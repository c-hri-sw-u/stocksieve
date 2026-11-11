"""下载 A 股数据到本地（股票列表：新浪 / 东方财富；日线：腾讯，东方财富备用）。

用法：
    python -m app.sync                  # 增量更新：最近 3 年日线（前复权），跳过今天已下载的
    python -m app.sync --full           # 全部重新下载
    python -m app.sync --only-failed    # 只重试上次失败的
    python -m app.sync --years 5        # 下载最近 5 年
    python -m app.sync --limit 50       # 只下 50 只，先试试
    python -m app.sync --industry       # 同时下载东财行业分类（很慢，默认沿用上次的）

每只股票先单独存到 data/cn_a/hist/<code>.parquet，每 500 只合并一次到 data/cn_a/daily.parquet，
所以下载过程中网站就能用上已下载的部分。进度写在 data/cn_a/sync_status.json，
创建 data/cn_a/sync_cancel 文件可以中途取消。
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from . import config
from .data import DAILY_COLUMNS, STOCK_COLUMNS, market_dir

MARKET = "cn_a"

SPOT_RENAME = {
    "代码": "code",
    "名称": "name",
    "最新价": "price",
    "市盈率-动态": "pe",
    "市净率": "pb",
    "总市值": "total_mv",
    "流通市值": "float_mv",
}
HIST_RENAME = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "涨跌幅": "pct_chg",
    "换手率": "turnover",
}


def _retry(fn, *args, tries=3, **kwargs):
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))


SINA_URL = "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData"


def fetch_spot_sina(max_seconds: float | None = None) -> pd.DataFrame:
    """新浪全部 A 股行情，逐页下载、单页失败重试。新浪的市值单位是万元，这里换成元。
    max_seconds：超过这个时间还没下完就放弃（新浪被频繁请求后会变得很慢）。"""
    import requests

    rows, page, t0 = [], 1, time.time()
    while True:
        if max_seconds and time.time() - t0 > max_seconds:
            raise TimeoutError(f"新浪列表超过 {max_seconds:.0f} 秒未完成")
        params = {"page": page, "num": 100, "sort": "symbol", "asc": 1, "node": "hs_a", "symbol": "", "_s_r_a": "page"}
        data = _retry(lambda: requests.get(SINA_URL, params=params, timeout=20).json(), tries=5)
        if not data:
            break
        rows.extend(data)
        page += 1
        time.sleep(0.3)
    df = pd.DataFrame(rows)
    out = pd.DataFrame({
        "code": df["code"].astype(str).str.zfill(6),
        "name": df["name"],
        "price": pd.to_numeric(df["trade"], errors="coerce"),
        "pe": pd.to_numeric(df["per"], errors="coerce"),
        "pb": pd.to_numeric(df["pb"], errors="coerce"),
        "total_mv": pd.to_numeric(df["mktcap"], errors="coerce") * 1e4,
        "float_mv": pd.to_numeric(df["nmc"], errors="coerce") * 1e4,
        # 当天行情，增量更新用。新浪成交量单位是股，换成手；换手率已经是百分比
        "open": pd.to_numeric(df["open"], errors="coerce"),
        "high": pd.to_numeric(df["high"], errors="coerce"),
        "low": pd.to_numeric(df["low"], errors="coerce"),
        "prev_close": pd.to_numeric(df["settlement"], errors="coerce"),
        "volume": pd.to_numeric(df["volume"], errors="coerce") / 100,
        "amount": pd.to_numeric(df["amount"], errors="coerce"),
        "pct_chg": pd.to_numeric(df["changepercent"], errors="coerce"),
        "turnover": pd.to_numeric(df["turnoverratio"], errors="coerce"),
        "ticktime": df["ticktime"].astype(str),
    })
    print(f"  新浪列表：{len(out)} 只")
    return out


def _tx_symbol(code: str) -> str:
    if code.startswith(("92", "4", "8")):
        return "bj" + code
    if code.startswith(("6", "9")):
        return "sh" + code
    return "sz" + code


def fetch_one_tx(ak, code: str, start: str, end: str) -> pd.DataFrame:
    """腾讯日线（前复权）。腾讯的成交量单位是股、换手率是小数，这里换成和东财一致：手、百分比。"""
    df = _retry(ak.stock_zh_a_hist_tx, symbol=_tx_symbol(code), start_date=start, end_date=end, adjust="qfq")
    if df is None or df.empty:
        return pd.DataFrame(columns=DAILY_COLUMNS)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["volume"] = df["volume"] / 100
    df["turnover"] = df["turnover"] * 100
    df["pct_chg"] = df["close"].pct_change() * 100
    df["code"] = code
    return df[DAILY_COLUMNS]


def fetch_one(ak, code: str, start: str, end: str, sources: dict | None = None) -> pd.DataFrame:
    # 东财在国内家用网络经常超时，所以默认先用腾讯；在 GitHub Actions（美国）上东财又快又稳，设 XG_PREFER_EM=1 先用东财
    if os.environ.get("XG_PREFER_EM"):
        try:
            df = _retry(ak.stock_zh_a_hist, symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq", tries=2)
            if df is not None and not df.empty:
                if sources is not None:
                    sources["em"] += 1
                df = df.rename(columns=HIST_RENAME)
                df["code"] = code
                df["date"] = pd.to_datetime(df["date"])
                return df[DAILY_COLUMNS]
        except Exception:
            pass
    try:
        df = fetch_one_tx(ak, code, start, end)
        if not df.empty:
            if sources is not None:
                sources["tx"] += 1
            return df
    except Exception:
        pass
    if sources is not None:
        sources["em"] += 1
    df = _retry(ak.stock_zh_a_hist, symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
    if df is None or df.empty:
        return pd.DataFrame(columns=DAILY_COLUMNS)
    df = df.rename(columns=HIST_RENAME)
    df["code"] = code
    df["date"] = pd.to_datetime(df["date"])
    return df[DAILY_COLUMNS]


def DAILY_LABEL() -> str:
    return "东方财富（腾讯备用）" if os.environ.get("XG_PREFER_EM") else "腾讯（东方财富备用）"


PHASES = {"list": "获取股票列表", "industry": "下载行业分类", "daily": "下载日线", "append": "写入新数据", "merge": "合并数据"}
MODES = {"industry": "更新行业分类", "incremental": "增量更新", "update": "补全下载", "full": "全部重新下载", "retry": "重试失败"}
MERGE_EVERY = 500


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_atomic(path: Path, write) -> None:
    """先写临时文件再改名，读的一方永远看到完整文件。"""
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def write_json(path: Path, obj: dict) -> None:
    write_atomic(path, lambda t: t.write_text(json.dumps(obj, ensure_ascii=False, indent=2), "utf-8"))


class Status:
    def __init__(self, out: Path, **init):
        self.path = out / "sync_status.json"
        self.lock = threading.Lock()
        self.s = {"state": "running", "phase": "list", "phase_label": PHASES["list"], "done": 0, "total": 0,
                  "failed": [], "list_source": "", "daily_source": "", "years": 3, "started_at": _now(),
                  "updated_at": _now(), "finished_at": None, "message": "", "pid": os.getpid(), **init}
        self.save()

    def update(self, **kw):
        with self.lock:
            if "phase" in kw:
                kw["phase_label"] = PHASES[kw["phase"]]
            self.s.update(kw, updated_at=_now())
            self.save()

    def save(self):
        write_json(self.path, self.s)


def load_status(market: str = MARKET) -> dict:
    f = market_dir(market) / "sync_status.json"
    try:
        return json.loads(f.read_text("utf-8"))
    except Exception:
        return {}


def get_stock_list(ak, out: Path, fresh: bool = False, max_seconds: float | None = None) -> tuple[pd.DataFrame, str]:
    """返回 (股票列表, 来源)。今天已经拿过就直接用缓存，避免反复请求被封；fresh=True 时必须在线拿最新行情。"""
    cache = out / "spot_cache.parquet"
    if not fresh and cache.exists() and datetime.fromtimestamp(cache.stat().st_mtime).date() == date.today():
        return pd.read_parquet(cache), "本地缓存（今天）"
    errors = []
    try:
        spot = fetch_spot_sina(max_seconds)
        spot.to_parquet(cache, index=False)
        return spot, "新浪"
    except Exception as e:
        errors.append(f"新浪：{type(e).__name__}")
        print(f"  新浪列表失败（{type(e).__name__}），改用东方财富")
    try:
        spot = _retry(ak.stock_zh_a_spot_em, tries=1).rename(columns=SPOT_RENAME)
        spot["code"] = spot["code"].astype(str)
        spot = spot.rename(columns={"今开": "open", "最高": "high", "最低": "low", "昨收": "prev_close", "成交量": "volume",
                                    "成交额": "amount", "涨跌幅": "pct_chg", "换手率": "turnover"})
        spot["ticktime"] = ""  # 东财没有时间戳
        keep = ["code", "name", "price", "pe", "pb", "total_mv", "float_mv", "open", "high", "low", "prev_close",
                "volume", "amount", "pct_chg", "turnover", "ticktime"]
        spot = spot[[c for c in keep if c in spot.columns]]
        spot.to_parquet(cache, index=False)
        return spot, "东方财富"
    except Exception as e:
        errors.append(f"东方财富：{type(e).__name__}")
    if cache.exists() and not fresh:
        print("  在线列表都失败，使用上次缓存的股票列表")
        return pd.read_parquet(cache), "本地缓存"
    raise RuntimeError("股票列表下载失败（" + "；".join(errors) + "）")


def merge(out: Path, hist_dir: Path, stocks: pd.DataFrame, extra: dict) -> dict | None:
    codes = stocks["code"].tolist()
    parts = [pd.read_parquet(hist_dir / f"{c}.parquet") for c in codes if (hist_dir / f"{c}.parquet").exists()]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return None
    daily = pd.concat(parts, ignore_index=True).sort_values(["code", "date"]).reset_index(drop=True)
    last_close = daily.groupby("code")["close"].last()
    st = stocks.copy()
    st["price"] = st["price"].fillna(st["code"].map(last_close))
    st = st[st["code"].isin(last_close.index)].reset_index(drop=True)
    write_atomic(out / "daily.parquet", lambda t: daily.to_parquet(t, index=False))
    write_atomic(out / "stocks.parquet", lambda t: st.to_parquet(t, index=False))
    meta = {
        "stocks": int(daily["code"].nunique()),
        "first_date": daily["date"].min().strftime("%Y-%m-%d"),
        "last_date": daily["date"].max().strftime("%Y-%m-%d"),
        "synced_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        **extra,
    }
    write_json(out / "meta.json", meta)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="下载 A 股数据")
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="只下载前 N 只（调试用）")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--full", action="store_true", help="忽略今天已下载的，全部重新下载")
    ap.add_argument("--only-failed", action="store_true", help="只重试上次失败的")
    ap.add_argument("--incremental", action="store_true", help="只补最新交易日（已有数据时）")
    ap.add_argument("--industry", action="store_true", help="强制重新下载行业分类（默认 7 天内用缓存）")
    ap.add_argument("--industry-only", action="store_true", help="只更新行业分类，不下载日线")
    ap.add_argument("--chunk", default="", help="只下载第 i 组（共 N 组），格式 i/N，不合并；给并行下载用")
    ap.add_argument("--merge-only", action="store_true", help="不下载，只把 hist/ 里已有的文件合并成 daily.parquet")
    ap.add_argument("--list-only", action="store_true", help="只获取股票列表和行业分类（给并行下载做准备）")
    ap.add_argument("--apply-industry", action="store_true", help="不联网，把已有的 industry.parquet 写进股票表")
    ap.add_argument("--no-industry", action="store_true", help="不下载行业分类（有缓存就用缓存）")
    args = ap.parse_args()

    out = market_dir(MARKET)
    hist_dir = out / "hist"
    hist_dir.mkdir(parents=True, exist_ok=True)
    cancel_flag = out / "sync_cancel"
    cancel_flag.unlink(missing_ok=True)
    last_failed = load_status().get("failed", []) if args.only_failed else []

    if args.apply_industry:
        apply_industry_cache(out)
        return
    incremental = args.incremental and (out / "daily.parquet").exists()
    mode = ("industry" if args.industry_only else "incremental" if incremental else "full" if args.full
            else "retry" if args.only_failed else "update")
    status = Status(out, years=args.years, mode=mode, mode_label=MODES[mode])
    try:
        if args.industry_only:
            run_industry_only(out, status)
        elif incremental:
            run_incremental(args, out, hist_dir, cancel_flag, status)
        else:
            run(args, out, hist_dir, cancel_flag, last_failed, status)
    except Exception as e:
        status.update(state="error", message=f"{type(e).__name__}: {e}", finished_at=_now())
        raise
    finally:
        cancel_flag.unlink(missing_ok=True)


def run(args, out: Path, hist_dir: Path, cancel_flag: Path, last_failed: list, status: Status) -> None:
    import akshare as ak

    print("获取股票列表…")
    spot, list_source = get_stock_list(ak, out)
    status.update(list_source=list_source)
    spot["code"] = spot["code"].astype(str).str.zfill(6)
    spot["price"] = pd.to_numeric(spot["price"], errors="coerce")
    spot = spot[spot["price"].notna() & (spot["price"] > 0)].drop_duplicates("code")
    if args.limit:
        spot = spot.head(args.limit)

    ind = load_industry(out) if args.no_industry else ensure_industry(ak, out, spot["code"].tolist(), status, force=args.industry or args.full)
    spot = apply_industry(spot, ind)
    stocks = spot[STOCK_COLUMNS].reset_index(drop=True)
    if args.list_only:
        status.update(state="done", finished_at=_now(), message=f"股票列表 {len(stocks)} 只（{list_source}）")
        print(f"股票列表：{len(stocks)} 只，来源 {list_source}")
        return

    end = date.today()
    start = end - timedelta(days=365 * args.years)
    s, e = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
    today = datetime.now().date()

    codes = stocks["code"].tolist()
    if args.only_failed:
        todo = [c for c in codes if c in set(last_failed)]
    else:
        todo = []
        for c in codes:
            f = hist_dir / f"{c}.parquet"
            if not args.full and f.exists() and datetime.fromtimestamp(f.stat().st_mtime).date() == today:
                continue
            todo.append(c)
    if args.merge_only:
        todo = []
    if args.chunk:
        i, n = (int(x) for x in args.chunk.split("/"))
        todo = [c for k, c in enumerate(todo) if k % n == i]
    print(f"下载日线：共 {len(codes)} 只，需要下载 {len(todo)} 只（{s} ~ {e}，前复权）")
    sources = {"tx": 0, "em": 0}
    daily_label = DAILY_LABEL()
    status.update(phase="daily", done=0, total=len(todo), daily_source=daily_label)
    extra = {"source": f"列表 {list_source} / 日线 {daily_label}", "list_source": list_source,
             "daily_source": daily_label, "years": args.years, "last_mode": status.s["mode_label"],
             "industry_source": ind["source"].iloc[0] if ind is not None and not ind.empty else ""}

    failed: list[str] = []
    cancelled = False
    pool = ThreadPoolExecutor(max_workers=args.workers)
    try:
        futs = {pool.submit(fetch_one, ak, c, s, e, sources): c for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            c = futs[fut]
            try:
                df = fut.result()
                write_atomic(hist_dir / f"{c}.parquet", lambda t: df.to_parquet(t, index=False))
            except Exception as ex:
                failed.append(c)
                if len(failed) <= 10:
                    print(f"  {c} 失败：{ex}")
            status.update(done=i, failed=failed)
            if i % 200 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}")
            if cancel_flag.exists():
                cancelled = True
                print("收到取消请求，停止下载")
                break
            if i % MERGE_EVERY == 0:
                status.update(phase="merge")
                merge(out, hist_dir, stocks, {**extra, "failed": failed, "partial": True})
                status.update(phase="daily")
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    if args.chunk:  # 并行下载的一组：只写 hist/，合并交给最后一步
        status.update(state="done", finished_at=_now(), message=f"第 {args.chunk} 组完成 · 失败 {len(failed)} 只")
        write_json(out / f"failed_{args.chunk.replace('/', '_')}.json", failed)
        print(f"第 {args.chunk} 组完成：{len(todo)} 只，失败 {len(failed)} 只")
        return
    if sources["em"] and not sources["tx"]:
        extra["daily_source"] = "东方财富"
    elif sources["em"]:
        extra["daily_source"] = f"腾讯 {sources['tx']} 只 / 东方财富 {sources['em']} 只"
    if args.merge_only:  # 合并并行下载的各组：把各组失败的代码汇总起来
        for f in out.glob("failed_*.json"):
            failed += json.loads(f.read_text())
    status.update(phase="merge", daily_source=extra["daily_source"])
    print("合并日线…")
    meta = merge(out, hist_dir, stocks, {**extra, "failed": failed, "partial": cancelled})
    n = meta["stocks"] if meta else 0
    if cancelled:
        status.update(state="cancelled", finished_at=_now(), message=f"已取消，已合并 {n} 只")
        print(f"已取消：已合并 {n} 只股票")
        return
    status.update(state="done", finished_at=_now(), message=f"{status.s['mode_label']}完成 · {n} 只 · 失败 {len(failed)} 只")
    if meta:
        print(f"完成：{n} 只股票，{meta['first_date']} ~ {meta['last_date']}，失败 {len(failed)} 只")
    if failed:
        print("失败的可以运行 python -m app.sync --only-failed 补下。")


# ---------------------------------------------------------------- 增量更新
def trade_calendar(ak, out: Path) -> list[date]:
    """交易日历（新浪），按天缓存；拿不到就退化成工作日。"""
    cache = out / "trade_calendar.json"
    if cache.exists() and datetime.fromtimestamp(cache.stat().st_mtime).date() == date.today():
        return [date.fromisoformat(x) for x in json.loads(cache.read_text())]
    try:
        cal = sorted(pd.to_datetime(_retry(ak.tool_trade_date_hist_sina)["trade_date"]).dt.date)
        write_json(cache, [d.isoformat() for d in cal])  # type: ignore[arg-type]
        return cal
    except Exception as e:
        print(f"  交易日历获取失败（{type(e).__name__}），按工作日估算")
        return list(pd.bdate_range("2000-01-01", date.today() + timedelta(days=30)).date)


def beijing_now() -> datetime:
    from datetime import timezone
    return datetime.now(timezone(timedelta(hours=8))).replace(tzinfo=None)


def latest_closed_day(cal: list[date], now_bj: datetime) -> date:
    """已经收盘的最近一个交易日（北京时间 15:05 之后才算当天收盘）。"""
    today = now_bj.date()
    after_close = (now_bj.hour, now_bj.minute) >= (15, 5)
    return max(d for d in cal if d < today or (d == today and after_close))


def snapshot_rows(spot: pd.DataFrame, last_close: pd.Series, day: date) -> tuple[pd.DataFrame, list[str]]:
    """把收盘后的全市场行情变成 day 这一天的日线。
    返回 (可以直接追加的行, 需要整只重下的代码)。昨收和本地最后收盘价对不上的是除权除息或新股，前复权价格要整只重算。"""
    spot = spot[pd.to_numeric(spot["volume"], errors="coerce") > 0].copy()  # 停牌的不追加
    known = spot["code"].isin(last_close.index)
    stored = spot["code"].map(last_close)
    diff = (spot["prev_close"] - stored).abs()
    match = known & ((diff <= 0.011) | (diff / stored <= 0.001))
    redo = spot.loc[~match, "code"].tolist()
    ok = spot[match]
    rows = pd.DataFrame({
        "date": pd.Timestamp(day), "code": ok["code"], "open": ok["open"], "high": ok["high"], "low": ok["low"],
        "close": ok["price"], "volume": ok["volume"], "amount": ok["amount"], "pct_chg": ok["pct_chg"],
        "turnover": ok["turnover"],
    })[DAILY_COLUMNS]
    return rows, redo


def snapshot_is_for(spot: pd.DataFrame, last_close: pd.Series) -> str:
    """判断行情快照是哪一天的：'next' = 本地最后一天的下一个交易日（已收盘），'same' = 和本地最后一天相同，'unknown'。"""
    s = spot[spot["code"].isin(last_close.index) & (pd.to_numeric(spot["volume"], errors="coerce") > 0)]
    if len(s) < 50:
        return "unknown"
    stored = s["code"].map(last_close)
    prev_match = ((s["prev_close"] - stored).abs() <= 0.011).mean()
    close_match = ((s["price"] - stored).abs() <= 0.011).mean()
    closed = (s["ticktime"].fillna("") >= "15:00").mean() if s["ticktime"].astype(bool).any() else 1.0
    if close_match > 0.8 and prev_match < 0.5:
        return "same"
    if prev_match > 0.8 and closed > 0.8:
        return "next"
    return "unknown"


def append_rows(hist_dir: Path, rows: pd.DataFrame, status: "Status", cancel_flag: Path) -> int:
    groups = list(rows.groupby("code"))
    status.update(phase="append", done=0, total=len(groups))
    for i, (code, g) in enumerate(groups, 1):
        f = hist_dir / f"{code}.parquet"
        old = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=DAILY_COLUMNS)
        new = pd.concat([old, g], ignore_index=True).drop_duplicates("date", keep="last").sort_values("date")
        write_atomic(f, lambda t: new.to_parquet(t, index=False))
        if i % 100 == 0 or i == len(groups):
            status.update(done=i)
        if cancel_flag.exists():
            return i
    return len(groups)


def ensure_hist(out: Path, hist_dir: Path) -> None:
    """全新的机器上（比如 GitHub Actions）只有合并好的 daily.parquet：按股票拆回 hist/，增量更新和合并都基于它。"""
    if any(hist_dir.glob("*.parquet")) or not (out / "daily.parquet").exists():
        return
    daily = pd.read_parquet(out / "daily.parquet")
    print(f"从 daily.parquet 拆出 {daily['code'].nunique()} 只股票的单独文件…")
    for code, g in daily.groupby("code"):
        g[DAILY_COLUMNS].to_parquet(hist_dir / f"{code}.parquet", index=False)


def run_incremental(args, out: Path, hist_dir: Path, cancel_flag: Path, status: "Status") -> None:
    """只补最新的交易日：
    - 只缺 1 个交易日且行情快照是收盘后的：直接用全市场快照追加一行（几十个请求），除权除息和新股整只重下；
    - 缺多天或快照对不上：逐只下载缺的那几天（和本地最后一天一起下，用来核对是否除权）；
    - 已是最新：什么都不下载。
    """
    import akshare as ak

    ensure_hist(out, hist_dir)
    daily = pd.read_parquet(out / "daily.parquet", columns=["date", "code", "close"])
    daily = daily.sort_values(["code", "date"])
    last_close = daily.groupby("code")["close"].last()
    last_date = daily["date"].max().date()

    cal = trade_calendar(ak, out)
    target = latest_closed_day(cal, beijing_now())
    missing = [d for d in cal if last_date < d <= target]
    print(f"本地数据截至 {last_date}，最近收盘交易日 {target}，缺 {len(missing)} 个交易日")

    status.update(phase="list")
    spot, list_source, fresh_ok = None, "", False
    if len(missing) == 1:
        # 只缺一天：拿收盘后的全市场行情，几十个请求就能补完
        try:
            spot, list_source = get_stock_list(ak, out, fresh=True, max_seconds=120)
            fresh_ok = True
        except Exception as e:
            print(f"  全市场行情获取失败（{e}），改为逐只补")
    if spot is None:
        spot, list_source = get_stock_list(ak, out)
    spot["code"] = spot["code"].astype(str).str.zfill(6)
    status.update(list_source=list_source, daily_source=f"{list_source}全市场行情")

    old_stocks = pd.read_parquet(out / "stocks.parquet") if (out / "stocks.parquet").exists() else pd.DataFrame(columns=STOCK_COLUMNS)
    # 股票表：用最新快照的名称/市值/PE，快照里没有的（停牌、退市）保留旧的
    fresh = spot.copy()
    fresh["price"] = pd.to_numeric(fresh["price"], errors="coerce").where(lambda x: x > 0)
    old_stocks = apply_industry(old_stocks, None)
    stocks = pd.concat([apply_industry(fresh, None)[STOCK_COLUMNS],
                        old_stocks[~old_stocks["code"].isin(fresh["code"])][STOCK_COLUMNS]], ignore_index=True)
    ind = load_industry(out) if args.no_industry else ensure_industry(ak, out, stocks["code"].tolist(), status)
    stocks = apply_industry(stocks, ind)
    if args.limit:  # 调试用：只处理本地已有的前 N 只
        stocks = stocks[stocks["code"].isin(last_close.index)].head(args.limit)
        spot = spot[spot["code"].isin(stocks["code"])]
    new_codes = [c for c in stocks["code"] if c not in last_close.index and c in set(spot["code"])]

    extra = {"source": f"列表 {list_source} / 日线 腾讯 + {list_source}行情", "list_source": list_source,
             "daily_source": "腾讯（东方财富备用）", "years": load_meta(out).get("years", args.years),
             "last_mode": MODES["incremental"],
             "industry_source": ind["source"].iloc[0] if ind is not None and not ind.empty else ""}
    failed: list[str] = []
    appended_days: list[date] = []
    redo: list[str] = []
    how = ""

    if not missing:
        how = f"已是最新（截至 {last_date}）"
    else:
        kind = snapshot_is_for(spot, last_close) if fresh_ok and "prev_close" in spot.columns else "unknown"
        if len(missing) == 1 and kind == "next":
            rows, redo = snapshot_rows(spot[~spot["code"].isin(new_codes)], last_close, missing[0])
            how = f"用{list_source}行情补 {missing[0]}"
            append_rows(hist_dir, rows, status, cancel_flag)
            appended_days = missing
        else:
            # 逐只补缺的几天：从本地最后一天开始下，最后一天的收盘价对不上说明除权了，整只重下
            s_, e_ = last_date.strftime("%Y%m%d"), target.strftime("%Y%m%d")
            codes = [c for c in stocks["code"] if c in last_close.index]
            how = f"逐只补 {missing[0]}" + (f" ~ {missing[-1]}" if len(missing) > 1 else "")
            status.update(phase="daily", done=0, total=len(codes), daily_source="腾讯（东方财富备用）")
            sources = {"tx": 0, "em": 0}

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futs = {pool.submit(fetch_one, ak, c, s_, e_, sources): c for c in codes}
                for i, fut in enumerate(as_completed(futs), 1):
                    c = futs[fut]
                    try:
                        df = fut.result()
                        base = df[df["date"].dt.date == last_date]
                        if not base.empty and abs(base["close"].iloc[0] - last_close[c]) > max(0.011, last_close[c] * 0.001):
                            redo.append(c)
                        else:
                            new = df[df["date"].dt.date > last_date]
                            if not new.empty:
                                f = hist_dir / f"{c}.parquet"
                                old = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=DAILY_COLUMNS)
                                merged = pd.concat([old, new], ignore_index=True).drop_duplicates("date", keep="last")
                                write_atomic(f, lambda t: merged.sort_values("date").to_parquet(t, index=False))
                    except Exception:
                        failed.append(c)
                    status.update(done=i, failed=failed)
                    if cancel_flag.exists():
                        pool.shutdown(wait=False, cancel_futures=True)
                        break
            appended_days = missing

    # 除权除息：整只重下；本地还没有的（新股，或上次没下完的）：整只下载
    n_exright = len(redo)
    redo = list(dict.fromkeys(redo + new_codes))
    if redo and not cancel_flag.exists():
        end = date.today()
        s_, e_ = (end - timedelta(days=365 * extra["years"])).strftime("%Y%m%d"), target.strftime("%Y%m%d")
        status.update(phase="daily", done=0, total=len(redo), daily_source="腾讯（东方财富备用）")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(fetch_one, ak, c, s_, e_): c for c in redo}
            for i, fut in enumerate(as_completed(futs), 1):
                c = futs[fut]
                try:
                    df = fut.result()
                    if not df.empty:
                        write_atomic(hist_dir / f"{c}.parquet", lambda t: df.to_parquet(t, index=False))
                except Exception:
                    failed.append(c)
                status.update(done=i, failed=failed)
                if cancel_flag.exists():
                    pool.shutdown(wait=False, cancel_futures=True)
                    break

    cancelled = cancel_flag.exists()
    status.update(phase="merge")
    meta = merge(out, hist_dir, stocks, {**extra, "failed": failed, "partial": cancelled})
    n = meta["stocks"] if meta else 0
    parts = [how]
    if n_exright:
        parts.append(f"除权除息重下 {n_exright} 只")
    if new_codes:
        parts.append(f"补下本地缺的 {len(new_codes)} 只")
    if failed:
        parts.append(f"失败 {len(failed)} 只")
    msg = "增量更新 · " + "，".join(parts)
    if cancelled:
        status.update(state="cancelled", finished_at=_now(), message=f"已取消 · {msg}")
    else:
        status.update(state="done", finished_at=_now(), message=msg)
    print(msg + f"，共 {n} 只")


def load_meta(out: Path) -> dict:
    try:
        return json.loads((out / "meta.json").read_text("utf-8"))
    except Exception:
        return {}


# ---------------------------------------------------------------- 行业分类
INDUSTRY_MAX_AGE_DAYS = 7
INDUSTRY_SOURCES = {"sw": "申万一级", "sina": "新浪行业", "em": "东财行业"}


def _with_budget(items, fn, workers, budget, progress=None):
    """并发对 items 调 fn，超过 budget 秒就停下，返回已经拿到的部分 [(item, 结果)]，失败的跳过。"""
    t0, out = time.time(), []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fn, it): it for it in items}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                out.append((futs[fut], fut.result()))
            except Exception:
                pass
            if progress:
                progress(i, len(items))
            if time.time() - t0 > budget:
                print(f"  超过 {budget} 秒，先用已拿到的 {i}/{len(items)}")
                pool.shutdown(wait=False, cancel_futures=True)
                break
    return out


def industry_sw(ak, progress=None, budget=None, out: Path | None = None) -> pd.DataFrame:
    budget = budget or float(os.environ.get("XG_SW_BUDGET", 3600))  # 在 GitHub Actions 上限时，避免拖住整条流水线
    """申万行业（数据来自乐咕乐股）：按 336 个三级行业取成分股，再映射到一级、二级。"""
    def slow_retry(fn, tries=5):
        # 乐咕乐股频繁请求时会返回空页面（解析报 'NoneType' ... find_all），隔久一点再试
        for i in range(tries):
            try:
                return fn()
            except Exception:
                if i == tries - 1:
                    raise
                time.sleep(5 * (i + 1))

    # 一二三级行业表很少变，拿到一次就缓存；乐咕乐股限流时直接用缓存
    lv = (out / "sw_levels.json") if out else None
    try:
        l1 = slow_retry(ak.sw_index_first_info, tries=3)
        time.sleep(2)
        l2 = slow_retry(ak.sw_index_second_info, tries=3)
        time.sleep(2)
        l3 = slow_retry(ak.sw_index_third_info, tries=3)
        if lv:
            write_json(lv, {k: v[["行业代码", "行业名称", "上级行业"] if "上级行业" in v else ["行业代码", "行业名称"]].to_dict("records")
                            for k, v in (("l1", l1), ("l2", l2), ("l3", l3))})
    except Exception:
        if not (lv and lv.exists()):
            raise
        print("  申万行业表获取失败，使用缓存")
        c = json.loads(lv.read_text("utf-8"))
        l1, l2, l3 = (pd.DataFrame(c[k]) for k in ("l1", "l2", "l3"))
    l2_parent = dict(zip(l2["行业名称"], l2["上级行业"]))
    l3_parent = dict(zip(l3["行业名称"], l3["上级行业"]))
    first_names = set(l1["行业名称"])

    # 乐咕乐股会限流，所以每拿到一个三级行业就记下来（7 天内有效），中断后再跑只补没拿到的
    part = (out / "industry_sw_partial.json") if out else None
    done: dict[str, list[str]] = {}
    if part and part.exists() and (time.time() - part.stat().st_mtime) < INDUSTRY_MAX_AGE_DAYS * 86400:
        try:
            done = json.loads(part.read_text("utf-8"))
        except Exception:
            done = {}
    lock = threading.Lock()

    def cons(code):
        df = _retry(ak.sw_index_third_cons, symbol=code, tries=3)
        codes = [c.split(".")[0].zfill(6) for c in df["股票代码"].astype(str)]
        with lock:
            done[code] = codes
            if part:
                write_json(part, done)
        time.sleep(0.5)
        return codes

    items = list(zip(l3["行业代码"], l3["行业名称"]))
    todo = [x for x in items if x[0] not in done]
    base = len(items) - len(todo)
    prog = (lambda i, n: progress(base + i, len(items))) if progress else None
    if progress:
        progress(base, len(items))
    _with_budget(todo, lambda x: cons(x[0]), 2, budget, prog)

    rows = []
    for code, name in items:
        second = l3_parent.get(name, "")
        first = l2_parent.get(second, "")
        if first not in first_names or code not in done:
            continue
        for c in done[code]:
            rows.append({"code": c, "industry": first, "industry2": second})
    if len(done) < len(items):
        print(f"  申万：拿到 {len(done)}/{len(items)} 个三级行业，下次运行会接着补")
    return pd.DataFrame(rows, columns=["code", "industry", "industry2"]).drop_duplicates("code")


def industry_sina(ak, progress=None, budget=180) -> pd.DataFrame:
    """新浪行业：约 50 个行业，每个一次请求。只有一级。"""
    sectors = _retry(ak.stock_sector_spot, indicator="新浪行业")
    rows = []
    for (label, name), df in _with_budget(list(zip(sectors["label"], sectors["板块"])),
                                          lambda x: _retry(ak.stock_sector_detail, sector=x[0], tries=2), 4, budget, progress):
        for c in df["code"].astype(str):
            rows.append({"code": c.zfill(6), "industry": name, "industry2": ""})
    return pd.DataFrame(rows).drop_duplicates("code")


def industry_em(ak, progress=None, budget=180) -> pd.DataFrame:
    boards = _retry(ak.stock_board_industry_name_em, tries=1)
    rows = []
    for name, df in _with_budget(boards["板块名称"].tolist(),
                                 lambda n: _retry(ak.stock_board_industry_cons_em, symbol=n, tries=1), 4, budget, progress):
        for c in df["代码"].astype(str):
            rows.append({"code": c.zfill(6), "industry": name, "industry2": ""})
    return pd.DataFrame(rows).drop_duplicates("code")


def load_industry(out: Path) -> pd.DataFrame | None:
    f = out / "industry.parquet"
    try:
        return pd.read_parquet(f) if f.exists() else None
    except Exception:
        return None


def ensure_industry(ak, out: Path, universe: list[str], status: "Status | None" = None, force: bool = False) -> pd.DataFrame | None:
    """返回行业表（code, industry, industry2, source, fetched_at）。
    缓存不到 7 天就直接用；否则按 申万 → 新浪 → 东财 的顺序尝试，第一个能覆盖 80% 以上股票的就用它。
    全部失败时保留旧缓存，不会把行业清空。"""
    cache = load_industry(out)
    if cache is not None and not force and not cache.empty:
        age = (datetime.now() - pd.to_datetime(cache["fetched_at"].iloc[0])).days
        if age < INDUSTRY_MAX_AGE_DAYS:
            return cache
    if status:
        status.update(phase="industry", done=0, total=0)
    uni = set(universe)
    best, errors = None, []
    for key, fn in (("sw", industry_sw), ("sina", industry_sina), ("em", industry_em)):
        label = INDUSTRY_SOURCES[key]
        print(f"下载行业分类：{label}…")
        if status:
            status.update(industry_source=label, done=0, total=0)
        prog = (lambda i, n: status.update(done=i, total=n)) if status else None
        try:
            df = fn(ak, prog, out=out) if key == "sw" else fn(ak, prog)
        except Exception as e:
            errors.append(f"{label}：{type(e).__name__} {e}"[:200])
            print(f"  {label}失败：{type(e).__name__}: {e}")
            continue
        cover = df["code"].isin(uni).sum() / max(len(uni), 1)
        print(f"  {label}：覆盖 {cover:.0%}")
        if best is None or cover > best[1]:
            best = (key, cover, df)
        if cover >= 0.8:
            break
    if best is None or best[1] < 0.3:
        print("行业分类：所有来源都失败，沿用旧数据。" + "；".join(errors))
        if status and cache is not None and not cache.empty:
            status.update(industry_source=cache["source"].iloc[0])
        return cache
    key, cover, df = best
    if status:
        status.update(industry_source=INDUSTRY_SOURCES[key])
    df = df.assign(source=INDUSTRY_SOURCES[key], fetched_at=datetime.now().strftime("%Y-%m-%d %H:%M"))
    write_atomic(out / "industry.parquet", lambda t: df.to_parquet(t, index=False))
    return df


def apply_industry(stocks: pd.DataFrame, ind: pd.DataFrame | None) -> pd.DataFrame:
    stocks = stocks.copy()
    if ind is not None and not ind.empty:
        m = ind.set_index("code")
        stocks["industry"] = stocks["code"].map(m["industry"])
        stocks["industry2"] = stocks["code"].map(m["industry2"])
    for c in ("industry", "industry2"):
        if c not in stocks.columns:
            stocks[c] = ""
        stocks[c] = stocks[c].astype(object).fillna("")
    return stocks


def apply_industry_cache(out: Path) -> None:
    """不联网：把 industry.parquet（如果有）写进 stocks.parquet 和 meta.json。"""
    ind = load_industry(out)
    if ind is None or ind.empty or not (out / "stocks.parquet").exists():
        print("没有行业分类缓存，跳过")
        return
    st = apply_industry(pd.read_parquet(out / "stocks.parquet"), ind)[STOCK_COLUMNS]
    write_atomic(out / "stocks.parquet", lambda t: st.to_parquet(t, index=False))
    meta = load_meta(out)
    meta["industry_source"] = ind["source"].iloc[0]
    write_json(out / "meta.json", meta)
    print(f"已应用行业分类：{meta['industry_source']}，覆盖 {(st['industry'] != '').sum()}/{len(st)}")


def run_industry_only(out: Path, status: "Status") -> None:
    import akshare as ak

    if not (out / "stocks.parquet").exists():  # 全量下载还没合并时：只按股票列表拿行业分类，存成 industry.parquet
        codes = pd.read_parquet(out / "spot_cache.parquet")["code"].astype(str).str.zfill(6).tolist()
        ind = ensure_industry(ak, out, codes, status, force=True)
        n = 0 if ind is None else ind["code"].isin(set(codes)).sum()
        status.update(state="done", finished_at=_now(), message=f"行业分类完成 · 覆盖 {n}/{len(codes)} 只")
        print(f"行业分类：覆盖 {n}/{len(codes)}")
        return
    st = pd.read_parquet(out / "stocks.parquet")
    ind = ensure_industry(ak, out, st["code"].tolist(), status, force=True)
    st = apply_industry(st, ind)[STOCK_COLUMNS]
    write_atomic(out / "stocks.parquet", lambda t: st.to_parquet(t, index=False))
    meta = load_meta(out)
    src = ind["source"].iloc[0] if ind is not None and not ind.empty else ""
    meta["industry_source"] = src
    write_json(out / "meta.json", meta)
    n = (st["industry"] != "").sum()
    status.update(state="done", finished_at=_now(), message=f"行业分类完成 · {src} · 覆盖 {n}/{len(st)} 只")
    print(f"行业分类完成：{src}，覆盖 {n}/{len(st)}")


if __name__ == "__main__":
    main()
