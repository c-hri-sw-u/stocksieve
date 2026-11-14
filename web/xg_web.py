"""浏览器（Pyodide）里的入口：复用 app/ 下同一份 Python 代码，只把「读数据」换成读本机缓存的文件。

worker.js 调这里的函数，参数和返回值都是 JSON 字符串。
"""
from __future__ import annotations

import gc
import json
import re
import time

import numpy as np
import pandas as pd

from app import data, prompts, runner, sandbox
from app.examples import EXAMPLES

STATE = {"daily": None, "stocks": None, "manifest": {}, "cats": None, "idx": None, "first": None, "ratio": None}
PRICES = ["open", "high", "low", "close"]
FLOATS = ["open", "high", "low", "close", "volume", "amount", "pct_chg", "turnover"]
COLS = ["date", "code", *FLOATS]


def _j(x) -> str:
    return json.dumps(x, ensure_ascii=False, default=str)


# ---------------------------------------------------------------- 数据
# 浏览器里的 Python 内存有限（最多约 4 GB，还要留给筛选计算），所以读数据时尽量不多复制：
# - 直接用 pyarrow 读，每列单独转成 numpy、单独排序，不做整张表的 concat + sort_values
# - 股票代码用同一批字符串对象（按代码编号取），不为每一行新建字符串
# - 价格等 8 列直接填进一整块数组再交给 DataFrame（不让 pandas 之后自己合并、临时再复制一份）
# - 价格等一直用 float32（和下载的文件一样），筛选时也不转 float64：转了内存翻倍，MACD 这类规则在长范围下会撑爆
def _read(path: str):
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=COLS)


def _codes(tables) -> np.ndarray:
    import pyarrow as pa
    import pyarrow.compute as pc

    u = pa.chunked_array([pc.unique(t.column("code")) for t in tables]).unique().to_pylist()
    return np.array(sorted(set(u) | set(STATE["cats"] if STATE["cats"] is not None else [])), dtype=object)


def _index(t, cats: np.ndarray) -> np.ndarray:
    import pyarrow as pa
    import pyarrow.compute as pc

    return pc.index_in(t.column("code"), value_set=pa.array(cats.tolist())).to_numpy(zero_copy_only=False).astype(np.int32)


def _col(t, c: str, mask=None) -> np.ndarray:
    a = t.column(c).to_numpy()
    if c == "date":
        a = a.astype("datetime64[ns]")
    elif c in FLOATS:
        a = a.astype(np.float32, copy=False)
    return a if mask is None else a[mask]


def _frame(date: np.ndarray, code: np.ndarray, floats, dtype) -> pd.DataFrame:
    """floats：依次产出 FLOATS 每一列的函数。逐列填进一整块数组（每列内存连续），不额外复制。"""
    block = np.empty((len(FLOATS), len(date)), dtype=dtype)
    for i, c in enumerate(FLOATS):
        block[i] = floats(c)
    df = pd.DataFrame(block.T, columns=FLOATS, copy=False)
    df.insert(0, "date", date)
    df.insert(1, "code", code)
    return df


def load_files(manifest_json: str, folder: str) -> str:
    """worker 已经把本机缓存的文件写进 Pyodide 的虚拟文件夹 folder。全部读进内存，之后每次筛选不再读盘。"""
    import os

    t0 = time.time()
    manifest = json.loads(manifest_json)
    STATE.update(daily=None, cats=None, idx=None, first=None, ratio=None)
    files = sorted((f for f in manifest["files"] if f["kind"] == "daily"), key=lambda f: f.get("from", ""))
    tables = [_read(os.path.join(folder, f["name"])) for f in files if os.path.exists(os.path.join(folder, f["name"]))]
    st = next((f for f in manifest["files"] if f["kind"] == "stocks"), None)
    stocks = pd.read_parquet(os.path.join(folder, st["name"])) if st and os.path.exists(os.path.join(folder, st["name"])) else None
    if not tables or stocks is None:
        STATE.update(stocks=None, manifest={})
        return _j({"ok": False, "error": "本机还没有完整的数据"})
    cats = _codes(tables)
    idx = np.concatenate([_index(t, cats) for t in tables])
    # 各文件按时间先后、每个文件内按 (代码, 日期) 排好；按代码做稳定排序就得到整体的 (代码, 日期) 顺序
    order = np.argsort(idx, kind="stable")
    idx = idx[order]
    daily = _frame(np.concatenate([_col(t, "date") for t in tables])[order], cats[idx],
                   lambda c: np.concatenate([_col(t, c) for t in tables])[order], np.float32)
    del tables, order
    if "industry2" not in stocks:
        stocks["industry2"] = ""
    # 每只股票在现有数据里的第一行（按代码编号）：往前拼接更早的历史时用
    n = len(cats)
    starts = np.searchsorted(idx, np.arange(n))
    has = starts < len(idx)
    has[has] = idx[starts[has]] == np.arange(n)[has]
    first = {"has": has, "date": np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]"), "close": np.full(n, np.nan), "pct": np.full(n, np.nan)}
    first["date"][has] = daily["date"].to_numpy()[starts[has]]
    first["close"][has] = daily["close"].to_numpy(dtype=np.float64)[starts[has]]
    first["pct"][has] = daily["pct_chg"].to_numpy(dtype=np.float64)[starts[has]]
    STATE.update(daily=daily, stocks=stocks.reset_index(drop=True), manifest=manifest, cats=cats, idx=idx, first=first)
    return _j({"ok": True, "rows": len(daily), "stocks": int(has.sum()), "seconds": round(time.time() - t0, 2)})


def _ratios(newest, cats: np.ndarray) -> np.ndarray:
    """每只股票：现有数据的价格 ÷ 历史数据的价格（按代码编号，算不出来的是 NaN）。

    两段都是前复权，但历史是之前某天下载的，之后的分红送转会让现有数据里的价格整体按比例变化。
    - 两段有重叠的日子：用现有数据第一天的收盘价 ÷ 历史里同一天的收盘价
    - 没有重叠（那几天停牌）：用现有数据第一天的涨跌幅倒推前一天收盘价，再 ÷ 历史里最后一天的收盘价
    - 都算不出来的股票不拼接（宁可没有，也不要错的价格）
    newest：最新的那个历史年份文件（和现有数据重叠的那个）
    """
    f = STATE["first"]
    ix = _index(newest, cats)
    date, close = _col(newest, "date"), _col(newest, "close").astype(np.float64)
    ok = (ix >= 0) & f["has"][np.maximum(ix, 0)]
    ix, date, close = ix[ok], date[ok], close[ok]
    fdate = f["date"][ix]
    ratio = np.full(len(cats), np.nan)
    same = date == fdate
    ratio[ix[same]] = f["close"][ix[same]] / close[same]
    # 只有历史的下载范围盖住了那只股票现有数据的第一天，中间才不会缺交易日（缺的那几天就是停牌）
    before = (date < fdate) & (fdate <= date.max()) & np.isnan(ratio[ix])
    if before.any():
        b = pd.DataFrame({"ix": ix[before], "date": date[before], "close": close[before]}).sort_values(["ix", "date"]).groupby("ix").last()
        prev = f["close"][b.index] / (1 + f["pct"][b.index] / 100)
        ratio[b.index] = prev / b["close"].to_numpy()
    ratio[~(np.isfinite(ratio) & (ratio > 0) & (ratio < 1e3))] = np.nan
    return ratio


def add_history(names_json: str, folder: str) -> str:
    """把更早的历史（hist-YYYY.parquet）接到现有数据前面。worker 已经把这些文件写进 folder。

    每次调用传的是还没载入的、从最新往前连续的几个年份文件；第一次一定包含和现有数据重叠的那个，用它算拼接比例。
    """
    import os

    if STATE["daily"] is None:
        return _j({"ok": False, "error": "还没有载入现有数据"})
    t0 = time.time()
    names = [n for n in json.loads(names_json) if os.path.exists(os.path.join(folder, n))]
    if not names:
        return _j({"ok": False, "error": "本机没有这些历史文件"})
    names.sort()  # hist-YYYY：按年份从早到晚
    cats = STATE["cats"]
    if STATE["ratio"] is None:
        STATE["ratio"] = _ratios(_read(os.path.join(folder, names[-1])), cats)
    ratio, first = STATE["ratio"], STATE["first"]
    hidx, hcols = [], {c: [] for c in COLS if c != "code"}
    for n in names:  # 一个文件一个文件处理，读完就丢
        t = _read(os.path.join(folder, n))
        ix = _index(t, cats)
        date = _col(t, "date")
        # 已经退市的（股票列表里没有）、算不出比例的不要；和现有数据重叠的那几天以现有数据为准
        keep = (ix >= 0)
        keep[keep] = np.isfinite(ratio[ix[keep]]) & (date[keep] < first["date"][ix[keep]])
        ix = ix[keep]
        hidx.append(ix)
        hcols["date"].append(date[keep])
        r = ratio[ix]
        for c in FLOATS:
            a = _col(t, c, keep)
            hcols[c].append((a * r).astype(np.float32) if c in PRICES else a)
        del t
    hidx = np.concatenate(hidx)
    rows = len(hidx)
    if not rows:
        return _j({"ok": True, "rows": 0, "first_date": str(STATE["daily"]["date"].min())[:10], "seconds": round(time.time() - t0, 2)})
    # 历史在前、现有数据在后，按代码稳定排序：每只股票的日期自然是从早到晚
    allidx = np.concatenate([hidx, STATE["idx"]])
    order = np.argsort(allidx, kind="stable")
    del hidx
    STATE["idx"] = allidx[order]
    del allidx
    old = STATE["daily"]
    both = lambda c: np.concatenate([np.concatenate(hcols.pop(c)), old[c].to_numpy()])[order]  # noqa: E731
    new = _frame(both("date"), cats[STATE["idx"]], both, np.float32)
    STATE["daily"] = new
    del old, order, new
    return _j({"ok": True, "rows": rows, "first_date": str(STATE["daily"]["date"].min())[:10],
               "seconds": round(time.time() - t0, 2)})


# 一次筛选最多用多少行（含预热）。浏览器里 Python 最多 4 GB 内存，规则代码本身还会复制几份数据，
# 用 float32 时，2015 年起的全部历史（约 1150 万行）在 Chrome 上一次能算完；再多就可能把计算环境撑崩。
# 超过时：只看单只股票自身走势的规则，把股票分成几批分别算再合起来（慢一些）；
# 要在股票之间比较的规则（排名、前 N 名、全市场平均等）分批会算错，只能提示缩短时间范围。
MAX_ROWS = 14_000_000
BATCH_ROWS = 5_000_000
BATCH = {"codes": None}  # 分批计算时，这一批的股票（按代码编号的布尔数组）


class RangeTooLarge(Exception):
    pass


def _has_data() -> bool:
    return STATE["daily"] is not None


def _load(market, start=None, end=None):
    if not _has_data():
        d, s = data._demo_tables()
        return d.copy(), s.copy()
    d, s = STATE["daily"], STATE["stocks"]
    # 只复制一次：筛选区间（含预热）的那部分。runner 不再整表复制（runner.COPY_INPUT = False）
    mask = np.ones(len(d), dtype=bool)
    if start:
        mask &= (d["date"] >= pd.Timestamp(start) - pd.Timedelta(days=prompts.LOOKBACK_DAYS)).to_numpy()
    if end:
        mask &= (d["date"] <= pd.Timestamp(end)).to_numpy()
    if BATCH["codes"] is not None:
        mask &= BATCH["codes"][STATE["idx"]]
        s = s[s["code"].isin(set(STATE["cats"][BATCH["codes"]]))]
    n = int(mask.sum())
    if n > MAX_ROWS and BATCH["codes"] is None:
        # 算出最早能从哪天开始：从结束日往前数，行数不超过上限（再留出预热的天数）
        dates = d["date"].to_numpy()[mask]
        u, cnt = np.unique(dates, return_counts=True)
        keep = np.cumsum(cnt[::-1]) <= MAX_ROWS
        earliest = pd.Timestamp(u[::-1][keep][-1]) + pd.Timedelta(days=prompts.LOOKBACK_DAYS)
        raise RangeTooLarge(f"{earliest:%Y-%m-%d}")
    out = _frame(d["date"].to_numpy()[mask], d["code"].to_numpy()[mask], lambda c: d[c].to_numpy()[mask], np.float32)
    return out, s.copy()


# 把 app.data 里「读文件」的地方换成读内存
data.load = _load
data.use_demo = lambda market: not _has_data()
runner.COPY_INPUT = False  # _load 已经给了一份独立的数据，不用再整表复制一次


# ---------------------------------------------------------------- 筛选
def run(market: str, start: str, end: str, code: str) -> str:
    try:
        sandbox.validate(code)
    except sandbox.UnsafeCode as e:
        return _j({"ok": False, "error": str(e)})
    try:
        try:
            return _j(runner.execute(market, start, end, code))
        except RangeTooLarge as e:
            if _cross_sectional(code):
                tip = f"请把开始日期改到 {e} 或之后再试" if str(e) <= end else "请缩短时间范围再试"
                return _j({"ok": False, "error": f"时间范围太长：这条规则要在股票之间比较（排名、前几名、全市场平均等），没法分批算，"
                                                  f"浏览器内存不够一次算完。{tip}"})
            return _j(_run_batched(market, start, end, code))
    except MemoryError:
        return _j({"ok": False, "error": "浏览器内存不够：时间范围太长、数据太多。请缩短时间范围再试"})
    except Exception as e:  # noqa: BLE001
        import traceback

        return _j({"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(limit=-3)[-2000:]})


# 规则代码里出现这些写法，说明要在股票之间比较，不能分批算（宁可多判，不分批只是提示缩短范围）
CROSS = re.compile(r"""groupby\(\s*\[?\s*["']date|\.rank\(|nlargest|nsmallest|quantile\(|\.head\(|axis\s*=\s*1|["']rank["']|pct\s*=\s*True""")


def _cross_sectional(code: str) -> bool:
    return bool(CROSS.search(code))


def _run_batched(market: str, start: str, end: str, code: str) -> dict:
    """股票分几批，每批数据量在 BATCH_ROWS 以内，分别跑同一段规则，再把结果合起来。"""
    d = STATE["daily"]
    mask = np.ones(len(d), dtype=bool)
    mask &= (d["date"] >= pd.Timestamp(start) - pd.Timedelta(days=prompts.LOOKBACK_DAYS)).to_numpy()
    mask &= (d["date"] <= pd.Timestamp(end)).to_numpy()
    rows_per_code = np.bincount(STATE["idx"][mask], minlength=len(STATE["cats"]))
    del mask
    groups, cur, acc = [], [], 0
    for i, n in enumerate(rows_per_code):
        if cur and acc + n > BATCH_ROWS:
            groups.append(cur)
            cur, acc = [], 0
        cur.append(i)
        acc += n
    if cur:
        groups.append(cur)
    results = []
    try:
        for g in groups:
            sel = np.zeros(len(STATE["cats"]), dtype=bool)
            sel[g] = True
            BATCH["codes"] = sel
            results.append(runner.execute(market, start, end, code))
            gc.collect()
    finally:
        BATCH["codes"] = None
    cols = next((r["columns"] for r in results if r["rows"]), results[0]["columns"])
    if any(r["rows"] and r["columns"] != cols for r in results):
        raise ValueError("分批计算时各批返回的列不一样，请把规则写成每只股票单独判断")
    pairs = [(row, b) for r in results for row, b in zip(r["rows"], r["builtin_rows"])]
    sort = next((r["default_sort"] for r in results if r["default_sort"] and r["default_sort"]["column"] in cols), None)
    if sort:  # 各批各自排好序，合起来以后按同一列重新排
        k = cols.index(sort["column"])
        num = [p for p in pairs if isinstance(p[0][k], (int, float))]
        rest = [p for p in pairs if not isinstance(p[0][k], (int, float))]
        num.sort(key=lambda p: p[0][k], reverse=sort["dir"] == "desc")
        pairs = num + rest
    pairs = pairs[: runner.config.MAX_ROWS]
    out = dict(results[0])
    out.update(rows=[p[0] for p in pairs], builtin_rows=[p[1] for p in pairs], columns=cols, default_sort=sort,
               total=sum(r["total"] for r in results), universe=sum(r["universe"] for r in results),
               load_seconds=round(sum(r["load_seconds"] for r in results), 2),
               run_seconds=round(sum(r["run_seconds"] for r in results), 2), batches=len(groups))
    return out


def examples() -> str:
    return _j(EXAMPLES)


# ---------------------------------------------------------------- 给 AI 的提示词（和本地版同一份 prompts.py）
def _industry_desc() -> tuple[str, str]:
    if not _has_data():
        return "示例行业", "空"
    names = sorted(n for n in STATE["stocks"]["industry"].dropna().unique() if n)
    src = (STATE["manifest"].get("meta") or {}).get("industry_source") or "行业"
    if not names:
        return "暂无行业数据，均为空字符串", "空"
    l2 = "申万二级行业名称" if src.startswith("申万") else "空"
    return f"{src}行业名称，取值：{'、'.join(names)}；没有分类的为空字符串", l2


def system_prompt() -> str:
    return prompts.system_prompt(*_industry_desc())


def user_message(query: str, start: str, end: str) -> str:
    return prompts.user_message(query, start, end)


def refine_message(refine_json: str, start: str, end: str) -> str:
    return prompts.refine_message(json.loads(refine_json), start, end)


def fix_message(error: str, tb: str) -> str:
    return prompts.fix_message(error, tb)


def parse_answer(text: str) -> str:
    try:
        return _j({"ok": True, "plan": prompts.parse_answer(text)})
    except Exception as e:  # noqa: BLE001
        return _j({"ok": False, "error": f"无法解析模型输出：{e}"})


def rewrote(old: str, new: str, change: str) -> bool:
    return prompts.rewrote(old, new, change)


def call_params(provider: str, model: str, effort: str) -> str:
    r, extra = prompts.call_params(provider, model, effort or None)
    return _j({"reasoning": r, "extra": extra, "max_tokens": prompts.MAX_TOKENS.get(extra.get("reasoning_effort") or "", 32000 if r else 8000)})


def fix_model(provider: str, model: str) -> str:
    m, extra = prompts.fix_model(provider, model)
    r = prompts.call_params(provider, m, extra.get("reasoning_effort"))[0] if extra else prompts.looks_reasoning(provider, m)
    return _j({"model": m, "extra": extra, "reasoning": r,
               "max_tokens": prompts.MAX_TOKENS.get(extra.get("reasoning_effort") or "", 32000 if r else 8000)})


REWRITE_RETRY = prompts.REWRITE_RETRY
REWRITE_NOTE = prompts.REWRITE_NOTE
