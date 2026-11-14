"""浏览器（Pyodide）里的入口：复用 app/ 下同一份 Python 代码，只把「读数据」换成读本机缓存的文件。

worker.js 调这里的函数，参数和返回值都是 JSON 字符串。
"""
from __future__ import annotations

import json
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
# - 常驻内存里的价格等用 float32（和下载的文件一样，不损失精度），每次筛选复制出区间时再转成 float64
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
# 实测 900 万行（约全市场 6 年）时峰值约 2.3 GB，再多就可能把计算环境撑崩。
MAX_ROWS = 9_000_000


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
    n = int(mask.sum())
    if n > MAX_ROWS:
        # 算出最早能从哪天开始：从结束日往前数，行数不超过上限（再留出预热的天数）
        dates = d["date"].to_numpy()[mask]
        u, cnt = np.unique(dates, return_counts=True)
        keep = np.cumsum(cnt[::-1]) <= MAX_ROWS
        earliest = pd.Timestamp(u[::-1][keep][-1]) + pd.Timedelta(days=prompts.LOOKBACK_DAYS)
        raise RangeTooLarge(f"时间范围太长，浏览器内存不够算这么多数据。请把开始日期改到 {earliest:%Y-%m-%d} 或之后再试")
    out = _frame(d["date"].to_numpy()[mask], d["code"].to_numpy()[mask], lambda c: d[c].to_numpy()[mask], np.float64)
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
        return _j(runner.execute(market, start, end, code))
    except RangeTooLarge as e:
        return _j({"ok": False, "error": str(e)})
    except MemoryError:
        return _j({"ok": False, "error": "浏览器内存不够：时间范围太长、数据太多。请缩短时间范围再试"})
    except Exception as e:  # noqa: BLE001
        import traceback

        return _j({"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(limit=-3)[-2000:]})


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
