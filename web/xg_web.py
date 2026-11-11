"""浏览器（Pyodide）里的入口：复用 app/ 下同一份 Python 代码，只把「读数据」换成读本机缓存的文件。

worker.js 调这里的函数，参数和返回值都是 JSON 字符串。
"""
from __future__ import annotations

import json
import time

import pandas as pd

from app import data, prompts, runner, sandbox
from app.examples import EXAMPLES

STATE = {"daily": None, "stocks": None, "manifest": {}}


def _j(x) -> str:
    return json.dumps(x, ensure_ascii=False, default=str)


# ---------------------------------------------------------------- 数据
def load_files(manifest_json: str, folder: str) -> str:
    """worker 已经把本机缓存的文件写进 Pyodide 的虚拟文件夹 folder。全部读进内存，之后每次筛选不再读盘。"""
    import os

    t0 = time.time()
    manifest = json.loads(manifest_json)
    parts, stocks = [], None
    for f in manifest["files"]:
        path = os.path.join(folder, f["name"])
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        if f["kind"] == "stocks":
            stocks = df
        else:
            parts.append(df)
    if not parts or stocks is None:
        STATE.update(daily=None, stocks=None, manifest={})
        return _j({"ok": False, "error": "本机还没有完整的数据"})
    daily = pd.concat(parts, ignore_index=True)
    daily["date"] = pd.to_datetime(daily["date"])
    for c in ["open", "high", "low", "close", "volume", "amount", "pct_chg", "turnover"]:
        daily[c] = daily[c].astype("float64")
    daily = daily.sort_values(["code", "date"]).reset_index(drop=True)
    if "industry2" not in stocks:
        stocks["industry2"] = ""
    STATE.update(daily=daily, stocks=stocks.reset_index(drop=True), manifest=manifest)
    return _j({"ok": True, "rows": len(daily), "stocks": int(daily["code"].nunique()), "seconds": round(time.time() - t0, 2)})


def _has_data() -> bool:
    return STATE["daily"] is not None


def _load(market, start=None, end=None):
    if not _has_data():
        d, s = data._demo_tables()
        d = d.copy()
    else:
        d, s = STATE["daily"], STATE["stocks"]
    if start:
        d = d[d["date"] >= pd.Timestamp(start) - pd.Timedelta(days=prompts.LOOKBACK_DAYS)]
    if end:
        d = d[d["date"] <= pd.Timestamp(end)]
    return d.reset_index(drop=True), s.reset_index(drop=True)


# 把 app.data 里「读文件」的地方换成读内存
data.load = _load
data.use_demo = lambda market: not _has_data()


# ---------------------------------------------------------------- 筛选
def run(market: str, start: str, end: str, code: str) -> str:
    try:
        sandbox.validate(code)
    except sandbox.UnsafeCode as e:
        return _j({"ok": False, "error": str(e)})
    try:
        return _j(runner.execute(market, start, end, code))
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
