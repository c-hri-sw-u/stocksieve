"""子进程入口：从 stdin 读 JSON，加载数据，执行 screen()，把结果 JSON 打到 stdout 最后一行。"""
from __future__ import annotations

import builtins
import json
import sys
import time
import traceback

import numpy as np
import pandas as pd

from . import config, data
from .helpers import make_helpers

SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in [
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "isinstance",
        "len", "list", "map", "max", "min", "print", "range", "reversed", "round", "set", "slice",
        "sorted", "str", "sum", "tuple", "zip", "True", "False", "None", "ValueError", "KeyError",
        "Exception", "ZeroDivisionError",
    ]
}


def _limit_resources() -> None:
    try:
        import resource

        cpu = config.EXEC_TIMEOUT + 5
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    except Exception:
        pass


def _clean(v):
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else round(float(v), 4)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    return v


def execute(market: str, start: str, end: str, code: str) -> dict:
    t0 = time.time()
    daily, stocks = data.load(market, start, end)
    t_load = time.time() - t0

    g = {"__builtins__": SAFE_BUILTINS, "pd": pd, "np": np, **make_helpers(daily)}
    exec(compile(code, "<screen>", "exec"), g)
    screen = g.get("screen")
    if not callable(screen):
        raise ValueError("没有找到 screen 函数")

    result = screen(daily.copy(), stocks.copy(), pd.Timestamp(start), pd.Timestamp(end))
    if isinstance(result, pd.Series):
        result = result.to_frame()
    if not isinstance(result, pd.DataFrame):
        raise ValueError(f"screen 需要返回 DataFrame，实际返回了 {type(result).__name__}")
    if "code" not in result.columns:
        if result.index.name == "code":
            result = result.reset_index()
        else:
            raise ValueError("返回的 DataFrame 需要有 code 列")

    result = result.drop_duplicates("code")
    total = len(result)
    result = result.head(config.MAX_ROWS)
    extra = [c for c in result.columns if c not in ("code", "name", "industry", "date")]
    info = stocks[["code", "name"]]
    merged = result[["code", *extra]].merge(info, on="code", how="left")
    cols = ["code", "name", *extra]
    merged = merged[cols]

    rows = [[_clean(v) for v in row] for row in merged.itertuples(index=False, name=None)]
    bcols, brows = builtin_columns(merged["code"], daily, stocks, start, end)
    return {
        "builtin_columns": bcols,
        "builtin_rows": brows,
        "default_sort": _default_sort(merged, extra),
        "ok": True,
        "columns": [str(c) for c in cols],
        "rows": rows,
        "total": int(total),
        "universe": int(stocks["code"].nunique()),
        "load_seconds": round(t_load, 2),
        "run_seconds": round(time.time() - t0 - t_load, 2),
        "demo": data.use_demo(market),
    }


# 内置列：不用 AI，直接从行情和股票表算出来，前端可以随时加减
BUILTIN = [
    ("industry", "行业", "股票信息"), ("industry2", "二级行业", "股票信息"), ("price", "最新价", "股票信息"),
    ("pe", "市盈率(动态)", "股票信息"), ("pb", "市净率", "股票信息"),
    ("total_mv", "总市值(亿)", "股票信息"), ("float_mv", "流通市值(亿)", "股票信息"),
    ("end_pct", "涨跌幅%", "结束日当天"), ("end_turnover", "换手率%", "结束日当天"), ("end_amount", "成交额(亿)", "结束日当天"),
    ("range_ret", "区间涨幅%", "所选区间"), ("range_mdd", "最大回撤%", "所选区间"),
    ("range_amount", "日均成交额(亿)", "所选区间"), ("range_vol", "年化波动率%", "所选区间"),
]
DECIMALS = {"price": 2, "pe": 2, "pb": 2, "total_mv": 1, "float_mv": 1, "end_pct": 2, "end_turnover": 2,
            "end_amount": 2, "range_ret": 2, "range_mdd": 2, "range_amount": 2, "range_vol": 1}


def builtin_columns(codes: pd.Series, daily: pd.DataFrame, stocks: pd.DataFrame, start: str, end: str):
    s = stocks.set_index("code").reindex(codes.values)
    out = pd.DataFrame(index=codes.values)
    out["industry"] = s["industry"]
    out["industry2"] = s["industry2"] if "industry2" in s.columns else ""
    out["price"] = s["price"]
    out["pe"] = s["pe"]
    out["pb"] = s["pb"]
    out["total_mv"] = s["total_mv"] / 1e8
    out["float_mv"] = s["float_mv"] / 1e8

    d = daily[daily["code"].isin(set(codes)) & (daily["date"] <= pd.Timestamp(end))]
    last = d.groupby("code").tail(1).set_index("code")
    out["end_pct"] = last["pct_chg"]
    out["end_turnover"] = last["turnover"]
    out["end_amount"] = last["amount"] / 1e8

    r = d[d["date"] >= pd.Timestamp(start)]
    if not r.empty:
        g = r.groupby("code")
        first, lastc = g["close"].first(), g["close"].last()
        out["range_ret"] = (lastc / first - 1) * 100
        dd = r["close"] / g["close"].cummax() - 1
        out["range_mdd"] = dd.groupby(r["code"]).min() * 100
        out["range_amount"] = g["amount"].mean() / 1e8
        out["range_vol"] = r.assign(rr=g["close"].pct_change()).groupby("code")["rr"].std() * np.sqrt(252) * 100
    cols = [{"id": i, "label": lb, "group": gr} for i, lb, gr in BUILTIN]
    rows = []
    for row in out.reindex(columns=[c[0] for c in BUILTIN]).itertuples(index=False, name=None):
        vals = []
        for (cid, _, _), v in zip(BUILTIN, row):
            v = _clean(v)
            if isinstance(v, float) and cid in DECIMALS:
                v = round(v, DECIMALS[cid])
            vals.append("" if v is None and cid.startswith("industry") else v)
        rows.append(vals)
    return cols, rows


def _default_sort(df, extra) -> dict | None:
    """找出规则本身是按哪一列排的：第一个按返回顺序单调的数值列。"""
    import pandas as pd

    for c in extra:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        v = df[c].dropna()
        if len(v) < 2 or v.nunique() < 2:
            continue
        if v.is_monotonic_decreasing:
            return {"column": str(c), "dir": "desc"}
        if v.is_monotonic_increasing:
            return {"column": str(c), "dir": "asc"}
    return None


def main() -> None:
    _limit_resources()
    req = json.loads(sys.stdin.read())
    try:
        out = execute(req["market"], req["start"], req["end"], req["code"])
    except Exception as e:
        tb = traceback.format_exc(limit=-3)
        out = {"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": tb[-2000:]}
    sys.stdout.write("\n" + json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
