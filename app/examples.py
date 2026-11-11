"""写死的示例规则：不调用 AI 也能跑通整个筛选流程，同时作为给 AI 的参考写法。"""

EXAMPLES = [
    {
        "id": "range_gain",
        "title": "区间涨幅超 20% 的大市值股",
        "query": "在所选时间范围内涨幅超过 20%，并且总市值大于 100 亿，按涨幅从高到低排",
        "understanding": "计算每只股票从开始日期到结束日期的收盘价涨幅，保留涨幅 > 20% 且总市值 > 100 亿元的股票，按涨幅降序。",
        "code": '''def screen(daily, stocks, start, end):
    d = daily[in_range(daily, start, end)]
    first = d.groupby("code")["close"].first()
    last = d.groupby("code")["close"].last()
    out = pd.DataFrame({"区间涨幅%": (last / first - 1) * 100, "收盘价": last}).reset_index()
    out = out.merge(stocks[["code", "total_mv"]], on="code")
    out["总市值(亿)"] = out["total_mv"] / 1e8
    out = out[(out["区间涨幅%"] > 20) & (out["total_mv"] > 1e10)]
    return out.sort_values("区间涨幅%", ascending=False)[["code", "区间涨幅%", "收盘价", "总市值(亿)"]]
''',
    },
    {
        "id": "macd_golden",
        "title": "MACD 金叉 + 站上 60 日线",
        "query": "时间范围内出现过 MACD 金叉，而且结束日收盘价在 60 日均线之上",
        "understanding": "用全部历史计算 MACD（12,26,9）和 60 日均线；筛出在所选区间内至少出现一次 DIF 上穿 DEA，且结束日收盘价高于 MA60 的股票。",
        "code": '''def screen(daily, stocks, start, end):
    d = daily[daily["date"] <= end].copy()
    dif, dea, hist = macd(d["close"])
    d["金叉"] = cross_up(dif, dea)
    d["MA60"] = ma(d["close"], 60)
    win = d[in_range(d, start, end)]
    crosses = win.groupby("code")["金叉"].sum().rename("金叉次数")
    last = latest(d)[["code", "close", "MA60"]]
    out = last.merge(crosses.reset_index(), on="code")
    out = out[(out["金叉次数"] > 0) & (out["close"] > out["MA60"])]
    out["高于MA60%"] = (out["close"] / out["MA60"] - 1) * 100
    out = out.rename(columns={"close": "收盘价"})
    return out.sort_values("高于MA60%", ascending=False)[["code", "收盘价", "MA60", "高于MA60%", "金叉次数"]]
''',
    },
    {
        "id": "value_stable",
        "title": "低估值且回撤小",
        "query": "市盈率在 0 到 15 之间、市净率小于 1.5，并且时间范围内最大回撤小于 15%",
        "understanding": "用最新的动态市盈率和市净率筛低估值股票，再计算所选区间内的最大回撤（从区间内最高点到之后最低点的跌幅），保留回撤 < 15% 的。注意：PE/PB 是最新值，不是历史值。",
        "code": '''def screen(daily, stocks, start, end):
    cheap = stocks[(stocks["pe"] > 0) & (stocks["pe"] < 15) & (stocks["pb"] < 1.5)]
    d = daily[in_range(daily, start, end) & daily["code"].isin(cheap["code"])].copy()
    peak = d.groupby("code")["close"].cummax()
    d["dd"] = (d["close"] / peak - 1) * 100
    mdd = d.groupby("code")["dd"].min().rename("最大回撤%").reset_index()
    out = mdd.merge(cheap[["code", "pe", "pb"]], on="code")
    out = out[out["最大回撤%"] > -15].rename(columns={"pe": "市盈率", "pb": "市净率"})
    return out.sort_values("最大回撤%", ascending=False)
''',
    },
]

BY_ID = {e["id"]: e for e in EXAMPLES}
