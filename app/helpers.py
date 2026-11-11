"""筛选代码里可以直接用的技术指标函数。

所有函数都自动「按股票分组」计算：传入 daily 里的一列（例如 daily["close"]），
返回同样索引的一列，不会把上一只股票的数据串到下一只。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ST_10PCT_FROM = pd.Timestamp("2026-07-06")  # 主板 ST 涨跌幅从 5% 调整为 10% 的日期

HELPER_DOC = """
可直接调用的函数（都已自动按股票分组，传入 daily 的列即可，返回与输入对齐的 Series）：
  ma(s, n)              n 日简单均线
  ema(s, n)             n 日指数均线
  ref(s, n=1)           n 日前的值（同一只股票内 shift）
  ret(s, n)             n 日涨跌幅，单位 %，等于 (s / ref(s, n) - 1) * 100
  highest(s, n)         n 日最高值（含当日）
  lowest(s, n)          n 日最低值（含当日）
  std(s, n)             n 日标准差
  rsi(s, n=14)          RSI
  macd(s, fast=12, slow=26, signal=9)  返回 (dif, dea, hist)，hist = 2*(dif-dea)
  boll(s, n=20, k=2)    返回 (mid, upper, lower)
  kdj(high, low, close, n=9)  返回 (k, d, j)
  cross_up(a, b)        a 当日上穿 b（前一日 a<=b 且当日 a>b），b 可以是数字
  cross_down(a, b)      a 当日下穿 b
  count(cond, n)        最近 n 日内 cond 为 True 的天数
  in_range(daily, start, end)  返回布尔 Series：date 在 [start, end] 内
  latest(df)            每只股票取最后一行（一般在计算完指标并截到 end 之后用）
  limit_up(daily, stocks)    当日是否涨停（布尔 Series）。已按板块区分涨停幅度：主板 10%、创业板(30开头)/科创板(68开头) 20%、
                             北交所(4/8/92开头) 30%、主板 ST 在 2026-07-06 前 5%（之后 10%）；新股上市前 5 个交易日不算。凡是「涨停」一律用它，不要自己写阈值
  limit_down(daily, stocks)  当日是否跌停，规则同上
  streak(cond)          连续计数：截至当日，cond 已经连续为 True 多少天（False 当天为 0）
  runs(daily, cond)     把每只股票里 cond 连续为 True 的每一段列出来，返回 DataFrame：
                        code, start(该段第一天), end(该段最后一天), length(连续天数)
  board(code)           股票所属板块：主板 / 创业板 / 科创板 / 北交所（传 code 列返回 Series）
""".strip()


def make_helpers(daily: pd.DataFrame) -> dict:
    codes = daily["code"]

    def _g(s: pd.Series):
        return s.groupby(codes.reindex(s.index))

    def ma(s, n):
        return _g(s).transform(lambda x: x.rolling(n, min_periods=n).mean())

    def ema(s, n):
        return _g(s).transform(lambda x: x.ewm(span=n, adjust=False).mean())

    def ref(s, n=1):
        return _g(s).shift(n)

    def ret(s, n):
        return (s / ref(s, n) - 1) * 100

    def highest(s, n):
        return _g(s).transform(lambda x: x.rolling(n, min_periods=n).max())

    def lowest(s, n):
        return _g(s).transform(lambda x: x.rolling(n, min_periods=n).min())

    def std(s, n):
        return _g(s).transform(lambda x: x.rolling(n, min_periods=n).std())

    def rsi(s, n=14):
        diff = _g(s).diff()
        up = diff.clip(lower=0)
        down = -diff.clip(upper=0)
        au = _g(up).transform(lambda x: x.ewm(alpha=1 / n, adjust=False).mean())
        ad = _g(down).transform(lambda x: x.ewm(alpha=1 / n, adjust=False).mean())
        return 100 - 100 / (1 + au / ad.replace(0, np.nan))

    def macd(s, fast=12, slow=26, signal=9):
        dif = ema(s, fast) - ema(s, slow)
        dea = ema(dif, signal)
        return dif, dea, 2 * (dif - dea)

    def boll(s, n=20, k=2):
        mid = ma(s, n)
        sd = std(s, n)
        return mid, mid + k * sd, mid - k * sd

    def kdj(high, low, close, n=9):
        hh, ll = highest(high, n), lowest(low, n)
        rsv = (close - ll) / (hh - ll).replace(0, np.nan) * 100
        k = _g(rsv).transform(lambda x: x.ewm(alpha=1 / 3, adjust=False).mean())
        d = _g(k).transform(lambda x: x.ewm(alpha=1 / 3, adjust=False).mean())
        return k, d, 3 * k - 2 * d

    def _prev(x, like):
        if isinstance(x, pd.Series):
            return ref(x, 1)
        return pd.Series(x, index=like.index)

    def cross_up(a, b):
        return (a > b) & (_prev(a, a) <= _prev(b, a))

    def cross_down(a, b):
        return (a < b) & (_prev(a, a) >= _prev(b, a))

    def count(cond, n):
        return _g(cond.astype(float)).transform(lambda x: x.rolling(n, min_periods=1).sum())

    def in_range(df, start, end):
        return (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))

    def latest(df):
        return df.sort_values(["code", "date"]).groupby("code", as_index=False).tail(1)

    def _limit_pct(df, stocks):
        code = df["code"].astype(str)
        lim = pd.Series(10.0, index=df.index)
        lim[code.str.startswith(("30", "68"))] = 20.0
        lim[code.str.startswith(("4", "8", "92"))] = 30.0
        if stocks is not None and "name" in stocks:
            # 主板 ST 股原来是 5%，从 2026-07-06 起和普通主板一样是 10%（从行情数据里看出来的切换点）。
            # 只能按「现在」的名称判断是否 ST，历史上摘帽 / 戴帽的时间无法区分，属于近似。
            st = set(stocks.loc[stocks["name"].astype(str).str.contains("ST", na=False), "code"])
            lim[code.isin(st) & (lim == 10.0) & (df["date"] < ST_10PCT_FROM)] = 5.0
        return lim

    def _new_listing(df):
        # 新股上市前 5 个交易日不设或另设涨跌幅限制，不算涨停/跌停（只针对在数据起点之后才上市的股票）
        first_day = df["date"].min()
        n = df.groupby("code").cumcount()
        listed_later = df.groupby("code")["date"].transform("min") > first_day
        return listed_later & (n < 5)

    def limit_up(df, stocks=None):
        # 前复权价格会让涨跌幅有零点几的误差，所以留 0.2 个百分点的余量
        return (df["pct_chg"] >= _limit_pct(df, stocks) - 0.2) & ~_new_listing(df)

    def limit_down(df, stocks=None):
        return (df["pct_chg"] <= -(_limit_pct(df, stocks) - 0.2)) & ~_new_listing(df)

    def board(code):
        c = pd.Series(code).astype(str)
        out = pd.Series("主板", index=c.index)
        out[c.str.startswith(("300", "301"))] = "创业板"
        out[c.str.startswith(("688", "689"))] = "科创板"
        out[c.str.startswith(("4", "8", "92"))] = "北交所"
        return out if isinstance(code, pd.Series) else out.iloc[0]

    def streak(cond):
        c = cond.astype(bool)
        grp = (~c).groupby(codes.reindex(c.index)).cumsum()
        return c.astype(int).groupby([codes.reindex(c.index), grp]).cumsum()

    def runs(df, cond):
        c = cond.astype(bool)
        d = df.loc[:, ["code", "date"]].assign(_c=c.values)
        d["_g"] = (d["_c"] != d.groupby("code")["_c"].shift()).groupby(d["code"]).cumsum()
        d = d[d["_c"]]
        out = d.groupby(["code", "_g"]).agg(start=("date", "first"), end=("date", "last"), length=("date", "size")).reset_index()
        return out[["code", "start", "end", "length"]]

    return {
        "ma": ma,
        "ema": ema,
        "ref": ref,
        "ret": ret,
        "highest": highest,
        "lowest": lowest,
        "std": std,
        "rsi": rsi,
        "macd": macd,
        "boll": boll,
        "kdj": kdj,
        "cross_up": cross_up,
        "cross_down": cross_down,
        "count": count,
        "in_range": in_range,
        "latest": latest,
        "limit_up": limit_up,
        "limit_down": limit_down,
        "streak": streak,
        "runs": runs,
        "board": board,
    }
