"""行情数据的读取。

数据统一成两张表：
  daily  长表，每行一只股票一天：date, code, open, high, low, close, volume, amount, pct_chg, turnover
  stocks 每只股票一行：code, name, industry(申万一级), industry2(申万二级), price, pe, pb, total_mv, float_mv
真实数据由 `python -m app.sync` 从 AKShare 下载到 data/<market>/ 下；
没有真实数据或开启 XG_DEMO=1 时，用随机生成的示例数据。
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

DAILY_COLUMNS = ["date", "code", "open", "high", "low", "close", "volume", "amount", "pct_chg", "turnover"]
STOCK_COLUMNS = ["code", "name", "industry", "industry2", "price", "pe", "pb", "total_mv", "float_mv"]


def market_dir(market: str) -> Path:
    return config.DATA_DIR / market


def has_real_data(market: str) -> bool:
    d = market_dir(market)
    return (d / "daily.parquet").exists() and (d / "stocks.parquet").exists()


def use_demo(market: str) -> bool:
    return config.DEMO_MODE or not has_real_data(market)


# ---------------------------------------------------------------- 示例数据
@lru_cache(maxsize=1)
def _demo_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    n = 300
    end = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end - pd.Timedelta(days=365 * 3), end)
    industries = ["银行", "白酒", "半导体", "新能源", "医药", "军工", "汽车", "软件", "有色金属", "房地产"]
    codes = [f"{600000 + i:06d}" if i < 150 else f"{1 + i:06d}" for i in range(n)]
    frames = []
    for code in codes:
        drift = rng.normal(0.0003, 0.0006)
        vol = rng.uniform(0.012, 0.035)
        ret = rng.normal(drift, vol, len(dates))
        close = rng.uniform(4, 120) * np.exp(np.cumsum(ret))
        open_ = close * (1 + rng.normal(0, vol / 3, len(dates)))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, len(dates))))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, len(dates))))
        volume = rng.lognormal(12, 0.6, len(dates)).round()
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "code": code,
                    "open": open_.round(2),
                    "high": high.round(2),
                    "low": low.round(2),
                    "close": close.round(2),
                    "volume": volume,
                    "amount": (volume * close * 100).round(),
                    "turnover": rng.uniform(0.2, 8, len(dates)).round(2),
                }
            )
        )
    daily = pd.concat(frames, ignore_index=True)
    daily["pct_chg"] = (daily.groupby("code")["close"].pct_change() * 100).round(2)
    daily = daily[DAILY_COLUMNS]
    last = daily.groupby("code").tail(1).set_index("code")
    float_shares = rng.uniform(1e8, 5e9, n)
    stocks = pd.DataFrame(
        {
            "code": codes,
            "name": [f"示例股份{i + 1:03d}" for i in range(n)],
            "industry": rng.choice(industries, n),
            "industry2": "",
            "price": last.loc[codes, "close"].values,
            "pe": rng.uniform(-20, 80, n).round(2),
            "pb": rng.uniform(0.5, 10, n).round(2),
        }
    )
    stocks["float_mv"] = (stocks["price"] * float_shares).round()
    stocks["total_mv"] = (stocks["float_mv"] * rng.uniform(1, 1.6, n)).round()
    return daily, stocks[STOCK_COLUMNS]


# ---------------------------------------------------------------- 读取
def load(market: str, start: str | None = None, end: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取截至 end 的全部日线和股票表。

    start 只是用户关心的事件时间段的开始，不用来截数据：规则需要多长的历史就用多长（「历史新低」「上市以来」等），
    时间段由规则代码自己用 in_range 截取。
    """
    if use_demo(market):
        daily, stocks = _demo_tables()
        daily = daily.copy()
    else:
        d = market_dir(market)
        filters = None
        if end:
            filters = [("date", "<=", pd.Timestamp(end))]
        daily = pd.read_parquet(d / "daily.parquet", filters=filters)
        stocks = pd.read_parquet(d / "stocks.parquet")
        if "industry2" not in stocks.columns:
            stocks["industry2"] = ""
    if end:
        daily = daily[daily["date"] <= pd.Timestamp(end)]
    daily = daily.sort_values(["code", "date"]).reset_index(drop=True)
    return daily, stocks.reset_index(drop=True)


def status(market: str) -> dict:
    if use_demo(market):
        daily, stocks = _demo_tables()
        return {
            "market": market,
            "demo": True,
            "stocks": int(len(stocks)),
            "first_date": daily["date"].min().strftime("%Y-%m-%d"),
            "last_date": daily["date"].max().strftime("%Y-%m-%d"),
        }
    meta_file = market_dir(market) / "meta.json"
    meta = json.loads(meta_file.read_text("utf-8")) if meta_file.exists() else {}
    return {"market": market, "demo": False, **meta}
