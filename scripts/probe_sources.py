"""在 GitHub Actions 上逐个测试 A 股数据源能不能连通、要多久。

每一项单独计时，单项最多等 30 秒；任何一项失败都不影响其他项，也不会让整个任务失败。
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import akshare as ak  # noqa: E402

TIMEOUT = 30
SYMBOLS = ["600519", "000001", "300750", "688981", "830799"]
pool = ThreadPoolExecutor(max_workers=4)
rows: list[tuple[str, str, float, str]] = []


def tx_symbol(code: str) -> str:
    if code.startswith(("92", "4", "8")):
        return "bj" + code
    return ("sh" if code.startswith(("6", "9")) else "sz") + code


def probe(name: str, fn):
    t0 = time.time()
    try:
        n = pool.submit(fn).result(timeout=TIMEOUT)
        rows.append((name, "成功", time.time() - t0, f"{n} 行"))
    except FutTimeout:
        rows.append((name, "超时", time.time() - t0, f"超过 {TIMEOUT} 秒"))
    except Exception as e:  # noqa: BLE001
        rows.append((name, "失败", time.time() - t0, f"{type(e).__name__}: {str(e)[:120]}"))
    r = rows[-1]
    print(f"{r[0]:<28} {r[1]:<4} {r[2]:6.1f}s  {r[3]}", flush=True)


def end_date():
    return time.strftime("%Y%m%d")


def start_date():
    return time.strftime("%Y%m%d", time.localtime(time.time() - 365 * 86400))


# 股票列表 / 全市场快照
probe("交易所股票列表", lambda: len(ak.stock_info_a_code_name()))
probe("新浪全市场行情(第1页)", lambda: len(__import__("requests").get(
    "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData",
    params={"page": 1, "num": 100, "sort": "symbol", "asc": 1, "node": "hs_a", "symbol": "", "_s_r_a": "page"},
    timeout=20).json()))
probe("东财全市场快照 spot_em", lambda: len(ak.stock_zh_a_spot_em()))

# 日线
for c in SYMBOLS:
    probe(f"腾讯日线 {c}", lambda c=c: len(ak.stock_zh_a_hist_tx(symbol=tx_symbol(c), start_date=start_date(), end_date=end_date(), adjust="qfq")))
for c in SYMBOLS:
    probe(f"东财日线 {c}", lambda c=c: len(ak.stock_zh_a_hist(symbol=c, period="daily", start_date=start_date(), end_date=end_date(), adjust="qfq")))

# 交易日历、行业
probe("新浪交易日历", lambda: len(ak.tool_trade_date_hist_sina()))
probe("申万行业表(乐咕乐股)", lambda: len(ak.sw_index_third_info()))
probe("申万三级成分(乐咕乐股)", lambda: len(ak.sw_index_third_cons(symbol="850111.SI")))
probe("新浪行业列表", lambda: len(ak.stock_sector_spot(indicator="新浪行业")))
probe("新浪行业成分", lambda: len(ak.stock_sector_detail(sector="new_blhy")))

# 完整的新浪全市场快照（增量更新用），给 120 秒
TIMEOUT = 120
from app.sync import fetch_spot_sina  # noqa: E402
probe("新浪全市场快照(全部页)", lambda: len(fetch_spot_sina(max_seconds=110)))

print("\n| 数据源 | 结果 | 耗时 | 说明 |\n|---|---|---|---|")
for name, ok, sec, note in rows:
    print(f"| {name} | {ok} | {sec:.1f}s | {note} |")
sys.stdout.flush()
os._exit(0)  # 超时的线程可能还在等，直接退出
