"""把 data/cn_a 里的行情导出成给网页用的静态数据（site/data/）。

- 日线按年拆成文件，当年按月拆（每天只有当月那个小文件会变，浏览器只需重新下载它）
- 列精简、价格等用 float32、zstd 压缩
- manifest.json 记录每个文件的大小、sha256、日期范围，浏览器据此只下载缺的或变了的文件

用法：
    python scripts/export_site_data.py                     # 全量
    python scripts/export_site_data.py --limit 200 --years 1   # 本地测试用的小切片
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.data import DAILY_COLUMNS, STOCK_COLUMNS  # noqa: E402

F32 = ["open", "high", "low", "close", "volume", "amount", "pct_chg", "turnover"]


def write(df: pd.DataFrame, path: Path) -> dict:
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="zstd", compression_level=9, use_dictionary=["code"] if "code" in df else True)
    data = path.read_bytes()
    return {"name": path.name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "rows": len(df)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data" / "cn_a"))
    ap.add_argument("--out", default=str(ROOT / "site" / "data"))
    ap.add_argument("--limit", type=int, default=0, help="只导出前 N 只（测试用）")
    ap.add_argument("--years", type=float, default=0, help="只导出最近 N 年（测试用）")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.parquet"):
        f.unlink()

    daily = pd.read_parquet(src / "daily.parquet")
    stocks = pd.read_parquet(src / "stocks.parquet")
    if "industry2" not in stocks:
        stocks["industry2"] = ""
    meta = json.loads((src / "meta.json").read_text("utf-8")) if (src / "meta.json").exists() else {}

    if args.limit:
        keep = sorted(daily["code"].unique())[: args.limit]
        daily = daily[daily["code"].isin(keep)]
    if args.years:
        daily = daily[daily["date"] >= daily["date"].max() - pd.Timedelta(days=int(365 * args.years))]
    stocks = stocks[stocks["code"].isin(set(daily["code"]))]

    daily = daily[DAILY_COLUMNS].copy()
    daily["date"] = pd.to_datetime(daily["date"]).astype("datetime64[ms]")
    for c in F32:
        daily[c] = daily[c].astype("float32")
    daily = daily.sort_values(["date", "code"])

    files = []
    last = daily["date"].max()
    for year, g in daily.groupby(daily["date"].dt.year):
        if year < last.year:
            parts = [(f"daily-{year}.parquet", g)]
        else:  # 当年按月拆
            parts = [(f"daily-{year}-{m:02d}.parquet", gm) for m, gm in g.groupby(g["date"].dt.month)]
        for name, part in parts:
            info = write(part.sort_values(["code", "date"]), out / name)
            info.update(kind="daily", from_=part["date"].min().strftime("%Y-%m-%d"), to=part["date"].max().strftime("%Y-%m-%d"))
            files.append(info)
    st = stocks[STOCK_COLUMNS].copy()
    for c in ["price", "pe", "pb", "total_mv", "float_mv"]:
        st[c] = st[c].astype("float64")
    info = write(st, out / "stocks.parquet")
    info.update(kind="stocks")
    files.append(info)

    for f in files:
        if "from_" in f:
            f["from"] = f.pop("from_")
    manifest = {
        "version": 1,
        "market": "cn_a",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "first_date": daily["date"].min().strftime("%Y-%m-%d"),
        "last_date": last.strftime("%Y-%m-%d"),
        "stocks": int(daily["code"].nunique()),
        "total_size": sum(f["size"] for f in files),
        "meta": {k: meta.get(k) for k in ("synced_at", "list_source", "daily_source", "industry_source", "years", "last_mode") if k in meta},
        "files": files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), "utf-8")
    print(f"导出 {len(files)} 个文件，共 {manifest['total_size'] / 1e6:.1f} MB，{manifest['stocks']} 只，{manifest['first_date']} ~ {manifest['last_date']}")


if __name__ == "__main__":
    main()
