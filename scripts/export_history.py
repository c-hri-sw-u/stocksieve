"""把下载好的「更早的历史日线」（hist/ 下每只股票一个文件）按年打包，给网页的「时间范围」往前补全用。

更早的历史只准备一次，之后不再变：
- 文件名 hist-YYYY.parquet，格式和 export_site_data.py 导出的日线一样（float32、zstd）
- 最后一个文件比现有数据的第一天（base_from）多下载了几周，两段数据在这几天重叠。
  前复权价格会随以后的分红送转整体变化，浏览器用重叠那天的收盘价算出每只股票的比例，把历史价格换到和现有数据一致
- history.json 记录每个文件的大小、sha256、日期范围

用法（GitHub Actions 的 history 流程里）：
    python scripts/export_history.py --hist data/cn_a/hist --from 2015-01-01 --base-from 2023-11-13 --out data/history
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.data import DAILY_COLUMNS  # noqa: E402
from export_site_data import F32, write  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", required=True, help="每只股票一个 parquet 的目录")
    ap.add_argument("--from", dest="from_", required=True, help="历史从哪天开始 YYYY-MM-DD")
    ap.add_argument("--base-from", required=True, help="现有数据（每天更新的那部分）的第一天 YYYY-MM-DD")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("hist-*.parquet"):
        f.unlink()

    parts = [pd.read_parquet(f) for f in sorted(Path(args.hist).glob("*.parquet"))]
    parts = [p for p in parts if not p.empty]
    if not parts:
        sys.exit("没有下载到任何历史数据")
    daily = pd.concat(parts, ignore_index=True)[DAILY_COLUMNS]
    daily["date"] = pd.to_datetime(daily["date"]).astype("datetime64[ms]")
    daily = daily[daily["date"] >= pd.Timestamp(args.from_)]
    for c in F32:
        daily[c] = daily[c].astype("float32")
    daily = daily.drop_duplicates(["code", "date"]).sort_values(["date", "code"])

    files = []
    for year, g in daily.groupby(daily["date"].dt.year):
        info = write(g.sort_values(["code", "date"]), out / f"hist-{year}.parquet")
        info.update(kind="history", year=int(year), **{"from": g["date"].min().strftime("%Y-%m-%d"), "to": g["date"].max().strftime("%Y-%m-%d")})
        files.append(info)

    meta = {
        "version": 1,
        "from": daily["date"].min().strftime("%Y-%m-%d"),
        "to": daily["date"].max().strftime("%Y-%m-%d"),
        "base_from": args.base_from,
        "stocks": int(daily["code"].nunique()),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total_size": sum(f["size"] for f in files),
        "files": files,
    }
    (out / "history.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), "utf-8")
    print(f"历史数据：{len(files)} 个文件，共 {meta['total_size'] / 1e6:.1f} MB，{meta['stocks']} 只，{meta['from']} ~ {meta['to']}")


if __name__ == "__main__":
    main()
