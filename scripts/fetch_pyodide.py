"""把网页版需要的 Pyodide（浏览器里的 Python）下载到 site/pyodide/，和网页放在同一个网站上。

不再依赖 cdn.jsdelivr.net：它在部分网络环境（比如国内一些网络）下连不上，计算环境就加载失败。
只下载核心文件和 numpy / pandas / pyarrow 及其依赖（按 pyodide-lock.json 解析），约 30 MB。
下载过的文件缓存在 data/pyodide-cache/，本地反复生成不用重下。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION = "0.29.3"  # 和 web/worker.js 里的版本保持一致
CDN = f"https://cdn.jsdelivr.net/pyodide/v{VERSION}/full/"
CORE = ["pyodide.js", "pyodide.asm.js", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json"]
PACKAGES = ["numpy", "pandas", "pyarrow"]


def fetch(name: str, cache: Path) -> Path:
    p = cache / name
    if not p.exists():
        print(f"  下载 {name}")
        tmp = p.with_suffix(p.suffix + ".tmp")
        with urllib.request.urlopen(CDN + name, timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.replace(p)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "site" / "pyodide"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "pyodide-cache" / VERSION))
    args = ap.parse_args()
    out, cache = Path(args.out), Path(args.cache)
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    lock = json.loads(fetch("pyodide-lock.json", cache).read_text())
    pkgs = lock["packages"]
    need, stack = set(), list(PACKAGES)
    while stack:
        k = stack.pop()
        if k in need:
            continue
        need.add(k)
        stack += [d.lower() for d in pkgs[k].get("depends", [])]
    files = CORE + [pkgs[k]["file_name"] for k in sorted(need)]
    total = 0
    for name in files:
        src = fetch(name, cache)
        if name.endswith(".whl"):  # 和 lock 里的校验值对一下，防止下载不完整
            k = next(k for k in need if pkgs[k]["file_name"] == name)
            if hashlib.sha256(src.read_bytes()).hexdigest() != pkgs[k]["sha256"]:
                src.unlink()
                sys.exit(f"{name} 校验失败，请重试")
        shutil.copy(src, out / name)
        total += src.stat().st_size
    print(f"Pyodide {VERSION}：{len(files)} 个文件，{total / 1e6:.1f} MB → {out}（包：{', '.join(sorted(need))}）")


if __name__ == "__main__":
    main()
