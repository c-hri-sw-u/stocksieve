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
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION = "0.29.3"  # 和 web/worker.js 里的版本保持一致
# jsdelivr 偶尔返回 503（2026-11-16 的 data 流程就因此没部署成功）：每个地址重试几次，再换同一份文件的其他镜像
MIRRORS = [f"https://{host}/pyodide/v{VERSION}/full/" for host in ("cdn.jsdelivr.net", "fastly.jsdelivr.net", "gcore.jsdelivr.net")]
TRIES = 3
CORE = ["pyodide.js", "pyodide.asm.js", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json"]
PACKAGES = ["numpy", "pandas", "pyarrow"]


def fetch(name: str, cache: Path) -> Path:
    p = cache / name
    if p.exists():
        return p
    print(f"  下载 {name}")
    tmp = p.with_suffix(p.suffix + ".tmp")
    errors = []
    for base in list(MIRRORS):
        for i in range(TRIES):
            try:
                with urllib.request.urlopen(base + name, timeout=120) as r, open(tmp, "wb") as f:
                    shutil.copyfileobj(r, f)
                tmp.replace(p)
                if base != MIRRORS[0]:  # 之后的文件先用这个能下载的地址
                    MIRRORS.remove(base)
                    MIRRORS.insert(0, base)
                return p
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                errors.append(f"{base}{name}: {e}")
                if isinstance(e, urllib.error.HTTPError) and e.code == 404:
                    break  # 这个镜像没有这个文件，重试也没用
                wait = 2 ** (i + 1)
                print(f"    失败（{e}），{wait} 秒后重试", flush=True)
                time.sleep(wait)
        print(f"    换一个地址", flush=True)
    tmp.unlink(missing_ok=True)
    sys.exit(f"{name} 下载失败：\n" + "\n".join(errors[-len(MIRRORS):]))


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
