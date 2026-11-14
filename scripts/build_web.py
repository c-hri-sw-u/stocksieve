"""生成静态网页版（GitHub Pages 用）到 site/。

- 页面：app/static/index.html（和本地版同一份），注入 storage.js + runtime.js，把 /api/... 换成在浏览器里就地实现
- Python：app/ 下的筛选相关源码原样拷到 site/py/，在浏览器的 Pyodide 里运行（不维护两份）
- 数据：默认从 data/cn_a 导出到 site/data/（--no-data 跳过；--limit / --years 只导出一小部分做本地测试）

本地预览：
    python scripts/build_web.py --limit 200 --years 1
    python -m http.server -d site 8080      # 打开 http://127.0.0.1:8080
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PY_FILES = ["app/__init__.py", "app/config.py", "app/data.py", "app/helpers.py", "app/runner.py",
            "app/sandbox.py", "app/examples.py", "app/prompts.py"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "site"))
    ap.add_argument("--no-data", action="store_true", help="不导出数据（数据由别的步骤放进 site/data）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--years", type=float, default=0)
    ap.add_argument("--no-pyodide", action="store_true", help="不把 Pyodide 放进网站（只用 CDN）")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 页面
    html = (ROOT / "app" / "static" / "index.html").read_text("utf-8")
    marker = "<script>\nconst $ = (s) => document.querySelector(s);"
    assert marker in html, "index.html 结构变了，找不到主脚本"
    html = html.replace(marker, '<script src="storage.js"></script>\n<script src="runtime.js"></script>\n' + marker, 1)
    (out / "index.html").write_text(html, "utf-8")
    for f in ["storage.js", "runtime.js", "worker.js"]:
        shutil.copy(ROOT / "web" / f, out / f)

    # Python 源码（和本地版同一份）
    py = out / "py"
    if py.exists():
        shutil.rmtree(py)
    for f in PY_FILES:
        (py / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / f, py / f)
    shutil.copy(ROOT / "web" / "xg_web.py", py / "xg_web.py")
    (py / "files.json").write_text(json.dumps(PY_FILES + ["xg_web.py"]), "utf-8")

    from app.examples import EXAMPLES
    (out / "examples.json").write_text(json.dumps(EXAMPLES, ensure_ascii=False), "utf-8")
    (out / ".nojekyll").write_text("")

    if not args.no_pyodide:  # 自托管 Pyodide，不依赖 cdn.jsdelivr.net
        subprocess.run([sys.executable, str(ROOT / "scripts" / "fetch_pyodide.py"), "--out", str(out / "pyodide")], check=True)

    if not args.no_data:
        cmd = [sys.executable, str(ROOT / "scripts" / "export_site_data.py"), "--out", str(out / "data")]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if args.years:
            cmd += ["--years", str(args.years)]
        subprocess.run(cmd, check=True)
    print(f"已生成 {out}")


if __name__ == "__main__":
    main()
