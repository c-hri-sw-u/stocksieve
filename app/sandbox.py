"""运行筛选代码的受限环境。

两层防护：
1. 静态检查（AST）：禁止 import、双下划线/私有属性、文件和网络相关的方法名。
2. 在独立子进程里执行，限时，只给 pandas / numpy / 指标函数和少量内置函数。
这是给自己用的工具级别的防护，不是面向公网多用户的安全沙箱。
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys

from . import config

FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "open", "input", "globals", "locals", "vars", "getattr",
    "setattr", "delattr", "__import__", "breakpoint", "help", "exit", "quit", "memoryview",
    "type", "object", "super", "classmethod", "staticmethod", "property",
}
FORBIDDEN_ATTRS = {
    "system", "popen", "spawn", "fork", "remove", "unlink", "rmdir", "load", "loads", "save",
    "savez", "savetxt", "loadtxt", "genfromtxt", "fromfile", "tofile", "memmap", "ctypeslib",
    "lib", "eval", "pipe", "style", "plot", "options", "set_option", "io", "api", "testing",
}
FORBIDDEN_PREFIXES = ("read_", "to_csv", "to_pickle", "to_parquet", "to_excel", "to_json", "to_hdf",
                      "to_sql", "to_feather", "to_html", "to_latex", "to_clipboard", "to_markdown",
                      "to_stata", "to_xml", "to_orc", "to_gbq", "f_", "gi_", "co_", "tb_", "cr_")


class UnsafeCode(ValueError):
    pass


def validate(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise UnsafeCode(f"语法错误：第 {e.lineno} 行 {e.msg}") from e

    has_screen = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeCode("不允许 import（pd、np 和指标函数已经可以直接用）")
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.ClassDef, ast.AsyncFunctionDef, ast.Await)):
            raise UnsafeCode(f"不允许使用 {type(node).__name__}")
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in FORBIDDEN_NAMES:
                raise UnsafeCode(f"不允许使用名称 {node.id}")
        if isinstance(node, ast.Attribute):
            a = node.attr
            if a.startswith("_") or a in FORBIDDEN_ATTRS or a.startswith(FORBIDDEN_PREFIXES):
                raise UnsafeCode(f"不允许访问属性 .{a}")
        if isinstance(node, ast.FunctionDef) and node.name == "screen":
            has_screen = True
    if not has_screen:
        raise UnsafeCode("代码里需要定义 screen(daily, stocks, start, end) 函数")


def run(market: str, start: str, end: str, code: str) -> dict:
    """在子进程里运行筛选代码，返回 {"ok": True, "columns": [...], "rows": [...], ...} 或 {"ok": False, "error": ...}"""
    try:
        validate(code)
    except UnsafeCode as e:
        return {"ok": False, "error": str(e)}

    payload = json.dumps({"market": market, "start": start, "end": end, "code": code})
    # 子进程拿不到任何 API key
    secret = ("API_KEY", "TOKEN", "SECRET")
    env = {k: v for k, v in os.environ.items() if not any(x in k.upper() for x in secret)}
    env["XG_NO_DOTENV"] = "1"
    root = str(config.ROOT)
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "app.runner"],
            input=payload,
            capture_output=True,
            text=True,
            timeout=config.EXEC_TIMEOUT,
            cwd=root,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"运行超时（超过 {config.EXEC_TIMEOUT} 秒），请简化计算"}
    out = proc.stdout.strip().splitlines()
    if not out:
        return {"ok": False, "error": "执行进程异常退出：" + proc.stderr[-1500:]}
    try:
        return json.loads(out[-1])
    except json.JSONDecodeError:
        return {"ok": False, "error": "无法解析执行结果：" + (proc.stderr or proc.stdout)[-1500:]}
