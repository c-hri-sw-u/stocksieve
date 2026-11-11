"""网站后端。运行：uvicorn app.main:app --reload"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import date
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from . import config, data, sandbox
from .examples import BY_ID, EXAMPLES

STATIC = config.ROOT / "app" / "static"
app = FastAPI(title="选股神器")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class ScreenRequest(BaseModel):
    query: str = Field("", max_length=2000)
    market: str = "cn_a"
    start: date
    end: date
    example_id: str | None = None  # 指定时直接跑写死的示例代码，不调用 AI
    code: str | None = None  # 用户手动修改后的代码
    refine: dict | None = None  # 在当前规则上修改：{base_query, base_understanding, base_code, instruction, result_summary}


def _ai_available() -> bool:
    return config.active_provider() is not None


@app.get("/")
def index():
    # 页面经常改，禁止浏览器缓存旧版本
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/api/meta")
def meta():
    return {
        "markets": config.MARKETS,
        "status": data.status("cn_a"),
        "examples": [{k: e[k] for k in ("id", "title", "query")} for e in EXAMPLES],
        "ai": _ai_available(),
        "provider": config.PROVIDERS[config.active_provider()]["label"] if _ai_available() else None,
        "model": config.PROVIDERS[config.active_provider()]["model"] if _ai_available() else None,
        "ai_defaults": {k: p["model"] for k, p in config.PROVIDERS.items()},
    }


def _ai_from_headers(request: Request) -> dict | None:
    """页面在请求头里带上用户自己的 key（存在用户浏览器里）。后端只在内存里用，不落盘、不写日志。"""
    key = (request.headers.get("x-xg-key") or "").strip()
    if not key:
        return None
    return {"provider": request.headers.get("x-xg-provider") or "deepseek", "key": key,
            "model": (request.headers.get("x-xg-model") or "").strip() or None,
            "effort": (request.headers.get("x-xg-effort") or "").strip() or None}


@app.post("/api/screen")
async def screen(req: ScreenRequest, request: Request):
    m = config.MARKETS.get(req.market)
    if not m:
        raise HTTPException(400, "未知市场")
    if not m["enabled"]:
        raise HTTPException(400, f"{m['label']}暂未支持，目前先支持 A 股")
    if req.start > req.end:
        raise HTTPException(400, "开始日期不能晚于结束日期")
    start, end = req.start.isoformat(), req.end.isoformat()

    t0 = time.time()
    if req.code:
        r = await run_in_threadpool(sandbox.run, req.market, start, end, req.code)
        return {**r, "code": req.code, "understanding": "手动修改的代码", "assumptions": [], "attempts": 1,
                "seconds": round(time.time() - t0, 1), "source": "manual"}
    if req.example_id:
        ex = BY_ID.get(req.example_id)
        if not ex:
            raise HTTPException(404, "示例不存在")
        r = await run_in_threadpool(sandbox.run, req.market, start, end, ex["code"])
        return {**r, "code": ex["code"], "understanding": ex["understanding"], "assumptions": [],
                "attempts": 1, "seconds": round(time.time() - t0, 1), "source": "example"}

    if req.refine:
        if not (req.refine.get("instruction") or "").strip() or not req.refine.get("base_code"):
            raise HTTPException(400, "请输入修改要求")
    elif not req.query.strip():
        raise HTTPException(400, "请输入筛选规则")
    from . import llm

    ai = llm.resolve_ai(_ai_from_headers(request))
    if not ai:
        raise HTTPException(400, detail={"code": "no_ai", "message": "先填一个 API Key 才能用自己写的规则，示例和收藏不需要"})
    try:
        r = await run_in_threadpool(llm.generate_and_run, req.query, req.market, start, end, 2, ai, req.refine)
    except Exception as e:  # 网络、额度、解析等
        kind, msg = llm.explain_error(e, ai["key"])
        raise HTTPException(502, detail={"code": f"ai_{kind}", "message": f"AI 调用失败：{msg}"})
    return {**r, "source": "ai"}


# ---------------------------------------------------------------- 数据管理
def _pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _sync_status() -> dict:
    f = data.market_dir("cn_a") / "sync_status.json"
    try:
        s = json.loads(f.read_text("utf-8"))
    except Exception:
        return {"state": "idle"}
    if s.get("state") == "running" and not _pid_alive(s.get("pid")):
        s["state"] = "interrupted"
        s["message"] = "上次下载被中断，已下载的部分保留"
    return s


@app.get("/api/data/status")
def data_status():
    return {"data": data.status("cn_a"), "sync": _sync_status()}


class SyncRequest(BaseModel):
    mode: Literal["update", "full", "retry_failed", "industry"] = "update"
    years: Literal[1, 3, 5] = 3
    limit: int = Field(0, ge=0)  # 调试用


@app.post("/api/data/sync")
def data_sync(req: SyncRequest):
    prev = _sync_status()
    if prev.get("state") == "running":
        raise HTTPException(409, "已经有下载在进行中")
    if req.mode == "retry_failed" and not prev.get("failed"):
        raise HTTPException(400, "上次没有失败的股票")
    d = data.market_dir("cn_a")
    d.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", "-m", "app.sync", "--years", str(req.years)]
    if req.mode == "update":
        cmd.append("--incremental")  # 已有数据时只补最新交易日；没有数据时自动退回逐只下载
    elif req.mode == "full":
        cmd.append("--full")
    elif req.mode == "retry_failed":
        cmd.append("--only-failed")
    elif req.mode == "industry":
        cmd.append("--industry-only")
    if req.limit:
        cmd += ["--limit", str(req.limit)]
    (d / "sync_cancel").unlink(missing_ok=True)
    log = open(config.DATA_DIR / "sync.log", "w")
    p = subprocess.Popen(cmd, cwd=config.ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)
    log.close()
    threading.Thread(target=p.wait, daemon=True).start()  # 回收子进程，避免僵尸进程被误判为还在运行
    # 先写一个 running 状态，避免轮询时短暂看到旧状态
    from .sync import write_json, PHASES, MODES
    has_data = (d / "daily.parquet").exists()
    mode = {"update": "incremental" if has_data else "update", "full": "full", "retry_failed": "retry", "industry": "industry"}[req.mode]
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(d / "sync_status.json", {"state": "running", "phase": "list", "phase_label": PHASES["list"], "done": 0,
                                        "total": 0, "failed": prev.get("failed", []) if req.mode == "retry_failed" else [], "list_source": "", "daily_source": "",
                                        "years": req.years, "mode": mode, "mode_label": MODES[mode], "started_at": now, "updated_at": now,
                                        "finished_at": None, "message": "", "pid": p.pid})
    return {"ok": True, "pid": p.pid}


@app.post("/api/data/cancel")
def data_cancel():
    if _sync_status().get("state") != "running":
        raise HTTPException(409, "当前没有进行中的下载")
    (data.market_dir("cn_a") / "sync_cancel").touch()
    return {"ok": True}



@app.post("/api/ai/test")
async def ai_test(request: Request):
    """用页面传来的 key 发一个极小的请求，验证 key、余额、模型是否可用。"""
    from . import llm

    ai = llm.resolve_ai(_ai_from_headers(request))
    if not ai:
        return {"ok": False, "ms": 0, "error_kind": "nokey", "message": "请先填写 API key"}
    t0 = time.time()

    def ping():
        c = llm.make_client(ai, timeout=30, max_retries=0)
        _, extra = llm.call_params(ai["provider"], ai["model"], "none")  # 测试连接不需要思考
        c.chat.completions.create(model=ai["model"], messages=[{"role": "user", "content": "ping"}], max_tokens=1,
                                  **({"extra_body": extra} if extra else {}))

    try:
        await run_in_threadpool(ping)
    except Exception as e:
        kind, msg = llm.explain_error(e, ai["key"])
        return {"ok": False, "ms": int((time.time() - t0) * 1000), "error_kind": kind, "message": msg}
    return {"ok": True, "ms": int((time.time() - t0) * 1000), "error_kind": None,
            "message": f"连接成功 · {config.PROVIDERS[ai['provider']]['label']} · {ai['model']}"}



# ---------------------------------------------------------------- 流式筛选（AI 规则）：边算边把进度和思考过程推给页面
@app.post("/api/screen/stream")
async def screen_stream(req: ScreenRequest, request: Request):
    """返回 NDJSON，每行一个事件：
    {"type":"status","stage":"thinking|writing|running|fixing","attempt":n}
    {"type":"reasoning","text":"..."}            思考模型的思考过程（增量）
    {"type":"result", ...和 /api/screen 一样的结果}
    {"type":"error","code":"...","message":"..."}"""
    import asyncio
    import queue
    import threading as th

    from fastapi.responses import StreamingResponse

    from . import llm

    m = config.MARKETS.get(req.market)
    if not m or not m["enabled"]:
        raise HTTPException(400, "这个市场暂未支持")
    if req.start > req.end:
        raise HTTPException(400, "开始日期不能晚于结束日期")
    if req.refine:
        if not (req.refine.get("instruction") or "").strip() or not req.refine.get("base_code"):
            raise HTTPException(400, "请输入修改要求")
    elif not req.query.strip():
        raise HTTPException(400, "请输入筛选规则")
    ai = llm.resolve_ai(_ai_from_headers(request))
    if not ai:
        raise HTTPException(400, detail={"code": "no_ai", "message": "先填一个 API Key 才能用自己写的规则，示例和收藏不需要"})
    start, end = req.start.isoformat(), req.end.isoformat()

    q: queue.Queue = queue.Queue()
    DONE = object()

    def work():
        try:
            r = llm.generate_and_run(req.query, req.market, start, end, 2, ai, req.refine, emit=q.put)
            q.put({"type": "result", **r, "source": "ai"})
        except Exception as e:  # 网络、额度、解析等；报错里不会带 key
            kind, msg = llm.explain_error(e, ai["key"])
            q.put({"type": "error", "code": f"ai_{kind}", "message": f"AI 调用失败：{msg}"})
        finally:
            q.put(DONE)

    th.Thread(target=work, daemon=True).start()

    async def gen():
        loop = asyncio.get_running_loop()
        while True:
            ev = await loop.run_in_executor(None, q.get)
            if ev is DONE:
                break
            yield json.dumps(ev, ensure_ascii=False, default=str) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
