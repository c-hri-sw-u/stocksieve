"""AI 层：大模型把自然语言规则翻译成 screen() 代码，执行出错时把报错发回去让它修。

支持 DeepSeek 原生 API 和 OpenRouter（两者都是 OpenAI 兼容接口），在 .env 里配置。
不是开放式 agent，而是「生成 → 执行 → 出错就修」的固定流程，最多修 2 次。
"""
from __future__ import annotations

import json
import os
import time

from openai import BadRequestError, OpenAI

from . import config, sandbox
from . import prompts
from .prompts import (LEGACY_DEEPSEEK, MAX_TOKENS, call_params, deepseek_new, fix_model,  # noqa: F401
                      looks_reasoning, parse_answer as _parse, refine_message as _refine_message, rewrote as _rewrote)

def _industry_desc() -> tuple[str, str]:
    """告诉模型行业列用的是哪套分类，以及全部行业名称，方便它写出能精确匹配的条件。"""
    try:
        import pandas as pd

        from . import data

        st = data.status("cn_a")
        if st.get("demo"):
            return "示例行业", "空"
        stocks = pd.read_parquet(data.market_dir("cn_a") / "stocks.parquet")
        src = st.get("industry_source") or "行业"
        names = sorted(n for n in stocks["industry"].dropna().unique() if n)
        if not names:
            return "暂无行业数据，均为空字符串", "空"
        l2 = "申万二级行业名称" if src.startswith("申万") else "空"
        return f"{src}行业名称，取值：{'、'.join(names)}；没有分类的为空字符串", l2
    except Exception:
        return "行业名称", "二级行业名称"


def _system_prompt() -> str:
    return prompts.system_prompt(*_industry_desc())


def resolve_ai(override: dict | None = None) -> dict | None:
    """确定这次用哪个 AI：页面传来的 key 优先，否则用 .env。返回 {provider, key, model} 或 None。"""
    if override and override.get("key") and override.get("provider") in config.PROVIDERS:
        p = config.PROVIDERS[override["provider"]]
        return {"provider": override["provider"], "key": override["key"], "model": override.get("model") or p["model"],
                "effort": override.get("effort")}
    name = config.active_provider()
    if not name:
        return None
    p = config.PROVIDERS[name]
    return {"provider": name, "key": os.environ[p["key_env"]], "model": p["model"], "effort": os.environ.get("DEEPSEEK_REASONING_EFFORT")}


def make_client(ai: dict, timeout: float = 300, max_retries: int = 2) -> OpenAI:
    p = config.PROVIDERS[ai["provider"]]
    headers = {"X-Title": "xuangu"} if ai["provider"] == "openrouter" else None
    return OpenAI(api_key=ai["key"], base_url=p["base_url"], default_headers=headers, timeout=timeout, max_retries=max_retries)


def _client(override: dict | None = None) -> tuple[OpenAI, str]:
    ai = resolve_ai(override)
    if not ai:
        raise RuntimeError("还没有设置 AI key")
    return make_client(ai), ai["model"]


def explain_error(e: Exception, key: str | None = None) -> tuple[str, str]:
    """把 AI 接口的报错翻成人话，返回 (类型, 说明)。说明里不会出现 key。"""
    import openai

    raw = str(e)
    if key:
        raw = raw.replace(key, "***")
    status = getattr(e, "status_code", None)
    low = raw.lower()
    if isinstance(e, openai.AuthenticationError) or status == 401:
        return "auth", "API key 不正确或已失效，请检查是否复制完整"
    if status == 402 or "insufficient" in low or "balance" in low or "credits" in low:
        return "balance", "账户余额不足，请到服务商后台充值"
    if isinstance(e, openai.NotFoundError) or status == 404 or "model" in low and ("not exist" in low or "not found" in low or "invalid model" in low or "not a valid model" in low):
        return "model", "模型不存在，请检查模型名称"
    if isinstance(e, openai.RateLimitError) or status == 429:
        return "rate", "请求太频繁或额度用完，稍后再试"
    if isinstance(e, (openai.APIConnectionError, openai.APITimeoutError)):
        return "network", "连不上服务商，请检查网络或代理"
    return "other", f"{type(e).__name__}: {raw[:200]}"


def _call(client: OpenAI, model: str, messages: list, emit=None, reasoning: bool = False,
          extra: dict | None = None, provider: str = "deepseek") -> tuple[dict, str, str]:
    """流式调用，边收边把思考过程 / 阶段推给 emit。返回 (解析后的 JSON, 原始答案文本, 思考过程)。"""
    effort = (extra or {}).get("reasoning_effort")
    kwargs = dict(model=model, messages=messages, stream=True,
                  max_tokens=MAX_TOKENS.get(effort or "", 32000 if reasoning else 8000))
    if not reasoning:
        kwargs["temperature"] = 0  # 推理模型不支持 / 忽略 temperature
    body = dict(extra or {})
    if provider == "openrouter" and reasoning:
        body.setdefault("reasoning", {"exclude": False})  # 让 OpenRouter 把思考内容返回来
    if body:
        kwargs["extra_body"] = body
    try:
        stream = client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
    except BadRequestError:
        # 有的模型（比如部分推理模型）不支持 JSON 模式，退回普通输出再自己解析
        stream = client.chat.completions.create(**kwargs)

    text, think, finish, stage = [], [], None, None
    for chunk in stream:
        if not chunk.choices:
            continue
        ch = chunk.choices[0]
        d = ch.delta
        r = getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None)
        if r is None and getattr(d, "model_extra", None):
            r = d.model_extra.get("reasoning_content") or d.model_extra.get("reasoning")
        if r:
            think.append(r)
            if emit:
                if stage != "thinking":
                    stage = "thinking"
                    emit({"type": "status", "stage": "thinking"})
                emit({"type": "reasoning", "text": r})
        if d.content:
            text.append(d.content)
            if emit and stage != "writing":
                stage = "writing"
                emit({"type": "status", "stage": "writing"})
        if ch.finish_reason:
            finish = ch.finish_reason
    raw = "".join(text)
    if finish == "length":
        raise RuntimeError("模型输出过长被截断，请把规则拆简单一点")
    try:
        return _parse(raw), raw, "".join(think)
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(f"无法解析模型输出：{e}") from e


def generate_and_run(query: str, market: str, start: str, end: str, max_fixes: int = 2, ai: dict | None = None,
                     refine: dict | None = None, emit=None) -> dict:
    """生成代码 → 运行 → 出错就修。emit(event) 用来把进度推给页面（流式接口用），可以不传。"""
    ai = resolve_ai(ai)
    if not ai:
        raise RuntimeError("还没有设置 AI key")
    client, model, provider = make_client(ai), ai["model"], ai["provider"]
    reasoning, base_extra = call_params(provider, model, ai.get("effort"))
    say = emit or (lambda e: None)
    if refine:
        user = _refine_message(refine, start, end)
    else:
        user = prompts.user_message(query, start, end)
    messages: list = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": user},
    ]
    t0 = time.time()
    plan: dict = {}
    result: dict = {"ok": False, "error": "未执行"}
    attempts = 0
    thoughts: list[str] = []

    def ask(use_fix: bool = False):
        # 自动修复时推理模型换成同家族的快模型；回传给模型的只有答案，不带思考过程
        if use_fix and reasoning:
            m, extra = fix_model(provider, model)
            r = call_params(provider, m, extra.get("reasoning_effort"))[0] if extra else looks_reasoning(provider, m)
        else:
            m, extra, r = model, base_extra, reasoning
        say({"type": "status", "stage": "thinking" if r else "writing"})
        p, raw, think = _call(client, m, messages, say, r, extra, provider)
        if think:
            thoughts.append(think)
        return p, raw

    rewrote_retry = False
    for i in range(max_fixes + 1):
        attempts += 1
        if i:
            say({"type": "status", "stage": "fixing", "attempt": i})
        plan, raw = ask(use_fix=i > 0)
        # 护栏：修改时如果新代码和原代码几乎不像、又没说明是按新规则重写，就要求它在原代码上最小修改再来一次
        if refine and not rewrote_retry and _rewrote(refine.get("base_code", ""), plan["code"], plan.get("change", "")):
            rewrote_retry = True
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": prompts.REWRITE_RETRY})
            attempts += 1
            plan, raw = ask()
            if _rewrote(refine.get("base_code", ""), plan["code"], plan.get("change", "")):
                plan["change"] = ((plan.get("change") or "").rstrip("。") + prompts.REWRITE_NOTE).lstrip("（")
        say({"type": "status", "stage": "running"})
        result = sandbox.run(market, start, end, plan["code"])
        if result.get("ok") or i == max_fixes:
            break
        messages.append({"role": "assistant", "content": raw})
        messages.append(
            {
                "role": "user",
                "content": prompts.fix_message(result.get("error", ""), result.get("traceback", "")),
            }
        )

    return {
        **result,
        "plan": plan.get("plan", ""),
        "reasoning": "\n\n".join(thoughts)[-20000:],
        "understanding": plan.get("understanding", ""),
        "assumptions": plan.get("assumptions", []),
        "query": (plan.get("query") or "").strip() if refine else query,
        "change": (plan.get("change") or "").strip() if refine else "",
        "change_detail": [str(x) for x in (plan.get("change_detail") or [])][:3] if refine else [],
        "code": plan.get("code", ""),
        "attempts": attempts,
        "seconds": round(time.time() - t0, 1),
    }
