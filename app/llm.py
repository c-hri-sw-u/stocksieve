"""AI 层：大模型把自然语言规则翻译成 screen() 代码，执行出错时把报错发回去让它修。

支持 DeepSeek 原生 API 和 OpenRouter（两者都是 OpenAI 兼容接口），在 .env 里配置。
不是开放式 agent，而是「生成 → 执行 → 出错就修」的固定流程，最多修 2 次。
"""
from __future__ import annotations

import json
import os
import re
import time

from openai import BadRequestError, OpenAI

from . import config, sandbox
from .examples import EXAMPLES
from .helpers import HELPER_DOC

OUTPUT_FORMAT = """## 输出格式
只输出一个 JSON 对象，不要输出其他文字：
{"plan": "先打草稿：用几句中文写清楚关键概念的定义、时间口径和计算步骤（写代码前想清楚，比如「连板」怎么算、阈值是多少）",
 "understanding": "用中文一两句话复述你理解的筛选规则，具体到参数和时间口径",
 "assumptions": ["用户没说清、你自己做的假设，或数据不支持只能近似的地方，用中文；没有就给空数组"],
 "code": "完整的 Python 代码字符串，定义 screen(daily, stocks, start, end)"}"""

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
    shots = "\n\n".join(
        f"示例规则：{e['query']}\n代码：\n```python\n{e['code']}```" for e in EXAMPLES[:2]
    )
    ind_label, ind2_label = _industry_desc()
    return f"""你是一个 A 股选股规则编译器。用户用自然语言描述筛选条件，你把它翻译成一个 Python 函数，在本地行情数据上执行。

## 函数约定
定义 `def screen(daily, stocks, start, end):`，返回一个 pandas DataFrame：
- 必须有 `code` 列（6 位股票代码字符串），每只股票一行。用与规则最相关的「一个」指标列排序（sort_values，最符合的在前），并把这一列放在返回结果里，界面会把它标成默认排序列。
- 另外放 2~6 列能说明「为什么选中」的指标，列名用简短中文并注明单位，例如「20日涨幅%」「总市值(亿)」。不用放名称和行业，系统会自动补上。
- start、end 是 pd.Timestamp，是用户选择的时间范围。daily 里包含 start 之前约 {config.LOOKBACK_DAYS} 个自然日的预热数据，方便计算均线等指标；判断条件时注意用 in_range(d, start, end) 截取区间。
- 用户说「最近 N 天/日」时，指截至 end 的最近 N 个交易日；没说时间口径时，默认条件在 end 这一天成立，区间类条件（涨幅、回撤、出现过某信号）用 [start, end]。
- A 股术语：「涨停/跌停」一律用 limit_up / limit_down 判断（已区分主板、创业板、科创板、北交所、ST 和新股），不要自己写阈值。
- 「连续 N 个涨停」「N 连板」= 连续 N 个交易日涨停。默认理解为「区间内出现过至少 N 连板」（runs(daily, limit_up(daily, stocks)) 里 length >= N 且该段落在区间内）；用户说「正好 N 个」「只要 N 个」「不要更多」时，才是某段连板的最终长度恰好为 N（length == N）。
- 输出列名必须让不懂 A 股的人也能看懂，写清含义和单位，例如「最长连板天数」「最近一次连板结束日」「连板段数」；不要用「次数」这种含糊的名字。

## 数据
daily：长表，按 code、date 排好序，每行一只股票一个交易日（前复权）
  date(datetime64) code(str) open high low close volume(手) amount(元) pct_chg(当日涨跌幅%) turnover(换手率%)
stocks：每只股票一行（最新快照，不是历史值）
  code name industry({ind_label}) industry2({ind2_label}) price(最新价) pe(动态市盈率) pb(市净率) total_mv(总市值,元) float_mv(流通市值,元)
没有的数据：财报（营收、利润、ROE 等）、分红、资金流、龙虎榜、概念板块、分钟线。如果规则需要这些，用现有数据做最合理的近似，并在 assumptions 里说明；完全无法近似的条件就忽略并说明。

## 可用工具
已经可以直接用 pd、np，以及：
{HELPER_DOC}
不能 import，不能读写文件，不能访问以下划线开头的属性。只用到 pandas/numpy 的常规向量化写法，避免对每只股票写 Python 循环（全市场 5000+ 只，要在几十秒内跑完）。groupby 时不要用 apply，用 transform / agg / 上面的函数。

## 参考写法
{shots}

{OUTPUT_FORMAT}
"""


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


def _parse(text: str) -> dict:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    elif not text.startswith("{"):
        i, j = text.find("{"), text.rfind("}")
        if i >= 0 and j > i:
            text = text[i : j + 1]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj, _ = json.JSONDecoder().raw_decode(text[text.find("{"):])  # 只取第一个完整的 JSON 对象
    if not isinstance(obj, dict) or not isinstance(obj.get("code"), str):
        raise ValueError("模型返回的 JSON 里没有 code 字段")
    obj.setdefault("understanding", "")
    obj.setdefault("assumptions", [])
    return obj


REASONING_HINTS = ("reasoner", "r1", "think", "reason", "o1", "o3", "o4")


EFFORTS = ("none", "low", "high", "max")
LEGACY_DEEPSEEK = ("deepseek-chat", "deepseek-reasoner")


def deepseek_new(provider: str, model: str) -> bool:
    """DeepSeek 新版模型（deepseek-flash、deepseek-v4-pro 等）用 reasoning_effort 控制思考；旧名字没有这个参数。"""
    return provider == "deepseek" and model.lower().startswith("deepseek-") and model.lower() not in LEGACY_DEEPSEEK


def call_params(provider: str, model: str, effort: str | None) -> tuple[bool, dict]:
    """返回 (是否会思考, 额外请求参数)。DeepSeek 新版模型：reasoning_effort = none/low/high/max，默认 high。"""
    if deepseek_new(provider, model):
        e = effort if effort in EFFORTS else "high"
        return e != "none", {"reasoning_effort": e}
    m = model.lower()
    if provider == "deepseek":
        return m == "deepseek-reasoner", {}
    return any(h in m for h in REASONING_HINTS), {}


def looks_reasoning(provider: str, model: str, effort: str | None = None) -> bool:
    return call_params(provider, model, effort)[0]


def fix_model(provider: str, model: str) -> tuple[str, dict]:
    """自动修复代码时用的模型：推理太慢，换成不思考的方式。返回 (模型名, 额外参数)。"""
    m = model.lower()
    if provider == "deepseek":
        if m == "deepseek-reasoner":
            return "deepseek-chat", {}
        if deepseek_new(provider, model):  # 同一个模型，关掉思考
            return model, {"reasoning_effort": "none"}
        return model, {}
    if m.startswith("deepseek/deepseek-r1"):
        return "deepseek/deepseek-chat", {}
    return model, {}


MAX_TOKENS = {"none": 8000, "low": 32000, "high": 64000, "max": 128000}


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


REFINE_FORMAT = """## 输出格式
只输出一个 JSON 对象，不要输出其他文字：
{"plan": "先打草稿：用几句中文写清楚要改哪几处、改成什么，以及哪些地方保持不变",
 "query": "合并修改后的完整规则，一句自然的中文，不看原规则也能独立读懂",
 "understanding": "用中文一两句话复述修改后的完整筛选规则，具体到参数和时间口径",
 "change": "给普通用户看的一句话，不超过 25 个汉字，只说结果上的变化；不出现代码、变量名、比较符号和英文，例如「只保留正好 2 连板的股票」；全新的规则写「按新规则重写」",
 "change_detail": ["可选，给想看细节的人：列和排序有什么变化，同样用大白话，最多 3 条短句；没有就给空数组"],
 "assumptions": ["用户没说清、你自己做的假设，或数据不支持只能近似的地方；没有就给空数组"],
 "code": "完整的 Python 代码字符串，定义 screen(daily, stocks, start, end)"}"""


def _refine_message(refine: dict, start: str, end: str) -> str:
    rs = refine.get("result_summary") or {}
    cols = "、".join(str(c) for c in (rs.get("columns") or [])) or "无"
    assumptions = "；".join(str(a) for a in (refine.get("base_assumptions") or [])) or "无"
    top = "\n".join("  " + " | ".join("" if v is None else str(v) for v in row) for row in (rs.get("top_rows") or [])[:5])
    return f"""这是一次「在现有规则上修改」。请以下面的「当前代码」为基础逐处修改，只改用户要求的部分：
- 未提到的条件、阈值、判定方法（例如涨停的判断方式）、排序和输出列都必须原样保留，不要重写或简化。
- 如果修改要求和现有输出列冲突（例如要求「只要连续两个」，而现有列是「连续涨停次数」），要同时改掉或删掉那一列，不要留下会误导的指标。
- change 是给不懂代码的用户看的，一句大白话、不超过 25 个字；具体改了哪些代码写在 plan 里。
如果修改要求和原规则明显无关（是一条全新的规则），可以直接按新规则重写。

时间范围：{start} 至 {end}
原规则：{refine.get("base_query") or "（无文字，见代码）"}
原规则的理解：{refine.get("base_understanding") or "无"}
上次的假设：{assumptions}
当前结果：{rs.get("total", "未知")} 只股票，列：{cols}
前 5 行：
{top or "  （无）"}
当前代码：
```python
{refine.get("base_code", "")}
```
修改要求：{refine.get("instruction", "")}

{REFINE_FORMAT}"""


def _rewrote(old: str, new: str, change: str) -> bool:
    import difflib

    if "新规则" in (change or ""):
        return False
    return difflib.SequenceMatcher(None, old, new).ratio() < 0.45


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
        user = f"时间范围：{start} 至 {end}\n筛选规则：{query}"
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
            messages.append({"role": "user", "content": "你几乎重写了整段代码。请以原来的「当前代码」为基础做最小修改：只改修改要求涉及的部分，其他逻辑（包括涨停判断、阈值、排序、输出列）保持原样。按同样的 JSON 格式重新输出。"})
            attempts += 1
            plan, raw = ask()
            if _rewrote(refine.get("base_code", ""), plan["code"], plan.get("change", "")):
                plan["change"] = ((plan.get("change") or "").rstrip("。") + "（注意：AI 重写了整段规则，不是在原规则上小改）").lstrip("（")
        say({"type": "status", "stage": "running"})
        result = sandbox.run(market, start, end, plan["code"])
        if result.get("ok") or i == max_fixes:
            break
        messages.append({"role": "assistant", "content": raw})
        messages.append(
            {
                "role": "user",
                "content": f"代码执行失败：\n{result.get('error')}\n{result.get('traceback', '')}\n请修正后按同样的 JSON 格式重新输出完整结果。",
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
