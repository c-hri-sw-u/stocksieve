"""给大模型的提示词和模型参数规则。

纯 Python、不依赖网络库：本地网站（app/llm.py）和静态网页版（在浏览器的 Pyodide 里）共用同一份。
"""
from __future__ import annotations

import json
import re

from .examples import EXAMPLES
from .helpers import HELPER_DOC

LOOKBACK_DAYS = 400  # 和 config.LOOKBACK_DAYS 保持一致（这里不 import config，避免浏览器里读 .env）

REWRITE_RETRY = "你几乎重写了整段代码。请以原来的「当前代码」为基础做最小修改：只改修改要求涉及的部分，其他逻辑（包括涨停判断、阈值、排序、输出列）保持原样。按同样的 JSON 格式重新输出。"
REWRITE_NOTE = "（注意：AI 重写了整段规则，不是在原规则上小改）"


def fix_message(error: str, traceback: str = "") -> str:
    return f"代码执行失败：\n{error}\n{traceback}\n请修正后按同样的 JSON 格式重新输出完整结果。"


def user_message(query: str, start: str, end: str) -> str:
    return f"时间范围：{start} 至 {end}\n筛选规则：{query}"


OUTPUT_FORMAT = """## 输出格式
只输出一个 JSON 对象，不要输出其他文字：
{"plan": "先打草稿：用几句中文写清楚关键概念的定义、时间口径和计算步骤（写代码前想清楚，比如「连板」怎么算、阈值是多少）",
 "understanding": "用中文一两句话复述你理解的筛选规则，具体到参数和时间口径",
 "assumptions": ["用户没说清、你自己做的假设，或数据不支持只能近似的地方，用中文；没有就给空数组"],
 "code": "完整的 Python 代码字符串，定义 screen(daily, stocks, start, end)"}"""

def system_prompt(ind_label: str = "行业名称", ind2_label: str = "二级行业名称") -> str:
    shots = "\n\n".join(
        f"示例规则：{e['query']}\n代码：\n```python\n{e['code']}```" for e in EXAMPLES[:2]
    )
    return f"""你是一个 A 股选股规则编译器。用户用自然语言描述筛选条件，你把它翻译成一个 Python 函数，在本地行情数据上执行。

## 函数约定
定义 `def screen(daily, stocks, start, end):`，返回一个 pandas DataFrame：
- 必须有 `code` 列（6 位股票代码字符串），每只股票一行。用与规则最相关的「一个」指标列排序（sort_values，最符合的在前），并把这一列放在返回结果里，界面会把它标成默认排序列。
- 另外放 2~6 列能说明「为什么选中」的指标，列名用简短中文并注明单位，例如「20日涨幅%」「总市值(亿)」。不用放名称和行业，系统会自动补上。
- start、end 是 pd.Timestamp，是用户选择的时间范围，表示「事件发生在哪段时间」。daily 是本机已下载的全部日线（截至 end，开始日期因用户下载了多少而不同，通常比 start 早很多），计算均线等指标直接用；判断条件时用 in_range(d, start, end) 截取区间。
- 「历史新低/新高」「上市以来」「有数据以来」= 用 daily 里该股票的全部数据，不要只用 [start, end]；如果数据开始日期晚于该股上市日，在 assumptions 里说明「历史」指本机已有数据的范围。
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



def parse_answer(text: str) -> dict:
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


REFINE_FORMAT = """## 输出格式
只输出一个 JSON 对象，不要输出其他文字：
{"plan": "先打草稿：用几句中文写清楚要改哪几处、改成什么，以及哪些地方保持不变",
 "query": "合并修改后的完整规则，一句自然的中文，不看原规则也能独立读懂",
 "understanding": "用中文一两句话复述修改后的完整筛选规则，具体到参数和时间口径",
 "change": "给普通用户看的一句话，不超过 25 个汉字，只说结果上的变化；不出现代码、变量名、比较符号和英文，例如「只保留正好 2 连板的股票」；全新的规则写「按新规则重写」",
 "change_detail": ["可选，给想看细节的人：列和排序有什么变化，同样用大白话，最多 3 条短句；没有就给空数组"],
 "assumptions": ["用户没说清、你自己做的假设，或数据不支持只能近似的地方；没有就给空数组"],
 "code": "完整的 Python 代码字符串，定义 screen(daily, stocks, start, end)"}"""


def refine_message(refine: dict, start: str, end: str) -> str:
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


def rewrote(old: str, new: str, change: str) -> bool:
    import difflib

    if "新规则" in (change or ""):
        return False
    return difflib.SequenceMatcher(None, old, new).ratio() < 0.45
