"""全局配置。可以在项目根目录的 .env 里覆盖。"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    if not os.environ.get("XG_NO_DOTENV"):  # 执行筛选代码的子进程不读 .env，拿不到 key
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:  # python-dotenv 是可选的
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("XG_DATA_DIR", ROOT / "data"))

# 没有本地数据时，用随机生成的示例数据（方便先体验界面流程）
DEMO_MODE = os.environ.get("XG_DEMO", "").lower() in ("1", "true", "yes")

# AI 服务：DeepSeek 原生 API 或 OpenRouter（都是 OpenAI 兼容接口）
PROVIDERS = {
    "deepseek": {
        "label": "DeepSeek",
        "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        "key_env": "DEEPSEEK_API_KEY",
        "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
    },
    "openrouter": {
        "label": "OpenRouter",
        "base_url": os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "key_env": "OPENROUTER_API_KEY",
        "model": os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-chat"),
    },
}


def active_provider() -> str | None:
    """XG_PROVIDER 指定的优先；没指定就用第一个填了 key 的。"""
    want = os.environ.get("XG_PROVIDER", "").strip().lower()
    if want in PROVIDERS and os.environ.get(PROVIDERS[want]["key_env"]):
        return want
    for name, p in PROVIDERS.items():
        if os.environ.get(p["key_env"]):
            return name
    return None

# 沙箱执行
EXEC_TIMEOUT = int(os.environ.get("XG_EXEC_TIMEOUT", "90"))  # 秒
MAX_ROWS = int(os.environ.get("XG_MAX_ROWS", "10000"))  # 返回全部结果，前端排序才完整
# 指标计算需要的预热期：在用户选择的开始日期之前多加载的自然日
LOOKBACK_DAYS = int(os.environ.get("XG_LOOKBACK_DAYS", "400"))

MARKETS = {
    "cn_a": {"label": "A股", "enabled": True},
    "hk": {"label": "港股", "enabled": False},
    "us": {"label": "美股", "enabled": False},
}
