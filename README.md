# 选股神器

用一句话描述选股规则，选市场和时间范围，AI（DeepSeek 或 OpenRouter 上的任意模型）把规则翻译成筛选代码，在本地行情数据上跑一遍，返回股票列表。目前先支持 A 股，港股、美股按钮已留好。

## 快速开始

需要 Python 3.10 及以上。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # 填 DEEPSEEK_API_KEY 或 OPENROUTER_API_KEY（不填也能先跑示例）
uvicorn app.main:app --reload # 打开 http://127.0.0.1:8000
```

没有下载真实数据时，网站会自动使用随机生成的示例数据（页面顶部会标出「示例数据」），可以先点示例规则体验完整流程。

## AI 配置

在 `.env` 里填一个 key 即可：

| 服务 | key | 默认模型 | 换模型 |
|---|---|---|---|
| DeepSeek 原生 API | `DEEPSEEK_API_KEY` | `deepseek-chat` | `DEEPSEEK_MODEL=deepseek-reasoner` |
| OpenRouter | `OPENROUTER_API_KEY` | `deepseek/deepseek-chat` | `OPENROUTER_MODEL=<OpenRouter 上的模型 id>` |

两个都填时默认用 DeepSeek，设置 `XG_PROVIDER=openrouter` 可切换。页面底部会显示当前用的服务和模型。

## 下载真实 A 股数据

```bash
python -m app.sync --limit 50   # 先下 50 只试试接口是否通
python -m app.sync              # 全市场最近 3 年日线（前复权），首次约 10-30 分钟
```

数据存在 `data/cn_a/`，中断后重跑会跳过当天已经下载的股票。建议每个交易日收盘后跑一次更新。

数据源是 [AKShare](https://akshare.akfamily.xyz/)（免费，底层是东方财富），包括：
- 日线：开高低收、成交量、成交额、涨跌幅、换手率
- 最新快照：名称、动态市盈率、市净率、总市值、流通市值
- 东财行业分类

暂时没有财报、资金流、概念板块等数据，AI 遇到这类条件会在「AI 的理解」里说明它怎么近似或忽略了。

## 工作原理

```
自然语言规则 + 时间范围
      │
      ▼
大模型（JSON 输出）→ 理解说明 + 假设 + screen() 函数代码
      │
      ▼
静态检查（禁止 import / 文件读写 / 私有属性）→ 独立子进程限时执行
      │            └─ 出错时把报错发回给模型自动修正，最多 2 次
      ▼
股票列表（可排序、导出 CSV），代码可查看和手动修改后重跑
```

不是开放式 agent，而是「生成 → 执行 → 出错就修」的固定流程，速度快、行为可控。

## 目录

```
app/
  main.py      网站后端（FastAPI）
  llm.py       AI 层：调用 DeepSeek / OpenRouter 生成代码、自动修错
  sandbox.py   代码静态检查 + 子进程执行
  runner.py    子进程入口：加载数据、执行 screen()
  helpers.py   给筛选代码用的指标函数（ma、macd、rsi、cross_up…）
  examples.py  写死的示例规则
  data.py      数据读取（真实数据 / 示例数据）
  sync.py      从 AKShare 下载数据
  static/index.html  前端页面（深色 / 亮色）
```

## 注意

- 沙箱是给自己本地用的防护级别，不适合直接放到公网给陌生人用。
- 仅供研究参考，不构成投资建议。
