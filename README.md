# 选股神器

用一句话描述选股规则，选市场和时间范围，AI（DeepSeek 或 OpenRouter 上的任意模型）把规则翻译成筛选代码，在本地行情数据上跑一遍，返回股票列表。目前先支持 A 股，港股、美股按钮已留好。

## 快速开始

需要 Python 3.10 及以上。

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --port 8000   # 打开 http://127.0.0.1:8000
```

第一次打开时还没有真实行情，网站会先用随机示例数据（顶部「数据」旁的圆点是灰色）。点右上角「数据」→「更新到最新」下载全市场数据（第一次约 1 小时，之后每天只补当天，1～2 分钟）。

## 功能

- **一句话选股**：输入规则，AI 生成筛选代码并在本地数据上执行；等待时显示「思考中 / 写代码 / 运行中」和计时，可以展开看 AI 的思考过程。
- **在当前结果上修改**：结果上方的修改框写「排除 ST、市值大于 50 亿」，AI 在原代码上做最小修改；每次修改生成一个版本，可以切回。
- **规则库**：示例规则 + 收藏（收藏存了代码，重跑不调 AI，秒出）。输入框里的书签图标或 ⌘K 打开，支持搜索、重命名、删除撤销。
- **结果表**：点表头排序（显示规则默认排序）、表头 × 隐藏列、最右边「＋」添加内置列（市盈率、市值、换手率、区间涨幅、最大回撤等）、点代码 / 名称复制、导出 CSV。
- **AI 设置**：右上角「AI」里填 DeepSeek 或 OpenRouter 的 key（只存在浏览器里），选模型和思考强度（默认 deepseek-flash + 高）。也可以写在 `.env` 里。
- **数据管理**：右上角「数据」查看数据范围和来源，增量更新 / 全部重新下载 / 重试失败，后台下载有进度。

## AI 配置

在页面右上角「AI」里填写即可，也可以复制 `.env.example` 为 `.env` 填写：

| 服务 | key | 默认模型 |
|---|---|---|
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-flash`（思考强度用 `DEEPSEEK_REASONING_EFFORT`：none / low / high / max） |
| OpenRouter | `OPENROUTER_API_KEY` | `deepseek/deepseek-chat`（`OPENROUTER_MODEL` 可换） |

页面里填的 key 优先于 `.env`。

## 数据

```bash
.venv/bin/python -m app.sync --incremental   # 增量：只补最新交易日
.venv/bin/python -m app.sync --full          # 全部重新下载（默认最近 3 年）
.venv/bin/python -m app.sync --only-failed   # 重试上次失败的
.venv/bin/python -m app.sync --industry-only # 只更新行业分类
```

数据存在 `data/cn_a/`（不进 git）。来源都是免费接口：股票列表用新浪（东方财富备用），日线用腾讯（东方财富备用，前复权），行业分类用申万（乐咕乐股，新浪备用）。暂时没有财报、资金流、概念板块等数据。

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
  helpers.py   给筛选代码用的指标函数（ma、macd、rsi、cross_up、limit_up、runs…）
  examples.py  写死的示例规则
  data.py      数据读取（真实数据 / 示例数据）
  sync.py      下载 / 增量更新行情和行业数据
  static/index.html  前端页面（深色 / 亮色）
```

## 注意

- 沙箱是给自己本地用的防护级别，不适合直接放到公网给陌生人用。
- 仅供研究参考，不构成投资建议。
