// 静态网页版的运行时：没有后端服务器，页面原来请求的 /api/... 都在这里就地实现。
// - 行情数据：从同一网站的 data/ 目录下载，存在本机（OPFS / IndexedDB），按 manifest 只下载缺的或变了的文件
// - 计算：worker.js 里的 Pyodide 跑和本地版同一份 Python 代码
// - AI：浏览器直接调 DeepSeek / OpenRouter，key 只在本机
(function () {
  const realFetch = window.fetch.bind(window);
  const INDEX_KEY = "xg.data.v1";
  const EXEC_TIMEOUT = 240000;  // 时间范围很长时会分批算，慢一些
  const MARKETS = { cn_a: { label: "A股", enabled: true }, hk: { label: "港股", enabled: false }, us: { label: "美股", enabled: false } };
  const AI_DEFAULTS = { deepseek: "deepseek-flash", openrouter: "deepseek/deepseek-chat" };
  const PROVIDERS = {
    deepseek: { label: "DeepSeek", url: "https://api.deepseek.com/chat/completions" },
    openrouter: { label: "OpenRouter", url: "https://openrouter.ai/api/v1/chat/completions" },
  };

  // ---------------------------------------------------------------- 本机数据索引
  const loadIndex = () => { try { return JSON.parse(localStorage.getItem(INDEX_KEY) || "null") || { files: {} }; } catch (e) { return { files: {} }; } };
  const saveIndex = (ix) => { try { localStorage.setItem(INDEX_KEY, JSON.stringify(ix)); } catch (e) {} };
  // { manifest: 本机这份数据对应的 manifest, files: {名字: sha256}, remote: 最近看到的远端 manifest 概要,
  //   histFrom: 用户在「时间范围」里往前补到的日期（更早的历史按年一个文件，只下载这一天以后的） }
  let index = loadIndex();
  const has = (f) => index.files[f.name] === f.sha256;
  const baseFiles = (m) => m.files.filter((f) => f.kind !== "history");
  // 更早的历史：从新到旧
  const histFiles = (m) => m.files.filter((f) => f.kind === "history").sort((a, b) => (a.from < b.from ? 1 : -1));
  const wantedHist = (m) => (index.histFrom ? histFiles(m).filter((f) => f.to >= index.histFrom) : []);
  const wanted = (m) => [...baseFiles(m), ...wantedHist(m)];
  // 能用的历史：从最新的年份往前、连续都已下载的那几个（中途取消的话只用接得上的部分）
  function usableHist(m) {
    const out = [];
    for (const f of wantedHist(m)) { if (!has(f)) break; out.push(f); }
    return out;
  }
  const complete = () => !!index.manifest && baseFiles(index.manifest).every(has);

  // ---------------------------------------------------------------- 计算 worker
  let worker = null, seq = 0, pending = new Map(), workerReady = null, dataLoaded = null, histLoaded = new Set();
  const engine = { stage: "", readyMs: null };
  const LOAD_TIMEOUT = 150000;  // 计算环境（Python + pandas）最多等 150 秒
  // 计算环境起不来时：所有等待中的调用立刻失败，下次用到时重新启动 worker（相当于「重试」）
  function workerFailed(msg) {
    const err = new Error("计算环境加载失败：" + msg);
    err.kind = "engine";
    engine.stage = ""; engine.error = err.message; engine.ready = false;
    try { worker?.terminate(); } catch (e) {}
    worker = null; dataLoaded = null; histLoaded = new Set();
    for (const q of pending.values()) { clearTimeout(q.timer); q.reject(err); }
    pending.clear();
  }
  function startWorker() {
    engine.error = null; engine.ready = false;
    worker = new Worker("worker.js");
    workerReady = new Promise((res) => {
      worker.onmessage = (e) => {
        const m = e.data;
        if (m.type === "progress") { engine.stage = m.stage; return; }
        if (m.type === "ready") { engine.stage = ""; engine.readyMs = m.ms; engine.ready = true; res(); return; }
        if (m.type === "fatal") { workerFailed(m.error); return; }
        // 运行中崩溃（内存不够）：这次调用已经收到报错，丢掉这个 worker，下次用到时重新启动
        if (m.type === "crashed") { try { worker?.terminate(); } catch (err) {} worker = null; dataLoaded = null; histLoaded = new Set(); engine.ready = false; return; }
        const p = pending.get(m.id); if (!p) return;
        pending.delete(m.id); clearTimeout(p.timer);
        m.ok ? p.resolve(m.result) : p.reject(new Error(m.error));
      };
      worker.onerror = (e) => { e.preventDefault?.(); workerFailed(e.message || "worker 出错"); };
    });
    dataLoaded = null;
  }
  function call(cmd, args = [], timeout = 0) {
    if (!worker) startWorker();
    const id = ++seq;
    // 计算环境还没准备好时，多给加载的时间；永远不会无限等待
    const loading = !engine.ready;
    const limit = (timeout || 60000) + (loading ? LOAD_TIMEOUT : 0);
    return new Promise((resolve, reject) => {
      const p = { resolve, reject, timer: null };
      p.timer = setTimeout(() => {
        // 超时：直接结束 worker（停止计算），下次用到时重新启动
        pending.delete(id); worker?.terminate(); worker = null; dataLoaded = null; histLoaded = new Set();
        for (const q of pending.values()) { clearTimeout(q.timer); q.reject(new Error("计算已中止")); }
        pending.clear();
        reject(new Error(loading && !engine.ready ? "计算环境加载超时：Python 运行环境下载太慢（网络问题），请重试"
                                                  : `运行超时（超过 ${(timeout || 60000) / 1000} 秒），请简化计算`));
      }, limit);
      pending.set(id, p);
      worker.postMessage({ id, cmd, args });
    });
  }
  async function ensureData() {
    if (!worker) startWorker();
    if (!dataLoaded) {
      dataLoaded = complete() ? withStage("载入行情数据", "", () => loadBase(index.manifest)) : Promise.resolve({ ok: false });
      dataLoaded.catch(() => { dataLoaded = null; });
    }
    return dataLoaded;
  }
  const py = (fn, ...args) => call("call", [fn, ...args]);
  // 页面上显示「现在在做什么」：engine.stage 是 Python 环境自己的加载进度（优先显示），
  // engine.task 是之后的步骤 { stage, hint }，hint 是括号里的说明（空的话页面显示「只有第一次需要」）
  async function withStage(stage, hint, fn) {
    const task = { stage, hint };
    engine.task = task;
    try { return await fn(); }
    finally { if (engine.task === task) engine.task = null; }
  }
  function loadBase(m) { histLoaded = new Set(); return call("loadData", [m]); }

  // 已下载的数据全部交给规则：用户补全过更早的历史，筛选前先把它接上（第一次要等一会，之后一直在内存里）
  async function ensureHistory(start, emit) {
    const m = index.manifest;
    if (!m || !index.histFrom) return false;
    const usable = usableHist(m);
    if (!usable.length) return false;
    const todo = usable.filter((f) => !histLoaded.has(f.name));
    if (!todo.length) return true;
    const years = todo.map((f) => +f.from.slice(0, 4));
    const lo = Math.min(...years), hi = Math.max(...years);
    const stage = `正在载入${lo === hi ? ` ${lo} 年` : ` ${lo}–${hi} 年`}的历史数据`;
    const hint = todo.length > 2 ? "第一次需要 1～2 分钟，之后不用再等" : "只有第一次需要";
    emit?.({ type: "status", stage: "history", text: `${stage}（${hint}）` });
    const r = await withStage(stage, hint, () => call("addHistory", [todo.map((f) => f.name)], 180000));
    if (r && r.ok === false) throw new Error(r.error);
    todo.forEach((f) => histLoaded.add(f.name));
    return true;
  }

  // ---------------------------------------------------------------- 检查更新（只看 manifest，不下载）
  // state: idle | checking | latest | available | none（本机还没数据） | error
  const check = { state: "idle" };
  async function runCheck() {
    if (check.state === "checking" || sync.state === "running") return;
    Object.assign(check, { state: "checking", at: now() });
    try {
      const r = await realFetch("data/manifest.json?t=" + Date.now(), { cache: "no-store" });
      if (!r.ok) throw new Error("HTTP " + r.status);
      const m = await r.json();
      index.remote = { last_date: m.last_date, generated_at: m.generated_at, total_size: m.total_size }; saveIndex(index);
      const need = wanted(m).filter((f) => !has(f));
      const bytes = need.reduce((a, f) => a + f.size, 0);
      Object.assign(check, { state: !complete() ? "none" : need.length ? "available" : "latest", last_date: m.last_date,
                             mb: +(bytes / 1e6).toFixed(1), bytes, total_mb: +(m.total_size / 1e6).toFixed(1), message: "" });
    } catch (e) {
      Object.assign(check, { state: "error", message: e.message });
    }
  }

  // ---------------------------------------------------------------- 下载数据
  const sync = { state: "idle" };
  let abort = null;
  const now = () => new Date().toISOString().slice(0, 19).replace("T", " ");

  async function startSync(mode, from) {
    if (sync.state === "running") throw Object.assign(new Error("已经有下载在进行中"), { status: 409 });
    if (mode === "extend") {
      if (!/^\d{4}-\d{2}-\d{2}$/.test(from || "")) throw Object.assign(new Error("开始日期不正确"), { status: 400 });
      if (!index.histFrom || from < index.histFrom) { index.histFrom = from; saveIndex(index); }
    }
    Object.assign(sync, { state: "running", phase: "list", phase_label: "获取文件清单", mode,
                          mode_label: mode === "clear" ? "清除本地数据" : mode === "extend" ? "补全更早的数据" : "下载到本机",
                          done: 0, total: 0, failed: [], message: "", started_at: now(), updated_at: now(), finished_at: null,
                          extend_from: mode === "extend" ? from : null, extended_to: null,
                          list_source: "", daily_source: "" });
    abort = new AbortController();
    runSync(mode, abort.signal).catch((e) => {
      Object.assign(sync, { state: abort.signal.aborted ? "cancelled" : "error", message: abort.signal.aborted ? "已取消，已下载的文件保留" : e.message, finished_at: now() });
    });
  }

  async function runSync(mode, signal) {
    if (mode === "clear") {
      await XGStore.clear(); index = { files: {}, remote: index.remote }; saveIndex(index); histLoaded = new Set();
      if (worker) { worker.terminate(); worker = null; }
      Object.assign(sync, { state: "idle", message: "本地数据已清除", finished_at: now() });
      Object.assign(check, { state: "none", mb: index.remote?.total_size ? +(index.remote.total_size / 1e6).toFixed(1) : 0 });
      return;
    }
    if (mode === "full") { await XGStore.clear(); index.files = {}; saveIndex(index); }
    const m = await (await realFetch("data/manifest.json", { cache: "no-cache", signal })).json();
    index.remote = { last_date: m.last_date, generated_at: m.generated_at, total_size: m.total_size };
    // 先下现有数据，再从新到旧下历史（中途取消时，已下的历史也能接上用）
    const need = wanted(m).filter((f) => !has(f));
    const baseChanged = need.some((f) => f.kind !== "history");
    const total = need.reduce((a, f) => a + f.size, 0);
    Object.assign(sync, { phase: "daily", phase_label: "下载数据（MB）", total: +(total / 1e6).toFixed(1), done: 0, updated_at: now() });
    let got = 0;
    for (const f of need) {
      const r = await realFetch("data/" + f.name + "?v=" + f.sha256.slice(0, 12), { signal });
      if (!r.ok) throw new Error(`下载 ${f.name} 失败：HTTP ${r.status}`);
      const reader = r.body.getReader(), chunks = [];
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        chunks.push(value); got += value.length;
        sync.done = +(got / 1e6).toFixed(1); sync.updated_at = now();
      }
      const buf = await new Blob(chunks).arrayBuffer();
      const hash = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", buf)), (b) => b.toString(16).padStart(2, "0")).join("");
      if (hash !== f.sha256) throw new Error(`${f.name} 校验失败，请重试`);
      await XGStore.put(f.name, buf);
      index.files[f.name] = f.sha256; saveIndex(index);
    }
    // 删掉远端已经没有的旧文件（比如上个月的月文件并进了年文件）
    const keep = new Set(m.files.map((f) => f.name));
    for (const n of Object.keys(index.files)) if (!keep.has(n)) { await XGStore.del(n); delete index.files[n]; }
    const histChanged = !!index.manifest && JSON.stringify(histFiles(index.manifest).map((f) => f.sha256)) !== JSON.stringify(histFiles(m).map((f) => f.sha256));
    index.manifest = m; saveIndex(index);
    Object.assign(sync, { phase: "merge", phase_label: "载入数据" });
    // 只新下了历史的话不用重载：下次筛选用到时再接上
    if (worker && (baseChanged || histChanged || !dataLoaded)) { dataLoaded = loadBase(m); await dataLoaded; }
    const hist = usableHist(m);
    Object.assign(sync, { state: "done", finished_at: now(), done_at: Date.now(), updated_to: m.last_date,
      extended_to: mode === "extend" && hist.length ? hist[hist.length - 1].from : null,
      message: need.length ? `已更新到 ${m.last_date} · 下载 ${(total / 1e6).toFixed(1)} MB` : `已是最新（${m.last_date}）` });
    Object.assign(check, { state: "latest", last_date: m.last_date, mb: 0, total_mb: +(m.total_size / 1e6).toFixed(1), message: "" });
  }

  function dataStatus() {
    const m = index.manifest;
    if (!complete()) {
      const end = new Date(), start = new Date(end.getTime() - 3 * 365 * 86400000);
      return { market: "cn_a", demo: true, stocks: 300, first_date: start.toISOString().slice(0, 10), last_date: end.toISOString().slice(0, 10),
               remote: index.remote || null, cached_mb: 0 };
    }
    const meta = m.meta || {};
    const usable = usableHist(m), hf = histFiles(m);
    const cached = m.files.filter(has).reduce((a, f) => a + f.size, 0);
    return { market: "cn_a", demo: false, stocks: m.stocks, first_date: usable.length ? usable[usable.length - 1].from : m.first_date,
             base_first_date: m.first_date, last_date: m.last_date, synced_at: m.generated_at,
             list_source: meta.list_source || "", daily_source: meta.daily_source || "", industry_source: meta.industry_source || "",
             years: meta.years, remote: index.remote || null, cached_mb: +(cached / 1e6).toFixed(1), static: true,
             // 时间范围：网站上有的更早历史（按年），以及本机已经补到哪
             history: hf.length ? { from: hf[hf.length - 1].from, files: hf.map((f) => ({ year: f.year, from: f.from, to: f.to, size: f.size, have: has(f) })) } : null };
  }

  // ---------------------------------------------------------------- 筛选
  let examplesCache = null;
  const examples = async () => examplesCache || (examplesCache = await (await realFetch("examples.json")).json());

  async function screen(req) {
    const t0 = performance.now();
    let code = req.code, understanding = "手动修改的代码", source = "manual";
    if (!code && req.example_id) {
      const ex = (await examples()).find((e) => e.id === req.example_id);
      if (!ex) throw Object.assign(new Error("示例不存在"), { status: 404 });
      code = ex.code; understanding = ex.understanding; source = "example";
    }
    await ensureData();
    const r = await runCode(req.market, req.start, req.end, code);
    return { ...r, code, understanding, assumptions: [], attempts: 1, seconds: +((performance.now() - t0) / 1000).toFixed(1), source };
  }
  async function runCode(market, start, end, code, emit) {
    let long = false;
    try { long = await ensureHistory(start, emit); }
    catch (e) { if (e.kind === "engine" || /计算环境/.test(e.message)) throw e; return { ok: false, error: "载入更早的历史数据失败：" + e.message }; }
    if (long) emit?.({ type: "status", stage: "running", text: "运行中（数据多，可能要半分钟左右）" });
    try {
      const go = () => call("run", [market, start, end, code], EXEC_TIMEOUT);
      return await (long ? withStage("筛选中", "数据多，可能要半分钟左右", go) : go());
    }
    catch (e) { if (e.kind === "engine" || /计算环境/.test(e.message)) throw e; return { ok: false, error: e.message }; }
  }

  // ---------------------------------------------------------------- AI（浏览器直接调服务商，key 只在本机）
  function aiFromHeaders(h) {
    const key = h["X-XG-Key"]; if (!key) return null;
    const provider = h["X-XG-Provider"] || "deepseek";
    return { provider, key, model: h["X-XG-Model"] || AI_DEFAULTS[provider], effort: h["X-XG-Effort"] || null };
  }
  function explain(status, msg) {
    const low = (msg || "").toLowerCase();
    if (status === 401) return ["auth", "API key 不正确或已失效，请检查是否复制完整"];
    if (status === 402 || /insufficient|balance|credits/.test(low)) return ["balance", "账户余额不足，请到服务商后台充值"];
    if (status === 404 || (/model/.test(low) && /not exist|not found|invalid model|not a valid model/.test(low))) return ["model", "模型不存在，请检查模型名称"];
    if (status === 429) return ["rate", "请求太频繁或额度用完，稍后再试"];
    if (status === 0) return ["network", "连不上服务商，请检查网络或代理"];
    return ["other", (msg || `HTTP ${status}`).slice(0, 200)];
  }
  class AIError extends Error { constructor(kind, msg) { super(msg); this.kind = kind; } }

  // 流式调用一次模型；onReasoning / onStage 推进度。返回 { text, think }
  async function chat(ai, model, messages, p, emit, signal) {
    const prov = { ...PROVIDERS[ai.provider] };
    try { const dev = localStorage.getItem("xg.dev.ai." + ai.provider); if (dev) prov.url = dev; } catch (e) {}  // 仅开发测试用：指向本地模拟服务
    const body = { model, messages, stream: true, max_tokens: p.max_tokens, response_format: { type: "json_object" }, ...p.extra };
    if (!p.reasoning) body.temperature = 0;
    if (ai.provider === "openrouter" && p.reasoning) body.reasoning = { exclude: false };
    const headers = { "Content-Type": "application/json", Authorization: "Bearer " + ai.key };
    if (ai.provider === "openrouter") Object.assign(headers, { "X-Title": "StockSieve", "HTTP-Referer": location.origin });
    let r;
    try {
      r = await realFetch(prov.url, { method: "POST", headers, body: JSON.stringify(body), signal });
      if (r.status === 400) {  // 有的模型不支持 JSON 模式，退回普通输出
        delete body.response_format;
        r = await realFetch(prov.url, { method: "POST", headers, body: JSON.stringify(body), signal });
      }
    } catch (e) { throw new AIError("network", "连不上服务商，请检查网络或代理"); }
    if (!r.ok) {
      let msg = ""; try { const j = await r.json(); msg = j.error?.message || j.message || JSON.stringify(j); } catch (e) {}
      const [k, m] = explain(r.status, msg.replaceAll(ai.key, "***")); throw new AIError(k, m);
    }
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = "", text = "", think = "", stage = "", finish = null;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1);
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") continue;
        let j; try { j = JSON.parse(data); } catch (e) { continue; }
        const ch = j.choices?.[0]; if (!ch) continue;
        const d = ch.delta || {};
        const rz = d.reasoning_content || d.reasoning;
        if (rz) { think += rz; if (stage !== "thinking") { stage = "thinking"; emit({ type: "status", stage }); } emit({ type: "reasoning", text: rz }); }
        if (d.content) { text += d.content; if (stage !== "writing") { stage = "writing"; emit({ type: "status", stage }); } }
        if (ch.finish_reason) finish = ch.finish_reason;
      }
    }
    if (finish === "length") throw new AIError("other", "模型输出过长被截断，请把规则拆简单一点");
    return { text, think };
  }

  // 和本地版 llm.generate_and_run 同样的流程：生成 → 运行 → 出错就修（最多 2 次）
  async function generateAndRun(req, ai, emit, signal) {
    const t0 = performance.now();
    if (!dataLoaded) emit({ type: "status", stage: "loading" });  // 第一次：先在浏览器里启动 Python、载入数据
    await ensureData();
    const params = await py("call_params", ai.provider, ai.model, ai.effort || "");
    const user = req.refine ? await py("refine_message", JSON.stringify(req.refine), req.start, req.end) : await py("user_message", req.query, req.start, req.end);
    const messages = [{ role: "system", content: await py("system_prompt") }, { role: "user", content: user }];
    const thoughts = [];
    let plan = {}, result = { ok: false, error: "未执行" }, attempts = 0, rewroteRetry = false;

    async function ask(useFix) {
      let model = ai.model, p = params;
      if (useFix && params.reasoning) { const f = await py("fix_model", ai.provider, ai.model); model = f.model; p = f; }
      emit({ type: "status", stage: p.reasoning ? "thinking" : "writing" });
      const { text, think } = await chat(ai, model, messages, p, emit, signal);
      if (think) thoughts.push(think);
      const parsed = await py("parse_answer", text);
      if (!parsed.ok) throw new AIError("other", parsed.error);
      return { p: parsed.plan, raw: text };
    }

    for (let i = 0; i <= 2; i++) {
      attempts++;
      if (i) emit({ type: "status", stage: "fixing", attempt: i });
      let { p, raw } = await ask(i > 0);
      if (req.refine && !rewroteRetry && await py("rewrote", req.refine.base_code || "", p.code, p.change || "")) {
        rewroteRetry = true; attempts++;
        messages.push({ role: "assistant", content: raw }, { role: "user", content: await call("get", ["REWRITE_RETRY"]) });
        ({ p, raw } = await ask(false));
        if (await py("rewrote", req.refine.base_code || "", p.code, p.change || "")) {
          p.change = ((p.change || "").replace(/。$/, "") + await call("get", ["REWRITE_NOTE"])).replace(/^（/, "");
        }
      }
      plan = p;
      emit({ type: "status", stage: "running" });
      result = await runCode(req.market, req.start, req.end, plan.code, emit);
      if (result.ok || i === 2) break;
      messages.push({ role: "assistant", content: raw }, { role: "user", content: await py("fix_message", result.error || "", result.traceback || "") });
    }
    return {
      ...result, plan: plan.plan || "", reasoning: thoughts.join("\n\n").slice(-20000),
      understanding: plan.understanding || "", assumptions: plan.assumptions || [],
      query: req.refine ? (plan.query || "").trim() : req.query, change: req.refine ? (plan.change || "").trim() : "",
      change_detail: req.refine ? (plan.change_detail || []).slice(0, 3).map(String) : [],
      code: plan.code || "", attempts, seconds: +((performance.now() - t0) / 1000).toFixed(1), source: "ai",
    };
  }

  function streamResponse(req, ai) {
    const enc = new TextEncoder();
    const ctrl = new AbortController();
    const stream = new ReadableStream({
      async start(c) {
        const emit = (e) => c.enqueue(enc.encode(JSON.stringify(e) + "\n"));
        try { emit({ type: "result", ...(await generateAndRun(req, ai, emit, ctrl.signal)) }); }
        catch (e) { emit({ type: "error", code: "ai_" + (e.kind || "other"), message: e.kind === "engine" || /计算环境/.test(e.message) ? e.message : "AI 调用失败：" + e.message }); }
        c.close();
      },
      cancel() { ctrl.abort(); },
    });
    return new Response(stream, { headers: { "Content-Type": "application/x-ndjson" } });
  }

  // 模型参数规则的 JS 版：只给「测试连接」用，这样测试不依赖浏览器里的 Python（它可能还没加载好或加载失败）。
  // 必须和 app/prompts.py 的 call_params / MAX_TOKENS 保持一致（两处同步修改）。
  const EFFORTS = ["none", "low", "high", "max"], LEGACY_DS = ["deepseek-chat", "deepseek-reasoner"];
  const HINTS = ["reasoner", "r1", "think", "reason", "o1", "o3", "o4"];
  const MAX_TOKENS = { none: 8000, low: 32000, high: 64000, max: 128000 };
  function callParamsJS(provider, model, effort) {
    const m = (model || "").toLowerCase();
    if (provider === "deepseek" && m.startsWith("deepseek-") && !LEGACY_DS.includes(m)) {
      const e = EFFORTS.includes(effort) ? effort : "high";
      return { reasoning: e !== "none", extra: { reasoning_effort: e }, max_tokens: MAX_TOKENS[e] };
    }
    const r = provider === "deepseek" ? m === "deepseek-reasoner" : HINTS.some((h) => m.includes(h));
    return { reasoning: r, extra: {}, max_tokens: r ? 32000 : 8000 };
  }

  async function aiTest(ai) {
    const t0 = performance.now();
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 20000);
    try {
      const p = callParamsJS(ai.provider, ai.model, "none");
      await chat(ai, ai.model, [{ role: "user", content: "ping" }], { ...p, max_tokens: 1, reasoning: false }, () => {}, ctrl.signal);
      return { ok: true, ms: Math.round(performance.now() - t0), error_kind: null, message: `连接成功 · ${PROVIDERS[ai.provider].label} · ${ai.model}` };
    } catch (e) {
      // 只要求 1 个 token，JSON 模式下可能被判为截断，这也说明 key 和模型都是通的
      if (e.message.includes("截断")) return { ok: true, ms: Math.round(performance.now() - t0), error_kind: null, message: `连接成功 · ${PROVIDERS[ai.provider].label} · ${ai.model}` };
      if (ctrl.signal.aborted) return { ok: false, ms: Math.round(performance.now() - t0), error_kind: "network", message: "连接超时（20 秒没有响应），请检查网络或代理" };
      return { ok: false, ms: Math.round(performance.now() - t0), error_kind: e.kind || "other", message: e.message };
    } finally { clearTimeout(timer); }
  }

  // ---------------------------------------------------------------- /api/* 就地实现
  const json = (obj, status = 200) => new Response(JSON.stringify(obj), { status, headers: { "Content-Type": "application/json" } });
  const fail = (e) => json({ detail: e.detail || e.message }, e.status || 400);

  async function handle(path, opt) {
    const method = (opt?.method || "GET").toUpperCase();
    const body = opt?.body ? JSON.parse(opt.body) : {};
    const headers = opt?.headers || {};
    if (path === "/api/meta") {
      const ex = await examples();
      return json({ markets: MARKETS, status: dataStatus(), examples: ex.map(({ id, title, query }) => ({ id, title, query })),
                    ai: false, provider: null, model: null, ai_defaults: AI_DEFAULTS, static: true });
    }
    if (path === "/api/screen" && method === "POST") return json(await screen(body));
    if (path === "/api/screen/stream" && method === "POST") {
      const ai = aiFromHeaders(headers);
      if (!ai) return json({ detail: { code: "no_ai", message: "先填一个 API Key 才能用自己写的规则，示例和收藏不需要" } }, 400);
      if (body.refine && !(body.refine.instruction || "").trim()) return json({ detail: "请输入修改要求" }, 400);
      if (!body.refine && !(body.query || "").trim()) return json({ detail: "请输入筛选规则" }, 400);
      return streamResponse(body, ai);
    }
    if (path === "/api/data/status") return json({ data: dataStatus(), sync: { ...sync }, check: { ...check } });
    if (path === "/api/data/check" && method === "POST") { runCheck(); return json({ ok: true }); }
    if (path === "/api/data/sync" && method === "POST") { await startSync(["full", "clear", "extend"].includes(body.mode) ? body.mode : "update", body.from); return json({ ok: true }); }
    if (path === "/api/data/cancel" && method === "POST") { if (sync.state !== "running") return json({ detail: "当前没有进行中的下载" }, 409); abort?.abort(); return json({ ok: true }); }
    if (path === "/api/ai/test" && method === "POST") {
      const ai = aiFromHeaders(headers);
      return json(ai ? await aiTest(ai) : { ok: false, ms: 0, error_kind: "nokey", message: "请先填写 API key" });
    }
    return json({ detail: "未知接口" }, 404);
  }

  window.fetch = async (input, opt) => {
    const url = typeof input === "string" ? input : input.url;
    if (url.startsWith("/api/")) {
      try { return await handle(url.split("?")[0], opt); } catch (e) { return fail(e); }
    }
    return realFetch(input, opt);
  };

  // 页面可以读的一些状态
  window.XG_STATIC = {
    engine, complete, index: () => index,
    preload() { ensureData(); },  // 空闲时先把 Python 环境和数据载入，第一次筛选更快
    async persist() { try { return await navigator.storage.persist(); } catch (e) { return false; } },
    storeKind: () => XGStore.kind(),
  };
})();
