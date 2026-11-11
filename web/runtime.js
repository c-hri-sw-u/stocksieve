// 静态网页版的运行时：没有后端服务器，页面原来请求的 /api/... 都在这里就地实现。
// - 行情数据：从同一网站的 data/ 目录下载，存在本机（OPFS / IndexedDB），按 manifest 只下载缺的或变了的文件
// - 计算：worker.js 里的 Pyodide 跑和本地版同一份 Python 代码
// - AI：浏览器直接调 DeepSeek / OpenRouter，key 只在本机
(function () {
  const realFetch = window.fetch.bind(window);
  const INDEX_KEY = "xg.data.v1";
  const EXEC_TIMEOUT = 90000;
  const MARKETS = { cn_a: { label: "A股", enabled: true }, hk: { label: "港股", enabled: false }, us: { label: "美股", enabled: false } };
  const AI_DEFAULTS = { deepseek: "deepseek-flash", openrouter: "deepseek/deepseek-chat" };
  const PROVIDERS = {
    deepseek: { label: "DeepSeek", url: "https://api.deepseek.com/chat/completions" },
    openrouter: { label: "OpenRouter", url: "https://openrouter.ai/api/v1/chat/completions" },
  };

  // ---------------------------------------------------------------- 本机数据索引
  const loadIndex = () => { try { return JSON.parse(localStorage.getItem(INDEX_KEY) || "null") || { files: {} }; } catch (e) { return { files: {} }; } };
  const saveIndex = (ix) => { try { localStorage.setItem(INDEX_KEY, JSON.stringify(ix)); } catch (e) {} };
  let index = loadIndex();  // { manifest: 本机这份数据对应的 manifest, files: {名字: sha256}, remote: 最近看到的远端 manifest 概要 }
  const complete = () => !!index.manifest && index.manifest.files.every((f) => index.files[f.name] === f.sha256);

  // ---------------------------------------------------------------- 计算 worker
  let worker = null, seq = 0, pending = new Map(), workerReady = null, dataLoaded = null;
  const engine = { stage: "", readyMs: null };
  function startWorker() {
    worker = new Worker("worker.js");
    workerReady = new Promise((res) => {
      worker.onmessage = (e) => {
        const m = e.data;
        if (m.type === "progress") { engine.stage = m.stage; return; }
        if (m.type === "ready") { engine.stage = ""; engine.readyMs = m.ms; res(); return; }
        const p = pending.get(m.id); if (!p) return;
        pending.delete(m.id); clearTimeout(p.timer);
        m.ok ? p.resolve(m.result) : p.reject(new Error(m.error));
      };
    });
    dataLoaded = null;
  }
  function call(cmd, args = [], timeout = 0) {
    if (!worker) startWorker();
    const id = ++seq;
    return new Promise((resolve, reject) => {
      const p = { resolve, reject, timer: null };
      if (timeout) p.timer = setTimeout(() => {
        // 超时：直接结束 worker（停止计算），下次用到时重新启动
        pending.delete(id); worker.terminate(); worker = null;
        for (const q of pending.values()) q.reject(new Error("计算已中止"));
        pending.clear();
        reject(new Error(`运行超时（超过 ${timeout / 1000} 秒），请简化计算`));
      }, timeout);
      pending.set(id, p);
      worker.postMessage({ id, cmd, args });
    });
  }
  async function ensureData() {
    if (!worker) startWorker();
    if (!dataLoaded) {
      dataLoaded = complete() ? call("loadData", [index.manifest]) : Promise.resolve({ ok: false });
    }
    return dataLoaded;
  }
  const py = (fn, ...args) => call("call", [fn, ...args]);

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
      const need = m.files.filter((f) => index.files[f.name] !== f.sha256);
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

  async function startSync(mode) {
    if (sync.state === "running") throw Object.assign(new Error("已经有下载在进行中"), { status: 409 });
    Object.assign(sync, { state: "running", phase: "list", phase_label: "获取文件清单", mode, mode_label: mode === "clear" ? "清除本地数据" : "下载到本机",
                          done: 0, total: 0, failed: [], message: "", started_at: now(), updated_at: now(), finished_at: null,
                          list_source: "", daily_source: "" });
    abort = new AbortController();
    runSync(mode, abort.signal).catch((e) => {
      Object.assign(sync, { state: abort.signal.aborted ? "cancelled" : "error", message: abort.signal.aborted ? "已取消，已下载的文件保留" : e.message, finished_at: now() });
    });
  }

  async function runSync(mode, signal) {
    if (mode === "clear") {
      await XGStore.clear(); index = { files: {}, remote: index.remote }; saveIndex(index);
      if (worker) { worker.terminate(); worker = null; }
      Object.assign(sync, { state: "idle", message: "本地数据已清除", finished_at: now() });
      Object.assign(check, { state: "none", mb: index.remote?.total_size ? +(index.remote.total_size / 1e6).toFixed(1) : 0 });
      return;
    }
    if (mode === "full") { await XGStore.clear(); index.files = {}; saveIndex(index); }
    const m = await (await realFetch("data/manifest.json", { cache: "no-cache", signal })).json();
    index.remote = { last_date: m.last_date, generated_at: m.generated_at, total_size: m.total_size };
    const need = m.files.filter((f) => index.files[f.name] !== f.sha256);
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
    index.manifest = m; saveIndex(index);
    Object.assign(sync, { phase: "merge", phase_label: "载入数据" });
    if (worker) { dataLoaded = call("loadData", [m]); await dataLoaded; }
    Object.assign(sync, { state: "done", finished_at: now(), done_at: Date.now(), updated_to: m.last_date,
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
    return { market: "cn_a", demo: false, stocks: m.stocks, first_date: m.first_date, last_date: m.last_date, synced_at: m.generated_at,
             list_source: meta.list_source || "", daily_source: meta.daily_source || "", industry_source: meta.industry_source || "",
             years: meta.years, remote: index.remote || null, cached_mb: +(m.total_size / 1e6).toFixed(1), static: true };
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
  async function runCode(market, start, end, code) {
    try { return await call("run", [market, start, end, code], EXEC_TIMEOUT); }
    catch (e) { return { ok: false, error: e.message }; }
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
      result = await runCode(req.market, req.start, req.end, plan.code);
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
        catch (e) { emit({ type: "error", code: "ai_" + (e.kind || "other"), message: "AI 调用失败：" + e.message }); }
        c.close();
      },
      cancel() { ctrl.abort(); },
    });
    return new Response(stream, { headers: { "Content-Type": "application/x-ndjson" } });
  }

  async function aiTest(ai) {
    const t0 = performance.now();
    try {
      const p = await py("call_params", ai.provider, ai.model, "none");
      await chat(ai, ai.model, [{ role: "user", content: "ping" }], { ...p, max_tokens: 1, reasoning: false }, () => {});
      return { ok: true, ms: Math.round(performance.now() - t0), error_kind: null, message: `连接成功 · ${PROVIDERS[ai.provider].label} · ${ai.model}` };
    } catch (e) {
      // 只要求 1 个 token，JSON 模式下可能被判为截断，这也说明 key 和模型都是通的
      if (e.message.includes("截断")) return { ok: true, ms: Math.round(performance.now() - t0), error_kind: null, message: `连接成功 · ${PROVIDERS[ai.provider].label} · ${ai.model}` };
      return { ok: false, ms: Math.round(performance.now() - t0), error_kind: e.kind || "other", message: e.message };
    }
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
    if (path === "/api/data/sync" && method === "POST") { await startSync(body.mode === "full" || body.mode === "clear" ? body.mode : "update"); return json({ ok: true }); }
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
