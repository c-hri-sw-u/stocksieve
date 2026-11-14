// 计算 worker：在浏览器里用 Pyodide 跑 Python（pandas），执行筛选代码。
// 和本地网站共用同一份 Python 源码（app/*.py），由 scripts/build_web.py 拷到 py/ 下。
importScripts("storage.js");
// Pyodide 优先从同一网站的 pyodide/ 目录加载（scripts/fetch_pyodide.py 放进去的），失败再试 CDN。
// 只靠 cdn.jsdelivr.net 的话，在连不上它的网络里计算环境会一直加载失败。
const PYODIDE_VERSION = "0.29.3";  // 和 scripts/fetch_pyodide.py 保持一致
const SOURCES = [
  new URL("pyodide/", self.location.href).href,
  `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`,
];

let py = null, X = null;
const say = (stage) => postMessage({ type: "progress", stage });

async function boot() {
  const t0 = performance.now();
  say("加载 Python 运行环境");
  let lastErr = null;
  for (const url of SOURCES) {
    try {
      importScripts(url + "pyodide.js");
      py = await loadPyodide({ indexURL: url });
      say("加载 pandas");
      await py.loadPackage(["numpy", "pandas", "pyarrow"]);
      lastErr = null;
      break;
    } catch (e) {
      lastErr = e; py = null;
      say("换一个地址加载 Python 运行环境");
    }
  }
  if (lastErr || !py) throw new Error("无法下载 Python 运行环境（网络问题）：" + String(lastErr && lastErr.message || lastErr).slice(0, 160));
  say("加载筛选程序");
  const files = await (await fetch("py/files.json", { cache: "no-cache" })).json();
  for (const f of files) {
    const path = "/home/pyodide/" + f;
    py.FS.mkdirTree(path.slice(0, path.lastIndexOf("/")));
    py.FS.writeFile(path, await (await fetch("py/" + f, { cache: "no-cache" })).text());
  }
  py.runPython("import sys; sys.path.insert(0, '/home/pyodide')");
  X = py.pyimport("xg_web");
  postMessage({ type: "ready", ms: Math.round(performance.now() - t0) });
}

const ready = boot().catch((e) => {
  // 告诉页面：计算环境起不来了，别让调用一直等
  postMessage({ type: "fatal", error: String(e && e.message || e) });
  throw e;
});

const handlers = {
  // 把本机缓存的数据文件交给 Python，读进内存
  async loadData(manifest) {
    const dir = "/xgdata";
    py.FS.mkdirTree(dir);
    for (const n of py.FS.readdir(dir)) if (n !== "." && n !== "..") py.FS.unlink(dir + "/" + n);
    const missing = [];  // 索引里记着已下载、本机缓存里却没有的文件（浏览器清掉了网站数据）
    for (const f of manifest.files) {
      if (f.kind === "history") continue;  // 更早的历史用到时才载入（addHistory）
      const buf = await XGStore.get(f.name);
      if (buf) py.FS.writeFile(dir + "/" + f.name, new Uint8Array(buf));
      else missing.push(f.name);
    }
    const r = { ...JSON.parse(X.load_files(JSON.stringify(manifest), dir)), missing };
    for (const n of py.FS.readdir(dir)) if (n !== "." && n !== "..") py.FS.unlink(dir + "/" + n);  // 已经在内存里了，删掉省空间
    return r;
  },
  // 把更早的历史（几个年份文件）接到已载入的数据前面
  async addHistory(names) {
    const dir = "/xghist";
    py.FS.mkdirTree(dir);
    try {
      const missing = [];
      for (const n of names) {
        const buf = await XGStore.get(n);
        if (buf) py.FS.writeFile(dir + "/" + n, new Uint8Array(buf));
        else missing.push(n);
      }
      if (missing.length) return { ok: false, missing, error: "本机缓存里的部分历史数据不见了" };
      return JSON.parse(X.add_history(JSON.stringify(names), dir));
    } finally {
      for (const n of py.FS.readdir(dir)) if (n !== "." && n !== "..") py.FS.unlink(dir + "/" + n);
    }
  },
  run(market, start, end, code) { return JSON.parse(X.run(market, start, end, code)); },
  // 调 xg_web 里的其他函数（提示词、解析模型输出等）
  call(fn, ...args) {
    const v = X[fn](...args);
    return typeof v === "string" && (fn === "parse_answer" || fn === "call_params" || fn === "fix_model" || fn === "examples") ? JSON.parse(v) : v;
  },
  get(name) { return X[name]; },
};

onmessage = async (e) => {
  const { id, cmd, args } = e.data;
  try {
    await ready;
    postMessage({ id, ok: true, result: await handlers[cmd](...(args || [])) });
  } catch (err) {
    const msg = String(err && err.message || err);
    // Python 运行环境崩了（多半是内存不够）：之后的调用都会失败，让页面重启 worker
    if (py && /fatal/i.test(msg)) {
      postMessage({ id, ok: false, error: "计算环境内存不够崩溃了，已重新启动。请缩短时间范围再试" });
      postMessage({ type: "crashed" });
      return;
    }
    postMessage({ id, ok: false, error: msg });
  }
};
