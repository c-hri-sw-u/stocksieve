// 计算 worker：在浏览器里用 Pyodide 跑 Python（pandas），执行筛选代码。
// 和本地网站共用同一份 Python 源码（app/*.py），由 scripts/build_web.py 拷到 py/ 下。
importScripts("storage.js");
const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.29.3/full/";
importScripts(PYODIDE_URL + "pyodide.js");

let py = null, X = null;
const say = (stage) => postMessage({ type: "progress", stage });

const ready = (async () => {
  const t0 = performance.now();
  say("加载 Python 运行环境");
  py = await loadPyodide({ indexURL: PYODIDE_URL });
  say("加载 pandas");
  await py.loadPackage(["numpy", "pandas", "pyarrow"]);
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
})();

const handlers = {
  // 把本机缓存的数据文件交给 Python，读进内存
  async loadData(manifest) {
    const dir = "/xgdata";
    py.FS.mkdirTree(dir);
    for (const n of py.FS.readdir(dir)) if (n !== "." && n !== "..") py.FS.unlink(dir + "/" + n);
    for (const f of manifest.files) {
      const buf = await XGStore.get(f.name);
      if (buf) py.FS.writeFile(dir + "/" + f.name, new Uint8Array(buf));
    }
    const r = JSON.parse(X.load_files(JSON.stringify(manifest), dir));
    for (const n of py.FS.readdir(dir)) if (n !== "." && n !== "..") py.FS.unlink(dir + "/" + n);  // 已经在内存里了，删掉省空间
    return r;
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
    postMessage({ id, ok: false, error: String(err && err.message || err) });
  }
};
