// 本机数据存储：优先用 OPFS（浏览器给网站的私有文件夹），不支持时退回 IndexedDB。
// 页面主线程和计算 worker 都用这同一份代码（worker 里用 importScripts 加载）。
(function (g) {
  const DIR = "xg-data";
  let opfs = null;

  async function dir() {
    if (opfs !== null) return opfs;
    try {
      const root = await navigator.storage.getDirectory();
      opfs = await root.getDirectoryHandle(DIR, { create: true });
    } catch (e) { opfs = false; }
    return opfs;
  }

  // ---- IndexedDB 退路
  let dbp = null;
  function db() {
    if (!dbp) dbp = new Promise((res, rej) => {
      const r = indexedDB.open("xg-data", 1);
      r.onupgradeneeded = () => r.result.createObjectStore("files");
      r.onsuccess = () => res(r.result);
      r.onerror = () => rej(r.error);
    });
    return dbp;
  }
  async function idb(mode, fn) {
    const d = await db();
    return new Promise((res, rej) => {
      const tx = d.transaction("files", mode);
      const req = fn(tx.objectStore("files"));
      tx.oncomplete = () => res(req && req.result);
      tx.onerror = () => rej(tx.error);
    });
  }

  g.XGStore = {
    async kind() { return (await dir()) ? "OPFS" : "IndexedDB"; },
    async put(name, buf) {
      const d = await dir();
      if (d) {
        const fh = await d.getFileHandle(name, { create: true });
        const w = await fh.createWritable();
        await w.write(buf); await w.close();
        return;
      }
      await idb("readwrite", (s) => s.put(new Blob([buf]), name));
    },
    async get(name) {
      const d = await dir();
      if (d) {
        try { return await (await (await d.getFileHandle(name)).getFile()).arrayBuffer(); } catch (e) { return null; }
      }
      const b = await idb("readonly", (s) => s.get(name));
      return b ? await b.arrayBuffer() : null;
    },
    async del(name) {
      const d = await dir();
      if (d) { try { await d.removeEntry(name); } catch (e) {} return; }
      await idb("readwrite", (s) => s.delete(name));
    },
    async list() {
      const d = await dir();
      if (d) { const out = []; for await (const k of d.keys()) out.push(k); return out; }
      return (await idb("readonly", (s) => s.getAllKeys())) || [];
    },
    async clear() { for (const n of await this.list()) await this.del(n); },
  };
})(self);
