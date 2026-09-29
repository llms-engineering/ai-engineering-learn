/** localStorage progress for 从零造 AI */
(function (global) {
  const KEY = "clzai_progress_v1";
  const THEME_KEY = "clzai_theme";

  function load() {
    try {
      const raw = localStorage.getItem(KEY);
      if (!raw) return { completed: {}, last: null, updatedAt: null };
      const data = JSON.parse(raw);
      return {
        completed: data.completed || {},
        last: data.last || null,
        updatedAt: data.updatedAt || null,
      };
    } catch (_) {
      return { completed: {}, last: null, updatedAt: null };
    }
  }

  function save(data) {
    data.updatedAt = new Date().toISOString();
    localStorage.setItem(KEY, JSON.stringify(data));
    return data;
  }

  function lessonKey(path) {
    // path like phases/00-setup/01-dev → lesson:0:01
    const m = String(path).match(/phases\/(\d{2})-[^/]+\/(\d{2})-/);
    if (m) return `lesson:${parseInt(m[1], 10)}:${m[2]}`;
    return `lesson:${path}`;
  }

  const Progress = {
    load,
    isDone(path) {
      const k = lessonKey(path);
      return !!load().completed[k];
    },
    markDone(path, meta) {
      const data = load();
      const k = lessonKey(path);
      data.completed[k] = {
        at: new Date().toISOString(),
        path,
        title: (meta && meta.title) || "",
        phaseId: meta && meta.phaseId,
      };
      data.last = { path, title: (meta && meta.title) || "", phaseId: meta && meta.phaseId, at: data.completed[k].at };
      return save(data);
    },
    unmark(path) {
      const data = load();
      delete data.completed[lessonKey(path)];
      return save(data);
    },
    setLast(path, meta) {
      const data = load();
      data.last = {
        path,
        title: (meta && meta.title) || "",
        phaseId: meta && meta.phaseId,
        at: new Date().toISOString(),
      };
      return save(data);
    },
    countCompleted() {
      return Object.keys(load().completed).length;
    },
    completedInPhase(phaseId, lessons) {
      const data = load();
      let n = 0;
      for (const l of lessons || []) {
        if (data.completed[lessonKey(l.path)]) n++;
      }
      return n;
    },
    recent(limit) {
      const data = load();
      return Object.entries(data.completed)
        .map(([k, v]) => ({ key: k, ...v }))
        .sort((a, b) => (b.at || "").localeCompare(a.at || ""))
        .slice(0, limit || 5);
    },
    clear() {
      localStorage.removeItem(KEY);
      return load();
    },
    export() {
      return JSON.stringify(load(), null, 2);
    },
    getTheme() {
      return localStorage.getItem(THEME_KEY) || "light";
    },
    setTheme(t) {
      localStorage.setItem(THEME_KEY, t);
      document.documentElement.setAttribute("data-theme", t === "dark" ? "dark" : "light");
    },
    applyTheme() {
      const t = Progress.getTheme();
      document.documentElement.setAttribute("data-theme", t === "dark" ? "dark" : "light");
    },
    lessonKey,
  };

  global.CLZAI = global.CLZAI || {};
  global.CLZAI.Progress = Progress;
})(window);
