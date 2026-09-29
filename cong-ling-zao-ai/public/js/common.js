(function (global) {
  const CLZAI = global.CLZAI || (global.CLZAI = {});

  function qs(sel, el) { return (el || document).querySelector(sel); }
  function qsa(sel, el) { return Array.from((el || document).querySelectorAll(sel)); }

  function headerHTML(active) {
    const items = [
      ["catalog.html", "课程目录"],
      ["paths.html", "学习路径"],
      ["progress.html", "我的进度"],
      ["about.html", "关于"],
    ];
    const links = items.map(([href, label]) => {
      const cls = active === href ? "active" : "";
      return `<li><a class="${cls}" href="${href}">${label}</a></li>`;
    }).join("");
    return `
<header class="site-header">
  <div class="wrap nav">
    <a class="logo" href="index.html">从零造 <span>AI</span></a>
    <button type="button" class="nav-toggle" aria-label="菜单" id="navToggle">菜单</button>
    <ul class="nav-links" id="navLinks">${links}</ul>
  </div>
</header>`;
  }

  function footerHTML() {
    return `
<footer class="site-footer">
  <div class="wrap footer-inner">
    <div>© 从零造 AI · 学习优先</div>
    <div>
      内容改编自
      <a href="https://github.com/rohitg00/ai-engineering-from-scratch" target="_blank" rel="noopener">AI Engineering from Scratch</a>
      （MIT，Rohit Ghumare 等）·
      <strong>非官方中文版</strong>
    </div>
  </div>
</footer>`;
  }

  function mountChrome(active) {
    const h = qs("#site-header-slot");
    const f = qs("#site-footer-slot");
    if (h) h.outerHTML = headerHTML(active);
    if (f) f.outerHTML = footerHTML();
    const toggle = qs("#navToggle");
    const links = qs("#navLinks");
    if (toggle && links) {
      toggle.addEventListener("click", () => links.classList.toggle("open"));
    }
    if (CLZAI.Progress) CLZAI.Progress.applyTheme();
  }

  let catalogCache = null;
  async function loadCatalog() {
    if (catalogCache) return catalogCache;
    const res = await fetch("/api/catalog");
    if (!res.ok) {
      const res2 = await fetch("/data/catalog.json");
      if (!res2.ok) throw new Error("无法加载课程目录");
      catalogCache = await res2.json();
      return catalogCache;
    }
    catalogCache = await res.json();
    return catalogCache;
  }

  function findLesson(catalog, path) {
    const norm = String(path).replace(/^\/+/, "").replace(/\/docs\/.*$/, "");
    for (const ph of catalog.phases) {
      for (const l of ph.lessons) {
        if (l.path === norm) return { phase: ph, lesson: l };
      }
    }
    return null;
  }

  function nextLesson(catalog, path) {
    const found = findLesson(catalog, path);
    if (!found) return null;
    const { phase, lesson } = found;
    const idx = phase.lessons.findIndex((l) => l.path === lesson.path);
    if (idx >= 0 && idx < phase.lessons.length - 1) {
      return { phase, lesson: phase.lessons[idx + 1] };
    }
    // next phase first lesson
    const pidx = catalog.phases.findIndex((p) => p.id === phase.id);
    if (pidx >= 0 && pidx < catalog.phases.length - 1) {
      const np = catalog.phases[pidx + 1];
      if (np.lessons.length) return { phase: np, lesson: np.lessons[0] };
    }
    return null;
  }

  function prevLesson(catalog, path) {
    const found = findLesson(catalog, path);
    if (!found) return null;
    const { phase, lesson } = found;
    const idx = phase.lessons.findIndex((l) => l.path === lesson.path);
    if (idx > 0) return { phase, lesson: phase.lessons[idx - 1] };
    const pidx = catalog.phases.findIndex((p) => p.id === phase.id);
    if (pidx > 0) {
      const pp = catalog.phases[pidx - 1];
      if (pp.lessons.length) return { phase: pp, lesson: pp.lessons[pp.lessons.length - 1] };
    }
    return null;
  }

  function lessonUrl(path) {
    return `lesson.html?path=${encodeURIComponent(path)}`;
  }

  function continueTarget(catalog) {
    const data = CLZAI.Progress.load();
    if (data.last && data.last.path) {
      const found = findLesson(catalog, data.last.path);
      if (found) {
        // If last is done, go to next; else resume last
        if (CLZAI.Progress.isDone(data.last.path)) {
          const n = nextLesson(catalog, data.last.path);
          if (n) return n;
        }
        return found;
      }
    }
    // default: first lesson of phase 0
    const p0 = catalog.phases.find((p) => p.id === 0);
    if (p0 && p0.lessons[0]) return { phase: p0, lesson: p0.lessons[0] };
    return null;
  }

  CLZAI.qs = qs;
  CLZAI.qsa = qsa;
  CLZAI.mountChrome = mountChrome;
  CLZAI.loadCatalog = loadCatalog;
  CLZAI.findLesson = findLesson;
  CLZAI.nextLesson = nextLesson;
  CLZAI.prevLesson = prevLesson;
  CLZAI.lessonUrl = lessonUrl;
  CLZAI.continueTarget = continueTarget;
})(window);
