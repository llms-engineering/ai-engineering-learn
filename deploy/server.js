#!/usr/bin/env node
/**
 * Self-host server for AI Engineering from Scratch site.
 * Serves site/ statically and routes Vercel-style API handlers for lessons.
 * Runtime is GitHub-free: all curriculum/i18n is read from local disk only.
 * Based on https://github.com/rohitg00/ai-engineering-from-scratch (MIT).
 */
"use strict";

const http = require("http");
const fs = require("fs");
const path = require("path");
const { URL } = require("url");

const ROOT = path.resolve(__dirname, "..");
const SITE = path.join(ROOT, "site");
const PORT = Number(process.env.PORT || 80);
const HOST = process.env.HOST || "0.0.0.0";

const lessonHandler = require(path.join(ROOT, "api", "lesson.js"));
const certificationHandler = require(path.join(ROOT, "api", "certification.js"));
const markdownHandler = require(path.join(ROOT, "api", "markdown.js"));

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".svg": "image/svg+xml",
  ".webp": "image/webp",
  ".ico": "image/x-icon",
  ".txt": "text/plain; charset=utf-8",
  ".md": "text/markdown; charset=utf-8",
  ".xml": "application/xml; charset=utf-8",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
  ".ttf": "font/ttf",
  ".map": "application/json; charset=utf-8",
};

const CLEAN_HTML = {
  "/": "index.html",
  "/catalog": "catalog.html",
  "/glossary": "glossary.html",
  "/certifications": "certifications.html",
  "/assessment": "assessment.html",
  "/path": "prereqs.html",
  "/roadmap": "prereqs.html",
  "/about": "about.html",
  "/developer": "developer.html",
  "/docs": "developer.html",
  "/contact": "contact.html",
  "/privacy": "privacy.html",
  "/sponsors": "sponsors.html",
  "/learning-paths": "learning-paths.html",
};

// Also serve curriculum source trees so local content-source can resolve them
const EXTRA_ROOTS = ["phases", "certifications", "glossary", "learning-paths", "assets", "outputs", "skills", "i18n", "docs", "book"];

function parseQuery(urlObj) {
  const q = {};
  for (const [k, v] of urlObj.searchParams.entries()) {
    if (Object.prototype.hasOwnProperty.call(q, k)) {
      if (!Array.isArray(q[k])) q[k] = [q[k]];
      q[k].push(v);
    } else {
      q[k] = v;
    }
  }
  return q;
}

function wrapReq(req, urlObj) {
  req.query = parseQuery(urlObj);
  return req;
}

function sendFile(res, filePath, method) {
  fs.stat(filePath, (err, st) => {
    if (err || !st.isFile()) {
      const notFound = path.join(SITE, "404.html");
      fs.readFile(notFound, (e2, body) => {
        res.statusCode = 404;
        res.setHeader("Content-Type", "text/html; charset=utf-8");
        res.end(e2 ? "Not Found" : body);
      });
      return;
    }
    const ext = path.extname(filePath).toLowerCase();
    res.statusCode = 200;
    res.setHeader("Content-Type", MIME[ext] || "application/octet-stream");
    if ([".css", ".js", ".png", ".jpg", ".jpeg", ".svg", ".webp", ".woff2", ".woff", ".ttf", ".ico"].includes(ext)) {
      res.setHeader("Cache-Control", "public, max-age=86400");
    } else {
      res.setHeader("Cache-Control", "public, max-age=300");
    }
    if (method === "HEAD") {
      res.setHeader("Content-Length", String(st.size));
      res.end();
      return;
    }
    fs.createReadStream(filePath).pipe(res);
  });
}

function safeJoin(base, rel) {
  const resolved = path.resolve(base, rel);
  if (!resolved.startsWith(base + path.sep) && resolved !== base) return null;
  return resolved;
}

function wantsMarkdown(req) {
  const accept = String(req.headers.accept || "").toLowerCase();
  if (!accept) return false;
  const md = /text\/markdown/.test(accept);
  const html = /text\/html/.test(accept);
  return md && (!html || accept.indexOf("text/markdown") < accept.indexOf("text/html"));
}

function isSafeI18nRel(rel) {
  if (!rel || rel.includes("\\") || rel.includes("\0")) return false;
  if (rel.includes("..")) return false;
  // i18n/<lang>/ui.json or i18n/<lang>/phases/.../docs/<lang>.md etc.
  return /^i18n\/[A-Za-z0-9._-]+(?:\/[A-Za-z0-9._-]+)+$/.test(rel);
}

function sendI18n(res, rel, method) {
  const filePath = safeJoin(ROOT, rel);
  if (!filePath || !isSafeI18nRel(rel)) {
    res.statusCode = 403;
    return res.end("Forbidden");
  }
  // Local disk only — never proxy to GitHub at runtime.
  return sendFile(res, filePath, method);
}

function isSafeContentsPath(rel) {
  if (!rel || rel.includes("\\") || rel.includes("\0") || rel.includes("..")) return false;
  const top = rel.split("/").filter(Boolean)[0];
  if (!EXTRA_ROOTS.includes(top)) return false;
  return /^[A-Za-z0-9._/-]+$/.test(rel);
}

function sendContentsListing(res, rel) {
  const clean = String(rel || "").replace(/^\/+/, "").replace(/\/+$/, "");
  if (!isSafeContentsPath(clean)) {
    res.statusCode = 403;
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    return res.end(JSON.stringify({ error: "Forbidden" }));
  }
  const dirPath = safeJoin(ROOT, clean);
  if (!dirPath) {
    res.statusCode = 403;
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    return res.end(JSON.stringify({ error: "Forbidden" }));
  }
  fs.readdir(dirPath, { withFileTypes: true }, (err, entries) => {
    if (err) {
      res.statusCode = 404;
      res.setHeader("Content-Type", "application/json; charset=utf-8");
      return res.end(JSON.stringify({ error: "Not Found" }));
    }
    const out = [];
    for (const ent of entries) {
      if (!ent || !ent.name || ent.name.startsWith(".")) continue;
      const entryPath = clean + "/" + ent.name;
      const abs = path.join(dirPath, ent.name);
      let size = 0;
      try {
        const st = fs.statSync(abs);
        size = st.isFile() ? st.size : 0;
      } catch (_) {}
      const type = ent.isDirectory() ? "dir" : "file";
      out.push({
        name: ent.name,
        path: entryPath,
        size: size,
        type: type,
        html_url: "/" + entryPath,
        download_url: type === "file" ? "/" + entryPath : null,
      });
    }
    res.statusCode = 200;
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.setHeader("Cache-Control", "public, max-age=60");
    res.end(JSON.stringify(out));
  });
}

const server = http.createServer((req, res) => {
  try {
    const method = (req.method || "GET").toUpperCase();
    const urlObj = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);
    let pathname = decodeURIComponent(urlObj.pathname);
    if (pathname.length > 1 && pathname.endsWith("/")) pathname = pathname.slice(0, -1);

    // API / dynamic routes
    if (pathname === "/lesson" || pathname === "/api/lesson" || pathname === "/lesson.html") {
      if (pathname === "/lesson.html") urlObj.searchParams.set("legacy", "1");
      return lessonHandler(wrapReq(req, urlObj), res);
    }
    if (pathname === "/certification" || pathname === "/api/certification" || pathname === "/certification.html") {
      if (pathname === "/certification.html") urlObj.searchParams.set("legacy", "1");
      return certificationHandler(wrapReq(req, urlObj), res);
    }
    if (pathname === "/api/markdown" || pathname === "/api/v1/markdown") {
      return markdownHandler(wrapReq(req, urlObj), res);
    }
    if (pathname === "/api/contents") {
      return sendContentsListing(res, urlObj.searchParams.get("path") || "");
    }

    // Accept: text/markdown negotiation for key pages
    if (CLEAN_HTML[pathname] && wantsMarkdown(req)) {
      urlObj.searchParams.set("path", pathname);
      return markdownHandler(wrapReq(req, urlObj), res);
    }

    // Clean URL -> HTML
    if (CLEAN_HTML[pathname]) {
      return sendFile(res, path.join(SITE, CLEAN_HTML[pathname]), method);
    }

    // Extra curriculum roots (phases/, etc.)
    const top = pathname.split("/").filter(Boolean)[0];
    if (EXTRA_ROOTS.includes(top)) {
      const rel = pathname.replace(/^\//, "");
      if (top === "i18n") {
        return sendI18n(res, rel, method);
      }
      const filePath = safeJoin(ROOT, rel);
      if (!filePath) {
        res.statusCode = 403;
        return res.end("Forbidden");
      }
      return sendFile(res, filePath, method);
    }

    // Static site assets
    const rel = pathname === "/" ? "index.html" : pathname.replace(/^\//, "");
    const filePath = safeJoin(SITE, rel);
    if (!filePath) {
      res.statusCode = 403;
      return res.end("Forbidden");
    }
    return sendFile(res, filePath, method);
  } catch (err) {
    console.error(err);
    res.statusCode = 500;
    res.end("Internal Server Error");
  }
});

server.listen(PORT, HOST, () => {
  console.log(`AI Engineering Learn listening on http://${HOST}:${PORT}`);
  console.log(`Serving ${SITE} (GitHub-free runtime)`);
});
