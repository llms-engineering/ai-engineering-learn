#!/usr/bin/env node
/**
 * 从零造 AI — static UI + local lesson API.
 * Reads curriculum from CONTENT_ROOT (default: sibling ai-engineering-learn).
 * No GitHub runtime dependency.
 */
"use strict";

const http = require("http");
const fs = require("fs");
const path = require("path");
const { URL } = require("url");

const ROOT = __dirname;
const PUBLIC = path.join(ROOT, "public");
const CONTENT_ROOT = process.env.CONTENT_ROOT || "/root/code/ai-engineering-learn";
const PORT = Number(process.env.PORT || 80);
const HOST = process.env.HOST || "0.0.0.0";

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".webp": "image/webp",
  ".ico": "image/x-icon",
  ".md": "text/markdown; charset=utf-8",
  ".woff2": "font/woff2",
  ".map": "application/json; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
};

const CLEAN = {
  "/": "index.html",
  "/catalog": "catalog.html",
  "/paths": "paths.html",
  "/progress": "progress.html",
  "/about": "about.html",
  "/lesson": "lesson.html",
};

function send(res, status, type, body) {
  res.writeHead(status, {
    "Content-Type": type,
    "Cache-Control": status === 200 ? "public, max-age=60" : "no-store",
  });
  res.end(body);
}

function sendJson(res, status, obj) {
  send(res, status, "application/json; charset=utf-8", JSON.stringify(obj));
}

function safeResolve(base, rel) {
  const resolved = path.resolve(base, rel);
  if (!resolved.startsWith(path.resolve(base) + path.sep) && resolved !== path.resolve(base)) {
    return null;
  }
  return resolved;
}

function normalizeLessonPath(raw) {
  if (!raw || typeof raw !== "string") return null;
  let p = raw.trim().replace(/^\/+/, "").replace(/\\/g, "/");
  // Accept phases/00-.../01-... or 00-.../01-...
  if (!p.startsWith("phases/")) {
    if (/^\d{2}-/.test(p)) p = "phases/" + p;
    else return null;
  }
  // Strip trailing /docs/... if present
  p = p.replace(/\/docs\/.*$/, "");
  const parts = p.split("/").filter(Boolean);
  if (parts.length !== 3) return null;
  if (parts[0] !== "phases") return null;
  if (!/^\d{2}-[\w-]+$/.test(parts[1])) return null;
  if (!/^\d{2}-[\w-]+$/.test(parts[2])) return null;
  return parts.join("/");
}

function readLessonMarkdown(rel) {
  const zh = path.join(CONTENT_ROOT, "i18n/zh", rel, "docs/zh.md");
  const en = path.join(CONTENT_ROOT, rel, "docs/en.md");
  if (fs.existsSync(zh)) {
    return { lang: "zh", markdown: fs.readFileSync(zh, "utf8"), source: "i18n/zh/" + rel + "/docs/zh.md" };
  }
  if (fs.existsSync(en)) {
    return { lang: "en", markdown: fs.readFileSync(en, "utf8"), source: rel + "/docs/en.md" };
  }
  return null;
}

function serveStatic(reqPath, res) {
  let fileRel = CLEAN[reqPath] || reqPath;
  if (fileRel.includes("..")) {
    send(res, 400, "text/plain", "Bad request");
    return;
  }
  if (fileRel.startsWith("/")) fileRel = fileRel.slice(1);
  if (!fileRel) fileRel = "index.html";

  const filePath = safeResolve(PUBLIC, fileRel);
  if (!filePath || !fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
    // SPA-ish fallback for unknown: 404 page as plain
    const notFound = path.join(PUBLIC, "404.html");
    if (fs.existsSync(notFound)) {
      send(res, 404, "text/html; charset=utf-8", fs.readFileSync(notFound));
    } else {
      send(res, 404, "text/plain; charset=utf-8", "Not found");
    }
    return;
  }
  const ext = path.extname(filePath).toLowerCase();
  const type = MIME[ext] || "application/octet-stream";
  const cache = ext === ".html" ? "public, max-age=30" : "public, max-age=3600";
  res.writeHead(200, { "Content-Type": type, "Cache-Control": cache });
  fs.createReadStream(filePath).pipe(res);
}

function handleApi(urlObj, res) {
  const apiPath = urlObj.pathname;

  if (apiPath === "/api/health") {
    sendJson(res, 200, {
      ok: true,
      brand: "从零造 AI",
      contentRoot: CONTENT_ROOT,
      contentExists: fs.existsSync(path.join(CONTENT_ROOT, "phases")),
    });
    return;
  }

  if (apiPath === "/api/catalog") {
    const catalogPath = path.join(PUBLIC, "data", "catalog.json");
    if (!fs.existsSync(catalogPath)) {
      sendJson(res, 503, { error: "catalog_missing", hint: "Run scripts/build-catalog.js" });
      return;
    }
    send(res, 200, "application/json; charset=utf-8", fs.readFileSync(catalogPath));
    return;
  }

  if (apiPath === "/api/lesson") {
    const rel = normalizeLessonPath(urlObj.searchParams.get("path") || "");
    if (!rel) {
      sendJson(res, 400, { error: "invalid_path" });
      return;
    }
    const data = readLessonMarkdown(rel);
    if (!data) {
      sendJson(res, 404, { error: "lesson_not_found", path: rel });
      return;
    }
    // Parse phase/lesson for navigation helpers
    const parts = rel.split("/");
    const phaseSlug = parts[1];
    const lessonSlug = parts[2];
    const phaseId = parseInt(phaseSlug.slice(0, 2), 10);
    const lessonNum = lessonSlug.slice(0, 2);

    let quiz = null;
    const quizPath = path.join(CONTENT_ROOT, rel, "quiz.json");
    if (fs.existsSync(quizPath)) {
      try { quiz = JSON.parse(fs.readFileSync(quizPath, "utf8")); } catch (_) {}
    }

    sendJson(res, 200, {
      path: rel,
      phaseId,
      phaseSlug,
      lessonSlug,
      lessonNum,
      lang: data.lang,
      source: data.source,
      markdown: data.markdown,
      hasQuiz: !!quiz,
      quiz: quiz,
    });
    return;
  }

  sendJson(res, 404, { error: "not_found" });
}

const server = http.createServer((req, res) => {
  try {
    const urlObj = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);
    if (urlObj.pathname.startsWith("/api/")) {
      handleApi(urlObj, res);
      return;
    }
    // Clean URLs
    let p = urlObj.pathname;
    if (p.length > 1 && p.endsWith("/")) p = p.slice(0, -1);
    serveStatic(p, res);
  } catch (err) {
    console.error(err);
    send(res, 500, "text/plain; charset=utf-8", "Internal error");
  }
});

server.listen(PORT, HOST, () => {
  console.log(`从零造 AI listening on http://${HOST}:${PORT}`);
  console.log(`CONTENT_ROOT=${CONTENT_ROOT}`);
  console.log(`PUBLIC=${PUBLIC}`);
});
