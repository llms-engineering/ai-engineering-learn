#!/usr/bin/env node
/**
 * Build catalog.json from local phases/ + i18n/zh/ trees.
 * No network. CONTENT_ROOT defaults to sibling ai-engineering-learn.
 */
"use strict";

const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const CONTENT_ROOT = process.env.CONTENT_ROOT
  || path.resolve(ROOT, "../ai-engineering-learn")
  || "/root/code/ai-engineering-learn";
const OUT = path.join(ROOT, "public", "data", "catalog.json");

const PHASE_META = {
  0: { zh: "环境与工程工具链", en: "Setup & Tooling", short: "环境" },
  1: { zh: "数学基础", en: "Math Foundations", short: "数学" },
  2: { zh: "机器学习基础", en: "ML Fundamentals", short: "ML" },
  3: { zh: "深度学习核心", en: "Deep Learning Core", short: "深度学习" },
  4: { zh: "计算机视觉", en: "Computer Vision", short: "视觉" },
  5: { zh: "NLP 基础到进阶", en: "NLP Foundations to Advanced", short: "NLP" },
  6: { zh: "语音与音频", en: "Speech & Audio", short: "语音" },
  7: { zh: "Transformer 深潜", en: "Transformers Deep Dive", short: "Transformer" },
  8: { zh: "生成式 AI", en: "Generative AI", short: "生成式" },
  9: { zh: "强化学习", en: "Reinforcement Learning", short: "强化学习" },
  10: { zh: "手写 LLM", en: "LLMs from Scratch", short: "手写 LLM" },
  11: { zh: "LLM 应用工程", en: "LLM Engineering", short: "LLM 工程" },
  12: { zh: "多模态 AI", en: "Multimodal AI", short: "多模态" },
  13: { zh: "工具与协议", en: "Tools & Protocols", short: "工具协议" },
  14: { zh: "Agent 工程", en: "Agent Engineering", short: "Agent" },
  15: { zh: "自主系统", en: "Autonomous Systems", short: "自主系统" },
  16: { zh: "多智能体与蜂群", en: "Multi-Agent & Swarms", short: "多智能体" },
  17: { zh: "基础设施与生产", en: "Infrastructure & Production", short: "生产" },
  18: { zh: "伦理·安全·对齐", en: "Ethics, Safety & Alignment", short: "对齐" },
  19: { zh: "综合实战项目", en: "Capstone Projects", short: "实战" },
};

const SKU_A_MAX = 3; // Phase 0–3 fully learnable

function readTitle(mdPath) {
  try {
    const text = fs.readFileSync(mdPath, "utf8");
    const m = text.match(/^#\s+(.+)$/m);
    if (m) return m[1].trim().replace(/\s+/g, " ");
  } catch (_) {}
  return null;
}

function slugToTitle(slug) {
  // 01-dev-environment -> Dev Environment
  const parts = slug.split("-").slice(1);
  return parts.map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
}

function extractTime(mdPath) {
  try {
    const text = fs.readFileSync(mdPath, "utf8").slice(0, 2000);
    const m = text.match(/\*\*Time:\*\*\s*~?\s*([^\n*]+)/i)
      || text.match(/建议时长[：:]\s*约?\s*([^\n]+)/);
    if (m) return m[1].trim();
  } catch (_) {}
  return null;
}

function build() {
  const phasesDir = path.join(CONTENT_ROOT, "phases");
  if (!fs.existsSync(phasesDir)) {
    console.error("CONTENT_ROOT phases missing:", phasesDir);
    process.exit(1);
  }

  const phaseDirs = fs.readdirSync(phasesDir)
    .filter((d) => /^\d{2}-/.test(d) && fs.statSync(path.join(phasesDir, d)).isDirectory())
    .sort();

  const phases = [];
  let totalLessons = 0;
  let totalZh = 0;

  for (const phaseSlug of phaseDirs) {
    const id = parseInt(phaseSlug.slice(0, 2), 10);
    const meta = PHASE_META[id] || { zh: phaseSlug, en: phaseSlug, short: phaseSlug };
    const pdir = path.join(phasesDir, phaseSlug);
    const lessonDirs = fs.readdirSync(pdir)
      .filter((d) => /^\d{2}-/.test(d) && fs.statSync(path.join(pdir, d)).isDirectory())
      .sort();

    const lessons = [];
    for (const lessonSlug of lessonDirs) {
      const num = lessonSlug.slice(0, 2);
      const rel = `phases/${phaseSlug}/${lessonSlug}`;
      const zhPath = path.join(CONTENT_ROOT, "i18n/zh", rel, "docs/zh.md");
      const enPath = path.join(pdir, lessonSlug, "docs/en.md");
      const hasZh = fs.existsSync(zhPath);
      const hasEn = fs.existsSync(enPath);
      const titleZh = hasZh ? readTitle(zhPath) : null;
      const titleEn = hasEn ? readTitle(enPath) : slugToTitle(lessonSlug);
      const time = extractTime(hasZh ? zhPath : enPath);
      const open = id <= SKU_A_MAX || hasZh; // link if SKU-A or zh exists
      lessons.push({
        id: `${id}:${num}`,
        num,
        slug: lessonSlug,
        path: rel,
        title: titleZh || titleEn || slugToTitle(lessonSlug),
        titleEn: titleEn || slugToTitle(lessonSlug),
        hasZh,
        hasEn,
        open,
        time: time || "~45m",
      });
      totalLessons++;
      if (hasZh) totalZh++;
    }

    const skuOpen = id <= SKU_A_MAX;
    let badge = "可学";
    if (!skuOpen) {
      const allZh = lessons.every((l) => l.hasZh);
      badge = allZh ? "可预览" : "翻译中";
      // Design: non-SKU-A marked 即将上线/翻译中
      if (!allZh) badge = "翻译中";
      else badge = "即将上线";
    }

    phases.push({
      id,
      slug: phaseSlug,
      name: meta.zh,
      nameEn: meta.en,
      short: meta.short,
      lessonCount: lessons.length,
      skuOpen,
      badge,
      lessons,
    });
  }

  const paths = [
    {
      slug: "foundations",
      name: "基础营",
      desc: "环境 → 数学直觉 → 经典 ML → 手写神经网络",
      meta: "Phase 0–3 · 入门推荐",
      phases: [0, 1, 2, 3],
      open: true,
      recommended: true,
    },
    {
      slug: "transformer",
      name: "手写到 Transformer",
      desc: "NLP 基础到注意力机制与完整 Transformer",
      meta: "精选 Phase 5 · 7",
      phases: [5, 7],
      open: false,
      recommended: false,
    },
    {
      slug: "llm-app",
      name: "LLM 应用工程",
      desc: "Prompt · RAG · 评测 · 护栏 · 成本",
      meta: "Phase 11 主轴",
      phases: [11],
      open: false,
      recommended: false,
    },
    {
      slug: "agent-mcp",
      name: "Agent 与 MCP",
      desc: "工具协议、编排、记忆与安全底线",
      meta: "精选 Phase 13–14",
      phases: [13, 14],
      open: false,
      recommended: false,
    },
  ];

  // Attach lesson counts to paths
  for (const p of paths) {
    p.lessonCount = phases
      .filter((ph) => p.phases.includes(ph.id))
      .reduce((n, ph) => n + ph.lessonCount, 0);
  }

  const catalog = {
    generatedAt: new Date().toISOString(),
    brand: "从零造 AI",
    totalLessons,
    totalZh,
    skuAMax: SKU_A_MAX,
    phases,
    paths,
    contentRoot: CONTENT_ROOT,
  };

  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, JSON.stringify(catalog, null, 2));
  console.log(`Wrote ${OUT}`);
  console.log(`Phases: ${phases.length}, lessons: ${totalLessons}, zh: ${totalZh}`);
}

build();
