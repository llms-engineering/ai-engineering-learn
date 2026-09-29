# Self-host deployment

## Current production UI: 从零造 AI

Public site: `http://43.153.0.177/`

The live UI is the Chinese learning product **从零造 AI**, served from:

- App: `/root/code/cong-ling-zao-ai` (systemd `cong-ling-zao-ai.service`)
- Curriculum (read-only): this repo (`phases/`, `i18n/zh/`, …)

See [`cong-ling-zao-ai/DEPLOY.md`](./cong-ling-zao-ai/DEPLOY.md) for cutover, catalog rebuild, and acceptance checks.

**Runtime is GitHub-free** for lessons: markdown is read from local disk only.

### Legacy English UI

The previous `site/` + `deploy/server.js` stack remains in-tree for reference but is **not** bound to port 80. Service `ai-engineering-learn` is disabled.

Upstream: https://github.com/rohitg00/ai-engineering-from-scratch (MIT).
