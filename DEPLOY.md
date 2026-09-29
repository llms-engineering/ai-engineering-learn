# Self-host deployment

This instance is a self-hosted copy of the MIT-licensed curriculum
[AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
by [Rohit Ghumare](https://github.com/rohitg00) and contributors.

- Upstream: https://github.com/rohitg00/ai-engineering-from-scratch
- Canonical site: https://aiengineeringfromscratch.com
- License: MIT (see `LICENSE`)

## Runtime is GitHub-free

**Serving this site does not call GitHub** (`raw.githubusercontent.com`,
`api.github.com`, or `github.com` HTTP APIs) for lessons, i18n UI strings,
translated markdown, code/output directory listings, or curriculum assets.

All of that is read from local disk under this repository:

| Path | Purpose |
|------|---------|
| `site/` | Built static UI |
| `phases/`, `certifications/`, … | English curriculum source |
| `i18n/<lang>/` | Vendored UI dictionaries + translated lesson markdown |
| `deploy/server.js` | Static + API server (local files only) |

Updates are **deploy-time only**: `git pull` / re-vendor translations / rebuild,
then restart the service. There is no cache-miss proxy to upstream at runtime.

## How this server is run

```bash
# Build generated site assets
node site/build.js

# Serve site/ + /lesson + /certification APIs
PORT=80 node deploy/server.js
# or: systemctl start ai-engineering-learn
```

Service unit: `/etc/systemd/system/ai-engineering-learn.service`

## Internationalization (vendored)

Switcher languages (see `site/langs.js` / `languages.json`) ship with full trees
under `i18n/` (at minimum complete `zh`, plus `hi` `es` `ar` `fr` `pt` `tr` `vi`).

Client code sets `window.__AIFS_SELF_HOST = true` and loads:

- `/i18n/<lang>/ui.json`
- `/i18n/<lang>/phases/.../docs/<lang>.md`
- `/api/contents?path=...` for lesson `code/` / `outputs/` listings

### Refreshing translations (build / deploy time)

Upstream translations live on the `translations` branch. Re-vendor once, then
restart — do **not** rely on live GitHub fetches while serving:

```bash
git clone --depth 1 --branch translations --single-branch \
  https://github.com/rohitg00/ai-engineering-from-scratch.git /tmp/aifs-translations
rsync -a --delete /tmp/aifs-translations/i18n/ ./i18n/
systemctl restart ai-engineering-learn
```

## Residual non-critical remotes

Optional / secondary third-party loads (not required for lessons or language
switch) may still appear in HTML as ordinary links or CDNs, e.g. sponsor badge
images, fonts, or outbound “open on GitHub” hyperlinks. Those are not used by
the Node server when handling page/lesson/i18n requests.
