# Self-host deployment

This instance is a self-hosted copy of the MIT-licensed curriculum
[AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
by [Rohit Ghumare](https://github.com/rohitg00) and contributors.

- Upstream: https://github.com/rohitg00/ai-engineering-from-scratch
- Canonical site: https://aiengineeringfromscratch.com
- License: MIT (see `LICENSE`)

## How this server is run

```bash
# Build generated site assets
node site/build.js

# Serve site/ + /lesson + /certification APIs
PORT=80 node deploy/server.js
# or: systemctl start ai-engineering-learn
```

Service unit: `/etc/systemd/system/ai-engineering-learn.service`

Lesson page HTML is assembled by `api/lesson.js`. Lesson markdown bodies are
fetched by the browser from the public GitHub raw content (same as the
production site) unless you browse via localhost.


## Internationalization (self-host)

Upstream loads UI strings and translated lesson markdown from the GitHub
`translations` branch via `raw.githubusercontent.com`. That often fails for
browsers in China, so this fork:

1. Sets `window.__AIFS_SELF_HOST = true` (in `site/header.js` / lesson pages).
2. Fetches `/i18n/<lang>/ui.json` and `/i18n/<lang>/phases/.../docs/<lang>.md`
   from this origin instead.
3. `deploy/server.js` serves local `i18n/` files and, on cache miss, pulls from
   the upstream `translations` branch and caches them under `i18n/`.

Optional: pre-seed UI dictionaries:

```bash
for lang in zh hi es ar fr pt tr vi; do
  mkdir -p i18n/$lang
  curl -fsSL \
    "https://raw.githubusercontent.com/rohitg00/ai-engineering-from-scratch/translations/i18n/$lang/ui.json" \
    -o "i18n/$lang/ui.json"
done
systemctl restart ai-engineering-learn
```
