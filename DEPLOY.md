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
