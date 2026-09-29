# 从零造 AI · 部署说明

产品名：**从零造 AI**  
公共地址：`http://43.153.0.177/`  
内容来源：同机 `/root/code/ai-engineering-learn`（`phases/` + `i18n/zh/`）  
**运行时不访问 GitHub。**

## 架构

```
/root/code/cong-ling-zao-ai/     ← 本站 UI + server.js
/root/code/ai-engineering-learn/ ← 课程 markdown / quiz（只读）
```

- `server.js` 提供静态页 + `/api/catalog` + `/api/lesson?path=...`
- 课文优先 `i18n/zh/.../docs/zh.md`，否则回退 `phases/.../docs/en.md`
- 进度仅浏览器 `localStorage`（键 `clzai_progress_v1`）

## 首次 / 更新部署

```bash
# 1. 同步代码到服务器
rsync -avz --delete ./ /root/code/cong-ling-zao-ai/ \
  --exclude node_modules --exclude .git

# 2. 生成目录数据（读 CONTENT_ROOT）
cd /root/code/cong-ling-zao-ai
CONTENT_ROOT=/root/code/ai-engineering-learn node scripts/build-catalog.js

# 3. systemd（见下方 unit）
systemctl daemon-reload
systemctl restart cong-ling-zao-ai
systemctl status cong-ling-zao-ai --no-pager
```

## systemd unit

`/etc/systemd/system/cong-ling-zao-ai.service`：

```ini
[Unit]
Description=从零造 AI Chinese learning site
After=network.target

[Service]
Type=simple
WorkingDirectory=/root/code/cong-ling-zao-ai
Environment=PORT=80
Environment=HOST=0.0.0.0
Environment=NODE_ENV=production
Environment=CONTENT_ROOT=/root/code/ai-engineering-learn
ExecStart=/usr/bin/node /root/code/cong-ling-zao-ai/server.js
Restart=on-failure
RestartSec=3
MemoryMax=512M

[Install]
WantedBy=multi-user.target
```

停用旧服务（若仍占用 80）：

```bash
systemctl disable --now ai-engineering-learn
systemctl enable --now cong-ling-zao-ai
```

## 验收

- 首页标题含「从零造 AI」
- `/catalog` 列出 Phase 0–19；0–3 标「可学」
- `/lesson.html?path=phases/00-setup-and-tooling/01-dev-environment` 显示中文正文
- 课页「标记完成」写入 localStorage
- 页脚含 MIT 上游署名与「非官方中文版」

## 工作区副本

开发稿与源码也在：`/workspace/paid-tutorial/cong-ling-zao-ai/`
