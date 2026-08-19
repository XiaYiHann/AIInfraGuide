---
name: aiinfraguide-deploy
description: "把 AIInfraGuide 仓库的内容发布到用户的 GitHub Pages 站点（xiayihann.github.io/AIInfraGuide）。当用户说 push、推送、发布、上线、部署、推到 github.io、更新站点、网站没更新、看不到新文章 时使用，也适用于在 docs/ 下写完文章或修复后需要用户在线上看到的任何场景。覆盖完整流程：本地 build 检查 → 提交 → 推送到 XiaYiHann fork → 手动触发 Pages 部署（fork 不会自动触发 workflow）→ 验证线上页面。即使用户只说了'push'，也必须走完验证一步——只 push 不部署等于没上线。"
---

# AIInfraGuide 发布到 GitHub Pages

## 为什么需要这个 skill（固定事实，2026-08-19 验证）

本仓库是 **fork**（`XiaYiHann/AIInfraGuide` ← 上游 `caomaolufei/AIInfraGuide`）。GitHub 对 fork 的安全策略：
**push 到 fork 不会自动触发 Actions workflow**——所有 Pages 部署都必须是 `workflow_dispatch`（手动触发）。
历史上 48 个 push 只有 0 个自动部署，线上长期停在旧内容。所以"push 了"≠"上线了"，必须手动触发部署并验证。

- 站点：`https://xiayihann.github.io/AIInfraGuide/`（由本 fork 的 Pages 提供，base 路径 `/AIInfraGuide`）
- 部署 workflow：`.github/workflows/deploy.yml`（npm ci + build + upload-pages + deploy-pages）
- 构建+部署耗时约 5~6 分钟

## 两个必踩的坑

1. **gh CLI 仓库自动检测会指错**：本机 gh 可能把当前目录解析成 fork 上游 `caomaolufei/AIInfraGuide`。
   **所有 `gh` 命令一律显式 `-R XiaYiHann/AIInfraGuide`**（`gh run list`、`gh workflow run`、`gh run view` 全部）。
2. **`gh workflow run` 的仓库参数必须用 `-R`**：`gh workflow run deploy.yml -R XiaYiHann/AIInfraGuide`
   （把仓库名当位置参数会报 `requires a value separated by an '=' sign`）。

## 流程

### 0. 推送前检查（防推错仓库）

```bash
git remote get-url origin   # 必须是 git@github.com:XiaYiHann/AIInfraGuide.git
git status --short          # 看清楚要提交什么；无关改动（favicon 之类）单独处理，不混进内容提交
```

### 1. 本地 build（必须先过）

```bash
npm run build 2>&1 | grep -iE "error|Unrecognized" | head
```

build 报错或 KaTeX `Unrecognized Unicode` 警告（中文 `×` 等裸字符进了公式）先修掉再推。

### 2. 提交

提交信息用仓库既有风格（中文，前缀 `guides:` / `papers:` / `style:` / `docs:`），正文写清改了什么：

```
guides: 5.7 DFlash 补 DFlash 2（Inco AI 博客 2026-08）——新增第7节……；5.8 加对照注；更新日志条目
```

### 3. 推送

```bash
git push origin main
```

### 4. 触发部署

`.git/hooks/post-push` 正常情况下会自动触发。**必须验证它真的触发了**——看最新 run 的 headSha 是不是刚推的 commit：

```bash
gh run list -R XiaYiHann/AIInfraGuide --limit 1 --json headSha,event,createdAt,status
```

- 有新的 `workflow_dispatch` 且 SHA 匹配 → 跳到第 5 步；
- 没有（hook 没跑/gh 失败）→ 手动触发：

```bash
gh workflow run deploy.yml -R XiaYiHann/AIInfraGuide
```

### 5. 等待并验证（跳过这步 = 没上线）

```bash
gh run watch <run-id> -R XiaYiHann/AIInfraGuide   # 约 5~6 分钟，等到 completed/success
```

然后验证线上（中文路径必须整体 percent-encode，直接 curl 中文 URL 会 400/404）：

```bash
bash scripts/check_live.sh "/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/57-dflash-块扩散并行起草" "DFlash 2"
```

`check_live.sh` 两个参数：站点内路径（以 `/AIInfraGuide` 开头）、可选的正文签名串（grep 验证新内容真的上线了）。
退出码 0 = 200 且（若给了签名串）签名串出现；非 0 = 没上线。

## 验收标准

- `git ls-remote origin refs/heads/main` 的 SHA == 本地 HEAD（代码到了 GitHub）
- 最新 Pages run `completed/success` 且 headSha 是该 SHA（构建部署完成）
- 线上变更页 200 + 新内容签名串可见（**用户真正看到的**）

三步都过才算"发布完成"，回复用户时只报告这个结论，不报告中间的猜测。

## 排障

- **hook 没触发**：手动跑 `gh workflow run deploy.yml -R XiaYiHann/AIInfraGuide` 兜底；查 hook 是否存在/可执行（`ls -la .git/hooks/post-push`）。hook 只在它所在的那个 clone 生效，换机器 push 必须手动触发。
- **线上还是旧内容但 run 成功了**：等 1~2 分钟 CDN 刷新再验；确认验证的 URL 编码正确（用 check_live.sh，别手拼）。
- **run 一直 in_progress 超过 10 分钟**：`gh run view <id> -R XiaYiHann/AIInfraGuide --log-failed` 看日志，常见是 npm ci 拉包慢，再等即可。
