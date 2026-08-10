# Checkpoints

## Checkpoint — 2026-08-11T01:30:39+08:00

### Root Problem
在不迁移 Markdown、不改变既有 URL、内容集合与文档能力的前提下，将 AIInfraGuide 的全站前端收敛为克制、适合中文长文阅读的 Editorial Docs 主题，并保留可靠回滚路径。

### Current Round Goal
冻结并提交用户已认可的 Editorial 视觉 PoC、主题调研与验证证据，作为后续全站优化的可恢复基线；本轮不开始扩大全站覆盖。

### Completed
- [x] 在 `docs/FRONTEND_THEME_RESEARCH.md` 完成主题候选、设计 token、迁移阶段、视觉验收与回滚门禁调研。
- [x] 在现有 Astro 4 / Tailwind 3 架构内实现 `classic` / `editorial` 双主题切换，不移动 Markdown，不修改现有路由。
- [x] 将首页、长文阅读壳、课程侧栏、右侧 TOC、搜索及代码样式纳入 Editorial PoC，同时保留 Pagefind、KaTeX、Mermaid 和现有导航能力。
- [x] 产出桌面首页、PyTorch 4.8、PyTorch 4.12、移动首页和 Editorial Dark 共 5 张视觉证据，位于 `artifacts/theme-preview/`。
- [x] 运行 `npm run build`：Astro check 与生产构建完成，484 页生成；首页到 PyTorch 4.12 导航检查通过；Pagefind 索引 484 页、19,898 词。

### Next Steps
- [ ] 建立 484 个 URL、页面标题、canonical 与状态基线，并固定代表性视觉回归页和中文搜索查询。
- [ ] 将 Editorial token 与组件表现扩展到模块 landing、Paper、Interview、About、404、表格、公式、Mermaid 和代码等全站页面类型。
- [ ] 在 360、768、1280、1536 宽度及亮色、暗色、系统主题下完成交互、溢出、键盘与 200% zoom 验收。
- [ ] 复核 URL、内部链接、Pagefind base、中文 slug、排序、prev/next、KaTeX、Mermaid 与代码交互，任何门禁失败即停止扩面。
- [ ] 全站验收通过后再决定是否将 Editorial 设为默认主题；在此之前保留 Classic 回滚路径且不升级 Astro/Starlight。

### Key Decisions & Belief States

- **Claim**: 现有架构内的 Editorial Docs 重设计是当前最小风险方案，优于直接迁移第三方 starter。
- **Confidence**: high — 调研显示该路径可复用现有 URL、collections、侧栏、搜索与 Markdown pipeline；PoC 已在现有栈构建成功。
- **Strongest alternative**: Astro/Starlight 或第三方文档主题可用更少自定义代码提供更一致的全站文档壳。
- **Falsification condition**: 若全站扩面暴露出当前架构无法以合理复杂度满足统一 IA、可访问性或维护性门禁，并且小范围 Starlight 壳在相同 URL/内容约束下通过完整回归，则重新评估平台迁移。

- **Claim**: 用户认可当前 Editorial PoC 的视觉方向，可将其作为全站优化基线。
- **Confidence**: high — 用户明确反馈“这个方案不错”并要求在提交当前状态后设计全站优化 goal。
- **Strongest alternative**: 用户认可的只是三个代表页面的局部气质，其他页面类型需要不同的信息密度或组件表达。
- **Falsification condition**: 代表性模块 landing、Paper、Interview 或复杂内容页在同一 token 下出现明显可读性、信息架构或交互退化。

- **Claim**: 双主题切换足以支撑当前阶段的安全回滚。
- **Confidence**: medium — `?theme=editorial|classic` 与本地持久化已实现且代表页面可见，但尚未完成全站浏览器矩阵与可访问性回归。
- **Strongest alternative**: CSS 覆盖范围扩大后会产生 classic/editorial 串扰，单纯依赖 data attribute 不再易于维护。
- **Falsification condition**: 任一固定代表页无法稳定恢复 Classic，或主题切换导致首屏闪烁、低对比、状态错乱、布局或核心功能失效。

### Superseded Entries
无；这是仓库内首个 checkpoint。

### Blockers
- [ ] 无当前 blocker；全站视觉方向已由用户确认，后续仍需以 URL、内容能力、可访问性和跨页面视觉门禁约束扩面。

### Current Branch
`main`

### Files of Interest
- `docs/FRONTEND_THEME_RESEARCH.md` — 主题选择、设计原则、阶段门禁与回滚策略的调研真源。
- `src/styles/global.css` — Editorial 设计 token 与当前视觉覆盖的主要实现。
- `src/layouts/Layout.astro` — 全站壳、主题初始化和既有导航能力的接入点。
- `src/components/SiteThemePreview.astro` — Classic / Editorial 预览切换入口。
- `src/pages/index.astro` — Editorial 首页 PoC 与首页信息架构。
- `src/components/GuideContent.astro` — 长文主内容壳及元信息布局。
- `src/components/GuideSidebar.astro` — 左侧课程树。
- `src/components/TableOfContents.astro` — 右侧页内目录。
- `src/components/Search.astro` — Pagefind 搜索交互。
- `artifacts/theme-preview/` — 5 张桌面、移动、亮色与暗色视觉基线截图。

---
