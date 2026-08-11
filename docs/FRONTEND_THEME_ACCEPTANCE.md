# AIInfraGuide Editorial Docs 全站验收

## 1. 验收结论

Editorial Docs 已成为站点默认视觉主题，Classic 仍作为可逆回滚路径保留。本次变更只重构布局语义、主题样式与视觉回归工具，没有迁移 Markdown、content collections 或 URL。

验收状态：**通过**。

- 基线提交：`d54d2b95ec76461df4f595b10ea7c40c53b9f7cf`（`feat: add editorial theme preview`）
- URL 基线：`artifacts/theme-full-site/url-baseline.json`
- 浏览器证据：`artifacts/theme-full-site/browser/browser-check.json`
- 浏览器截图：`artifacts/theme-full-site/browser/`
- 默认主题：`editorial`
- 回滚主题：`classic`

## 2. 页面类型覆盖矩阵

| 页面类型 | 代表 URL | 主要验收点 |
|---|---|---|
| 首页 | `/AIInfraGuide/` | Editorial 首页、站点定位、课程目录、导航 |
| 模块 Landing | `/AIInfraGuide/prerequisites/` | 左侧课程树、章节目录、主操作 |
| PyTorch 4.8 | `/AIInfraGuide/prerequisites/模块一-前置知识/pytorch/48-benchmark与profiler/` | 三栏长文、代码、表格、桌面 TOC |
| PyTorch 4.12 | `/AIInfraGuide/prerequisites/模块一-前置知识/pytorch/412-minigpt综合项目/` | 长文、移动 TOC、宽表格、长代码行 |
| Interview 首页 | `/AIInfraGuide/interview/` | 梯队目录、公司分组、左侧索引 |
| Interview 文章 | `/AIInfraGuide/interview/美团-ai-infra-一面/` | 文章阅读面、侧栏、TOC、分页 |
| Paper 首页 | `/AIInfraGuide/papers/` | 三列精读目录、元信息层级 |
| Paper 精读 | `/AIInfraGuide/papers/kimi-k3-notes/` | 长标题、来源信息、正文、移动布局 |
| Blog 首页 | `/AIInfraGuide/blog/` | 博客目录和卡片语义 |
| Blog 文章 | `/AIInfraGuide/blog/update-log/` | 独立文章、长英文标识符断行 |
| About | `/AIInfraGuide/about/` | Ethan 生成式标识、AI Researcher 身份与 fork 链接 |
| Roadmap | `/AIInfraGuide/about/roadmap/` | 独立信息页、模块目录和 callout |
| 404 | `/AIInfraGuide/404.html` | 错误信息、返回首页和移动页脚 |

每个页面类型均在以下四种模式保存截图并运行 DOM 门禁：

1. `light-1440`：1440 × 1100，Light
2. `dark-1440`：1440 × 1100，Dark
3. `light-390`：390 × 844，Light
4. `dark-390`：390 × 844，Dark

共生成 52 张页面截图，加上搜索、移动菜单和移动 TOC 3 张交互截图，总计 55 张。

## 3. 自动化验证

### 3.1 构建、导航与搜索

执行：

```bash
npm run build
```

结果：

- Astro check：`0 errors`、`0 warnings`；已有生成文件与未使用变量共报告 `304 hints`
- Astro build：生成 `484 page(s)`
- 首页导航门禁：`首页 → PyTorch 第4章 → 4.12` 路径完整
- Pagefind：索引 `484 pages`、`19,886 words`、语言 `zh-cn`

### 3.2 精确 URL 回归

执行：

```bash
npm run check:theme-regression
```

结果：当前 `dist` 的 484 个 HTML URL 与提交 `d54d2b9` 建立的基线完全一致；缺失页面 `0`，新增页面 `0`。

### 3.3 浏览器运行与视觉回归

先启动预览：

```bash
npm run preview -- --host 0.0.0.0
```

再执行：

```bash
npm run check:theme-browser
```

结果：`13 page types × 4 modes` 全部通过。每个页面均验证：

- `data-site-theme="editorial"` 为默认主题
- Light/Dark 与测试模式一致
- `document.scrollWidth <= document.clientWidth`
- 无重复 DOM `id`
- 全站只有一个阅读进度条和一个返回顶部按钮
- 无 `error`、`unhandledrejection` 或 `console.error`

交互结果：

| 交互 | 结果 |
|---|---|
| `?theme=classic` | 页面与 `localStorage` 均切换为 Classic |
| 页头主题按钮 | Classic → Editorial，状态同步写入 `localStorage` |
| 刷新持久化 | Editorial 状态保留 |
| `?theme=editorial` | 页面与 `localStorage` 均切换为 Editorial |
| 搜索 | 弹窗打开、Pagefind 输入框就绪、背景滚动锁定 |
| 移动菜单 | 抽屉与遮罩打开、背景滚动锁定 |
| 移动 TOC | 目录可展开，长目录保持在正文宽度内 |

### 3.4 失败路径与修复证据

首轮浏览器矩阵发现 PyTorch 4.8、4.12 和 Blog 文章在 390px 下存在横向溢出，明暗模式共 6 个失败。DOM 几何审计确认根因是长代码行、宽表格和连续英文标识符扩大文章容器的 `scrollWidth`；移动抽屉只是随文档宽度偏移，不是根因。

修复后：

- 代码块与表格在自身容器内横向滚动
- 普通长文本使用安全断行
- 正文外层阻止内部滚动传播为整页横向滚动
- 未使用根级 `overflow: hidden` 掩盖越界

定向 `3 page types × 1 mode` 回归通过，随后完整 `13 × 4` 矩阵通过。

## 4. 视觉与架构边界

已验证的视觉结果：

- 暖白/墨黑纸面、单一低饱和蓝绿色强调色
- 正文约 70ch、中文阅读优先的行高和标题层级
- 普通目录卡片无悬浮抬升、厚阴影或发光
- 容器使用小圆角和细边框
- Editorial 代码块取消红黄绿灯与厚阴影
- 导航和新增 UI 使用文字或线性 SVG；Classic 专属 emoji 在 Editorial 隐藏
- 首页、目录页、长文、集合页和独立信息页共享语义样式层
- 页脚以 Ethan 为主署名，仅用一行小字链接原始仓库；首页镜像提示和原作者个人链接已移除

保留的产品与技术行为：

- Astro 4、Tailwind 3 与现有四个 content collections
- 484 个页面 URL、别名、排序和 GitHub Pages base path
- 左侧课程树、桌面 TOC、移动 TOC、Pagefind、KaTeX、Mermaid、Shiki、代码复制、SEO 和部署方式
- `?theme=editorial`、`?theme=classic`、页头切换和 `localStorage` 持久化

未扩大 claim 的边界：

- 验收基于本地静态构建与 Chromium 150，不等同于生产部署验收
- 没有修改 Markdown 教程正文、课程结构、schema、搜索语义或 Mermaid/KaTeX 内容；About 与 Roadmap 的作者品牌文案按用户授权更新为 Ethan
- 没有新增运行时依赖、升级框架、下载字体/素材、提交、push 或发布本轮全站优化

## 5. 关键截图

- 首页：`artifacts/theme-full-site/browser/home-light-1440.png`
- PyTorch 4.8：`artifacts/theme-full-site/browser/guide-4-8-light-1440.png`
- PyTorch 4.12 移动暗色：`artifacts/theme-full-site/browser/guide-4-12-dark-390.png`
- Interview 首页：`artifacts/theme-full-site/browser/interview-index-light-1440.png`
- Paper 首页：`artifacts/theme-full-site/browser/paper-index-light-1440.png`
- Paper 精读移动暗色：`artifacts/theme-full-site/browser/paper-article-dark-390.png`
- About 移动：`artifacts/theme-full-site/browser/about-light-390.png`
- 搜索弹窗：`artifacts/theme-full-site/browser/interaction-search-light-1440.png`
- 移动菜单：`artifacts/theme-full-site/browser/interaction-mobile-menu-light-390.png`
- 移动 TOC：`artifacts/theme-full-site/browser/interaction-mobile-toc-light-390.png`
