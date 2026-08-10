# AIInfraGuide 前端主题迁移调研

> 调研截止：**2026-08-10**
> 范围：Astro 主题与文档框架、跨框架对照、当前仓库适配风险、视觉收敛方案与 PoC。
> 结论先行：**不要把任何候选当成可“一键换皮”的依赖。最低风险、最高收益的方案是保留现有内容模型和 URL 路由，先在原 Astro 架构内做设计系统收敛；若后续确实需要标准文档能力，再升级 Astro 并以 Starlight 的页面壳逐步替换，而不是一次性搬迁全部内容。**

## 1. 明确推荐

### 推荐顺序

1. **首选：现有 Astro 架构内重设计，不换主题。**
   - 保留四个自定义 content collections、现有动态路由、章节分组逻辑、Pagefind、KaTeX、Mermaid、Shiki 和 GitHub Pages 发布链。
   - 只收敛视觉 token 与组件表现，移除高频“AI 模板感”装饰。
   - 这是唯一能把 484 个页面、固定 URL 和内容 schema 风险基本隔离在视觉层的方案。

2. **中期首选框架：Astro Starlight，但采用“页面壳渐进接入”，不是整站导入。**
   - Starlight 是候选中唯一成熟、持续维护、以技术文档信息架构为核心的方案；自带侧栏、页内 TOC、移动端、暗色、Pagefind、分页、可访问性与组件覆盖机制。[官方文档](https://starlight.astro.build/) [配置参考](https://starlight.astro.build/reference/configuration/) [自定义页面](https://starlight.astro.build/guides/pages/)
   - 最新 `@astrojs/starlight@0.41.7` 要求 Astro `^7.0.2`，不能直接装进当前 Astro 4.16 项目。[package.json](https://github.com/withastro/starlight/blob/main/packages/starlight/package.json) [0.41.7 release](https://github.com/withastro/starlight/releases/tag/%40astrojs%2Fstarlight%400.41.7)
   - PoC 应先让现有 `[...slug].astro` 路由继续生成 URL，并用 `<StarlightPage>` 包裹现有 collection 渲染结果；官方明确支持自定义 Astro 页面使用 Starlight 布局、传入 headings 和自定义 sidebar。[官方自定义页面文档](https://starlight.astro.build/guides/pages/#using-starlights-design-in-custom-pages)

3. **若采用 Starlight，视觉层优先比较 Starlight Black，Rapide 作为备选。**
   - Starlight Black 受 shadcn 文档站启发，黑白、中性、边界清晰，和“克制的专业技术文档”目标更接近；Rapide 是 Vitesse 风格，辨识度更强但配色也更明显。[Starlight Black 仓库](https://github.com/adrian-ub/starlight-theme-black) [Black demo](https://starlight-theme-black.vercel.app/) [Rapide 仓库](https://github.com/HiDeoo/starlight-theme-rapide)
   - 两者都只是 Starlight theme/plugin，不是独立文档引擎；必须在 Starlight 核心 PoC 通过后再比较，不能代替迁移方案。

4. **SupportGenix Docs 是最接近“一条命令安装”的候选，但仍不是本项目的一键迁移。**
   - 官方 CLI 可执行 `npm create supportgenix-docs@latest`，支持 `--dry-run`，并把文件放在 namespaced 目录；但要求 Node 22.12+、Astro 6+，而且明确不能自动合并 `astro.config.mjs` 与 content config。[CLI README](https://github.com/supportgenix/supportgenix-docs/blob/main/packages/create-supportgenix-docs/README.md)
   - 它会新建 `src/content/docs/` 和新的文档路由，并不识别本项目已有四个 collection、重复别名 URL 与章节排序。因此可用于参考“可回滚脚手架”的工程方式，不宜直接运行到当前主分支。

5. **Retypeset 仅作为“长文排版参考”，不建议直接迁入其代码。**
   - 它的纸书式排版、中文/多语言、LaTeX、Mermaid、TOC 很接近阅读目标，但本质是博客模板，使用 Astro 6 + UnoCSS，缺少本项目所需的课程级全局章节树。[README](https://github.com/radishzzz/astro-theme-retypeset/blob/master/README.md) [package.json](https://github.com/radishzzz/astro-theme-retypeset/blob/master/package.json)

6. **AstroPaper、Astro Cactus 不作为课程站主框架。**
   - 两者是优秀的博客/个人站 starter，长文表现好，但没有现成的多模块课程侧栏与复杂章节信息架构。借鉴排版即可，直接迁移反而要重写本项目已有能力。[AstroPaper README](https://github.com/satnaing/astro-paper/blob/main/README.md) [Cactus README](https://github.com/chrismwilliams/astro-theme-cactus/blob/main/README.md)

7. **VitePress、Nextra、Docusaurus 只作对照，不默认推荐。**
   - 三者文档能力都成熟，但会分别引入 Vue、Next.js/React、Docusaurus/React 技术栈。当前问题是视觉与阅读体验，不是 Astro 能力缺失；为换外观承担全框架、路由、Markdown pipeline 和部署重写，没有成本收益优势。

## 2. 当前仓库基线与不可破坏项

本地审计以提交 `54722c659b8d0ef730800745181a8aa5bf23df59` 为基线。

| 项目 | 当前事实 | 迁移约束 |
|---|---|---|
| 栈 | Astro `^4.16.19`、Tailwind `^3.4.19`、`@astrojs/tailwind`、TypeScript。[package.json](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/package.json) | 候选主分支大多已到 Astro 6/7 与 Tailwind 4 或 UnoCSS，主题替换同时会变成平台升级。 |
| 内容 | `guides`、`posts`、`interview`、`papers` 四个 glob loader collection，各有自定义 Zod schema；内容仍在仓库根 `docs/` 下。[content config](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/src/content/config.ts) | 不能假设把 Markdown 复制进新目录即可；schema、ID、draft、排序、章节、company/tier、paper source metadata 都要保留。 |
| 页面规模 | 当前 `dist` 审计为 **484 个 HTML 页面**；其中 `/guides/` 143、`/interview/` 182，并同时存在 `/cuda/`、`/distributed/`、`/inference/`、`/prerequisites/` 等分类路由。 | 不能只比较 Markdown 文件数；验收对象必须是完整 URL manifest。 |
| 固定路由 | 同一 guides collection 既由 `/guides/[...slug]` 生成，也按 category 由 `/cuda/`、`/distributed/`、`/inference/`、`/prerequisites/` 生成。[通用路由](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/src/pages/guides/%5B...slug%5D.astro) [CUDA 路由](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/src/pages/cuda/%5B...slug%5D.astro) | “一个内容文件对应一个新框架页面”会丢失别名 URL；GitHub Pages 又没有通用服务端重写，必须继续静态生成原路径。 |
| 章节导航 | `chapter` 字段、标题正则与 `order` 一起生成章节树和上一篇/下一篇顺序。[chapterGrouping.ts](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/src/utils/chapterGrouping.ts) [GuideSidebar](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/src/components/GuideSidebar.astro) | 新框架的文件名字母排序不能替代当前课程顺序。 |
| 阅读功能 | 桌面左侧章节树、右侧 H2/H3 TOC、移动端折叠 TOC、上一篇/下一篇、阅读进度、暗色和回到顶部。[GuideContent](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/src/components/GuideContent.astro) [TOC](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/src/components/TableOfContents.astro) | PoC 不能只验首页截图，必须覆盖长文章和三栏/移动布局。 |
| Markdown | Astro/Shiki 双主题；remark-math + rehype-katex；Mermaid 11 在客户端渲染；宽表与公式防溢出。[Astro config](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/astro.config.mjs) [Layout](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/src/layouts/Layout.astro) [global.css](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/src/styles/global.css) | 任意新 Markdown pipeline 都需做 KaTeX、Mermaid、原始 HTML、中文 heading ID 和 Shiki 回归。 |
| 搜索 | Pagefind 1.5.2，构建后索引并复制到 `public/pagefind`，客户端按 `BASE_URL` 加载。[package.json](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/package.json) [Search.astro](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/src/components/Search.astro) | 新主题即使也使用 Pagefind，索引范围、中文结果、base path 和构建顺序仍需重验。 |
| 部署 | 纯静态 GitHub Pages，`site=https://XiaYiHann.github.io`、`base=/AIInfraGuide`，Node 20，上传 `dist`。[Astro config](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/astro.config.mjs) [deploy workflow](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/.github/workflows/deploy.yml) | 所有导航、图片、搜索 bundle、canonical、sitemap 和 404 都必须在子路径部署下工作。 |

## 3. “直接换主题能否一键迁移”的真实性判断

**答案：不能。对本仓库而言，“一键迁移”是错误预期。**

原因不是样式覆盖不够，而是候选大多是完整 starter 或有自身内容路由约定：

1. **版本不兼容。**
   - Starlight 0.41.7：Astro 7。[package](https://github.com/withastro/starlight/blob/main/packages/starlight/package.json)
   - AstroPaper 6.1：Astro 7 + Tailwind 4。[package](https://github.com/satnaing/astro-paper/blob/main/package.json)
   - Astro Cactus 8.2：Astro 7 + Tailwind 4。[package](https://github.com/chrismwilliams/astro-theme-cactus/blob/main/package.json)
   - Retypeset：Astro 6 + UnoCSS。[package](https://github.com/radishzzz/astro-theme-retypeset/blob/master/package.json)
   - ReallySimpleDocs：Astro 6 + Tailwind 4 + Basecoat。[package](https://github.com/hunvreus/reallysimpledocs/blob/main/package.json)
   - Starlight Black 0.7.1：Node 22.12+、Starlight 0.41+。[package](https://github.com/adrian-ub/starlight-theme-black/blob/main/packages/starlight-theme-black/package.json)
   - SupportGenix Docs：Node 22.12+、Astro 6+、Tailwind 4；CLI 仍留下两项手工配置合并。[CLI README](https://github.com/supportgenix/supportgenix-docs/blob/main/packages/create-supportgenix-docs/README.md)

2. **内容契约不兼容。** Starlight 最新版要求名为 `docs` 的 collection，`docsLoader()` 默认只读取 `src/content/docs/`，并使用 `docsSchema()`；虽可扩展 schema 和自定义 `generateId()`，但不是现有四 collection 的原位替换。[手动安装](https://starlight.astro.build/manual-setup/#configure-content-collections) [loader/schema 参考](https://starlight.astro.build/reference/configuration/#configure-content-collections)

3. **URL 模型不兼容。** Starlight 的标准 docs loader 是文件式路由；本项目则从一个 collection 同时生成 `/guides/*` 和分类前缀页面。移动文件还可能改变 Unicode/大小写 slug；官方特意提供 `generateId()` 才能覆盖默认 slug 化行为。[docsLoader `generateId`](https://starlight.astro.build/reference/configuration/#generateid)

4. **信息架构不兼容。** 当前侧栏由 category、order、chapter 和标题正则动态组装；Starlight 自动侧栏默认按文件名字母排序，除非手工配置 sidebar。[Starlight sidebar 配置](https://starlight.astro.build/reference/configuration/#sidebar)

5. **Markdown 与客户端行为不等价。** “都支持 Markdown/代码高亮”不等于 KaTeX、Mermaid、原始 HTML、中文 anchor、代码复制、暗色同步和 Pagefind 范围完全一致。

6. **主题更新并不会自动合并本地改造。** AstroPaper、Cactus、Retypeset 都是可 fork/create 的模板；一旦把当前业务组件迁进去，后续上游更新仍是 Git 合并问题，而不是包升级。Cactus 和 Retypeset 的官方说明也直接描述了模板同步/冲突处理。[Cactus updating](https://github.com/chrismwilliams/astro-theme-cactus#updating) [Retypeset updates](https://github.com/radishzzz/astro-theme-retypeset#updates)

因此应把“换主题”拆成三个独立决策：**设计语言是否借鉴、布局壳是否复用、内容/路由是否迁移**。本项目建议前两项渐进进行，第三项默认不做。

## 4. 候选维护状态、许可证、技术栈与功能

### 4.1 Astro 候选

| 候选 | 截止日维护证据 | 许可证 | 当前技术栈 | 主要能力与适配判断 |
|---|---|---|---|---|
| **Astro Starlight** | 0.41.7 于 2026-08-05 发布；同日仍有提交，未归档。[release](https://github.com/withastro/starlight/releases/tag/%40astrojs%2Fstarlight%400.41.7) [commit](https://github.com/withastro/starlight/commit/656ffd54e5b27483f542c9eb8b12fd32f44372ae) | MIT。[LICENSE](https://github.com/withastro/starlight/blob/main/LICENSE) | Astro 7 integration、MD/MDX/Markdoc、Expressive Code、Pagefind、i18n。[package](https://github.com/withastro/starlight/blob/main/packages/starlight/package.json) | 文档侧栏、TOC、暗色、移动端、搜索、分页、编辑链接、组件 override、custom CSS；能力匹配最好。代价是先升级 Astro，并处理现有 collection/URL。[配置](https://starlight.astro.build/reference/configuration/) [搜索](https://starlight.astro.build/guides/site-search/) [demo](https://starlight.astro.build/) |
| **Starlight Theme Black** | v0.7.1 于 2026-07-22 发布；2026-07-23 后仍有仓库更新，未归档。[release](https://github.com/adrian-ub/starlight-theme-black/releases/tag/v0.7.1) [repo](https://github.com/adrian-ub/starlight-theme-black) | MIT。[LICENSE](https://github.com/adrian-ub/starlight-theme-black/blob/main/LICENSE) | Starlight plugin，要求 Starlight `>=0.41`、Node `>=22.12`。[package](https://github.com/adrian-ub/starlight-theme-black/blob/main/packages/starlight-theme-black/package.json) | 候选中最接近克制的 shadcn 文档视觉：低彩度、清晰边界、密度适中；继承 Starlight 的能力与迁移前提，只能在 Starlight PoC 后评估。[demo](https://starlight-theme-black.vercel.app/) |
| **Starlight Theme Rapide** | 0.5.2 于 2025-10-16 发布；最后核验提交 2026-05-13，未归档。[release](https://github.com/HiDeoo/starlight-theme-rapide/releases/tag/starlight-theme-rapide%400.5.2) [commit](https://github.com/HiDeoo/starlight-theme-rapide/commit/65632fffad4a20ca97b5610f52900449aa4d0fd1) | MIT。[LICENSE](https://github.com/HiDeoo/starlight-theme-rapide/blob/main/LICENSE) | CSS 为主的 Starlight plugin，peer Starlight `>=0.34`，Node `>=22.12`。[package](https://github.com/HiDeoo/starlight-theme-rapide/blob/main/packages/starlight-theme-rapide/package.json) | 更简洁的 Vitesse 风格，继承 Starlight 全部文档能力；多一层主题依赖，且本仓库 CI 当前 Node 20。只能在 Starlight PoC 后评估。[demo](https://starlight-theme-rapide.vercel.app/) |
| **SupportGenix Docs** | v1.1.0 于 2026-05-29 发布，仓库未归档。[release](https://github.com/supportgenix/supportgenix-docs/releases/tag/v1.1.0) [repo](https://github.com/supportgenix/supportgenix-docs) | ISC。[LICENSE](https://github.com/supportgenix/supportgenix-docs/blob/main/LICENSE) | Astro 6、Tailwind 4、MDX、Alpine、Pagefind；安装器要求 Node 22.12+。[package](https://github.com/supportgenix/supportgenix-docs/blob/main/package.json) [CLI](https://github.com/supportgenix/supportgenix-docs/blob/main/packages/create-supportgenix-docs/README.md) | 唯一提供面向现有 Astro 项目的 one-command installer 和 dry-run 的候选；但 CLI 不自动合并关键配置，也不迁移既有 collections/URL，只能算“脚手架一键安装”，不能算本项目“一键迁移”。[demo](https://astro.supportgenix.com/) |
| **AstroPaper** | v6.1.0 于 2026-06-06 发布；2026-08-05 仍有提交。[release](https://github.com/satnaing/astro-paper/releases/tag/v6.1.0) [commit](https://github.com/satnaing/astro-paper/commit/35cfa7fbe0b897306d27670d3819e55d5205f3dd) | MIT。[LICENSE](https://github.com/satnaing/astro-paper/blob/main/LICENSE) | Astro 7、Tailwind 4、TypeScript、MDX、Pagefind、Shiki transformer、Satori/Sharp。[package](https://github.com/satnaing/astro-paper/blob/main/package.json) | 最小、可访问、SEO、暗色、响应式、静态搜索、可折叠 TOC；长文友好，但博客归档模型不等于课程章节树。[README](https://github.com/satnaing/astro-paper/blob/main/README.md) [demo](https://astro-paper.pages.dev/) |
| **Astro Cactus** | v8.2.0 于 2026-07-14 发布；2026-07-24 仍有提交。[release](https://github.com/chrismwilliams/astro-theme-cactus/releases/tag/v8.2.0) [commit](https://github.com/chrismwilliams/astro-theme-cactus/commit/210d96d7e286535f183d34d60afdbb9f28b52df1) | MIT。[LICENSE](https://github.com/chrismwilliams/astro-theme-cactus/blob/main/LICENSE) | Astro 7、Tailwind 4、MDX、Pagefind、Expressive Code、Satori/Sharp。[package](https://github.com/chrismwilliams/astro-theme-cactus/blob/main/package.json) | 语义 HTML、暗色、响应式、Pagefind、admonition、RSS/SEO；仍是 post/note/tag 博客模型，且默认单栏/mono 气质并不适合直接承载大型课程树。[README](https://github.com/chrismwilliams/astro-theme-cactus/blob/main/README.md) [demo](https://astro-cactus.chriswilliams.dev/) |
| **Retypeset** | v1.0.0 release 为 2025-08-07；最后核验提交 2026-04-12，未归档，但更新节奏弱于前三者。[release](https://github.com/radishzzz/astro-theme-retypeset/releases/tag/v1.0.0) [commit](https://github.com/radishzzz/astro-theme-retypeset/commit/a636b6d393be714cab52d3fc4baddd3f3905f701) | MIT。[LICENSE](https://github.com/radishzzz/astro-theme-retypeset/blob/master/LICENSE) | Astro 6、UnoCSS、MDX、KaTeX、Mermaid、TOC、i18n、评论系统。[package](https://github.com/radishzzz/astro-theme-retypeset/blob/master/package.json) | 候选中长文排版取向最明确，中文 demo 完整；但博客结构、UnoCSS 和自带脚本让代码迁入成本高。建议只提取字体、行宽、留白和弱装饰原则。[README](https://github.com/radishzzz/astro-theme-retypeset/blob/master/README.md) [demo](https://retypeset.radishzz.cc/) |
| **ReallySimpleDocs** | npm/repo manifest 已到 1.0.13，2026-07-05 有提交；GitHub 的 latest release 页面仍标 1.0.2（2026-06-24），说明项目活跃但发布标记尚不完全一致。[package](https://github.com/hunvreus/reallysimpledocs/blob/main/package.json) [release](https://github.com/hunvreus/reallysimpledocs/releases/tag/1.0.2) [commit](https://github.com/hunvreus/reallysimpledocs/commit/8c43b7f574d05ee209aa5de57727a1f41f71edb9) | MIT。[LICENSE](https://github.com/hunvreus/reallysimpledocs/blob/main/LICENSE.md) | Astro 6 integration、Tailwind 4、Basecoat、MD/MDX、Lunr、Shiki。[package](https://github.com/hunvreus/reallysimpledocs/blob/main/package.json) | `docs.json` 侧栏、命令搜索、代码复制、`llms.txt`/Markdown 导出、组件 slots；比博客主题更贴近技术文档，但生态和实战规模明显早期，且要把 Pagefind/现有 collection 改成它的 docs 约定。[README](https://github.com/hunvreus/reallysimpledocs/blob/main/README.md) [demo](https://reallysimpledocs.com/) |

### 4.2 跨框架对照

| 候选 | 截止日状态、许可证与栈 | 对照价值 | 为什么不默认推荐 |
|---|---|---|---|
| **VitePress** | MIT；Vue + Vite 的静态文档生成器。npm stable 为 1.6.4，2.0.0-alpha.19 于 2026-08-02 发布，2026-08-10 仍有提交。[repo](https://github.com/vuejs/vitepress) [npm](https://www.npmjs.com/package/vitepress) [alpha release](https://github.com/vuejs/vitepress/releases/tag/v2.0.0-alpha.19) [commit](https://github.com/vuejs/vitepress/commit/08cc11092b14f1fb97b36dcb1f6b8c61ad823c62) | 默认文档外观克制，sidebar/outline/local search/dark mode/GitHub Pages 路径配置成熟。[官方文档](https://vitepress.dev/) [部署](https://vitepress.dev/guide/deploy#github-pages) | 需要把 Astro collections、Astro 组件和 Markdown pipeline 改为 Vue/VitePress；2.x 仍是 alpha 线。 |
| **Nextra** | MIT；Next.js + React + MDX。latest release `nextra-theme-docs@4.6.1` 为 2025-12-04，2026-06-23 仍有提交；主分支 manifest 使用 Next 16/React 19 开发。[repo](https://github.com/shuding/nextra) [release](https://github.com/shuding/nextra/releases/tag/nextra-theme-docs%404.6.1) [package](https://github.com/shuding/nextra/blob/main/packages/nextra-theme-docs/package.json) [commit](https://github.com/shuding/nextra/commit/d6e80e1dd627b781429a6ee989b15ebba688c8ea) | docs theme 的页面地图、侧栏、TOC、MDX 组件与搜索可作为功能对照。[官方文档/demo](https://nextra.site/) | 引入 Next/React 的运行与构建约定，只为静态课程站换外观不划算；固定路径和 base 部署都要重验。 |
| **Docusaurus** | MIT（代码；官方仓库另说明 docs 内容使用 CC）；v3.10.2 于 2026-07-10 发布，2026-08-07 仍有提交。React 文档平台。[repo/LICENSE 说明](https://github.com/facebook/docusaurus) [release](https://github.com/facebook/docusaurus/releases/tag/v3.10.2) [commit](https://github.com/facebook/docusaurus/commit/3f483e80e326cc646b54b83d564b3f0c4881b9a6) | 大型文档的版本化、i18n、docs/blog、插件体系和部署能力最全面。[官方文档](https://docusaurus.io/docs) | 功能明显超出当前需求；迁移到 React/Docusaurus 的重写面最大，默认视觉也不自动解决“AI 风格”。 |

## 5. 候选评分矩阵

评分 1–5。加权：文档 IA 25%、现有功能覆盖 20%、URL/内容复用 20%、长文与视觉克制 15%、维护成熟度 10%、可定制 5%、GitHub Pages 静态适配 5%。这是针对 **AIInfraGuide 当前约束** 的适配分，不是项目质量排名。

| 方案 | 文档 IA | 功能覆盖 | URL/内容复用 | 长文/克制 | 维护 | 定制 | Pages | 加权分 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **现有架构 + 设计系统重构** | 5.0 | 5.0 | 5.0 | 3.5→5.0 | 4.0 | 5.0 | 5.0 | **4.68** |
| **Starlight 核心，渐进页面壳** | 5.0 | 5.0 | 3.5 | 4.0 | 5.0 | 4.5 | 5.0 | **4.53** |
| **Starlight + Black** | 5.0 | 5.0 | 3.5 | 4.7 | 4.0 | 4.0 | 5.0 | **4.51** |
| **Starlight + Rapide** | 5.0 | 5.0 | 3.5 | 4.5 | 4.0 | 4.0 | 5.0 | **4.48** |
| SupportGenix Docs | 4.5 | 4.0 | 2.0 | 3.5 | 3.0 | 4.0 | 4.5 | 3.58 |
| ReallySimpleDocs | 4.5 | 3.5 | 2.5 | 4.0 | 3.0 | 3.5 | 4.5 | 3.63 |
| Retypeset | 2.5 | 4.0 | 2.0 | 5.0 | 3.5 | 4.0 | 4.5 | 3.35 |
| AstroPaper | 2.0 | 3.5 | 2.0 | 4.5 | 4.5 | 4.0 | 4.5 | 3.15 |
| Astro Cactus | 2.0 | 3.5 | 2.0 | 4.0 | 4.5 | 4.0 | 4.5 | 3.08 |

跨框架对照没有纳入主排名：即便其通用文档能力得分高，**URL/内容复用与迁移成本在本项目上会成为否决项**。

## 6. 当前“AI 生成感”的具体来源

问题不在蓝色本身，而在多个流行模板信号同时出现且重复使用：

1. **渐变字和多色强调。** 首页标题、Logo、导航指示、阅读进度条同时使用蓝—紫—青渐变。[首页](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/src/pages/index.astro) [Layout](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/src/layouts/Layout.astro)
2. **蓝紫光球与大面积 blur。** 首页和 About 使用 400–500px 的蓝/紫圆形背景与 `blur-[120px]`，这是常见生成式 SaaS landing page 语汇。[首页](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/src/pages/index.astro) [About](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/src/pages/about/index.astro)
3. **玻璃拟态叠加。** 固定 header、移动抽屉、搜索遮罩和悬浮按钮反复使用半透明背景、`backdrop-blur` 与大阴影。[Layout](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/src/layouts/Layout.astro) [Search](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/src/components/Search.astro)
4. **pill、圆角卡片与标签过量。** 状态、分类、tag、按钮、卡片、导航都用 `rounded-xl/2xl/full`，使信息层级被统一成“组件展示墙”。[首页](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/src/pages/index.astro) [GuideContent](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/src/components/GuideContent.astro)
5. **hover 浮起、放大、发光。** 卡片 `-translate-y-1 + shadow-xl`、图标 scale、渐变 glow 和箭头位移同时使用，动效在阅读型站点中抢夺注意力。[CategoryCard](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/src/components/CategoryCard.astro) [PaperCard](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/src/components/PaperCard.astro)
6. **emoji 被当作导航/内容类型图标。** 分类、导航热点和 Paper 类型混用 🗺️📚⚡🌐🚀📊🔥📖📄 等，跨平台字形不一致，也会加强模板拼装感。[categories.ts](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/src/utils/categories.ts) [Layout](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/src/layouts/Layout.astro) [Paper page](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/src/pages/papers/%5B...slug%5D.astro)
7. **Mac 红黄绿灯代码块。** 伪窗口圆点、厚阴影、绿色左边框、悬浮按钮和语言标签叠加，视觉装饰超过代码本身。[global.css](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/src/styles/global.css)

这些元素单独使用并非错误；问题是密度高、层层叠加、几乎所有页面都出现。高级感应来自一致的字阶、行宽、节奏和边界，而不是更多特效。

## 7. 更克制的设计 token 方向

### 7.1 原则

- **一套中性色 + 一个低饱和 accent**，不使用标题渐变。
- **正文优先于容器**：默认不用卡片包正文或列表；用空白、细分隔线和字重建立层级。
- **圆角有语义**：小控件 4–6px，普通容器最多 8px；`999px` 只给真实状态 chip，不给所有标签。
- **阴影只表达浮层**：搜索弹层和移动抽屉可有弱阴影，普通卡片和代码块没有阴影。
- **动效只反馈状态**：颜色/透明度 120–160ms；去除 hover 浮起、放大、发光和通用 `transition-all`。
- **图标统一为线性 SVG**；emoji 只保留在文章正文作者表达中，不作为 UI 组件。

### 7.2 建议 token

```css
:root {
  color-scheme: light;
  --color-bg: #fafaf8;
  --color-surface: #ffffff;
  --color-surface-subtle: #f4f4f1;
  --color-text: #202124;
  --color-text-muted: #62666b;
  --color-border: #dedfdb;
  --color-border-strong: #c7c9c5;
  --color-accent: #25627a;
  --color-accent-soft: #e9f1f3;

  --font-sans: Inter, -apple-system, BlinkMacSystemFont, "PingFang SC",
    "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
  --font-mono: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;

  --text-body: 1rem;
  --leading-body: 1.75;
  --content-width: 72ch;
  --space-unit: 0.25rem;
  --radius-control: 0.375rem;
  --radius-panel: 0.5rem;
  --shadow-overlay: 0 12px 32px rgb(0 0 0 / 0.12);
  --duration-fast: 140ms;
}

html.dark {
  color-scheme: dark;
  --color-bg: #111210;
  --color-surface: #171816;
  --color-surface-subtle: #1d1f1c;
  --color-text: #e7e8e3;
  --color-text-muted: #a2a59f;
  --color-border: #30322e;
  --color-border-strong: #444740;
  --color-accent: #75aebe;
  --color-accent-soft: #1b3035;
}
```

落地重点：

- 正文中文行高从当前约 1.6 提高到 1.7–1.8，行宽控制在 68–72ch；段落间距大于行距。
- H1/H2 使用纯色、略紧字距；H2 可保留细下边线，但不加渐变装饰条。
- 侧栏保持树结构，用字重、缩进和 2px 单色 active indicator 表达层级。
- 代码块改为平面容器：1px border、4–6px 圆角、无交通灯、无厚阴影；保留语言标签、复制按钮、横向滚动、行号和 Shiki 双主题。
- 首页从“hero + 光球 + 卡片墙”改为“简短定位 + 课程目录 + 最近更新”；模块列表优先采用文本目录或带细分隔线的两栏列表。
- Mermaid、KaTeX、表格只做可读性与溢出处理，不把每个元素再包成彩色卡片。

## 8. 低风险迁移策略

### Phase 0：冻结基线，不改结构

产物：

- 导出 484 个 URL 的 manifest、status、canonical、title。
- 固定 10–12 个视觉回归页面和 10 个中文 Pagefind 查询。
- 记录当前构建命令、页面数、搜索索引文件、控制台错误和关键截图。

门禁：后续每阶段 **URL 零丢失、页面数不下降、内容文件零批量移动**。

### Phase 1：现有 Astro 4 + Tailwind 3 内完成视觉收敛

仅改设计 token 与已有组件表现：

- 去掉渐变字、光球、玻璃模糊、普通卡片阴影/浮起、UI emoji、Mac 红绿灯。
- 统一正文、侧栏、TOC、代码、表格、callout、首页目录。
- 不改 content config、route、slug、构建脚本和 Markdown 插件。

这是应先上线的版本。它可以验证“问题是否主要来自视觉”，而不把框架迁移噪声混入结论。

### Phase 2：建立布局适配层

在现有数据逻辑与 UI 间形成稳定接口，例如：

- `buildGuideSidebar(category, currentId) -> SidebarItem[]`
- `buildPagination(entries, currentId) -> { prev, next }`
- `ArticleShell` 统一接收 frontmatter、headings、sidebar、content slot。

先让现有 Layout 使用该接口；这一步不引入 Starlight。目的在于让未来页面壳可替换，而不是让内容模型依赖某个主题。

### Phase 3：平台升级与主题 PoC 分开

单独分支/工作树按 Astro 官方升级路径推进 4→5→6→7；Tailwind 3→4 另做一次提交。每个 major 都运行完整 URL、Markdown 和 Pagefind 回归。不要在同一提交同时搬内容、改路由、换主题、升级 Tailwind。

原因：Starlight 0.41.7 已要求 Astro 7；Rapide 和多个 starter 还要求 Node 22，而当前 Pages workflow 使用 Node 20。[Starlight package](https://github.com/withastro/starlight/blob/main/packages/starlight/package.json) [Rapide package](https://github.com/HiDeoo/starlight-theme-rapide/blob/main/packages/starlight-theme-rapide/package.json) [当前 workflow](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/.github/workflows/deploy.yml)

### Phase 4：Starlight 页面壳 PoC

只选择代表性路由，不移动 Markdown：

- 保留当前 collection 和 `getStaticPaths()`。
- 用 `<StarlightPage>` 接收现有 `Content`、`headings` 和由 Phase 2 生成的 sidebar。
- 先保留现有 Pagefind 构建链、KaTeX/Mermaid 插件和固定 URL。
- 首页、About、模块 landing 暂时保持自定义 Astro 页面。

PoC 通过后，再决定是否扩大 Starlight 壳覆盖面。**默认仍不把四个 collection 合并为 Starlight `docs` collection**；只有当 schema、重复 URL 和文章类型都能无损表达时才讨论内容迁移。

### Phase 5：可选比较 Black / Rapide

同一组页面在纯 Starlight token、Starlight Black 与 Rapide 下做盲评。若第三方主题只改变配色且没有显著提高阅读体验，就不增加依赖。Retypeset/AstroPaper/Cactus 只用于视觉参考截图，不纳入生产依赖。

## 9. PoC 验收项

### 9.1 代表页面

至少覆盖：

1. 首页；
2. 一个模块 landing；
3. 一篇含长代码块、行号和横向滚动的 CUDA 文章；
4. 一篇含 KaTeX block/inline 公式的文章；
5. 一篇含 Mermaid 的文章；
6. 一篇含宽表格、图片和原始 HTML 的 Paper 精读；
7. 一篇 interview 页面；
8. 404 页面。

### 9.2 自动门禁

- `astro check` 和生产构建成功。
- HTML 页面数仍为 484；基线 URL manifest 全部存在，无意外新增/删除。
- 所有内部链接、canonical、sitemap、静态资源和 Pagefind 结果都包含正确的 `/AIInfraGuide` base。
- 固定抽测 Unicode 中文文件名、空格、大小写、深层目录 slug。
- Pagefind 对 10 个固定中文查询有结果，结果链接可在 GitHub Pages 子路径打开。
- KaTeX 与 Mermaid 均成功渲染，页面无未处理源码、无控制台异常。
- Shiki 亮/暗主题均可读；代码复制、行号、横向滚动可用。
- 当前 category/order/chapter/company 排序与 prev/next 关系保持一致。

### 9.3 视觉与交互门禁

- 视口：360×800、768×1024、1280×800、1536×960。
- 亮色、暗色、系统主题首次加载均无闪烁或低对比文本。
- 仅键盘可完成 skip link、搜索、移动菜单、侧栏折叠、TOC、主题切换和复制代码。
- 200% zoom 不丢内容；移动端公式、表格和代码只在自身容器横向滚动。
- 正文行宽 68–72ch、中文行高 1.7–1.8；主内容不会因左右侧栏挤压到不可读。
- 普通列表/卡片 hover 不发生位移或缩放；`prefers-reduced-motion` 下无非必要动画。
- Lighthouse Accessibility 目标 ≥95；性能相对基线不得出现超过 5 分的回退。最终以真实 GitHub Pages URL复测。

### 9.4 主观盲评

让至少 3 名读者在不知道候选名称的情况下比较同一篇长文，问题只问：

- 10 分钟阅读后哪版最不疲劳？
- 能否快速定位当前章节和 H2/H3？
- 代码、公式和表格哪版最清楚？
- 哪版最像稳定维护的工程文档，而不是模板 landing page？

“更炫”不作为验收指标。

## 10. 回滚方案

1. **内容不搬家。** PoC 阶段不批量修改 Markdown/frontmatter，因此回滚不涉及内容恢复。
2. **提交粒度可逆。** 平台 major、Tailwind major、布局适配层、视觉 token、Starlight 壳分别提交；禁止 squash 成一次不可分离的大改。
3. **保留旧壳。** 新旧 `ArticleShell` 并存到完整门禁通过；用单一配置或 import 切换，不在验证前删除旧组件。
4. **冻结基线。** 给现网提交打 tag，并保存 URL manifest、截图、Pagefind 查询结果与构建日志。
5. **发布回滚。** 若 GitHub Pages 验证失败，直接从基线 SHA 重新运行现有 deploy workflow，而不是在生产分支现场修补。
6. **触发回滚条件。** 任一固定 URL 404、Pagefind base 错误、中文 slug 改变、KaTeX/Mermaid 失效、页面数下降、移动端导航不可用，均立即停止扩面并回到上一阶段。

## 11. 最终决策

- **现在就做：** 在当前 Astro 4 + Tailwind 3 代码中执行视觉去模板化；这能直接解决用户感知问题，且不碰 484 页的内容与 URL 风险面。
- **PoC 主候选：** Starlight 0.41.7 的 `<StarlightPage>` 渐进壳；先完成 Astro major 升级，再对代表页面验证。
- **可选皮肤：** 首选比较 Starlight Black，Rapide 为备选；只在纯 Starlight PoC 通过后 A/B。
- **“一键”候选结论：** SupportGenix Docs 的 CLI 最接近，但它只能一键搭脚手架，不能自动迁移本项目内容、配置和 URL；不建议直接运行到当前主分支。
- **排版参考：** Retypeset 第一，AstroPaper 第二；不直接合并其 starter 代码。
- **不选为主框架：** Cactus、ReallySimpleDocs（前者博客导向，后者尚早期）；VitePress/Nextra/Docusaurus（跨框架成本没有业务理由支撑）。
- **明确否定：** 不存在对本仓库安全的“一键换主题”。成功路径是 **视觉层先行 → 数据/布局接口稳定 → 平台升级 → 小范围 Starlight 壳 → 门禁通过后扩面**。

## 12. 来源清单

### 当前仓库

- [package.json](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/package.json)
- [astro.config.mjs](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/astro.config.mjs)
- [content collections](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/src/content/config.ts)
- [Layout](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/src/layouts/Layout.astro)
- [GuideContent](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/src/components/GuideContent.astro)
- [GuideSidebar](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/src/components/GuideSidebar.astro)
- [global.css](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/src/styles/global.css)
- [GitHub Pages workflow](https://github.com/XiaYiHann/AIInfraGuide/blob/54722c659b8d0ef730800745181a8aa5bf23df59/.github/workflows/deploy.yml)

### Astro 候选官方来源

- Starlight：[docs/demo](https://starlight.astro.build/) · [repo](https://github.com/withastro/starlight) · [package](https://github.com/withastro/starlight/blob/main/packages/starlight/package.json) · [license](https://github.com/withastro/starlight/blob/main/LICENSE) · [configuration](https://starlight.astro.build/reference/configuration/) · [custom pages](https://starlight.astro.build/guides/pages/) · [search](https://starlight.astro.build/guides/site-search/)
- AstroPaper：[repo/README](https://github.com/satnaing/astro-paper) · [demo](https://astro-paper.pages.dev/) · [package](https://github.com/satnaing/astro-paper/blob/main/package.json) · [license](https://github.com/satnaing/astro-paper/blob/main/LICENSE)
- Astro Cactus：[repo/README](https://github.com/chrismwilliams/astro-theme-cactus) · [demo](https://astro-cactus.chriswilliams.dev/) · [package](https://github.com/chrismwilliams/astro-theme-cactus/blob/main/package.json) · [license](https://github.com/chrismwilliams/astro-theme-cactus/blob/main/LICENSE)
- Retypeset：[repo/README](https://github.com/radishzzz/astro-theme-retypeset) · [demo](https://retypeset.radishzz.cc/) · [package](https://github.com/radishzzz/astro-theme-retypeset/blob/master/package.json) · [license](https://github.com/radishzzz/astro-theme-retypeset/blob/master/LICENSE)
- Starlight Black：[repo](https://github.com/adrian-ub/starlight-theme-black) · [demo](https://starlight-theme-black.vercel.app/) · [package](https://github.com/adrian-ub/starlight-theme-black/blob/main/packages/starlight-theme-black/package.json) · [license](https://github.com/adrian-ub/starlight-theme-black/blob/main/LICENSE)
- Starlight Rapide：[repo](https://github.com/HiDeoo/starlight-theme-rapide) · [demo](https://starlight-theme-rapide.vercel.app/) · [package](https://github.com/HiDeoo/starlight-theme-rapide/blob/main/packages/starlight-theme-rapide/package.json) · [license](https://github.com/HiDeoo/starlight-theme-rapide/blob/main/LICENSE)
- SupportGenix Docs：[repo](https://github.com/supportgenix/supportgenix-docs) · [demo](https://astro.supportgenix.com/) · [CLI](https://github.com/supportgenix/supportgenix-docs/blob/main/packages/create-supportgenix-docs/README.md) · [license](https://github.com/supportgenix/supportgenix-docs/blob/main/LICENSE)
- ReallySimpleDocs：[repo/README](https://github.com/hunvreus/reallysimpledocs) · [demo](https://reallysimpledocs.com/) · [package](https://github.com/hunvreus/reallysimpledocs/blob/main/package.json) · [license](https://github.com/hunvreus/reallysimpledocs/blob/main/LICENSE.md)

### 跨框架官方来源

- VitePress：[docs/demo](https://vitepress.dev/) · [repo](https://github.com/vuejs/vitepress) · [package](https://github.com/vuejs/vitepress/blob/main/package.json) · [license](https://github.com/vuejs/vitepress/blob/main/LICENSE)
- Nextra：[docs/demo](https://nextra.site/) · [repo](https://github.com/shuding/nextra) · [docs-theme package](https://github.com/shuding/nextra/blob/main/packages/nextra-theme-docs/package.json) · [license](https://github.com/shuding/nextra/blob/main/LICENSE)
- Docusaurus：[docs/demo](https://docusaurus.io/docs) · [repo](https://github.com/facebook/docusaurus) · [core package](https://github.com/facebook/docusaurus/blob/main/packages/docusaurus/package.json) · [license](https://github.com/facebook/docusaurus/blob/main/LICENSE)
