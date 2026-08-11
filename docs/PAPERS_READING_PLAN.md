# Paper 精读:候选清单与学习路线规划

> 状态:**第一批已上线(2026-08-11)**
> 第一批:PagedAttention / Orca / Splitwise 三篇精读 + 精读路线站点改造,均已部署上线
> 目标:让精读从"按日期平铺的列表"升级为"有时间顺序的阅读路线",并给出新增候选的优先级与写作流程。

## 1. 现状盘点

现有 7 篇精读:

| 文章 | 类型 | 定位 |
| --- | --- | --- |
| assisted-generation-notes | 博客 | 投机解码直觉入门 |
| dspark-notes | 论文 | 投机解码前沿顶点 |
| deepseek-v4-notes / kimi-k3-notes / glm-5-notes / gemma-4-notes / kimi-k25-notes | 技术报告 | 前沿模型解读(时效性强) |

问题:

1. 索引页按 `pubDate` 倒序平铺,没有学习顺序;读者不知道先读哪篇;
2. 5 篇模型报告同质化,挤占了算法/系统精读的空间;
3. 没有前置关系、难度、时长信息,无法规划投入;
4. 首页只取最新 3 篇,和路线无关。

## 2. 两条主线:原理精读 vs 前沿解读

- **A 线 · 原理精读(evergreen)**:推理系统、投机解码、注意力、分布式训练。文章之间有依赖,必须排序阅读;面试价值长期有效。
- **B 线 · 前沿解读(时效)**:模型技术报告、生产系统博客。可并行阅读,不构成前置依赖。

站点 `docs/guides/AI Infra学习路线.md` 已经为每个章节挂接了"对应论文"清单(FA1/2/3、DeepSeekMoE、Megatron-LM、ZeRO、vLLM、SGLang、Orca、SmoothQuant/GPTQ/AWQ/KIVI/Marlin、Speculative Sampling/Medusa/EAGLE-2/Block Verification 等)。**精读 = 这些论文的原文视角补课**,与站内教程互补而非重复。

## 3. 候选池与优先级

标注:P0 = 首批应写;P1 = 第二批;P2 = 可选/低优先。

### P0(6 篇,求职最高频)

| 候选 | 来源 | 对应站内章节 | 理由 |
| --- | --- | --- | --- |
| PagedAttention(vLLM) | arXiv 2309.06180 | 模块四 §2.1 | KV cache 管理是推理岗第一高频考点 |
| Orca: Continuous Batching | OSDI'22 | 模块四 §2.2 | batching 机制源头,引擎架构必答 |
| EAGLE-2 | arXiv 2406.16858 | 模块四 §5.3 | 自回归 drafter 代表;DSpark 精读的前置对照 |
| Splitwise | arXiv 2311.18677 | 模块四 §7 | PD 解耦第一篇文章,硬件分相核心 |
| FlashAttention-2 | arXiv 2307.08691 | 模块二 §6.2 | 注意力 kernel 标杆,算子岗必答 |
| Mooncake | arXiv 2407.00079 | 模块四 §10 | Kimi 生产 serving 架构,KV-centric 解耦代表作 |

### P1(7 篇)

| 候选 | 来源 | 对应站内章节 | 理由 |
| --- | --- | --- | --- |
| SARATHI | arXiv 2308.16369 | 模块四 §2.4 | chunked prefill 源头,与 Orca 成对 |
| SGLang / RadixAttention | arXiv 2312.07104 | 模块四 §2.3 | prefix cache 机制 + 结构化前端 |
| DistServe | arXiv 2401.09670 | 模块四 §7 | goodput 视角的 PD 解耦,与 Splitwise 对照 |
| DFlash | arXiv 2602.06036 | 模块四 §5 | 并行 drafter 代表;与 DSpark 精读成对 |
| DeepSeek-V3.2(DSA) | arXiv 2512.02556 | 模块四 §4/长上下文 | 稀疏注意力工程化,面经新热点 |
| ZeRO | arXiv 1910.02054 | 模块三 §5 | 显存优化的核心方法 |
| DeepSeek-V3 技术报告 | arXiv 2412.19437 | 模块三 §7/§8 | MLA+FP8+DualPipe 一次讲清,训练推理通吃(DualPipe 无独立 arXiv,并入本篇) |

### P2(可选,10 篇)

| 候选 | 来源 | 说明 |
| --- | --- | --- |
| Speculative Sampling | arXiv 2302.01318 | 理论基石,可并入投机谱系开篇 |
| Medusa | arXiv 2401.10774 | 多头解码,可并入 EAGLE 篇做对比 |
| Block Verification | arXiv 2403.10444 | 投机解码 block 验证理论 |
| FlashDecoding / FlashDecoding++ | PyTorch/CRFM 博客 / arXiv 2311.01282 | 长上下文 decode 优化 |
| NSA(Native Sparse Attention) | arXiv 2502.11089 | 训练侧稀疏注意力,与 DSA 对照 |
| Ring Attention | arXiv 2310.06226 | 长序列训练,模块三 §9 配套 |
| Megatron-LM | arXiv 1909.08053 | TP/PP 里程碑,模块三 §6/§7 配套 |
| Efficiently Serving LLM Mixtures | LMSYS 博客 2024-01 | MoE 混合流量服务 |
| H2O / SnapKV | arXiv 2306.14048 / 2404.14469 | KV 驱逐,可选 |
| Marlin / FlashInfer / SpecInfer 等 | 写时核实 ID | kernel/引擎级,站内章节已深,低优先 |

**明确不做**:FA3、FlashInfer、SmoothQuant/AWQ/GPTQ/KIVI、DeepSeekMoE —— 站内对应章节已深度覆盖,精读边际价值低;Qwen3、DeepSeek-R1 等模型报告 —— 视求职需要再定。

## 4. 学习路线(时间顺序)

五站 + 一个并行区。每站给出"读完你能做什么",前置依赖同时标注站内章节。

```
第 0 站 直觉启动(0.5h)            已有 1 篇
  ① assisted generation(已有)
  → 产出:投机解码在省什么、三笔账

第 1 站 推理引擎主干(2-3 周)     前置:模块四 §1-2
  ② PagedAttention → ③ Orca → ④ SARATHI → ⑤ RadixAttention → ⑥ Splitwise/DistServe
  → 产出:能讲清 vLLM/SGLang 的 KV 管理、batching、prefix cache、PD 解耦;
        推理岗面经 80% 的系统题落在这站

第 2 站 投机解码谱系(1-2 周)     前置:第 1 站 + 模块四 §5
  ⑦ EAGLE-2 → ⑧ DFlash → ⑨ DSpark(已有)
  → 产出:能对比 自回归/并行/半自回归 三条 drafter 路线,
        解释接受率、验证预算与引擎执行的关系

第 3 站 注意力与长上下文(1-2 周)  前置:模块二 §6
  ⑩ FlashAttention-2 → ⑪ FlashDecoding → ⑫ DSA/NSA → (可选)Ring Attention
  → 产出:能解释在线 softmax、split-K、稀疏注意力如何省 FLOPs 和显存

第 4 站 分布式训练(1-2 周,训练方向) 前置:模块三 §1-2
  ⑬ ZeRO → ⑭ Megatron-LM → ⑮ DeepSeek-V3 报告(MLA/FP8/DualPipe)
  → 产出:能讲清显存账本、TP/PP/EP 与通信重叠

第 5 站 生产与前沿(随时并行)      前置:任意
  ⑯ Mooncake → ⑰ LLM Mixtures 博客 → 已有 5 篇模型报告
  → 产出:生产架构视野 + 面试谈资
```

总投入:约 20-22 篇 × 0.5-1 天 ≈ 4-6 周(与模块学习并行,不是前置)。

## 5. 站点接入方案(不新增栏目)

1. **frontmatter 扩展**(`src/content/config.ts`,全部可选字段 + 默认值,老文章零破坏):
   - `stage`: 枚举 `intuition | engine | speculative | attention | distributed | production`
   - `order`: 站内序号(决定组内顺序)
   - `prereqs`: 前置文章 id 数组
   - `minutes`: 预计阅读时长
   - `difficulty`: 1-3
2. **`papers/index.astro` 重构**:顶部"精读路线"概览(阶段 → 目标 → 前置依赖);主体按 stage 分组渲染,每组含"读完你能…";底部放"前沿解读(并行阅读)"区。
3. **PaperCard**:增加阶段标签、组内序号、难度、时长;前置未读的卡片显示"建议先读:xxx"。
4. **文章页**:增加"上一篇/下一篇"(按路线顺序)与"下一站建议"。
5. **首页精读区**:由"最新 3 篇"改为"精读路线 · 从这里开始"(取第 0/1 站前 3 篇),保留"查看全部"。
6. **`docs/guides/AI Infra学习路线.md`**:增加精读路线小节,链接 `/papers`。
7. URL 全部不变(`/papers/<id>`),不新增顶级栏目。

## 6. 写作与发布流程(每篇固定门禁,同 DSpark 流程)

1. **选题**:绑定对应章节 + 岗位线,在本文档登记;
2. **证据账本**:来源、公式、数字、工程差异分层(论文事实 vs 当前工程),冻结来源;
3. **撰写**:按 tech-report-writing 契约(覆盖地图、最小例子、机制卡、局限、来源锚点),无正文 H1,加 `<!-- more -->`;
4. **独立证据审校**:冻结来源逐项核对,必改项清零才算 PASS;
5. **构建门禁**:`npm run build` + 站内链接检查 + Pagefind + 图片存在性;
6. **提交部署**:push → workflow → 线上 HTTP + 渲染断言。

## 7. 落地节奏建议

- **第 1 批**(求职最高频):PagedAttention → Orca → Splitwise
- **第 2 批**:EAGLE-2 → DFlash → SARATHI
- **第 3 批**:FlashAttention-2 → Mooncake → DeepSeek-V3 报告
- 之后:DSA、RadixAttention、DistServe、ZeRO、Megatron 按需推进
- 站点改造(第 5 节)与第 1 批同步实施,首批三篇上线时索引页即呈现路线形态


---

## 8. 执行状态(2026-08-11 更新)

### 第一批:已完成并上线

| 文章 | 阶段/顺序 | 审校 | 提交 | 部署 run |
| --- | --- | --- | --- | --- |
| PagedAttention | 第 1 站 · 第 1 篇 | 独立审校 PASS(含一次人工裁定) | `ef5646a` | 31459385571 |
| Orca | 第 1 站 · 第 2 篇 | 独立审校 PASS | `ad9b40c` | 31461035569 |
| Splitwise | 第 1 站 · 第 5 篇 | 独立审校 PASS(含一次人工裁定) | `9574c55` | 31464357649 |

### 站点改造:已完成

- `src/content/config.ts`:papers schema 增加可选字段 stage/order/prereqs/minutes/difficulty;
- `src/pages/papers/index.astro`:精读路线概览 + 按阶段分组渲染(既有文章由代码侧路线配置归类,未改已上线文章);
- `src/components/PaperCard.astro`:卡片显示阶段、站内顺序、预计时长、难度、前置阅读;
- `src/pages/index.astro`:精读区改为「从这里开始」,展示路线开头 3 篇;
- `docs/guides/AI Infra学习路线.md`:新增「精读路线(Paper 精读)」小节;
- 每篇上线后构建 486→487→488 页,0 errors,Pagefind 可搜,线上断言全部通过。

### 后续批次(按第 3 节优先级推进)

- P0 剩余:FlashAttention-2、Mooncake;
- P1:EAGLE-2、DFlash、SARATHI、RadixAttention、DistServe、DeepSeek-V3.2(DSA)、ZeRO、DeepSeek-V3 技术报告;
- P2:按需。
