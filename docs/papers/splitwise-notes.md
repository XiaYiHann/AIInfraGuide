---
title: "Splitwise 精读：把 prompt 计算和 token 生成拆到不同机器，同功率同成本吞吐提升 2.35×"
description: "拆解 Splitwise 的阶段拆分架构：prompt 计算与 token 生成分池部署、per-layer KV-cache 异步转移把开销压到 prompt 计算时间的 7% 以内，换来同功率同成本吞吐 2.35×、成本低 20% 时吞吐 1.4×,并对照 2026 年 vLLM、Dynamo、Mooncake 的 PD 解耦现状。"
pubDate: 2026-08-11
originalUrl: "https://arxiv.org/abs/2311.18677"
sourceType: "paper"
originalAuthor: "Pratyush Patel et al. (University of Washington & Microsoft)"
tags: ["Splitwise", "PD 解耦", "Phase Splitting", "KV Cache", "LLM Serving"]
stage: engine
order: 5
prereqs: ["pagedattention-notes", "orca-notes"]
minutes: 40
difficulty: 3
---

> 原文：[Splitwise: Efficient Generative LLM Inference Using Phase Splitting](https://arxiv.org/abs/2311.18677)(Pratyush Patel et al.,University of Washington & Microsoft,MICRO 2024,arXiv:2311.18677 v2,2023-11)

LLM serving 的一个请求要经历两段需求完全相反的旅程：先是 **prompt computation**（现代引擎所说的 prefill），吃算力、要快；再是 token generation(decode)，吃带宽、慢慢来。传统系统把两段混在同一台机器、同一个 batch 里，两边互相将就——从 A100 到 H100,算力涨了 3.43×，带宽只涨 1.64×，显存干脆没动(Table I)；而 conversation 负载的 decode 阶段 60–70% 的时间只跑 ≤20 个活跃 token(Fig.4)，最新 GPU 的算力大部分时间在睡觉。Splitwise 的答案一句话：把两段拆到不同机器上，prompt 机算完，把 KV-cache 经 InfiniBand 转给 token 机继续生成，各用各的硬件、各做各的调度。 由此换来：同功率同成本下吞吐提升 2.35×(Abstract/§I/Conclusion)，以及吞吐 1.4× 且成本低 20%(Abstract/§I)。代价是 KV-cache 转移这个新开销——论文用 per-layer 异步转移把它压到 prompt 计算时间的 7% 以内，端到端只多付 0.8%。本文先把三笔账算清，再拆三池架构、两级调度与转移机制，最后对照 2026 年的 vLLM/Dynamo/Mooncake 看看 PD 解耦今天走到了哪。

<!-- more -->

## 📑 目录

- [🗺️ 原文阅读地图](#️-原文阅读地图)
- [0. 读前 3 分钟：先把硬件账、负载账和时间账算清](#0-读前-3-分钟先把硬件账负载账和时间账算清)
- [1. 问题：一个请求的两段旅程，需求完全相反](#1-问题一个请求的两段旅程需求完全相反)
- [2. 核心思想：阶段拆分，两段旅程各配一台机器](#2-核心思想阶段拆分两段旅程各配一台机器)
- [3. 两级调度:CLS 路由,MLS 批处理](#3-两级调度cls-路由mls-批处理)
- [4. 真正的工程难点:KV-cache 转移](#4-真正的工程难点kv-cache-转移)
- [5. 设计空间：机型配对与拆分比例](#5-设计空间机型配对与拆分比例)
- [6. 端到端收益：数字账与三种口径](#6-端到端收益数字账与三种口径)
- [7. 局限与选型边界](#7-局限与选型边界)
- [8. 从论文到 2026 年的工程:PD 解耦现状](#8-从论文到-2026-年的工程pd-解耦现状)
- [9. 面试官视角：三问三答](#9-面试官视角三问三答)
- [📝 总结](#-总结)
- [🎯 自我检验清单](#-自我检验清单)
- [📚 参考资料](#-参考资料)

## 🗺️ 原文阅读地图

这篇论文既是“洞察论文”（对两阶段资源需求的量化表征）也是“系统论文”（拆分架构 + 调度 + 转移实现 + 设计空间搜索）。本文选择性精讲如下，避免把一篇中文解读误当成原文逐段翻译。

| 原文单元 | 处理深度 | 本文位置与理由 | 来源锚点 |
| --- | --- | --- | --- |
| Abstract / §I（2.35×、1.4× 承诺） | 精讲 | 第 6 节，先立条件再谈收益，含口径纪律 | Abstract;§I;p.1–2 |
| §II Background（两阶段、batching、硬件与网络） | 简述 | 第 0、1 节，读者已有 serving 基础，只保留 Table I 硬件账 | §II;Table I;p.1–3 |
| §III Characterization（Fig.3–9 六个 Insight） | 精讲 | 第 0、1、5 节，全部数字账与硬件不对称证据在这里 | §III-A~G;Fig.3–9;p.3–5 |
| §III-G / Table IV（A100 vs H100 单机对比） | 精讲 | 第 1 节，decode 不需要新算力的直接证据 | Table IV;§III-G;p.5 |
| §IV-A/B 两级调度 CLS/MLS | 精讲 | 第 3 节，机制卡 2 | §IV-A/B;p.6–7 |
| §IV-C KV-cache 转移与 per-layer 优化 | 精讲 | 第 4 节，机制卡 3,全文最值得抄的工程点 | §IV-C;Fig.11;p.7 |
| §IV-D 设计空间搜索与异构设计 | 精讲 | 第 5 节，原文无闭式公式，以机制清单与数字账代替 | §IV-D;Fig.12;p.7–8 |
| §V 实现与性能模型 | 简述 | 第 4、5 节，只保留 MSCCL++ 与 MAPE <3% | §V;p.8–9 |
| §VI Evaluation(Fig.14–20) | 精讲（数字表） | 第 4、6 节，所有端到端数字集中呈现 | §VI-A~E;Fig.14–20;p.9–11 |
| §VII Discussion / §IX Conclusion | 精讲（局限） | 第 7、8 节，选型边界与 1.76× 来自这里 | §VII;p.12;§IX;p.12 |
| Related Work / References 全量谱系 | 不展开 | 只做定位，不影响本文机制主线 | §VIII;References |

📌 **本文承诺**：读完后，你应该能说清 2.35×、两种 1.4×、2.15×、1.18×、1.76× 各自的比较对象与条件，画出 prompt/token/mixed 三池的请求流转，复述 KV 转移的 <7% / ~8ms / ~5ms / 0.8% / +16.5% 数字账，并解释为什么“power cap 50% 对 token 阶段几乎无影响”能支撑异构与功率封顶设计。

## 0. 读前 3 分钟：先把硬件账、负载账和时间账算清

### 0.1 硬件账：A100 到 H100,算力涨 3.43×,显存没动

论文的出发点是一张硬件对比表(Table I,p.1)：

| 指标 | A100 | H100 | 比值 |
| --- | ---: | ---: | ---: |
| 算力 (TFLOPs) | 19.5 | 66.9 | **3.43×** |
| HBM 带宽 (GBps) | 2039 | 3352 | **1.64×** |
| HBM 容量 (GB) | 80 | 80 | **1.00×**（原文：no increase） |
| 功率 (W) | 400 | 700 | **1.75×** |
| 机器成本 ($/hr) | 17.6 | 38 | **2.16×** |

（成本来源 CoreWeave [5];同一张表里 NVLink 50→100 Gbps、InfiniBand 200→400 GBps,都是 2×。）

算力增长(3.43×)远快于带宽增长(1.64×)，显存容量零增长。**这意味着：如果你的负载只吃带宽不吃算力，换新 GPU 的钱基本白花。** 做个除法感受一下失衡（本文自设例子，数字来自 Table I）：每 GBps 带宽摊到的算力，A100 是 19.5/2039 ≈ 9.6 GFLOP,H100 是 66.9/3352 ≈ 20.0 GFLOP——同样的带宽只能喂饱一半的算力。后面你会看到，decode 恰恰就是那种“只吃带宽”的负载。

### 0.2 负载账：decode 阶段的 GPU 大多数时间在“空转”

论文用两个 Azure 生产服务的 trace（coding 与 conversation,2023-11-11 采集 20 分钟，§III）做了表征。Fig.4 画的是：**机器运行不同活跃 token 数的累计时间占比**。注意计数约定(§III-B,p.4)：prompt 阶段 100 个 token 按 100 计，decode 阶段按 1 计(beam=1)——所以“活跃 token 数”直接反映 batch 占用。

<img src="/AIInfraGuide/images/splitwise-fig4-batched-token-cdf.png" alt="Splitwise 论文 Fig.4:混合连续 batching 下机器运行不同活跃 token 数的累计时间分布，Coding 与 Conversation 两个负载的 CDF 曲线，conversation 60% 到 70% 的时间只跑不超过 20 个活跃 token,coding 超过 20% 的时间只有 1 个 token" style="max-width: 98%; display: block; margin: 0 auto;" />

*图源：Splitwise 论文 Figure 4(MICRO 2024,arXiv:2311.18677)*

两条结论(§III-B Insight II,p.4)：

- **conversation 服务 60–70% 的时间只跑 ≤20 个活跃 token**;
- **coding 服务 >20% 的时间只跑 1 个 token**("runs with a single token for more than 20% of the time")。

混合连续 batching 下，decode 阶段的批次稀疏到这种程度，GPU 的 compute 资源利用率自然极低。论文没有给 SM 利用率百分比（账本 §8 注明原文未提供），就用“活跃 token 数”与功率归一化(Fig.8)来表达。

### 0.3 时间账：一个 prompt token 只值 1/250 个 output token

第三个关键数字是两阶段的时间不对称(§III-C Insight III,p.4)：**BLOOM-176B 上，1500 个 prompt token 的耗时 ≈ 6 个 output token 的耗时。** 也就是说一个 prompt token 平均只花一个 output token 的 6/1500 = 1/250（本文自设换算）。

含义：**对大多数请求，端到端时间的大头在 token 生成阶段**——prompt 阶段是“贵但短”的尖峰：它短，所以 TTFT 要快；它贵（单位时间烧算力），所以批处理收益大。原文没有给出“prefill 占 X% / decode 占 Y%”的统一百分比（账本 §8），这个 1500≈6 的等价关系就是替代它的锚点。

### 0.4 实验设置快速对齐

- **模型**：BLOOM-176B（70 层，hidden 14336,112 heads）与 Llama2-70B（80 层，8192,32 heads）(Table III,p.3);
- **默认平台**：8×H100 + vLLM,tensor parallelism 横跨 8 卡、限制在单机内(§III,§II-E,p.3);
- **负载**：两个 Azure LLM 服务 trace(coding / conversation),2023-11-11 采集、20 分钟，发布子集见 AzureLLMInferenceDataset2023(§III,p.3);median prompt:coding 1500、conversation 1020 tokens;median output：13、129 tokens(§III-A Fig.3,p.3)；表征实验用 2 req/sec 缩放 trace(§III-B,p.4);
- **硬件实验**：2×DGX-A100 + 2×DGX-H100 Azure VMs,InfiniBand(§V-A,p.8)；云端 InfiniBand 带宽 25–50 GBps per GPU pair(§II-F,p.3)，实验机 H100 400 Gbps = A100 200 Gbps 的 2 倍(§V-A,§VI-A,p.8–9)。

## 1. 问题：一个请求的两段旅程，需求完全相反

### 1.1 先对齐术语：prompt computation 与 token generation

论文通篇使用 **prompt computation**（处理整个输入 prompt、产出第一个 token）与 token generation（逐个生成后续 token）两个词——这就是现代引擎所说的 prefill / decode。本文叙述论文事实时沿用原文术语，只在首现处标注现代叫法，不把 vLLM PD 模式等后续工程的词倒灌成论文原话。

指标按原文 Table II 定义：TTFT（time to first token,首个 token 时延）、TBT（time between tokens,相邻 token 间隔）、E2E（整请求端到端时延）。

两阶段的资源画像：

| 阶段 | 计算特征 | 瓶颈 | 理想硬件 |
| --- | --- | --- | --- |
| prompt computation | 短而尖的 compute 爆发，可并行 | 算力(FLOPS) | 最新 GPU,越多越好 |
| token generation | 长尾的逐 token 序列化，每步很小 | 内存带宽(HBM) | 带宽够即可，算力过剩 |

### 1.2 混合 batching 是将就，不是最优

Orca 式 mixed continuous batching 把 prompt 和 token 塞进同一批，看似填满了 GPU,实则两阶段互相拖累(§III-D Insight IV,p.4)：

- **prompt 批总量 ≥2048 tokens 后吞吐反而下降**(Fig.6(a))——2048 对应 median prompt 下 batch<2,批已经太稀；
- **token 阶段吞吐随 batch 增长，直到 batch 64 时内存耗尽**(Fig.6(b))——“keeps increasing with batching until 64 batch-size, at which point, the machine runs out of memory”；
- 同时 TBT 对 batching 并不敏感：batch=64 时 TBT 只有 2× 影响(§III-C Fig.5(b),p.4)。

把这三条放在一起读：prompt 批想大、被 2048 封顶；token 批想大、被内存封顶；而 token 的时延指标又不怎么怕批大。**混合批是“一锅乱炖”——两种菜都做不极致。** 另外论文实现基于 vLLM,但 vanilla vLLM 只有带 preemption 的 continuous batching、TBT 更高，论文自行补了 mixed continuous batching(§II-D,§VI)。

### 1.3 单机对比：decode 根本不需要最新的 GPU

最直接的反直觉证据是单机对比（Table IV,§III-G,p.5,Llama-70B,无 batching,P50）：

| 指标 | Coding A100 / H100（H100 比值） | Conversation A100 / H100（H100 比值） |
| --- | ---: | ---: |
| TTFT | 185 / 95 ms(0.51×) | 155 / 84 ms(0.54×) |
| TBT | 52 / 31 ms(0.70×) | 40 / 28 ms(0.70×) |
| E2E | 856 / 493 ms(0.58×) | 4957 / 3387 ms(0.68×) |
| Cost | $0.42 / $0.52(1.24×) | $2.4 / $3.6(1.5×) |
| Energy | 1.37 / 1.37 Whr(1×) | 7.9 / 9.4 Whr(1.2×) |

读这张表的关键：**TBT 只慢约 30%(0.70×)，而 TTFT 差到约一半(0.51×/0.54×)。** E2E 的劣化集中在 prompt 主导的 coding 负载上。结论就是 Insight VII(§III-G,p.5)：token 生成阶段是 memory-bound,不依赖最新 GPU 的算力——这为后面的异构设计（A100 当 token 机、功率封顶的 H100 当 token 机）提供了全部依据。

## 2. 核心思想：阶段拆分，两段旅程各配一台机器

### 2.1 三池架构与请求流转

Splitwise 把一次推理请求的两阶段分配到**不同机器**上执行(§IV,p.6–7)：prompt 机算完 prompt、生成第一个 token 并产生 KV-cache,把 KV-cache 经 InfiniBand 转给 token 机，由 token 机继续逐 token 生成直到结束。

<img src="/AIInfraGuide/images/splitwise-fig10-phase-split-architecture.png" alt="Splitwise 论文 Fig.10:系统架构图，CLS 集群调度器连接 Prompt 池、Token 池与 Mixed 池三组机器，每台机器上有 MLS 机器级调度器与 GPU,机器之间以 InfiniBand 相连，箭头标示 KV-cache 从 prompt 机转移到 token 机" style="max-width: 95%; display: block; margin: 0 auto;" />

*图源：Splitwise 论文 Figure 10(MICRO 2024,arXiv:2311.18677)*

沿 Fig.10 走一轮：

1. CLS(cluster-level scheduler)把请求路由到 prompt 机；
2. prompt 机完成 prompt computation,生成第一个 token 与整份 KV-cache;
3. KV-cache 经 InfiniBand 转给指定的 token 机（与后续计算重叠，见第 4 节）；
4. token 机从第一个 token 开始继续 token generation,直到请求结束；
5. 机器按需伸缩：**mixed pool** 的机器保留原身份，队列清空后归位，切换无感知时延(§IV)。

**机制卡 1：阶段拆分(phase splitting)**

| 项目 | 内容 |
| --- | --- |
| 要解决什么 | 两阶段硬件需求相反，混在一台机器上只能互相将就，算力/带宽利用率双低 |
| 最小前置 | 理解两阶段各自瓶颈（prompt 吃算力、token 吃带宽）；KV-cache 是唯一需要跨机传的状态 |
| 输入 → 状态 → 输出 | 输入：请求到达；状态：三池机器占用、每请求 KV 的位置；输出：prompt 机 / token 机指派 |
| 因果步骤 | ① CLS 把请求路由到 prompt 机 → ② prompt 机算完 prompt 与首 token,产生 KV-cache → ③ KV-cache 经 InfiniBand 转给 token 机 → ④ token 机从首 token 继续逐 token 生成 → ⑤ mixed pool 按需伸缩吸收负载波动 |
| 公式语义 | 原文未给闭式公式；拆分比例（每池几台机器）由设计空间搜索决定（见机制卡 4） |
| 最小例子 | Splitwise-HH 的 cost-optimal 配置 = 27 台 prompt 机 + 3 台 token 机，70 RPS(Fig.12,§IV-D,p.7) |
| 边界与来源 | 依赖 InfiniBand 级网络；小 prompt 的转移不可完全隐藏(§IV-C,§VI-A);mixed 机切换身份无感知时延(§IV,p.6–7) |

### 2.2 快餐区和慢炖区：mixed pool 是机动摊位

> 打个比方：prompt 机像快餐窗口——接单快、出餐快，一份做完立刻传走，窗口翻台率决定流量上限(compute-bound);token 机像慢炖锅——每桌一个灶眼慢慢咕嘟，主要占着灶台（带宽）而不是大厨（算力）；mixed pool 是机动摊位，忙时去帮快餐、闲时回慢炖，摊主身份不变、摊位可以随时切换。机制一一映射：快餐窗口对应 prompt 机（KV 立即转走、机器立刻接下一单），慢炖锅对应 token 机（memory-bound、每台能扛的并发高），机动摊位对应 mixed 机（按需伸缩，切换无感知时延）。

拆开之后，每个阶段都能做**独立的最优资源管理**：token 机可以纯 decode batching,吞吐一路涨到内存上限(batch 64);prompt 机可以纯 prompt batching,把批压到 2048 的拐点以内；两个阶段还能用不同型号的硬件——这就是第 5 节的设计空间。

## 3. 两级调度：CLS 路由，MLS 批处理

### 3.1 CLS：池管理、JSQ 与跨池借调

CLS（cluster-level scheduler,集群级调度器）负责三件事(§IV-A,p.6–7)：

1. **池管理**：维护 prompt / token / mixed 三池的机器状态；
2. **路由**：用 JSQ(Join the Shortest Queue)选机器，队列长度按 pending tokens 计，而不是按请求数；
3. **重叠**：同时指派 prompt 机与 token 机，让 KV 转移与计算重叠。

队列超阈值时，先查 mixed pool,再**跨池借调**——例如把 token 机临时拉来跑 prompt(§IV-A)。负载变化的效果在 Fig.17(§VI-B,p.10–11)里看得最清楚：低负载 70 RPS 时，40 台 Baseline-H100 有 70% 的时间只跑 ≤15 个活跃 token；高负载 130 RPS 时，混合池把负载拉平，批次不再稀疏。

### 3.2 MLS:2048 封顶的 prompt 批与尽力而为的 decode 批

MLS（machine-level scheduler,机器级调度器）按机型分策略(§IV-B,p.7)：

- **prompt 机**：FCFS + 2048-token 批上限(正是 Fig.6(a) 的吞吐拐点);
- **token 机**：FCFS + 尽力 batch 到内存上限（正是 batch 64 的边界）；
- **mixed 机**：为 TTFT SLO 抢占 token 工作——按年龄(age)提权，并限制抢占次数防止饿死。

**机制卡 2：两级调度(CLS + MLS)**

| 项目 | 内容 |
| --- | --- |
| 要解决什么 | 拆池之后，谁去哪台机器、机器内部怎么批、波动时怎么办 |
| 最小前置 | 三池架构；TTFT/TBT/E2E 九个 SLO(Table VI) |
| 输入 → 状态 → 输出 | 输入：请求与 SLO;状态：各池队列长度（按 pending tokens）、mixed 池可用性；输出：机器指派与批策略 |
| 因果步骤 | ① CLS 按 pending tokens 的 JSQ 选 prompt 机 → ② 同时预留 token 机以重叠转移 → ③ 超阈值先查 mixed pool、再跨池借调 → ④ MLS 按机型各自批处理（2048 上限 / 内存上限） → ⑤ mixed 机为 TTFT 抢占并限次防饿死 |
| 公式语义 | 无；路由与批决策都是启发式（JSQ、FCFS、年龄提权） |
| 最小例子 | 70 RPS 低负载：40 台 H100 有 70% 时间 ≤15 活跃 token;130 RPS 高负载：混合池拉平(Fig.17) |
| 边界与来源 | CLS 调度开销对 LLM 长请求可忽略，但大集群下 CLS 可能成为瓶颈(§IV-E,p.8) |

## 4. 真正的工程难点：KV-cache 转移

### 4.1 转移为什么是主要开销

阶段拆分引入的新开销只有一个：KV-cache 跨机转移。KV-cache 大小与 prompt token 数成正比（原文未给绝对字节数，Fig.14 以时延呈现；账本 §8 注明）。最朴素的**串行方案**是：prompt 阶段完成 → 转移 → decode,把转移时延直接加进第二 token 与 E2E——第二 token 时延因此暴涨 +64%(§VI-A Fig.15,p.9)。

### 4.2 per-layer 异步转移：边算边传

论文的优化(§IV-C,§VI-A;p.7,p.9)是 **per-layer 异步转移**：每算完一层，就把该层的 KV 块通过 InfiniBand 异步 put 出去，与下一层的计算重叠(Fig.11(b))；层间做细粒度同步保证正确性。这样转移时延被“藏”在计算后面，只剩一个常数尾巴。小 prompt（<512 tokens,H100 语境）改用串行转移——总 KV 太小，不值得为 per-layer 的同步与干扰付工程成本(§IV-C,§VI-A)。

<img src="/AIInfraGuide/images/splitwise-fig14-kv-transfer-overhead.png" alt="Splitwise 论文 Fig.14:KV-cache 转移时延随 prompt token 数(0 到 2048)的变化，串行转移随 KV 大小线性增长，per-layer 异步转移把不可重叠时延压到近似常数，A100 约 8ms、H100 约 5ms,总开销小于 prompt 计算时间的 7%" style="max-width: 95%; display: block; margin: 0 auto;" />

*图源：Splitwise 论文 Figure 14(MICRO 2024,arXiv:2311.18677)*

Fig.14 的形态：串行转移时延随 KV 大小线性增长；per-layer 后不可重叠部分变成**近似常数**——A100 约 8ms、H100 约 5ms（H100 带宽 2×,故约 2× 快），而总开销不到 prompt 计算时间的 7%(§VI-A,p.9)。

实现层面(§V-A,p.8)：用 **MSCCL++ 的 zero-copy one-sided put** + 每请求独立 semaphore（同一 InfiniBand 连接）；vLLM 中按 block 发送，利用块的连续性。

### 4.3 数字账：7%、8ms/5ms、0.8%、16.5% 对 64%

| 数字 | 含义 | 锚点 |
| --- | --- | --- |
| **<7%** | 转移总开销 vs prompt 计算时间 | §VI-A Fig.14,p.9 |
| **~8ms / ~5ms** | per-layer 后不可重叠时延，A100 / H100,近似常数 | §VI-A Fig.14,p.9 |
| **≤3% vs 0.8%** | 串行转移对 E2E 的影响最多 3%;Splitwise 仅 **0.8%**（coding trace,2 机、无 batching） | §VI-A Fig.15,p.9 |
| **+64% vs +16.5%** | 第二 token 时延增幅：串行转移 vs Splitwise——"only visible impact is the latency for the second token" | §VI-A Fig.15,p.9 |
| **10×** | 讨论中：KV 转移带宽即使降 10× 仍可能有利（可配 KV 压缩 [55]） | §VII,p.12 |

最小数字例子（本文自设）：设一次 prompt 计算耗时 100ms(H100)，则转移总开销 <7ms,其中约 5ms 不可重叠——转移几乎被计算盖住；如果改用串行转移，同样规模的 KV 转移时延随 KV 大小线性增长，prompt 算完还要再等数十毫秒量级，第二 token 时延自然暴涨。

**机制卡 3:KV-cache per-layer 异步转移**

| 项目 | 内容 |
| --- | --- |
| 要解决什么 | 跨机转移是拆分引入的新开销；串行转移直接抬高第二 token 时延与 E2E |
| 最小前置 | KV-cache 概念；RDMA / one-sided put 的异步语义 |
| 输入 → 状态 → 输出 | 输入：每层算好的 KV 块；状态：每请求 semaphore 与共享 InfiniBand 连接；输出：token 机可直接消费的 KV |
| 因果步骤 | ① prompt 机算完第 L 层 → ② 该层 KV 块异步 put 到 token 机 → ③ 计算第 L+1 层与传输重叠 → ④ 层间细粒度同步保序 → ⑤ 全部层完成后 token 机直接开始 decode |
| 公式语义 | 原文未给闭式公式；开销以实验曲线表征(Fig.14) |
| 最小例子 | <512 tokens(H100)走串行；per-layer 后不可重叠 ~5ms(H100) |
| 边界与来源 | 转移时延与 KV 大小成正比，per-layer 重叠后只剩常数尾巴(§VI-A)；小 prompt 的 per-layer 干扰风险用串行规避(§IV-C)；总开销 <7% prompt 计算时间(§VI-A Fig.14) |

## 5. 设计空间：机型配对与拆分比例

### 5.1 Splitwise-XY 记法与硬件不对称

论文用 **Splitwise-XY** 记法表示设计：X = prompt 池机型，Y = token 池机型；A = A100,H = H100,Hcap = 功率封顶的 H100（每 GPU cap 50%、整机 70% 额定功率）。(P, T) 表示 prompt 池 P 台 + token 池 T 台(§IV-D,p.7–8)。

Table V(p.8)把各机型的成本/功率做了归一化：H100 机成本约为 A100 机的 **2.35×**、功率 1.75×;token 池配 H100 的成本系数 2.5×。

⚠️ **注意**：2.35× 在原文出现两处、语义完全不同——一处是 Abstract/§I/Conclusion 的吞吐倍数，一处是 Table V 的 H100 机成本归一化。别混。（Table I 按 CoreWeave 列表价算出的成本比是 2.16×,与 Table V 的 2.35× 又是两个口径，各记各的锚点。）

为什么敢用 A100 当 token 机？就是第 1.3 节的 Insight VII:TBT 只慢 ~30%,而 token 机省钱省功率。Table V 的账算下来，异构(HA)与功率封顶(HHcap)在成本/功率上都有明确动机。

### 5.2 没有闭式公式：性能模型 + 模拟器 + 设计空间搜索

⚠️ **重要口径：原文没有给出“拆分比例 = f(……)”的闭式公式。** 拆分决策不是解公式，而是“性能模型 + 集群模拟器 + 设计空间搜索”三件套(§IV-D,§V-B;p.7–9)：

1. **分段线性性能模型**(piece-wise linear,§V-B,p.9)：在 A100/H100 上按 （batch size, #prompt tokens, #output tokens, 并行度） 打点、插值建表，输出 TTFT/TBT 估计；MAPE <3%(80:20 train:test);
2. **事件驱动集群模拟器**(SplitwiseSim,§V-B,Fig.13)：输入 = request trace、九个 SLO、性能模型、集群与调度配置；输出 = 每请求 TTFT/TBT/E2E 百分位 + 机器利用率；模拟器用 >50K iterations 做端到端验证；
3. **设计空间搜索**(§IV-D,Fig.12)：目标函数 ∈ {throughput, cost, power},约束 = 九个 SLO 与吞吐，决策变量 = (P, T)。示例：iso-throughput cost-optimized 的 Splitwise-HH = (27P, 3T),70 RPS(Fig.12)。

九个 SLO 的定义(Table VI,§V-B,p.9)——全部相对 **A100 无竞争时延**的倍数：

| 指标 | P50 | P90 | P99 |
| --- | ---: | ---: | ---: |
| TTFT | 2× | 3× | 6× |
| TBT | 1.25× | 1.5× | 5× |
| E2E | 1.25× | 1.5× | 5× |

**九个 SLO 全部满足，一个设计才算达标。** 后面所有收益数字都默认这个前提。

**机制卡 4：设计空间搜索（性能模型 + 模拟器）**

| 项目 | 内容 |
| --- | --- |
| 要解决什么 | 拆分没有解析公式：机型配对、池大小、功率分配都要在满足 SLO 的前提下求最优 |
| 最小前置 | 分段线性性能模型(MAPE <3%)；事件驱动模拟器；九个 SLO 定义 |
| 输入 → 状态 → 输出 | 输入：trace、SLO、候选机型；状态：性能表 + 模拟器配置；输出：(P, T) 最优配置与吞吐/成本/功率 |
| 因果步骤 | ① 在真实硬件打点建性能表 → ② 模拟器按 trace 跑调度与排队 → ③ 在 (P, T) 网格上搜索目标函数 → ④ 用九个 SLO 过滤不可行解 → ⑤ 输出 Pareto 面上的配置 |
| 公式语义 | 原文未给闭式公式；目标函数与约束以清单形式给出(throughput/cost/power vs 9 SLO) |
| 最小例子 | iso-throughput cost-optimized Splitwise-HH → (27P, 3T),70 RPS(Fig.12) |
| 边界与来源 | 模型假设 InfiniBand 与给定硬件谱系；搜索结果是特定 trace 与 SLO 下的最优，不能外推（§VI-D 负载漂移回退 7%） |

### 5.3 功率封顶：decode 机可以“降频不降速”

支撑 Hcap 设计的证据来自功率表征(§III-F,p.5)：

- **power cap 50%(700→350W)对 token 阶段时延几乎无影响；prompt 阶段则高度敏感**(Insight VI,Fig.9);
- prompt 阶段功率随 batch 增大，token 阶段功率不变(Fig.8)——token 阶段 memory-bound 的直接证据。

于是 **Splitwise-HHcap** 把 token 机每 GPU 功率封顶 50%（整机 70% 额定功率）：decode 不受影响，省下的功率预算可以加给 prompt 机或直接降总功率(§IV-D)。

## 6. 端到端收益：数字账与三种口径

### 6.1 三种口径：iso-power、iso-cost、iso-throughput

论文报告收益时用三种预算口径(§VI-B/C,p.9–11)：

- **iso-power（等功率）**：给定功率预算，比吞吐与成本；
- **iso-cost（等成本）**：给定成本预算，比吞吐与功率；
- **iso-throughput（等吞吐）**：给定吞吐与 SLO,比成本与功率。

基线：Baseline-H100 = **40 台**(40P/T),Baseline-A100 = 70 台(70P/T)，功率预算相同(§VI-B,p.9)。所有数字都默认九个 SLO 全部满足。

### 6.2 主结果表：2.35×、两种 1.4×、2.15×、1.18×、1.76×

| 收益 | 条件与比较对象 | 锚点 |
| --- | --- | --- |
| **2.35×** 吞吐 | **同功率同成本**，相对现有设计 | Abstract p.1;§I p.2;Conclusion p.12 |
| **1.4×** 吞吐 | 相对现有设计，且**成本低 20%** | Abstract p.1;§I p.2 |
| **1.4×** 吞吐 | **iso-cost** 下 Splitwise-AA vs Baseline-H100,代价是多耗 **25% 功率、2× 空间** | §VI-C,Fig.18(b),p.10 |
| **2.15×** 吞吐 | Splitwise-AA vs Baseline-A100,**iso-power**,conversation trace | §VI-B Fig.18(a),p.10 |
| **1.18×** 吞吐 | Splitwise-HA,**同功率**，成本低 **10%** | §VI-B Fig.18(a),p.10 |
| **1.76×** 吞吐 | **同成本、功率低 15%**，且满足性能 SLO | Conclusion p.12 |
| 同吞吐，功率低 **25%** | Splitwise-HHcap vs Baseline-H100(iso-throughput,power-optimized) | §VI-C Fig.19(a),p.11 |
| 同吞吐，成本低 **25%** | Splitwise-AA vs Baseline-H100(iso-throughput,cost-optimized) | §VI-C Fig.19(b),p.11 |

⚠️ **两种 1.4× 的语境不同，必须分开写**：第一种来自 Abstract/§I,是相对现有设计的“吞吐 1.4× 且成本低 20%”；第二种是 iso-cost 预算下 Splitwise-AA 相对 Baseline-H100 的 1.4× 吞吐，代价是多耗 25% 功率和 2× 空间(Fig.18(b))。别把“便宜 20%”安到第二种头上。

### 6.3 集群怎么配：(P, T) 的具体数字

<img src="/AIInfraGuide/images/splitwise-fig18-throughput-summary.png" alt="Splitwise 论文 Fig.18:吞吐优化集群汇总，双面板水平条形图，(a) iso-power 下各设计的服务器数、吞吐与成本归一化，Splitwise-AA 相对 Baseline-A100 吞吐 2.15 倍、Splitwise-HA 1.18 倍且成本低 10%;(b) iso-cost 下 Splitwise-AA 相对 Baseline-H100 吞吐 1.4 倍，多耗 25% 功率与 2 倍空间" style="max-width: 98%; display: block; margin: 0 auto;" />

*图源：Splitwise 论文 Figure 18(MICRO 2024,arXiv:2311.18677)*

iso-power 下各设计的池配置（§VI-B,Fig.18 图例，p.10–11）：

- **coding**：Splitwise-AA(55P, 15T)，比 Baseline-H100 多 75% 机器；Splitwise-HH(35P, 5T);
- **conversation**：AA(45P, 25T)、HH(25P, 15T)、HA(25P, 26T)、HHcap(25P, 21T)。

iso-cost 下(Fig.18(b) 图例，p.10–11)：AA(51P, 35T)、HH(25P, 15T)、HA(30P, 21T)、HHcap(30P, 10T);Baseline-A100(86P/T)、Baseline-H100(40P/T)。

iso-throughput 下（Fig.19 图例，p.11）：Baseline-A100(88P/T)、Baseline-H100(24P/T);AA(25P, 16T)、HH(5P, 17T);HA 在 Fig.19(a)/(b) 分别为 (21P, 1T)/(11P, 19T),HHcap 分别为 (8P, 16T)/(19P, 3T)。

注意 pattern:cost-optimal 的 HH 只需 3 台 token 机——**token 机吃带宽不吃算力，每台能扛的并发远高于 prompt 机**；而 conversation 的 token 池（15–26 台）显著大于 coding（5–15 台），因为长输出负载的 decode 才是主线。另外 Splitwise-HA 在 >90 RPS 时会把 H100 机器拉进 mixed pool 以降低 TBT P50(§VI-B,p.10)。

### 6.4 两个可复算的最小数字例子

**例子 A:iso-power 的预算自洽检查（本文自设例子，数字来自 §VI-B 与 Table I）。** 40 台 H100 × 700W = 28 kW;70 台 A100 × 400W = 28 kW——两个基线在功率预算上确实相等。Splitwise 用同样的 28 kW 配出 (55P, 15T) 等混合配置，把功率从“全给 prompt 机”挪到“按阶段分配”，这就是 2.15×/2.35× 的功率侧来源。

**例子 B：时间等价换算（本文自设例子，基于 §III-C Insight III 与 §III-A Fig.3 的 median）。** 由"1500 prompt tokens ≈ 6 output tokens"得 1 个 output token ≈ 250 个 prompt token 的耗时。

- coding(median 1500 prompt + 13 output)：prompt 耗时 ≈ 6 个 output 当量，占 E2E ≈ 6/(6+13) ≈ **32%**;
- conversation(median 1020 prompt + 129 output)：prompt 耗时 ≈ 4.1 个 output 当量，占 E2E ≈ 4.1/(4.1+129) ≈ **3%**。

与 Insight III“大多数 E2E 时间花在 token 生成”一致。推论：把 prompt 挪到更快的机器上，对 conversation 的 E2E 几乎无感，但对 coding 的 TTFT 影响显著——**这也解释了为什么 conversation 负载上拆分收益更大（AA 2.15× vs HA 1.18×,均为 conversation trace 结果）**。

### 6.5 口径纪律：别把条件丢了

- **2.35× / 1.4× 是吞吐，不是单请求时延加速**；前提是九个 SLO 全部满足；
- **2.35× 的准确值是 2.35×**——不少二手资料写成“2.3×”，原文 Abstract/§I/Conclusion 三处一致；全文没有 “2.9×”（账本 §8 更正）；
- 两种 1.4×、Table V 的 2.35× 成本系数，各记各的语境（见 5.1、6.2）；
- **负载漂移**：conversation trace 跑在按 coding 设计的集群上，HA/HHcap 吞吐回退 7%,AA/HH 不受影响(§VI-D Fig.20,p.11);
- **高负载边界**：batch 场景下拆分收益消失——Baseline-A100 与 Splitwise-AA 同为 0.89 RPS/$,Splitwise-HH 与 Baseline-H100 同为 0.75 RPS/$，退化为 iso-count baseline(§VI-E,p.11)。

## 7. 局限与选型边界

### 7.1 论文自己承认的十条边界

1. **KV-cache 转移是主要开销**：小 prompt 场景转移不可完全隐藏，<512 tokens(H100)改用串行(§IV-C,§VI-A);
2. **网络假设**：所有设计假定 InfiniBand;Splitwise-HA（H100↔A100 异构）在现实中不易获得 IB 直连，替代是 HPC CPU 桥接或 RoCE;带宽降 10× 仍可能有利，但未实测(§VII);
3. **规模化**：CLS 可能成为大集群的调度瓶颈（引分区/复制式调度，正交）(§IV-E);
4. **可靠性/容错**：失败即从头重算，或把 KV-cache checkpoint 到内存数据库；安全高效的恢复设计不在本文范围(§VII);
5. **数据中心碎片化**：异构集群(HA)会给 CSP 带来运营挑战(§VII);
6. **会话级 KV 复用未覆盖**：未来若服务端缓存对话上下文，prompt 内存画像会变，且需要把 KV 传回 prompt 机(§VII);
7. **硬件范围**：仅 A100/H100;T4 显存不足；MI250、Sapphire Rapids(HBM)只作展望(§VII);
8. **工作负载变化**：HA/HHcap 遇负载漂移回退 7%;高负载 batch 场景退化为 iso-count baseline(§VI-D/E);
9. **功率口径**：只考虑 provisioned power,不考虑动态功耗(§IV-D);
10. **表征不复用 KV-cache**（模拟安全隔离云），可能低估现实中的缓存收益(§III)。

### 7.2 牺牲什么，换取什么

| 换取 | 牺牲 |
| --- | --- |
| 阶段独立资源管理：decode 机可用便宜硬件（A100 或功率封顶 H100），TBT 只慢 ~30%(Table IV) | KV-cache 转移开销：第二 token +16.5%（vs 串行 +64%），不可重叠 ~8ms/~5ms(§VI-A) |
| 同功率同成本吞吐 2.35×;成本低 20% 时吞吐 1.4×(Abstract/§I) | 需要 InfiniBand 级网络、MSCCL++ per-layer 传输工程与 CLS/MLS 两级调度复杂度(§IV,V) |
| 功率预算灵活：HHcap 同吞吐省 25% 功率(Fig.19(a)) | 负载漂移时 HA/HHcap 吞吐回退 7%(Fig.20) |
| 各阶段 batch 独立最优：prompt 2048 封顶 / decode 到内存上限(§III-D) | 高负载 batch 场景收益消失：0.89 vs 0.75 RPS/$(§VI-E) |

### 7.3 一句话选型规则

> **如果负载 decode 占比高、机房有低延迟高带宽网络、且能接受两套池子与两级调度的运维复杂度，阶段拆分值得；如果只有单机、带宽一般、或负载长期高并发 batch,先别拆——拆分收益在那里会退化，而转移与调度成本是实打实的。**

## 8. 从论文到 2026 年的工程：PD 解耦现状

> 本节工程事实访问日期：**2026-08-11**。论文数字（2.35× 等）是 MICRO 2024 时代在其实验设置下的结论；工程实现细节已大幅演进，下面分层对照，不把后续工程术语倒灌成论文原话。

### 8.1 论文时代：没有官方仓库，只有 PR 和模拟器

- **时间线**：arXiv 2311.18677 v1(2023-11-30),MICRO 2024 正式发表；
- **官方实现**：没有独立官方仓库。论文自引 vLLM PR #2809 "Add Splitwise implementation to vLLM"（2026-08-11 经 GitHub API 核实：closed、未合并）；
- **模拟器开源**：SplitwiseSim（github.com/Mutinifni/splitwise-sim,157★,最后推送 2024-04-25,2026-08-11 核实）——论文实验体系的模拟侧是公开的；
- **实现栈**：基于 vLLM + MSCCL++;tensor parallelism 限制在单机内（8 卡），跨机通信只有 KV 转移。

### 8.2 当前工程：experimental 的 vLLM PD 与生产化的生态

- **vLLM 官方 PD(prefill-decode)disaggregation**：已是内置特性，标注 experimental；文档明确 "Disaggregated prefill DOES NOT improve throughput."——它的价值在独立调 TTFT/ITL 与尾部隔离，而不是自动提吞吐。这和 Splitwise 的叙事并不矛盾：Splitwise 的 2.35× 是“机型配对 + 功率分配 + 两级调度 + SLO 约束”的整体结果，不是“拆开”这一个动作；
- **KV 转移**：vLLM 的 KVTransferConfig + connector 体系——NixlConnector（NVIDIA NIXL,GPU 直传、异步）、MooncakeConnector（KVCacheConnector 系列）；多轮对话支持双向 KV 复用：decode 保留上一轮 KV,prefill 直接拉取，避免重算上下文（论文 §VII 预告的“会话级 KV”如今成为工程特性）；
- **路由与池管理**：vLLM 示例用 proxy 路由 client→prefill→decode;NVIDIA Dynamo 提供生产级 P/D worker pool + KV-aware/topology-aware 路由；Dynamo 官方建议与聚合(aggregation)baseline 对比——短 prompt、小模型、低并发、慢 KV 转移时聚合更优；
- **生态对照**：Mooncake（kvcache-ai/Mooncake,Moonshot Kimi 的生产系统，FAST'25）——prefill/decode 集群分离 + Mooncake Store 分布式 KV 缓存层 + RDMA Transfer Engine,报告 59–498% 有效请求容量提升（依赖 workload 与 baseline）；DistServe(OSDI'24)与 Splitwise 同期独立提出 P/D 分离（arXiv 晚约一个半月，OSDI'24 正式发表在前；phase-specific placement,goodput 视角）；
- **未核实项如实说明**：vLLM/Dynamo/Mooncake 的精确版本号未逐版核实；Mooncake 仓库自述的 "75% more requests" 未复核。

### 8.3 时间差异说明

| 维度 | 2023-11 论文(MICRO 2024) | 2026-08-11 工程 |
| --- | --- | --- |
| 官方代码 | 无独立仓库；vLLM PR #2809 未合并 | vLLM 内置 PD(experimental,"Disaggregated prefill DOES NOT improve throughput.") |
| KV 转移 | MSCCL++ zero-copy one-sided put,逐层异步 | NixlConnector / MooncakeConnector;双向 KV 复用 |
| 路由/池管理 | CLS+MLS 两级、JSQ、mixed pool | proxy 路由；Dynamo P/D worker pool(KV-aware) |
| 生态 | — | Mooncake(FAST'25)、DistServe(OSDI'24)、Dynamo |

## 9. 面试官视角：三问三答

**Q1：“Splitwise 的 2.35× 是怎么来的？条件是什么？”**

答：先立条件——**同功率、同成本、九个 SLO 全部满足**，相对现有设计(Abstract/§I/Conclusion)；准确值是 2.35×,不是二手资料常写的 2.3×,全文也没有 2.9×。来源不是单机基准，而是“性能模型(MAPE<3%)+ 模拟器 + 设计空间搜索”在 (P, T) 网格上求出的最优。吞吐提升来自三件事的叠加：decode 机改用便宜硬件（A100 或功率封顶 H100）、两个阶段各自做最优 batch（prompt 2048 封顶 / decode 到内存上限）、功率预算按阶段重分配。补一句：还有 iso-cost 口径的 1.4×（多耗 25% 功率、2× 空间）和 Abstract 里“吞吐 1.4× 且成本低 20%”，两种 1.4× 条件不同，别混。

**Q2：“KV-cache 转移那么贵，为什么 Splitwise 还能赢？”**

答：贵，但可控。per-layer 异步转移把不可重叠时延压到 ~8ms(A100)/~5ms(H100)的常数，总开销 <prompt 计算时间的 7%,E2E 只多 0.8%,第二 token 时延 +16.5%（串行方案是 +64%）(§VI-A)。这是典型的“牺牲什么/换取什么”：**用一次跨机转移，换整段 decode 的资源独立**——decode 机可以按 memory-bound 特性配最便宜的硬件，整池吞吐/成本大幅改善。小 prompt 转移藏不住，所以 <512 tokens 走串行；带宽降 10× 仍可能有利，但没实测(§VII)。

**Q3：“vLLM 不是早就支持 PD 解耦了吗？为什么文档还说 'Disaggregated prefill DOES NOT improve throughput.'？”**

答：vLLM 的 PD(prefill-decode)disaggregation 是内置特性但标注 experimental,文档的原话是 **"Disaggregated prefill DOES NOT improve throughput."**——单机 PD 分离若不做异构机型、不做阶段独立批优化、不把转移开销用异步重叠吸收，吞吐不会自动提升；它的价值在独立调 TTFT/ITL 与尾部隔离。Splitwise 的 2.35× 是设计空间搜索的整体结果（机型配对 + 功率分配 + SLO 约束），“拆开”只是第一步。工程上今天的 KV 转移走 Nixl/Mooncake connector,路由层有 Dynamo 的 P/D worker pool,还有 Mooncake(FAST'25)与 DistServe(OSDI'24)做生态对照。

## 📝 总结

1. **硬件错配是根源**：A100→H100 算力 3.43×、带宽 1.64×、显存 1.00×、功率 1.75×、成本 2.16×(Table I);compute-bound 的活在新卡上划算，memory-bound 的活换卡白换。
2. **两阶段需求相反**：prompt computation 吃算力，token generation 吃带宽；conversation 负载 60–70% 时间只跑 ≤20 个活跃 token,coding 负载 >20% 时间只有 1 个(Fig.4)——decode 阶段算力利用率极低。
3. **混合 batching 是将就**：prompt 批 2048 封顶、decode 批 64 内存耗尽、TBT 对批不敏感(§III-D)，一锅乱炖两头都做不极致。
4. **阶段拆分**：prompt 机算完把 KV-cache 经 InfiniBand 转给 token 机；三池架构 + mixed pool 按需伸缩，各阶段独立资源管理、允许异构机型(§IV,Fig.10)。
5. **两级调度**：CLS 按 pending tokens 的 JSQ 路由 + 跨池借调；MLS 按机型各自批处理，mixed 机为 TTFT 抢占并限次防饿死(§IV-A/B)。
6. **KV 转移是主要工程难点**：per-layer 异步转移 + MSCCL++ one-sided put,把不可重叠时延压到 ~8ms/~5ms 常数，总开销 <7% prompt 计算时间，E2E 只多 0.8%,第二 token +16.5%（vs 串行 +64%）(§VI-A)。
7. **原文未给闭式公式**：拆分决策 = 分段线性性能模型(MAPE<3%)+ SplitwiseSim 模拟器 + 设计空间搜索；九个 SLO(Table VI)全满足才达标。
8. **收益与条件**：同功率同成本吞吐 2.35×;成本低 20% 时吞吐 1.4×;iso-power conversation 下 Splitwise-AA vs Baseline-A100 为 2.15×、Splitwise-HA 为 1.18×（成本还低 10%）；同成本、功率低 15% 时吞吐 1.76×(Conclusion);HHcap 同吞吐省 25% 功率。
9. **边界诚实**：负载漂移时 HA/HHcap 回退 7%;高负载 batch 退化为 iso-count baseline(0.89 vs 0.75 RPS/$)；只对 InfiniBand 场景成立；可靠性、会话级 KV 复用、动态功耗均不在范围内(§VI-D/E,§VII)。
10. **工程传承**：vLLM PR #2809 未合并，但 PD 已成为 vLLM 内置特性(experimental,"Disaggregated prefill DOES NOT improve throughput.")；转移走 Nixl/Mooncake connector,路由有 Dynamo,生态有 Mooncake(FAST'25)与 DistServe(OSDI'24)。

## 🎯 自我检验清单

- 能复述 Table I 的五组比值(3.43×/1.64×/1.00×/1.75×/2.16×)，并解释为什么 token 阶段不需要最新 GPU。
- 能解释 Fig.4 的计数约定（prompt 100 tokens 计 100,decode 计 1）与“conversation 60–70% 时间 ≤20 活跃 token、coding >20% 时间只有 1 个”的含义。
- 能说出 2048 与 64 两个批上限各自的阶段、原因与锚点(Fig.6(a)/(b))。
- 能画出三池架构的请求流转：CLS 路由 → prompt 机算首 token → KV 转移 → token 机续生成 → mixed pool 按需伸缩。
- 能讲清 CLS 的 JSQ（pending tokens 计长）与跨池借调、MLS 按机型分策略、mixed 机为 TTFT 抢占与限次防饿死。
- 能复述 per-layer 异步转移的机制与数字账：<7%、~8ms/~5ms、0.8% vs 3%、+16.5% vs +64%,并解释小 prompt(<512,H100)为何走串行。
- 能说清“原文未给闭式公式”：拆分决策 = 性能模型(MAPE<3%)+ SplitwiseSim + 设计空间搜索，九个 SLO(Table VI)全满足才算达标。
- 能准确区分 2.35×、两种 1.4×（成本低 20% vs iso-cost 多耗 25% 功率）、2.15×、1.18×、1.76× 各自的比较对象与条件。
- 能复述 HHcap 的设计依据（power cap 50% 对 token 阶段几乎无影响，Fig.9）与收益（同吞吐省 25% 功率，Fig.19(a))。
- 能举出至少四条论文自述局限，并说明负载漂移 7% 回退只影响 HA/HHcap、高负载 batch 下收益消失的原因。
- 能复算例子 A(40×700W = 70×400W)与例子 B（1500≈6 → coding E2E 约 32% 在 prompt）并说明各自依据的锚点。
- 能对照 2026 年工程：vLLM PD 为何标注 experimental、"Disaggregated prefill DOES NOT improve throughput."的含义、Nixl/Mooncake connector 与 Dynamo/Mooncake/DistServe 的定位。

## 📚 参考资料

- 原文：
  - [Splitwise: Efficient Generative LLM Inference Using Phase Splitting(arXiv:2311.18677)](https://arxiv.org/abs/2311.18677)：本文精读对象，MICRO 2024;账本核对 arXiv v2(v1 2023-11-30)。
  - [SplitwiseSim 模拟器(Mutinifni/splitwise-sim)](https://github.com/Mutinifni/splitwise-sim)：论文引用的开源事件驱动模拟器；157★,最后推送 2024-04-25（2026-08-11 核实）。
  - [vLLM PR #2809 "Add Splitwise implementation to vLLM"](https://github.com/vllm-project/vllm/pull/2809)：论文自引的官方实现 PR,closed、未合并（2026-08-11 核实）。
  - [AzureLLMInferenceDataset2023(AzurePublicDataset)](https://github.com/Azure/AzurePublicDataset)：论文生产 trace(coding/conversation)的发布子集。
- 当前工程（访问日期 2026-08-11）：
  - [vLLM PD disaggregation 文档](https://docs.vllm.ai/en/stable/features/disagg_prefill/)：内置 PD 特性，标注 experimental,文档明确 "Disaggregated prefill DOES NOT improve throughput.";KV 转移经 KVTransferConfig 走 NixlConnector / MooncakeConnector。
  - [NVIDIA Dynamo](https://github.com/NVIDIA/Dynamo)：生产级 P/D worker pool + KV-aware/topology-aware 路由；官方建议与聚合 baseline 对比。
  - [Mooncake(kvcache-ai/Mooncake)](https://github.com/kvcache-ai/Mooncake)：Moonshot Kimi 生产系统，FAST'25;prefill/decode 分离 + Mooncake Store + RDMA Transfer Engine;仓库自述数字未复核。
  - [DistServe(LLMServe/DistServe)](https://github.com/LLMServe/DistServe)：OSDI'24,与 Splitwise 同期独立提出 P/D 分离（arXiv 晚约一个半月，OSDI'24 正式发表在前；phase-specific placement,goodput 视角）。
- 站内相关：
  - [7.2 PD 解耦架构设计](/AIInfraGuide/inference/模块四-推理优化/第7章-pd解耦架构/72-解耦架构设计)：本文思想在现代引擎中的落地形态。
  - [7.3 KV 传输与 Connector](/AIInfraGuide/inference/模块四-推理优化/第7章-pd解耦架构/73-kv传输与connector)：Nixl/Mooncake connector 与转移工程的实践细节。
  - [7.4 Goodput 与 SLO 调度](/AIInfraGuide/inference/模块四-推理优化/第7章-pd解耦架构/74-goodput与slo调度)：九个 SLO 与 goodput 视角的工程化。
  - [7.6 vLLM 解耦实战](/AIInfraGuide/inference/模块四-推理优化/第7章-pd解耦架构/76-vllm解耦实战)：当前 vLLM PD 模式的配置与踩坑。
  - [2.2 Continuous Batching 连续批处理](/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/22-continuous-batching)：混合 batching 与 2048/64 批上限的前置知识。
  - [9.1 推理指标体系](/AIInfraGuide/inference/模块四-推理优化/第9章-性能分析与benchmark/91-推理指标体系)：TTFT/TBT/E2E 指标定义与测量口径。
  - [2.1 PagedAttention 分页注意力](/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/21-pagedattention)：KV-cache 块管理是理解转移机制的前置。
