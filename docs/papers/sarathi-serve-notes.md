---
title: "Sarathi-Serve 精读：chunked prefill 让吞吐与延迟不再二选一"
description: "拆解 OSDI 2024 的 Sarathi-Serve：为什么 decode 批是内存受限而 prefill 是计算受限，chunked-prefills 如何把长 prompt 切块塞进 decode 批的空闲算力，stall-free batching 如何消除 generation stall，以及 2.6×/3.7×/5.6× 容量的真实口径。"
pubDate: 2026-08-15
originalUrl: "https://www.usenix.org/conference/osdi24/presentation/agrawal"
sourceType: "paper"
originalAuthor: "Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Alexey Tumanov, Ramachandran Ramjee (Georgia Tech & Microsoft Research India)"
tags: ["Sarathi-Serve", "Chunked Prefill", "推理调度", "生成停顿", "LLM Serving"]
stage: engine
order: 3
prereqs: ["orca-notes", "pagedattention-notes"]
minutes: 40
difficulty: 2
---

> 原文：[Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal)（Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Alexey Tumanov, Ramachandran Ramjee,OSDI 2024,arXiv:2403.02310 v2;本文访问日期 2026-08-15）

LLM 推理请求有两个阶段：**prefill**（一次性处理整个输入 prompt，产生第一个输出 token）和 decode（逐 token 生成）。前者吃算力、后者吃带宽——decode 批天然内存受限，算力闲着；prefill 批天然计算受限，批大小加了也没用（§3.1,Takeaway-1/2）。现有调度器要么“先来先服务地塞 prefill”（vLLM 系，prefill-prioritizing），造成生成停顿（generation stall：一个长 prefill 迭代能打断正在解码的请求好几秒，Fig.1a）；要么“等所有 decode 跑完再进新请求”（FasterTransformer 系，decode-prioritizing），吞吐上不去。Sarathi-Serve 的回答是两招：chunked-prefills（把长 prompt 切成计算量均匀的块，分多个迭代算完）与 stall-free batching（每轮迭代的 token 总量被预算卡住，decode 永不被打断）。效果：Mistral-7B 单卡容量 2.6×、Yi-34B 双卡 3.7×、Falcon-180B 流水线并行 5.6×（摘要）。本文先建立“两类批各自受限”的直觉，再拆 chunked prefill 与 token budget 机制，最后核对容量数字与它在 2026 年工程中的位置。

<!-- more -->

## 📑 目录

- [🗺️ 原文阅读地图](#️-原文阅读地图)
- [0. 读前 3 分钟：两组直觉](#0-读前-3-分钟两组直觉)
- [1. 问题：两类迭代的错配](#1-问题两类迭代的错配)
- [2. 核心机制：chunked-prefills](#2-核心机制chunked-prefills)
- [3. 核心机制：stall-free batching 与 token budget](#3-核心机制stall-free-batching-与-token-budget)
- [4. 为什么 pipeline 气泡也一起解决了](#4-为什么-pipeline-气泡也一起解决了)
- [5. 评测：容量数字逐项核对](#5-评测容量数字逐项核对)
- [6. 与 Orca/vLLM/Splitwise 的关系](#6-与-orcavllmsplitwise-的关系)
- [🕰️ 原文时代 vs 当前工程](#️-原文时代-vs-当前工程)
- [7. 常见误读与错误做法](#7-常见误读与错误做法)
- [📝 总结](#-总结)
- [🎯 自我检验清单](#-自我检验清单)
- [📚 参考资料](#-参考资料)

## 🗺️ 原文阅读地图

Sarathi-Serve 是“动机分析 + 两个机制 + 评测”的标准系统论文结构（7 节）。下表标出会改变本文主张的单元。

| 原文单元 | 处理 | 本文位置/省略理由 | 来源锚点 |
| --- | --- | --- | --- |
| §1 Introduction（Fig.1 生成停顿与尾延迟、Fig.2 权衡示意、2.6×/3.7×/5.6× 承诺） | 精讲 | 开篇与第 1、5 节 | §1;Fig.1-2 |
| §2 Background（2.2 推理过程与 Algorithm 1/2、2.4 TTFT/TBT/Capacity、2.5 调度分类） | 精讲（2.4、2.5） | 第 0、1 节：术语与基线 | §2.2;§2.4;§2.5;Algorithm 1-2 |
| §3 Motivation（3.1 成本分析 Fig.3-6、3.2 权衡 Fig.7、3.3 pipeline 气泡 Fig.8） | 精讲 | 第 1、4 节：四组 Takeaway | §3.1-3.3;Fig.3-8 |
| §4.1 chunked-prefills | 精讲 | 第 2 节：核心机制 | §4.1;Fig.9 |
| §4.2 stall-free batching（Algorithm 3） | 精讲 | 第 3 节：调度算法走查 | §4.2;Algorithm 3 |
| §4.3 token budget（SLO、tile-quantization、Vidur） | 精讲 | 第 3 节：预算怎么定 | §4.3 |
| §4.4 实现（vLLM 之上、FlashAttention v2/FlashInfer） | 简述 | 第 6 节 | §4.4 |
| §5 评测（5.1 容量 Fig.10-11、5.2 权衡 Fig.12、5.3 PP Fig.13、5.4 消融 Fig.14/Table 4） | 精讲 | 第 4、5 节：数字逐项核对 | §5.1-5.4;Table 1-4 |
| §6 Related Work（Splitwise/DistServe 解耦对比） | 简述 | 第 6 节 | §6 |
| §7 Conclusion | 简述 | 开篇与总结 | §7 |

📌 **本文承诺**：读完后，你应该能解释“为什么 decode 批里塞 prefill 块不会明显变慢”（算术强度差）、能按 Algorithm 3 走一遍调度决策（预算怎么分配），并说清 token budget 的 512/2048 两种配置在严格/宽松 SLO 下各自的取舍。

## 0. 读前 3 分钟：两组直觉

**第一组：同一张卡，两种活，一种闲算力、一种闲带宽。** decode 迭代一次只处理每请求一个 token：从 HBM 搬权重的时间远大于计算时间（内存受限，arithmetic intensity 算术强度低，Fig.5），GPU 的计算单元大部分时间在等数据。prefill 迭代一次处理几百上千个 prompt token：计算饱和（计算受限），但搬一次权重摊到很多 token 上。关键推论（Takeaway-2）：decode 批内存受限意味着“往 decode 批里多加一些 token，几乎不增加它的执行时间”——空闲算力可以白捡。

**第二组：调度器在“吞吐”和“尾延迟”之间被逼二选一。** 把新请求的 prefill 立刻插进批（prefill-prioritizing，vLLM/Orca 的做法）→ decode 被长 prefill 打断（生成停顿，Fig.1a 显示停顿可达数秒）→ TBT 尾延迟爆掉；反过来等 decode 全跑完再接新请求（decode-prioritizing，FasterTransformer）→ decode 批越来越小 → 吞吐崩掉（§2.5,Algorithm 1 与 2 对照）。Sarathi-Serve 说：两个都不选，把 prefill 切成小块塞进 decode 批的“空闲算力”里，让每个迭代都既满算力又不超时。

## 1. 问题：两类迭代的错配

**数字证据（§3.1,Fig.3-6,Mistral-7B 单 A100）**：

- **decode 吞吐随 batch 近似线性增长**（批大了一卡能同时喂更多请求），而 prefill 吞吐在 batch=1 时就已饱和（一个 prompt 就够吃满算力）——batching 对 decode 是灵药、对 prefill 无感（Takeaway-1）。
- **线性层占运行时间大头**：即使序列很长、注意力开销随长度平方增长，线性层仍贡献 >80% 的运行时间（Fig.4）——优化线性层是主线。
- **算术强度对比（Fig.5）**：prefill 批高算术强度（权重搬运摊到大量 token），decode 批低算术强度（每次迭代只有少数 token，权重搬运成本摊不开）。理论上 A100 上线性层在约 200 token 时从内存受限转为计算受限，实测（高 TP 度下）在 500-600 token 附近（Fig.6 注）。
- **生成停顿（§3.2,Fig.7）**：vLLM 的迭代级批处理（站内 [orca-notes](/AIInfraGuide/papers/orca-notes) 讲的 iteration-level scheduling 的 vLLM 实现）会“尽可能多地先算 prefill”，长 prompt 的 prefill 迭代（几秒级）直接把正在 decode 的请求打断。Orca 支持混合批（prefill+decode 同批）但长 prompt 同样导致高延迟。
- **pipeline 气泡（§3.3,Fig.8）**：流水线并行（PP）下，每个 micro-batch 的执行时间差异巨大（prefill 块 vs decode 块、不同上下文长度），造成 GPU 空转。Falcon-180B 上：一个 4K prompt 的 prefill 迭代约 1150ms，而 batch 32 的 decode 迭代约 200ms——两者交错会产生约 950ms 的气泡。

## 2. 核心机制：chunked-prefills

**要解决什么**：长 prompt 的 prefill 迭代太长（几秒级），是生成停顿与气泡的根源；但 prefill 吞吐在 ~512 token 就饱和（Fig.4），说明一个 prompt 不需要一次算完也能保持 GPU 利用率。

**做法（§4.1）**：把 prefill 请求按计算量切成多个“近等大小”的块（chunk），每个块在一个迭代里算完，跨多个迭代完成整个 prefill。关键不是均匀切长度，而是按计算量切——因为每块都要重新访问前面块的 KV cache，块与块之间的计算量要大致一致才能维持批的均匀性。

**为什么不会变慢**：chunked prefill 的注意力对每个块都要重读前面块的 KV cache（§4.3：切成 N 块时，第一块的 KV 被读 N-1 次、第二块 N-2 次……），但 prefill 注意力是计算受限操作，多读 KV 不改变瓶颈；实测切块开销：chunk 512 时 prefill 总时间增加最多约 25%，chunk 2048 时几乎无开销（§5.4.1,Fig.14,Yi-34B）。

**最小例子（块级走查）**：一个 4000 token 的 prompt。vLLM 一次 prefill 迭代算完 4000 token（几秒级，打断 decode）。Sarathi-Serve 设 token budget τ=1024：这个 prompt 被切成 4 块（1024×4），分 4 个迭代与 decode 批混跑——每个迭代只增加约 1024 token 的计算量，decode 的 TBT 只被小幅推高。论文 Fig.9 对比了两种混合批：Orca 式的“decode + 完整 prefill”在长 prompt 下 TBT 暴涨（Fig.9 标题：naive hybrid 相对 decode-only 批 TBT 最高涨 28.3×），而"decode + chunked prefill"把影响压到预算内。

## 3. 核心机制：stall-free batching 与 token budget

**要解决什么**：chunked prefill 只是把 prefill 切小了，调度器还得决定“每轮迭代放多少 prefill token 才不打断 decode”。

**做法（§4.2,Algorithm 3）**：每轮调度按以下顺序填批：

1. **先装所有进行中的 decode token**（行 6-8）——decode 永远优先，这是"stall-free"的保证；
2. **再装进行中 prefill 的下一块**（行 9-12）——部分完成的 prefill 不能丢；
3. **最后装新请求的 prefill 块**（行 13-20），但总 token 数不超过预算 τ——预算按 TBT SLO 算出来（行 2），装不下就 break，等下一轮。

关键点：**新请求进入批的前提是“本轮预算还有余量”**，而不是“显存还有余量”（vLLM 的准入条件）。这样每个迭代的计算量都被 τ 卡住，decode 永远不会因为一个超长 prefill 而停顿。论文 Fig.7 展示了同样四个请求 A/B/C/D 在三种调度下的时间线：Sarathi-Serve 的 C/D prefill 被切成 p₀/p₁ 两片，与 A/B 的 decode 同批执行，谁也不等谁。

**token budget 怎么定（§4.3）**——三个互相打架的因素：

- **TBT SLO**：τ 越小，单迭代延迟越低（利于尾延迟）；
- **切块开销**：τ 太小 → 块太多 → 每块都有 kernel 启动等固定开销 + 重复读 KV（不利于吞吐）；
- **tile-quantization（块量化效应）**：GPU matmul 按 tile 分块，矩阵维度不是 tile 整数倍时部分 thread block 白算。实测 chunk 257 vs 256 会让 prefill 时间增加 32%——预算要避开“差一点对齐”的尴尬值。

论文的做法是用 Vidur（自家的 LLM 推理模拟器，§4.3）离线配置 token budget；评测中严格 SLO 用 **τ=512**、宽松 SLO 用 τ=2048（LLaMA2-70B 宽松用 1536 以减少气泡，§5.1）。

## 4. 为什么 pipeline 气泡也一起解决了

**背景**：跨节点部署大模型常用 PP（流水线并行），因为 TP（张量并行）跨节点要 all-reduce，延迟高（§2.3）。PP 的问题是有气泡：micro-batch 执行时间不均 → 后级等前级。

**Sarathi-Serve 的均匀批（§3.3,§4.2）**：混合批（decode + prefill 块）的计算量被 token budget 压得近似均匀，micro-batch 执行时间方差大幅缩小 → 气泡消失。Falcon-180B 的实验（§5.3,Fig.13）：TP-8 的 decode median TBT 比 TP4-PP2 约高 2×（跨节点 all-reduce 贵），而 Sarathi-Serve 的 PP 部署在宽松 SLO 下容量提高 1.48×、严格 SLO 下 3.6×（相对 vLLM 的 TP4-PP2 混合并行）——让“商品化网络上的 PP”第一次变得可用。

## 5. 评测：容量数字逐项核对

**指标（§2.4,§5）**：Capacity = 系统在满足延迟目标（SLO）下能承受的最大请求负载（QPS）；TTFT 看中位数、TBT 看 P99。SLO 定义（§5.1）：P99 TBT ≤ 无干扰 decode 迭代时间的 5×（严格）/ 25×（宽松）。数据集 openchat_sharegpt4（median prompt 1730 token）与 arxiv_summarization（median prompt 7059 token，Table 2）。

| 实验 | 相对基线 | 数字与出处 |
| --- | --- | --- |
| Mistral-7B 单 A100 | vLLM | 最高 **2.6×** 容量（摘要/§7） |
| Yi-34B 双 A100 TP-2,openchat_sharegpt4,严格 SLO | vLLM / Orca | **3.7×** / **4.0×**（§5.1,Fig.10b） |
| Mistral-7B,严格 SLO(100ms) | vLLM | **3.5×**（§5.2,Fig.12） |
| Yi-34B,宽松 SLO(1s) | vLLM | **1.65×**（§5.2） |
| LLaMA2-70B（TP4-PP2），openchat_sharegpt4 | vLLM / Orca | **4.3×** / **6.3×**（§5.1,Fig.11a） |
| Falcon-180B（TP4-PP2 跨 100Gbps）严格/宽松 | vLLM 混合并行 | **3.6×** / **1.48×**（§5.3,Fig.13b） |
| Falcon-180B 端到端（PP 场景上限） | vLLM | 最高 **5.6×**（摘要/§7） |

**消融（§5.4.2,Table 4）**：chunked-prefills 单独用 → TTFT 变差（切块有开销）；hybrid-batching 单独用 → TBT 变差（长 prefill 仍造成停顿）；两者合用 → 两个指标同时最优。这就是论文“两个机制缺一不可”的证明。

## 6. 与 Orca/vLLM/Splitwise 的关系

Sarathi-Serve 站在三棵树上（§6 Related Work 与 §2 背景）：

- **Orca**（站内 [orca-notes](/AIInfraGuide/papers/orca-notes)）：沿用迭代级批处理（Algorithm 2 即 vLLM 的 Orca 式调度），但给 prefill 加上了 token 上限——这是它对 Orca 唯一的“颠覆”，其余是工程化；
- **vLLM**：在 vLLM 代码库上实现（§4.4），用 FlashAttention v2/FlashInfer 内核做分页 chunked prefill——它不是另起炉灶，而是给 vLLM 换调度器；今天 vLLM 的 chunked prefill 功能就是这条线的直系产物；
- **Splitwise/DistServe**（站内 [splitwise-notes](/AIInfraGuide/papers/splitwise-notes)）：PD 解耦（prefill/decode 分集群）可以彻底消除两类迭代的干扰，但要搬 KV cache 且 prefill 集群显存闲置（§6 明确讨论）；Sarathi-Serve 是“不拆集群”的替代路线——两条路线今天共存：单机/小集群用 chunked prefill，大规模生产用 PD 解耦（如 DeepSeek-V3 的 prefill/decode 分池部署）。

## 🕰️ 原文时代 vs 当前工程

Sarathi-Serve 论文（OSDI 2024,arXiv v2）写于 vLLM v0.4 时代。2026 年 8 月复核的当前状态（来源标注如下）：

- **vLLM 已原生支持 chunked prefill**（`--enable-chunked-prefill` 参数，官方文档；评测用 vLLM v0.5.1 时代的功能）——Sarathi-Serve 的机制已成为主流引擎的默认选项之一，而非学术孤品；
- **Mooncake/SGLang 等系统把“PD 解耦 + chunked prefill”组合使用**（Mooncake 论文 §5.2 明确对比了 chunked prefill 与 P/D 分离的取舍）——论文 §6 预言的“两条路线共存”已成为现实；
- **TBT/TTFT 指标与 SLO 定义方式仍是行业标准**（vLLM 官方性能文档、LLMPerf 等基准沿用）。

**结论边界**：论文的核心主张（decode 内存受限 → 可搭车）与机制（chunked prefill + 预算调度）没有过时；但“token budget 用 Vidur 模拟器离线定”的做法，在今天的 vLLM 里是启动参数 + 运行时启发式，读者应按当前版本文档操作。

## 7. 常见误读与错误做法

- **误读 1："Sarathi-Serve = chunked prefill。"** 错。chunked prefill 只是把 prefill 切小；没有 stall-free batching（预算调度）时，单独切块甚至会让 TTFT 更差（§5.4.2,Table 4）。两个机制是配套的。
- **误读 2：“chunked prefill 会显著拖慢 prefill 本身。”** 部分错。切块确实有开销（重复读 KV、kernel 启动），但实测 chunk 2048 几乎无开销、chunk 512 最多约 25%（§5.4.1,Fig.14）——而且这是“为了换 decode 不被打断”的合理代价。真正要避的是 tile-quantization 的“差一点对齐”（257 vs 256 差 32%,§4.3）。
- **误读 3：“PD 解耦（Splitwise）比 Sarathi-Serve 严格更优。”** 错。论文 §6 明确列出解耦路线的代价：KV 迁移需要高带宽、prefill 集群显存闲置；Sarathi-Serve 的目标场景是“没有高速互联的商品化集群”。今天生产系统两者都在用（见 🕰️ 节）。
- **错误做法 1（调参）**：把 token budget 拉满以追求吞吐，忽略 P99 TBT。预算的本质是 SLO 的换算；论文在严格 SLO 下用 512、宽松用 2048,这个差距本身就说明“预算越大，延迟越差”。
- **错误做法 2（理解基线）**：拿“vLLM 默认配置”当 vLLM 的全部。vLLM 的调度策略是插件化的，论文对比的是 prefill-prioritizing 配置；今天的 vLLM 默认已含 chunked prefill 调度——拿旧结论套新版本会得出过时判断。

## 📝 总结

1. **两类迭代的算术强度错配是全部问题的根源**：decode 内存受限（有空闲算力）、prefill 计算受限（一个 prompt 就饱和）——“往 decode 批里塞 prefill 块”因此可行，而“一次塞整个长 prompt”不可行（28.3× TBT）。
2. **chunked-prefills + stall-free batching 是两个配套机制**：前者把 prefill 切成计算量均匀的块，后者用 token budget 保证每轮迭代 decode 优先、总量受限——decode 永不被打断，吞吐与延迟同时保住。
3. **实测与影响**：Mistral-7B 2.6×、Yi-34B 3.7×/4.0×、Falcon-180B PP 5.6×;它把“商品化网络上的 PP”变成可行选项；今天 vLLM 的 chunked prefill 与生产系统的 PD 解耦都是这条线的延续。

## 🎯 自我检验清单

- [ ] 能解释为什么 decode 批可以“免费”搭载 prefill token（算术强度、Fig.5-6 与 200/500-600 token 阈值）。
- [ ] 能说出生成停顿的机制（prefill-prioritizing 调度 + 长 prompt 迭代）与论文 Fig.1a 的量级（数秒）。
- [ ] 能按 Algorithm 3 走一遍“decode 优先 → 续 prefill → 新请求填预算”的决策顺序。
- [ ] 能解释 token budget 的三个约束（SLO、切块开销、tile-quantization）与 512/2048 配置的取舍。
- [ ] 能复述至少 4 个容量数字（2.6×/3.7×/4.3×/5.6× 等）及其相对基线与出处（§5.1-5.3）。
- [ ] 能说清 Sarathi-Serve 与 Orca/vLLM/Splitwise 三条路线的异同与今天的共存状态。

## 📚 参考资料

- Sarathi-Serve 原文（OSDI 2024）：https://www.usenix.org/conference/osdi24/presentation/agrawal 与 arXiv:2403.02310
- 官方代码：https://github.com/microsoft/sarathi-serve
- 站内关联：[orca-notes](/AIInfraGuide/papers/orca-notes)（迭代级调度）、[pagedattention-notes](/AIInfraGuide/papers/pagedattention-notes)（KV cache 分页）、[splitwise-notes](/AIInfraGuide/papers/splitwise-notes)（PD 解耦）
