---
title: "NanoFlow 精读：LLM serving 其实是计算受限的，而引擎们把 GPU 用废了一半"
description: "拆解 OSDI 2025 的 NanoFlow：端到端 LLM serving 为什么是 compute-bound（成本模型推导）、单算子利用率 80% 而整体只有 40% 的差距从哪来、nano-batch 与执行单元调度如何让计算/内存/网络三类操作在一张卡内重叠，以及 1.91×/68.5% 的评测口径。"
pubDate: 2026-08-15
originalUrl: "https://www.usenix.org/conference/osdi25/presentation/zhu-kan"
sourceType: "paper"
originalAuthor: "Kan Zhu, Yilong Zhao, Liangyu Zhao, Gefei Zuo, Yile Gu, Dedong Xie, Yufei Gao, Tian Tang, Qinyu Xu, Zihao Ye, Keisuke Kamahori, Chien-Yu Lin, Ziren Wang, Stephanie Wang, Arvind Krishnamurthy, Baris Kasikci (University of Washington 等)"
tags: ["NanoFlow", "Intra-device 并行", "吞吐优化", "推理引擎", "LLM Serving"]
stage: engine
order: 7
prereqs: ["orca-notes", "splitwise-notes"]
minutes: 45
difficulty: 3
---

> 原文：[NanoFlow: Towards Optimal Large Language Model Serving Throughput](https://www.usenix.org/conference/osdi25/presentation/zhu-kan)（Kan Zhu 等，OSDI 2025,arXiv:2408.12757 v2;本文访问日期 2026-08-15）

“LLM serving 是内存受限的”——这个常识被 NanoFlow 用成本模型推翻了：**端到端看，现代 LLM serving 是计算受限（compute-bound）的**（§3）。推理里确实有内存受限的部分（decode 注意力）和网络受限的部分（张量并行的通信），但引擎把它们串行执行：先跑计算、再跑通信、再跑注意力……每个算子单独看利用率不低（约 80%），但整卡的计算利用率只有约 40%（§1,§6.2）——GPU 的计算单元在等内存和网络。NanoFlow 的答案是 intra-device parallelism（设备内并行）：把输入批切成 nano-batch（微批），让同一操作在多个微批上重复执行（nano-operation），于是“计算型算子处理微批 A”的同时，“内存型算子处理微批 B”、“网络型算子处理微批 C”可以在一张卡内重叠。流水线由 auto-search 自动生成（两阶段：先无干扰调度，再按实测干扰重排，§4.1）。效果：LLaMA-2-70B 上相对 vLLM/DeepSpeed-FastGen/TensorRT-LLM 平均 1.91× 吞吐提升，达到理论最优的 68.5%（§6.2）。本文先推导“为什么是计算受限”，再拆 nano-batch 机制与 auto-search，最后核对评测数字。

<!-- more -->

## 📑 目录

- [🗺️ 原文阅读地图](#️-原文阅读地图)
- [0. 读前 3 分钟：两组直觉](#0-读前-3-分钟两组直觉)
- [1. 反直觉主张：端到端 serving 是计算受限的](#1-反直觉主张端到端-serving-是计算受限的)
- [2. 差距：每个算子 80%，整卡只有 40%](#2-差距每个算子-80整卡只有-40)
- [3. 核心机制：nano-batch 与设备内流水线](#3-核心机制nano-batch-与设备内流水线)
- [4. auto-search：两阶段搜索流水线](#4-auto-search两阶段搜索流水线)
- [5. 运行时：异步调度与 KV cache offload](#5-运行时异步调度与-kv-cache-offload)
- [6. 评测：1.91× 与 68.5% 的数字口径](#6-评测191-与-685-的数字口径)
- [7. 与已有系统的关系](#7-与已有系统的关系)
- [🕰️ 原文时代 vs 当前工程](#️-原文时代-vs-当前工程)
- [8. 常见误读与错误做法](#8-常见误读与错误做法)
- [📝 总结](#-总结)
- [🎯 自我检验清单](#-自我检验清单)
- [📚 参考资料](#-参考资料)

## 🗺️ 原文阅读地图

NanoFlow 论文 7 节（OSDI 版），本文精讲“分析 → 机制 → 搜索 → 评测”主线；Transformer 背景与相关工作简述。

| 原文单元 | 处理 | 本文位置/省略理由 | 来源锚点 |
| --- | --- | --- | --- |
| §1 Introduction（40% vs 80%、1.91×、68.5% 承诺） | 精讲 | 开篇与第 2 节 | §1 |
| §2 Background（2.1 工作流、2.2 操作四分类、2.3 TP/PP） | 简述 | 第 0、1 节少量引用 | §2.1-2.3;Fig.1 |
| §3 Analysis（3.1-3.5 成本模型式 1-4、Table 1、3.6 差距、3.7 intra-device 并行） | 精讲 | 第 1、2、3 节：公式逐项拆解 | §3.1-3.7;式(1)-(4);Table 1 |
| §4 设计（4.1 auto-search 两阶段、4.2 runtime） | 精讲（4.1） | 第 4 节 | §4.1-4.2 |
| §5 实现（拓扑排序 + 贪心搜索、干扰 profiling） | 简述 | 第 4 节 | §5 |
| §6 评测（6.2 吞吐 Fig.7、6.3 延迟 Fig.8、6.4 消融） | 精讲 | 第 6 节：数字逐项核对 | §6.2-6.4;Fig.7-11 |
| §7 相关工作与结论 | 简述 | 第 7 节 | §7 |

📌 **本文承诺**：读完后，你应该能用自己的话推导“为什么大模型 + GQA + 大 batch 下网络和内存都不再是瓶颈”（式 4 的量级分析），能画出 nano-batch 重叠的时间线，并说清 1.91×/68.5%/2.66× 各自的比较对象。

## 0. 读前 3 分钟：两组直觉

**第一组：GPU 上“算”和“搬”是两套资源，串行用就是浪费。** 一次 LLM 迭代里，GEMM（矩阵乘）主要用计算单元（SM 的 tensor core），decode 注意力主要用内存带宽（搬 KV cache），张量并行还要用网络（NVLink 通信）。现有引擎让它们一个接一个跑：跑 GEMM 时内存单元闲着，跑注意力时计算单元闲着。如果能把批切开，让 GEMM 处理第 1 块的同时、注意力处理第 2 块、通信处理第 3 块，三种资源就同时忙起来——这就是 NanoFlow 的全部思想。

**第二组：“内存受限”与“计算受限”取决于怎么数账。** 单看 decode 注意力，它是内存受限的（每迭代只处理 1 个 token/请求，却要搬整个 KV cache）。但端到端看整个迭代：GEMM 的计算量随模型变大而变大，而 GQA（分组查询注意力）又把 KV cache 压小了、让批能装更多请求——大 batch 让 GEMM 成为大头。NanoFlow 的结论：对现代大模型 + 大 batch，计算时间 > 内存时间 > 网络时间，瓶颈是计算。

## 1. 反直觉主张：端到端 serving 是计算受限的

**成本模型（§3.2,式 1-3）**：一次迭代的延迟从三个资源分别估算：

- **内存**：$T_{mem} = \frac{MemSize}{MemBW}$（式 1）——最大 batch 下，整卡内存内容每迭代都要过一遍（权重复用距离太长，缓存不住）；
- **计算**：$T_{Compute} \approx \frac{2B_{Dense} \cdot P_{Model}}{Compute}$（式 2）——GEMM 的计算量 = 2 × 稠密批大小 × 模型参数量（每个权重元素参与一次乘加），除以卡的计算能力；
- **网络**：$T_{net} \approx 4 \cdot \frac{N_{GPU} B_{Dense} D_{model} S_{type} L}{NetBW}$（式 3）——张量并行每层约 4 次激活量级的跨卡传输。

**为什么网络不是瓶颈（§3.3）**：$T_{Net}/T_{Compute} = \frac{2D_{model}L}{P_{model}} \cdot \frac{N_{GPU}Compute}{NetBW/S_{type}}$。第一项约等于 $1/(6D_{model})$（因为 $P_{model} \approx 12D_{model}^2 L$），对 $D_{model} > 4096$ 的现代模型小于 $10^{-5}$；第二项（卡数 × 算力/带宽比）在数据中心 GPU 上是 $10^4$-$10^5$ 量级。两者相乘通常 <1——网络时间小于计算时间。

**为什么内存不是瓶颈（§3.3,式 4）**：$T_R = \frac{T_{Mem}}{T_{Compute}} \approx \frac{Compute}{MemBW} \cdot \frac{MemSize}{P_{model}} \cdot \frac{1}{2B_{dense}}$。GQA 是关键：KV 头共享让同样显存能装更多请求，LLaMA-2-70B 在 8×A100 上最大 decode 批约 1024，加上 prefill token 后 $B_{dense}$ 可达 2048；而同样大小的非 GQA 模型只有 $B_{dense}=256$（§3.3 原文例子）。大 $B_{dense}$ 把 $T_R$ 压到 <1。模型越大（$P_{model}$ 越大），$T_R$ 越低——计算越来越是瓶颈。Table 1 列了 13 款加速器的 Compute/MemBW 比值（A100 约 200、H100 约 295、B200 约 281），配合 $MemSize/P_{model}$ 与 $1/(2B_{dense})$ 即可判断任意配置。

## 2. 差距：每个算子 80%，整卡只有 40%

**单个算子的效率很高**：论文测量各操作对各自瓶颈资源的利用率约 80%（§1）。

**但整体计算利用率只有约 40%**（§1,§6.2）——为什么？因为算子按“计算型 → 内存型 → 网络型”串行执行：GEMM 跑的时候内存/网络闲着；decode 注意力跑的时候计算闲着；TP 通信跑的时候计算也闲着。每个算子的 80% 是“它在跑的那段时间里”的利用率，乘上它占的时间比例后，计算单元的总体利用率只剩 40% 左右。

**现有引擎为什么差（§3.6,Fig.7）**：离线吞吐对比中，vLLM、DeepSpeed-FastGen、TensorRT-LLM 分别只达到理论最优的 22.0%、22.9%、37.8%——即使是最好的引擎，也有六成以上的理论吞吐没拿到。论文把这归因于“设备内资源的串行使用”，而不是单个内核写得不好。

## 3. 核心机制：nano-batch 与设备内流水线

**要解决什么**：让“计算、内存、网络”三类操作在一张卡内同时进行。

**做法（§1,§3.7）**：把输入批切成多个 nano-batch（微批），同一个操作在多个微批上重复执行（如 Up 投影在 batch 2048 上切成 UP1 处理 0-768、UP2 处理 768-2048，§3.7 原文例子）。因为不同微批之间没有数据依赖，计算型、内存型、网络型的 nano-operation 可以并行跑在不同微批上。代价：权重被重复加载（内存 I/O 增加）——但整体是计算受限时，多出来的内存 I/O 可以被流水线隐藏（§1）。

**执行单元调度（§4.1 语境）**：并行执行需要显式分配 GPU 资源——大部分 SM 给计算重的 GEMM，剩余 SM 给内存型操作（decode 注意力用小部分 SM 也能拿到可观带宽，因为它是带宽瓶颈不是 SM 瓶颈）。LLaMA-2-70B 的例子：decode 注意力以 0.4 的资源利用率运行（牺牲 40% GEMM 性能）就能达到其自身最大性能的 80%，于是 auto-search 在 KQV 生成段用 4 个 nano-operation 重叠它们，其余流水线段 GEMM 优先、只用 2 个 nano-operation（§4.1 原文，Fig.6）。

**MoE 模型（§4.1）**：专家负载不均，NanoFlow 用张量并行处理 MoE，FFN 用 grouped-GEMM，加上 gate 路由操作；auto-search 同样自动生成流水线（§6 验证了 Deepseek-67B、Mixtral 8×7B 等）。

## 4. auto-search：两阶段搜索流水线

**要解决什么（§1）**：nano-batch 数量、每个 nano-operation 的批大小、执行顺序、资源分配——搜索空间巨大，且并行 kernel 之间会互相干扰（抢占执行单元/缓存，性能不可预测）。

**两阶段搜索（§1,§4.1）**：

1. **第一阶段（无干扰假设）**：先假设 kernel 之间互不干扰，用成本模型 + 拓扑排序与贪心搜索确定初始流水线（nano-operation 的数量、大小、顺序）；
2. **第二阶段（干扰修正）**：按实际 profiling 的 kernel 干扰重新规划——把“并行 kernel 抢资源”造成的性能损失纳入调度决策。

搜索输入是离线 profiling 的 compute/memory/network-bound kernel 特征（§5 语境），输出是一个可直接执行的设备内流水线。**这使 NanoFlow 可以自动移植到不同模型**（LLaMA-3-8B 等单卡模型不需要网络操作，auto-search 自动把操作切成 2 个 nano-operation 重叠 decode 注意力与 Up/Gate/Down 投影，§4.1）。

## 5. 运行时：异步调度与 KV cache offload

**异步请求调度（§4.2.1）**：NanoFlow 假设控制面（auto-scaling、负载均衡、优先级路由）由外部系统负责；实例内用稠密 batch（如 LLaMA-2-70B 用 batch 2048 最优，§6.2）持续处理请求——稠密且恒定的批大小让执行时间可预测，P99 延迟接近平均（§6.3:P99 仅为平均的 1.07×）。

**KV cache SSD offload（§4.2.2）**：多轮对话的旧 KV cache 换出到 SSD，节省显存放更多请求——注意这与 [pagedattention-notes](/AIInfraGuide/papers/pagedattention-notes) 的分页机制是互补的（分页管显存内的碎片与共享，offload 管显存外的容量）。

## 6. 评测：1.91× 与 68.5% 的数字口径

**主实验（§6,LLaMA-2-70B,8×A100 单节点）**：

- **理论最优吞吐（§3.5）**：用 CUTLASS profiling 出 FP16 峰值计算能力 280 TFLOPS（8×A100 聚合），代入式 2 得最优吞吐 1857 tokens/s/GPU（与用户查询统计无关，恒为常数）；
- **离线吞吐（§6.2,Fig.7）**：NanoFlow 最高达理论最优的 68.5%；常量长度下平均相对 vLLM 2.62×、DeepSpeed-FastGen 2.78×、TensorRT-LLM 1.73×；按数据集采样长度时相对 vLLM 4.18×、DeepSpeed-FastGen 3.45×、TensorRT-LLM 1.91×（摘要的 1.91× 即此口径）；
- **在线延迟（§6.3,Fig.8）**：200ms 平均归一化延迟 SLO 下，NanoFlow 相对 TensorRT-LLM 可承受 1.64× 更高的请求率（LMSys-Chat 1M 数据集）；低请求率下延迟与最优基线相当（略高，因为面向吞吐的大 batch）；
- **其他模型（§6.2,Fig.11）**：LLaMA-3-70B/8B、QWen2-72B、Deepseek-67B、Mixtral 8×7B 达到理论最优的 50%-72%（arXiv v2 口径；OSDI 正式版为 59%-72%），相对 vLLM 平均 2.66×；
- **消融（§6.4）**：nano-batching、nano-operation 重叠、KV offload 各自贡献分离验证。

**版本口径**：本文核对 arXiv:2408.12757 v2;OSDI '25 正式版摘要中“59%-72%”与 v2 的“50%-72%”有出入，引用时按版本注明。

## 7. 与已有系统的关系

- **与 Orca/vLLM（[orca-notes](/AIInfraGuide/papers/orca-notes)）**：迭代级调度解决的是“请求之间”的粒度问题；NanoFlow 解决的是“单卡内部资源”的粒度问题——两者正交（NanoFlow 的批内调度 + 连续批处理的请求调度可以共存）；
- **与 Splitwise（[splitwise-notes](/AIInfraGuide/papers/splitwise-notes)）**：Splitwise 把 prefill/decode 拆到不同机器；NanoFlow 在同一卡内重叠它们——论文评测直接用 Splitwise 数据集（§6 语境）；
- **与 FlashAttention（站内 [pagedattention-notes](/AIInfraGuide/papers/pagedattention-notes) 的注意力内核语境）**：FlashAttention 类工作优化的是单个注意力 kernel 的访存；NanoFlow 优化的是kernel 之间的并行——论文指出单个 kernel 已做到 80% 利用率，瓶颈在调度不在内核。

## 🕰️ 原文时代 vs 当前工程

NanoFlow 论文（arXiv 2024-08,OSDI 2025-07）与 2026 年 8 月现状对比（来源标注如下）：

- **代码与生态**：论文未公开完整源码（OSDI 版未提供官方仓库链接，「待核实」——截至 2026-08-15 未见官方开源仓库），后续进展需以论文与引用为准；
- **思想影响**：“LLM serving 端到端计算受限”的分析框架被后续系统论文广泛引用；“单卡内重叠异构操作”的思路与同期/后续工作（如利用 CUDA streams 重叠通信与计算的实践）方向一致；
- **硬件趋势**：Table 1 里 B200 的 Compute/MemBW ≈ 281 与 H100 的 295 同量级——计算/带宽比没有下降,意味着“计算受限”的判断对 Blackwell 时代依然成立，论文的结论边界未变。

**结论边界**：NanoFlow 的目标是吞吐优先场景（论文明说 batch=1 的延迟型负载不是它的目标，§6.3 低请求率下延迟略高）；在线交互型服务（强 TTFT/TBT SLO）仍需要 chunked prefill 与 token 预算调度（如 Sarathi-Serve 思路，站内 [splitwise-notes](/AIInfraGuide/papers/splitwise-notes) 的 PD 解耦是另一条路线）的配合。

## 8. 常见误读与错误做法

- **误读 1：“NanoFlow 说注意力不是瓶颈。”** 错。它说的是端到端迭代里 GEMM 计算占主导；decode 注意力本身仍是内存受限的（论文明确，§2.2）——正因为它内存受限，才能用少量 SM 边角料跑它，把 SM 让给 GEMM。
- **误读 2：“nano-batch 会重复加载权重，所以更慢。”** 只在“整体不是计算受限”时成立。计算受限时，多出的内存 I/O 被流水线隐藏（§1）；这就是为什么论文先花一整节证明“现代负载计算受限”——前提不成立，机制就不成立。
- **误读 3：“1.91× 是相对所有引擎在所有场景。”** 错。1.91× 是相对 TensorRT-LLM（最强基线）在数据集采样长度下的离线吞吐；相对 vLLM 是 4.18×（同一口径）。不同基线、不同长度口径的数字不能混用（§6.2）。
- **错误做法 1（移植）**：手写 nano-operation 数量与资源分配。论文的卖点就是 auto-search 自动做（两阶段，§4.1）；手工配置既难调又不可移植。
- **错误做法 2（选型）**：给延迟敏感型在线服务上 NanoFlow 式的大稠密 batch。论文自认延迟不是其目标（§6.3 低负载下延迟略高于基线）；混合负载需要把“设备内并行”与“预算调度”结合，而不是二选一。

## 📝 总结

1. **端到端 serving 是计算受限的**：$T_{Net}/T_{Compute}$ 与 $T_{Mem}/T_{Compute}$ 两个比值在现代大模型 + GQA + 大 batch 下都小于 1（式 3-4 量级分析）——瓶颈是计算，不是内存/网络。
2. **瓶颈不在内核，在调度**：单算子 80% 利用率，串行执行后整卡计算利用率只有约 40%;nano-batch 让异构操作在同一张卡内重叠，auto-search 自动生成流水线。
3. **实测**：LLaMA-2-70B 上 68.5% 理论最优、相对 TensorRT-LLM 1.91×（数据集口径）/ 相对 vLLM 4.18×,其他模型 50%-72% 最优、平均 2.66× vs vLLM——“设备内并行”是继连续批处理、PD 解耦之后又一个独立的吞吐杠杆。

## 🎯 自我检验清单

- [ ] 能写出式 1-4 并解释每项含义，能用量级分析说明为什么网络与内存不再是瓶颈（GQA 的 $B_{dense}$ 与 $P_{model}$ 的角色）。
- [ ] 能解释“单算子 80% vs 整卡 40%”的差距来源（串行执行、资源错配）。
- [ ] 能画出 nano-batch 重叠的基本时间线，并解释“权重重复加载为什么可以被隐藏”。
- [ ] 能说清 auto-search 两阶段各做什么（无干扰调度 → 干扰修正）。
- [ ] 能区分三组吞吐数字：1.91×（vs TRT-LLM,数据集）、4.18×（vs vLLM,数据集）、68.5%（vs 理论最优），并指出出处（§6.2）。
- [ ] 能说明 NanoFlow 与 Orca/Splitwise/FlashAttention 的互补关系，以及它的适用边界（吞吐优先，非延迟型）。

## 📚 参考资料

- NanoFlow 原文（OSDI 2025）：https://www.usenix.org/conference/osdi25/presentation/zhu-kan 与 arXiv:2408.12757（v2）
- 站内关联：[orca-notes](/AIInfraGuide/papers/orca-notes)（迭代级调度）、[splitwise-notes](/AIInfraGuide/papers/splitwise-notes)（PD 解耦）、[pagedattention-notes](/AIInfraGuide/papers/pagedattention-notes)（KV cache 管理）
