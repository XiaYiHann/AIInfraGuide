---
title: "Mooncake 精读：把整个集群的 CPU 内存和 SSD 变成一张全局 KV cache 网"
description: "拆解 FAST 2025 的 Mooncake(Kimi 的 serving 平台）：KVCache-centric 的解耦架构如何用 CPU/DRAM/SSD 做全局前缀缓存、Conductor 如何做 cache-aware 调度与热块迁移、过载场景下基于预测的 early rejection,以及 50%-525% 吞吐提升与 75% 请求增量的真实口径。"
pubDate: 2026-08-15
originalUrl: "https://arxiv.org/abs/2407.00079"
sourceType: "paper"
originalAuthor: "Ruoyu Qin, Zheming Li, Weiran He, Mingxing Zhang, Yongwei Wu, Weimin Zheng, Xinran Xu (Moonshot AI & Tsinghua University)"
tags: ["Mooncake", "KV Cache", "PD 解耦", "前缀缓存", "Kimi", "LLM Serving"]
stage: engine
order: 6
prereqs: ["splitwise-notes", "pagedattention-notes"]
minutes: 45
difficulty: 3
---

> 原文：[Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079)（Ruoyu Qin, Zheming Li, Weiran He, Mingxing Zhang, Yongwei Wu, Weimin Zheng, Xinran Xu,FAST 2025,arXiv:2407.00079 v3;本文访问日期 2026-08-15）

Kimi 的 serving 平台 Mooncake 回答一个问题：**当请求带着几万 token 的上下文涌进来时，重复计算和显存容量哪个更贵？** 答案是都贵，而它的解法是把问题变成“缓存调度问题”：KVCache-centric（以 KV 缓存为中心）的解耦架构——prefill 集群与 decode 集群分离（沿袭 [splitwise-notes](/AIInfraGuide/papers/splitwise-notes) 的 PD 解耦），再把 GPU 集群里闲置的 CPU 内存、SSD 和 RDMA 带宽组织成一张全局 KV cache 池（Fig.1）。每个请求先复用尽可能多的前缀缓存，prefill 时按层流式把新 KV 传给 decode 节点（§3,Fig.4 四步工作流）。中央调度器 Conductor 同时干三件事：cache-aware 的 prefill 实例选择（§6.1,Algorithm 1）、热块自动复制迁移（§6.2）、以及过载场景下基于预测的 early rejection（提前拒绝）（§7）。效果：长上下文模拟场景吞吐提升 50%-525%，真实负载下 Kimi 多处理 75% 的请求（§8.1）。本文先建立“缓存复用 vs SLO”的直觉，再拆四步工作流、Conductor 调度算法与 early rejection 的机制，最后核对评测数字的版本口径。

<!-- more -->

## 📑 目录

- [🗺️ 原文阅读地图](#️-原文阅读地图)
- [0. 读前 3 分钟：两组直觉](#0-读前-3-分钟两组直觉)
- [1. 架构总览：Conductor 与四步工作流](#1-架构总览conductor-与四步工作流)
- [2. 全局 KV cache：CPU 内存池、块 hash 与热块迁移](#2-全局-kv-cachecpu-内存池块-hash-与热块迁移)
- [3. prefill 池：chunked pipeline parallelism 与按层流式传输](#3-prefill-池chunked-pipeline-parallelism-与按层流式传输)
- [4. Conductor 调度：cache-aware 的实例选择](#4-conductor-调度cache-aware-的实例选择)
- [5. 过载调度：early rejection 与负载波动](#5-过载调度early-rejection-与负载波动)
- [6. 评测：50%-525% 与 75% 的数字口径](#6-评测50-525-与-75-的数字口径)
- [🕰️ 原文时代 vs 当前工程：arXiv 版与 FAST '25 版](#️-原文时代-vs-当前工程arxiv-版与-fast-25-版)
- [7. 常见误读与错误做法](#7-常见误读与错误做法)
- [📝 总结](#-总结)
- [🎯 自我检验清单](#-自我检验清单)
- [📚 参考资料](#-参考资料)

## 🗺️ 原文阅读地图

Mooncake 论文 10 节，本文精讲“架构 + 调度 + 过载策略 + 评测”主线；trace 数据分析与未来方向简述。

| 原文单元 | 处理 | 本文位置/省略理由 | 来源锚点 |
| --- | --- | --- | --- |
| §1 Introduction（动机、525%/75% 承诺） | 精讲 | 开篇与第 6 节 | §1;Fig.1 |
| §2 Preliminary（TTFT/TBT、SLO 定义、goodput） | 简述 | 第 0 节 | §2 |
| §3 架构总览（Fig.3 缓存池、Fig.4 四步工作流、Messenger） | 精讲 | 第 1 节：四步流程 | §3;Fig.3-4 |
| §4 真实请求 trace（23608 条、平均输入 7590、Table 1 缓存策略） | 简述 | 第 2 节引用关键统计 | §4;Table 1;Fig.5-6 |
| §5 prefill 池（5.1 CPP、5.2 layer-wise prefill） | 精讲 | 第 3 节：为什么不用 SP | §5.1-5.2;Fig.7 |
| §6 调度（6.1 Algorithm 1 cache-aware 调度、6.2 热块迁移） | 精讲 | 第 2、4 节：机制与算法走查 | §6.1-6.2;Algorithm 1;Fig.8 |
| §7 过载调度（7.1-7.4：SLO 负载、early rejection、波动、预测） | 精讲 | 第 5 节：四阶段波动分析 | §7.1-7.4;Fig.9-10 |
| §8 评测（8.1.1 公开数据集、8.1.2 模拟、8.1.3 真实负载、8.2 分析） | 精讲（8.1） | 第 6 节：数字逐项核对 | §8.1;Fig.11-13 |
| §9 Related Work、§10 Conclusion | 简述 | 第 7 节与 🕰️ 节 | §9-10 |

📌 **本文承诺**：读完后，你应该能画出 Mooncake 的四步请求工作流、按 Algorithm 1 走一遍“选 prefill 实例”的决策（含缓存阈值分支），并解释 early rejection 为什么会导致负载反相波动、预测为什么能压住它。

## 0. 读前 3 分钟：两组直觉

**第一组：KV cache 是“可复用但会过期”的中间产物。** 两个请求的 prompt 前缀相同时（比如系统提示词、共享文档开头），前面算好的 KV 可以直接复用，省掉重复 prefill 计算。但 KV cache 很大（每 token 每层都要存），GPU 显存放不下多少——于是 Mooncake 把 CPU 内存、SSD 变成“第二层缓存”，用 RDMA 高速搬运。核心权衡：复用远程缓存能省计算，但等缓存传输可能违反 TTFT SLO；缓存服务器太热还会网络拥塞（§1.1）。所以调度器必须在“多复用缓存”与“满足 SLO”之间找平衡。

**第二组：过载时，“拒绝”也要讲策略。** 大多数 serving 研究假设“所有请求都会被处理”，但 Kimi 实际长期过载（GPU 供给不足，§1.1）。在解耦架构里，如果请求在 prefill 池算完才发现 decode 池满了，prefill 的计算就白费了——所以要“提前拒绝”：到达时就看 decode 侧未来的负载。但直接提前拒绝又会导致 prefill/decode 两池负载反相波动（§7.3），需要用负载预测来平滑。

## 1. 架构总览：Conductor 与四步工作流

**组件（§3,Fig.1）**：

- **prefill 实例池与 decode 实例池**：两类节点分开部署；
- **全局 KV cache 池**：把 GPU 集群节点上的 CPU DRAM 与 SSD 组织起来，KV cache 以分页块（paged block）存储，用 LRU/LFU 等策略淘汰（§3,Fig.3）；
- **Messenger**：每节点一个独立进程，用 (GPUDirect) RDMA 做跨机 KV cache 高速传输；
- **Conductor**：全局调度器，负责实例选择、块复制/换出决策、过载拒绝。

**四步工作流（§3,Fig.4）**：

1. **KV cache 复用**：请求到达后，Conductor 选定 prefill 节点（组），把可复用的前缀缓存块 ID 交给它，从远端 CPU 内存加载到 GPU 显存作为起点（无前缀则跳过）；
2. **增量 prefill**：prefill 节点用前缀缓存完成剩余的 prefill，把新产生的 KV 写回 CPU 内存池。如果未缓存输入 token 数超过阈值 `prefill_chunk`（通常大于 1000 token，选这个值是为了吃满 GPU 算力），prefill 会切成多块按流水线执行；
3. **KV 传输**：Messenger 把每层新产生的 KV 按层流式（layer-by-layer）异步传给目标 decode 节点的 CPU 内存，与第 2 步的 prefill 计算重叠——不等全部算完再搬（§5.2 详细讲）；
4. **解码**：KV 全部到达 decode 节点后，请求加入 continuous batching 的下一批。decode 节点由 Conductor 预选（按当前负载），但本地调度器会二次检查 TBT SLO——如果负载已变化，请求可能在此被拒，前序 prefill 成本作废（这正是 §7 early rejection 要解决的问题）。

## 2. 全局 KV cache：CPU 内存池、块 hash 与热块迁移

**块与 hash（§3,Fig.3；§4）**：KV cache 以 512 token 为一块存储，每个块挂一个前缀 hash（本块 token 的 hash 与前面所有块的 hash 拼接后再 hash）——相同 hash 即代表“从开头到这块的 token 完全相同”，可直接复用。论文公开了 1 小时真实请求 trace（23,608 条，含时间戳/输入输出长度/重映射块 hash，§4）：平均输入 7590 token、平均输出 182 token，输入输出比约 720——这是 Kimi 长上下文负载的典型画像。

**缓存策略分析（§4.2,Table 1）**：缓存容量从 1000 块加到 50000 块，命中率从 30% 升到 50%，再大收益甚微；LRU 在该 trace 上表现最好（请求在时间上邻近）。另一个关键统计：超过 50% 的缓存块从未被使用，而少数热块被访问数万次（Fig.6）——热块必须复制，否则传输拥塞。

**热块迁移（§6.2）**：Conductor 无法精确预测未来缓存使用（负载动态性太强），于是用启发式：当请求被派到“本地前缀较短”的实例时，如果远程最佳前缀长度与本地前缀长度之比超过阈值 `kvcache_balancing_threshold`（Algorithm 1 行 30），就执行 TransferKVCache 把热块从持有者搬到当前实例——“顺路复制”：既省了本次请求的 prefill 时间，又自动完成了热块的分布复制。阈值以下则选择直接计算（多算一点但省传输时间）。

## 3. prefill 池：chunked pipeline parallelism 与按层流式传输

**为什么 prefill 也要多节点（§5.1）**：长上下文请求（8K 到 128K、甚至 1M）输入 token 可能是输出的 10-100 倍，TTFT 主要花在 prefill。单个 8×GPU 节点可能不够，但跨节点张量并行（TP）每层要两次 RDMA all-reduce，MFU 很差；序列并行（SP，如 Ring Attention）每层至少一次跨节点通信，且要动态弹性扩缩节点组，复杂。

**CPP（chunked pipeline parallelism，分块流水线并行）**：把 prefill 节点按 X 个一组组成流水线组，请求的输入按 `prefill_chunk` 切成块，不同块可在不同节点同时处理——像训练里的流水线并行，跨节点通信只发生在阶段边界，可轻易与计算重叠。论文强调这是推理侧首次应用（训练侧已有），因为它“天然适配长短上下文、无需频繁动态调整节点划分”。

**Layer-wise prefill（按层 prefill，§5.2,Fig.7）**：prefill 是逐层计算且计算受限的，所以 KV 的存储/传输可以按层异步重叠：每层注意力开始前等该层 KV 加载完成、同时触发下一层的异步加载；该层注意力算完立刻启动该层 KV 的异步存储。效果：prefill 实例的执行时间约等于“KV 加载时间”与“标准 prefill 时间”的较大者，长上下文下存储延迟大幅下降（Fig.7）。这还带来一个推论：prefill 调度可以忽略显存大小（只要放得下一个请求），因为 KV 不驻留显存——prefill 实例调度只看 KV 分布与 CPU DRAM 可用量（Fig.1 标注）。

## 4. Conductor 调度：cache-aware 的实例选择

**要解决什么（§6.1）**：传统调度按“请求数/负载”选实例；Mooncake 要同时考虑前缀缓存命中长度（影响 prefill 时间）与实例排队时间（影响 TTFT）。

**Algorithm 1 走查（§6.1）**：

1. 对请求 prompt 按块算前缀 hash（行 3）；
2. 对每个 prefill 实例，计算它的本地前缀匹配长度 `prefix_len`，并找出全局最佳匹配（行 5-6）；
3. **分支判断（行 10）**：如果 `best_prefix_len / prefix_len < kvcache_balancing_threshold`——即“最佳远程缓存比本地好得有限”——走 cache-aware 分支：只按本地实例的排队时间 + 预计 prefill 时间估 TTFT（行 11-15）；
4. **否则走 cache-aware-and-balancing 分支**（行 16-23）：把远端缓存的传输时间也算进 TTFT（行 17-19）；
5. 选 TTFT 最短的实例；decode 实例单独按负载选（行 26）；**任何一项超 SLO 就返回 HTTP 429 拒绝**（行 27-29）；
6. 若最佳远程前缀显著优于所选实例的本地前缀（行 30），触发热块迁移。

**工程细节（§6.1）**：prefill 时间用离线数据训练的预测模型（Transformer 计算规律规则，误差小）；排队时间 = 队列里请求 prefill 时间之和；传输时间要考虑发送节点拥塞——这正需要热块复制来缓解。

## 5. 过载调度：early rejection 与负载波动

**负载怎么定义（§7.1）**：解耦架构里 prefill 与 decode 互不干扰，所以直接拿 SLO 满足度当负载：prefill 侧预测最大 TTFT 与 `l_ttft` 比，decode 侧预测最大 TBT 与 `l_tbt` 比。

**Early rejection（提前拒绝，§7.2）**：请求到达时，Conductor 用 prefill 与 decode 两池较大的那个负载决定是否接受——把 decode 侧的负载检查提前到 prefill 开始之前，避免“prefill 白算”。

**负载波动（§7.3,Fig.9-10a）**：直接 early rejection 会在 20 分钟窗口里出现 prefill/decode 两池负载反相波动。论文用四阶段解释：①两池都空 → 大量接受直到 prefill 满载；②prefill 产出的请求涌向 decode → decode 满载 → 开始拒绝 → prefill 负载下降；③decode 消化完 → 又大量接受 → prefill 满载；④循环。根源：基于当前 decode 负载的调度天然滞后（预测与实际执行之间有延迟）。

**基于预测的 early rejection（§7.4,Fig.10b）**：预判“prefill 完成后 decode 侧的负载”。论文当前用系统级预测：假设每个请求的 decode 阶段耗时统一为 $t_d$，把 t 时刻 prefill 完成的请求加入解码池、把已完成的移除，然后算所有 decode 实例的平均 TBT 与 `l_tbt` 之比作为预测负载。请求级预测（预测每个请求的输出长度）成本高、过载时精度差，留作未来工作。

## 6. 评测：50%-525% 与 75% 的数字口径

**实验设置（§8.1）**：为保护隐私与可复现，全部实验用 dummy LLaMA2-70B 模型（架构相同、权重无关）+ 回放真实 trace（23,000 条带时间戳的请求）。测试床：每节点 8×A800-80GB + 800Gbps RDMA。指标：P90 TTFT/TBT 与 SLO 上限之比（TTFT 上限 = 最低负载下 P90 的 10×，TBT = 5×，§2），超过即算失败；吞吐 = 满足 SLO 的有效请求率（goodput 口径：只有完整完成的请求才算数，§2）。

| 实验（§8.1） | 配置 | 结果 | 出处 |
| --- | --- | --- | --- |
| 公开数据集 ArXiv Summarization | Mooncake-[3P+1D] vs vLLM-[4M] | 吞吐 **+20%**，满足 SLO | Fig.11 |
| 公开数据集 L-Eval | 同上 | 吞吐 **+40%**（前缀缓存显著降低 prefill 时间） | Fig.11 |
| 模拟数据（16K/32K/64K/128K prompt） | Mooncake vs vLLM | 吞吐 **+50% 到 +525%**（vLLM 长上下文下 decode 被严重打断，甚至逐请求处理） | Fig.12 |
| 真实负载 | [10P+10D] vs 20 个 vLLM 实例 | Kimi 多处理 **75%** 的请求（TTFT 分布几乎一致，TBT 达标率 ~100% vs vLLM 57%） | Fig.13 |
| 调度器对比（§6.2） | 8P+8D 集群回放 23,000 请求 | KVCache-centric 调度在平均 TTFT 与 SLO 达标率上同时优于随机/纯负载均衡 | Fig.8 |

**版本口径提醒**：本文核对的是 arXiv:2407.00079 v3。FAST '25 正式版论文的数字口径略有不同（摘要中：模拟场景吞吐提升 59%-498%，A800/H800 集群上 Kimi 分别多处理 115%/107% 请求）——两组数字都来自 Mooncake 团队但对应不同论文版本与实验口径，引用时务必注明版本。

## 🕰️ 原文时代 vs 当前工程：arXiv 版与 FAST '25 版

Mooncake 是 Kimi 的**生产平台**（论文明示"currently the primary platform for serving Kimi"，§1.2），这是它与学术原型最大的不同。2026 年 8 月复核（来源标注如下）：

- **代码开源**：官方仓库 github.com/kvcache-ai/Mooncake 已公开（论文 §1.2 承诺 trace 开源），含 KVCache 传输引擎与调度器实现；社区有基于 Mooncake 的 PD 解耦 + 缓存复用集成实践（SGLang/vLLM 的 Mooncake 集成讨论）——工程事实以官方仓库 README 与文档为准（访问日期 2026-08-15）；
- **论文版本差异**：arXiv v3（2025-02 更新）与 FAST '25 正式版的评测数字不同（见第 6 节表格），引用时按版本对号入座；
- **思想扩散**：Mooncake 的“全局缓存池 + cache-aware 调度”与 Sarathi 系（chunked prefill）共同塑造了 2025-2026 年 PD 解耦系统的设计语言；后续同主题工作（如以 KV cache 为中心的调度研究）普遍引用本文。

**结论边界**：论文的实验用 dummy LLaMA2-70B 而非 Kimi 真实模型（§1.2），真实 Kimi 的具体架构细节（如专家并行、模型结构）未公开；“75% 更多请求”是 Kimi 当时真实集群的增益，不能直接外推到其他负载。

## 7. 常见误读与错误做法

- **误读 1：“Mooncake = 简单的 prefill/decode 分离。”** 错。PD 解耦只是起点（沿袭 Splitwise）；核心贡献是把缓存当一级公民：全局 KV cache 池、cache-aware 调度、热块复制、以及过载场景的拒绝策略——后两者是解耦系统独有的新问题（§6-7）。
- **误读 2：“前缀缓存命中率高就万事大吉。”** 错。命中率高意味着复用多、prefill 计算少，但远程缓存传输会挤占 TTFT（等缓存到达才开算）且热块会造成网络拥塞（§1.1,§6.1）——所以才有“阈值分支”（本地算 vs 远端搬）与热块复制。缓存命中与 SLO 是两本账。
- **误读 3：“early rejection 就是过载时拒绝请求。”** 部分对。难点在“拒绝多少、何时拒绝”：无脑拒绝会导致两池负载反相波动、资源利用率暴跌（§7.3 四阶段分析）；正确做法是基于预测的拒绝（§7.4）。
- **错误做法 1（复现实验）**：拿 FAST '25 摘要的 115%/107% 与 arXiv 版的 75% 混用。两组数字口径不同（正式版/预印版、A800/H800），引用必须注明版本（见第 6 节表格）。
- **错误做法 2（部署理解）**：把 prefill 节点显存规划成“要装下所有并发请求的 KV”。Mooncake 的 layer-wise prefill 让 KV 不驻留 GPU 显存（§5.2），prefill 节点的显存只需装下一个请求——按旧思路规划 prefill 池会浪费一半资源。

## 📝 总结

1. **KV cache 是调度的一等公民**：Mooncake 把 CPU/DRAM/SSD/RDMA 组织成全局缓存池，四步工作流（复用 → 增量 prefill → 按层流式传输 → 解码）让“缓存复用”与“SLO 满足”同时可调度。
2. **Conductor 三件套**：cache-aware 实例选择（Algorithm 1，阈值分支权衡“本地算 vs 远端搬”）、启发式热块复制（顺路迁移）、基于预测的 early rejection（压住过载下的负载波动）。
3. **真实系统数字**：长上下文模拟场景 +50%-525%、真实负载多处理 75% 请求（Kimi 生产平台证据；两版本数字差异见第 6 节版本口径提醒）。

## 🎯 自我检验清单

- [ ] 能画出并解释四步请求工作流（复用、增量 prefill、按层传输、解码）各步的输入输出（§3,Fig.4）。
- [ ] 能按 Algorithm 1 走一遍：什么条件下走 cache-aware 分支、什么条件下算传输时间、什么条件触发热块迁移（行 10/16/30）。
- [ ] 能解释 CPP 与 SP/跨节点 TP 的取舍（通信边界次数、弹性复杂度），以及 layer-wise prefill 为什么让 prefill 调度不用管显存。
- [ ] 能复述 early rejection 负载波动的四阶段循环（§7.3）与系统级预测的做法（统一 $t_d$，§7.4）。
- [ ] 能列出至少 4 个评测数字（20%/40%/50-525%/75%）并注明版本口径（arXiv v3 vs FAST '25）。
- [ ] 能说出“缓存命中率高”为什么不一定满足 SLO（传输时间、拥塞，§6.1）。

## 📚 参考资料

- Mooncake 原文（FAST 2025）：https://www.usenix.org/conference/fast25/presentation/qin 与 arXiv:2407.00079（v3）
- 官方代码与 trace:https://github.com/kvcache-ai/Mooncake
- 站内关联：[splitwise-notes](/AIInfraGuide/papers/splitwise-notes)（PD 解耦）、[pagedattention-notes](/AIInfraGuide/papers/pagedattention-notes)（KV cache 分页）、[orca-notes](/AIInfraGuide/papers/orca-notes)（迭代级调度）
