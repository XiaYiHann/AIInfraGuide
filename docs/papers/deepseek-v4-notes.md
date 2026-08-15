---
title: "DeepSeek-V4 解读：百万 token 上下文背后的架构与训练账本"
description: "拆解 DeepSeek-V4 技术报告：CSA+HCA 混合注意力如何把 1M 上下文下的单 token 推理 FLOPs 压到 V3.2 的 27%、KV cache 压到 10%,以及 mHC、Muon、两阶段后训练与 32T/33T token 的训练账本。"
pubDate: 2026-08-09
originalUrl: "https://arxiv.org/abs/2606.19348"
sourceType: "paper"
originalAuthor: "DeepSeek-AI"
tags: ["DeepSeek-V4", "稀疏注意力", "KV Cache", "MoE", "Muon", "长上下文"]
---

> 原文：[DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/abs/2606.19348)(DeepSeek-AI,arXiv 2606.19348 v1,提交于 2026-04-26,preview 版技术报告;本文访问日期 2026-08-15)

把上下文拉到 100 万 token(约 75 万英文单词，能塞下十几本长篇小说)，传统 Transformer 的注意力会"算不起也存不起"。DeepSeek-V4 给出的答案是**两层压缩注意力(CSA + HCA)交替排列**：在 1M 上下文下,**V4-Pro(总参数 1.6T、单 token 激活 49B)的单 token 推理 FLOPs 只有 DeepSeek-V3.2 的 27%,KV cache 只有 V3.2 的 10%**；小一号的 V4-Flash(284B 总参、13B 激活)更是压到 **10% FLOPs、7% KV cache**(以上数字均为报告原文口径，按等效 FP8 FLOPs 计)。更夸张的是，报告称其 KV cache 相对常见的 BF16 GQA8 注意力基线只有约 **2%**。

也就是说，V4 把"注意力省下的钱"反手花在了更大的 MoE 上——参数量从 V3.2 的 671B 涨到 1.6T,激活参数从 37B 涨到 49B,1M 上下文的账单反而更便宜。本文按"问题 → 架构 → 训练 → 后训练 → 推理系统 → 评测 → 意义"的顺序精读这份报告，所有数字标注来源口径:「报告」= arXiv 原文、「模型卡」= HuggingFace 官方模型卡、「公告」= DeepSeek API 官方发布博客、「解读」= 第三方解读(标注作者)。

<!-- more -->

## 📑 目录

- [1. 为什么需要：长上下文的注意力成本](#1-为什么需要长上下文的注意力成本)
- [2. 架构:CSA 与 HCA 混合注意力](#2-架构csa-与-hca-混合注意力)
- [3. mHC 与 Muon 带来更稳的残差与更快的收敛](#3-mhc-与-muon-带来更稳的残差与更快的收敛)
- [4. 预训练账本:32T token 与训练稳定性](#4-预训练账本32t-token-与训练稳定性)
- [5. 后训练：领域专家与蒸馏合流](#5-后训练领域专家与蒸馏合流)
- [6. 推理系统：百万 token 的工程账](#6-推理系统百万-token-的工程账)
- [7. 评测与局限：自报数字怎么读](#7-评测与局限自报数字怎么读)
- [8. 意义：开源模型把 1M 上下文变成默认](#8-意义开源模型把-1m-上下文变成默认)
- [📝 总结](#-总结)
- [🎯 延伸思考：自我检验清单](#-延伸思考自我检验清单)
- [📚 参考资料](#-参考资料)

## 🗺️ 原文阅读地图

这篇报告同时是"算法报告"与"系统报告":算法上提出 CSA/HCA 混合压缩注意力与 mHC 残差,系统上给出了 1M 上下文的训练与推理账本。本文选择性精讲如下,避免把中文解读误当成原文全貌。

| 原文单元 | 处理深度 | 本文位置与理由 | 来源锚点 |
| --- | --- | --- | --- |
| §1 Introduction(测试时扩展的动机、平方复杂度瓶颈、27% FLOPs 与 10% KV 的效率承诺) | 精讲 | 第 1 节,先量化问题再谈方案 | §1;Fig. 1 右半 |
| §2 Architecture(继承自 V3 的 MoE/MTP、2.2 mHC、2.3 CSA 与 HCA 细节) | 精讲 | 第 2、3 节,机制卡 1-3,含公式逐项拆解与数值算例 | §2.2;§2.3;Fig. 2;Fig. 3 |
| §3 预训练(训练稳定性、Muon 优化器、成本账) | 精讲(成本) | 第 3、4 节,含 Algorithm 1 走查 | §3;Algorithm 1 |
| §4 Pre-Training(数据、序列长度渐进、Base 评测) | 简述 | 第 4 节引用关键设置与 Table 1 结论,不逐行展开评测 | §4;Table 1 |
| §5 Post-Training(领域专家、蒸馏、标准评测与真实任务) | 简述 | 第 5 节只保留改变结论的对比(Table 7 与真实任务胜率) | §5.3;Table 7;§5.4 |
| 附录 A/B(作者列表、评测细节) | 不展开 | 不改变本文的机制承诺 | Appendix A/B |

📌 **本文承诺**：读完后，你应该能手算 1M 上下文下 CSA($m=4$)与 HCA($m'=128$)把 KV 条目数压到什么量级，解释 27% FLOPs、10% KV cache 这两个数字从哪来，并说清 mHC 与 Muon 各自解决了什么问题。

## 1. 为什么需要：长上下文的注意力成本

**问题是：长上下文到底贵在哪?** 自回归模型每生成一个新 token,都要"回顾"一遍此前所有的 token。这个回顾(注意力)的时间和显存开销，随序列长度 $n$ **平方级**上涨:

- **计算**：每个 query 要和所有 key 算相关分，总操作量 $O(n^2)$(序列多长，两两配对多少次);
- **显存**:KV cache(键值缓存，把算过的 K/V 向量存下来供后续复用，站内 [1.1-LLM推理基础](/AIInfraGuide/inference/模块四-推理优化/第1章-llm推理基础/11-llm推理基础) 有完整账本)随 $n$ **线性**增长——$n$ 从 4K 涨到 1M(250 倍),KV cache 涨 250 倍。

把 1M token 全部塞进普通稠密注意力：光 KV cache 就大到单机装不下，每个新 token 还要扫一遍全部 KV 做计算。这不是"优化一下"能解决的，是结构性的。

<img src="/AIInfraGuide/images/deepseek-v4-fig1-efficiency-benchmark.png" alt="V4 系列效率与基准对比" style="max-width: 75%; display: block; margin: 0 auto;" />
*图源：DeepSeek-V4 技术报告 Figure 1(arXiv:2606.19348)*

> 打个比方：注意力就像**图书馆查资料**。$O(n^2)$ 的意思是——你每写一句话，都要把馆里每一本书重新翻一遍找相关内容;KV cache 是你做的读书笔记，笔记量随馆藏线性增长。馆藏从 1 个书架涨到 250 个书架，你每次写作都要翻 250 个书架，笔记也堆满房间。常规做法是"换更大的房间"(加显存、加卡),DeepSeek 的选择是**把书先压缩成摘要卡片，再决定翻哪几张**。

**为什么这个问题现在非解不可?** 报告的动机很直接：推理模型(reasoning model)的测试时扩展(test-time scaling,推理时花更多 token 思考换取更高准确率)让单次会话的序列长度暴涨；同时 agent 类任务(多轮工具调用、读整个代码仓库、跨文档分析)天然需要超长上下文。报告原文的原话是：vanilla attention 的平方级复杂度是"prohibitive bottleneck"(令人却步的瓶颈)。谁先把长上下文做便宜，谁就解锁了测试时扩展和 long-horizon 任务(长时间跨度、多轮交互的任务)的下一个台阶。

## 2. 架构：CSA 与 HCA 混合注意力

> 🔗 **来源锚点**：本节机制拆解对应报告 §2.3(CSA/HCA 压缩与稀疏选择)与 §2.1(继承自 V3 的 MoE/MTP 配置)；压缩权重公式与索引打分公式均按报告原文转写。

**核心思想一句话：把 KV 序列"压缩"后再做注意力，压缩得轻的层做稀疏选择，压缩得狠的层做全量扫描——两类层交错排列。** 报告设计了两种注意力并交替使用(Pro 前两层用纯 HCA,之后 CSA/HCA 交错;Flash 前两层用纯滑动窗口注意力，之后交错):

<img src="/AIInfraGuide/images/deepseek-v4-fig2-architecture.png" alt="DeepSeek-V4 总体架构" style="max-width: 75%; display: block; margin: 0 auto;" />
*图源：DeepSeek-V4 技术报告 Figure 2(arXiv:2606.19348)*

- **CSA(Compressed Sparse Attention,压缩稀疏注意力)**：先把每 $m=4$ 个 token 的 KV 压缩成 1 条，再在压缩后的条目上做 DSA(DeepSeek Sparse Attention,稀疏注意力：每个 query 只选分数最高的 top-$k$ 条)。Pro 的 top-$k$=1024,Flash 的 top-$k$=512。
- **HCA(Heavily Compressed Attention,重度压缩注意力)**：把每 $m'=128$ 个 token 的 KV 压成 1 条，然后对全部压缩条目做**稠密**注意力——因为压缩后序列极短，稠密扫一遍也便宜。

<img src="/AIInfraGuide/images/deepseek-v4-fig3-csa.png" alt="CSA 的压缩与稀疏选择流程" style="max-width: 75%; display: block; margin: 0 auto;" />
*图源：DeepSeek-V4 技术报告 Figure 3(arXiv:2606.19348)*

**压缩是怎么做的?** 压缩不是简单平均，而是"加权合并"。对 HCA,每个压缩条目是:

$$C_i^{\text{Comp}} = \sum_{j=mi}^{m(i+1)-1} \underbrace{S_j}_{\text{每个 token 的合并权重}} \odot \underbrace{C_j}_{\text{原始 KV 条目}}$$

其中权重 $S_j = \mathrm{Softmax}_{\text{row}}(Z_j + B)$,$Z_j$ 是模型学出来的"这个 token 的信息值打分",$B$ 是可学习的位置偏置。直觉：**一个块里，重要的 token 多保留，不重要的少保留**——像做摘要卡片时，关键句抄全、废话略写。CSA 的版本还带一点"跨块重叠"(第 $i$ 条压缩条目会参考前一块末尾的 token)，让块边界的信息不丢失。

**稀疏选择怎么做?** CSA 里还有一个"闪电索引器"(Lightning Indexer)：先用低秩投影给每个 query 生成若干个小号索引 query,和压缩后的索引 key 算相关分，取 top-$k$。索引打分公式:

$$I_{t,s} = \sum_{h=1}^{n_h^I} \underbrace{w^I_{t,h}}_{\text{每个索引头的权重}} \cdot \mathrm{ReLU}(\underbrace{\mathbf{q}^I_{t,h}}_{\text{索引 query}} \cdot \underbrace{K_s^{\text{IComp}}}_{\text{压缩后的索引 key}})$$

即：多个"小头"各自投票，加权求和得到"第 $s$ 个压缩块与 query $t$ 的相关分",只保留 top-$k$ 进主注意力。

**数值算例(报告配置):** 序列 1M token 时——CSA($m=4$)把 KV 条目数压到 250K;HCA($m'=128$)压到约 7.8K;再叠加 CSA 每 query 只取 top-1024,注意力计算量从"扫 1M 条"变成"扫 1024 条 + 局部窗口"。这就是 27% FLOPs、10% KV cache 的来源。

**还有几个配套细节(报告 2.3.3):**

- **滑动窗口分支(Sliding Window Attention)**：因为压缩块是"回头看"的，query 看不到自己所在块内部的 token,而最近的 token 通常最重要——所以每个 CSA/HCA 层都额外保留最近 $n_{\text{win}}=128$ 个 token 的**未压缩** KV,与压缩条目一起参与注意力。
- **Attention Sink(注意力汇)**：给每个注意力头加一个可学习的"兜底分值"加到 softmax 分母上，让总注意力权重可以不等于 1(甚至可以接近 0)，防止某些头被迫把分数摊给无关内容。
- **混合精度 KV**:RoPE(旋转位置编码，给向量加位置信息的变换)维度用 BF16,其余维度用 FP8——KV cache 相比纯 BF16 再省近一半。
- **QK 归一化**：注意力的 query 和压缩 KV 在做注意力前过一层 RMSNorm,防止 logit 爆炸(这个细节后面 Muon 一节还会用到)。

💡 **提示**：压缩注意力不是"无损压缩"——它牺牲了长距离的逐 token 细节，换取 KV/FLOPs 数量级下降。报告用三类机制补细节：滑动窗口保近期细粒度、CSA 的 top-$k$ 选择保"相关"的远处内容、HCA 保"全局大致印象"。**牺牲的是长距离精确检索能力，换取的是一百万 token 真的能用起来**；评测里 1M 长度性能相对 128K 有可见退化(见第 7 节)，就是这份代价的实证。

## 3. mHC 与 Muon 带来更稳的残差与更快的收敛

> 🔗 **来源锚点**：mHC 对应报告 §2.2，Muon 对应报告 Algorithm 1(含完整优化器伪代码)；混合 ZeRO 与随机舍入量化见报告 §3.4(训练系统实现)。

注意力之外，报告还有两处"地基级"改动。

### 3.1 mHC:把残差连接"钉"在稳定流形上

传统残差连接(ResNet 以来的标配)是 $X_{l+1} = X_l + \mathcal{F}_l(X_l)$——每层输出和输入直接相加。Hyper-Connections(超连接，HC)把它推广成三条可学习线性映射:

$$X_{l+1} = \underbrace{B_l X_l}_{\text{残差变换:上一层的状态}} + \underbrace{C_l \mathcal{F}_l(A_l X_l)}_{\text{层输出:先投影再进层,输出再投影}}$$

$B_l \in \mathbb{R}^{n_{\text{hc}}\times n_{\text{hc}}}$ 把残差流的宽度从 $d$ 扩到 $n_{\text{hc}}\times d$(V4 取 $n_{\text{hc}}=4$,即残差流变宽 4 倍),$A_l$、$C_l$ 负责进出投影。HC 的好处是给模型一条"额外缩放轴",但堆很多层时数值不稳定。

> 打个比方：普通残差像**接力传话**——每棒都"把原话加上自己的发挥"传给下一棒，层数一多，传着传着就跑偏(信号被不断放大或抵消)。mHC 的做法是给传话人立规矩：你只能"重新分配"原话，不能凭空加量减量。

mHC 的规矩就是**把 $B_l$ 约束成双随机矩阵(doubly stochastic matrix,每行和为 1、每列和为 1、所有元素非负)**:

$$B_l \in \mathcal{M} \coloneq \{M \mid M\mathbf{1}_n = \mathbf{1}_n,\ \mathbf{1}_n^T M = \mathbf{1}_n^T,\ M \geqslant 0\}$$

这个约束保证 $B_l$ 的谱范数(矩阵对向量的最大放大倍数)$\|B_l\|_2 \le 1$——残差变换"不扩张"(non-expansive)，信号在深层传播时不会越传越炸；而且这个集合对矩阵乘法封闭，深层堆叠也稳定。$A_l$、$C_l$ 则用 Sigmoid 约束为非负有界，避免信号相消。约束通过 **Sinkhorn-Knopp 算法**(交替做行归一化、列归一化)施加，报告取 $t_{\max}=20$ 次迭代。

**这笔账怎么算?** mHC 增加了激活显存和流水线通信量，报告通过算子融合 + 选择性重计算 + 调整 DualPipe(双向流水线并行：训练批次从流水线两端同时灌入)1F1B(one-forward-one-backward 调度：每卡交替执行一个前向、一个后向，填满流水线气泡)重叠，把 mHC 的额外墙钟开销压到**重叠后流水线单阶段的 6.7%**(报告 3.4.2)。**牺牲的是每层多三组小映射的计算与工程复杂度，换取的是 61 层 × 1.6T 参数能稳定训练不炸。**

### 3.2 Muon:动量 + 正交化，替换 AdamW

报告用 **Muon 优化器**替换大部分模块的 AdamW:每个权重矩阵的更新前，先把"动量 + 当前梯度"做一次**近似正交化**(Newton-Schulz 迭代逼近 SVD(奇异值分解：把矩阵拆成 U、S、V 三个矩阵的通用分解)的 $UV^T$)，再按固定 RMS 重缩放。算法主干(报告 Algorithm 1):

$$
\begin{aligned}
M_t &= \mu M_{t-1} + G_t &&\text{① 动量累积(}\mu=0.95\text{)} \\
O'_t &= \mathrm{HybridNewtonSchulz}(\mu M_t + G_t) &&\text{② Nesterov 技巧 + 正交化} \\
O_t &= O'_t \cdot \sqrt{\max(n,m)} \cdot \gamma &&\text{③ 重缩放更新 RMS(}\gamma\text{使 RMS}=0.18\text{)} \\
W_t &= W_{t-1}(1-\eta\lambda) - \eta O_t &&\text{④ 权重衰减(}\lambda=0.1\text{)并更新}
\end{aligned}
$$

直觉:**AdamW 是给每个参数单独配"步长调节器";Muon 是对整个矩阵做"旋转校准"——把梯度里沿相关性强的方向重复走的分量去掉，让每次更新都朝新方向推进**，所以收敛更快、训练更稳。正交化用混合 Newton-Schulz:10 次迭代分两段，前 8 步用系数 $(3.4445, -4.7750, 2.0315)$ 快速逼近，后 2 步换 $(2, -1.5, 0.5)$ 精修到奇异值恰为 1。

**保留 AdamW 的部分**(报告明确列出):embedding、预测头、RMSNorm 权重、mHC 的静态偏置与门控因子。**不用 QK-Clip**(Kimi K2 在 Muon 下需要它防止注意力 logit 爆炸，Moonshot 的做法):V4 因为 2.3.3 节已经给 query 和压缩 KV 做了 RMSNorm,logit 不会爆，所以省掉这个技巧。

💡 **提示**:Muon 有个工程上的"不合群"之处——它需要**完整梯度矩阵**才能做正交化，而 ZeRO(把优化器状态切到多卡、站内 [第5章-ZeRO系列](/AIInfraGuide/distributed/模块三-分布式训练/第5章-zero系列))按元素切分参数。报告的解法是"混合 ZeRO":稠密参数限制 ZeRO 并行度、用背包算法(knapsack)分配矩阵、桶补齐(补齐开销 <10% 显存);MoE 参数则展平后整体均分。数据并行 rank 间同步的 MoE 梯度用**随机舍入量化到 BF16**，通信量减半，再用 all-to-all(全交换：每张卡都要给其他每张卡各发一份数据) + FP32 本地求和避免低精度累加误差。

### 3.3 MoE 配置账本

V4 沿用 DeepSeekMoE(细粒度路由专家 + 共享专家)，但有几处调整：亲和度激活函数从 Sigmoid 换成 $\mathrm{Sqrt}(\mathrm{Softplus}(\cdot))$;去掉路由目标节点数约束；前 3 层 FFN 用 **Hash 路由**(按 token ID 的哈希函数决定专家，省路由计算)。两份模型的具体账本:

| 📊 配置项 | V4-Flash | V4-Pro | V3.2(对照) | 来源口径 |
|---|---|---|---|---|
| 总参数 / 激活参数 | 284B / 13B | **1.6T / 49B** | 671B / 37B | 报告 |
| Transformer 层数 / 隐藏维 | 43 / 4096 | 61 / 7168 | — | 报告 |
| CSA 压缩率 $m$ / top-$k$ | 4 / 512 | 4 / 1024 | — | 报告 |
| HCA 压缩率 $m'$ | 128 | 128 | — | 报告 |
| 路由专家数(每层)/ 每 token 激活 | 256 / 6 | 384 / 6 | — | 报告 |
| 滑动窗口 $n_{\text{win}}$ | 128 | 128 | — | 报告 |
| mHC 扩展因子 $n_{\text{hc}}$ / Sinkhorn 迭代 | 4 / 20 | 4 / 20 | — | 报告 |

📌 **关键点**：激活参数 49B 是目前头部开源模型里最高的(Labonne 解读对比：K2.6 激活 32B、GLM-5.1 激活 40B、Qwen3.5 激活 17B)。别家都在压激活参数，DeepSeek 反着来——**注意力省下来的钱，买更多的激活专家**，因为激活参数直接决定单 token 算力，而注意力省下的占比更大。

## 4. 预训练账本：32T token 与训练稳定性

### 4.1 数据：长文档优先 + 防"模型崩溃"

报告的数据策略(4.1 节):

- 在 V3 语料基础上重建，总量 **Flash 32T token、Pro 33T token**(报告 4.2.2);
- **长文档优先**：重点收集科学论文、技术报告等"有独特学术价值"的长文档——长上下文模型需要真长文喂;
- 过滤"批量自动生成、模板化"的网页内容，报告明确说这是为了**降低模型崩溃(model collapse)风险**(模型吃自己生成的文本越吃越差);
- 数学和代码仍是核心;mid-training 阶段加入 agentic 数据(工具调用轨迹等)强化编码;
- 多语言语料扩容，补长尾文化知识；沿用 V3 的 token-splitting(把超长 token 拆分)与 FIM(Fill-in-Middle,填空式训练目标，强化代码/文本中段补全)；新增**样本级注意力掩码**(pack 时跨文档不互相可见)。

### 4.2 训练设置：序列长度 4K → 1M 渐进

- **序列长度渐进**:4K → 16K → 64K → 1M(Flash 与 Pro 相同);
- **稀疏注意力晚引入**：前 **1T token** 用稠密注意力"热身",到 64K 序列阶段才引入稀疏注意力，并先短训一段专门"热身"CSA 的闪电索引器;
- **Batch 调度**:batch 从小学起，Flash 最大到 75.5M token、Pro 最大到 94.4M token;
- **学习率**:Flash 峰值 $2.7\times10^{-4}$ 衰减到 $2.7\times10^{-5}$;Pro 峰值 $2.0\times10^{-4}$ 衰减到 $2.0\times10^{-5}$;前 2000 步线性热身，cosine 衰减;
- **MTP(Multi-Token Prediction,多 token 预测：一个位置同时预测后面多个 token)**：深度 1,权重 0.3,进入学习率衰减后降为 0.1——MTP 配置与 V3 完全一致。

### 4.3 训练不稳定：两个"知其然不知其所以然"的招

训练万亿级 MoE,报告坦承遇到显著不稳定问题(loss spike,损失值尖峰)，简单回滚只能救一时。他们定位到 spike 与 MoE 层异常值强相关、路由机制会放大异常值，于是上两招(报告 4.2.3,**并明说机制尚未被完全理解**):

- **Anticipatory Routing(前瞻路由)**：把主干网络和路由网络的更新解耦——第 $t$ 步用当前参数 $\theta_t$ 算特征，但路由索引用历史参数 $\theta_{t-\Delta t}$ 算(提前取数据、提前算好缓存)。额外墙钟开销约 20%,但只在检测到 loss spike 时自动启用一小段，之后回到标准训练，总体开销可忽略;
- **SwiGLU Clamping**：把 SwiGLU 的线性分量夹在 $[-10,10]$、门控分量上限夹在 10,直接压掉异常值。

⚠️ **注意**：这两招是经验性的。报告原话是"comprehensive theoretical understanding of their underlying mechanisms remains an open question"(其机制的综合理论理解仍是开放问题)——解读文章时别把"有效"说成"懂了"。

### 4.4 成本账：报告没说的部分

**训练硬件、总算力、电力、成本：报告全部未公开。** 报告只提了一句：细粒度 EP 方案"在 NVIDIA GPU 和华为 Ascend NPU 两个平台都验证过"(3.1 节)。第三方报道口径相互冲突，此处明确标注:

- Labonne 解读转述 ChinaTalk/36Kr 报道：**训练实际跑在 NVIDIA Hopper 上**,2025 年中尝试迁移华为 Ascend 失败，绕路 Nvidia 耽搁数月;
- kenhuangus 解读声称 V4 是"首个完全跑在华为 Ascend 950PR 上的前沿模型"——**与该报道直接冲突**;
- 报告本身对"用谁的卡训练的"只字未提。

以上第三方说法均**「待核实」**，写代码时请以报告原文为准：报告确认的只有"EP 内核在 Ascend 上验证过"和"权重发布为 FP4+FP8 混合精度，推理可在国产芯片上跑(解读口径)"。

## 5. 后训练：领域专家与蒸馏合流

> 🔗 **来源锚点**：本节对应报告 §5 Post-Training(领域专家、蒸馏与标准评测)；关键对比数字见报告 §5.3 与 Table 7。

报告 5.1 节把后训练从 V3.2 的"混合 RL"换成**两阶段范式**，这是本报告除架构外最值得注意的方法论变化:

1. **第一阶段：培养领域专家**。把 base 模型 fork 成多个专用专家——数学、代码、agent、指令遵循等。每个专家先做领域 SFT(有监督微调，打底基础能力)，再做 GRPO(Group Relative Policy Optimization,V3 引入的 RL 算法，组内归一化优势，不需要独立的 critic 价值网络)强化。每个领域一条独立 RL 流水线，这是配方里最贵的一半(Labonne 解读语)。
2. **第二阶段：蒸馏合流**。用一个"学生"模型，通过 **On-Policy Distillation(OPD,在线策略蒸馏：学生用自己的采样轨迹学习)** 对齐 **超过十个**专家教师:

$$\mathcal{L}_{\text{OPD}}(\theta) = \sum_{i=1}^{N} \underbrace{w_i}_{\text{各专家权重}} \cdot D_{\mathrm{KL}}\Big(\underbrace{\pi_\theta \| \pi_{E_i}}_{\text{学生分布对教师分布的反向 KL}}\Big)$$

> 打个比方：**十位特级教师教一个学生**。数学老师、代码老师、写作老师各教各的领域；学生自己做题(自己采样轨迹)，每道题所有老师按"标准答案分布"打分，学生按反向 KL 朝老师们靠拢。难点在于"多位老师同时教",学生要学会**看题选老师**——做数学题时主要跟数学老师对齐。

**为什么用反向 KL 蒸馏而不是再跑一轮混合 RL?** 报告与解读的一致说法：**混合 RL 的 reward 混合会引入 reward hacking(钻奖励函数漏洞)与能力互相干扰**；专家各自训到收敛再合流，避免"一锅炖"。学生只在概率分布层面吸收多个专家，而不是做权重合并(weight merging 常见性能塌陷)。

**工程上最难的一点：全词表蒸馏(Full-Vocabulary OPD)。** 常规 OPD 只对"学生采样到的那个 token"算 KL(便宜但梯度方差大、不稳)；报告坚持**每个位置对完整词表(>10 万)算 KL**，代价是 logits 张量巨大。工程解法(报告 5.2.2):

- 教师前向时**只缓存最后一层隐状态**，训练时现过预测头重建 logits,避免物化超大 logits;
- 训练样本按教师索引排序，**同一时刻只有一位教师的预测头驻留显存**;
- 精确 KL 用定制 TileLang kernel 算。

另外两个后训练基础设施点:

- **FP4 量化感知训练(QAT)**：对 MoE 专家权重(显存大头)和 CSA 索引器的 QK 路径做 **MXFP4** 量化，训练时模拟量化、推理/RL 采样时直接用原生 FP4 权重(保证线上行为一致)。报告强调 **FP4→FP8 反量化无损**:FP8(E4M3)比 FP4(E2M1)多 2 个指数位，动态范围更大，只要 FP4 子块尺度比不超过阈值，细粒度尺度信息能完整装进 FP8。索引分数从 FP32 量化到 BF16,top-$k$ 选择器**提速 2×,KV 召回率保持 99.7%**(报告 5.2.1);
- **可抢占、容错的采样服务**：集群用可抢占调度，任何任务可能随时被抢；报告实现了 **token 级 WAL(Write-Ahead Log,预写日志)**——每生成一个 token 立即落盘，被抢占时保存 KV cache,恢复后接着生成。报告特别指出"从零重生成"在数学上不正确(短回复更容易活下来，会产生长度偏差)，所以必须续跑而不是重跑。

## 6. 推理系统：百万 token 的工程账

> 🔗 **来源锚点**：本节工程账本综合自报告 §2.3.3(其他实现细节：KV 归一化、精度优化等)与 §3.4(训练与推理系统实现)；报告未完整公开的部分已标注。

架构省的是"理论 FLOPs",要把 1M 上下文变成日常服务，还需要一套系统。报告第 3 章主要讲四件事:

### 6.1 异构 KV cache:一个模型，四类 KV

V4 的 KV cache 不是一个均匀的大数组，而是多种条目混在一起：CSA 压缩条目(每 $m$ token 一条)、HCA 压缩条目(每 $m'$ token 一条)、SWA 的未压缩近端条目、以及"还没凑够一个压缩块"的尾部未压缩状态。报告 3.5.1 节的设计:

- **State Cache(状态缓存)**:SWA 近端 KV + 未压缩尾部，按序列固定分配小块——因为 SWA 本来就只依赖最近 $n_{\text{win}}$ 个 token,可以当"状态"而不是"历史"管理;
- **Classical KV Cache**:CSA/HCA 压缩条目。每个 cache block 覆盖 $\mathrm{lcm}(m, m') = 128$ 个原始 token,包含 $k_1 = \mathrm{lcm}/m = 32$ 条 CSA 压缩 + $k_2 = \mathrm{lcm}/m' = 1$ 条 HCA 压缩——取最小公倍数是为了让两类压缩在块边界对齐(算例：128 token 一块，32:1 的条目配比)。

<img src="/AIInfraGuide/images/deepseek-v4-fig6-kv-cache-layout.png" alt="DeepSeek-V4 异构 KV cache 布局" style="max-width: 75%; display: block; margin: 0 auto;" />
*图源：DeepSeek-V4 技术报告 Figure 6(arXiv:2606.19348)*

**为什么不能直接用 PagedAttention?** 报告 3.5.1 明确说，混合注意力**违反了 PagedAttention 的底层假设**(站内 [2.1-PagedAttention](/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/21-pagedattention) 讲过它的分页模型)：一是 SWA 有自己的淘汰策略(老条目要丢)，二是高性能注意力 kernel 有对齐约束，没法统一塞进"定长页"。所以自研了上面的布局 + 与稀疏注意力 kernel 协同设计(块大小取 $\mathrm{lcm}$ 的倍数以对齐 cache line)。

### 6.2 磁盘 KV cache:省掉重复 prefill

1M 上下文的 prefill(预填充，把整个 prompt 算一遍出 KV)很贵。报告 3.5.2 的方案：**把压缩 KV 条目全部落盘**，命中共享前缀(比如多轮对话的前几轮、团队共享的代码库)时直接读盘复用，不用重算。SWA 条目没压缩、体积大(报告给数:**SWA KV 体积约为压缩 CSA/HCA KV 的 8 倍**)，于是给了三种策略按场景选:

| 🎯 策略 | 做法 | 牺牲 | 换取 |
|---|---|---|---|
| Full SWA Caching | SWA KV 全存盘 | 写盘量大、SSD 写放大(每次命中只读尾部一小截) | 计算零冗余，命中即用 |
| Periodic Checkpointing | 每 $p$ 个 token 存一次 SWA 检查点，命中后重算尾部 | 少量重算 | 存储与计算按 $p$ 可调 |
| Zero SWA Caching | 完全不存 SWA | 重算最后 $n_{\text{win}}\cdot L$ 个 token(L=层数) | 存储最小 |

### 6.3 专家并行：通信藏进计算里

MoE 的专家并行(EP,专家分布在多卡，见站内 [6.4-数据并行与专家并行](/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/64-数据并行与专家并行))在 V4 里被重做成**细粒度 wave 调度 + 单 fused mega-kernel**(开源为 DeepGEMM 里的 MegaMoE)：一个 MoE 层的 Dispatch(把 token 发给目标专家)/Combine(收回结果)两次通信，和 Linear-1/Linear-2 两次计算，按 wave(一小撮专家为一波)流水化重叠——当前 wave 在算、下一 wave 的 token 在传、完成的专家结果在回收，三者并行。实测(报告 3.1):

- 通用推理负载比强非融合基线 **快 1.50~1.73×**;
- RL rollout、高速 agent 服务等延迟敏感场景**最高 1.96×**;
- 两个平台都验证：NVIDIA GPU 和华为 Ascend NPU。

报告还给硬件厂商算了笔平衡账($C$ 为算力，单位 FLOPs/s;$B$ 为互联带宽，单位 Byte/s)：每 token-专家对需要 $6hd$ FLOPs(SwiGLU 三投影)、只传 $3h$ 字节(FP8 Dispatch + BF16 Combine)，因此**当算力带宽比 $C/B \le 2d = 6144$ FLOPs/Byte 时通信可以被完全藏住**——即每 GBps 互联带宽足以掩盖 6.1 TFLOP/s 的算力，带宽超过这个平衡点后继续加带宽收益递减。

### 6.4 确定性：为了 RL 的正确性与可调试

报告 3.3 节强调 **batch-invariant(批不变性：一个 token 的输出与它在 batch 里的位置无关，bit 级一致)与确定性(逐位可复现)**——这既是调试 loss spike 的工具，也是 5.2.3 节 WAL 续跑正确性的前提。代价很具体：不能用 split-KV/split-k(把一条序列的注意力或小 GEMM 拆到多个 SM 并行)这类标准加速技巧，因为累加顺序不同会破坏逐位一致。报告用双 kernel 策略(满波用单 SM 处理整序列、尾波用多 SM + 分布式共享内存，且两 kernel 累加顺序精心一致)、DeepGEMM 替换 cuBLAS、per-SM 独立累加缓冲等补回性能。

**权衡总结**:V4 在推理系统上反复做同一类交易——**牺牲"标准库的现成加速技巧"与"存储",换取"1M 上下文服务化"与"训练/推理逐位可复现"**。

## 7. 评测与局限：自报数字怎么读

⚠️ **先看口径**：以下所有分数均为**报告自报**(内部评测框架、统一设置)；且报告自己声明了几个空白：GPT-5.4 在 1M 长上下文评测中因 API 大量失败未测(Table 6 中部分条目留空);K2.6 与 GLM-5.1 部分条目因 API 繁忙留空。第三方独立复测见本节约尾。

### 7.1 Base 模型：更小反而更强

| 📊 基准(EM) | V3.2-Base | V4-Flash-Base | V4-Pro-Base | 来源口径 |
|---|---|---|---|---|
| 激活参数 | 37B | 13B | 49B | 报告 |
| MMLU | 87.8 | 88.7 | **90.1** | 报告 Table 1 |
| MMLU-Pro | 65.5 | 68.3 | **73.5** | 报告 Table 1 |
| SimpleQA verified | 28.3 | 30.1 | **55.2** | 报告 Table 1 |
| HumanEval | 62.8 | 69.5 | **76.8** | 报告 Table 1 |
| MATH | 60.5 | 57.4 | **64.5** | 报告 Table 1 |
| LongBench-V2 | 40.2 | 44.7 | **51.5** | 报告 Table 1 |

📌 **关键点**:Flash-Base 用 13B 激活参数(约为 V3.2 的三分之一)在大部分基准上反超 V3.2-Base——报告把功劳归于架构、数据质量与训练优化，这正是"效率优先"路线最硬的证据。

### 7.2 Pro-Max 对阵前沿模型：有赢有输

Pro-Max(Max 推理档)在报告里的表现，挑有代表性的行:

| 📊 基准 | DS-V4-Pro-Max | 对标最强(闭源) | 来源口径 |
|---|---|---|---|
| LiveCodeBench | **93.5** | GPT-5.4 未测;Gemini-3.1-Pro 91.7 | 报告 Table 6 |
| Codeforces(Rating) | **3206**(人类第 23 名) | GPT-5.4 3168、Gemini 3052 | 报告 Table 6 |
| SimpleQA-Verified | 57.9 | Gemini-3.1-Pro **75.6** | 报告 Table 6 |
| GPQA Diamond | 90.1 | Gemini-3.1-Pro **94.3** | 报告 Table 6 |
| HLE | 37.7 | Gemini-3.1-Pro **44.4** | 报告 Table 6 |
| MRCR 1M(长上下文检索) | 83.5 | Opus-4.6 **92.9**、Gemini 76.3 | 报告 Table 6 |
| CorpusQA 1M | 62.0 | Opus-4.6 **71.7**、Gemini 53.8 | 报告 Table 6 |
| SWE Verified(代码 agent) | 80.6 | Opus-4.6 **80.8** | 报告 Table 6 |
| Terminal Bench 2.0 | 67.9 | GPT-5.4 **75.1** | 报告 Table 6 |

📌 **关键点**：报告自己的定性总结很克制——(1) 知识类(SimpleQA、GPQA、HLE)**大幅拉开与开源模型的差距，但仍落后 Gemini-3.1-Pro**；(2) 推理类"与 GPT-5.2/Gemini-3.0-Pro 相当，距 GPT-5.4/Gemini-3.1-Pro 约 **3~6 个月**差距";(3) agent 公开基准与 Kimi-K2.6、GLM-5.1 持平、略逊闭源，但内部评测超 Claude Sonnet 4.5、接近 Opus 4.5;(4) **1M 长上下文在 MRCR 上反超 Gemini-3.1-Pro**——这正是本报告的主场。

### 7.3 报告承认的局限(第 6 节原文 + 评测细节)

- **架构复杂**：为了降风险保留了许多"初步验证过但不够优雅"的组件，报告说未来要做更原则性的精简;
- **稳定性技巧原理不明**:Anticipatory Routing 与 SwiGLU Clamping 有效但机制未完全理解;
- **长上下文仍有退化**:MRCR 检索性能 128K 内平稳，超过 128K 可见退化，1M 时"仍然很强"但确实在下滑;
- **搜索短板**：对比类/推荐类任务上 V3.2 仍能与 V4-Pro 抗衡(内部搜索评测);
- **中文写作的极限场景**：高复杂度约束、多轮场景下 Claude Opus 4.5 仍以 52.0% 对 45.9% 胜出;
- **白领任务的短板**：偶尔忽略格式约束、长文压缩为摘要能力弱、PPT 视觉设计一般(内部 30 题人评，整体非失败率 63% vs Opus-4.6-Max);
- **工程师内部调查(N=85)**:52% 愿意让 V4-Pro 当默认编码模型、39% 倾向愿意、<9% 不愿意；反馈集中在琐碎错误、模糊 prompt 误读、偶尔过度思考;
- 未来方向(报告点名):embedding 稀疏化(引用了 Engram 论文)、低延迟系统、长周期 agent 任务、多模态、更好的数据策展。

<img src="/AIInfraGuide/images/deepseek-v4-fig9-mrcr-long-context.png" alt="DeepSeek-V4 在 MRCR 长上下文任务上的表现" style="max-width: 75%; display: block; margin: 0 auto;" />
*图源：DeepSeek-V4 技术报告 Figure 9(arXiv:2606.19348)*

⚠️ **注意**：有解读文章(kenhuangus)把 **Engram 条件记忆模块**列为 V4 的架构组件——报告原文只在"未来方向"里提了一句"explore more sparse embedding modules such as engram",**V4 本体并没有这个模块**。引用时以报告为准。

### 7.4 第三方视角(解读口径，非报告)

- **Artificial Analysis 独立复测**(Labonne 转述):V4-Pro 综合指数 52,紧咬开源并列第一的 K2.6/MiMo-V2.5-Pro(54);**事实性短板**突出——AA-Omniscience 上 V4 在不知道答案时"猜"的比例 94%(Pro)/96%(Flash)，而 K2.6 为 39%;
- **啰嗦成本**(Labonne 转述):V4-Pro 跑完 Intelligence Index 烧了 190M 输出 token、Flash 烧 240M,单价便宜但 token 量大；即便这样 Pro 全套成本约为 Claude Opus 4.7 的四分之一;
- **V4-Pro-Max 自报 R&D 编码基准**(报告 5.4.4):30 个内部真实任务上显著超 Sonnet 4.5、接近 Opus 4.5——注意这是内部基准 + 自报，与公开 SWE 类基准的结论要分开读。

## 8. 意义：开源模型把 1M 上下文变成默认

**第一层：效率数字本身。** 报告开篇就给了两个坐标：V4-Pro 在 1M 上下文下单 token FLOPs 是 V3.2 的 27%、KV cache 是 10%;相对 BF16 GQA8 基线 KV cache 约 2%。这意味着 **1M 上下文从"演示级"变成"日常级"**——报告原文说这"enables us to routinely support one-million-token contexts"(让我们能常规化支持百万 token 上下文)，并明确把长上下文效率与测试时扩展、long-horizon 任务、在线学习(online learning)绑定为下一代研究方向。

**第二层：开源与部署。** 权重 MIT 协议开源(HF 模型卡),Base 版 FP8、对话版 FP4+FP8 混合(专家权重 FP4、其余 FP8)；官方 API 当天上线，1M 上下文成为所有官方服务的默认(公告)，旧的 deepseek-chat / deepseek-reasoner 在 2026-07-24 退役(公告)。三种推理档(Non-think / Think High / Think Max)对应不同 RL 训练配置，Max 档官方建议至少 384K 上下文窗口(模型卡)。

**第三层：国产芯片叙事(谨慎区分口径)。** 报告**确认**的事实只有：细粒度 EP 内核在**华为 Ascend NPU** 上验证过(3.1 节);MXFP4 权重(QAT 后直接以原生 FP4 推理)被解读为便于在未训练芯片上部署——此为解读口径，报告原文未明确。第三方解读(Labonne)补充：**推理**可跑 Ascend/寒武纪/壁仞等国产芯片，但**训练**据 ChinaTalk/36Kr 报道用的是 NVIDIA Hopper(2025 年中 Ascend 迁移失败);kenhuangus 则称全栈 Ascend——**两说冲突，训练硬件「待核实」**。写系统文章时，能站得住的事实是:"国产芯片全栈"目前只有"推理侧 + EP 内核验证"这两个证据点。

**落成一句话**:DeepSeek-V4 报告展示的是一条完整的"长上下文降本"路线——**架构上压缩注意力(CSA/HCA)、训练上换优化器与数据(Muon + 长文档优先)、后训练上专家蒸馏(OPD)、系统上把 KV 管到块级与磁盘级**——四层一起，才把 1M token 从理论变成默认。

## 📝 总结

1. **问题**：注意力随序列平方级上涨 + KV cache 线性上涨，1M 上下文在传统架构下"算不起、存不起";V4 用 **CSA + HCA 混合压缩注意力**把 1M 上下文下 Pro 的单 token FLOPs 压到 V3.2 的 **27%**、KV cache 压到 **10%**(Flash:10% / 7%;相对 BF16 GQA8 基线约 2%——均报告口径)。
2. **架构**:CSA(压缩率 $m=4$ + top-$k$ 稀疏选择)负责"按需精读",HCA(压缩率 $m'=128$ + 稠密注意力)负责"全局概览",加滑动窗口补近期细节;**mHC** 用双随机矩阵约束残差映射(谱范数 ≤1),**Muon** 用动量 + 正交化换更快更稳的收敛。
3. **训练**:Flash 32T / Pro 33T token,序列 4K→1M 渐进、前 1T token 稠密热身；训练不稳定靠前瞻路由与 SwiGLU Clamping 两个经验性技巧压住(机制未完全理解);**硬件与训练成本报告未公开**(第三方报道冲突，待核实)。
4. **后训练**：领域专家(SFT + GRPO)各自培养，再用 **>10 位教师的全词表反向 KL 蒸馏(OPD)** 合流成一个模型——取代 V3.2 的混合 RL,规避 reward hacking 与能力干扰。
5. **系统**：异构 KV cache(状态缓存 + 压缩缓存，块对齐取 $\mathrm{lcm}=128$)+ 磁盘 KV 前缀复用 + wave 级 EP 重叠(1.50~1.73×,最高 1.96×)+ 逐位确定性内核;SWA KV 体积约为压缩 KV 的 8 倍，三种磁盘策略按存储/计算权衡选。
6. **边界**：知识类落后 Gemini-3.1-Pro(SimpleQA 57.9 vs 75.6)、推理差前沿 3~6 个月、长上下文 128K 后可见退化；报告自报分数需配合"内部框架 + 部分对手未测"的口径阅读。

## 🎯 延伸思考：自我检验清单

- 能解释为什么注意力对长上下文是平方级瓶颈，并用手算说明 KV cache 随序列线性增长(用站内 [1.1-LLM推理基础](/AIInfraGuide/inference/模块四-推理优化/第1章-llm推理基础/11-llm推理基础) 的算账方法)。
- 能说出 CSA 与 HCA 各自"压缩率、是否稀疏、负责什么信息"三要素，并解释为什么两者要交替而不是只用一种。
- 能给 CSA 的压缩公式写出直觉：压缩条目 = 原始 KV 的加权合并，权重来自可学习打分 + 位置偏置；并算 1M token 在 $m=4$ / $m'=128$ 下的压缩条目数(250K / ~7.8K)。
- 能解释滑动窗口分支存在的必要性：压缩块内 query 看不到同块 token,而近期 token 最重要。
- 能说明 mHC 的"双随机矩阵"约束为什么能稳定深堆叠(行/列和为 1 → 谱范数 ≤1 → 非扩张)，以及 Sinkhorn-Knopp 在做什么(交替行列归一化)。
- 能对照站内 [第5章-ZeRO系列](/AIInfraGuide/distributed/模块三-分布式训练/第5章-zero系列) 说清 Muon 与 ZeRO 的冲突点(需要完整梯度矩阵)和报告的混合解法。
- 能解释为什么 OPD 用反向 KL、为什么坚持全词表蒸馏(梯度方差 vs 存储成本)，并说出三个工程技巧(缓存隐状态、按教师排序、TileLang KL kernel)。
- 能说清 FP4→FP8 反量化为什么可以无损(FP8 多 2 个指数位，动态范围更大)，以及"无损"的前提条件(子块尺度比阈值)。
- 能解释为什么 V4 的 KV cache 管理不能直接套 PagedAttention(混合淘汰策略 + kernel 对齐)，并说明 state cache 与 classical cache 的分工。
- 能复述磁盘 KV 的三种 SWA 策略并给出选型直觉：命中率高/存储宽裕 → Full;存储紧 → Periodic;极致省存储 → Zero + 重算。
- 能凭记忆列出报告自报的至少三个"输给闭源"的基准(如 SimpleQA、GPQA、HLE)与两个"赢"的基准(如 LiveCodeBench、Codeforces、MRCR 1M)，并说明评测口径(内部框架、GPT-5.4 未测)。
- 能区分"报告确认事实"(EP 内核 Ascend 验证、权重 FP4+FP8、KV 比例)与"第三方待核实说法"(训练硬件、全栈国产芯片)。

## 📚 参考资料

**报告原文**

- [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence(arXiv 2606.19348)](https://arxiv.org/abs/2606.19348)——DeepSeek-AI,2026-04-26 preview 版；本文主源，所有「报告」口径数字出处
- [PDF 原文](https://arxiv.org/pdf/2606.19348)

**官方模型卡与发布公告**

- [DeepSeek-V4-Pro 模型卡(HuggingFace)](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)——权重下载、Base/对话版精度说明(FP8 / FP4+FP8)、推理模式表、本地运行建议
- [DeepSeek-V4 权重合集(HuggingFace Collections)](https://huggingface.co/collections/deepseek-ai/deepseek-v4)
- [DeepSeek V4 Preview Release 官方公告](https://api-docs.deepseek.com/news/news260424)——2026-04-24 上线、API 兼容、旧模型退役时间

**权威解读**

- [Maxime Labonne: DeepSeek V4: ten teachers, one student](https://maximelabonne.substack.com/p/deepseek-v4-ten-teachers-one-student)——OPD 方法论定位、激活参数对比、第三方复测与硬件报道转述(标注解读口径)
- [Ken Huang: DeepSeek V4: The Next Frontier of Open-Source AI](https://kenhuangus.substack.com/p/deepseek-v4-the-next-frontier-of)——架构与商业解读；其中 Engram 模块、Ascend 950PR 全栈训练等说法与报告冲突，已在上文标注「待核实」
- [Framia: DeepSeek V4 Architecture: CSA, HCA, mHC, MoE Deep Dive](https://framia.converge.ai/page/en-US/news/deepseek-v4-model-architecture)——架构四支柱的通俗拆解

**站内相关(建议配合阅读)**

- [6.4-数据并行与专家并行](/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/64-数据并行与专家并行)——EP 的通信账，对应报告 3.1 节 wave 重叠
- [7.3-KV 传输与 Connector](/AIInfraGuide/inference/模块四-推理优化/第7章-pd解耦架构/73-kv传输与connector)——KV 在系统中的搬运成本，呼应 V4 磁盘 KV 设计
- [4.5-FP8 与 NVFP4](/AIInfraGuide/inference/模块四-推理优化/第4章-量化/45-fp8与nvfp4)——FP4/FP8 精度格式基础，对应报告 MXFP4 QAT
- [1.1-LLM 推理基础](/AIInfraGuide/inference/模块四-推理优化/第1章-llm推理基础/11-llm推理基础)——KV cache 概念与显存账本起点
- [2.1-PagedAttention](/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/21-pagedattention)——为什么 V4 说混合注意力"违反 PagedAttention 假设"
- [第9章-长序列训练与上下文并行](/AIInfraGuide/distributed/模块三-分布式训练/第9章-长序列训练与上下文并行)——CP 与长序列训练，对应报告 3.4.3 两阶段 CP
- [第10章-MoE 并行](/AIInfraGuide/distributed/模块三-分布式训练/第10章-moe并行)——MoE 并行全景，对应 MegaMoE 内核
- [6.1-FlashAttention V1 详解](/AIInfraGuide/cuda/模块二-cuda编程与算子优化/61-flashattention-v1详解)——注意力算子的显存/带宽优化起点
