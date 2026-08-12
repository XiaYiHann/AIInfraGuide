---
title: "GLM-5 解读：从'一句话编程'到能自主干活 8 小时的 Agent 模型"
description: "拆解 GLM-5 技术报告：744B 总参/40B 激活的稀疏 MoE 用 DSA 把长序列注意力算力砍掉约 1.5–2 倍，28.5T 语料 + 全异步 Agentic RL 让模型学会连续数小时自主编程，八大 ARC 基准平均比 GLM-4.7 提升约 20%,SWE-bench Verified 77.8 逼近闭源。"
pubDate: 2026-08-09
originalUrl: "https://arxiv.org/abs/2602.15763"
sourceType: "paper"
originalAuthor: "Zhipu AI & Tsinghua University"
tags: ["GLM-5", "MoE", "DSA稀疏注意力", "Agentic RL", "长上下文", "SWE-bench"]
---

> 原文：[GLM-5: from Vibe Coding to Agentic Engineering](https://arxiv.org/abs/2602.15763)(GLM-5-Team,Zhipu AI & Tsinghua University,arXiv 2602.15763,2026-02-17)

GLM-5 要回答的问题不是"下一个 token 预测得准不准",而是"**模型能不能像工程师一样自己干完一整件活**"。这份技术报告交出的答卷是:**744B 总参数、40B 激活**的稀疏 MoE(总规模是前代 GLM-4.5 的两倍)，把上下文拉到 **200K**；用 **DSA 稀疏注意力**把长序列的注意力算力砍掉约 **1.5–2 倍**；用 **28.5T tokens** 语料训练，再用一套**全异步的 Agentic RL** 管线让模型学会"边干活边反思、一干就是几小时"。结果：GLM-5 成为**首个在 Artificial Analysis Intelligence Index v4.0 上拿到 50 分的开源模型**(GLM-4.7 为 42 分)，八大 Agentic/推理/编码基准平均比前代提升约 **20%**;SWE-bench Verified 拿到 **77.8**，逼近闭源的 Claude Opus 4.5(80.9)。对做推理系统的人，这份报告还藏着一半干货：长程 Agent 的 rollout 推理怎么解耦、怎么用 DP-attention(数据并行注意力：注意力按数据并行切分，各 rank 自算自己的部分，不用跨卡搬 KV)省 KV 搬运、怎么用 FP8 + MTP 压长尾延迟——这些正是站内 [8.2 Tool Calling/Agent](/AIInfraGuide/inference/模块四-推理优化/第8章-生产级服务特性/82-tool-calling与reasoning) 和 [7.2 PD 解耦](/AIInfraGuide/inference/模块四-推理优化/第7章-pd解耦架构/72-解耦架构设计) 两章的实战延伸。

<!-- more -->

## 📑 目录

- [1. 背景：从"一句话编程"到 Agentic 工程](#1-背景从一句话编程到-agentic-工程)
- [2. 架构:744B MoE 与"划重点"的稀疏注意力](#2-架构744b-moe-与划重点的稀疏注意力)
- [3. 训练:28.5T 语料与"异步干活"学出来的 Agent](#3-训练285t-语料与异步干活学出来的-agent)
- [4. 评测：自报基准、对比口径与自认的差距](#4-评测自报基准对比口径与自认的差距)
- [5. 系统与部署：长程 Agent 的推理需求](#5-系统与部署长程-agent-的推理需求)
- [6. 局限与意义](#6-局限与意义)
- [📝 总结](#-总结)
- [🎯 延伸思考：自我检验清单](#-延伸思考自我检验清单)
- [📚 参考资料](#-参考资料)

## 1. 背景：从"一句话编程"到 Agentic 工程

**问题是：去年大家还在讨论"一句话生成代码",为什么今年风向变成了"让模型自己干几小时活"?**

因为静态基准测不出真实生产力。SWE-bench 这类基准把任务切成"单个 commit、孤立修改",模型改完一个文件就交卷；但真实软件工程是**多文件、多步骤、状态累积**的——改完 A 步骤，代码库状态变了，下一步的决策必须基于新状态。报告的原话是:**coding agents can now write code autonomously for hours**(编码 Agent 现在可以自主写代码数小时)，任务的长度和广度只会继续增长。

打个比方:**Vibe Coding 像你对着厨师口述菜谱**——你说一句，厨师做一步，你盯着看，不满意再喊停;Agentic Engineering 像你放手让厨师自己进货、备菜、试菜、调整火候，几个小时后端上一桌菜，你只负责验收。关键词是"自己"——模型要自己规划(plan)、自己实现(implement)、自己迭代(iterate)，中间没有任何人类盯着。

GLM-4.5 已经把 Agentic(智能体)、Reasoning(推理)、Coding(编码)三种能力(报告简称 **ARC**)融进一个 MoE 模型，拿到当时的 SOTA。GLM-5 的增量是两件事：**更便宜**(DSA 稀疏注意力砍算力)和**更自主**(异步 RL 学长程决策)。报告列了四项技术贡献:

1. **DSA(DeepSeek Sparse Attention,DeepSeek 稀疏注意力)**：动态分配注意力资源，把长序列推理成本大幅压低，使参数规模能冲到 744B、训练预算能加到 28.5T tokens;
2. **异步强化学习基础设施**：把"生成轨迹"和"训练更新"彻底解耦，最大化 GPU 利用率，支撑大规模 Agent 轨迹探索;
3. **新的异步 Agent RL 算法**：让模型从多样、长程的交互中持续学习，直接贡献于真实编码场景的领先;
4. **首日即全栈适配国产芯片**：昇腾、摩尔线程、海光、寒武纪、昆仑芯、MetaX、燧原 7 个平台(详见第 5 节)。

## 2. 架构：744B MoE 与"划重点"的稀疏注意力

### 2.1 模型规模：总参数翻倍，激活参数只加 8B

| 📊 维度 | GLM-4.5 | GLM-5 | 变化 |
|---|---|---|---|
| 总参数 | 355B | **744B** | ≈2 倍 |
| 激活参数 | 32B | **40B** | +8B |
| 层数 | 3 dense + 89 MoE | 3 dense + **75 MoE** | 层数变少 |
| 专家数 | 160(8 routed + 1 shared) | **256**(8 routed + 1 shared) | 专家变多 |
| 注意力头数 | 96 | **64** | -1/3 |
| 上下文(后训练) | 128K 上限 | **200K** | +72K |

📌 **关键点**：总参数翻倍，但每 token 只激活 40B——MoE 的意义就是"模型很大、每次只请一部分专家干活"。层数从 89 减到 75、专家数从 160 增到 256,报告给出的动机是**减少专家并行(Expert Parallelism)的通信开销**：层越少，跨设备通信的次数越少；专家越多，单个专家更"专"。

### 2.2 DSA:注意力也有"划重点"功能

**问题是：上下文 200K 时，标准的稠密注意力(Dense Attention)为什么扛不住?**

因为注意力计算的复杂度是 $O(L^2)$(L 是序列长度)——200K 上下文的注意力矩阵有 $4\times10^{10}$ 个条目，每个都要算一遍相关性，这在推理时是巨大的算力浪费。DeepSeek-V3.2 给出的答案是 DSA,GLM-5 直接采纳。

打个比方：DSA 像一个**"划重点"的导读员**——传统注意力要求你把一本书的每一页都精读一遍;DSA 先让一个轻量组件快速扫完全书，挑出"这页跟当前问题相关"的重点页，然后注意力只精读这些重点页。关键是**重点由内容动态决定**，不是固定窗口。

术语对照(与比喻一一映射):

- **Lightning Indexer(闪电索引器)**：那个"快速扫书"的轻量组件。对每个 query 位置，给前面所有 token 打分，取 **top-k(k=2048)** 个 KV 条目;
- **稀疏注意力**：主注意力模块只在这 2048 个被选中的条目上计算，复杂度从 $O(L^2)$ 降到 $O(Lk)$;
- **划重点的依据**:DeepSeek-V3.2 的实验结论(报告转述)是——**长上下文里约 90% 的注意力条目确实是冗余的**，所以划掉它们不影响质量。

报告给的两个关键数字:

- **长序列的注意力计算减少约 1.5–2 倍**；报告原话是"能以一半 GPU 成本处理 128K 上下文"——注意限定：这是**长序列、推理侧**的收益，不是端到端加速;
- **DSA 是"接枝"出来的，不是从零训的**：从 dense(稠密)基座模型做 **Continued Pre-Training(继续预训练)**。GLM-5 的 DSA 适配只用了 **20B tokens**(1000 步 warmup + 稀疏适配阶段)，而 DeepSeek-V3.2 花了 **943.7B tokens**——差了约 47 倍，这是报告特别强调的成本优势。

<img src="/AIInfraGuide/images/glm-5-fig6-sft-loss.png" alt="MLA 与 DSA 的训练损失曲线对比" style="max-width: 75%; display: block; margin: 0 auto;" />
*图源：GLM-5 技术报告 Figure 6(arXiv:2602.15763)*

💡 **提示**:DSA 不是唯一的省算力方案。报告用 GLM-9B 做了消融，对比了滑动窗口注意力(SWA)、Gated DeltaNet(GDN)等方案，结论是：**固定模式的 SWA 在 RULER@128K 上暴跌 30.35 分；搜索式 SWA 模式好一些但仍有损;DSA 因为是 token 级动态稀疏、不丢任何长程依赖，可以无损应用到所有层**。这是 GLM-5 选 DSA 不选其他方案的依据。

还有一个工程细节值得记:**DSA 的 top-k 算子必须是确定性的**。报告发现，SGLang 里基于 CUDA 的非确定性 top-k 实现会让 RL 训练几步之内熵骤降、性能崩盘；换成朴素但确定性的 `torch.topk`,RL 就稳了。原因和 MoE 的 routing replay 一样：训练和推理如果选了不同的 top-k 集合，策略梯度就是歪的。为此报告在 RL 阶段**默认冻结 indexer 参数**。

### 2.3 MLA 与 Muon Split:KV 显存与解码计算的双重账本

DSA 管的是"注意力算得少一点",MLA(Multi-latent Attention,多潜在注意力)管的是"KV 缓存存得小一点"。MLA 用低维潜在向量压缩 KV:GLM-5 的 MLA 的 **KV 缓存维度是 576**，而 GQA-8(8 组查询的分组查询注意力)是 **2048 维**——长上下文下 KV 显存差近 4 倍，报告明确说 MLA 对长序列"显存更省、处理更快",是 200K 长上下文显存账本的关键。

但 MLA 有两个坑，报告各自给了解法:

1. **训练效果打不过 GQA-8**：在 Muon 优化器下，576 维 MLA 的基准分全面落后 GQA-8。解法是 **Muon Split**：原来对多头 Q/K/V 的上投影矩阵 $W^{UQ}, W^{UK}, W^{UV}$ 整体做矩阵正交化，改成**按 head 拆成小块、各自独立正交化**，让不同注意力头的投影权重能以不同尺度更新。消融表显示这一改，MLA 追平 GQA-8(表 1:MMLU 61.5→62.5,Hellaswag 77.3→77.8)。
2. **解码计算贵**:MLA 解码时要做 576 维点积，而 GQA 只要 128 维。解法是 **MLA-256**：把 head 维度从 192 提到 256,注意力头数减少 1/3(96→64)——训练和 Prefill 阶段计算量不变，但**解码阶段点积维度降了 1/3**。

💡 **提示**：这里的权衡很典型——MLA 省的是 KV 显存(存储)，代价是解码时单次点积更贵(计算);MLA-256 再把"解码计算"这一项找回来，代价是注意力头数变少。没有免费午餐，只是把账从"显存"挪到了"算力"再挪回来一部分。

### 2.4 MTP:投机解码的草稿层，三层共享参数

MTP(Multi-Token Prediction,多 token 预测)在训练时让模型同时预测后面 n 个 token,推理时这些层就是**投机解码(Speculative Decoding)的草稿模型**——小步快跑猜一串，主模型一次验收。DeepSeek-V3 只训 1 个 MTP 层，推理时预测 2 个 token,存在训练-推理不一致(第二个 token 接受率低)。GLM-5 的做法是**训练时 3 个 MTP 层共享参数**：草稿模型的显存开销和 DeepSeek-V3 一致，但接受率更高。报告在私有 prompt 集上测的 **accept length(接受长度)是 2.76,DeepSeek-V3.2 是 2.55**(同样 4 步投机)。

### 2.5 上下文：从 4K 一路拉到 200K 的三段式 Mid-training

基座模型预训练(通用语言 + 编码能力)之后，是**Mid-training(中期训练)**，专门负责"长上下文 + Agentic"。上下文长度分三档渐进扩展:

| 阶段 | 上下文长度 | 训练量 |
|---|---|---|
| 1 | 32K | 1T tokens |
| 2 | 128K | 500B tokens |
| 3 | **200K** | 50B tokens |

报告特别指出：**后面的 200K 阶段即使只看 128K 以内的任务，也进一步提升了性能**——长上下文训练不是"只对长文本有用"。长上下文数据里混了合成数据：受 NextLong / EntropyLong 启发，把高度相似的文本用交错打包(interleaved packing)拼成长序列，缓解"lost-in-the-middle"(中间内容被遗忘)现象;200K 阶段还加了 MRCR 类多轮对话数据。

## 3. 训练：28.5T 语料与"异步干活"学出来的 Agent

<img src="/AIInfraGuide/images/glm-5-fig5-training-pipeline.png" alt="GLM-5 总体训练管线" style="max-width: 75%; display: block; margin: 0 auto;" />
*图源：GLM-5 技术报告 Figure 5(arXiv:2602.15763)*

### 3.1 预训练数据：代码语料 +28%,Issue-PR 对约 1000 万

基座模型总预算 **28.5T tokens**。几个有数字的重点:

- **代码语料**：刷新了主流代码托管平台的快照，去重后唯一 token 数**增加 28%**；还修了 Software Heritage 代码文件的元数据对齐问题，并对 Scala/Swift/Lua 等低资源语言训了专用分类器;
- **软件工程数据**：保持"仓库文件 + commit diff + Issue + PR"拼成统一训练序列的做法，放宽仓库级过滤(约 **1000 万 issue–PR 对**)，加强 issue 级质量过滤，过滤后 issue–PR 部分约 **160B 唯一 tokens**;
- **数学与科学**:LLM 给文档打分，只留教育价值最高的；长文档用 **chunk-and-aggregate(分块聚合)打分**；过滤管线**严格排除合成/AI 生成/模板数据**;
- **Web 数据**：引入基于句子嵌入的 DCLM 分类器和"世界知识"分类器，专门捞长尾知识。

⚠️ **注意**：报告没有公开训练集群规模、总算力(FLOPs)和训练时长——这些对评估"这个 744B 模型训得贵不贵"至关重要，但报告未公开，别从别处脑补。

### 3.2 SFT:交错的思考模式

SFT 语料分三大类：General Chat、Reasoning、**Coding & Agent**(前端/后端工程代码、工具调用、编码 Agent、搜索 Agent、通用 Agent——比 GLM-4.5 显著扩容)。SFT 最大上下文 **202,752 tokens**。两个值得记的设计:

- **三种思考模式**：报告图 7 展示了 Interleaved Thinking(交错思考，边行动边想)、Preserved Thinking 和 **Turn-level Thinking(按轮控制思考)**——轻量请求关掉思考省延迟/成本，复杂任务打开思考提准确率。这就是"thinking effort"开关的由来;
- **错误段掩码**：轨迹里的错误片段**保留但 mask 掉 loss**，让模型学到"纠错行为"但不强化"错误动作"。

### 3.3 Reasoning RL:GRPO + IcePop,先稳住推理

后训练是**串行三阶段 RL**:Reasoning RL → Agentic RL → General RL,最后再来一轮跨阶段蒸馏。Reasoning RL 的算法骨架是 **GRPO(群体相对策略优化：一种 RL 算法，对同一问题采样一组回答，按组内相对优劣更新策略) + IcePop 变体**。

**问题是：RL 训练为什么需要 IcePop 这种怪东西?** 因为 RL 的"训练分布"和"推理分布"会漂移——训练时用的采样方式和推理时不一样，梯度就失真了(报告叫 training-inference mismatch,训练-推理失配)。IcePop 的做法是算一个失配比:

$$\rho_{i,t}=\underbrace{\frac{\pi^{\text{train}}_{\theta_{\text{old}}}(y_{i,t}\mid x,y_{i,<t})}{\pi^{\text{infer}}_{\theta_{\text{old}}}(y_{i,t}\mid x,y_{i,<t})}}_{\text{同一 token:训练策略概率 ÷ 推理策略概率}}$$

失配太离谱的样本(比例超出 $[1/\beta, \beta]$)直接 pop 掉(抑制，不参与梯度)。GLM-5 去掉了原版 IcePop 的 KL 正则项加速收敛，超参 $\beta=2,\ \epsilon_{\text{low}}=0.2,\ \epsilon_{\text{high}}=0.28$,组大小 32、batch 32。

训练域是混合的：数学、科学、代码(Codeforces/TACO)、工具集成推理(TIR)，用域相关的 judge 模型给二元奖励。难度过滤的原则很明确:**GLM-4.7 大概率做不对、但更强的教师模型(GPT-5.2 xhigh、Gemini 3 Pro Preview)能做的题**，才进 RL。

### 3.4 Agentic RL:全异步解耦，让模型学会长程决策

**问题是：Agent 的轨迹又长又慢，同步 RL 为什么浪费 GPU?**

同步 RL 的流程是"采样一批轨迹 → 全部完成 → 更新权重 → 再来一批"。Agent 轨迹动辄几十分钟，而且**长尾严重**——99 条 1 分钟轨迹 + 1 条 50 分钟轨迹，整批都要等那 1 条(报告叫 straggler,掉队者)。GPU 大量时间在空转。

打个比方：**同步 RL 像全班等最后一个人打完饭才开吃；异步 RL 像外卖平台——订单(轨迹)源源不断地进来，攒够一车就发车(训练更新一次)，不等所有订单**。比如 99 条 1 分钟的轨迹配上 1 条 50 分钟的"掉队者",同步做法整批都要等它;

GLM-5 的异步架构(报告第 4 节核心):

- **训练引擎和推理引擎彻底解耦到不同 GPU 上**：推理引擎持续生成轨迹，攒够阈值就发给训练引擎更新；训练引擎每 K 次梯度更新后把新权重推回推理引擎(策略滞后靠周期性同步来控制);
- **Multi-Task Rollout Orchestrator(多任务轨迹编排器)**：每个任务(软件工程/终端/搜索)的 rollout 和奖励逻辑是独立微服务，统一注册到编排器，标准化成统一消息列表，支持 **1000+ 并发 rollout**、动态调整各任务采样比例;
- **TITO(Token-in-Token-out)网关**：训练直接消费推理引擎吐出的**原始 token 流**，而不是把文本重新 tokenize——重 tokenize 会在轮次边界、截断、特殊 token 上产生细微错位，把"动作和奖励的对应关系"搞坏。TITO 是异步训练稳定的关键之一。

异步引入了 off-policy(离策略)问题——轨迹是旧权重生成的，直接学会偏。三个机制控制:

1. **直接双侧重要性采样**：旧策略概率 $\pi_{\theta_{\text{old}}}$ 太贵不追踪了，直接用 rollout 时的 log-prob 当行为代理:

$$r_t(\theta)=\exp\!\Big(\underbrace{\log\pi_\theta(a_t\mid s_t)}_{\text{当前策略概率}}-\underbrace{\log\pi_{\text{rollout}}(a_t\mid s_t)}_{\text{生成轨迹时的概率}}\Big)$$

再套一个**双侧裁剪掩码**：比值落在 $[1-\epsilon_\ell,\ 1+\epsilon_h]$ 之外就整体 mask 掉，不让极端策略漂移破坏训练:

$$f(x;\epsilon_\ell,\epsilon_h)=\begin{cases}x, & 1-\epsilon_\ell < x < 1+\epsilon_h\\ 0, & \text{otherwise}\end{cases}$$

2. **丢弃过时样本**：记录每条轨迹由哪几个权重版本 $(w_0,\dots,w_k)$ 生成，若最老版本落后当前策略超过阈值 $\tau$,整条丢弃;
3. **丢弃环境崩溃样本 + 组填充**：沙箱崩溃不代表模型不行，记下失败原因并排除;GRPO 组不完整时，有效样本过半就复制填充，否则整组丢弃。

**Agent 环境规模化**是这节的另一半：软件工程用 RepoLaunch 管线搭出 **10000+ 个可验证环境，覆盖 9 种语言**(Python/Java/Go/C/C++/JS/TS/PHP/Ruby)，自动从测试输出里抽 F2P/P2P 测试；终端任务用 Harbor 格式合成，Docker 构建准确率 **>90%**；搜索任务用网页知识图谱(WKG,200 万+ 页面)自动合成多跳问答，难度过滤分三阶段(无工具模型 8 次全错、早期 Agent 几步内解不出、双向验证答案唯一性)。

### 3.5 General RL 与跨阶段蒸馏

General RL 把目标拆成三个维度：**基础正确性**(指令遵循、逻辑、事实、幻觉、流畅度的错误率——先达到"可用"基线)、**情商**(共情、自然、像人)、**任务特定质量**(写作/翻译/角色扮演等逐任务的精细质量)。奖励系统是**混合的**：规则奖励(精确可解释但覆盖面窄)+ 结果奖励模型 ORM(低方差高效但易被 reward hacking)+ 生成式奖励模型 GRM(更抗 hack 但方差高)——三者互补。还引入**人类专家写作的回复作为风格锚点**，避免纯模型生成数据收敛成"模型腔"。

**跨阶段蒸馏(On-Policy Cross-Stage Distillation)** 解决的是"灾难性遗忘":每个新 RL 阶段都会冲掉前面阶段的能力。做法是把前序阶段的最终 checkpoint 当教师，蒸馏损失就是把主损失里的 advantage 换成:

$$\hat{A}_{i,t}=\text{sg}\!\left[\log\frac{\pi^{\text{infer}}_{\theta_{\text{teacher}}}(y_{i,t}\mid x,y_{i,<t})}{\pi^{\text{train}}_{\theta}(y_{i,t}\mid x,y_{i,<t})}\right]\underbrace{\quad\text{sg}=\text{stop gradient,教师只提供'差距'信号,不被优化}}_{\text{蒸馏阶段组大小降到 1、batch 提到 1024:不需要同 prompt 多条采样估 advantage 了}}$$

### 3.6 训练系统：显存与并行的"抠门账本"

预训练基础设施(2.4 节)全是工程账，挑四个能带走的口诀:

- **Pipeline ZeRO2 梯度分片**：每个 pipeline stage 只存 $1/dp$ 的梯度，且只用"双缓冲"保留两个 stage 的完整累加缓冲——梯度显存从"每 stage 全量"降到"分片 + 2 份滚动";
- **Muon 优化器零冗余通信**:all-gather(全收集：把各卡数据收集齐，每张卡都拿到完整版)只取本 rank 拥有的参数分片，和本地计算重叠;
- **激活 offloading + 细粒度重计算**:warmup 阶段 forward 领先 backward,激活存活期长，就挪到主机内存，backward 前再搬回，按层粒度做，和 P2P(点对点通信：两张卡直接互传数据，不经过第三方)通信、MoE token 路由错峰;
- **sequence-chunked 输出投影**：输出投影 + cross-entropy 的激活峰值显存按 chunk 切分，前向反向走完一块放一块。

SFT 阶段还做了 **INT4 量化感知训练(QAT)**，且量化核在训练和推理间**逐位一致**(bitwise-identical)——量化训练不白练。

## 4. 评测：自报基准、对比口径与自认的差距

### 4.1 八大 ARC 基准：平均 +20%,逼近闭源

报告主表(表 7)对比 GLM-5 / GLM-4.7 / DeepSeek-V3.2 / Kimi K2.5 / Claude Opus 4.5 / Gemini 3 Pro / GPT-5.2 (xhigh)。**八大基准平均比 GLM-4.7 提升约 20%,与 Claude Opus 4.5 和 GPT-5.2 (xhigh) 相当，优于 Gemini 3 Pro**——这是报告自己的口径。

<img src="/AIInfraGuide/images/glm-5-fig1-benchmarks.png" alt="GLM-5 与其他模型的基准对比" style="max-width: 75%; display: block; margin: 0 auto;" />
*图源：GLM-5 技术报告 Figure 1(arXiv:2602.15763)*

挑关键数字(均为论文报告):

| 📊 基准 | GLM-5 | GLM-4.7 | Claude Opus 4.5 | GPT-5.2 (xhigh) | 来源口径 |
|---|---|---|---|---|---|
| HLE(仅文本子集) | 30.5 | 24.8 | 28.4 | 35.4 | 论文报告 |
| HLE(w/ Tools) | **50.4** | 42.8 | 43.4* | 45.5* | 论文报告 |
| GPQA-Diamond | 86.0 | 85.7 | 87.0 | 92.4 | 论文报告 |
| LongBench v2 | 64.5 | 59.1 | 64.4 | 59.8 | 论文报告 |
| SWE-bench Verified | 77.8 | 73.8 | 80.9 | 80.0 | 论文报告 |
| SWE-bench Multilingual | **73.3** | 66.7 | 77.5 | 72.0 | 论文报告 |
| Terminal-Bench 2.0 (Terminus-2) | 56.2 / 60.7† | 41.0 | 59.3 | 54.0 | 论文报告 |
| BrowseComp(w/ Context Manage) | **75.9** | 67.5 | 57.8 | 65.8 | 论文报告 |
| BrowseComp-ZH | **72.7** | 66.6 | 62.4 | 76.1 | 论文报告 |
| τ2-Bench | 89.7 | 87.4 | 91.6 | 85.5 | 论文报告 |
| MCP-Atlas (Public Set) | 67.8 | 52.0 | 65.2 | 68.0 | 论文报告 |
| Vending-Bench 2(期末账户余额) | **\$4,432** | \$2,377 | \$4,967 | \$3,591 | 论文报告 |
| GDPval-AA Elo(2026-02-15 记录) | 1,409 | 1,198 | 1,400 | 1,462 | 论文报告 |

(*为全量 HLE 集;†为修正歧义指令后的验证版)

💡 **提示**：完整表 7 还含 DeepSeek-V3.2、Kimi K2.5、Gemini 3 Pro 三列，本表节选四列；涉及"全场最强"的判断以完整表为准，下面关键点已逐项对照。

📌 **关键点**：严格对照完整表，GLM-5 真正"全场(含闭源)第一"的只有 **BrowseComp(带上下文管理)75.9**；其余亮点要逐项说清边界：带工具的 HLE 50.4 超过 Claude Opus 4.5(43.4)和 Gemini 3 Pro(45.8)，但 Kimi K2.5 的 51.8 更高;SWE-bench Multilingual 73.3 超过 Gemini 3 Pro 和 GPT-5.2(xhigh)，但 Claude Opus 4.5 的 77.5 更高;BrowseComp-ZH 72.7 超过 Claude 和 Gemini,但 GPT-5.2(xhigh)的 76.1 更高;Vending-Bench 2 以 \$4,432 排开源第一、逼近 Claude Opus 4.5(\$4,967)，但 Gemini 3 Pro(\$5,478)更高。**"局部领先、整体逼近"才是这节的准确基调**。推理侧,"裸 HLE"(30.5 vs 35.4)和 GPQA-Diamond(86.0 vs 92.4)上仍落后 GPT-5.2 xhigh 约 5–6 分，深度推理仍有差距。Vending-Bench 2 是"模拟经营一年自动售货机生意，期末看账户余额",它测的不是单轮智能，而是长程规划与资源管理。

另外两个总览数字:

- **Artificial Analysis Intelligence Index v4.0:GLM-5 得 50 分，是第一个达到 50 的开源权重模型**(GLM-4.7 为 42,+8 分主要由 agentic 能力和知识/幻觉维度贡献);
- **LMArena Text Arena 与 Code Arena:开源 #1**，整体与 Claude Opus 4.5、Gemini 3 Pro 相当。

<img src="/AIInfraGuide/images/glm-5-fig4-long-horizon.jpeg" alt="Vending-Bench 2 长程任务资金余额曲线" style="max-width: 75%; display: block; margin: 0 auto;" />
*图源：GLM-5 技术报告 Figure 4(arXiv:2602.15763)*

### 4.2 CC-Bench-V2:端到端工程任务的照妖镜

报告自己搭的内部评测 **CC-Bench-V2**，彻底去掉人工标注，用"构建检查 + 单元测试 + **Agent-as-a-Judge**"全自动跑。三块:

| 📊 任务 | 指标 | GLM-5 | GLM-4.7 | Claude Opus 4.5 |
|---|---|---|---|---|
| 前端 HTML | ISR(完整通过率) | 38.9 | 35.4 | **52.2** |
| 前端 React | ISR | 34.6 | 17.2 | **39.7** |
| 前端 Vue | ISR | 32.7 | 24.5 | **46.9** |
| 构建 React/Vue/Svelte/Next.js | BSR(构建成功率) | 100/100/100/95 | 65/70/60/70 | 95/100/90/80 |
| 后端工程 | Pass@1 | 25.8 | 19.6 | **26.9** |
| 大仓库探索 | Pass@1 | **65.6** | 47.8 | 64.5 |
| 链式任务 | Pass@1 | 52.3 | 43.0 | **61.6** |

(来源口径：论文报告，内部评测集)

📌 **关键点**:GLM-5 的构建成功率(BSR)基本满分(四个技术栈 100/100/100/95)，检查项完成率(CSR)逼近 Claude;但**完整实例通过率(ISR)三栈全输**——"满足大多数单项要求，但端到端完整交付仍差一口气"。唯一的胜场是**大仓库探索(65.6 超过 Claude 的 64.5)**：这正好呼应训练里海量 Agent 轨迹的作用——找文件靠的是"目录级推理 + 语义联想"的策略性搜索，不是生成代码的能力。

**链式任务**设计值得一提：从真实合并的 PR(3–15 个 commit、线性历史)里挖出任务链，Agent 从 base commit 开始**顺序完成任务 k,提交，再自动打上下一个任务的补丁**——每一步都改变代码库状态，评价时累积应用所有测试补丁，既抓当前任务的错也抓前面任务的回归。GLM-5 从 43.0 涨到 52.3,但离 Claude(61.6)差距不小，报告直言：**错误会沿链累积，一个次优编辑可能悄悄弄坏后面所有任务的测试**。

### 4.3 对比口径：哪些数字要打折看

这节是"工程师视角审视自报基准"的部分——报告其实交代得比较诚实，但也必须逐条标注:

- **SWE-bench 系用 OpenHands + 定制指令 prompt**(对 GLM-5 有利的提示词),Claude 系通常用 Claude Code 框架——框架不同，跨模型对比天然有偏差;
- **Terminal-Bench 2.0 报告了两个框架**(Terminus-2 与 Claude Code),GLM-5 分别是 56.2/60.7† 和 56.2/61.1†——**同一模型在两种框架下都拿到 56.2,报告借此论证 GLM-5 编码能力的框架泛化性**；但其他模型换框架差异很大(DeepSeek-V3.2 从 39.3 跳到 46.4,GLM-4.7 从 41.0 掉到 32.8)，说明**框架本身就是跨模型对比的重要变量**;
- **MCP-Atlas 把 500 题公开集全部重评**，超时从 4 分钟放宽到 10 分钟(避免部署条件导致任务失败),judge 用 Gemini 3 Pro;
- **BrowseComp 的 judge 全部标准化为 OpenAI 官方评测 prompt + o3-mini**(报告发现开源 judge 有系统性偏差);
- **HLE 只评文本子集**,judge 是 GPT-5.2 (medium);
- **τ2-Bench 改了两处**:Retail/Telecom 换了优化后的用户模拟器 prompt(防"用户提前挂断"),Airline 套用 Claude Opus 4.5 system card 的 domain fixes——注意，这不是"作弊",而是修正基准本身的歧义，但确实意味着**分数不能直接和原榜上的其他模型横比**;
- **SWE-rebench**(2026-01,自动挖掘新 Issue 的去污染评测):GLM-5 **42.1%**,GLM-4.7 41.3%、Kimi K2.5 37.9%;闭源这边 Claude Opus 4.6 52.9%、GPT-5.2 (xhigh) 51.7%——**静态基准之外的"新题"上，开源和闭源差约 10 个点**。

## 5. 系统与部署：长程 Agent 的推理需求

### 5.1 长程 RL rollout 的推理系统设计

报告 3.6 节几乎是一篇独立的推理系统 mini-paper,主题是:**RL rollout 的优化目标不是吞吐，是端到端延迟，尤其是长尾延迟**——一个 straggler 卡住同步点，整步训练就白等。四个设计:

1. **PD 解耦(Prefill-Decode Disaggregation)**：多轮 Agent 场景里长前缀 prefill 频繁出现(对话历史、工具轨迹、代码上下文)，如果和 decode 混在同一批资源上，**一次重 prefill 就能打断一堆正在 decode 的请求**，长尾雪上加霜。把 prefill 和 decode 分到专用资源上，decode 稳定推进——这正是站内 [7.2 解耦架构设计](/AIInfraGuide/inference/模块四-推理优化/第7章-pd解耦架构/72-解耦架构设计) 讲的那个架构，在这里是 RL 系统的基本盘;
2. **多节点推理 + DP-attention**:EP64/DP64(专家并行、数据并行各切成 64 份)摊到 8 个节点，为突发流量备足分布式 KV 容量;DP-attention 专门避免跨 rank 拷贝 KV(想想站内 [7.3 KV 传输](/AIInfraGuide/inference/模块四-推理优化/第7章-pd解耦架构/73-kv传输与connector) 的账);
3. **FP8 rollout + MTP**:FP8 降每 token 延迟;MTP 在 RL 的小 batch 解码场景特别值钱——长尾 straggler 往往是稀有长上下文/复杂多轮推理/工具重轨迹，投机解码对它们的完成时间改善最大;
4. **心跳故障容错**:rollout 服务器周期性发心跳，挂了就从路由摘掉，重试自动绕开——长程训练跑几天，单机故障是常态不是意外。

还有 **DP-aware routing**(第 4 节)：同一 agent 实例的所有请求按一致哈希钉到同一 DP rank,多轮之间 KV 缓存不跨 rank 迁移，prefill 成本只随增量 token 增长。对多轮 Agent 负载，这是"KV 复用"的调度实现，和站内 [10.4 容量规划](/AIInfraGuide/inference/模块四-推理优化/第10章-生产部署与运维/104-容量规划) 的 KV 账本直接相关。

**搜索 Agent 的上下文管理**是另一个推理侧工程点：模型在超长上下文(>100K)下准确率明显下降。报告用 **keep-recent-k**(只保留最近 k 轮工具观测，更早的折叠掉;k=5)把 BrowseComp 从 55.3% 提到 62.0%;再叠加 discard-all 分层策略(总长超 32K 就清空工具历史重启)，最终 75.9。**牺牲了什么?** 折叠/清空会丢信息，但换来的是更多可执行的搜索步数——预算有限时，省上下文 = 更多轮次 = 更高分数。

<img src="/AIInfraGuide/images/glm-5-fig8-context-management.png" alt="BrowseComp 不同上下文管理策略的准确率" style="max-width: 75%; display: block; margin: 0 auto;" />
*图源：GLM-5 技术报告 Figure 8(arXiv:2602.15763)*

### 5.2 国产芯片全栈适配

报告声称"从第一天起"就全栈适配 7 家国产芯片(昇腾、摩尔线程、海光、寒武纪、昆仑芯、MetaX、燧原)，以昇腾 Atlas 800T A3 为例给了三个支柱:

- **W4A8 混合精度量化**：把 ~750B 模型塞进**单台 Atlas 800T A3**——Attention/MLP 用 W8A8(INT8),MoE 专家用 **W4A8(INT4)**，配 QuaRot(旋转去离群点)+ Flex_AWQ_SSZ 缩放校准;
- **融合内核**:Lightning Indexer(打分+ReLU+TopK 合成一个 kernel)、Sparse Flash Attention(选 token 与稀疏计算并行)、MLAPO(把 13 个小算子熔成"超级算子",Vector 与 Cube 单元并行);
- **推理引擎优化**:vLLM-Ascend 与 SGLang 适配——D2H 采样拷贝与下一步 decode 重叠消气泡、RadixCache/Prefix Cache 复用 KV、Attention DP + MoE EP 混合并行 + FlashComm 通信隐藏、MTP 提 NPU 计算密度。

官方口径：**单台国产节点 ≈ 双 GPU 国际集群的性能，长序列场景部署成本降 50%**。

## 6. 局限与意义

报告没有单独的 Limitations 章节，但正文里自认的边界非常清楚，逐条列出:

1. **前端端到端完成率全面落后 Claude Opus 4.5**(ISR:38.9/34.6/32.7 vs 52.2/39.7/46.9)——单项要求满足得很好，完整任务交付不行;
2. **链式任务误差累积**(52.3 vs 61.6)：需要长上下文一致性与长程自我纠错，报告点名这是"正在进行的研究";
3. **SWE-rebench 落后闭源约 10 个点**(42.1% vs Claude Opus 4.6 52.9%)——"新题"泛化仍弱;
4. **Vending-Bench 2 接近但未超过 Claude/Gemini**(\$4,432 vs \$4,967/\$5,478);
5. **GPQA-Diamond 86.0 vs GPT-5.2 (xhigh) 92.4,差约 6 分**——深度推理仍落后闭源前沿;
6. 训练集群规模、总算力未公开，成本不可独立评估。

还有一个"彩蛋"章节值得一提：报告透露 GLM-5 曾以 **"Pony Alpha"** 匿名身份在 OpenRouter 上线，社区猜测 25% 猜 Claude Sonnet 5、20% 猜 DeepSeek、10% 猜 Grok——这段更像品牌叙事，但说明了团队对"去掉牌子后能力是否被认可"的在意。

**工程视角的总结**:GLM-5 的意义不在单点指标，而在于**验证了三条可复制的路线**——DSA 式"接枝"稀疏化(20B tokens 适配 vs 从头训的 943.7B)、异步 RL 基础设施(解耦 + 确定性 top-k + TITO 是稳定的三根柱子)、以及"环境规模化"(10000+ 可验证 SWE 环境)驱动的长程能力。对推理工程师，这份报告最值钱的章节是 3.6 和 4.2——长尾延迟、PD 解耦、KV 复用路由，每一个都能在自家负载上复刻。

**延伸(官方博客口径，非论文)**:GLM-5 发布后两个月，GLM-5.1(2026-04)与 GLM-5.2(2026-06)相继登场。5.1 主打长程任务：SWE-bench Pro 58.4 开源 SOTA,VectorDBBench 上自主迭代 **600+ 轮、6000+ 次工具调用**、QPS 约 6 倍于单次 50 轮会话——"能干几小时"从论文叙事变成了实测;5.2 把上下文推到 **1M**，用 **IndexShare**(每 4 层共享一个 indexer)把 1M 上下文下每 token FLOPs 降 **2.9 倍**(动机：相邻 DSA 层的 top-k 选择有 70–100% 重叠，来自 IndexCache 论文),MTP 接受长度再 +20%,并转为 **MIT 开源**。

## 📝 总结

1. **范式转变**:GLM-5 的命题从"预测下一个 token"变成"自主完成数小时的长程工程任务"——从 Vibe Coding(人喊一句、模型写一段)到 Agentic Engineering(模型自己规划、实现、迭代)。
2. **架构**:744B/40B 稀疏 MoE(总参数翻倍、层数反减到 75 以省 EP 通信);DSA 稀疏注意力把长序列注意力算力降约 1.5–2 倍，且只需 20B tokens 接枝适配;MLA + Muon Split + MLA-256 把 KV 显存和解码计算两个账本同时管住;200K 上下文三段式 Mid-training。
3. **训练**:28.5T tokens 语料(代码去重后 +28%)；后训练是 Reasoning RL → Agentic RL → General RL 串行 + 跨阶段蒸馏;Agentic RL 的三大支柱是解耦(推理/训练分离)、TITO 保 token 级对齐、双侧裁剪 + 样本丢弃控 off-policy。
4. **评测**：八大 ARC 基准平均比 GLM-4.7 高约 20%,Intelligence Index v4.0 首个开源 50 分;SWE-bench Verified 77.8、BrowseComp(带上下文管理)75.9 全场最强；但自报口径要打折看(OpenHands 定制 prompt、judge 模型选择、τ2 基准修正),CC-Bench-V2 的 ISR 与链式任务仍落后 Claude。
5. **系统**：长程 Agent 推理 = PD 解耦 + 多节点分布式 KV + FP8/MTP 压长尾 + 心跳容错 + DP-aware 路由省 KV 复用；国产芯片 W4A8 方案单机装 750B。
6. **局限自认**：端到端交付差距、链式误差累积、SWE-rebench 落后闭源约 10 个点、训练成本未公开。

## 🎯 延伸思考：自我检验清单

- 能说出 GLM-5 的总参数/激活参数(744B/40B)与前代(355B/32B)的对比，以及"总参翻倍但层数减少"的动机(省专家并行通信)。
- 能解释 DSA 为什么是"动态"稀疏(lightning indexer 按内容划 top-k 重点)，并复述长序列注意力算力约 1.5–2 倍、适配预算 20B vs DeepSeek 943.7B 这两个数字。
- 能说明为什么 DSA 的 top-k 算子必须是确定性的(torch.topk vs CUDA top-k 的 RL 稳定性差异)，以及 RL 阶段为什么默认冻结 indexer。
- 能对比 MLA 与 GQA-8 的 KV 缓存维度(576 vs 2048)，并解释 Muon Split(按 head 正交化)和 MLA-256(head dim 192→256、头数减 1/3)各自修了什么账。
- 能画出异步 Agentic RL 的"推理引擎—编排器—训练引擎"解耦结构，并说出 TITO 解决什么问题(重 tokenize 破坏动作-奖励对齐)。
- 能写出 token 级重要性比 $r_t=\exp(\log\pi_\theta-\log\pi_{\text{rollout}})$ 和双侧裁剪 $[1-\epsilon_\ell,1+\epsilon_h]$,并解释它和 PPO 单侧 clip 的差异。
- 能复述 keep-recent-k 在 BrowseComp 上的量化收益(55.3%→62.0%,k=5;分层阈值 32K,最终 75.9)。
- 能说出 CC-Bench-V2 三个指标的含义(BSR/ISR/CSR)和 GLM-5 的相对位置(BSR 基本满分 100/100/100/95、ISR 三栈落后 Claude 约 5–14 点、大仓库探索 65.6 反超 64.5)。
- 能列举报告自认的三个局限(前端 ISR、链式任务误差累积、SWE-rebench 落后约 10 点)，不把"接近闭源"说成"超过闭源"。
- 能指出 GLM-5 评测中至少三处"口径"(OpenHands 定制 prompt、MCP-Atlas 超时 10 分钟重评、τ2-Bench 用户模拟器修正)，并说明它们如何影响横向比较。
- 能说明长程 Agent rollout 推理的四个系统件(PD 解耦、DP-attention 免 KV 拷贝、FP8+MTP 压长尾、心跳容错)各自对应站内哪一章。
- 能区分"论文报告数字"与"官方博客数字"(如 5.1 的 600+ 轮迭代、5.2 的 1M 上下文/2.9× FLOPs)，不混用口径。

## 📚 参考资料

- 论文/原文:
  - [GLM-5: from Vibe Coding to Agentic Engineering — arXiv 2602.15763](https://arxiv.org/abs/2602.15763)：本文解读对象，Zhipu AI & Tsinghua University,2026-02-17 发布(v2 2026-02-24)。
  - [GLM-5 全文 HTML 版](https://arxiv.org/html/2602.15763v2)：含全部表格(表 7/8/9/10/11)与附录评测细节。
  - [DeepSeek-V3.2 技术报告](https://arxiv.org/abs/2512.02556):DSA 的出处，报告大量引用的对比基线。
  - [SWE-rebench](https://arxiv.org/abs/2505.20411)：自动挖掘新 Issue 的去污染评测，报告 6.2.4 节的对比对象。
- 官方发布:
  - [zai-org/GLM-5 GitHub 仓库](https://github.com/zai-org/GLM-5)：模型、代码与 5.1/5.2 发布说明。
  - [GLM-5.1: Towards Long-Horizon Tasks — z.ai 官方博客](https://z.ai/blog/glm-5.1):600+ 轮自主迭代、6,000+ 次工具调用、SWE-bench Pro 58.4 的出处。
  - [GLM-5.2: Built for Long-Horizon Tasks — z.ai 官方博客](https://z.ai/blog/glm-5.2):1M 上下文、IndexShare(每 4 层共享 indexer、per-token FLOPs 降 2.9×)、MTP 接受长度 +20%、MIT 开源。
  - [GLM-5.2 模型卡 — Hugging Face](https://huggingface.co/zai-org/GLM-5.2)：配置与完整基准表。
  - [IndexCache — arXiv 2603.12201](https://arxiv.org/abs/2603.12201)：相邻 DSA 层 top-k 选择 70–100% 重叠，IndexShare 的动机依据。
- 站内相关:
  - [8.2 Tool Calling 与 Reasoning(站内)](/AIInfraGuide/inference/模块四-推理优化/第8章-生产级服务特性/82-tool-calling与reasoning):Agent 工作负载对推理系统的要求，本文第 5 节的前置。
  - [7.2 PD 解耦架构设计(站内)](/AIInfraGuide/inference/模块四-推理优化/第7章-pd解耦架构/72-解耦架构设计):GLM-5 的 rollout 系统把 prefill 和 decode 分离到专用资源，正是这一章的架构。
  - [10.4 容量规划(站内)](/AIInfraGuide/inference/模块四-推理优化/第10章-生产部署与运维/104-容量规划)：长程 Agent 的分布式 KV 容量账本(EP64/DP64 8 节点、DP-attention 免拷贝)。
