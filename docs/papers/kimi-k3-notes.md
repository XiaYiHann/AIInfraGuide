---
title: "Kimi K3 解读：2.8T 参数的开源前沿模型是怎么造出来的"
description: "拆解 Moonshot AI 的 47 页技术报告：2.8T 总参数/104B 激活的 MoE、896 专家每 token 激活 16 个、1M token 上下文，以及 KDA 混合注意力、Attention Residuals、三域三档 RL 与百万 token 强化学习基建背后的 2.5x 缩放效率提升"
pubDate: 2026-08-09
originalUrl: "https://arxiv.org/abs/2607.24653"
sourceType: "paper"
originalAuthor: "Moonshot AI (Kimi Team)"
tags: ["Kimi K3", "MoE", "混合注意力", "长上下文", "强化学习", "AI Infra"]
---

> 原文：[Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653)(Kimi Team / Moonshot AI,arXiv 2607.24653,2026-07-27)+ [官方仓库 README](https://github.com/moonshotai/Kimi-K3);本文访问日期 2026-08-15

2026 年 7 月 27 日，Moonshot AI 放出了 Kimi K3 的完整权重和一份 47 页技术报告：**2.8 万亿(2.8T)总参数、104B 激活参数的混合专家(MoE)模型，每个 token 只激活 896 个路由专家中的 16 个，原生视觉能力，1M（百万）token 上下文窗口**——报告称这是“世界上第一个开放的 3T 级模型”，也是迄今最大的开源权重模型。报告给出的最关键数字是：相比 Kimi K2,整体缩放效率(scaling efficiency)提升约 2.5 倍（报告拟合的缩放定律曲线，横轴训练 FLOPs、纵轴验证损失；TDS 的解读口径是“同等质量、不到一半的训练算力”——注意这是解读，不是报告原文）。但比数字更值得读的是报告的结构本身：它把注意力机制、MoE 稳定性、强化学习环境、训练基础设施全部摊开讲，唯独没有公开总训练 token 数、GPU 小时数和训练成本——这正是其他前沿实验室连提都不提的部分。这篇解读带你顺着报告的骨架走一遍：一个 2.8T 参数的模型，究竟是怎么从架构、数据、强化学习到基础设施一步步造出来的。

<!-- more -->

## 📑 目录

- [1. 背景:2.8T 为什么是最大开源](#1-背景28t-为什么是最大开源)
- [2. 架构：三条信息流，三个新组件](#2-架构三条信息流三个新组件)
- [3. 训练配方：数据、缩放定律与阶段](#3-训练配方数据缩放定律与阶段)
- [4. 后训练：九个专家模型，蒸馏成一个](#4-后训练九个专家模型蒸馏成一个)
- [5. 评测：报告自报与第三方口径](#5-评测报告自报与第三方口径)
- [6. 工程与系统：让 2.8T 模型跑起来的基建](#6-工程与系统让-28t-模型跑起来的基建)
- [7. 局限与开源意义](#7-局限与开源意义)
- [📝 总结](#-总结)
- [🎯 自我检验清单](#-自我检验清单)
- [📚 参考资料](#-参考资料)

## 🗺️ 原文阅读地图

Kimi K3 技术报告共 47 页，按“架构 → 预训练 → 后训练 → 基础设施 → 评测”组织。本文选择性精讲如下：

| 原文单元 | 处理深度 | 本文位置与理由 | 来源锚点 |
| --- | --- | --- | --- |
| §1 Introduction（2.8T/104B 激活、1M 上下文、开源定位） | 精讲 | 第 1 节，先给规模坐标 | §1 |
| §2.1 Hybrid Attention（KDA 便签本式注意力） | 精讲 | 第 2.1 节，机制卡 1,含公式 | §2.1 |
| §2.2 Attention Residuals（逐通道全秩输出门） | 精讲 | 第 2.3 节，机制卡 2 | §2.2 |
| §2.3 Stable LatentMoE（896 专家、quantile balancing） | 精讲 | 第 2.4 节，机制卡 3 | §2.3 |
| §2.4 Native Vision + §2.5 Per-Head Muon | 简述 | 第 2.5 节合并叙述 | §2.4;§2.5 |
| §3.1 Pre-Training Data（四域 + 大规模视觉） | 简述 | 第 3.1 节引用数据配比 | §3.1 |
| §3.2 Scaling Law（2.5× 缩放效率） | 精讲 | 第 3.2 节，区分“报告拟合”与“第三方解读口径” | §3.2 |
| §3.3/3.4 Training Recipe 与 Long-Context Extension | 简述 | 第 3.3-3.4 节引用关键设置 | §3.3;§3.4 |
| §4 Post-Training（MOPD 九师一徒蒸馏、部署感知后训练） | 精讲 | 第 4 节，机制卡 4 | §4 |
| §5 Infrastructure（KDA co-design、3T 预训练、1M RL、推理服务） | 简述 | 第 4.4 与第 5 节引用关键账本 | §5.1-5.4 |
| §6 Evaluations（自报、内部、第三方、成本效率） | 精讲（数字表） | 第 5 节，自报与第三方口径分开呈现 | §6.1-6.4 |
| §7 Case Studies、§8 Conclusion、附录 | 不展开 | 不改变本文的机制承诺 | §7-8;Appendix |

📌 **本文承诺**：读完后，你应该能解释 KDA 把“文件柜”换成“便签本”省了什么、Stable LatentMoE 的专家稳定性从哪来，并区分 2.5× 缩放效率是“报告拟合”还是“实测”，以及报告刻意没公开哪些训练成本。

## 1. 背景：2.8T 为什么是最大开源

**先问一个问题：一个模型 2.8T 参数，为什么说它每 token 只“用”104B？** 这就要分清两个概念：

- **总参数(Total Parameters)**：模型权重文件里存的所有数字，2.8T 个。它决定显存需求——推理时这些权重必须能被快速读到，所以全部要驻留在高速存储里。
- **激活参数(Activated Parameters)**：处理一个 token 时真正参与计算的那部分，104B 个。它决定算力需求。

**混合专家(Mixture-of-Experts,MoE)** 就是把两者拆开的机制：把前馈网络（Transformer 里负责“理解”的那一大块）换成几百个小型“专家”网络加一个路由器，每个 token 由路由器挑少数专家处理。MoE 省的是计算，不是显存——2.8T 参数一张卡根本放不下，这也是为什么 Tom's Hardware 报道里 Moonshot 建议用 64 个或更多加速器组成的 supernode 来服务它。这一点和站内 [6.4 数据并行与专家并行](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/64-数据并行与专家并行)讲的 DeepSeek 671B 是同一套账，只是 K3 把规模又推高了一个数量级。

报告给出的 K2 → K3 架构对比（来源：报告 Table 1），可以一眼看清“变大了什么”：

| 📊 维度 | Kimi K2 | Kimi K3 | 变化 | 来源口径 |
|---|---|---|---|---|
| 层数 | 61 | 93 | ↑ 52% | 报告 Table 1 |
| 总参数 | 1.04T | 2.78T | ↑ 167% | 报告 Table 1 |
| 激活参数 | 32.6B | 104.2B | ↑ 220% | 报告 Table 1 |
| 路由专家 / 每 token 激活 | 384 / 8 | 896 / 16 | ↑ 133% / ↑ 100% | 报告 Table 1 |
| 共享专家 | 1 | 2 | ↑ 100% | 报告 Table 1 |
| 训练上下文 | 128K | 1M | 8× | 报告 Table 1 |
| 注意力 | 纯 MLA | 混合 KDA + MLA（69 + 24 层） | 结构变化 | 报告 Table 1 |
| 激活函数 | SwiGLU | SiTU-GLU | 结构变化 | 报告 Table 1 |

📌 **关键点**：896/16 意味着稀疏度 56(896 ÷ 16)，即每 token 只走过专家池的 1.8%。专家越多、每个越“专”，模型容量越大；但稀疏度越高，训练越不稳定、负载越难平衡——报告用一整节(Stable LatentMoE)专门解决这个问题，后面第 2.4 节讲。

## 2. 架构：三条信息流，三个新组件

报告的架构叙述有个清晰的骨架：模型同时沿着**序列长度、网络深度、模型宽度**三个维度扩张信息流动：

- 序列维度：KDA（线性注意力）为主 + Gated MLA（全局注意力）为辅的**混合注意力**，每个 block 是 3 层 KDA + 1 层 MLA 交替（93 层里 69 层 KDA、24 层 MLA）；
- 深度维度：**Attention Residuals(AttnRes)**，让每一层能直接“回头看”前面所有层的输出；
- 宽度维度：**Stable LatentMoE**，把专家池扩到 896 个。

三个组件都是首次出现的概念，逐个拆。

<img src="/AIInfraGuide/images/kimi-k3-fig2-architecture.png" alt="Kimi K3 三维架构总览" style="max-width: 75%; display: block; margin: 0 auto;" />

*图源：Kimi K3 技术报告 Figure 2(arXiv:2607.24653)*

### 2.1 KDA：把“文件柜”换成“便签本”

**问题是：1M token 上下文为什么贵？** 传统注意力(softmax attention)处理每个 token 时，要回头看前面所有 token——实现方式是给每个 token 存一份 key-value 记录（KV Cache,键值缓存：注意力计算所需的中间量，存下来避免重复计算），新 token 要和缓存里每一份记录做比较。打个比方：这像一个文件柜，每来一个词就塞一张卡片，新词要翻遍整柜卡片才能决定“我跟谁有关”。上下文到 1M 时，柜子大得离谱，每次翻找都越来越贵。

**Kimi Delta Attention（KDA,三角洲注意力）** 的做法是：不存卡片，而是维护一个固定大小的“便签本”——一个不断被覆盖更新的摘要状态 $S_t$。便签本永远不长，所以 1M 上下文也付得起。代价是：摘要记不住每个早先 token 的精确内容，所以它必须学会遗忘，并且要能控制遗忘速度；而“精确回忆某个早先 token”这种能力，由每 4 层里的那 1 层 Gated MLA 补回来。

单头视角的状态更新公式（报告 Eq. 1）：

$$S_t = \underbrace{(I - \beta_t k_t k_t^\top \text{Diag}(\alpha_t))}_{\text{遗忘项:按通道衰减旧状态}} S_{t-1} + \underbrace{\beta_t k_t v_t^\top}_{\text{写入项:把新 token 的信息写进状态}}, \qquad \tilde{o}_t = S_t^\top q_t$$

逐项看：

- $S_t \in \mathbb{R}^{d_k \times d_v}$：当前摘要状态（固定大小，不随序列增长）；
- $\alpha_t \in (0,1)^{d_k}$：**
逐通道**的一步保留因子——每个特征维度有自己的遗忘旋钮，$\alpha$ 接近 1 就记得久，接近 0 就忘得快；
- $\beta_t \in (0,1)$：写入强度——新 token 的信息以多大力度写进状态；
- $q_t, k_t, v_t$：当前 token 的查询/键/值向量；$\tilde{o}_t$ 是输出前的一步（之后还有输出门）。

💡 **提示**：这是一个"delta rule"式的递推：先按通道衰减旧状态，再叠加新写入。$\alpha_t$ 和 $\beta_t$ 都由当前 token 的输入动态生成（报告 Eq. 2 的低秩投影 + sigmoid），不是固定超参——模型自己学会“这句该记住、那句该忘掉”。

**数值上还有个工程坑**：分块并行计算时，键向量要被“累计衰减的倒数”重缩放，而 $\alpha < 1$ 的连乘倒数可以涨到无穷大，在 BF16 低精度下直接溢出。Kimi Linear（前作）用负 Softplus 映射，对数衰减下界是 $-\infty$;K3 改用带下界的缩放 sigmoid：$g_t^h = g_{\min} \cdot \text{Sigmoid}(e^{A_h} z_t^h) \in (g_{\min}, 0)$,固定 $g_{\min} = -5$。这样每个保留因子 $\alpha > e^{-5} \approx 6.7 \times 10^{-3}$,16-token 分块内的累计对数衰减落在 $(-80, 0)$,倒数不超过 $e^{80}$,在 BF16 动态范围内——所有分块（包括对角块）都能用稠密 Tensor Core 矩阵乘，消灭了原来的“逐位置对”慢路径。这就是“带下界衰减”换来算力效率的具体故事。

<img src="/AIInfraGuide/images/kimi-k3-fig3-kda-lower-bound.png" alt="KDA 带下界衰减与分块计算" style="max-width: 75%; display: block; margin: 0 auto;" />

*图源：Kimi K3 技术报告 Figure 3(arXiv:2607.24653)*

**训练/推理怎么并行？** KDA 是递推，天然串行；GPU 讨厌串行。报告的做法是分块并行(chunkwise)：序列切成块，块内展开成几个稠密矩阵乘（张量核心的强项），块与块之间只传递固定大小的状态。剩下的串行尾巴由专门 kernel 处理——这是第 6 节基础设施的主角。

**位置编码呢？** K3 所有注意力层都不加显式位置编码(NoPE)：位置信息由 KDA 的递推衰减机制隐式携带。附带好处：扩展到 1M 上下文不需要改任何位置编码参数（不用调 RoPE 基频、不用 YaRN 插值），8K 阶段训练的权重直接就能跑到 1M。

### 2.2 Gated MLA：每四层留一层“精确回忆”

**Multi-head Latent Attention（MLA,多头潜在注意力）**：DeepSeek-V2 提出的方案，把每个 token 的键值压缩成一个低维潜在向量 $c_t$ 缓存，注意力计算时再用学到的升维投影重建完整键值。这样 KV Cache 体积大幅缩小，但保留了全局逐 token 注意力。K2 用过， K3 保留在每 4 层的 1 层里（24 层 MLA + 额外一层）。

与 K2/K2.5 不同的是：K3 的 MLA 层**不加位置编码**，且给输出加了一个输入相关的逐通道全秩输出门：$y_t = W_o[\text{Sigmoid}(W_g x_t) \odot \tilde{o}_t]$——每个 token 自己决定“从全局注意力里读哪些通道”。位置敏感性交给 KDA,全局内容交互交给 MLA,分工明确。

### 2.3 Attention Residuals：别玩“传话游戏”

**问题是：信息穿过 93 层之后还剩下什么？** 标准残差连接（Residual Connection,把每层输出加到主通路上）像一条传话游戏的队伍：每层在纸条上添一句，传到队尾时，开头几句早就被淹没。深层网络烧掉大量容量只是为了“别把早期信息弄丢”。

**Attention Residuals（AttnRes,注意力残差）** 把注意力那套“软查找”从序列维度搬到深度维度：每一层定义一个可学习的“伪查询” $w_l$,对前面所有层的输出（加上 token 嵌入）做 softmax 加权求和——每一层都能直接回头读前面任何一层，而不是只吃上一层的输出。早期信号不再被冲淡。

代价是内存：全量形式需要把每层输出都留着，内存和通信开销 $O(Ld)$（$L$=层数，$d$=隐藏维）。报告用 **Block 版本**解决：把 93 层分成 8 个块（每块约 12 层），块内输出先求和压缩成一个块表示，跨块才做完整注意力——开销从 $O(Ld)$ 降到 $O(Nd)$（$N$=块数），内存和通信都省，推理时还能用 online softmax 合并块间结果。经验上 $N \approx 8$ 就能恢复大部分收益（报告引用的 AttnRes 论文结论）。

### 2.4 Stable LatentMoE:896 个专家怎么不打架

**问题：专家池从 384 扩到 896,为什么普通 MoE 撑不住？** 两个原因：

1. **激活爆炸**：普通 MoE 里每个被选中的专家都吃完整的 $d$ 维表示，专家多了通信和权重搬运线性涨。LatentMoE（2026 年的论文）把路由专家的宽度从“全模型宽”降到“紧凑潜在空间” $\ell$：共享专家保留全宽通路处理通用变换，专门化专家只在一个窄得多的潜在空间里工作。K3 的潜在 MoE 维是 3584（恰好是隐藏维 7168 的一半）。这才能让 896 个专家 × 16 激活可行。
2. **极端稀疏下的两个失效模式**：路由分支里 down-projection → 专家 → up-projection 形成近四次连续矩阵乘的“病态链条”，2.8T 规模下内部激活直接爆炸；同时 ~10³ 个专家的负载平衡超出既有无辅助损失方法的舒适区。Stable LatentMoE 用三件套回应：

- **RMSNorm 插在专家聚合和升维投影之间**：抑制路由分支的尺度波动（还顺带降低了验证损失）；
- **SiTU-GLU 激活函数**(Sigmoid Tanh Unit GLU)：SwiGLU 的两个因子都无界，大坐标相乘容易出激活离群点、低精度溢出。SiTU-GLU 给两个分支各加一个平滑封顶 $\text{softcap}(x, \beta) = \beta \tanh(x/\beta)$：

$$\text{SiTU-GLU}(x) = \underbrace{\beta_1 \tanh\!\left(\frac{W_g x}{\beta_1}\right)}_{\text{门分支:封顶}} \odot \underbrace{\text{Sigmoid}(W_g x)}_{\text{保留 Swish 的线性正区}} \odot \underbrace{\beta_2 \tanh\!\left(\frac{W_u x}{\beta_2}\right)}_{\text{上分支:封顶}}$$

K3 取 $\beta_1 = 4$、$\beta_2 = 25$：原点附近近似线性（保留 SwiGLU 的局部响应），大输入被压到 $|f(x)| \leq \beta_1 \beta_2 = 100$——**用“大值封顶”换“低精度下的数值安全”**。

- **Quantile Balancing（QB,分位数平衡）** 做负载均衡：传统做法是给路由器加辅助损失(auxiliary loss)惩罚不均衡，但会干扰主目标；无辅助损失方案用专家偏置 $b_j$ 调节路由分数。QB 的关键改进是：每次前向用一次 Top-(k+1) 路由拿到“门槛” $\alpha_i$——一个 token 想进 Top-k 必须超过的分数线——然后给每个专家设偏置，使其恰好收到目标负载 $q = mk/n$（$m$ 个 token、$n$ 个专家）。偏置更新就是算一个分位数：

$$\hat{b}_j^{(t+1)} \leftarrow -\,\text{quantile}_{\,1 - k/n}\left(s_{:,j} - \alpha^{(\cdot)}\right)$$

即“各 token 对专家 j 的分数裕量”的 $(1 - k/n)$ 分位数。大规模训练时精确分位数算不起（裕量有几百万个、散在各 rank），报告用**直方图 + 一次 all-reduce** 估计：每专家只传几百个 bin 的计数，全局 batch 的分位数就能近似出来。偏置只影响派发，不参与混合权重和路由器梯度——用“不动优化目标”换“负载可控”。推理时偏置冻结。

### 2.5 视觉与优化器：两个“反直觉”决定

- **MoonViT-V2 从零训练**：以往视觉编码器（如 K2.5）用 SigLIP 这类对比预训练模型初始化。K3 反其道而行，完全从零训练（27 层、401M 参数），理由很工程：对比预训练编码器接上 LLM 后联合优化不稳定（梯度范数持续偏高且频繁尖峰），而从零训练稳定得多，而且最终视觉评测打平——说明“对比预训练作为初始化”在大规模多模态下并非必需。图像/视频共享参数，像素 2×2 下采样把视觉 token 数降 4 倍，3584×3584 的图也能装进 1M 上下文。
- **Per-Head Muon**：优化器沿用 Muon（对矩阵参数做 Newton-Schulz 正交化的优化器），但对注意力投影做了逐头变体——每个 head 的动量块单独正交化，避免大梯度 head 主导共享更新方向。换来更均衡的学习动态和更便宜的正交化迭代。

## 3. 训练配方：数据、缩放定律与阶段

### 3.1 数据：四域 + 大规模视觉

预训练语料横跨**网页文本、代码、数学、知识**四个文本域，外加覆盖 caption、图文交错文档、OCR、感知、视频和“视觉编码数据”的大规模视觉语料。管道沿用 K2 并在 K2.5 上打磨：规则启发式 + 分类器质量打分 + 去重；知识和数学语料沿用 K2 的“改写配方”（风格和视角多样的提示、分块自回归生成、对照原文的保真校验）。视觉侧的一个新动作是大幅扩充程序化多模态数据：把代码片段和渲染结果配对（SVG、3D 资产、网页、游戏、CAD 图纸），训练时坐标监督同时给绝对和归一化格式。

⚠️ **注意（忠实性声明）**：报告没有公开各域的具体配比、语料总量或去重后的 token 数。任何声称“K3 用了 XX T token”的说法都出自第三方猜测，不在本文引用范围。

### 3.2 缩放定律：2.5× 从哪来

架构、数据、训练三方面的改动改变了最优训练区间，所以团队重新做了缩放定律研究，重新调 batch size、学习率、**每参数 token 比(TPP,Tokens-Per-Parameter)** 和模型形状。在留出集（分布外验证数据）上拟合的曲线显示：这些改进合计带来约 2.5× 的整体缩放效率提升（报告图 7：同样 FLOPs 下验证损失显著更低）。

<img src="/AIInfraGuide/images/kimi-k3-fig7-scaling-law.png" alt="Kimi K2 与 K3 缩放定律" style="max-width: 75%; display: block; margin: 0 auto;" />

*图源：Kimi K3 技术报告 Figure 7(arXiv:2607.24653)*

顺带一个方法论亮点：报告专门对比了 **cosine 衰减和 WSD(Warmup Stable Decay)两种学习率调度**，结论是 cosine 更优——但前提是每种调度独立做缩放定律搜索。因为两种调度的最优峰值学习率和 batch size 差很多，共用一套超参比较会不公平地偏向其中一方。这个“公平比较”的自觉值得记下来。

### 3.3 训练阶段与显式配方

- **优化**：Per-Head Muon + K2 引入的权重裁剪(weight clipping)+ QB 负载均衡；cosine 学习率 + 1% 线性 warmup;权重衰减 0.1。
- **原生多模态**：语言和视觉从训练第一天就联合优化（视觉 token 和文本 token 在同一下一个 token 预测目标里交错），没有“事后接编码器”的对齐阶段。
- **上下文课程**：预训练从 8K 上下文开始，后段扩到 64K;cooldown（冷却）阶段再从 256K 一路扩到 1M——四阶段渐进扩展，把昂贵的长序列计算压缩在训练预算的很小一部分里。
- **长上下文数据**：自然长文档/长视频里垃圾很多（近似重复、二进制块、截断文件），要专门的清洗管道（精确+模糊去重、视频逐帧感知哈希、结构校验）；因为真正长而连贯的文档稀缺，要上采样防止被短文本淹没；还合成“必须跨全 1M 上下文才能解”的任务数据，防止注意力退化成局部模式。

### 3.4 算力与成本：报告没有说的部分

**报告未公开**：总训练 FLOPs、GPU 小时数、硬件规模、训练成本。我们能确认的只有侧写：Tom's Hardware 报道提到训练在 NVIDIA H200 和一台未具名的“替代厂商 GPGPU”上进行；报告 5.2 节描述的是 PP + EP + ZeRO-1 DP + Pipeline ZeRO-2 + CP 的多维并行（层间并行：模型按层切成几段，每卡负责一段；专家并行：专家分散到多卡；数据并行：每卡各算不同数据，ZeRO-1 只把优化器状态分片存储；上下文并行：一条长序列切成多段分给多卡）（见第 6 节）。任何关于 K3 训练成本的具体数字都只能算估算，本文不给。

## 4. 后训练：九个专家模型，蒸馏成一个

后训练是三段式：**SFT（监督微调）→ RL（强化学习）→ MOPD（多教师同策略蒸馏）**。

### 4.1 SFT：给 RL 一个像样的起点

在 K2/K2.5 管道上大幅扩了复杂 agentic 任务覆盖：用先前 Kimi 系列里的领域专用模型合成轨迹，再经过多阶段验证 + 人工标注。所有轨迹用自研 **XTML 模板**（可扩展 token 标记语言，报告附录 F）序列化。从 SFT 起就做量化感知训练(QAT)：专家权重用 MXFP4、激活用 MXFP8（详见 4.4）。

### 4.2 RL:3 个领域 × 3 档推理强度 = 9 个专家

**问题是：一个模型又要写代码、又要做助手、又要深度思考，还要能被指挥“别想太多”，一个策略够用吗？** 报告的选择是分开练：3 个领域（通用任务 / 通用 Agent / 编程 Agent）× 3 档推理强度(low / high / max)= 9 个专家模型，每个只练自己的切片，最后蒸馏回一个。报告图 8 显示：RL FLOPs 增长时，工具调用步数和各能力分数一起稳定上升。

<img src="/AIInfraGuide/images/kimi-k3-fig8-rl-scaling.png" alt="RL FLOPs 与能力分数变化" style="max-width: 75%; display: block; margin: 0 auto;" />

*图源：Kimi K3 技术报告 Figure 8(arXiv:2607.24653)*

配套机制：

- **Partial rollout（部分展开）**：长任务跑一轮可能要几小时，传统“等全部 rollout 结束再更新”会饿死流水线。做法是等 $\lambda$（如 0.5）比例的轨迹完成就暂停生成、先做策略更新，没跑完的轨迹排队到下一轮接着跑。代价是数据陈旧(off-policy)，靠策略优化里的逐 token 正则化兜住稳定性。
- **推理强度预算控制**：给每个问题按冷启动模型的消耗估一个初始 token 预算 $b_0(x)$,轨迹总 token 数超过 $\tau \cdot b_0(x)$ 就奖励判 $-1$;训练时把乘数 $\tau$ 从大往小退火(anneal)，先练出 max 档，再收紧出 high / low 档。用“超预算即惩罚”换“模型真的尊重思考强度指令”。对无法自动判分的通用任务，Agentic GRM（生成式奖励模型）先读输出、现写评分细则、逐条打分（强制“读→写 rubric→打分→记分”四步协议），同样套一个 verbosity 预算防“写越长越赢”。
- **环境即资产**：RL 的价值全在环境。报告列出七类自建环境：统一白盒 RL 环境（harness 可配置，能实例化 Kimi Code / Claude Code / Codex / OpenClaw 等，防止模型过拟合单一工具协议）、知识图谱引导的任务合成（Agent 自己长出一棵概念树，采样节点生成搜索词、检索真实资料、合成任务）、Agent 环境中的可验证问题（投资银行/数据分析/法律等几十上百步的交付物）、GPU kernel 优化任务（正确性+性能双奖励，配反作弊检测：识别 CUDA graph 回放、输入缓存、精度偷降等 hack）、个人助手任务（Gmail/Notion/Slack/Canvas 的 mock 实现，单次 rollout 可上千次工具调用、上百万上下文 token）、自主执行任务 AET（只给初始状态/目标/工具/预算/验证器，公开验证器给反馈、隐藏验证器打终局分）、Web 开发任务（确定性检查 + 模型裁判，项目构建失败或“假装实现”直接零分）。

### 4.3 MOPD：九个老师教一个学生

蒸馏目标不是传统的 KL 散度，而是**同策略逐 token 奖励**（报告 Eq. 15）：对域 $d$、强度档 $e$,学生 $\pi_\theta$ 在 token $y_t$ 上拿的奖励是

$$r_{\text{opd}}(y_t \mid e, x, y_{<t}) = \text{clip}\!\left(\text{sg}\!\left(\log \frac{\pi_{\text{teacher}}^{(d,e)}(y_t \mid x, y_{<t})}{\pi_\theta(y_t \mid e, x, y_{<t})}\right),\; -R_{\max},\; R_{\max}\right)$$

即“老师觉得这个 token 比我（学生）更该被选中的程度”，截断在 $[-R_{\max}, R_{\max}]$ 防止极端优势信号；`sg` 表示停梯度，老师不受学生影响。这份稠密奖励直接接进 RL 框架，天然支持 partial rollout。报告试过更细的 top-k 蒸馏目标，没看到收敛速度或最终性能优势，就用了这个简单版。

### 4.4 部署感知的后训练：训练时就在为省钱

- **MXFP4 量化感知训练**：专家权重（占参数内存大头）量化为 MXFP4（4 位浮点格式）、激活 MXFP8,非专家部分（注意力投影、潜在 MoE 投影、共享专家、路由器）保持高精度；SFT 和 RL 全程 QAT,且 RL 的 rollout 和训练用同一套量化——训练与推理的精度口径一致，没有“上线才量化”的落差。
- **投机解码的草稿模型**：预训练自带一层多 token 预测(MTP)层，结构恰好匹配 EAGLE-3 的草稿模型，于是把 MTP 层微调成草稿模型（目标模型冻结）。草稿输入融合第 1、第 4 和最后一个 AttnRes 块的低/中/高层特征，初始化矩阵 $[0\,0\,I]$ 保证起步时等价于纯高层特征。关键是损失函数直接用接受率的负对数（报告 Eq. 16）：

$$L_{\text{LK}} = -\log \sum_{x \in V} \min(p(x), q(x))$$

其中 $p, q$ 分别是目标模型和草稿模型在温度 1 下的分布。常规 KL 散度代理不保证最大化接受率，而接受率才真正决定投机解码加速多少——**直接优化要用的指标，而不是近似指标**。

## 5. 评测：报告自报与第三方口径

⚠️ **读这一节前先记住三个口径**（来源：报告 §6.1.3）：① 所有 Kimi K3 结果均为 reasoning effort=max、temperature=1.0;② 闭源对手中，Claude Fable 5 的结果“含 fallback 行为”、GPT-5.6 Sol 的结果“含可能的 cyberguard 拦截”；③ 不少对手分数引自 Artificial Analysis（截至 2026-07-23）等第三方榜单，且各模型用的 agent harness 不同(Kimi Code / Claude Code / Codex)——这不是严格同构评测。

### 5.1 报告自报的基准（精选）

| 📊 基准 | Kimi K3 | Claude Fable 5 | GPT-5.6 Sol | 备注（来源口径） |
|---|---|---|---|---|
| GPQA Diamond（研究生级科学） | 93.5 | 92.6 | 94.1 | 报告 Table 2 |
| HLE-Full（无工具 / 带工具） | 43.5 / 56.0 | 53.3 / 63.0 | 44.5 / 58.0 | 报告 Table 2 |
| CritPt（批判性思维） | 23.4 | 28.6 | 32.3 | 引自 Artificial Analysis,2026-07-23 |
| DeepSWE（软件工程） | 67.5 | 70.0 | 73.0 | 报告自测(Kimi Code harness) |
| ProgramBench | **77.8** | 76.8 | 77.6 | 报告 Table 2,最佳 |
| Terminal-Bench 2.1 | 88.3 | 88.0 | 88.8 | 各模型取跨 harness 最佳 |
| FrontierSWE（长程） | 81.2 | 86.6 | 71.3 | 截至 2026-07-16 重算 |
| SWE-Marathon（GPU kernel 向） | **42.0** | 35.0 | 39.0 | 报告自测，领先 Fable 5 达 7 分 |
| BrowseComp（浏览器搜索） | **91.2** | 88.0 | 90.4 | 报告自测；全 1M 上下文无管理时为 90.4 |
| DeepSearchQA(F1) | **95.0** | 94.2 | — | 报告 Table 2 |
| MCPMark-Verified（工具调用） | **94.5** | 87.4 | 92.9 | 报告 Table 2 |
| GDPval-AA v2(Elo) | 1686 | 1747 | 1736 | 引自 Artificial Analysis |
| OmniDocBench（文档理解） | **91.1** | 89.8 | 85.8 | 报告 Table 2 |
| Math-Vision（无 / 带 Python） | 94.3 / 97.8 | 94.8 / 98.6 | 95.8 / 97.8 | 报告 Table 2,三次平均 |
| ZeroBench-main（pass@5,无 / 带工具） | 23.0 / 41.0 | 23.0 / 46.0 | 17.0 / 35.0 | 官方设置跑 5 次 |

📌 **关键点**：K3 的画像很清晰——Agentic 与编程是强项（BrowseComp、MCPMark、SWE-Marathon、ProgramBench 拿第一），研究生级推理接近前沿（GPQA 93.5 与 GPT-5.5 并列、仅差 GPT-5.6 Sol 0.6），研究级推理是明确短板（CritPt 23.4、HLE-Full 56.0 双双落后）。报告原文结论：整体仍落后 Claude Fable 5 与 GPT-5.6 Sol,但稳定超过评测套件里其他所有开源与闭源模型。

### 5.2 第三方独立评测（截至 2026-07-23,报告 §6.3 汇总）

| 📊 榜单 | Kimi K3 | 位置 | 领跑者 | 来源 |
|---|---|---|---|---|
| Artificial Analysis 智能指数 v4.1 | 57.1 | #4/580（按模型家族合并变体后 #3） | Fable 5(59.9)、GPT-5.6 Sol(58.9) | Artificial Analysis |
| Vals AI 工业加权指数 | 74.7% | #2/39 | Fable 5(75.1%) | Vals AI |
| WebDev Arena(Elo) | 1,678 | **#1/99,首个登顶的开源模型** | 领先 Fable 5(1,634) | LMArena 众包盲测 |
| Text Arena(Elo) | 1,486 | #8/200 | — | LMArena |
| Agent Arena | 9.1 | #4/37 | Fable 5(12.7) | 2026-07-19 起开放投票 |

💡 **提示**：报告自报的 WebDev Arena 是 1,678;Tom's Hardware 报道的 Frontend Code Arena 是 1,679、且提到“6/7 个域第一、从 K2.6 的 #18 跳到 #1”。两个数字差 1,是因为 Elo 随对局持续漂移（报告自己也声明了）——同一榜单的不同时间快照，不是矛盾。

### 5.3 成本效率：报告最“卖货”的一节

报告把“分数 vs 每任务成本”画了四张散点图（图 13），口径：成本按 API 计费（部分引自 Artificial Analysis 的 token 定价，2026-07-23）：

- **BrowseComp**：K3 91.2% 拿下最高分，每任务 $2.03——GPT-5.6 Sol 90.4% 但贵约一倍，Claude 系在 max 强度下贵一个数量级；
- **Kimi Code Bench 2.0（内部基准）**：落后 Fable 5 4.0 分，但成本只有它的 38%;K3 的 high 档就用约 1/3 成本打平了 Opus 4.8 的 max 档分数；
- **GDPval-AA v2**：距 GPT-5.6 Sol 50 Elo 以内、成本低 13%,比 Fable 5 便宜 2.6×;
- **AA-Briefcase**：第二高分，约 Fable 5 半价。

<img src="/AIInfraGuide/images/kimi-k3-fig13-cost-efficiency.png" alt="分数与每任务成本对比" style="max-width: 75%; display: block; margin: 0 auto;" />

*图源：Kimi K3 技术报告 Figure 13(arXiv:2607.24653)*

Tom's Hardware 给的官方 API 定价可交叉核对：缓存命中输入 $0.30/M token、未命中 $3/M、输出 $15/M。

### 5.4 网络安全：一个容易忽略的长章节

报告 §6.2.2 自测（注意：Anthropic/OpenAI 的模型因拒绝此类任务未被纳入对比）：

- **Tier 1 漏洞发现**：在数十个广泛部署的系统（内核、数据库、Web 框架、区块链、VPN）中发现数百个候选漏洞，人工复核的发现中约 70% 被确认真实，含 16 个此前未知的漏洞，分布在 6 个项目；两个 Linux 内核例子：一个可远程触发的堆越界写（远程 DoS 原语）、一个 RDMA 子系统里“上游修复意外删掉权限检查”导致的本地提权。
- **Tier 2 漏洞利用**：36 个任务(16 个用户态 + 20 个内核态，K3 解出 14/36(38.9%),GLM-5.2 为 8/36(22.2%);14 个里有 10 个来自用户态，内核赛道四分之三未解。报告把差距归因于四个失败模式：最后一步利用链收尾难、缓解措施下策略选择差、陷入低效调试循环、提交前验证不足。
- **独立第三方**：英国 AISI 与 NIST CAISI 的联合评估与自测一致——ExploitBench 上 32% vs GLM-5.2 的 24%,但在 41 个任意代码执行任务上 0 成功，仍落后前沿网络能力模型。

### 5.5 案例研究：模型自己造编译器、设计芯片

报告第 7 节是“能力长什么样”的展示，摘几个量化的：

- **GPU kernel 优化**：独立沙箱、每任务 24 小时预算、Hopper GPU + 一台替代厂商 GPGPU 上，把 AttnRes 延迟从 283.6ms 压到 114.4ms,DSA 和 KDA 的运行时分别砍掉 55.1% 和 73.6%,MLA 达到峰值 TFLOPS 的一半以上；和 Fable 5（含 fallback）打平，明显超过 Opus 4.8 / GPT-5.6 Sol / GPT-5.5。报告还透露：开发后期“大部分内核优化工作已经是早期 K3 checkpoint 在做”。
- **MiniTriton 编译器**：K3 从零写了一个类 Triton 编译器（自研 tile 级 Python 前端、warp 级 MLIR 注解、PTX 代码生成），配套双模式张量库、反向自动微分、NCCL 分布式原语；在 L20 上其 tensor-core 矩阵乘在大 shape 下逼近 cuBLAS（约 90% 实测机器峰值），还能端到端训一个 GPT,梯度与 torch autograd 的差异在 $10^{-4}$ 量级。
- **芯片设计(nano-kpu)**：单次 48 小时自主运行，用开源 EDA 工具 + Nangate45 库设计了一个推理芯片原型：4mm² 面积内 100MHz 收敛时序、RTL 仿真解码吞吐 8,700+ token/s、1.46M 标准单元、INT4 MAC 阵列。代码已开源。
- **科研复现**：复现天体物理 I-Love-Q 关系——读了 20+ 篇论文交叉验证、评估 300+ 状态方程、写了 3,000+ 行 Python、产出交互式 HTML 仪表盘，约 2 小时完成，而人类专家通常要 1-2 周。

## 6. 工程与系统：让 2.8T 模型跑起来的基建

TDS 那篇解读的标题说得好：“一个前沿模型是怎么造出来的，读 Kimi K3 报告能学到——以及其中有多少不是模型本身”。报告第 5 节（约 10 页）是最像“AI Infra 教科书”的部分，三个挑战：混合注意力的系统协同设计、3T 级稀疏预训练、百万 token 的 Agentic RL。

### 6.1 KDA 的算法-系统协同设计

- **FlashKDA**：CUTLASS 写的分块 kernel,把“块内并行计算”和“跨块状态传播”重叠起来（计算和状态传递各自独立调度），显著超过 Triton 参考实现，同时服务训练和 prefill,并作为 flash-linear-attention 的后端被自动派发。
- **设备内上下文并行**：张量并行只切头数、不缩短递推链——超长序列 prefill 时每个 rank 只有几个头，SM 大量空闲。关键观察：每个片段的“状态转移”可以不依赖输入状态独立求值，事后精确组合。于是把序列在单个 rank 的 SM 间切分，并行算片段转移，再合并恢复每个片段的确切初始状态——零跨设备通信。
- **KCP（KDA 上下文并行）**：跨设备场景下，普通线性注意力可以“各算各的本地状态再求和”，但 KDA 不行——它的 delta rule 是用一个依赖 token 的矩阵 $M_t = I - \beta_t k_t k_t^\top \text{Diag}(\alpha_t)$ 乘输入状态，片段的效果取决于进入片段的状态。解法：每个 rank 算两个局部量——“片段对任意输入状态的累积转移矩阵”和“从零状态出发的本地状态”，然后用一次固定大小的 all-gather（全收集：把各卡持有的数据收集齐，让每张卡都拿到完整版） + 前缀扫描(prefix scan) 重建每个 rank 的真实输入状态。通信量恒定（不随序列增长），计算线性扩展。

### 6.2 3T 级预训练：MoonEP 与显存管理

**MoonEP**（已开源）：专家并行的老问题是 token 负载跨 rank 不均衡（计算歪斜 + 显存碎片）。MoonEP 用动态冗余专家达到完美均衡：证明每个 rank 预留 $E/R$ 个冗余专家槽位（E=专家数，R=EP 规模）就必然存在可行规划（§E 给出证明，且界是紧的）——对比 ECHO/UltraEP 预置冗余数或设 token 上限，超了就得停训。配套收益：完美均衡让每个 rank 恰好收到 $S \times K$ 个 token,计算形状静态已知，消除了逐层 MoE 的 host 同步；零拷贝通信（规划 kernel 预计算每个 token 的目的地，直接写到对端位置，最坏情况下通信缓冲从 $S \times K \times R$ 降到固定 $S \times K$）；专家 GEMM 用负载感知调度器（离线校准的分析代价模型选参数）。

显存侧（报告 5.2.2）三板斧：

- **统一激活管理器**：所有为 backward 保存的张量都挂一个可插拔存储后端，重计算/量化/offload/远程 offload 只是存储策略，可按张量粒度自由组合（FP8 块量化 + offload 为主，逐元素算子配重计算）；
- **MoE 梯度重写**：受 SonicMoE 启发，把 permuted probs 的梯度数学变形为只依赖中间激活和上游梯度，去掉对前向输出的依赖（多付一次轻量逐元素计算，省一份激活存储）；group GEMM 只存 dispatch 输入，backward 重算 dispatch,通信和计算重叠；
- **跨 PP rank 均衡 + Pipeline ZeRO-2**：1F1B 流水线下激活在 PP rank 间分布不均，用 Mooncake 传输引擎把激活远程 offload 到别的 rank 内存均衡；梯度跨 DP rank 分片并存进 CPU 内存（GPU 只留 double grad buffer）；Muon 的正交化需要完整参数矩阵，用 P2P 只取本地参数分片代替全量 all-gather,按 model-chunk 粒度流水线隐藏通信。

多模态侧：大图/长视频让视觉编码器成为关键路径，做法是动态 CP（单个大图按 patch 维度跨设备切分，gather-KV 算注意力，多个子 CP 组负载均衡分派大图）+ 把 ViT 的前后向**塞进流水线气泡**（1F1B 调度里天然存在的空闲时段）里执行，大部分编码器开销被隐藏。

### 6.3 百万 token 的 Agentic RL：状态是稀缺资源

1M 上下文的 RL 有一个独特矛盾：rollout 产生的 KV 缓存要留着跨迭代复用，而训练也要显存。报告的做法：

- **外部 KV 缓存池**：解码中的块留在 GPU,被逐出的可复用前缀写回 CPU DRAM（write-back 而非 write-through,避免冗余拷贝），用前再预取；KDA 状态和 MLA 缓存块的生命周期对齐。训练状态（权重、优化器）在迭代间隙 offload 到 NVMe,给缓存池腾 DRAM。
- **自动节流调度器**：固定并发要么早期浪费、后期爆缓存，改为用实时信号（活跃请求数、队列长度、KV 利用率）动态控制并发。
- **梯度缓冲复用**：参考模型等“只前向”模型权重太大放不下 GPU,就驻留 CPU、按块物化，参数张量直接借用策略模型的 FP32 梯度缓冲——“复用会被覆写的内存”换来零新增分配。
- **AgentENV 沙箱**（已开源）：微虚拟机沙箱，基于 Firecracker。增量 checkpoint 只存脏页，checkpoint 133ms、resume 49ms;三高层操作：Pause/Resume（沙箱等待模型推理时可占沙箱生命周期的 98%,暂停后零资源占用）、Fork（从精确状态克隆新沙箱做奖励判定）、Snapshot（定期快照容错）；OverlayBD 镜像 + 自研 ublk 驱动 + P2P 传输做到大规模亚秒级启动，COW 内存让真实负载内存超卖比达 6.5×。整个训练和评测周期共创建了 51,219,741 个沙箱，横跨 1,505,678 个镜像——这个数字本身就在说明“环境工程”的分量。

### 6.4 推理与服务：混合缓存的联合管理

生产侧最难的是 KDA 和 MLA **两种性质完全不同的缓存**要联合管理：MLA 的 KV 随序列增长、按 token 分页；KDA 状态固定大小、每请求一份，只在稀疏边界存 checkpoint。naive 的块哈希前缀缓存（块大小被 KDA checkpoint 逼到 1024–6144 token）会让短请求永远不可复用。解法是解耦两种粒度：

- 前缀哈希跑在细粒度 hash 块(512 token)上，物理块保持粗分配单位；KDA checkpoint 只存在 MLA hash 端点（查找可能引用的唯一位置）；
- 命中判定分两段：MLA 阶段先按链式哈希匹配完整物理块，缺失处回退到块内端点；KDA 阶段要求候选边界在每个 KDA 组都有 checkpoint——**命中点可以是 512 的任意倍数，不必是物理块倍数**（图 12 的例子在 6144-token 物理块内的 B=2560 处命中，免算 [0, B） 整个前缀);
- 并发一致性三规则：命中块跨所有组先钉住再分配；新注册块要等拷贝落地才可被匹配；一个组的 checkpoint 被逐出则跨组原子失效。

解码侧：KDA 的 in-place 状态更新和投机解码冲突（草案被拒时状态已前滚，无法回滚）。解法：**只缓存草案 token 的投影输入（远小于状态），验收后现场重建状态**,被接受 token 和 bonus token 的状态写回——与同期工作 ReplaySSM 独立同构思。Stable LatentMoE 的 kernel：潜在下投影与路由器融合成单个 GEMM、权重矩阵跨 rank 分片并把 all-gather 融进 GEMM epilogue（multimem 指令）、小 batch 下用 WarpDecode 式的 token 中心 kernel（warp 负责一个输出神经元直接流式读权重，再按 lane team 细分专家并行）。

**集群级调度**两个策略，值得每个做服务的工程师抄作业：

- **缓存亲和调度**：1M 上下文下典型编码会话是“400K token 前缀 + 4K token 新内容”，命中比未命中便宜数个数量级（解读口径，报告原文为 orders of magnitude）；请求路由到持有其前缀缓存的集群，同时用一致哈希给每个会话钉两个集群（主 + 备），单集群故障时重 prefill 的负担被均匀摊到全网；
- **预算准入控制**：线上流量从 2K 到 1M token 横跨三个数量级，“平均请求”式的容量规划全部失效；给每个请求类别独立资源预算，长上下文洪峰只能饿死自己，不能拖垮短请求的 TTFT（首 token 延迟）。

## 7. 局限与开源意义

### 7.1 报告自己承认的边界

- **整体仍落后最强闭源模型**（Claude Fable 5、GPT-5.6 Sol），报告摘要和结论都明确写了；评测里最刺眼的短板是研究级推理（CritPt 23.4、HLE-Full 带工具 56.0）和 Elo 式知识工作套件（GDPval-AA v2 第三、AA-Briefcase 第二）；
- **网络安全**：Tier 2 内核利用赛道四分之三未解，外部评估 41 个任意代码执行任务 0 成功；
- **评测口径**：部分分数引自第三方榜单、各模型 harness 不同、PostTrainBench 跑在 H20 而非官方 H100、SWE-Marathon 用的是 H20 校准分支，Claude Fable 5 在 35% 的任务上触发 fallback——这些限定词本身就是“报告写了什么”的一部分;
- **报告未公开**：训练算力、成本、数据规模与配比、以及 2.5× 中架构/数据/训练各自的贡献占比（报告原文只说"collectively"）。

### 7.2 开源意义与外部视角

- **生态意义**：第一个开放的 3T 级权重，且不只是权重——FlashKDA、MoonEP、AgentENV、MiniTriton、nano-kpu、EAGLE-3 草稿全部开源；Tom's Hardware 报道中 Bank of America 分析师的观点是：它证明“大规模预训练 + 架构工作仍能给受算力约束的中国旗舰模型带来阶跃式提升”。
- **Tom's Hardware 的提醒**：权重在 7 月 27 日公开前，所有 K3 数字都是 Moonshot 自报，无法独立验证；Anthropic 今年 2 月曾指控 Moonshot 用 340 万次 Claude 对话做蒸馏训练。这些是商业与信任层面的事实，与报告技术内容分开看。
- **TDS 的视角**：报告最有价值的不是架构（架构是一堆已发表想法的堆叠），而是环境、奖励机制、基础设施和数据管道——那些才是真正难做、真正构成壁垒的部分。K3 的 serving 章节对 3T 以下规模同样适用：前缀缓存、缓存亲和路由、预算准入控制，都是任何 busy service 的通用工具，只是这里的“每请求状态”大得离谱。

📌 **总结**：Kimi K3 = 2.8T 总参数 / 104B 激活的 MoE,用 KDA 混合注意力把 1M 上下文变得付得起，用 Stable LatentMoE 把 896 专家的极端稀疏度稳住，用三域三档 RL + 蒸馏把推理强度变成可调旋钮，再靠一整层为“状态复用”而生的基础设施把这一切跑起来——而它没能回答的（训练到底花了多少钱、多少 token），恰好是其他实验室同样守口如瓶的部分。

## 📝 总结

1. **规模账**：2.8T 总参数 / 104B 激活 / 896 专家激活 16 个 / 1M 上下文，稀疏度 56——MoE 省算力不省显存，这是“最大开源”的第一层含义（呼应站内 [6.4 数据并行与专家并行](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/64-数据并行与专家并行)与[第 10 章 MoE 并行](https://xiayihann.github.io/AIInfraGuide/distributed/模块三-分布式训练/第10章-moe并行)）。
2. **架构账**：KDA 用“固定大小状态 + 逐通道遗忘”把注意力从 $O(n)$ 缓存变成恒定状态，带下界衰减让全 Tensor Core 分块计算可行；AttnRes 让每层直接读前面所有层（Block 版把内存从 $O(Ld)$ 降到 $O(Nd)$）；Stable LatentMoE 用 RMSNorm + SiTU-GLU + Quantile Balancing 三个补丁稳住 896 专家的极端稀疏。
3. **训练账**：缩放定律显示合计 2.5× 效率提升（报告拟合口径；TDS 解读为“同等质量不到一半算力”）；cosine vs WSD 要各自独立调参才公平；上下文 8K→64K→256K→1M 四阶段渐进扩展；算力与成本报告未公开。
4. **后训练账**：3 域 × 3 强度 = 9 个专家模型分别 RL,再用 MOPD 逐 token 奖励蒸馏回一个；token 预算控制把“思考强度”变成可训练、可指挥的旋钮；MXFP4 QAT 和 EAGLE-3 草稿让训练目标直接对齐部署成本。
5. **评测账**：Agentic 与编程拿下一批第一（BrowseComp 91.2、MCPMark 94.5、SWE-Marathon 42.0），GPQA 93.5 贴近前沿，研究级推理（CritPt 23.4、HLE 56.0）是明确短板；第三方口径：AA 智能指数第三、WebDev Arena 登顶、Vals 第二；成本上 BrowseComp 每任务 $2.03 拿下最高分。
6. **基建账**：FlashKDA / KCP 让递推注意力跑得起来， MoonEP 的 $E/R$ 冗余专家界让 3T 预训练永不因负载失衡中断，AgentENV 的 133ms checkpoint 微虚拟机支撑 5,100 万个沙箱，混合前缀缓存让 1M 上下文 serving 的前缀复用粒度回到 512 token。

## 🎯 自我检验清单

- 能解释“总参数 2.8T、激活参数 104B”的含义，并说明 MoE 省的是算力还是显存（答：算力；2.8T 权重仍需全部驻留高速存储）。
- 能算出 896 专家激活 16 个的稀疏度（56,即每 token 只经过 1.8% 的专家池）。
- 能说出 KDA 状态更新公式 $S_t = (I - \beta_t k_t k_t^\top \text{Diag}(\alpha_t))S_{t-1} + \beta_t k_t v_t^\top$ 里每个符号的含义，以及 $\alpha_t$ 与 $\beta_t$ 分别控制什么（逐通道遗忘 / 写入强度）。
- 能解释为什么“带下界衰减($g_{\min}=-5$)”能让 KDA 的分块计算全部走 Tensor Core（倒数不超过 $e^{80}$,落在 BF16 动态范围）。
- 能说明混合注意力 3:1 的分工：KDA 提供位置敏感、recency 感知的混合，MLA 提供无位置编码的全局精确召回，以及 NoPE 为什么让 1M 扩展不需要改位置编码。
- 能对比 AttnRes 与标准残差连接的差别，并说出 Block AttnRes 牺牲了什么（全量 $O(Ld)$ 内存/通信 → Block 版 $O(Nd)$,块间注意力粒度变粗）换取了什么。
- 能解释 Quantile Balancing 与辅助损失负载均衡的本质区别（偏置只改派发、不进混合权重与路由器梯度；用一次 Top-(k+1) 的裕量分位数定偏置）。
- 能说出后训练 3×3 矩阵的含义（通用/Agent/编程 × low/high/max = 9 个专家）和 MOPD 蒸馏奖励的直觉（老师比学生更看好该 token 的程度，截断防极端）。
- 能解释 reasoning effort 预算控制如何工作（冷启动估计 $b_0(x)$,超 $\tau \cdot b_0(x)$ 判负，$\tau$ 退火出三档）。
- 能解释 KCP 为什么不能像普通线性注意力那样“本地状态直接求和”（KDA 的 delta rule 用 token 相关矩阵乘输入状态，片段效果依赖输入状态，所以要分解成转移矩阵 + 零状态本地片段，再做前缀扫描）。
- 能说出 MoonEP 的平衡保证：每 rank 预留 $E/R$ 个冗余专家槽位即存在可行规划，换来静态计算形状与零拷贝通信。
- 能说出报告评测的至少三个口径限定（所有 K3 分数为 effort=max、部分对手分数引自第三方榜单且 harness 不同、Fable 5 含 fallback / GPT-5.6 Sol 含 cyberguard)。
- 能列出报告未公开的三类信息（总训练 token / GPU 小时与成本、数据配比、2.5× 的内部归因），并知道哪些数字来自报告、哪些来自第三方、哪些是估算。

## 📚 参考资料

**技术报告与官方**

- [Kimi K3: Open Frontier Intelligence(arXiv 2607.24653)](https://arxiv.org/abs/2607.24653)：本文解读的 47 页技术报告，Kimi Team / Moonshot AI,2026-07-27;所有“报告”口径数字的来源。
- [moonshotai/Kimi-K3(GitHub)](https://github.com/moonshotai/Kimi-K3)：官方仓库，含完整模型卡（架构表格、评测表格、MXFP4 量化、API 用法）。
- [Kimi K3 技术博客](https://www.kimi.com/blog/kimi-k3)：Moonshot 官方发布文。
- [MoonEP](https://github.com/MoonshotAI/MoonEP) / [FlashKDA](https://github.com/MoonshotAI/FlashKDA) / [AgentENV](https://github.com/kvcache-ai/AgentENV) / [MiniTriton](https://github.com/MoonshotAI/minitriton) / [nano-kpu](https://github.com/MoonshotAI/nano-kpu)：报告配套开源的基础设施与产物。

**第三方解读与报道**

- [How a Frontier Model Gets Built, Read from the Kimi K3 Report(Towards Data Science)](https://towardsdatascience.com/how-a-frontier-model-gets-built-read-from-the-kimi-k3-report/)：“环境、奖励机制、基础设施才是壁垒”视角的深度解读；2.5× = “不到一半训练算力”的说法出自此文。
- [China's 2.8-trillion-parameter Kimi K3... (Tom's Hardware)](https://www.tomshardware.com/tech-industry/artificial-intelligence/moonshot-releases-2-8-trillion-parameter-kimi-k3)：Frontend Code Arena #1（1679 分）、API 定价（$0.30/$3/$15 每 M token）、H200 + 替代厂商 GPGPU 训练侧写、64+ 加速器部署建议、蒸馏指控背景。
- [Kimi K3 网络安全能力初步评估(UK AISI + NIST CAISI)](https://www.aisi.gov.uk/blog/preliminary-assessment-of-kimi-k3s-cyber-capabilities)：报告引用的独立网络能力评估。

**站内相关**

- [6.4 数据并行与专家并行(站内)](/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/64-数据并行与专家并行)：MoE 推理的显存账本与 EP 通信量推导，理解 K3 的 64 加速器部署的前置。
- [第 10 章 MoE 并行(站内)](/AIInfraGuide/distributed/模块三-分布式训练/第10章-moe并行)：Router、All-to-All、负载均衡的系统讲解，可与 MoonEP 的平衡方案对照。
- [6.1 推理并行策略总览(站内)](/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/61-推理并行策略总览)：PP/TP/EP/CP 全景，理解报告第 5 节并行组合的坐标系。
