---
title: "DSpark 精读:用半自回归草稿和置信度调度重画推理 Pareto 前沿"
description: "拆解 DeepSeek 与北大提出的 DSpark:并行骨干加轻量 Markov Head 如何缓解草稿后缀衰减,置信度与硬件吞吐曲线如何按请求分配验证预算,以及 DeepSeek-V4 线上匹配吞吐下单用户生成速度提升 60%–85% 的真实口径。"
pubDate: 2026-08-11
originalUrl: "https://arxiv.org/abs/2607.05147"
sourceType: "paper"
originalAuthor: "Xin Cheng et al. (Peking University & DeepSeek-AI)"
tags: ["DSpark", "Speculative Decoding", "DeepSeek-V4", "SGLang", "推理调度", "LLM Serving"]
---

> 原文:[DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation](https://arxiv.org/abs/2607.05147)(Xin Cheng et al.,北京大学与 DeepSeek-AI,arXiv 2607.05147 v1,2026-07)

投机解码过去常被概括成“让小模型先猜、大模型再验”，DSpark 把这个故事向前推进了两步：**草稿不能只追求一次猜得多，还要让长草稿前后连贯；验证也不能固定长度，而要把有限的 GPU 批容量分给最可能被接受的 token。** 论文在 Qwen3-4B/8B/14B 的离线实验中，相对 DFlash 将宏平均接受长度提高 **16.3%/18.4%/18.3%**；部署到 DeepSeek-V4 线上流量后，在匹配系统吞吐的口径下，V4-Flash 单用户生成速度提高 **60%–85%**，V4-Pro 提高 **57%–78%**。但这不是“所有模型无条件快 85%”：收益取决于草稿质量、请求类型、并发负载、硬件吞吐曲线和服务引擎能否真正执行变长验证。

<!-- more -->

## 📑 目录

- [🗺️ 原文阅读地图](#️-原文阅读地图)
- [0. 读前 3 分钟:先把三笔账算清](#0-读前-3-分钟先把三笔账算清)
- [1. DSpark 要解决的两个失败模式](#1-dspark-要解决的两个失败模式)
- [2. 总体架构:起草、排预算、再验证](#2-总体架构起草排预算再验证)
- [3. 半自回归草稿:并行骨干加一点顺序依赖](#3-半自回归草稿并行骨干加一点顺序依赖)
- [4. 置信度头:预测一段前缀能活多久](#4-置信度头预测一段前缀能活多久)
- [5. 硬件感知前缀调度:把验证预算花在刀刃上](#5-硬件感知前缀调度把验证预算花在刀刃上)
- [6. 训练目标与开源复现账本](#6-训练目标与开源复现账本)
- [7. 离线实验:接受长度为何真的提高](#7-离线实验接受长度为何真的提高)
- [8. 线上部署:真正改变的是吞吐与交互性的前沿](#8-线上部署真正改变的是吞吐与交互性的前沿)
- [9. 从论文到当前工程:SGLang 如何把调度收益落到 GPU](#9-从论文到当前工程sglang-如何把调度收益落到-gpu)
- [10. 权衡、局限与选型边界](#10-权衡局限与选型边界)
- [📝 总结](#-总结)
- [🎯 自我检验清单](#-自我检验清单)
- [📚 参考资料](#-参考资料)

## 🗺️ 原文阅读地图

这篇论文同时讲算法、训练和生产系统。本文选择性精讲如下，避免把一篇中文解读误当成原文逐段翻译。

| 原文单元 | 处理深度 | 本文位置与理由 | 来源锚点 |
| --- | --- | --- | --- |
| 标准投机解码与延迟公式 | 精讲 | 第 0 节，所有收益都要回到这三笔账 | 论文 §2.1、Eq. 1 |
| DFlash 的 KV Injection | 简述 | 第 1、3 节，只作为并行骨干前置，不展开投影细节 | 论文 §2.2、Eq. 2–3 |
| 半自回归 Markov/RNN Head | 精讲 | 第 3 节，解释“并行起点高、后缀仍连贯” | 论文 §3.1、Eq. 4–6、Figure 2–4 |
| Confidence Head 与 STS 校准 | 精讲 | 第 4 节，调度器必须拿到可解释的生存概率 | 论文 §3.2.1、Eq. 7–8、Figure 5–6 |
| Hardware-Aware Prefix Scheduler | 精讲 | 第 5 节，DSpark 从算法走向 serving 的关键 | 论文 §3.2.2、Algorithm 1、Appendix A |
| 三项训练损失 | 简述 | 第 6 节，保留目标与权重，不逐行推导优化器 | 论文 §3.3、Eq. 9–12 |
| 离线主表与消融 | 精讲 | 第 7 节，只保留改变结论的设置和数字 | 论文 §4、Table 1、Figure 2/4/5 |
| V4 内部训练基础设施 | 简述 | 第 8 节，隐藏状态通信与 anchor packing 各讲结论 | 论文 §5.1 |
| 线上流量与调度预算 | 精讲 | 第 8 节，严格区分匹配吞吐、中等 SLA 和极端 SLA | 论文 §5.2–5.4、Figure 7–8 |
| Related Work 全量谱系 | 不展开 | 只做定位，不影响本文三条机制主线 | 论文 §6 |
| 完整参考文献与全部模型配置 | 不展开 | 读者可从论文和 DeepSpec 追溯，避免淹没主线 | 论文 References、DeepSpec |

📌 **本文承诺**：读完后，你应该能手算一段草稿的前缀生存概率，解释为什么 Markov Head 能缓解后缀衰减，写出吞吐目标 $\Theta=\tau\cdot\mathrm{SPS}(B)$，并说明调度器为何必须满足 non-anticipating（不偷看未来 token）条件。

## 0. 读前 3 分钟:先把三笔账算清

### 0.1 投机解码到底省了什么

自回归大模型一次 forward 只生成一个 token。投机解码（Speculative Decoding）则让草稿模型 $M_d$ 先提出 $\gamma$ 个候选 $x_1,\ldots,x_\gamma$，目标模型 $M_t$ 一次并行验证整段候选。

对第 $k$ 个候选，标准 rejection sampling（拒绝采样）的接受概率是：

$$
P(\text{accept }x_k)=\min\!\left(1,\frac{p_k^t(x_k)}{p_k^d(x_k)}\right)
$$

其中 $p_k^t$、$p_k^d$ 分别是目标模型和草稿模型在第 $k$ 个位置给出的概率分布。验证从左到右进行：**第一个 token 被拒绝后，后面的候选全部作废**；目标模型从修正分布采一个 token，完成本轮。这个规则保证最终输出仍服从目标模型分布，因此论文把它称为 lossless（无质量损失、分布不变）的加速。

> 打个比方：草稿模型像实习生，一次写出一行；目标模型像主编，一次审完整行。前两个词正确、第三个词错误时，主编只能留下前两个词并改掉第三个，第四个词不能继续用——因为它建立在错误第三词的上下文上。这里“实习生”对应 draft model，“主编”对应 target model，“从首个错误截断”对应前缀验收。

### 0.2 一条公式看穿所有加速手段

论文把平均每生成一个 token 的延迟写成：

$$
\underbrace{L}_{\text{每 token 平均延迟}}
=
\frac{
\underbrace{T_{\text{draft}}}_{\text{起草耗时}}
+
\underbrace{T_{\text{verify}}}_{\text{目标模型验证耗时}}
}{
\underbrace{\tau}_{\text{每轮接受 token 数}}
}
$$

这条式子只有三个杠杆：

1. **Draft faster**：降低 $T_{\text{draft}}$，草稿更快；
2. **Draft better**：提高 $\tau$，每轮多接受几个 token；
3. **Verify smarter**：减少无效的 $T_{\text{verify}}$，别验证注定会被拒绝的后缀。

Eagle3 一类自回归 drafter 主要追求第二点；DFlash 一类并行 drafter 主要追求第一点；**DSpark 的目标是同时抓住三个杠杆。**

### 0.3 接受长度不等于加速比

假设一轮起草加验证共花 10 ms，平均提交 $\tau=5$ 个 token，则平均延迟是 2 ms/token。若接受长度涨到 6，但验证更长、总耗时涨到 15 ms，结果反而是 2.5 ms/token——**接受长度更高，端到端却更慢。**

这也是读 DSpark 时最重要的口径纪律：

- 离线实验的 accepted length（接受长度）衡量 drafter 质量；
- 线上实验的 tok/s/user 衡量单用户速度；
- aggregate throughput 衡量整套系统服务多少 token；
- 三者相关，但不能互相冒充。

## 1. DSpark 要解决的两个失败模式

### 1.1 并行草稿猜得快，却容易“串台”

**为什么不让并行 drafter 一口气猜 16 个 token？** 因为每个位置在同一次 forward 中独立预测，看不到本轮其他位置最终采样了什么。

假设上下文后面存在两种自然续写：

- `of course`
- `no problem`

独立预测的第一个位置可能采到 `of`，第二个位置却仍在“course / problem”两种模式之间平均，最后拼成 `of problem`。论文把这种现象联系到并行生成中的 **multi-modal collision（多模态碰撞）**：每个位置单独都像合理答案，组合起来却不连贯。

DFlash 的优势是草稿 backbone 只跑一次，$T_{\text{draft}}$ 几乎不随 $\gamma$ 线性增长；缺点是后面的 token 不知道前面实际选了什么，conditional acceptance（条件接受率）沿 block 后缀快速下降。

### 1.2 草稿生成得长，不代表都值得验证

第二个问题发生在目标模型侧。假设一个 batch 有 $B$ 个请求，每个请求都验证 $K$ 个草稿 token，目标模型这一步需要处理大约 $B\times K$ 个验证位置。

- **低并发**：GPU 还没吃满，多验证几个位置可能几乎免费；
- **高并发**：GPU 批容量已经紧张，一个大概率被拒绝的 token 也会占掉真实算力，挤走其他请求。

而且请求本身有差异：代码和数学通常结构更强、容易猜；开放式聊天熵更高、后缀更容易错。固定长度验证等于给所有请求发同样预算，既不看“这题难不难”，也不看“GPU 现在忙不忙”。

**DSpark 的两条主线因此一一对应：**

| 失败模式 | DSpark 组件 | 直接优化的量 |
| --- | --- | --- |
| 并行草稿缺块内依赖，后缀衰减 | Parallel Backbone + 轻量 Sequential Head | 提高 $\tau$，少量增加 $T_{\text{draft}}$ |
| 固定长度验证浪费批容量 | Confidence Head + Hardware-Aware Prefix Scheduler | 降低有效 $T_{\text{verify}}$ |

## 2. 总体架构:起草、排预算、再验证

<img src="/AIInfraGuide/images/dspark-fig1-architecture.png" alt="DSpark 一轮解码流程：目标模型生成锚点 D，并行骨干与轻量串行头起草 EFGH 及置信度，硬件感知前缀调度丢弃低置信 H，目标模型并行校验并接受 EF、拒绝 G 后产出纠正 token G 星" style="max-width: 88%; display: block; margin: 0 auto;" />

*图源:DSpark 论文 Figure 1(arXiv:2607.05147)*

沿着图走一轮：

1. **目标模型先产生 anchor**：已有 prompt `ABC`，目标模型生成 token `D`；
2. **半自回归起草**：并行 backbone 一次产出整块隐藏状态和 base logits，轻量 sequential head 从左到右采出 `E F G H`；
3. **同时预测置信度**：每个位置得到 $c_1,\ldots,c_4$；
4. **前缀调度**：调度器根据累计生存概率和当前硬件吞吐曲线留下 `E F G`，丢弃低收益的 `H`；
5. **目标模型并行验证**：`E`、`F` 被接受，`G` 被拒绝，目标模型给出修正 token $G^*$；
6. **进入下一轮**：新合法前缀变成 `ABC D E F G*`。

把它写成机制卡：

| 项目 | 内容 |
| --- | --- |
| 输入 | 当前合法上下文、目标模型上一轮产生的 anchor、每个请求的系统状态 |
| 可变状态 | 已采样 draft prefix、每位置 confidence、引擎 $\mathrm{SPS}(B)$ 成本表 |
| 输出 | 每请求不同的验证长度 $\ell_r$、目标模型无偏提交的新 token |
| 成功条件 | sequential head 足够轻；confidence 已校准；引擎能执行变长验证 |
| 失败条件 | 草稿普遍难猜；成本表失真；变长请求仍被 padding 回固定宽度 |

## 3. 半自回归草稿:并行骨干加一点顺序依赖

### 3.1 重活并行做，轻活串行做

DSpark 先让 DFlash 风格的 parallel backbone 对整块跑一次重计算，产生每个位置的隐藏状态 $h_k$ 和基础 logits $U_k$。然后再加一个非常轻的 sequential block，对第 $k$ 个位置补一个依赖已采样前缀的偏置 $B_k$：

$$
P(X\mid x_0)=\prod_{k=1}^{\gamma}p_k(x_k\mid x_0,x_{<k})
$$

$$
p_k(v\mid x_0,x_{<k})=
\frac{\exp\bigl(U_k(v)+B_k(x_0,x_{<k},v)\bigr)}
{\sum_{u\in\mathcal V}\exp\bigl(U_k(u)+B_k(x_0,x_{<k},u)\bigr)}
$$

逐项解释：

- $x_0$：上一轮目标模型产生的 anchor；
- $U_k(v)$：并行 backbone 对词表 token $v$ 给出的基础分；
- $B_k$：根据已经采样的块内前缀追加的顺序偏置；
- $\mathcal V$：词表；
- $p_k$：最终仍是一个普通 softmax 概率，因此可直接进入标准 rejection sampling。

关键约束是：

$$
T_{\text{sequential}}\ll T_{\text{parallel}}
$$

否则为了补顺序依赖又串行跑一个大模型，就退回了自回归 drafter 的老问题。

### 3.2 Markov Head:只看上一个 token，就能挡住大量串台

默认方案把偏置限制为一阶转移，只看前一个 token：

$$
B(x_{k-1},\cdot)=W_1[x_{k-1}]W_2\in\mathbb R^{|\mathcal V|}
$$

完整的词表转移矩阵本应是 $|\mathcal V|\times|\mathcal V|$，太大；论文用低秩分解 $B=W_1W_2$，默认 rank $r=256$：

- $W_1[x_{k-1}]$ 像查一行 token transition embedding；
- $W_2$ 把这个低维向量投回完整词表 logits；
- 若前一个 token 是 `of`，偏置可以提升 `course`、压低 `problem`。

这不是让 Markov Head 独自生成文本，而是让它在深并行 backbone 的 $U_k$ 上做小幅、条件化修正。**backbone 提供容量，Markov Head 提供块内因果性。**

论文还研究了 RNN Head：它维护递归状态，能看见完整 block prefix，而不只看上一个 token。结果是 RNN 在长 block 上略好，但收益有限、实现更复杂，因此生产和开源配置默认采用 Markov Head（论文 §4.3.2）。

### 3.3 为什么“只有一点自回归”就够了

<img src="/AIInfraGuide/images/dspark-fig2-position-acceptance.png" alt="Qwen3-4B 上 Math、Code、Chat 各 draft 位置的条件接受率：DFlash 起点高但后缀下滑，Eagle3 后期回升，DSpark 全程更高更稳" style="max-width: 95%; display: block; margin: 0 auto;" />

*图源:DSpark 论文 Figure 2(arXiv:2607.05147)*

Figure 2 不是普通的“前缀存活率”，而是**位置条件接受率**：只有前面位置全部已接受的样本，才进入第 $k$ 个位置的分母。它把“前面先错了”的惩罚剥掉，直接观察某个位置自身的预测质量。

论文给出三个反直觉观察：

1. **第一个位置，深并行模型更强。** Math 上 DFlash 约 0.88、Eagle3 约 0.81；Chat 上约 0.72 对 0.53。因为并行模型只跑一次，可以承担 5 层 backbone；自回归 drafter 每个位置都要跑，只能做得更浅。
2. **后缀位置，自回归更会利用已选前缀。** Eagle3 在 Chat 上从约 0.53 升到 0.74；DFlash 从约 0.72 降到 0.63。
3. **DSpark 把两条曲线的优点叠起来。** 它继承并行 backbone 的高起点，同时用 sequential head 稳住后缀。

<img src="/AIInfraGuide/images/dspark-fig4-block-size.png" alt="不同 draft 长度下 Math、Code、Chat 的接受长度与整轮时延：DSpark 的 Markov 和 RNN 方案持续优于 DFlash，优势随长度扩大，串行头额外时延约 0.2% 到 1.3%" style="max-width: 98%; display: block; margin: 0 auto;" />

*图源:DSpark 论文 Figure 4(arXiv:2607.05147)*

当 proposal length 从 $\gamma=7$ 增至 $15$ 时，DSpark 相对 DFlash 的接受长度提升从 Math/Code/Chat 的 **16%/15%/18%** 扩大到 **30%/26%/22%**。与此同时，在 batch size 128、上下文长度取 512/1024/2048/4096 平均的实验里，把总 draft 长度从 4 拉到 16，相对 DFlash 的整轮时延只增加 **0.2%–1.3%**。

⚠️ **注意口径**：这里的“串行开销很小”发生在论文给定的 batch、目标模型和引擎设置中，且目标模型验证占主导。不能据此断言任意小模型、batch=1 或任意 kernel 下 Markov loop 都可忽略。

## 4. 置信度头:预测一段前缀能活多久

### 4.1 它预测的不是“这个 token 看起来像不像”

Confidence Head 对每个位置输出 $c_k\in(0,1)$：

$$
c_k=\sigma\!\left(w^\top[h_k;W_1[x_{k-1}]]\right)
$$

它的语义是：**假设前面 token 都已经被目标模型接受，第 $k$ 个 draft token 继续被接受的条件概率。** 训练标签来自目标分布和草稿分布的总变差距离（Total Variation Distance）：

$$
c_k^*=1-\frac12\left\|p_k^d-p_k^t\right\|_1
$$

这不是拍脑袋的置信分。标准投机采样下，$1-\frac12\|p^d-p^t\|_1$ 正好对应该位置的期望接受概率。Draft 与 target 分布越接近，TV 距离越小，$c_k^*$ 越接近 1。

### 4.2 调度需要的是前缀生存概率

因为第一个拒绝会截断整个后缀，第 $j$ 个 token 真正有价值的概率不是单独的 $c_j$，而是前 $j$ 个条件概率的连乘：

$$
a_j=\prod_{i=1}^{j}c_i
$$

假设某请求的条件置信度是：

$$
[c_1,c_2,c_3,c_4]=[0.90,0.80,0.50,0.20]
$$

那么各前缀位置的生存概率是：

$$
[a_1,a_2,a_3,a_4]=[0.90,0.72,0.36,0.072]
$$

第四个 token 自己的 confidence 是 0.20，但它被验证后真正贡献接受 token 的概率只有 **7.2%**，因为前面三关也必须全部通过。这解释了为什么长草稿的尾部很容易成为 verification waste。

### 4.3 为什么还要做 STS 校准

一个 confidence 模型可以很会“排序”，却不一定报得准。例如它能正确判断 A 比 B 更容易接受，但把真实 70% 报成 90%。静态阈值只看排序时问题不大；硬件调度要计算期望吞吐，概率绝对值错了就会分错预算。

论文使用 Sequential Temperature Scaling（STS，顺序温度缩放）在 held-out validation set 上从左到右校准累计生存概率。它保持置信排序不变，只修正概率尺度。论文报告：

- 原始 confidence 的 ROC-AUC 为 **0.81–0.90**；
- 原始 ECE（Expected Calibration Error）为 **3%–8%**，整体偏自信；
- STS 后平均 ECE 降到约 **1%**。

<img src="/AIInfraGuide/images/dspark-fig5-confidence-threshold.png" alt="置信度阈值扫描：提高阈值会剪掉将被拒绝的后缀 token，接受率上升；Chat 场景剪枝最强，接受率由约 46% 升至约 96%" style="max-width: 95%; display: block; margin: 0 auto;" />

*图源:DSpark 论文 Figure 5(arXiv:2607.05147)*

Figure 5 是离线静态阈值诊断，不是最终硬件调度器。阈值提高后，Chat 接受率从 **45.7%** 升到 **95.7%**，Math 从 **76.9%** 升到 **92.5%**，Code 从 **67.6%** 升到 **92.0%**。接受率升高的同时，每步保留 token 数也在下降——**高接受率本身不是免费收益，它来自主动少验一些 token。** 最终要不要剪，仍要交给下一节的系统吞吐目标判断。

## 5. 硬件感知前缀调度:把验证预算花在刀刃上

### 5.1 把问题写成吞吐最大化

设一个 batch 有 $R$ 个请求，第 $r$ 个请求选择验证 $\ell_r$ 个 draft token。目标模型实际处理的 token batch size 是：

$$
B=\sum_{r=1}^{R}(1+\ell_r)
$$

每个请求里的 `1` 表示目标模型至少会贡献的 bonus/corrected token。期望提交 token 总数是：

$$
\tau=\sum_{r=1}^{R}\left(1+\sum_{j=1}^{\ell_r}a_{r,j}\right)
$$

引擎在 token batch size 为 $B$ 时，每秒能跑多少 decode step，记作 $\mathrm{SPS}(B)$（Steps Per Second）。这条硬件容量曲线在引擎初始化时 profiling 一次，保存为小型 cost table。调度目标是：

$$
\underbrace{\Theta}_{\text{期望系统 token 吞吐}}
=
\underbrace{\tau}_{\text{每 step 期望提交 token}}
\cdot
\underbrace{\mathrm{SPS}(B)}_{\text{每秒可跑 step}}
$$

这条式子把“草稿质量”和“硬件代价”接到一起：多加入一个候选，收益是它的前缀生存概率 $a_{r,j}$，代价是 $B$ 增大后 $\mathrm{SPS}(B)$ 可能下降。

### 5.2 为什么可以贪心分预算

对同一个请求，$a_{r,j}=\prod_{i\le j}c_{r,i}$ 必然单调不增。调度器把所有请求的候选扩展 $(r,j)$ 按 $a_{r,j}$ 从高到低放进一个池子，再依次尝试加入：

```python
# 论文 Algorithm 1 的简化示意，不是逐字源码
for request in active_requests:
    survival[request] = cumulative_product(confidence[request])

candidates = sort_all_prefix_extensions_by_survival_desc(survival)
best = requests * SPS(requests)  # 每请求只拿目标模型的基础 token

for request, position in candidates:
    extend_prefix(request, position)
    token_batch_size += 1
    expected_tokens += survival[request][position]
    throughput = expected_tokens * SPS(token_batch_size)

    if throughput > best:
        save_current_plan()
    else:
        break  # 理论算法的因果早停
```

固定总 $B$ 时，优先选 $a_{r,j}$ 最大的 token 就能最大化期望接受数；沿着这个 greedy admission path 再查 $\mathrm{SPS}(B)$，即可寻找最优预算。

**这与统一阈值有什么区别？** 同一个 batch 里，数学请求可能拿到 6 个验证位置，开放聊天可能只拿 2 个；同一请求在 GPU 空闲时可以多拿预算，在高并发时则缩短。调度单位从“整个 batch 一个 $K$”变成“每个请求一个 $\ell_r$”。

### 5.3 为什么调度器不能偷看未来 token

这篇论文最容易被跳过、却最重要的理论边界在 Appendix A：**是否验证第 $k$ 个 token 的决定，不能依赖这个 token 实际采成了什么。** 否则调度器会偏爱“后续看起来更顺”的 token，改变目标分布。

论文给出一个两 token 反例。设：

$$
a_1=0.8,\qquad \mathrm{SPS}(1)=1.0,\quad \mathrm{SPS}(2)=0.5,\quad \mathrm{SPS}(3)=0.45
$$

验证 0 或 1 个 draft token 的期望吞吐分别是：

$$
\Theta_0=1\times1.0=1.0
$$

$$
\Theta_1=(1+0.8)\times0.5=0.9
$$

按因果 early stop，看到 $\Theta_1<\Theta_0$ 就应立即停下，不再读取依赖 $x_1$ 的 $c_2$。如果仍向后全局搜索：

- 若 $x_1$ 导致 $c_2=0.9$，则 $a_2=0.72$、$\Theta_2=(1+0.8+0.72)\times0.45=1.134$，调度器回头选择验证 $x_1$；
- 若 $x_1$ 导致 $c_2=0$，则 $\Theta_2=0.81$，调度器回头选择不验证 $x_1$。

于是“是否给 $x_1$ 入场”取决于 $x_1$ 是什么。论文进一步令 target 分布为 $(0.7,0.3)$、draft 分布为 $(0.5,0.5)$，让 token A 对应高后续 confidence、B 对应低 confidence；回看式调度最后会得到输出分布 $(0.85,0.15)$，不再等于 target 的 $(0.7,0.3)$。

📌 **关键点**：投机解码的 lossless 不只取决于 rejection sampling；任何动态候选选择策略都必须满足 non-anticipating。论文指出，Algorithm 1 用第一次吞吐下降时 early stop 来建立因果屏障；只有当 $\Theta$ 沿 admission path 呈单峰、硬件容量曲线足够平滑时，这个逐步早停才同时给出全局最优吞吐。

### 5.4 理论曲线平滑，真实 GPU 曲线却有台阶

真实 $\mathrm{SPS}(B)$ 会受 kernel tile、CUDA Graph capture tier、并行拓扑等影响，常常呈锯齿和台阶，并不平滑。论文生产实现做了两项适配（§5.2）：

1. **用两步之前的 confidence 估计下一步容量 $K$**，让调度与当前 forward 异步重叠，不阻塞 Zero-Overhead Scheduling；
2. **在历史信息上做无 early-stop 的全局搜索**，跨过锯齿局部低谷；因为容量只依赖两步之前的信息，不依赖当前 token 实现值，因果屏障仍然成立。

这里要分清：Algorithm 1 是理论版本；两步延迟、动态 top-$K$ 是生产适配。后者牺牲一点容量估计时效性，换取 GPU pipeline 无气泡和对非平滑硬件曲线的适应。

## 6. 训练目标与开源复现账本

### 6.1 三项损失分别负责什么

目标模型全程冻结，embedding 与 LM Head 由 target/draft 共享并冻结；训练更新 parallel backbone、sequential block 和 confidence head。论文目标函数是：

$$
\mathcal L=
\underbrace{0.1\mathcal L_{\mathrm{ce}}}_{\text{预测正确 token}}
+
\underbrace{0.9\mathcal L_{\mathrm{tv}}}_{\text{贴近目标分布}}
+
\underbrace{1.0\mathcal L_{\mathrm{conf}}}_{\text{预测接受概率}}
$$

三项损失都按位置衰减，前面的 draft token 权重更大，因为 prefix verification 中越靠前的 token 杠杆越高。

| 损失 | 训练信号 | 为什么需要 |
| --- | --- | --- |
| $\mathcal L_{\mathrm{ce}}$ | ground-truth next token 的交叉熵 | 让 drafter 会预测正确答案 |
| $\mathcal L_{\mathrm{tv}}$ | $\|p_k^d-p_k^t\|_1$ | TV 距离直接对应接受概率，优化的不是“像不像标签”而是“能否通过 target 验收” |
| $\mathcal L_{\mathrm{conf}}$ | 软标签 $c_k^*=1-\frac12\|p_k^d-p_k^t\|_1$ 的 BCE | 给 scheduler 一条可校准的接受概率信号 |

### 6.2 离线论文设置

论文为了公平比较 Eagle3、DFlash、DSpark，统一使用：

- **目标模型**：Qwen3-4B/8B/14B、Gemma4-12B；
- **训练数据**：Open-PerfectBlend 的 130 万条 prompt，响应由各 target model 重新生成；
- **数据构成**：Chat 17.6%、Math 39.4%、Code 38.9%、Instruction 4.1%；
- **训练**：10 epochs，non-thinking mode；
- **草稿长度**：block size 7；Eagle3 TTT horizon 7；
- **模型深度**：Eagle3 1 层，DFlash/DSpark 5 层；
- **评估**：Math、Code、Chat 各 3 个 benchmark，temperature=1.0；
- **离线主实验关闭 confidence scheduler**，只比较 raw draft quality。

### 6.3 当前 DeepSpec 能复现到哪

截至 **2026-08-11**，官方仓库 [DeepSpec](https://github.com/deepseek-ai/DeepSpec)（本文核对 commit `005e03b`）已开源：

- Eagle3、DFlash、DSpark 的训练与评估代码；
- Qwen3-4B/8B/14B、Gemma4-12B 对应 checkpoint；
- 数据准备 → target cache → drafter training → acceptance evaluation 全流程；
- Qwen3-4B 当前配置中的 5 层 backbone、block size 7、Markov rank 256、global batch 512、BF16 和 10 epochs。

但“开源”不等于“轻量复现”：README 明确警告，默认 Qwen3-4B 数据设置的 target cache 约 **38 TB**，默认脚本按**单机 8 卡**设计。仓库还要求按相同训练设置比较，否则结果没有意义；目标模型若运行 thinking mode 或进入特定领域，建议重新微调 drafter。

⚠️ **不要把 Table 1 的 checkpoint 直接外推到任意模型。** Target-dependent drafter 学的是“怎样贴近这一份 target 分布”，模型、推理模式、领域和 tokenizer 变化都可能降低接受率。

## 7. 离线实验:接受长度为何真的提高

### 7.1 主结果先看宏平均

下面的宏平均由本文对论文 Table 1 的 9 个 benchmark 做等权算术平均；提升百分比与论文 §4.2 一致。$\tau$ 包含 target 产生的 bonus token。

| Target | Eagle3 $\tau$ | DFlash $\tau$ | DSpark $\tau$ | DSpark vs Eagle3 | DSpark vs DFlash | 来源口径 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3-4B | 3.611 | 4.064 | **4.727** | +30.9% | +16.3% | 本文据 Table 1 计算 |
| Qwen3-8B | 3.798 | 4.067 | **4.813** | +26.7% | +18.4% | 本文据 Table 1 计算 |
| Qwen3-14B | 3.676 | 4.039 | **4.779** | +30.0% | +18.3% | 本文据 Table 1 计算 |
| Gemma4-12B | 4.376 | 4.018 | **4.663** | +6.6% | +16.1% | 本文据 Table 1 计算 |

📌 **关键点**：DSpark 在 4 个 target、9 个 benchmark 上都取得最高接受长度，但“相对 Eagle3 约 30%”只适用于 Qwen3 三档；Gemma4 上 Eagle3 本来就强，DSpark 的宏平均优势约 6.6%。不能把 Qwen 结论复制到所有模型族。

### 7.2 任务结构决定可投机程度

以 Qwen3-4B 为例，论文 Table 1 的三域平均为：

| 领域 | Eagle3 | DFlash | DSpark | 来源口径 |
| --- | ---: | ---: | ---: | --- |
| Math | 4.56 | 4.80 | **5.57** | 本文据 Table 1 计算 |
| Code | 3.87 | 4.44 | **5.12** | 本文据 Table 1 计算 |
| Chat | 2.40 | 2.95 | **3.49** | 本文据 Table 1 计算 |

数学和代码更有结构，下一 token 的选择空间较窄；开放聊天允许更多合法续写，所以接受长度低。**这不仅影响 drafter 选型，也说明 per-request verification budget 比全 batch 固定 $K$ 更合理。**

### 7.3 论文证明了什么，没有证明什么

论文离线实验有三个强点：统一训练框架与数据、覆盖两种模型族、同时看位置条件接受率和长度/深度消融。但边界也清楚：

- 主实验是 non-thinking mode；
- scheduler 被关闭，只能说明 drafter 更准；
- 指标是 accepted length，不是公开引擎上的 wall-clock speedup；
- checkpoint 和 target 强绑定；
- 训练成本没有给出 GPU 小时账本。

## 8. 线上部署:真正改变的是吞吐与交互性的前沿

### 8.1 V4 生产版与开源实验不是同一配置

DeepSeek-V4 内部部署使用 DSpark-5，最大 draft length $\gamma=5$，默认 Markov Head；parallel backbone 是 3 层 MoE，并结合 mHC 与 sliding-window attention 128。训练系统还做了两项大规模适配：

- **Hidden-state communication**：跨 worker 不传约 $10^5$ 维完整词表 logits，只传 LM Head 前的 hidden state，通信从 $O(|\mathcal V|)$ 降为 $O(d)$；
- **Anchor-bounded sequence packing**：固定抽取若干 anchor，把独立 prediction block pack 成稠密 batch，让 drafter 成本不随原文档上下文长度增长。

这些是内部 V4 训练/部署设置，不等同于 DeepSpec 的 Qwen/Gemma 配置。

### 8.2 先看稳定口径:匹配吞吐与中等 SLA

<img src="/AIInfraGuide/images/dspark-fig7-production-frontier.png" alt="DeepSeek-V4-Flash 与 V4-Pro 线上吞吐和单用户 TPS 前沿：DSpark 相对 MTP-1 将 Pareto 前沿向右上方外推；匹配吞吐下 Flash 单用户速度提高 60% 到 85%，Pro 提高 57% 到 78%" style="max-width: 95%; display: block; margin: 0 auto;" />

*图源:DSpark 论文 Figure 7(arXiv:2607.05147);散点为线上 telemetry，实线为拟合前沿*

论文用真实用户流量比较 DSpark-5 与此前生产基线 MTP-1：

| 模型与口径 | DSpark 相对 MTP-1 | 应怎样解释 |
| --- | ---: | --- |
| V4-Flash，匹配 practical throughput | **单用户 +60%–85%** | 同等系统产能下用户看到 token 更快 |
| V4-Pro，匹配 system capacity | **单用户 +57%–78%** | 同上，Pro 的区间稍低 |
| V4-Flash，80 tok/s/user SLA | **aggregate throughput +51%** | 中等交互目标下多服务约一半 token |
| V4-Pro，35 tok/s/user SLA | **aggregate throughput +52%** | 中等交互目标下的吞吐提升 |

这些是论文的核心生产结论，也是开篇采用的数字。

### 8.3 极端倍率为什么不能写成“DSpark 快 7.6 倍”

论文还报告：

- V4-Flash 在 120 tok/s/user SLA 下 nominal throughput **+661%**；
- V4-Pro 在 50 tok/s/user SLA 下 nominal throughput **+406%**。

但原文主动提醒：此时 MTP-1 已接近 operational boundary，只能维持很小并发，分母接近“性能悬崖”。所以这两组数字主要证明 **DSpark 解锁了基线难以达到的交互性档位**，不代表常规负载下端到端稳定快 7.61 倍或 5.06 倍。

正确表述是：**DSpark 把 throughput–interactivity Pareto frontier 向右上方推了出去。** 在常规工作点看 51%/52% 吞吐，或分别看 Flash 的 60%–85% 与 Pro 的 57%–78% 单用户速度；在严格 SLA 点看“以前不可行、现在可行”。

### 8.4 预算如何随负载变化

论文 Figure 8 显示：在 V4-Flash 少于约 200 并发请求、V4-Pro 少于约 150 并发请求的生产常见区间，调度器把每请求验证预算从 MTP-1 的静态 2 个 token 扩到约 **4–6 个**；并发继续上升、target 逐渐饱和后，预算会平滑缩短。

这正对应 $\Theta=\tau\cdot\mathrm{SPS}(B)$：

- 空闲时 $\mathrm{SPS}(B)$ 对多几个 token 不敏感，应该尽量提高 $\tau$；
- 繁忙时 $\mathrm{SPS}(B)$ 对 batch token 数敏感，应该剪掉低生存概率后缀。

## 9. 从论文到当前工程:SGLang 如何把调度收益落到 GPU

论文发布后，SGLang 在 2026-07-06 的官方工程文章中公开了 DSpark 集成。**以下均是 SGLang 的后续工程实现，不是 DSpark 论文的实验栈。** 本文核对的是文章给出的复现 commit [`692c5f7d`](https://github.com/sgl-project/sglang/commit/692c5f7d532f129424b57961c262bbd253b411dc)。SGLang 明确声明：硬件、引擎和流量都与论文不同，因此复现的是**机制和曲线形状**，不是逐位复制论文数字。

### 9.1 三种 verify mode 分开正确性与观测

- **static**：整块都验证，固定长度基线；
- **compact**：只验证 scheduler 给每个请求选择的窗口，生产路径；
- **cap-accept**：整块验证，但只提交窗口内 token；输出与 compact 相同，同时看到“不剪枝本来能接受多少”的 ceiling。

为什么需要 `cap-accept`？Compact 把后缀剪掉后，你再也观测不到它原本是否会通过，无法判断剪枝是不是太激进。`cap-accept` 用额外计算换一个反事实观测基线。

### 9.2 Ragged verify 不能再 padding 回去

每请求窗口不同，若把所有请求 padding 到最大长度，逻辑上虽然“剪了”，GPU 仍计算完整矩形，收益等于零。SGLang 把变长请求 front-pack 到一个紧凑 buffer，以**总 token 数**选择最近的 CUDA Graph capture tier，并用 `cu_seqlens` 风格元数据执行 varlen attention。

这意味着预算缩短后，重放的是更小的图，attention/MLP 真正处理更少行，而不是在全宽 forward 上套 mask。

### 9.3 Cost table 与重叠调度

SGLang 把 step time 拟合为：

$$
T(bs,K)=\text{bias}+\alpha(bs)+\theta(M),\qquad M=bs+K
$$

- $\alpha(bs)$：请求数相关、剪枝省不掉的 draft 与部分 attention 基础成本；
- $\theta(M)$：target 验证 token 的增量成本，才是 trimming 能回收的部分。

同时，scheduler 和 forward 放到不同 stream 重叠，用两步延迟的 confidence relay 避免每 step CPU/GPU 同步气泡。SGLang 报告在 DeepSeek-V4-Pro、B300、TP=8、batch size 1 的特定设置下，经过 fused Triton kernel、sharded drafter matmul 和 overlap scheduler 后达到 **383.7 tok/s、接受长度约 5**；这是 SGLang 自己的工程测量，不是 DSpark 论文主结果。

### 9.4 混合流量才是 per-request 调度的主场

SGLang 把 GSM8K、Arena-Hard、Poetry 混在一个服务流量里，示例平均验证窗口分别为 **5.24、3.78、2.91**，相对不剪枝接受 ceiling 的利用率仍保持 **0.88–0.97**。这比单一数据集曲线更能说明价值：同一个 batch 内，容易猜的数学请求拿更长窗口，开放诗歌请求拿更短窗口。

⚠️ **原文时代 vs 当前工程**：DSpark 论文给出算法、内部 V4 生产实现和开源 DeepSpec；SGLang 随后公开了 ragged CUDA Graph、三种 verify mode、cost profiler 与复现命令。后者可以帮助理解“怎样落地”，但不能倒写成论文作者当时所有实验都使用了这些具体实现。

## 10. 权衡、局限与选型边界

### 10.1 DSpark 牺牲了什么

| 收益 | 付出的代价 |
| --- | --- |
| 并行 backbone 提供高容量、高 position-1 接受率 | 每轮无论最终验证多短，都先支付整块 draft forward |
| Markov Head 缓解后缀衰减 | 增加小型串行采样 loop 与 kernel 工程 |
| Confidence Scheduler 减少无效验证 | 需要训练、校准 confidence，维护硬件 cost table |
| Per-request 变长预算适应混合流量 | 引擎必须支持 ragged verify、CUDA Graph tier 和可观测性 |
| 保持 target 分布不变 | 调度必须满足 non-anticipating，不能随意“看完再挑” |

论文自己点名的主要局限是：对于天然低接受率的复杂请求，parallel backbone 生成 $\gamma$ token 的固定成本无法回收。未来可以做 difficulty-aware early exit，让难请求提前退出 drafter。

### 10.2 什么场景值得关注

**更可能受益：**

- 目标模型足够大，一次 target decode 很贵；
- 代码、数学、结构化输出等可预测负载占比高；
- 线上并发从低到高波动，固定 draft length 难兼顾；
- 服务引擎掌握 batch、CUDA Graph 和 cost profiling，能让剪枝变成真实算力减少；
- 有资源为特定 target/领域训练并校准 drafter。

**不应直接套用：**

- 极小 target，drafter 固定成本占比过大；
- target 或领域频繁变化，却不重训 drafter；
- 引擎只支持固定宽度验证，最后仍 padding；
- 只看离线 accepted length 就宣称线上加速；
- 把 V4 live traffic 的 60%–85% 外推到任意 GPU、模型和并发。

### 10.3 一句话选型规则

> **如果瓶颈只是 batch=1 的草稿质量，先看 Eagle/普通 speculative decoding；如果既想保留并行 drafter 的低起草延迟，又要在混合请求和高并发下动态控制 verification waste，DSpark 才是更完整的算法—系统联合方案。**

## 📝 总结

1. **三杠杆统一**：$L=(T_{\text{draft}}+T_{\text{verify}})/\tau$；DSpark 同时追求 draft faster、draft better、verify smarter。
2. **半自回归草稿**：深并行 backbone 提供第一个位置的容量，低秩 Markov Head 用上一个已采样 token 修正后续 logits，缓解 multi-modal collision 与 suffix decay。
3. **置信度可计算**：$c_k^*=1-\frac12\|p_k^d-p_k^t\|_1$ 对应条件接受概率；前缀价值是连乘 $a_j=\prod_{i\le j}c_i$，STS 将过度自信的分数校准到可用于吞吐计算的尺度。
4. **调度目标落到硬件**：每请求选择 $\ell_r$，最大化 $\Theta=\tau\cdot\mathrm{SPS}(B)$；GPU 空闲时多验，繁忙时剪掉低生存概率后缀。
5. **Lossless 有额外条件**：动态调度不能让 token 的入场决定依赖 token 自身实现值；论文用 early stop 和生产中的两步历史容量估计建立因果屏障。
6. **离线结果**：Qwen3-4B/8B/14B 上相对 DFlash 的宏平均接受长度提高 16.3%/18.4%/18.3%；优势在长 block 上扩大，串行头在论文指定设置中的整轮开销为 0.2%–1.3%。
7. **线上结果**：DeepSeek-V4 匹配吞吐下，Flash 单用户速度提高 60%–85%，Pro 提高 57%–78%；中等 SLA 下 aggregate throughput 约提高 51%–52%；极端 661%/406% 代表基线性能悬崖与可行前沿外移，不是普适倍数。
8. **工程闭环**：DeepSpec 开源训练和 checkpoint，SGLang 展示 ragged verify、CUDA Graph、cost table 与 overlap scheduling；没有这些引擎能力，算法剪枝未必变成墙钟收益。

## 🎯 自我检验清单

- 能解释 $L=(T_{\text{draft}}+T_{\text{verify}})/\tau$ 的三个优化杠杆，并给出“接受长度增加但延迟反而上升”的数值例子。
- 能用 `of course / no problem` 例子解释 parallel drafter 的 multi-modal collision。
- 能写出 $p_k\propto\exp(U_k+B_k)$，说明 parallel backbone 与 Markov Head 各自负责什么。
- 能解释为什么低秩 Markov Head 只看前一个 token，仍能显著缓解后缀衰减。
- 给定 $[0.9,0.8,0.5,0.2]$，能手算前缀生存概率 $[0.9,0.72,0.36,0.072]$。
- 能说明 confidence 排序准确仍不够，为什么硬件调度还要求 STS 校准概率绝对值。
- 能从 $B=\sum_r(1+\ell_r)$、$\tau=\sum_r(1+\sum_j a_{r,j})$ 推出吞吐目标 $\Theta=\tau\cdot\mathrm{SPS}(B)$。
- 能复算 Appendix A 中 $\Theta_0=1.0$、$\Theta_1=0.9$、$\Theta_2=1.134$ 的反例，并解释输出为何从 $(0.7,0.3)$ 偏到 $(0.85,0.15)$。
- 能区分 Table 1 accepted length、Figure 7 tok/s/user 和 aggregate throughput 三种指标口径。
- 能复述 V4-Flash 60%–85%、V4-Pro 57%–78%、中等 SLA +51%/+52% 各自的比较条件。
- 能解释为什么 +661%/+406% 应理解为性能前沿外移，而不是常规加速倍率。
- 能说明 compact ragged verify 为什么必须真正减少 GPU 处理行，不能只在 padding 后加 mask。
- 能列出至少三个 DSpark 不适合直接套用的条件，并说明要先 benchmark 哪些指标。

## 📚 参考资料

- 论文与代码:
  - [DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation](https://arxiv.org/abs/2607.05147):本文精读对象，算法、离线实验、V4 线上部署与 Appendix A 无偏性反例的原始来源。
  - [DeepSpec — DeepSeek-AI](https://github.com/deepseek-ai/DeepSpec):DSpark、DFlash、Eagle3 的训练/评估代码和公开 checkpoint；本文核对 commit `005e03b`。
  - [DeepSpec checkpoints](https://huggingface.co/collections/deepseek-ai/deepspec):论文 Table 1 对应的 Qwen3 与 Gemma4 草稿模型集合。
- 当前工程:
  - [DSpark in SGLang](https://www.lmsys.org/blog/2026-07-06-dspark-sglang/):Ragged verify、CUDA Graph、三种 verify mode、cost table 和复现命令；本文核对其公开 commit `692c5f7d`。
  - [NVIDIA NeMo AutoModel DSpark Recipe](https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/dspark-speculative-decoding):另一条当前训练路径，说明三项 loss、FSDP2 和 Qwen3/Gemma4 配置面。
- 站内相关:
  - [5.1 投机解码与 Rejection Sampling](/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/51-核心原理与rejection-sampling):补齐标准 speculative sampling 的无偏性推导。
  - [5.2 Draft 模型与 N-gram 方案](/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/52-draft模型与n-gram方案):理解 target-dependent drafter、草稿成本和接受率权衡。
  - [5.3 Medusa 与 EAGLE](/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/53-self-draft方案-medusa与eagle):对比多头、自回归特征 drafter 与 DSpark 的并行—半自回归路线。
  - [5.4 收益边界与限制](/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/54-收益边界与限制):从 $T_d/T_t$、接受率和 batch 解释何时投机不赚钱。
  - [5.5 vLLM 投机解码实战](/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/55-vllm投机解码实战):建立 benchmark 与生产参数的实践入口；DSpark 当前公开工程实现以 SGLang 为主。
