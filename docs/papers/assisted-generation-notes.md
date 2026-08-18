---
title: "Assisted Generation 解读：让 1/10 大小的小模型先写草稿，生成延迟最高降 10 倍"
description: "用‘实习生起草、主编审阅’的框架拆解 HuggingFace 的 Assisted Generation：为什么自回归慢在‘搬’不在‘算’，前向传播如何兼任‘验收’，三档实测加速(2x/3x/10x)各自的前提，以及它与 Batching、张量并行三条路线的账本对比。"
pubDate: 2026-08-09
originalUrl: "https://huggingface.co/blog/assisted-generation"
sourceType: "blog"
originalAuthor: "Joao Gante (Hugging Face)"
tags: ["Speculative Decoding", "Assisted Generation", "推理优化", "低延迟"]
---

> 原文：[Assisted Generation: a new direction toward low-latency text generation](https://huggingface.co/blog/assisted-generation)(Joao Gante,Hugging Face Blog,2023-05-11)

自回归生成慢的根源不是“算不过来”，而是“搬不过来”。HuggingFace 这篇博客提出 **Assisted Generation（辅助生成，Speculative Decoding 的一种工程实现）**：让一个比主模型小约一个数量级的小模型先“打草稿”，主模型一次前向（forward，输入→输出的一次完整计算）批量“验收”。在普通消费级硬件上，生成延迟最高能压低 10 倍（内存 offloading 场景：模型装不进显存，权重放内存、用到再搬）；模型装得进显存时，加速最高 2 倍，配合 INT8（8 位整数）量化最高 3 倍。做法简单到只改一个 API 参数，但背后的洞察——前向传播不仅能“预测”下一个 token（文本的基本单位，大约一个词），还能“验证”一串候选 token——值得每个做推理优化的人记住。

> 原文：[Assisted Generation: a new direction toward low-latency text generation](https://huggingface.co/blog/assisted-generation)（Joao Gante,Hugging Face Blog,发布于 2023-05-11;本文访问日期 2026-08-15）

<!-- more -->

## 📑 目录

- [🗺️ 原文阅读地图](#️-原文阅读地图)
- [1. 为什么自回归生成这么慢](#1-为什么自回归生成这么慢)
- [2. 核心思想：前向传播不只会预测，还会验收](#2-核心思想前向传播不只会预测还会验收)
- [3. 方法拆解：六步循环 + 一行代码](#3-方法拆解六步循环--一行代码)
- [4. 效果与对比：三组数字三个场景](#4-效果与对比三组数字三个场景)
- [5. 权衡与局限：不是银弹](#5-权衡与局限不是银弹)
- [🕰️ 原文时代 vs 当前工程](#️-原文时代-vs-当前工程)
- [📝 总结](#-总结)
- [🎯 延伸思考：自我检验清单](#-延伸思考自我检验清单)
- [📚 参考资料](#-参考资料)

## 🗺️ 原文阅读地图

这篇博客是叙述性文章，共六节。本文选择性精讲如下：

| 原文单元（博客节） | 处理深度 | 本文位置与理由 | 来源锚点 |
| --- | --- | --- | --- |
| Understanding text generation latency（延迟瓶颈定性、三条优化路线、DeepSpeed 1.5× 数据） | 精讲 | 第 1 节，先定性再展开 | 博客第 1 节 |
| Language decoder forward pass, revisited（不缓存时输出所有位置 logits、argmax 可复现输入） | 精讲 | 第 2 节，机制卡 1 | 博客第 2 节 |
| Greedy decoding with assisted generation（六步循环、候选数动态调整、2×/3×/10× 数字、batching 对比） | 精讲 | 第 3、4 节，机制卡 2-3,含轨迹示例 | 博客第 3 节 |
| Sample with assisted generation（采样模式、温度与命中率） | 简述 | 第 4 节结尾，只保留结论 | 博客第 4 节 |
| Future directions（对“固定规模计算生成 token”假设的反思） | 简述 | 第 5 节，并入权衡 | 博客第 5 节 |
| Related Work（Blockwise Parallel Decoding、Speculative Sampling） | 简述 | 第 5 节末尾点名两个后续工作 | 博客第 6 节 |

📌 **本文承诺**：读完后，你应该能手算一轮“候选 A-B-C、主模型选 A-X”的提交序列，说清三档加速数字(2×/3×/10×)各自的场景前提，并区分 2023 博客时代的约束与当前 Transformers 的 assisted decoding。

## 1. 为什么自回归生成这么慢

**问题是：生成一段文本为什么这么慢？** 答案的关键在于“自回归”三个字——模型一次前向只吐一个 token,生成 100 个 token 就要串行跑 100 次前向传播，而大模型单次前向本身就是毫秒级的重活。

**那单次前向又慢在哪？** 前向传播的主体是矩阵乘法，而矩阵乘法是内存带宽受限的：瓶颈在于把模型每层的权重从显存搬进计算核心，而不是核心里的计算本身。换句话说，GPU 的计算单元大部分时间在“等食材”，而不是在“炒菜”。

> 打个比方：GPU 是一座超级工厂，计算核心是厨师，显存是仓库，而显存和计算核心之间的数据通路是一条**传送带**。矩阵乘法这道菜每个步骤都要从仓库搬原料（权重），传送带只有那么宽——厨师再快，也只能等料。这就是为什么推理延迟的账，主要记在“搬运”头上，而不是“计算”头上。

**问题变成：搬运瓶颈已经存在，还有哪些路可走？** 博客盘点了当时的三条主流优化路线：

- **硬件级优化**：Flash Attention（重排注意力计算顺序）、INT8 量化（把权重变小，搬运量直接变少）——但这类优化有上限，做完了就没得再压。
- **Batching（批处理）**：把多个请求拼成一趟，权重只搬一次、服务多行输入，吞吐暴涨、延迟几乎不涨。代价是额外显存；走到极端就是 FlexGen 这类“牺牲延迟换吞吐”的方案。
- **Tensor Parallelism（张量并行）**：把权重拆到多张卡上，带宽负担被分摊，但引入设备间通信开销和真金白银的多卡成本。按博客引用的 [DeepSpeed 数据](https://www.microsoft.com/en-us/research/blog/deepspeed-accelerating-large-scale-model-inference-and-training-via-system-optimizations/),17B 模型拆 4 张 GPU,延迟只降了 1.5 倍。

三条路各有各的贵。博客的结论是：**硬件优化做尽之后，压延迟的选项屈指可数，而且都很贵**——于是它换了个思路：能不能从“解码方式”本身下手？

## 2. 核心思想：前向传播不只会预测，还会验收

**为什么前向传播只能“预测下一个 token”？** 博客先指出一个被忽略的事实：如果不开启缓存（KV Cache）），把已经生成的一整段序列喂给模型，输出其实是每个位置对应的 next-token logits（模型给每个候选词打的原始分）。在 greedy 解码（贪心：每次都挑得分最高的词）下，对这串 logits 取 argmax（选出得分最高的那个），你能把输入序列原样复现出来——代码里 `model(generated).logits.argmax(-1) == generated[0, 1:]` 验证返回 `True`。

**这意味着前向传播有了第二种用途：验收。** 喂一串候选 token 进去，模型会告诉你“这些 token 是我会生成的，还是不是”。

**凭什么这能省延迟？** 博客先讲了一个思想实验：假设你有一个零延迟的 oracle 助手，和你模型生成得一模一样。用它先产出候选 token,再让主模型一次前向“盖章确认”——理想情况下，生成延迟从

$$\underbrace{O(n)}_{\text{逐 token 串行: n 次前向}} \quad \rightarrow \quad \underbrace{O(1)}_{\text{助手一次生成 + 主模型一次验收}}$$

$n$ 是生成 token 数。长文本生成就是几个数量级的差距。现实中没有 oracle，但助手会犯错也没关系。规则是：**从第一个错位开始，后面的候选全部作废；主模型把错位纠正后，助手再重新起草，如此循环**。即使助手时不时错几个 token，延迟依然能低一个数量级。

**现实中谁当助手？** 博客的答案是：同架构、同训练方式、小得多的模型——主模型的小号版。当两者规模差距足够大（原文要求至少差一个数量级，越大越好），小模型跑一溜候选 token 的成本，相比省下的几次大模型前向，就成了"afterthought"（零头）。

> 打个比方：这就像让**实习生先起草一段文字，资深主编一次性审阅**。猜对的地方直接放行，猜错的地方主编当场改掉，然后实习生接着起草。审阅一稿的时间，顶得上自己逐字写几十稿。原文自己也点破了这个套娃结构：你在文本生成里跑了一次文本生成，像电影《Inception》——梦里套梦。

**还有个硬性要求：助手必须和主模型用完全相同的 tokenizer（分词器，把文本切成 token 的工具）。** 为什么？如果不一致，每次都要做昂贵的 token 解码 + 重编码，而且这步发生在 CPU 上，还牵扯慢速的设备间数据传输——助手“快”这个前提就没了。

**以及一个控制开销的启发式（经验规则）：候选数动态调整。** 你没法预知助手能猜对几个，但可以看历史成绩——首次调用默认生成 5 个候选 token;这一轮全部匹配就 +2，有错就 -1。输出里有的段落好猜（如常见话术），有的难猜，候选数随之伸缩。

## 3. 方法拆解：六步循环 + 一行代码

**具体到每一步，assisted generation 的循环长什么样？** 原文的实现可以拆成六步（伪代码为原文逻辑的简化示意）：

```python
# 简化伪代码:assisted generation 主循环
candidates = assistant_generate(prompt, num=5)   # ① 助手 greedy 生成候选(首次 5 个)
while not done:
    logits = model.forward(candidates)           # ② 主模型对整串候选做一次前向
    next_tokens = select(logits)                 # ③ greedy 用 argmax,sampling 用 multinomial
    n = count_matches(next_tokens, candidates)   # ④ 从左到右数匹配数,第一个错位之后全部作废
    emit(next_tokens[:n+1])                      # ⑤ 保留匹配的 n 个 + 第一个分歧 token(主模型自己选的)
    candidates = assistant_generate(继续, num=5+n_adjust)  # ⑥ 全匹配 +2,有错 -1,回到 ②
```

**为什么要从第一个错位处全部作废？** 自回归的因果性：一旦某个 token 和主模型不一致，它后面的候选全部建立在一个“主模型不会说出口”的基础上，一个都不能留。这也是第 ⑤ 步只保留“匹配串 + 第一个分歧 token”的原因——分歧 token 是主模型从合法前缀上真实选出的，直接可用。

**一个具体的 token 轨迹（手算一遍）：** 假设助手对前缀 "I love to" 起草了 3 个候选 `[code, write, stories]`，主模型一次前向对这 3 个位置逐一“验收”：

| 位置 | 助手候选 | 主模型选择(argmax) | 匹配？ |
| --- | --- | --- | --- |
| 1 | `code` | `code` | ✅ 匹配 |
| 2 | `write` | `read` | ❌ 错位，从这里作废 |
| 3 | `stories` | （不算数） | ❌ 作废 |

主模型第 1 位确实会输出 `code`，第 2 位主模型自己选了 `read`——所以最终提交的是 `[code, read]`：匹配的 `code` 直接采用，分歧处用主模型自己的选择 `read`，`stories` 因为建立在“主模型会说 write”这个错误前提上，一个都不能留。助手下一轮从 "I love to code read" 继续起草。这一轮实际赚到 1 次大模型前向（本来要 3 次）。

**工程上有多简单？** 在 🤗 Transformers 里，这一切被收敛成一个参数：

```python
outputs = model.generate(**inputs, assistant_model=assistant_model)
```

唯一要注意的：**发布时（2023 年 5 月）assisted generation 只支持 batch size = 1**——它服务的是“单请求低延迟”场景，和 batching 的“高吞吐”路线互补而非替代。

> 💡 **提示**：第 ④ 步的匹配数，就是这一轮“赚到”的 token 数——它省掉的是等量的大模型前向；而花出去的是 n 次小模型前向（每个候选 token 一次）。这笔账能不能赚，取决于小模型前向有多便宜、猜对率有多高，这正是下一节数字要回答的问题。

## 4. 效果与对比：三组数字三个场景

**assisted generation 到底能快多少？** 博客给出的数字全部来自 🤗 Transformers 直接拉取、无任何额外优化，测量设备是 RTX3090 消费级显卡，可以复现。按场景分三档：

| 📊 场景 | 加速 | 前提 |
| --- | --- | --- |
| 模型装不进显存，靠内存 offloading | **最高 10 倍** | 硬件条件最差，反而收益最大 |
| 模型装进显存 + INT8 量化 | **最高 3 倍** | 权重搬运量已减半，仍有 3 倍 |
| 模型装进显存，普通精度 | **最高 2 倍** | 最朴素场景 |

📌 **关键点**：加速最猛的不是配置最好的场景，而是最卡的场景(offloading)——因为搬运瓶颈越严重，少跑几次前向的价值就越大。但博客同时强调：这不是银弹，上生产前必须自己 benchmark。

> 🔗 **来源锚点**：以上三档加速数字与下方 Batching/TP 对比数字均出自博客 "Greedy decoding with assisted generation" 一节（博客第 3 节，标题直译“带辅助生成的贪心解码”），测量设备 RTX3090、数字为 🤗 Transformers 直接拉取的实测；1.5× 的 TP 数字出自同博客第 1 节 "Understanding text generation latency" 引用的 DeepSpeed 数据；“未来方向”与 Blockwise Parallel Decoding / Speculative Sampling 两个后续工作在博客 "Future directions" 与 "Related Work" 两节。

**对照一下另外两条路的账本**（同为博客实测/引用数据，注意口径）：

| 📊 路线 | 数字 | 代价 |
| --- | --- | --- |
| Batching(distilgpt2, RTX3090) | batch=1 时 **418.3 tok/s**,batch=64 时 **16266.2 tok/s（约 39 倍吞吐）** | 额外显存 |
| Tensor Parallelism（17B, 4×GPU, DeepSpeed 数据） | 延迟降 **1.5 倍** | 多卡成本 + 设备间通信 |
| Assisted Generation | 延迟降 2~10 倍 | 双模型显存 + 助手开销 |

📌 **关键点**：Batching 的数字是吞吐(tokens/s),assisted generation 的数字是延迟——前者多赚的是并发请求，后者多赚的是单请求的响应速度，两者可以叠加使用，不是二选一。

**什么任务最吃这套？** 博客的观察：assisted generation 在输入锚定任务（input-grounded）上最出彩——自动语音识别（ASR）、翻译、摘要——因为这类任务的输出高度可预测，助手猜对率高。而开放式的创作型任务（比如聊天机器人）用的是 sampling（按概率随机抽选）而非 greedy，助手会更容易猜错，收益缩水；补救办法是压低 temperature：温度接近 0 时 sampling 趋近 greedy，助手的命中率就回来了；温度远大于 1 时采样接近均匀分布，助手基本瞎猜。

## 5. 权衡与局限：不是银弹

**assisted generation 牺牲了什么，换取了什么？** 原文明确说它是“一场平衡术”(balancing act)：

- **牺牲一：双模型显存**。主模型之外，助手模型也要占设备内存，显存吃紧的场景要先算这笔账。
- **牺牲二：助手的推理开销**。每个候选 token 都要一次助手前向，助手质量差时，你会“付了助手钱，没捡到便宜”。
- **牺牲三：对助手的要求苛刻**。同 tokenizer 是硬约束；至少小一个数量级、同架构同训练方式，意味着不是随便找个模型就能当助手。
- **换取**：省下的是主模型一票前向传播——在内存带宽受限的世界里，这是最贵的资源。

**局限盘点**（全部来自原文）：
- 发布时仅支持 **batch size 1**；采样模式下收益依赖低温。
- 助手的选择无法自动化，需要自己试、自己 benchmark。
- 数字是消费级硬件 + 无额外优化下的结果，不代表所有环境。

**以及一个更根本的追问**：博客在结尾提出，assisted generation 动摇了一个默认假设——“每个新 token 都必须由固定规模的计算产生”。既然大段输出可以由小得多的模型生成、再由大模型把关，那新模型架构和新解码方法就还有巨大的优化空间；同时，高质量小模型的发布将是放大这套收益的关键。原文也补充了同思路的后续工作：Google Brain 的 Blockwise Parallel Decoding 和 DeepMind 的 Speculative Sampling。

## 🕰️ 原文时代 vs 当前工程

这篇博客发布于 **2023-05-11**，距今超过两年，其中的工程约束需要与当前 Transformers 分开看：

| 维度 | 原文时代（2023-05 博客） | 当前工程（截至 2026-08-15 复核官方文档） |
| --- | --- | --- |
| tokenizer 约束 | 助手与主模型必须共享 tokenizer，否则验证无从谈起 | 基础 assisted decoding 仍要求同 tokenizer；官方文档另列 **Universal Assisted Decoding(UAD)**，允许不同 tokenizer 的模型配对（通过解码/重编码桥接），不再是硬性全局要求 |
| 解码方式 | 主要演示 greedy；sampling 可用但收益依赖低温 | 当前文档明确支持 greedy 与 sampling 两种路径，采样验收路径仍是官方支持面 |
| batch 支持 | 发布时仅 batch size 1 | 官方文档仍将基础路径描述为单请求场景（不批处理多请求），与 batching 路线互补的定位不变 |
| 候选数控制 | 固定启发式：初值 5、全对 +2、有错 -1 | GenerationConfig 仍提供历史式候选数启发式，也提供置信度阈值等更细的控制面——接口选择更多，不构成对 2023 规则的追溯性改写 |

**结论边界**：博客的机制（前向验收、错位作废、候选数动态调整）与核心数字(2×/3×/10×)属于原文时代的记录；上生产前以所用 Transformers 版本的官方文档（assisted_decoding 页面）为准，并自己 benchmark。本节的“当前工程”描述基于 2026-08-15 对官方文档的复核，版本迭代后需重新核对。

## 📝 总结

1. **瓶颈定性**：自回归生成的延迟瓶颈是内存带宽（搬权重），不是算力——这是整篇博客的地基。
2. **核心洞察**：前向传播不止能预测下一个 token,还能“验收”一串候选 token（不缓存时输出所有位置的 logits,argmax 可复现输入序列）。
3. **方法**：小一个数量级、同 tokenizer 的助手模型 greedy 起草候选，主模型一次前向批量验收，从左到右匹配、错位即作废，候选数动态调整（初值 5、全对 +2、有错 -1）；Transformers 里一行 `assistant_model=` 启用。
4. **效果**：RTX3090 上延迟最高降 10 倍(offloading)/ 3 倍(INT8)/ 2 倍（显存内），输入锚定任务收益最大；但不是银弹，必须实测。
5. **权衡**：双模型显存 + 助手开销，换主模型前向次数的锐减；与 batching（吞吐）、tensor parallelism（多卡）是三条互补路线。

## 🎯 延伸思考：自我检验清单

- 能解释为什么自回归生成的延迟瓶颈是内存带宽而不是计算量，并举出“传送带”之外的自己的类比。
- 能说明不缓存时一次前向输出的是“所有位置的 next-token logits”，以及为什么 greedy 下 argmax 能复现输入序列。
- 能写出 assisted generation 六步循环，并解释为什么第一个错位之后的候选必须全部作废（自回归因果性）。
- 能说出候选数动态调整的完整规则（初值 5、全匹配 +2、有错 -1）和这样做的动机。
- 能复述三档加速数字(2x / 3x / 10x)各自对应的场景前提，不张冠李戴。
- 能说明为什么“模型装不进显存时加速反而最大”（搬运瓶颈越严重，省前向越值钱）。
- 能解释为什么助手必须与主模型共享 tokenizer（避免 CPU 上的解码/重编码和跨设备传输）。
- 能区分博客里的三组数字口径：吞吐(418.3 → 16266.2 tok/s)、延迟（2~10 倍）、TP 延迟（1.5 倍）。
- 能说明 sampling 模式下为什么温度越低、assisted generation 收益越大。
- 能说出本方法的两个已点名局限（batch size 1;助手选择无法自动化），以及原文点名的两个后续工作（Blockwise Parallel Decoding、Speculative Sampling）。

## 📚 参考资料

- 论文/原文：
  - [Assisted Generation: a new direction toward low-latency text generation — Joao Gante, Hugging Face Blog](https://huggingface.co/blog/assisted-generation)：本文解读对象，2023-05-11 发布，含全部代码示例与实测数字。
  - [Blockwise Parallel Decoding — Google Brain](https://arxiv.org/abs/2211.17192)：同思路的并行解码工作，原文点名。
  - [Speculative Sampling — DeepMind](https://arxiv.org/abs/2302.01318)：用 rejection sampling 保证无偏的投机采样，原文点名。
- 官方文档：
  - [🤗 Transformers assisted generation 文档](https://huggingface.co/docs/transformers/main/en/llm_opt)：`assistant_model` 参数的用法与后续演进。
  - [🤗 Transformers text generation 博客](https://huggingface.co/blog/how-to-generate)：原文引用的解码策略总览，讲清 greedy / sampling / beam search。
- 站内相关：
  - [5.1 Speculative Sampling 核心原理(站内)](/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/51-核心原理与rejection-sampling)：同主题的系统化讲解，含无偏性证明与加速比公式推导。
  - [1.1 LLM 推理基础(站内)](/AIInfraGuide/inference/模块四-推理优化/第1章-llm推理基础/11-llm推理基础)：Prefill/Decode 两阶段与 KV Cache 显存账本，理解本文瓶颈分析的前置。
- 延伸阅读：
  - [DeepSpeed 推理优化博客](https://www.microsoft.com/en-us/research/blog/deepspeed-accelerating-large-scale-model-inference-and-training-via-system-optimizations/)：原文引用 17B 模型 4 卡 TP 延迟降 1.5 倍的数据出处。
  - [FlexGen](https://github.com/FMInference/FlexGen)：原文点名的“牺牲延迟换吞吐”方案，offloading 路线的代表。
