---
title: "Orca 精读：把调度粒度从请求级降到迭代级，同延迟下吞吐提升 36.9×"
description: "拆解 OSDI 2022 的 Orca:iteration-level scheduling 如何让提前完成的请求立即出批、新请求最快下一迭代入批，selective batching 如何只批非 Attention 算子，以及 175B 模型上同延迟下 36.9× 吞吐提升的真实口径与 vLLM continuous batching 的承继关系。"
pubDate: 2026-08-11
originalUrl: "https://www.usenix.org/conference/osdi22/presentation/yu"
sourceType: "paper"
originalAuthor: "Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, Byung-Gon Chun (Seoul National University & FriendliAI)"
tags: ["Orca", "Continuous Batching", "迭代级调度", "推理引擎", "LLM Serving"]
stage: engine
order: 2
prereqs: ["pagedattention-notes"]
minutes: 40
difficulty: 2
---

> 原文：[Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)(Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, Byung-Gon Chun,Seoul National University & FriendliAI,OSDI '22,pp. 521–538)

LLM 服务系统里最隐蔽的浪费，藏在调度粒度里。Orca(OSDI 2022)把调度从请求级降到迭代级：每跑完一次 Transformer 前向，调度器就重新决定“下一轮跑谁”——提前完成的请求立刻出批返回，新到的请求最快下一个迭代就能上车。配套的 selective batching 只对没有参数复用损失的算子做批量计算，其余逐请求执行。这套组合在 GPT-3 175B 上做到**同延迟水平**下吞吐是 FasterTransformer 的 **36.9×**（median 归一化延迟 190ms 时 6.81 req/s vs 0.185 req/s），引擎微基准下最多快 **47%**，论文总结为“一个数量级”的吞吐提升。今天 vLLM 的 continuous batching、chunked prefill 都从这棵树上长出来——但注意，论文原文并不使用这些词，它说的是 iteration-level scheduling 与 initiation/increment phase。本文先把直觉建立起来，再拆机制、算账、对照 2026 年的工程现状。

<!-- more -->

## 📑 目录

- [🗺️ 原文阅读地图](#️-原文阅读地图)
- [0. 读前 3 分钟：先建立三组直觉](#0-读前-3-分钟先建立三组直觉)
- [1. 问题：请求级调度让提前完成的请求空等](#1-问题请求级调度让提前完成的请求空等)
- [2. 核心方案:iteration-level scheduling 与 ORCA 架构](#2-核心方案iteration-level-scheduling-与-orca-架构)
- [3. Selective batching:不是所有算子都值得批](#3-selective-batching不是所有算子都值得批)
- [4. 分布式执行：模型切分、控制面与数据面、pipeline](#4-分布式执行模型切分控制面与数据面pipeline)
- [5. 评测:36.9 倍的账怎么算](#5-评测369-倍的账怎么算)
- [6. 从论文到当前工程:continuous batching 的承继](#6-从论文到当前工程continuous-batching-的承继)
- [7. 局限与选型边界](#7-局限与选型边界)
- [8. 面试官视角：三问三答](#8-面试官视角三问三答)
- [📝 总结](#-总结)
- [🎯 自我检验清单](#-自我检验清单)
- [📚 参考资料](#-参考资料)

## 🗺️ 原文阅读地图

这篇论文同时讲调度算法、批处理策略和分布式系统架构，三块各对应一个核心机制。本文选择性精讲如下，避免把一篇中文解读误当成原文逐段翻译。

**页码约定**：锚点使用论文印刷页码(pp. 521–538)，换算关系为**印刷页码 = PDF 页码 + 519**（如 PDF 第 13 页 = p.532）。

| 原文单元 | 处理深度 | 本文位置与理由 | 来源锚点 |
| --- | --- | --- | --- |
| §2 自回归背景、initiation/increment 两阶段 | 简述 | 第 0 节，只保留两阶段直觉与 KV 复用 | §2,p.522–523;Figure 1 |
| §3 C1：请求级调度的低效 | 精讲 | 第 1 节，含 Figure 3 完整走查 | §3,p.524–525;Figure 2–3 |
| §3 S1:iteration-level scheduling | 精讲 | 第 2 节，机制卡 1 + Figure 4 走查 | §3,p.524–525;Figure 4 |
| §3 C2：三种不可批情形 | 精讲 | 第 1、3 节，selective batching 的动机 | §3,p.525–526 |
| §3 S2:selective batching | 精讲 | 第 3 节，机制卡 2 + Figure 5 | §3,p.526–527;Figure 5 |
| §4.1 分布式执行、intra/inter-layer | 精讲 | 第 4 节前半 | §4.1,p.526–528;Figure 6–7 |
| §4.2 Algorithm 1 与 pipeline 执行 | 精讲 | 第 2、4 节，伪代码抄录 | §4.2,p.528–529;Algorithm 1;Figure 8 |
| §5 实现（13K 行 C++、控制/数据面） | 简述 | 第 4、5 节引用，不逐模块展开 | §5,p.529 |
| §6.1 引擎微基准（47%、OOM） | 精讲（数字表） | 第 5 节 | §6.1,p.530–531;Figure 9 |
| §6.2 端到端评测（36.9× 等） | 精讲（数字表） | 第 5 节，含可复算例子 | §6.2,p.531–532;Figure 10–11 |
| §7 Related Work（BatchMaker 等） | 不展开 | 只引用 33–640 一个语境与紧耦合局限 | §7,p.533 |
| §8 Conclusion | 简述 | “一个数量级”表述并入第 5 节 | §8,p.533 |

📌 **本文承诺**：读完后，你应该能画出 Figure 3 的 x1/x2 走查并解释 “−” 的代价，默写 Algorithm 1 的三个关键分支，解释 selective batching 为什么敢不批 Attention,复算 6.81 ÷ 0.185 ≈ 36.9,并且严格分清哪些是论文原话、哪些是后来 vLLM 等系统补的术语。

## 0. 读前 3 分钟：先建立三组直觉

### 0.1 直觉一：自回归生成是“一次 initiation + N 次 increment”的迭代长跑

一个生成式请求在引擎里的生命周期，论文把它切成两个阶段(§2,p.522–523;Figure 1)：

- **initiation phase（起算阶段）**：一次迭代并行处理全部输入 token,产出**第一个**输出 token;
- **increment phase（增量阶段）**：之后每次迭代只吃一个 token,生成下一个 token,同时复用历史 key/value（论文用 fairseq 式 incremental decoding,§3,p.525）。

所以“生成 100 个 token”在引擎眼里是 1 次 initiation + 99 次 increment,一共 100 次迭代。每次迭代都是一次完整的 Transformer 前向，只是输入形状不同：initiation 吃 `[输入长度, H]`,increment 吃 `[1, H]`。**每个请求在引擎里待的时间天然长短不一**——这是全文所有问题的物理根源。

> 打个比方：initiation 像老师先把整段题目读一遍，增量阶段像每读一个字就写一个字。不同学生（请求）的题目长短不同、写字速度不同，有的两分钟交卷，有的要写十分钟。

**术语对账表（先记下来，后面不重复解释）**：论文原文只使用 inititation/increment phase 两个词，不使用 "prefill/decode";只使用 iteration-level scheduling,不使用 "continuous batching"。下面这些是**后续系统（vLLM 等）的术语**，不是论文原话，精读时不要倒灌：

| 论文原文术语 | 后续系统术语 | 首次出现处 |
| --- | --- | --- |
| initiation phase | prefill（预填充） | 论文 §2,p.522 |
| increment phase | decode（解码） | 论文 §2,p.522–523 |
| iteration-level scheduling | continuous batching（连续批处理） | 论文 §3,p.524–525;§4.2,p.528 |

### 0.2 直觉二：调度粒度决定 GPU 的“忙碌质量”

同样一张 GPU,调度粒度不同，干的活差很多。请求级调度(request-level scheduling)像**包车一日游**：一车人起点出发，必须陪最慢的乘客走完全程，提前到站的人也得在车上耗着，中途不能上客。迭代级调度像**地铁**：每站（每次迭代）到点，到站的乘客（完成请求）下车，站台上等的人（新请求）立刻上车——**下一站跑谁，重新决定**。

论文的关键表述是：迭代级调度下，“每次迭代，调度器完全控制跑哪些请求、跑几个”(§3,p.525)。这一句就是 Orca 的全部灵魂。

### 0.3 直觉三：先记住三个数字和一条纪律

- **36.9×**：175B 模型、同延迟水平下，ORCA 吞吐 vs FasterTransformer(Abstract,p.521;§6.2,p.532);
- **up to 47%**：引擎微基准（没有调度器）、175B/16 GPU、两系统都关 pipeline 时，ORCA engine 比 FasterTransformer 快最多 47%(§6.1,p.531);
- **一个数量级**：§8 结论对 175B/341B 多 inter-partition 场景的概括(§6.2,p.532;§8,p.533)。

纪律：本文所有数字只来自证据账本与原文锚点；论文没给的数字（比如 GPU 利用率具体百分比，原文只有定性描述，§3,p.524 与 §5,p.529）一概不编造。

## 1. 问题：请求级调度让提前完成的请求空等

### 1.1 Figure 3 走查：论文的驱动例子

现有 serving 系统（Triton 等 serving 层 + FasterTransformer 等引擎，§3,p.524）只在两个时点交互：调度器向空闲引擎提交下一批请求、引擎跑完当前批后返回。引擎一旦接单，**batch 就固定到整批全部完成**——这就是请求级调度。

论文 Figure 3 给了个具体例子(§3,p.524–525)：两个请求 x1("I think")、x2("I love") 输入长度相同（各 2 个 token），组成一个 batch：

- **iter 1**：两请求的 initiation,各并行处理 2 个输入 token;
- **iter 2**：x1 生成 "this",x2 生成 "you"——**x2 在此刻已经完成**（它只差一个输出 token）；
- **iter 3、iter 4**：x1 继续生成 "is"、"great",但引擎仍为 x2 执行计算——x2 的输入和输出都是 “−”，论文标注为**额外计算(extra computation)**。

<img src="/AIInfraGuide/images/orca-fig3-iteration-vs-request-scheduling.png" alt="请求级调度下两个请求 x1、x2 从 iter 1 到 iter 4 的 token 流：x1 生成 this is great,x2 在 iter 2 已生成 you 完成，但 iter 3、4 仍被计算，格子标记为 - 表示额外计算" style="max-width: 95%; display: block; margin: 0 auto;" />

*图源：Orca 论文 Figure 3(OSDI '22,pp. 521–538)*

两个代价(§3,p.524)：

1. **算力浪费**：x2 在 iter 3、4 的两次迭代是纯浪费——“Such extra computation for inactive requests (x2 at iter 3 and 4) limits the efficiency of batched execution”；
2. **延迟放大**：引擎只在整批结束时统一返回，所以 x2 明明 iter 2 就完成了，响应却拖到 iter 4 才回到客户端；反过来，iter 2 中途到达的新请求，也要等整批结束才能被调度。

> 还是包车比喻：x1 和 x2 拼同一辆车，车必须开到 x1 的终点才放人，x2 到站后还得在车上干坐两站，而站台的新乘客只能等这趟车结束。

### 1.2 为什么“提前完成/晚加入”在自回归负载里是常态

因为每个请求需要的迭代次数 = 1 + 生成 token 数，天然长短不一；而且生成长度在调度时**未知**（请求到达时你只知道 max_gen_tokens,不知道实际会停在哪）。负载越异构，请求级调度的浪费越大。论文的端到端实验用输入 token 数 ~ U(32,512)、max_gen_tokens ~ U(1,128) 的合成 trace 模拟这种异构性(§6.2,p.531)：最轻的请求只跑 1 次迭代，最重的要跑 128 次，绑在一辆车上的代价可想而知。

### 1.3 C2：就算换了粒度，batch 也批不起来

把粒度降下来之后，立刻撞上第二个问题：调度器每轮选出的请求，形状五花八门，没法整体批处理。论文列出**三种不可批的情形**(§3,p.525–526)：

1. **两请求都在 initiation phase,但输入 token 数不同**（如 Figure 4 中 x3 有 2 个输入 token、x4 有 3 个）；
2. **两请求都在 increment phase,但当前 token index 不同**（如 x1、x2）；
3. **两请求处于不同 phase**（一个 initiation、一个 increment）。

原因是：批处理要求所有请求的算子序列一致、输入张量形状一致；上述三种情形输入形状天然不同。相关工作的 BatchMaker 正是栽在这里：请求 token 总数 L 从 33（32 输入 + 1）到 640（512 输入 + 128 生成）不等，每个 Transformer cell 的形状都不同，无法批量(§7,p.533)。这是 C2,对应解法就是第 3 节的 selective batching。

## 2. 核心方案：iteration-level scheduling 与 ORCA 架构

### 2.1 每迭代三件事

ORCA 把 serving 系统与引擎的交互从“批粒度”降到“迭代粒度”(§3,p.524–525;Figure 4)。每个迭代循环做三件事：

1. **选**：调度器从请求池中选择本轮要跑的请求；
2. **跑**：引擎对这批请求只执行**一次**模型迭代；
3. **收**：调度器收回该迭代产出的输出 token。

完成请求被请求池移除、端点立刻发响应；新请求在下一迭代即可被选中。调度器从“发号施令后不管”变成“每步都在场”。

<img src="/AIInfraGuide/images/orca-fig4-system-overview.png" alt="ORCA 系统总览：Endpoint 到 Scheduler,调度器与 Request Pool 双向交互、与 Execution Engine 双向交互，四个请求 x1 到 x4 各有一条 token 序列，每迭代返回一个输出 token" style="max-width: 95%; display: block; margin: 0 auto;" />

*图源：Orca 论文 Figure 4(OSDI '22,pp. 521–538)*

沿 Figure 4 走一轮(§3,p.525)：调度器从请求池选中 (x1,x2,x3,x4);x3、x4 是首次被调度，调度器把它们的输入 token（x3 两个、x4 三个）交给引擎；引擎对这 4 个请求跑**一次**迭代，返回每个请求各一个输出 token（x15、x23、x33、x44）；请求池移除已完成的请求并通知端点返回。对比 Figure 2 的旧流程——“跑完一批的所有迭代才返回”——区别就在“可以每迭代改变请求集合”。

**机制卡 1:iteration-level scheduling（含 Algorithm 1）**

| 项目 | 内容 |
| --- | --- |
| 要解决什么 | 请求级调度下，提前完成的请求被捆绑计算（算力浪费）+ 延迟返回，新请求无法及时入批(C1) |
| 最小前置 | 理解自回归 = 1 次 initiation + N 次 increment;引擎接口支持迭代粒度驱动 |
| 输入 → 状态 → 输出 | 输入：请求池（每个请求带到达时间、阶段状态 INITIATION/RUNNING/INCREMENT、max_tokens）；状态：n_scheduled、n_rsrv（K/V 预留计数）；输出：每轮 batch 选择与引擎迭代调用 |
| 因果步骤 | ① 调度器按到达时间排序请求池(iteration-level FCFS)→ ② 依序选入 batch,直到 max_bs 或内存约束 → ③ 引擎对 batch 执行一次迭代 → ④ 返回请求标记 INCREMENT,完成的请求释放预留 → ⑤ 攒满 n_workers 个在途 batch 才等待返回（见第 4 节） |
| 公式语义 / 伪代码 | Algorithm 1(§4.2,p.528)，三个关键分支：Select 按到达时间排序；INITIATION 请求按 max_tokens 预留 K/V 槽位（超出 n_slots 即停）；`n_scheduled = n_workers` 才 wait |
| 最小例子 | Figure 3 走查：x2 本可 iter 2 完成，请求级调度把它拖到 iter 4;迭代级调度下 iter 2 结束即刻返回 |
| 边界与来源 | max_bs 需要人工权衡，增大未必不损延迟(§4.2,p.528);FCFS 保证公平；完成判定依赖请求产出 <EOS> 或达到 max_gen_tokens（§6,p.530 假设无 <EOS> 的合成负载） |

### 2.2 Algorithm 1：调度器每轮做什么

论文 Algorithm 1(§4.2,p.528)是核心伪代码，参数三个：`n_workers`（worker 数）、`max_bs`（最大 batch）、`n_slots`（K/V 槽位数）。主干如下（注释为本文所加）：

```
n_scheduled ← 0; n_rsrv ← 0
while true:
    batch, n_rsrv ← Select(request_pool, n_rsrv)      # 每迭代重新选一批
    调度 engine 对 batch 执行模型的一次迭代
    for req in batch:
        req.state ← RUNNING; n_scheduled += 1
        if n_scheduled = n_workers:                   # pipeline 关键:攒满 n_workers 批才等返回
            wait for return of a scheduled batch
            for req in returned batch:
                req.state ← INCREMENT
                if finished(req): n_rsrv -= req.max_tokens; n_scheduled -= 1

def Select(pool, n_rsrv):
    batch ← {}
    pool ← {req ∈ pool | req.state ≠ RUNNING}
    SortByArrivalTime(pool)                           # iteration-level FCFS
    for req in pool:
        if batch.size() = max_bs: break
        if req.state = INITIATION:                    # 首次调度的请求预留 K/V
            new_n_rsrv ← n_rsrv + req.max_tokens
            if new_n_rsrv > n_slots: break            # 内存约束
            n_rsrv ← new_n_rsrv
        batch ← batch ∪ {req}
    return batch, n_rsrv
```

三个值得记住的机制点（均锚定 p.528）：

1. **iteration-level FCFS**：按到达时间排序，防止饥饿，与后来 vLLM 的默认 `--scheduling-policy fcfs` 同一选择；
2. **K/V 槽位预留**：INITIATION 请求按 `max_tokens`（每请求属性，论文脚注 6）预占槽位，一个槽位 = 单个 token 的 Attention key+value 内存；预留不足 n_slots 才入批——这是内存约束的直接表达；
3. **n_slots vs max_bs 的分工**：n_slots 由运维配置为内存允许的最大值，**不需要实验调参**；而 max_bs 需要按硬件/模型/负载权衡（§4.2,p.528;§6.2,p.532 的实验都遍历了 max_bs）。

### 2.3 K/V 预留：比“按最大长度预分配”省在哪

对比 FasterTransformer 的做法：它按**最大序列长度(2048)**为每个请求预分配 K/V 内存，固定预分配导致 13B 模型 batch ≥ 8、101B 模型 batch ≥ 16 即 OOM（§6.1,p.530;Figure 9 中 FasterTransformer 曲线缺失项的原因）。ORCA 按每请求 max_tokens 单独分配，避免这种冗余——论文没有给出显存节省的具体百分比（原文未提供绝对值）。

> **最小数字例子（本文自设例子，用于感受量级）**：假设某请求实际只用了 256 个 token 的 K/V,而系统按 2048 预分配，那么它拿到的槽位里有 (2048−256)/2048 ≈ 87.5% 从未被写入。请求实际用量越短，固定预分配越浪费。注意：这里的 87.5% 是根据论文“按 2048 预分配 vs 按 max_tokens 分配”的机制推出来的示例算术，不是论文报告的数字。

### 2.4 最小数字例子：三个请求的时间线对比（本文自设例子）

设三个请求：A 需要 2 次迭代完成、B 需要 4 次、C 需要 6 次（数字为本文自设，仅示意）。请求级调度把三者绑成一车：

- 总计算量 = 3 请求 × 6 迭代 = **18 个“请求-迭代”单位**;
- 有用计算 = 2 + 4 + 6 = 12 个单位；
- 浪费 = 6 个单位，占 33%;A 的响应在第 6 迭代才返回。

迭代级调度下，每轮只算活跃请求：

- 总计算量 = 2 + 4 + 6 = **12 个“请求-迭代”单位**，零浪费；
- A 第 2 迭代完成即返回，新请求 E 第 3 迭代即可入批。

同样的 GPU、同样的请求，计算量差 18 vs 12（少 33%），这就是“调度粒度”的杠杆。真实负载里迭代次数由 U(1,128) 决定(§6.2,p.531)，方差更大，差距更悬殊。

## 3. Selective batching：不是所有算子都值得批

### 3.1 三种不可批情形下的“部分批处理”

迭代级调度让引擎每轮面对的形状更乱：同批请求可能跨阶段、不同 token index,整批无法合并成一个大张量跑统一 kernel(§3,p.525–526)。ORCA 的答案是：**只对一部分算子做 batching**(§3 S2,p.526–527)：

- **非 Attention 算子（Linear、LayerNorm、Add、GeLU）**：把输入展平成 2 维 `[ΣL, H]`（例如两个请求的 `[2,H]` 和 `[3,H]` 拼成 `[5,H]`），按 **token 级**批处理——这些算子不区分张量元素属于哪个请求；
- **Attention 算子**：需要请求边界（batch 维），逐请求单独执行，前后各插一个 Split/Merge 算子衔接(Figure 5)。

<img src="/AIInfraGuide/images/orca-fig5-selective-batching.png" alt="selective batching 执行示意：QKV Linear 接收 [7,H] 展平张量输出 [7,3H],Split 拆成各请求的 QKV,逐请求执行 Attention(Attn x1 到 x4,K/V 来自 Attention K/V Manager),Merge 合并后进入 Attn Out Linear" style="max-width: 95%; display: block; margin: 0 auto;" />

*图源：Orca 论文 Figure 5(OSDI '22,pp. 521–538)*

沿 Figure 5 走一遍(§3,p.526–527)：展平张量 `[7,H]` 进 QKV Linear 得 `[7,3H]` → Split 拆成各请求的 `[2,3H]`、`[3,3H]`、`[1,3H]`、`[1,3H]` → 每个请求的 Attention 单独执行，K/V 从 **Attention K/V Manager** 按请求取历史（key/value 由 manager 保存，直到调度器显式删除，p.526）→ Merge 拼回 `[7,H]` → Attn Out Linear。请求 x1 的 K/V 是 (x11,x12,x13),x2 的只有 (x21)，各算各的。

### 3.2 为什么敢不批 Attention：参数复用论

批处理最大的收益之一是**参数复用**：同一组权重一次加载、服务更多数据，摊薄内存带宽成本。但 Attention 恰恰是 Transformer 里**唯一没有可学习参数**的算子（权重都来自 QKV/输出投影）——不批它，不损失任何参数复用收益（§1,p.522;§6.1,p.530 重申）。这是 selective batching 的经济学依据：batch 的形状约束是 Attention 强加的，而 Attention 恰好是最不值得为批处理妥协的算子，于是把它踢出 batch。

实现上还有个工程细节：多个 split 出来的 Attention kernel 的 thread block 被拼接成**单一 kernel** 以减少 launch 开销(§5,p.529)。

**机制卡 2:selective batching**

| 项目 | 内容 |
| --- | --- |
| 要解决什么 | 迭代级调度下，同批请求形状不齐(C2)，整批 kernel 无法执行 |
| 最小前置 | 三种不可批情形；算子是否依赖请求边界的区分 |
| 输入 → 状态 → 输出 | 输入：各请求的 token 张量；状态：Attention K/V manager 中按请求保存的历史 key/value;输出：一次迭代的所有算子输出 |
| 因果步骤 | ① 判断可合并条件（同阶段且同输入长度/同 token index）→ ② 非 Attention 算子把输入展平为 [ΣL,H] 按 token 批处理 → ③ Attention 逐请求执行，前后 Split/Merge → ④ K/V manager 更新本迭代的 key/value |
| 公式语义 / 伪代码 | canonical batching 输入 `[B,L,H]`(p.526);selective batching 非 Attention 输入 `[ΣL,H]`,如 `[2,H]+[3,H] → [5,H]` |
| 最小例子 | 论文 §3(p.526)：x3([2,H])与 x4([3,H])无法拼成 [B,L,H],但可拼成 [5,H] 喂给所有非 Attention 算子 |
| 边界与来源 | 可合并 iff 同 initiation 且输入 token 数相同，或同 increment 且 token index 相同(p.525–526);Attention 无参数、不批不损失复用收益（§1,p.522;§6.1,p.530 重申）；引擎单独执行时 Attention 不批的优势消失，微基准与 FT 相当或略差(§6.1,p.530) |

## 4. 分布式执行：模型切分、控制面与数据面、pipeline

### 4.1 intra/inter-layer 切分：一个模型摊到多张卡

ORCA 的分布式执行沿两个维度切分模型(§4.1,p.526–528;Figure 6)：

- **intra-layer parallelism**：把矩阵乘(Linear/Attention)及其参数按 GPU 切分——训练系统的老手艺（论文引用 [55,58]）；
- **inter-layer parallelism**：按层切分，每张 GPU 分到等量层。

一个 **worker** = 一个 inter-layer partition,可以横跨多台机器；每个 worker 的 controller 管理多个 CPU 线程驱动 GPU(§4.1,p.527)。论文实验里：175B = 2 inter × 8 intra = **16 GPU**,341B = 4 × 8 = **32 GPU**(Table 1 + §6,p.529–530;Table 1 同时给出 13B/101B/175B/341B 的层数 40/80/96/120 与 hidden 5120/10240/12288/15360)。

### 4.2 控制面走 gRPC,数据面走 NCCL

这是 Orca 最容易被低估的工程贡献(§4.1,p.527–528;§5,p.529)：

- **控制消息**（request id、token index、输入长度等）走 **gRPC**：不经过 NCCL、不触发 CPU-GPU 同步；
- **tensor 数据**走 **NCCL**。

对比 FasterTransformer/Megatron：每次收到控制消息都要做 CPU-GPU 同步，控制流量直接卡在 GPU pipeline 上。ORCA 把“决策信息”和“数据”拆到两条管道，调度器每迭代发控制消息不再打扰 GPU 执行。引擎微基准里 ORCA engine 最多比 FasterTransformer 快 **47%**（175B/16 GPU、两系统均关闭 pipeline 时），论文把优势归因于 control/data plane 分离(§6.1,p.531)。整个系统约 **13K 行 C++**，基于 CUDA 生态(§5,p.529)。

### 4.3 Pipeline：让 n_workers 个 batch 同时流水

调度器不等上一批返回就继续注入新 batch——Algorithm 1 里 `n_scheduled = n_workers` 才等待返回，保证引擎里始终有 **n_workers 个 batch 并行流水**(§4.2,p.529;Figure 8)。这是迭代级调度带来的结构性红利：FasterTransformer 因 request-level 接口只能用 microbatch 流水，在 batch 大小与 pipeline bubble 之间做效率权衡(§6.2,p.532 的实验显示其最优配置是 (max_bs, mbs) = (1,1) 或 (8,8))。

> 工厂比喻：调度器像投料工，不等上一批成品下线就继续投料，流水线上始终同时有 n_workers 件产品在加工；请求级系统只能“一批做完再投下一批”，机器必然空转。

**机制卡 3：分布式执行（切分 + 双管道 + pipeline）**

| 项目 | 内容 |
| --- | --- |
| 要解决什么 | 单卡放不下大模型；控制消息同步拖慢 GPU;批间切换空窗 |
| 最小前置 | intra/inter-layer 切分语义；NCCL 集合通信；迭代级调度接口 |
| 输入 → 状态 → 输出 | 输入：模型配置与请求流；状态：n_scheduled 在途批数、各 worker 的 GPU 执行状态；输出：并行化的迭代执行 |
| 因果步骤 | ① intra-layer 切分矩阵乘 → ② inter-layer 切分层 → ③ 控制消息 gRPC 直达、tensor 走 NCCL → ④ 调度器持续注入 batch,攒满 n_workers 才等返回 → ⑤ 返回批完成、资源释放 |
| 公式语义 / 伪代码 | Algorithm 1 line 9–10：`n_scheduled = n_workers` 时 wait(§4.2,p.529) |
| 最小例子 | Figure 6:4 层模型 = 2 inter × 3 intra = 6 GPU(p.526–527);175B = 16 GPU、341B = 32 GPU(Table 1,p.529) |
| 边界与来源 | 47% 优势来自引擎微基准且双方关闭 pipeline(§6.1,p.531)；控制/数据面分离是相对 FT/Megatron 的对比结论(§4.1,p.527–528) |

## 5. 评测：36.9 倍的账怎么算

### 5.1 实验设置先对齐

- **硬件**：8×40GB A100（Azure ND96asr A100 v4 VM,NVLink 互联），最多 4 台 VM;每 VM 8 个 Mellanox 200Gbps HDR InfiniBand 适配器，VM 间合计 1.6Tb/s(§6,p.530);
- **模型**：13B/101B/175B/341B,最大序列长度统一 2048,沿用 GPT-3 论文设定；fp16 参数与中间激活(§6,p.530;Table 1,p.529);
- **负载**：端到端合成 trace,输入 token 数 ~ U(32,512),max_gen_tokens ~ U(1,128)（最轻 1 次迭代、最重 128 次；§6.2,p.531），到达时间按 Poisson 过程(§6,p.530)；引擎微基准用所有请求同输入长、同输出长（32 或 128 输入 token、生成 32 token），排除调度因素(§6.1,p.530–531);
- **基线**：FasterTransformer 遍历所有 (max_bs, mbs) 组合取最优配置，最终是 (1,1) 或 (8,8)(§6.2,p.532)。

### 5.2 核心数字表（全部锚定原文）

| 数字 | 口径 | 锚点 |
| --- | --- | --- |
| **36.9×** | 175B、同延迟水平，ORCA vs FasterTransformer | Abstract,p.521;§6.2,p.532 |
| **6.81 vs 0.185 req/s** | 175B、median 归一化延迟 = 190ms:ORCA(max_bs=128) 6.81,FasterTransformer 0.185 | §6.2,p.532 |
| **190ms** | 对比点，按生成 token 数归一的中位延迟 | §6.2,p.532 |
| **0.49 req/s** | 101B、异构 trace 下 FasterTransformer 的峰值吞吐（ORCA 显著更高） | §6.2,p.532 |
| **up to 47%** | 引擎微基准、175B/16 GPU、双方关 pipeline,ORCA engine 更快 | §6.1,p.531 |
| **一个数量级** | §8 对多 inter-partition 场景的结论表述 | §6.2,p.532;§8,p.533 |
| **batch≥8(13B)/≥16(101B) 即 OOM** | FasterTransformer 按 2048 固定预分配 K/V,大 batch 直接爆显存（Figure 9 缺失项） | §6.1,p.530 |
| **33 ↔ 640** | 请求 token 总数范围（32+1 到 512+128），BatchMaker 无法批量的原因语境 | §7,p.533 |

### 5.3 最小数字例子：复算 36.9×

这是全文最值得亲手按一次的计算器：

$$
\frac{6.81\ \text{req/s}}{0.185\ \text{req/s}} = 36.81 \approx 36.9×
$$

两个数字都来自 §6.2(p.532) 的同一对比点：175B 模型、median 归一化延迟 190ms。**36.9× 是“同延迟水平下的吞吐比”，不是延迟快 36.9 倍**——横着读是“ORCA 在 190ms 延迟点能扛住 6.81 req/s,而 FasterTransformer 只扛得住 0.185 req/s”，竖着读才是“FasterTransformer 想达到 6.81 req/s,延迟会飙到哪里去”（Figure 10 的曲线形状）。

### 5.4 Figure 10：端到端延迟-吞吐曲线

<img src="/AIInfraGuide/images/orca-fig10-throughput-latency.png" alt="端到端 median 归一化延迟 vs 吞吐对比曲线：(a) 101B/8GPU (b) 175B/16GPU (c) 341B/32GPU;图例 ft(1,1)、ft(8,8) 与 orca(1/8/16/32),ORCA 曲线整体位于左下方" style="max-width: 98%; display: block; margin: 0 auto;" />

*图源：Orca 论文 Figure 10(OSDI '22,pp. 521–538);36.9× 结论的出处图*

三个面板分别对应 101B/8 GPU、175B/16 GPU、341B/32 GPU(§6.2,p.531–532)，横轴吞吐、纵轴 median 归一化延迟，图例是 ft(1,1)、ft(8,8) 与 orca(1/8/16/32)。读图要点：**除 101B 低负载外，ORCA 曲线位于左下方——同延迟下吞吐更高，同吞吐下延迟更低**（101B 低负载例外，§6.2,p.531），而且是双优，不是拿延迟换吞吐。175B/341B（多于一个 inter-layer partition）上，ORCA 在所有负载水平下延迟与吞吐双优，同延迟下吞吐高一个数量级(§6.2,p.532;§8,p.533)。

**口径纪律，三条：**

1. **36.9× 是 175B 的端到端结论**，不能外推到 101B/341B（论文只给了这一个倍率数字）；
2. **up to 47% 是引擎微基准**（没有调度器、双方关 pipeline），不能与端到端 36.9× 混用；
3. **没有 TurboTransformers/LightSeq 对比数字**——它们只在 Related Work 被列为“行为类似的后端引擎”(§7,p.533)，无实验对比；GPU 利用率具体百分比原文也未提供（仅定性，§3,p.524、§5,p.529）。

## 6. 从论文到当前工程：continuous batching 的承继

> 本节工程事实访问日期：**2026-08-11**。论文是 OSDI 2022 的产物；下面所有“当前工程”的描述都基于当时可访问的官方文档/源码，不把后来的机制写进论文原话。

### 6.1 官方实现：未开源

论文 ORCA（13K 行 C++）**没有公开开源**。GitHub 上存在 github.com/microsoft/Orca,但它是 2015 年创建的 Java "orchestration engine"（Spinnaker 系），2023-06-13 后已 archived,**与论文无关**（访问日期 2026-08-11）。所以“读源码验证实现细节”这条路走不通，这也是后来 vLLM 论文里 Orca baseline 只能由作者自实现的原因。

### 6.2 vLLM：同一思想的“标准化封装”

vLLM(2023)把 Orca 的 iteration-level scheduling 以 **continuous batching** 的名字工程化（官方博客 "vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention" 与 vLLM 论文 arXiv:2309.06180;访问日期 2026-08-11)：调度器以迭代粒度调度、新请求即刻入批、完成请求即刻出批——与 Orca 的机制同源。vLLM 做对的两件事是：

1. **接上 PagedAttention**：Orca 的 K/V 预留仍是“每请求一段连续槽位”（n_slots 池），vLLM 改成按固定 token 数分块、block table 映射、按需分配，把显存碎片降到 <4%（对比传统预留 60%–80% 浪费）——这补上了 Orca 留下的内存管理短板；
2. **暴露调度参数**：`--max-num-batched-tokens`（单次迭代最多 token 数）、`--max-num-seqs`（单次迭代最多序列数）、`--scheduling-policy`（默认 fcfs,与 Orca 的 iteration-level FCFS 一致）、`--watermark`（KV 空闲块水位，防抖动），以及 **`--enable-chunked-prefill`**——把长 initiation(prefill)拆块与 increment(decode)混排，这是 Orca 没有的机制（vLLM 官方文档，访问日期 2026-08-11）。

### 6.3 SGLang：调度之上的前缀复用与扩展

SGLang 的调度同样属于 continuous batching 系（迭代级），核心差异化在 **RadixAttention**（前缀缓存复用）与 PD 分离、DP attention 等大规模扩展（官方文档，访问日期 2026-08-11）。相关参数：`--chunked-prefill-size`（chunked prefill 每块最大 token 数）、`--max-running-requests`、`--prefill-max-requests`/`--max-prefill-tokens`（initiation batch 上限）、`--schedule-policy`/`--enable-priority-scheduling`、`--radix-eviction-policy`（radix 树 LRU/LFU 逐出）。官方文档中没有与 Orca 的对比实验。

### 6.4 差异要点：Orca(OSDI '22)vs 当前主流引擎

| 维度 | Orca | vLLM / SGLang（2026-08-11 访问） |
| --- | --- | --- |
| 调度粒度 | iteration-level scheduling(§4.2,p.528) | continuous batching：同为迭代级；vLLM 默认 FCFS 与 Orca 一致 |
| initiation/increment | 两阶段，increment 每迭代 1 token(§2,p.522) | 同样区分 prefill/decode,并支持 **chunked prefill**（Orca 没有） |
| K/V cache 管理 | 按请求预分配 max_tokens 槽位（n_slots 池，Algorithm 1） | vLLM 分块 + block table 按需分配（碎片 <4%）；SGLang radix 树前缀复用 |
| Attention 是否批处理 | **不批**（selective batching,逐请求执行） | 批处理（FlashAttention/PagedAttention 等支持 ragged/batched attention），无 Split/Merge |
| 分布式 | intra+inter layer + 控制/数据面分离 + 多 batch 流水 | vLLM TP/PP/DP/EP;SGLang DP/PD/EPD;控制面与数据面同样分离（多进程架构） |
| 目标规模 | 最大测到 341B(Table 1,p.529) | 生态支持至万亿参数级，但论文层不做此对比 |

## 7. 局限与选型边界

### 7.1 论文自己声明的七条局限

1. **接口紧耦合**：ORCA 把 scheduler 与 engine 紧耦合以实现两种技术，“没有研究保持抽象分离的通用接口设计”，留给未来工作(§7,p.533);
2. **只在语言模型上实验**：虽声称其他生成域受益，但"conduct experiments only on language models"(§1 Introduction,p.522);
3. **合成 trace 与无 <EOS> 假设**：没有公开生成式请求 trace,自行按 Poisson 合成；没有真实 checkpoint 与输入文本，假设请求一律生成满 max_gen_tokens、模型永不出 <EOS>(§6,p.530);
4. **引擎微基准下无优势**：没有调度器时，ORCA engine 与 FasterTransformer 相当或略差（Attention 不批处理所致）(§6.1,p.530);
5. **101B 低负载不占优**：低负载下双方都没有足够请求组 batch,ORCA 无优势(§6.2,p.531);
6. **341B 微基准省略**：与 175B 结果类似故省略(§6.1,p.531);
7. **max_bs 需人工权衡**：增大 max_bs 未必不损延迟，需按硬件/模型/负载调参(§4.2,p.528;§6.2,p.532)。

### 7.2 牺牲什么，换取什么

| 换取 | 牺牲 |
| --- | --- |
| 同延迟下 36.9× 吞吐(175B,§6.2,p.532) | scheduler 与 engine 紧耦合，接口不可复用、难以独立演进(§7,p.533) |
| 提前完成请求即刻返回、新请求最快下迭代入批(§3,p.524–525) | 调度器每迭代介入，控制面成为关键路径——必须靠 gRPC/NCCL 双管道把开销压住(§4.1,p.527–528) |
| 按 max_tokens 预留，避免固定预分配 OOM（batch≥8 对 ≥16 的对比，§6.1,p.530） | 预留粒度仍是“每请求一段连续槽位”，比后来 PagedAttention 的分块按需分配粗；过预留依旧存在 |
| Attention 不批换来的实现简洁与可拼装 kernel(§5,p.529) | 引擎单独执行时 Attention 无批处理优势，微基准与 FT 相当或略差(§6.1,p.530) |
| 多 batch 流水消除批间空窗(§4.2,p.529) | 引擎内同时跑 n_workers 个 batch,显存占用更高，需要 n_slots 运维配置 |

**一句话选型规则**：如果你在做在线 LLM serving、负载天然长短不一（聊天、代码补全），iteration-level scheduling 不是“可选项”而是“及格线”——现代引擎(vLLM/SGLang)已把它做成默认；如果你要在它之上继续优化，论文留下的两个开放点正好是后来者的入场券：**K/V 内存管理（PagedAttention 的领域）** 与 **长 initiation 的切块（chunked prefill 的领域）**。

## 8. 面试官视角：三问三答

**Q1：“iteration-level scheduling 和 request-level scheduling 到底差在哪？”**

答：差在调度粒度与交互频次。请求级调度下，引擎接单后 batch 固定到整批跑完，提前完成的请求被捆绑计算、延迟返回，新请求等整批结束才能入批（论文 Figure 3,§3,p.524）。迭代级调度下，调度器每迭代重新决定跑谁：选请求 → 引擎跑一次迭代 → 收回输出 token,完成即出批、新请求最快下迭代入批(Figure 4)。可以补一句论文的量化结论：这个改动在 175B 上换到同延迟下 36.9× 的吞吐(§6.2,p.532)。再补一句承继：vLLM 的 continuous batching 就是同一个思想，只是换了名字。

**Q2：“selective batching 为什么敢不批 Attention？”**

答：因为批处理的核心收益是参数复用——同一组权重服务更多 token、摊薄访存成本；而 Attention 是 Transformer 里唯一没有可学习参数的算子，不批它不损失任何参数复用收益（§1,p.522;§6.1,p.530 重申）。具体做法：非 Attention 算子(Linear/LayerNorm/Add/GeLU)把输入展平成 [ΣL,H] 按 token 级批处理，Attention 逐请求执行、Split/Merge 衔接(Figure 5,§3,p.526–527)。代价是引擎单独执行时 Attention 没有批处理优势，微基准与 FasterTransformer 相当或略差(§6.1,p.530)——所以这套机制必须和迭代级调度配套才划算。

**Q3：“Orca 和 vLLM 是什么关系？”**

答：思想承继、实现换代。Orca(OSDI 2022)首次提出 iteration-level scheduling,这是 vLLM continuous batching 的同源机制；但注意术语纪律：论文原文不用 "continuous batching"、也不用 "prefill/decode",它说 iteration-level scheduling 与 initiation/increment phase,这些词是 vLLM 等后来系统标准化出来的。实现上 vLLM 补了两块 Orca 没有的：一是 PagedAttention 把 K/V 从“按请求预留连续槽位”改成“分块按需分配”；二是 chunked prefill 把长 initiation 拆块与 increment 混排。另外可以提一句：Orca 官方实现没有开源，github.com/microsoft/Orca 与论文无关，这也是后来者只能自实现 baseline 的原因（访问日期 2026-08-11）。

## 📝 总结

1. **问题定位**：请求级调度下，引擎 batch 固定到整批完成，提前完成的请求被捆绑计算、延迟返回，新请求无法及时入批——浪费算力且放大延迟(§3,p.524–525)。
2. **核心方案**：iteration-level scheduling——调度器每迭代做三件事：选请求、跑一次迭代、收输出 token;完成即出批、新请求最快下迭代入批(§3 S1;p.528 Algorithm 1)。
3. **调度算法**：按到达时间 FCFS;INITIATION 请求按 max_tokens 预留 K/V 槽位，n_slots 为内存约束；n_scheduled = n_workers 才等返回，形成多 batch pipeline(§4.2,p.528–529)。
4. **Selective batching**：非 Attention 算子把输入展平为 [ΣL,H] 按 token 批处理，Attention 逐请求执行、Split/Merge 衔接；依据是 Attention 无参数、不批不损失参数复用收益(§3 S2,p.526–527)。
5. **分布式**：intra/inter-layer 双维切分（175B=16 GPU、341B=32 GPU）+ 控制面 gRPC/数据面 NCCL 分离 + 13K 行 C++ 实现(§4.1–4.2,p.526–529)。
6. **量化结论**：36.9×(6.81 vs 0.185 req/s @ median 190ms,175B)、up to 47%（引擎微基准）、一个数量级(§6–8,p.521–533)。
7. **口径纪律**：36.9× 是“同延迟下的吞吐比”不是延迟倍数；47% 是关掉调度器的引擎对比；GPU 利用率百分比原文未提供；无 TurboTransformers 对比数字。
8. **诚实边界**：紧耦合接口、只在语言模型上实验、合成 trace 无 <EOS>、低负载无优势、max_bs 需人工调参(§6–7)。
9. **工程承继**：vLLM 的 continuous batching 即同源思想，并补上 PagedAttention（碎片 <4%）与 chunked prefill;SGLang 加前缀缓存与 PD/DP 扩展（访问日期 2026-08-11）。
10. **术语纪律**：initiation/increment、iteration-level scheduling 是论文原话；prefill/decode、continuous batching 是后续系统术语，精读时不倒灌。

## 🎯 自我检验清单

- 能画出 Figure 3 的 x1/x2 走查：为什么 x2 在 iter 2 已完成仍被算到 iter 4,“−”代表什么，两个代价分别是什么。
- 能用“包车 vs 地铁”的比喻解释 request-level 与 iteration-level scheduling,并说出每迭代三件事。
- 能默写 Algorithm 1 的三个关键分支：按到达时间排序、INITIATION 按 max_tokens 预留、n_scheduled = n_workers 才等待返回。
- 能解释 n_slots 与 max_bs 的分工：前者是内存上限、配置后无需调参，后者需按负载权衡。
- 能列出三种不可批情形，并说明 selective batching 对每类算子的处理（[ΣL,H] 展平 vs 逐请求 Attention）。
- 能解释“Attention 没有参数所以可以不批”的经济学，并说明其代价（引擎单独执行时无优势）。
- 能复算 6.81 ÷ 0.185 ≈ 36.9×,并说清比较口径（175B、同延迟、median 归一化 190ms）。
- 能区分 36.9×（端到端、175B）、up to 47%（引擎微基准、关 pipeline）、一个数量级（§8 概括）三个数字的适用场景。
- 能解释 13B batch≥8、101B batch≥16 即 OOM 的原因（FasterTransformer 按 2048 固定预分配）。
- 能指出论文原文不使用 "continuous batching"/"prefill/decode",并给出对应原文术语（iteration-level scheduling、initiation/increment phase）。
- 能说出 Orca 官方实现未开源、github.com/microsoft/Orca 与论文无关，以及这对后来系统意味着什么。
- 能列出 vLLM/SGLang 相对 Orca 补上的两块机制（chunked prefill、PagedAttention/radix 前缀复用），并说明差异要点表中至少三个维度。

## 📚 参考资料

- 原文：
  - [Orca: A Distributed Serving System for Transformer-Based Generative Models(USENIX 页面)](https://www.usenix.org/conference/osdi22/presentation/yu)：本文精读对象，OSDI 2022,pp. 521–538。
  - [OSDI '22 PDF(usenix.org/system/files/osdi22-yu.pdf)](https://www.usenix.org/system/files/osdi22-yu.pdf)：原文 PDF,本文所有锚点按“印刷页码 = PDF 页码 + 519”换算。
- 当前工程：
  - [vLLM 官方博客:Continuous Batching 与 PagedAttention(blog.vllm.ai,2023-06-20)](https://blog.vllm.ai/2023/06/20/vllm.html)：iteration-level scheduling 思想的工程化表述；vLLM 论文 arXiv:2309.06180。
  - [vLLM Engine Arguments 文档(docs.vllm.ai)](https://docs.vllm.ai/en/latest/configuration/engine_args/)：--max-num-batched-tokens、--enable-chunked-prefill、--scheduling-policy 等参数（访问日期 2026-08-11）。
  - [SGLang Server Arguments 文档(docs.sglang.ai)](https://docs.sglang.ai/docs/advanced_features/server_arguments.md)：--chunked-prefill-size、--radix-eviction-policy、PD disaggregation 等（访问日期 2026-08-11）。
  - [github.com/microsoft/Orca](https://github.com/microsoft/Orca)：同名仓库，2015 年创建的 Java orchestration engine,已 archived,**与论文无关**（访问日期 2026-08-11）。
- 站内相关：
  - [2.2 Continuous Batching 连续批处理](/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/22-continuous-batching)：iteration-level scheduling 的现代工程形态，本文第 6 节的展开版。
  - [2.1 PagedAttention 分页注意力](/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/21-pagedattention)：补上 Orca 留下的 K/V 内存管理短板的那块拼图。
  - [2.4 Chunked Prefill 与统一调度](/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/24-chunked-prefill-与统一调度)：Orca 没有、vLLM/SGLang 补上的长 initiation 切块机制。
  - [1.1 LLM 推理基础](/AIInfraGuide/inference/模块四-推理优化/第1章-llm推理基础/11-llm推理基础)：自回归生成、KV cache 与两阶段(initiation/increment)的入门前置。
  - [9.1 推理指标体系](/AIInfraGuide/inference/模块四-推理优化/第9章-性能分析与benchmark/91-推理指标体系)：理解 36.9×、6.81 req/s、median 归一化延迟这些口径的度量学基础。
