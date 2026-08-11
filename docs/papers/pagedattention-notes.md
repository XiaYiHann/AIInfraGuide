---
title: "PagedAttention 精读:像操作系统分页一样管理 KV Cache,把显存利用率从 20.4% 拉到 96.3%"
description: "拆解 vLLM 的 PagedAttention:分页式 KV cache 管理如何把 token states 占比从 20.4%–38.2% 提到 96.3%,同等延迟下吞吐提升 2-4×(峰值 22×),并手算 800 KB/token 的显存账与当前 V1 引擎的工程演进。"
pubDate: 2026-08-11
originalUrl: "https://arxiv.org/abs/2309.06180"
sourceType: "paper"
originalAuthor: "Woosuk Kwon, Zhuohan Li et al. (UC Berkeley, Stanford)"
tags: ["PagedAttention", "vLLM", "KV Cache", "推理引擎", "显存管理", "LLM Serving"]
stage: engine
order: 1
prereqs: []
minutes: 45
difficulty: 2
---

> 原文:[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)(Woosuk Kwon, Zhuohan Li et al.,UC Berkeley 等,SOSP '23,arXiv:2309.06180 v1,2023-09)

KV cache 是 LLM serving 的"不可压缩的内存税":OPT-13B 每生成一个 token 就要多占 **800 KB** 显存,一个最长 2048 token 的请求上限 **1.6 GB**。传统系统按最大长度给每个请求预留一整块连续显存,结果 token states(真正有用的 K/V 数据)只占 **20.4%–38.2%**,其余全被预留、内部碎片和外部碎片吃掉。PagedAttention 的答案一句话:像操作系统管物理内存一样管 KV cache——切成固定大小的块,逻辑连续、物理随意,按需分配、写时拷贝。论文(即 vLLM 引擎)靠这个把 token states 占比提到 **96.3%**,在同等延迟下把吞吐提升 **2-4×**,ShareGPT 负载上峰值达 FasterTransformer 的 **22×**。但请注意,这 2-4× 不是 attention kernel 算得更快——kernel 反而比 FasterTransformer 慢 **20-26%**——而是显存利用率换来的批处理收益。本文先把这笔账算清,再拆机制,最后对照 2026 年的 vLLM V1 引擎看哪些想法活了下来、哪些被工程化了。

<!-- more -->

## 📑 目录

- [🗺️ 原文阅读地图](#️-原文阅读地图)
- [0. 读前 3 分钟:先算清 KV cache 的三笔账](#0-读前-3-分钟先算清-kv-cache-的三笔账)
- [1. 问题:现有系统的显存只有两成在干正事](#1-问题现有系统的显存只有两成在干正事)
- [2. 核心思想:把 KV cache 当成虚拟内存来管](#2-核心思想把-kv-cache-当成虚拟内存来管)
- [3. PagedAttention kernel:按块算注意力](#3-pagedattention-kernel按块算注意力)
- [4. 解码走查:按需分配,一个块都不多给](#4-解码走查按需分配一个块都不多给)
- [5. Copy-on-Write:让并行采样和 beam search 共享 KV](#5-copy-on-write让并行采样和-beam-search-共享-kv)
- [6. 抢占与恢复:swap 还是 recompute](#6-抢占与恢复swap-还是-recompute)
- [7. 实验:2-4× 到底从哪来](#7-实验2-4-到底从哪来)
- [8. 从论文到 2026 年的 vLLM 工程](#8-从论文到-2026-年的-vllm-工程)
- [9. 局限与选型边界](#9-局限与选型边界)
- [10. 面试官视角:三问三答](#10-面试官视角三问三答)
- [📝 总结](#-总结)
- [🎯 自我检验清单](#-自我检验清单)
- [📚 参考资料](#-参考资料)

## 🗺️ 原文阅读地图

这篇论文同时是"算法论文"和"系统论文":算法上只改 KV cache 的存储布局,系统上重新设计了 scheduler 与 block manager。本文选择性精讲如下,避免把一篇中文解读误当成原文逐段翻译。

| 原文单元 | 处理深度 | 本文位置与理由 | 来源锚点 |
| --- | --- | --- | --- |
| §1 Introduction(显存布局、三类浪费、2-4× 承诺) | 精讲 | 第 1 节,先量化问题再谈方案 | §1;Fig.1;Fig.2;PAGE 1-2 |
| §2 Background(自回归、prompt/generation 两阶段、iteration-level scheduling) | 简述 | 第 0 节背景一笔带过,读者已有直觉 | §2.1;PAGE 3 |
| §3 Memory Challenges(800 KB/token、1.6 GB/请求、三类浪费定义) | 精讲 | 第 0、1 节,含一个完整手算例子 | §3;Fig.3;PAGE 4 |
| §4.1 PagedAttention 分块注意力 | 精讲 | 第 3 节,核心公式 Eq.(4) 逐项拆解 | §4.1;Eq. 4;PAGE 5 |
| §4.2 KV Cache Manager(逻辑/物理块 + block table) | 精讲 | 第 2 节,机制卡 1 | §4.2;Fig.6;PAGE 6 |
| §4.3 Decoding 走查(按需分配三步) | 精讲 | 第 4 节,含最小数字例子 | §4.3;Fig.6-7;PAGE 6-7 |
| §4.4 并行采样 / beam search / 共享前缀(CoW) | 精讲 | 第 5 节,机制卡 4,含共享收益数字 | §4.4;Fig.8-10;PAGE 7-8 |
| §4.5 Scheduling & Preemption(FCFS、all-or-nothing、swap vs recompute) | 精讲 | 第 6 节,机制卡 5 | §4.5;PAGE 8 |
| §4.6 Distributed(单一 KV manager + 张量并行) | 简述 | 第 2 节一句带过,不展开 all-reduce 细节 | §4.6;PAGE 8-9 |
| §5 Implementation(8.5K 行 Python + 2K 行 CUDA、fork/append/free) | 简述 | 第 5、8 节引用相关原语 | §5;PAGE 9 |
| §6 Evaluation(吞吐主表与多负载) | 精讲(数字表) | 第 7 节,全部 2.2×/4.3×/22× 等数字集中呈现 | §6;Fig.12-17;Table 1;PAGE 9-12 |
| §7 Ablation(kernel 开销、block size、swap vs recompute) | 简述 | 第 3、6、7 节引用结论,不逐面板展开 | §7;Fig.18-19;PAGE 12-13 |
| §8 Discussion(适用边界) | 精讲(局限) | 第 9 节,选型边界全部来自这里 | §8;PAGE 13 |
| §9 Related Work(Orca、FlashAttention 定位) | 简述 | 第 10 节回答"与 FlashAttention 的关系" | §9;PAGE 13-14 |
| §10 Conclusion | 不展开 | 一句话总结,并入本文 📝 总结 | §10;PAGE 14 |

📌 **本文承诺**:读完后,你应该能手算 OPT-13B 一个 token 的 KV cache 大小(800 KB)和单请求上限(1.6 GB),画出逻辑块→物理块的 block table 翻译过程,解释为什么 CoW 能让 beam search 省 37.6%–55.2% 的内存,并说清"kernel 慢了 20-26%,端到端却快 2-4×"这句话为什么不自相矛盾。

## 0. 读前 3 分钟:先算清 KV cache 的三笔账

### 0.1 KV cache 为什么是 serving 的命门

自回归模型生成第 $i$ 个 token 时,注意力要跟前面所有 token 的 key/value 做计算:

$$
P(x)=P(x_1)\cdot P(x_2\mid x_1)\cdots P(x_n\mid x_1,\dots,x_{n-1})
$$

其中 $x=(x_1,\dots,x_n)$ 是 token 序列,$P$ 是语言模型给出的联合概率(§2.1,PAGE 3)。如果每步都重新算前面所有位置的 K/V,复杂度随序列长度平方增长,谁都扛不住。所以 serving 系统把每个位置算好的 key、value 向量缓存下来——这就是 KV cache。它是**按 token 线性增长的"内存税"**:序列越长,税越重,而且**每个并发请求各交各的**。

> 打个比方:KV cache 像餐厅后厨为每桌客人提前备好的配菜。客人还没点完菜,配菜就得备着,而且一桌一桌独立备——来了 100 桌,就得备 100 份。问题不在"备菜"本身,而在"备菜间"怎么规划:是给每桌按最大可能人数预留一整面墙,还是客人真坐下、真加菜时再一格一格取?

### 0.2 手算:OPT-13B 一个 token 吃掉 800 KB

这是全文第一个值得亲手算的数字。OPT-13B 每个 token 的 KV cache 大小(§3,PAGE 4):

$$
\underbrace{2}_{\text{K 和 V 各一份}}
\times
\underbrace{5120}_{\text{hidden size }d}
\times
\underbrace{40}_{\text{transformer 层数}}
\times
\underbrace{2}_{\text{FP16 占 2 字节}}
=819{,}200\ \text{B}\approx\underbrace{800\ \text{KB}}_{\text{每个 token}}
$$

一步步来:每层每个 token 要存 key 和 value 两个向量(×2),每个向量长度等于 hidden size 5120,模型共 40 层(×40),FP16 存储每元素 2 字节(×2)。2×5120×40×2 = 819,200 字节 ≈ 800 KB。

那么单请求上限呢?论文里最大序列 2048 token:

$$
2048 \times 800\ \text{KB}\approx \underbrace{1.6\ \text{GB}}_{\text{单请求 KV cache 上限}}
$$

($§3,PAGE 4)。再把视角放到整张卡:1×A100 40GB 上跑 OPT-13B,参数占 26 GB,KV cache 分配 12 GB(Table 1,PAGE 9):

$$
12\ \text{GB}\div 800\ \text{KB/token}\approx \underbrace{15{,}700}_{\text{"max 15.7K 槽"}}
$$

这 12 GB 大概只能同时容纳约 1.5 万个 token 的 KV——注意这还是"槽位"上限,不是"实际利用"。**显存墙比你想的更近**:论文还点了一句,从 A100 到 H100,算力(FLOPS)约翻 2×,而显存容量维持在 80 GB 量级(§3,PAGE 4)——算力涨得比显存快,内存将日益成为瓶颈。这就是整篇论文的出发点。

### 0.3 三笔账的直觉:预留、碎片与共享

现有 serving 系统(FasterTransformer、Orca)的 KV cache 管理有三个毛病,对应三笔浪费(§3.1,PAGE 4):

1. **Reservation(预留)**:按最大可能序列长度给每个请求预留连续显存。预留了但没用到,就是浪费;
2. **Internal fragmentation(内部碎片)**:预留区域里实际只用了开头一小段,剩余槽位永远空着;
3. **External fragmentation(外部碎片)**:请求长度不同,预留区域参差不齐,中间产生无法利用的缝隙。

论文的实测数字(Fig.2,PAGE 2):现有系统里真正存储 token states 的比例只有 **20.4%–38.2%**,最低的是 Orca (Max) 的 20.4%;而 vLLM 高达 **96.3%**。换句话说,**现有系统每买 5 块钱显存,只有 1 块钱在干正事**。

**PagedAttention 的思路就是把这三笔浪费全部消掉:不预留(按需分配)、碎片收敛到一个块内(块粒度)、缝隙消失(全等大块)。** 下一节开始拆机制。

## 1. 问题:现有系统的显存只有两成在干正事

### 1.1 显存布局:参数 65%,KV cache 超过 30%

先看一张 A100 40GB 上服务 13B 模型时的显存布局(Fig.1 left;§1;PAGE 1):

- **约 65%(26 GB)**:模型参数,静态驻留,权重加载后不动;
- **超过 30%**:KV cache,随请求动态分配;
- 其余:activation 等临时张量。

注意 KV cache 的比例是"**超过 30%**"——它随并发请求数和序列长度增长,是 serving 系统里唯一可以靠调度和管理"省出来"的部分。参数没法省(除非量化),activation 是瞬时的,只有 KV cache 是长期、按请求线性占用的。这也是为什么一个 LLM 请求比传统 keyword query 贵约 **10×**(§1,PAGE 1,论文引用 Reuters 报道 [43])。

### 1.2 三类浪费:一张堆叠柱状图看穿

论文 Fig.2 把四种系统的 KV cache 内存按"Token states / Reservation / Internal frag. / External frag."四类拆开(PAGE 2):

| 系统 | Token states | Reservation | Internal frag. | External frag. |
| --- | ---: | ---: | ---: | ---: |
| Orca (Max) | 20.4% | 13.3% | 57.3% | 8.9% |
| Orca (Pow2) | 26.8% | 17.9% | 41.6% | 13.6% |
| Orca (Oracle) | 38.2% | 25.2% | 36.6% | —(图未单列) |
| **vLLM** | **96.3%** | 近零 | 近零 | 近零 |

<img src="/AIInfraGuide/images/pagedattention-fig1-memory-waste-comparison.png" alt="SOSP 23 原文 Fig.2:四种系统的 KV cache 内存浪费构成堆叠柱状图,Orca Max/Pow2/Oracle 的 token states 仅 20.4% 到 38.2%,vLLM 达 96.3%" style="max-width: 95%; display: block; margin: 0 auto;" />

*图源:PagedAttention 论文 Figure 2(SOSP '23,arXiv:2309.06180)*

⚠️ **注意口径**:Orca (Oracle) 是论文自实现的一种"上帝视角" baseline——它假设系统**提前知道每个请求的真实输出长度**,因此不用预留(见第 7 节)。即便这样,它的 token states 也只有 38.2%,内部碎片仍占 36.6%,因为它依然需要连续内存块。vLLM 的 96.3% 柱子里,其余约 3.7% 图中未细分(账本 §8:原文未提供)——这就是"近零浪费"的含义。**在显存墙面前,省出 3-4 倍的有效空间,直接兑换成 2-4× 的吞吐,这是全文最核心的因果链。**

### 1.3 浪费从哪来:一张"按最大长度预留"的示意图

Fig.3 用两个请求把浪费画成了具体的槽位(PAGE 4):

- **请求 A**:max 2048 token。实际 prompt 7 槽 + 已生成 1 槽 = 用了 8 槽;预留 2 槽;**内部碎片 2038 槽永未使用**;
- **请求 B**:max 512 token。实际 3 + 1 = 4 槽;预留 1 槽;**内部碎片 507 槽**。

<img src="/AIInfraGuide/images/pagedattention-fig2-memory-waste-sources.png" alt="SOSP 23 原文 Fig.3:现有系统按最大长度预留连续内存的三类浪费示意,请求 A max 2048 有 2038 个内部碎片槽,请求 B max 512 有 507 个,另有外部碎片" style="max-width: 88%; display: block; margin: 0 auto;" />

*图源:PagedAttention 论文 Figure 3(SOSP '23,arXiv:2309.06180)*

请求 A 的账很好算:2048(max)− 7(prompt)− 1(generated)− 2(reserved)= **2038 个槽**,从头到尾只服务了"确认存在"这个动作。为什么系统要这么干?因为解码阶段输出长度未知,而深度学习框架要求 KV cache 是**连续张量**,一次申请必须够用;长度超了就崩。**PagedAttention 的全部设计,就是绕开"连续张量"这个约束。**

## 2. 核心思想:把 KV cache 当成虚拟内存来管

### 2.1 逻辑块与物理块

PagedAttention 的灵感直接来自操作系统的分页(paging):把每个序列的 KV cache 切成固定大小的 **KV block**,每块容纳固定数量 $B$ 个 token 的 key/value 向量;块的**物理地址不必连续**(§4.1,PAGE 5)。

- **逻辑块**:序列视角,第 1、2、3… 块,天然连续;
- **物理块**:显存视角,散落在显存各处,由 block table 记录映射。

连续的逻辑块对应非连续的物理块,深度学习框架"要连续张量"的限制就被绕开了(§3.1,PAGE 4)。footnote 还交代了一个实现取舍:所有 key/value 向量既可以放进同一组 KV block 统一管理,也可以按 head/layer 粒度拆分、各自维护独立的 block table;论文实测两设计无性能差异,实现上选择了第二种——按 head/layer 拆分的方案(we choose the second one,§4.1 footnote,PAGE 5)。

> 还是餐厅的比方:传统系统要求"同一桌客人的配菜必须摆在同一面墙上",于是只能按最大人数预留整面墙;PagedAttention 把配菜装进统一大小的餐盒,分散在后厨任意货架,服务员手里一张"桌位地图"就能找到——**餐盒(块)等大,货架(显存)没有缝隙;客人(请求)来了才取餐盒,走了立刻翻台。**

### 2.2 block table:一张桌位地图

vLLM 的 KV cache manager 为每个请求维护一张 **block table**——逻辑块到物理块的映射,每条表项记录两样东西:物理块号,以及已填充位置数 `#filled`(§4.2,Fig.6,PAGE 6)。

<img src="/AIInfraGuide/images/pagedattention-fig3-block-table.png" alt="SOSP 23 原文 Fig.6:请求 A 的 3 个逻辑 KV block 通过 block table 映射到物理块 7/1/3,第 4 个逻辑块尚未分配(图中为空),表项记录物理块号与已填充位置数,解码时逐槽填充" style="max-width: 95%; display: block; margin: 0 auto;" />

*图源:PagedAttention 论文 Figure 6(SOSP '23,arXiv:2309.06180)*

图里请求 A 的 3 个逻辑块映射到物理块 **7、1、3**,第 4 个逻辑块尚未分配(图中为空)——物理上完全是乱的,但逻辑上顺序不变;`#filled` 随解码逐槽更新。翻译过程即:attention kernel 拿到逻辑位置,查 block table 得到物理块号和偏移,再去对应显存地址取数。

**机制卡 1:逻辑块 / 物理块 + block table**

| 项目 | 内容 |
| --- | --- |
| 要解决什么 | 深度学习框架要求 KV cache 连续存储,迫使系统按最大长度预留,产生 20.4%–38.2% 的利用率黑洞 |
| 最小前置 | 理解 KV cache 按 token 增长、注意力需要访问全部历史 K/V |
| 输入 → 状态 → 输出 | 输入:序列的 token 流;状态:每请求一张 block table(物理块号 + `#filled`);输出:任意位置 KV 的物理地址 |
| 因果步骤 | ① 序列 KV 按 $B$ 个 token 一组切成逻辑块 → ② manager 从空闲池取物理块 → ③ 表项登记逻辑块→物理块映射与填充数 → ④ kernel 按表取数 → ⑤ 请求结束整表释放,物理块回池 |
| 公式语义 | 无新公式;等价于把"连续数组"抽象成"页表"这一层间接寻址 |
| 最小例子 | 3 个逻辑块 → 物理块 7/1/3,第 4 个逻辑块尚未分配(Fig.6);任意表项可指向任意空闲物理块 |
| 边界与来源 | 块等大 ⇒ 外部碎片消除(§1,PAGE 2);只对"输出长度未知 + 显存受限"的负载有效(§8,PAGE 13);footnote 说明按 head/layer 拆分管理无性能差异(§4.1,PAGE 5) |

分布式执行只改一处:张量并行下每个模型分片只存自己负责 head 的 KV 部分,但**所有 worker 共享同一份逻辑→物理块映射**,只有一个集中式 KV cache manager,调度器每步广播 token id 与 block table(§4.6,PAGE 8-9)。这部分简述,记住"一份映射、各自存 KV"即可。

## 3. PagedAttention kernel:按块算注意力

### 3.1 从按位置求和到按块求和

标准注意力(§2.1,PAGE 3)对位置 $i$ 的 query $q_i$(第 $i$ 个 token 的 query 向量)计算对所有历史位置 $j$ 的分数:

$$
a_{ij}=\frac{\exp(q_i^\top k_j/\sqrt{d})}{\sum_{t=1}^{i}\exp(q_i^\top k_t/\sqrt{d})},\qquad o_i=\sum_{j=1}^{i}a_{ij}v_j
$$

其中 $k_j$、$v_j$ 是位置 $j$ 的 key、value 向量,$d$ 是 hidden size(即前面公式里的 5120),$a_{ij}$ 是注意力分数,$o_i$ 是位置 $i$ 的输出。求和按**位置**展开——这要求所有位置的 K/V 都在手边,且最好是连续内存。

PagedAttention 把它改写成按**块**求和(Eq. 4,§4.1,PAGE 5)。设块大小 $B$($B$ 为每块容纳的 token 数),第 $j$ 个 key 块、value 块定义为:

$$
\underbrace{K_j}_{\text{第 }j\text{ 个 key 块}}=(k_{(j-1)B+1},\dots,k_{jB}),\qquad
\underbrace{V_j}_{\text{第 }j\text{ 个 value 块}}=(v_{(j-1)B+1},\dots,v_{jB})
$$

分块注意力:

$$
\underbrace{A_{ij}}_{\text{第 }j\text{ 块上的分数行向量}}
=\frac{\exp(q_i^\top K_j/\sqrt{d})}{\sum_{t=1}^{\lceil i/B\rceil}\exp(q_i^\top K_t\mathbf{1}/\sqrt{d})},\qquad
\underbrace{o_i}_{\text{位置 }i\text{ 的输出}}=\sum_{j=1}^{\lceil i/B\rceil}V_jA_{ij}^{\top}
$$

逐项解释:

- $\lceil i/B\rceil$:位置 $i$ 所属的逻辑块号,决定要对多少块求和;
- $A_{ij}=(a_{i,(j-1)B+1},\dots,a_{i,jB})$:第 $j$ 个 KV block 上的一行分数;
- $\mathbf{1}$:全 1 向量,作用是把块内各 token 的 $\exp(q_i^\top k_t/\sqrt{d})$ 累加,对应原公式对 $t$ 的求和;
- 公式语义:把"对每个位置求和"改成"对每个块求和",**每个块可以独立地从非连续物理地址取出计算**,再累加。

数学上它与标准注意力**完全等价**——只是求和顺序按块分组了。这正是论文敢说"without affecting the model accuracy at all"(不影响模型精度,§1,PAGE 2)的原因:PagedAttention 不是近似注意力,它只改了存储布局和取数方式,不碰注意力分数的数值。

**机制卡 2:分块注意力 kernel**

| 项目 | 内容 |
| --- | --- |
| 要解决什么 | 让注意力计算能消费"非连续物理块"里的 K/V,而不是只认连续张量 |
| 最小前置 | 标准注意力公式 Eq.(3);block table 给出每个逻辑块的物理地址 |
| 输入 → 状态 → 输出 | 输入:$q_i$ 与 block table;状态:物理块内的 K/V;输出:$o_i$(与标准注意力逐位相同) |
| 因果步骤 | ① 由 $i$ 算出需要访问的逻辑块数 $\lceil i/B\rceil$ → ② 查表得各块物理地址 → ③ 对每块分别算 $\exp(q_i^\top K_j/\sqrt d)$ → ④ 块内求和(全 1 向量)得分母 → ⑤ 归一化得 $A_{ij}$ → ⑥ 累加 $V_jA_{ij}^\top$ |
| 公式语义 | $o_i=\sum_{j=1}^{\lceil i/B\rceil}V_jA_{ij}^\top$:输出 = 各块贡献之和,块间可乱序 |
| 最小例子 | 位置 $i=20$、$B=16$:需要 $\lceil 20/16\rceil=2$ 个块,分别从两个物理地址取 K/V,算完相加 |
| 边界与来源 | kernel 比 FasterTransformer 的 attention kernel 慢 **20-26%**(§7.1,Fig.18a,PAGE 12),靠批处理收益与其余算子不受影响弥补;间接寻址有开销,计算受限场景不划算(§8,PAGE 13) |

⚠️ **一个常被误解的点**:论文并不宣称"我的 attention kernel 更快"。恰恰相反,它承认 PagedAttention kernel 比 FasterTransformer 的 kernel 慢 20-26%(§7.1,PAGE 12)。**vLLM 赢在"同样一块 GPU 能塞下 3-4 倍的请求",batch 变大带来的吞吐收益远大于 kernel 的常数开销**。这也是读这篇论文最要紧的口径:别把 2-4× 记到 kernel 头上。

## 4. 解码走查:按需分配,一个块都不多给

### 4.1 三步走查

§4.3 用 Fig.6-7 演示了一个请求从 prefill 到 decode 的完整生命周期(PAGE 6-7):

1. **Prefill(prompt 阶段)**:按 prompt 实际长度分配**最少**的物理块。论文 Fig.6 的示例按块容量 $B=4$ 演示:7 个 prompt token 映射到前 2 个逻辑块(0 和 1),分别对应物理块 7 和 1;prefill 把前 4 个 token 的 KV 存进逻辑块 0、后 3 个 token 存进逻辑块 1,剩余槽留给自回归生成(§4.3,PAGE 6)——绝不按最大长度 2048 预留;
2. **Decode(生成阶段)**:每生成一个 token 填进当前块,`#filled` 加 1;当前块填满($=B$)再向 manager 要新块;
3. **释放**:请求结束,块归还空闲池,其他请求立刻复用。

因为块是从左到右填满的,**只有最后一个逻辑块可能有空槽**,单请求的浪费被限制在一个块以内(§4.3,PAGE 6);全部块等大,外部碎片从结构上消失(§1,PAGE 2)。

**机制卡 3:按需分配的解码走查**

| 项目 | 内容 |
| --- | --- |
| 要解决什么 | 预留策略下"最大长度×请求数"的显存天文数字;输出长度未知导致无法精确申请 |
| 最小前置 | block table 支持逻辑/物理解耦,block 可独立分配释放 |
| 输入 → 状态 → 输出 | 输入:新 token;状态:每请求的块列表与 `#filled`;输出:物理块分配/释放动作 |
| 因果步骤 | ① 分配:新请求按 prompt 长度取最少块 → ② 填充:decode 每 token 写槽、计数 +1 → ③ 换块:满块后分配新物理块 → ④ 释放:请求结束全部归还 |
| 公式语义 | 无;等价于"分配上限 = 实际 token 数 ÷ B 向上取整"的按需版本 |
| 最小例子 | 见 4.2 |
| 边界与来源 | 浪费 ≤ 一个块(§4.3,PAGE 6);块释放后立即复用(§4.3,Fig.7,PAGE 6-7) |

### 4.2 最小数字例子:16 token 的块怎么填

下面按 vLLM 生产默认 $B=16$ 重述一遍走查(论文 Fig.6 示例用的块容量是 4,7 个 token 占 2 个逻辑块;这里换成 16 贴近真实配置,例子为本文自设)。一个 prompt 7 token 的请求:

- 分配 1 个物理块,`#filled=7`,块内 9 个槽暂时空着;
- 继续生成 10 个 token:第 1 块填满(7+9=16),第 10 个 token 落在新分配的块 2 里,`#filled=1`;
- 此刻整请求浪费 = 块 2 的 15 个空槽,**上限就是 15 个槽**——因为只有最后一块可能不满。

对照 Fig.3 的老方案:同样的请求(实际 7+1 个 token、max 2048)要预留 2048 槽,其中 **2038 个槽从头到尾是死的**(Fig.3,PAGE 4)。**2038 vs ≤15,这就是"按需分配 + 块粒度"的量化差距。**

## 5. Copy-on-Write:让并行采样和 beam search 共享 KV

### 5.1 引用计数与写时拷贝

并行采样(n 路)和 beam search 有个天然结构:**多个输出序列共享同一个 prompt 前缀**。传统系统里每个序列各存一份完整 KV,浪费;PagedAttention 让共享前缀只存一份物理块,多个序列的 block table 指向同一批物理块。

这引出"谁能写"的问题。解法是操作系统的老手艺——**copy-on-write(写时拷贝,CoW)**:每个物理块带引用计数 `ref count`(§4.4,Fig.8,PAGE 7);当某个序列要往一个仍被共享的块里写新 token 时:

1. 分配一个新物理块;
2. 把旧块内容拷贝过去;
3. 新序列写入自己的新块,旧块 ref count 减 1;
4. 其他序列继续用旧块,互不干扰。

> 打个比方:并行采样像几个学生共读同一本教材的同一章(共享块)。谁要往书页空白处写自己的笔记,就去复印一份再写(拷贝块),原书继续流转;管理员(block manager)记着每本书有几个读者(ref count),最后一个读者还书时才把书放回书架(释放)。

beam search 里共享关系随解码动态演化,像 OS 的进程树(§4.4,Fig.9,PAGE 7):父候选分裂出多个子候选时共享块,被淘汰的分支 ref count 归零即释放。实现层 vLLM 只暴露三个原语支撑所有解码算法——`fork`(派生序列,共享块)、`append`(追加写)、`free`(释放)(§5.2,PAGE 9)。

### 5.2 共享前缀与缓存

更进一步的场景是**跨请求共享前缀**:同一服务下大量请求以相同系统提示词开头。论文设想服务商预先缓存前缀块,新请求的 block table 直接映射到缓存块,只有需要写入的末块标记 CoW(§4.4,Fig.10,PAGE 8)。注意 §4.4 这里只给了机制和一张示意图;跨请求共享前缀的实测收益,原文仅在 §6.4 的合成翻译负载(LLaMA-13B + WMT16,共享 1-shot 80 token / 5-shot 341 token 前缀)中给出(相对 Orca (Oracle) **1.67× / 3.58×**,Fig.16,PAGE 12),**真实多租户/系统提示词流量下的实测原文没有做**(账本 §8)。这个场景后来由 vLLM 的自动 prefix cache 默认开启(见第 8 节)——那是把 §4.4 的手动预留演进成 hash 命中即共享的工程演进,不是填补论文没做的实验。

### 5.3 数字例子:beam search 省了多少

§6.3 的实测(Alpaca + OPT-13B,Fig.15,PAGE 11):

- 并行采样 2/4/6 路:KV block 共享内存节省 **6.09% / 8.53% / 9.79%**;
- Beam search width 2/4/6:**37.56% / 53.13% / 55.16%**。

跨数据集看:并行采样节省 6.1%–9.8%(Alpaca)、16.2%–30.5%(ShareGPT);beam search 37.6%–55.2%(Alpaca)、44.3%–66.3%(ShareGPT)(§6.3,PAGE 11)。论文第 3 节前瞻性预告"up to 55% memory saving"(§3,PAGE 4),实测兑现。为什么 beam 比并行采样省得多?因为 beam 的候选共享前缀更长(整个已解码前缀),而并行采样只共享 prompt 段——§3 还给了个旁证:并行采样实验里 prompt 的 KV 只占总 KV 的约 12%(§3,PAGE 4,实验见 §6.3),共享面本来就小。

**机制卡 4:CoW 与共享**

| 项目 | 内容 |
| --- | --- |
| 要解决什么 | 多序列共享前缀时重复存 KV 的浪费;共享块被写时的数据竞争 |
| 最小前置 | 逻辑块可被多个 block table 引用(物理块级引用计数) |
| 输入 → 状态 → 输出 | 输入:序列的 fork/append/free 操作;状态:每物理块 ref count;输出:共享或拷贝的块分配决策 |
| 因果步骤 | ① fork:新序列 block table 复制并共享物理块,ref count +1 → ② append 遇共享块:分配新块、拷贝旧内容、写入、旧块 ref count −1 → ③ 淘汰分支:ref count 归零的块回收 |
| 公式语义 | 无;核心不变量是"写者必须独占",由 ref count 保证 |
| 最小例子 | beam width 2:Alpaca 上节省 37.56% KV 内存(Fig.15b,PAGE 11) |
| 边界与来源 | 共享收益取决于共享前缀占比:并行采样仅 6.1%–30.5%,beam search 37.6%–66.3%(§6.3,PAGE 11);跨请求共享前缀的实测仅见于 §6.4 合成翻译负载(1.67×/3.58×,Fig.16),真实多租户/系统提示词流量下的实测原文未做(账本 §8) |

## 6. 抢占与恢复:swap 还是 recompute

### 6.1 FCFS 与 all-or-nothing

显存是稀缺资源,请求多到放不下时必须有抢占策略。论文的选择(§4.5,PAGE 8):

- **调度**:FCFS(first-come-first-serve),保证公平、防止饥饿;资源不足时,**最后到达的请求先被抢占**;
- **驱逐粒度**:all-or-nothing——一个序列的所有块要么全驱逐、要么全保留。因为处理一个请求需要它的全部 token state,只抢一半无法继续;
- **调度单位**:同一请求内的多个序列(如 beam 候选)组成 **sequence group**,整体调度。

### 6.2 两条恢复路径与一条 20% 经验线

被抢占的请求怎么恢复?两条路(§4.5,PAGE 8):

1. **Swapping(换出)**:把块拷贝到 CPU RAM 的 swap 空间。swap 空间上限受限于 GPU KV cache 总量(没法凭空变多);
2. **Recomputation(重算)**:把 KV cache 重新算一遍。关键洞察:被抢占请求已经解码出的 token 可以拼回原 prompt,**一次 prefill 全部算完**,所以重算延迟显著低于当初的逐 token 解码延迟。

选哪条?取决于 **CPU-GPU 带宽 vs GPU 算力**的相对成本(§4.5,PAGE 8)。论文实测给出一条可迁移的工程经验(§7.3,PAGE 13):**recompute 的开销不超过 swap 延迟的 20%**——当 block size 小的时候尤其如此,因为 swap 被切成大量小传输,效率很低;block size 16–64 时两者端到端相当。微基准 Fig.19a 显示 swap 成本随 block size(1–256)变化,而 recompute 与 block size 无关(PAGE 13)。工程含义:**机器间带宽好、算力富余时,recompute 往往更省事**;这也是后来许多推理引擎的默认倾向。

**机制卡 5:抢占与恢复**

| 项目 | 内容 |
| --- | --- |
| 要解决什么 | 显存不足时谁让位、被让位的请求怎么低成本复活 |
| 最小前置 | block 可整体迁移(swap)或整体重算(recompute);请求状态可冻结 |
| 输入 → 状态 → 输出 | 输入:新请求到达;状态:序列组 + 块占用;输出:抢占决策与恢复路径 |
| 因果步骤 | ① FCFS 入队 → ② 显存不足:抢占最后到达的序列组 → ③ 全量驱逐其块 → ④ 选 swap 或 recompute → ⑤ 资源释放后恢复 |
| 公式语义 | 无;决策依据是"swap 成本(带宽 × 数据量)"vs"recompute 成本(算力 × 一次 prefill)" |
| 最小例子 | block size 小 → swap 碎传输低效 → recompute 更优(recompute ≤ swap 的 20%,§7.3,PAGE 13) |
| 边界与来源 | all-or-nothing 是"处理请求需要全部 token state"的直接推论(§4.5,PAGE 8);swap 的绝对带宽数字原文未给出(账本 §8) |

## 7. 实验:2-4× 到底从哪来

### 7.1 实验设置先对齐

论文对比对象:FasterTransformer,以及自实现的三种 Orca 变体(Orca 未开源,§6.1,PAGE 10):

- **Orca (Max)**:按最大长度 2048 token 预留;
- **Orca (Pow2)**:按 2 的幂预留,最多 2× 过预留(如 25 → 32);
- **Orca (Oracle)**:假设预知真实输出长度——论文自己承认这是实践不可行的上界(§6.1,PAGE 10)。

负载:ShareGPT(input 均值 161.31 / output 均值 337.99 token)与 Alpaca(input 19.31 / output 58.45)(Fig.11,PAGE 9);ShareGPT 输入平均是 Alpaca 的 **8.4×**、输出 **5.8×**(§6.1,PAGE 10)。大部分实验用 1 小时 traces,OPT-175B 因成本只跑 15 分钟(§6.1,PAGE 10)。

### 7.2 吞吐主结果:一组值得背的数字

**主结论(Abstract;§1;§10;PAGE 1/14):vLLM 相对现有系统在同等延迟下吞吐提升 2-4×。** 拆开看(§6.2;PAGE 10-11,OPT-13B + ShareGPT):

| 指标 | 数值 | 锚点 |
| --- | ---: | --- |
| 相对 Orca (Oracle) 可维持请求率 | 1.7×–2.7× | §6.2;PAGE 11 |
| 相对 Orca (Max) 可维持请求率 | 2.7×–8× | §6.2;PAGE 11 |
| 相对 FasterTransformer | 最多 22× | §6.2;PAGE 11 |
| 同时处理请求数 vs Orca (Oracle) | 2.2× | §6.2;Fig.13a;PAGE 10-11 |
| 同时处理请求数 vs Orca (Max) | 4.3× | §6.2;Fig.13a;PAGE 10-11 |

平均批处理请求数最能说明"显存利用率 → 并发"的因果链(Fig.13,PAGE 10):ShareGPT(2 req/s 到达率)下 Orca (Max)/Pow2/Oracle/vLLM 分别是 **7.00 / 9.81 / 13.62 / 30.42**;Alpaca(30 req/s)下是 **7.00 / 43.24 / 72.75 / 132.44**。vLLM 在同一张卡上同时服务 132 个请求而别人只能服务 43 个——**批越大,GPU 利用率越高,端到端吞吐自然越高**。

<img src="/AIInfraGuide/images/pagedattention-fig4-throughput-comparison.png" alt="SOSP 23 原文 Fig.12:OPT-13B/66B/175B 与 ShareGPT/Alpaca 六面板,normalized latency 随 request rate 变化的吞吐对比,vLLM 在更高请求率下仍保持低延迟" style="max-width: 98%; display: block; margin: 0 auto;" />

*图源:PagedAttention 论文 Figure 12(SOSP '23,arXiv:2309.06180);六面板覆盖 OPT-13B/66B/175B × ShareGPT/Alpaca*

Fig.12(上图)是 2-4× 的直观来源:横轴 request rate 推到很高时,其他系统的 normalized latency 开始飙升(显存耗尽、排队),vLLM 的曲线还能维持低位。

其他负载的补充数字:

- **Chatbot**:vLLM 相对三个 Orca baseline 可维持 **2×** 请求率(§6.5,Fig.17,PAGE 12;该实验上下文截断到最近 1024 token,最多输出 1024);
- **翻译(共享前缀)**:LLaMA-13B + WMT16,共享 1-shot(80 token)/ 5-shot(341 token)前缀时,相对 Orca (Oracle) 吞吐提升 **1.67× / 3.58×**(§6.4,Fig.16,PAGE 12)——共享前缀场景是 CoW + 缓存的直接受益者;
- **beam search 增幅**:OPT-13B + Alpaca,从基础采样到 beam width=6,vLLM 相对 Orca (Oracle) 的改善从 1.3× 升到 **2.3×**(§6.3,PAGE 11)——beam 越宽,共享收益越大(呼应 5.3 节的 55% 内存节省)。

### 7.3 一个诚实的边界

论文没有隐瞒:OPT-175B + Alpaca(短序列 + 大显存)时,vLLM 相对 Orca (Oracle)/(Pow2) 的优势"less pronounced"(明显减弱),因为系统转为 compute-bound(§6.2,PAGE 11)。**当显存不再是瓶颈,分页省出来的空间换不到吞吐**——这条边界在第 9 节展开。

## 8. 从论文到 2026 年的 vLLM 工程

> 本节工程事实访问日期:**2026-08-11**(来源:GitHub `vllm-project/vllm` main 分支源码与 release 信息;最新 release v0.27.0 发布于 2026-08-10,仓库 88.7K stars)。论文数字(2-4×、96.3% 等)是 SOSP '23 时代的结论,不因工程演进失效;但实现细节已大幅变化,下面分层对照。

### 8.1 论文时代 vs V1 引擎

论文(2023)的 vLLM 是 **V0 引擎**:scheduler + block manager 用 Python 实现,单一大 block table,自研 PagedAttention CUDA kernel,代码规模 8.5K 行 Python + 2K 行 C++/CUDA(§5,PAGE 9)。到今天,当前架构是 **V1 引擎**(`vllm/v1/core/kv_cache_manager.py`、`vllm/v1/core/sched/scheduler.py`、`vllm/v1/core/block_pool.py` 为主路径),V0 已移除。**论文的三件核心遗产——分块存储、ref count + CoW、抢占——在 V1 里都活着**:`KVCacheBlock` 仍带引用计数与 block copy 原语(`vllm/v1/core/kv_cache_utils.py` L118、L179),调度器仍有 `_preempt_request`。

### 8.2 三个关键演进

1. **Block size 仍是 16**:论文消融说默认 16 最优——"够大以利用 GPU 并行,够小以避免内部碎片",ShareGPT 上 16–128 都好,Alpaca 上 16/32 好(§7.2,Fig.18b,PAGE 12)。V1 的 `DEFAULT_BLOCK_SIZE = 16`(`vllm/config/cache.py` L48)一字未改;个别后端偏好更大倍数(flash_attn 后端的 `get_preferred_block_size` 在 XPU 上返回 max(默认, 64),CUDA 上沿用默认 16,且要求 block_size % 16 == 0)。**16 这个数字从论文活到今天,是"碎片 vs 并行度"权衡的实证沉淀。**

2. **前缀缓存从手动变自动**:论文时代共享前缀靠手动预留缓存块(§4.4,Fig.10);V1 实现**自动前缀缓存**——用 block hash 驱动的 `BlockHashToBlockMap`(`vllm/v1/core/block_pool.py`),同一前缀 hash 命中即共享物理块。论文只在 §6.4 的合成翻译负载下实测过共享前缀(1.67×/3.58×),未做真实多租户/系统提示词流量下的实测(账本 §8);自动 prefix cache 是工程演进——把手动预留变为默认开启的自动命中,而不是填补论文没做的实验。

3. **Chunked prefill(论文未提出)**:V1 scheduler 原生支持把长 prefill 切块调度(`scheduler.py` 含 prefill chunk limit 与 mamba 状态对齐切分),论文时代没有这个概念。另外注意力后端从"自研 kernel"走向多元化:`flash_attn`、`flashinfer`、`triton_attn`(vLLM 自研 paged attention kernel)、`flex_attention`、`mla`(DeepSeek 类)、`diffkv`、`gdn_attn`、`mamba`/`linear_attn` 等(`vllm/v1/attention/backends/`,2026-08-11)——**PagedAttention 的思想变成了引擎的默认内存层,而 kernel 层百花齐放**。

未核实项如实说明:V1 调度策略与论文 FCFS 描述的精确差异、CPU offload 默认开关,当前只以上述源码锚点为据,未做运行时验证。

## 9. 局限与选型边界

### 9.1 论文自己承认的边界

论文 §8 Discussion(PAGE 13)把适用边界写得很清楚:

1. **只对"输出长度先验未知 + 性能受显存容量约束"的负载有效**。DNN 训练张量形状静态、非 LLM serving 多为计算受限,强行套分页会因内存间接寻址与非连续块开销**降性能**;
2. **三个 LLM 特化增强**是它区别于 OS 分页的地方:all-or-nothing swap-out(处理请求需要全部 token state)、recompute 恢复(OS 里不可行)、kernel fusion 缓解间接寻址;
3. **kernel 慢 20-26%**(§7.1),靠其余算子不受影响与批处理收益弥补;
4. **block size 是权衡**:过大→内部碎片增加、共享概率下降(§7.2);
5. **优势有边界**:compute-bound 场景(OPT-175B + Alpaca)下相对 Orca 优势减弱(§6.2);
6. **baseline 限制**:Orca 未开源,三种变体为论文自实现,Oracle 被承认是实践不可行的上界(§6.1);
7. **未验证项**:真实多租户跨请求共享没有实验;swap 的绝对带宽数字未给出(账本 §8)。

### 9.2 牺牲什么,换取什么

| 换取 | 牺牲 |
| --- | --- |
| 显存利用率 96.3% vs 20.4%–38.2%,同卡并发 2-4×(峰值 22×)(Fig.2;§6.2) | attention kernel 慢 20-26%,间接寻址带来每块取数的常数开销(§7.1) |
| 按需分配,浪费 ≤ 一个块(§4.3) | block manager + block table 的维护复杂度,块分配/释放/拷贝都要进调度关键路径(§4.2;§5) |
| CoW 让并行采样/beam 共享前缀,省 37.6%–55.2% 内存(§6.3) | 写共享块要先拷贝,单序列路径多一次 memcpy 成本(§4.4) |
| recompute ≤ swap 20% 的低成本恢复(§7.3) | 抢占本身是 all-or-nothing 的,恢复前整个请求组停摆(§4.5) |
| 精确注意力、不影响精度(§1) | 无法像近似注意力那样进一步省内存——论文明确不碰这条路(§9 定位 FlashAttention 为正交的 tiling/I/O 优化) |

**一句话选型规则**:如果负载是"输出长度未知、并发受显存卡脖子"的在线 LLM serving,分页式 KV 管理是默认答案;如果输出长度已知、或系统早已 compute-bound,别为分页付间接寻址的税。

## 10. 面试官视角:三问三答

**Q1:"PagedAttention 凭什么让吞吐提升 2-4×?是 kernel 更快吗?"**

答:恰恰相反,kernel 比 FasterTransformer 慢 20-26%(§7.1)。2-4× 来自显存利用率:token states 占比从 20.4%–38.2% 提到 96.3%(Fig.2),同一张 A100 能同时服务的请求从 7 个涨到 30 个(Fig.13a),批处理让 GPU 始终吃满。**这是"省显存 → 提并发 → 提吞吐"的因果链,不是算子加速。** 顺带可以说:这也解释了为什么后来 vLLM 要把 kernel 换成 FA/flashinfer 后端——PagedAttention 管内存,attention kernel 负责算得快,两者正交。

**Q2:"block size 为什么默认 16?"**

答:两难权衡(§7.2):太大→最后一个块的内部碎片多、共享概率下降;太小→kernel 并行度不足、swap 时大量小传输低效。16 是论文消融的甜点:够大利用 GPU 并行,够小控制碎片;ShareGPT 上 16–128 都行,Alpaca 上 16/32 好。工程上这个默认值一直活到今天(V1 的 `DEFAULT_BLOCK_SIZE = 16`)。答完可以补一句:block size 还影响抢占恢复策略——小 block 时 swap 碎传输低效,recompute 更优(≤ swap 的 20%)。

**Q3:"swap 和 recompute 怎么选?"**

答:看瓶颈在哪(§4.5):swap 成本 ≈ CPU-GPU 带宽 × 数据量,recompute 成本 ≈ GPU 算力 × 一次 prefill。论文实测 recompute 开销 ≤ swap 延迟的 20%(§7.3),因为被抢占请求的解码 token 可以拼回 prompt 一次重算,比逐 token 重放便宜得多;block size 16–64 时两者端到端相当。工程上的现代倾向是把 preemption 当作"最后的兜底",先靠好的调度避免它。

## 📝 总结

1. **问题量化**:KV cache 是 serving 的内存税——OPT-13B 每 token 800 KB(2×5120×40×2),单请求上限 1.6 GB(§3);现有系统 token states 只占 20.4%–38.2%(Fig.2)。
2. **核心思想**:像 OS 分页一样管 KV cache——固定大小 KV block,逻辑连续、物理随意,block table 记录映射(§4.1-4.2)。
3. **三类浪费被结构性地消除**:不预留(按需分配)、内部碎片收敛到最后一个块(≤15 槽 vs 2038 槽)、等大块消灭外部碎片(§4.3;Fig.3)。
4. **分块注意力与标准注意力数值等价**:Eq.(4) 只是把按位置求和改成按块求和,论文声明不影响精度(§1;§4.1)。
5. **CoW 解锁共享**:ref count + 写时拷贝,beam search 省 37.6%–55.2% KV 内存,并行采样 6.1%–30.5%(§6.3)。
6. **抢占双路径**:FCFS + all-or-nothing,恢复可选 swap 或 recompute,后者开销 ≤ 前者的 20%(§7.3)。
7. **收益口径**:2-4× 吞吐来自显存利用率兑换的并发(7→30 请求),不是 kernel 更快——kernel 反而慢 20-26%(§7.1;§6.2)。
8. **峰值数字**:ShareGPT 上相对 FasterTransformer 最多 22×,共享前缀翻译场景 1.67×/3.58×(§6.2;§6.4)。
9. **边界诚实**:compute-bound 场景优势减弱;训练与计算受限负载不适合分页(§8;§6.2)。
10. **工程传承**:block size 16、ref count/CoW、抢占原语从 SOSP '23 活到 2026 年的 V1 引擎;自动前缀缓存与 chunked prefill 是论文之后补上的拼图(V1 源码,访问日期 2026-08-11)。

## 🎯 自我检验清单

- 能手算 OPT-13B 每 token 的 KV cache 大小(2×5120×40×2 ≈ 800 KB)并推出 2048 token 请求的 1.6 GB 上限。
- 能说出 65% 参数 / 超过 30% KV cache 的显存布局,并解释 KV cache 为什么是 serving 里唯一可省的大头。
- 能准确复述 Fig.2 的四类占比:Orca (Max) 20.4% vs vLLM 96.3%,以及三种 Orca baseline 各自的预留策略。
- 能用 Fig.3 的 2038/507 槽解释 internal fragmentation,并对比分页方案"浪费 ≤ 一个块"。
- 能画出 block table 的翻译过程:3 个逻辑块 → 物理块 7/1/3,第 4 个逻辑块尚未分配,表项含 `#filled` 计数。
- 能写出 Eq.(4) 分块注意力并解释 $\lceil i/B\rceil$、$\mathbf{1}$、$V_jA_{ij}^{\top}$ 各自的语义。
- 能解释"kernel 慢 20-26% 但端到端快 2-4×"为什么不是矛盾:批处理收益大于 kernel 常数开销。
- 能讲清 CoW 三步(分配新块 → 拷贝 → ref count −1),并给出 beam search 37.6%–55.2% 的节省数字。
- 能比较 swap 与 recompute 的决策变量(带宽 vs 算力),并复述 recompute ≤ swap 20% 的经验线。
- 能背出主结果表:2-4×、1.7×–2.7× / 2.7×–8× / 22×、2.2× / 4.3×、30.42 / 132.44,并说清各自的比较对象与负载。
- 能列出分页方案的适用边界(输出长度未知 + 显存受限)与不适用场景(训练、compute-bound)。
- 能说出论文之后 vLLM 工程补了哪三块拼图(自动前缀缓存、chunked prefill、多注意力后端),以及 block size 16 为何至今未变。

## 📚 参考资料

- 原文:
  - [Efficient Memory Management for Large Language Model Serving with PagedAttention(arXiv:2309.06180)](https://arxiv.org/abs/2309.06180):本文精读对象,SOSP '23 正式版(DOI 10.1145/3600006.3613165);账本核对的是 arXiv v1 PDF(16 页,无附录)。
- 官方实现:
  - [vllm-project/vllm](https://github.com/vllm-project/vllm):论文的官方实现仓库;本文工程对照核对 main 分支(2026-08-11,release v0.27.0),`vllm/v1/core/` 为 V1 引擎核心。
- 当前工程:
  - [vLLM V1 KV cache manager](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py) 与 [BlockPool / BlockHashToBlockMap](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py):自动前缀缓存与块池的当前实现。
  - [vLLM 注意力后端目录](https://github.com/vllm-project/vllm/tree/main/vllm/v1/attention/backends):flash_attn / flashinfer / triton_attn / mla 等多后端现状(访问日期 2026-08-11)。
- 站内相关:
  - [2.1 PagedAttention 分页注意力](/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/21-pagedattention):本文的工程落地篇,补 kernel 级实现细节。
  - [2.2 Continuous Batching 连续批处理](/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/22-continuous-batching):vLLM 与 Orca 共享的迭代级调度,是 2-4× 的另一半来源。
  - [2.3 Prefix Cache 与 RadixAttention](/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/23-prefix-cache-与-radixattention):论文第 4.4 节"共享前缀"的现代工程形态。
  - [2.4 Chunked Prefill 与统一调度](/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/24-chunked-prefill-与统一调度):论文未提出、V1 引擎补上的长 prefill 切块机制。
  - [6.2 FlashAttention-2 详解](/AIInfraGuide/cuda/模块二-cuda编程与算子优化/62-flashattention-v2详解):与 PagedAttention 正交的 tiling/I/O 优化,当前 vLLM 注意力后端的基础。
