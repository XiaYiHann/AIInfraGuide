---
title: "AIInfraGuide 文章更新日志"
description: "记录 AIInfraGuide 知识库的每一次内容更新，方便读者追踪最新变化"
pubDate: 2026-04-16
tags: ["公告", "更新日志"]
---

本文持续记录 AIInfraGuide 知识库的内容更新，按时间倒序排列，方便大家了解最新动态。

---

## 2026-08-20

### 九篇存文质量修复轮（公式走查补齐 + 数字核验 + 术语首现清零）

对前几批上线的 9 篇文章做了一轮逐条验收门体检（writer agent，DeepSeek V4 Flash）：对照一手源逐式逐表核验数字、补齐缺失的最小数值/矩阵走查、清理术语裸奔与悬空链接。重点修复：

- **3.10 MLA**（14 处）：压缩率数字对齐论文（93.3%）、RoPE/FP16/MoE 等首现定义补齐、低秩路径推导补全、3.3 四件套回链
- **12.1 InstructGPT**：修正一处算错的 next-token 损失（0.916→1.022，复算验证）、2 处数字对齐 arXiv 2203.02155、reference 必须冻结的三条理由
- **12.2 DPO**：chosen/rejected 首现定义、「令 lnZ=ln3」演示值与「Z 未知」的混淆澄清、Tülu 2 偏好数据六维对齐
- **3.12 MoE**：noisy top-k 采样算例、softmax gating 数值走查、「参数×512 而 FLOPs≈×1.06」双账本复算、99.994% 稀疏度公式
- **2.6 StreamingLLM**：Table 1/2/6 全数字对照 arXiv v4 核验（含甄别 Figure 10 曲线读数与正文数字的出处）
- **6.7 DeltaNet**：对照 arXiv 2406.06484 逐项核验 22 项数字（零错误），补术语首现与走查
- **3.11 DSA**：FLOPs 首现定义、index score 非负条件数学严谨性修正（抽查确认公式义务/数值走查达标）
- **1.2 测试时计算**：Roofline/Prefill/KV Cache 回顾块、Chinchilla 四件套（补齐与 chinchilla-notes 的对称交叉引用）、FLOPs/SFT 首现
- **chinchilla-notes**：两处悬空占位链接清理、Kaplan 批判段逻辑断裂修复（公式义务与推导完整度本已达标，未动结构）

### CS8803 第三批四篇完稿 + 四处存量补充（12.3/12.4/8.6/Precision + 3.9/3.3/3.8）

CS8803 扫描收尾批（新图 10 张入站；模块三第 12 章 12.3/12.4 上线、全章 4 篇齐整；另在 3 篇存量文章补了 4 处缺失块）：

- **文章：** [12.3 从 PPO 到 GRPO 再到 DAPO：LLM 后训练算法链与算力/显存账本](https://xiayihann.github.io/AIInfraGuide/distributed/模块三-分布式训练/123-grpo与dapo后训练算法链) | PPO→DPO→GRPO→DAPO 算法链：GRPO 组内相对优势消掉 critic（G=4 数值走查+无偏 baseline 论证）、DAPO 四技巧（Clip-Higher/动态采样/token 级 loss/长度塑形）、四算法模型副本与在线采样账本对比表
- **文章：** [12.4 零数据 RLVR：可验证奖励与 Absolute Zero 自博弈](https://xiayihann.github.io/AIInfraGuide/distributed/模块三-分布式训练/124-rlvr与absolutezero自博弈) | Absolute Zero 精读：自生成任务循环（模型出题→程序化 checker 验证→reward，配具体算例）、RLVR 谱系、自博弈训练额外开销与任务塌缩边界、Qwen2.5-7B-Coder 基准数字（按 arXiv 2505.03335 核验）
- **文章：** [8.6 小模型先答，答不了再升级：模型置信度与 LLM Cascade 路由](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第8章-生产级服务特性/86-模型置信度与llm-cascade路由) | Confidence Tokens 精读（源核验：真源为 arXiv 2410.13284 Self-REF，初标 2412.10148 实为凝聚态物理论文，文中已透明记录）：[confidence] 词表扩展与自报置信度训练配方、cascade 路由成本公式+数值走查、与 logprob 路线对比表、质量-成本 knee point
- **文章：** [precision-notes：训练该用几位精度？精度 × 参数 × 数据三方权衡](https://xiayihann.github.io/AIInfraGuide/papers/scaling-laws-for-precision-notes) | Precision Scaling Laws 精读（源核验：真源 arXiv 2411.04330+课程幻灯片）：b 位宽作为第三分配变量的 scaling 理论、最优精度解推导+175B@FP16 vs FP8 最小算例、与 Chinchilla 的正交性、训练/推理精度分工
- **补充：** [3.9 Tokenization 与词嵌入](https://xiayihann.github.io/AIInfraGuide/prerequisites/模块一-前置知识/transformer/39-tokenization与词嵌入) 新增 2.4「BPE 学习流程」（超小语料 4 轮 pair 计数→合并手算走查）+ 2.5「Byte-level BPE 与 UTF-8 字节 fallback」（无 UNK、多语种 token 数差异算例）+ 第 7 节「从词嵌入到文本嵌入：语义检索入门」（余弦算例、InfoNCE 三要素、E5 精简案例）
- **补充：** [3.3 Self-Attention §9.5](https://xiayihann.github.io/AIInfraGuide/prerequisites/模块一-前置知识/transformer/33-self-attention机制深入理解#951-absorb吸收把固定上投影折叠到-query-侧) 新增 Absorb 推理机制（2×2 折叠走查、行为≈MQA、K 侧 RoPE-free 前提）+ RoPE/NoPE 拆分 + TransMLA 一句话延伸；[3.8 §1.3](https://xiayihann.github.io/AIInfraGuide/prerequisites/模块一-前置知识/transformer/38-从transformer到llm自回归生成深入理解) 更新过时的 NAR 结论为 masked diffusion（前向掩码/反向并行预测+remask 小例、AR vs LLaDA 成本对比表）

### CS8803 第二批四篇完稿：DSA / RLHF / DPO / Chinchilla（模块三第12章后训练开章）

CS8803 课程扫描第二批（数字全部对照 arXiv 一手源核验，新图 12 张入站；模块三新建第 12 章「LLM 后训练与 RL 训练」）：

- **文章：** [3.11 从 O(L²) 到 O(Lk)：DeepSeek 稀疏注意力 DSA 详解](https://xiayihann.github.io/AIInfraGuide/prerequisites/模块一-前置知识/transformer/311-deepseek稀疏注意力dsa详解) | DeepSeek-V3.2 §2 精读：KV 压缩≠计算降低的 MHA/MQA/GQA/MLA/DSA 复杂度总表、Lightning Indexer FP8 低秩打分公式与 top-k 最小走查、为什么必须建在 MLA 之上、Dense Warm-up → top-k 稀疏两阶段训练配方与 detached indexer vs STE 取舍
- **文章：** [12.1 RLHF 的四模型成本：InstructGPT 三阶段流水线与 PPO 训练回路](https://xiayihann.github.io/AIInfraGuide/distributed/模块三-分布式训练/121-rlhf三阶段流水线与instructgpt) | InstructGPT 精读：next-token 与人类意图错位的证据、Bradley-Terry 奖励模型损失逐项推导+最小数值走查、PPO 训练回路（KL 惩罚与 reference 副本）、policy/reference/RM/critic 四份模型副本显存账+在线采样算力账、1.3B 胜 175B 的证据与三种失败模式
- **文章：** [12.2 去掉奖励模型和 PPO 采样回路：DPO 闭式推导与偏好数据工程](https://xiayihann.github.io/AIInfraGuide/distributed/模块三-分布式训练/122-dpo直接偏好优化) | DPO 闭式推导三步（KL 约束最优策略 π*∝π_ref·exp(r/β) → 奖励重参数化配分函数相消 → 二元交叉熵损失）每步带数值走查、PPO vs DPO 工程对比（四副本→两副本、在线采样→离线偏好）、UltraFeedback 合成数据与 RM 规模/Best-of-N 配方
- **文章：** [chinchilla-notes：算力预算固定时，模型该多大、训多少 token？](https://xiayihann.github.io/AIInfraGuide/papers/chinchilla-notes) | Chinchilla 精读：FLOPs≈6ND 逐项、三种互补估计（训练曲线包络 a≈b≈0.5 / IsoFLOP U 型谷底 / 参数化拟合）、N_opt∝C^0.5 ≈ 20 tokens/parameter 最小算例、Kaplan 2020 大模型优先路线的方法论批判与 70B/1.4T vs 280B/300B 同 FLOPs 实证
- **章索引：** [第12章 LLM 后训练与 RL 训练](https://xiayihann.github.io/AIInfraGuide/distributed/模块三-分布式训练/第12章-llm后训练与rl训练) | 模块三新开后训练章（12.1/12.2 已上线；12.3 GRPO/DAPO、12.4 RLVR/Absolute Zero 撰写中）

## 2026-08-19

### CS8803 首批五篇完稿：MLA / MoE / Attention Sink / 测试时计算 / DeltaNet

从 CS8803-LLM 课程扫描中挑出的 5 篇高优先新文章（论文精读+知识点精讲，数字全部对照 arXiv 一手源核验，新图 21 张入站；同批在 7 篇存量文章里加了交叉引用）：

- **文章：** [3.10 KV Cache 为什么能省 93%？MLA 低秩压缩与解耦 RoPE](https://xiayihann.github.io/AIInfraGuide/prerequisites/模块一-前置知识/transformer/310-mla与decoupled-rope精读) | DeepSeek-V2 §2.1 精读：低秩联合压缩 + 吸收 trick 把 KV 压到单一 latent 的逐步推导、解耦 RoPE（旋转只作用于 c_t 不破坏压缩）、93.4% KV 节省的最小数值走查与和 GQA 的路线对比
- **文章：** [3.12 参数扩 1000 倍但计算不变：MoE 的诞生（Shazeer 2017 精读）](https://xiayihann.github.io/AIInfraGuide/prerequisites/模块一-前置知识/transformer/312-moe-sparsely-gated-moe论文精读) | Sparsely-Gated MoE 精读：noisy top-k 门控（采样噪声的负载均衡作用）、router loss 与 aux-loss 系数、capacity factor=2 的溢出机制、为什么参数×1000 而 FLOPs 不变的账本
- **文章：** [2.6 滑动窗口把 KV 一驱逐模型就崩？Attention Sink 与 StreamingLLM](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/26-streamingllm与attention-sink) | StreamingLLM 精读：为什么驱逐初始 token 后 PPL 从 2.15 爆到 677（softmax 需要一个"落脚点"）、attention sink 四窗口 vs 全量 KV 的流式对账、固定显存下的无限长上下文工程化
- **文章：** [1.2 同样的 FLOPs 给预训练还是给推理？测试时计算与最优分配](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第1章-llm推理基础/12-测试时计算与推理算力分配) | Snell 2024 精读：compute-optimal 分配定律（α≈1 幂律）、"10× 搜索"为什么通常不是最优（训练/推理算力重新分配的边际收益）、最优推理算力随能力增长而缩小的反直觉结论
- **文章：** [6.7 不构造 L×L 矩阵：线性注意力、Delta Rule 与分块并行（DeltaNet 精读）](https://xiayihann.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/67-线性注意力与deltanet精读) | DeltaNet 精读：delta rule 为什么能当"线性注意力+梯度下降"、从 O(L²) 到 O(L) 的状态递推推导、WY 表示与分块并行（chunk 内并行 + chunk 间串行）的 GPU 实现路径与 SSM 的边界

### 3.4–3.7 四篇完稿：第3章 7 篇全部齐整

第3章补齐剩余四篇（3.1 的 frontmatter 顺序/标题编号同步修正），至此本章 7 篇全部上线：

- **文章：** [3.4 vLLM V1 引擎深度解析](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-深入vllm/34-v1引擎深度解析) | 冷启动五步序列（加载权重 → profile 测峰值 → 定 KV 池 → 分配 KV → 抓 CUDA Graph，实测 init 24.78 s 里大头是 102 张图）、KV 池定容公式与实测对账、CUDA Graph 的 FULL/PIECEWISE 分工、async scheduling（0.26.0 默认开）与零开销 Prefix Cache 的三机制
- **文章：** [3.5 vLLM 调度器源码导读](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-深入vllm/35-调度器源码导读) | schedule() 五块结构逐段导读、抢占（recompute）只由 running 循环触发、严格全序列准入（`scheduler_reserve_full_isl`）让抢占默认几乎不触发；附强制触发抢占的最小复现（Prometheus `vllm:num_preemptions_total` 实测 0 vs 2，prefix cache 对照实验 0.5 s）
- **文章：** [3.6 vLLM 关键配置调优](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-深入vllm/36-关键配置调优) | 内存/调度/执行三组旋钮逐项拆解（默认值源码出处 + 生效机制 + 副作用）、本机解析默认值 dump（batch 默认随入口×显存档位分叉）、"症状→旋钮"决策表
- **文章：** [3.7 推理框架横向对比](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-深入vllm/37-框架横向对比) | vLLM/SGLang/TensorRT-LLM 的三种架构赌注（宽兼容 vs RadixAttention 前缀复用+MoE vs 编译引擎）、prefix cache 与调度器两种实现路线深挖、2026 第三方基准数据（带硬件/版本/负载口径）与选型决策表

### 3.3 vLLM 整体架构（新增）

- **文章：** [3.3 vLLM 整体架构：一次请求如何穿过 LLMEngine、EngineCore 和 Worker](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-深入vllm/33-vllm-整体架构) | 以 vLLM 0.26.0 真实源码为对象拆解 V1 引擎三层进程架构：LLM/AsyncLLM 前端与 EngineCore 进程之间的 ZMQ 边界、`run_busy_loop` 主循环与 `step()` 心脏五步、调度器"无 prefill/decode 阶段"的统一分配与抢占=重算、KV 池启动时预分配与 prefix cache 零开销命中（实测 68-token prompt 第二轮命中 64 token）、Model Runner V1/V2 版本差异；附 2×RTX 6000D 最小复现（进程树 `VLLM::EngineCore`、GPU KV cache 1,012,240 tokens、默认参数 16,384/1,024/16）

### 5.7 DFlash 补充：DFlash 2（Inco AI 博客，2026-08）

- **文章：** [5.7 DFlash：块扩散并行起草](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/57-dflash-块扩散并行起草) | 新增第 7 节"并行起草留下的两个缺口：DFlash 2 的答案"——路径选择器（top-16 候选成对打分，+2M 参数/+0.6% 延迟）与二抽头局部卷积（+16.5M/+0.7%）分别修复选择头空间与后缀衰减，合计比 DFlash 接受长度 +21% 只花 1.3% 循环延迟（博客自测）；同步更新生态采用面（NVIDIA Blackwell 15×、Google TPU 3×、CoreWeave 生产默认、HF 下载 350 万+）；5.8 DSpark 加了一条路线对照注

## 2026-08-18

### 模块四第5章补全：MTP / DFlash / DSpark 三篇生产级投机解码

第 5 章从 5 篇扩充到 8 篇，把 2024-2026 年的三代生产级投机解码补进推理章节（数字全部对照 arXiv 一手源核验，论文原图 6 张入站）：

- **文章：** [5.6 MTP：模型原生的多 Token 预测](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/56-mtp-模型原生的多token预测) | DeepSeek 内置草稿器：MTP 模块结构与训练损失逐项拆解、Vanilla/Eagle 双路径、KV 回滚与宽松接受（R1-FP4 8 卡实测 2.16×~2.33×）与 V4 生产 MTP-1 现状
- **文章：** [5.7 DFlash：块扩散并行起草](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/57-dflash-块扩散并行起草) | ICML 2026 块扩散草稿器：T_draft=γ·t_step 天花板、KV 注入目标特征、随机锚点与位置衰减训练（Qwen3-8B greedy MATH-500 6.08×、SGLang B200 serving 5.1×）
- **文章：** [5.8 DSpark：置信度调度的生产级投机解码](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/58-dspark-置信度调度的生产级投机解码) | DeepSeek 生产方案：后缀衰减机制、Markov 头（0.2%~1.3% 延迟换 30% 接受长度）、置信度头+STS 校准+硬件感知调度器（V4 线上匹配吞吐单用户 +60%~85%，+661% 口径的正确读法）与 DeepSpec 落地

### Paper 精读新增

- **文章：** [Cross-Model KV Transfer 解读](https://xiayihann.github.io/AIInfraGuide/papers/cross-model-kv-transfer-notes) | 模型切换不再重读上下文：跨模型 KV 的线性结构（单源层 56%/32% 方差→多层 79%/65%）、closed-form per-head ridge mapper 三步（top-k 源层/去 RoPE/500 条校准）、6 对模型 4 对保留 73-98% 且 prefill 提速 2.7-25×、误差落点诊断（attention-output cosine r=+0.57）

---

## 2026-08-12

### 模块一第 6 章补全：集合通信基础

第 6 章从 1 篇补齐为完整章节(章首页 + 6 篇)：将原《集群通信网络与NCCL》大杂烩拆分为 6 篇新文章并大幅扩展（deepseek-v4-flash workflow 写作+审计，公式与 NCCL 环境变量对照官方文档核实）：

- **文章：** [6.1 集合通信原语](https://xiayihann.github.io/AIInfraGuide/prerequisites/模块一-前置知识/communication/61-集合通信原语) | 点对点 Send/Recv 与五类集合原语的数据流、通信量与适用场景
- **文章：** [6.2 Ring AllReduce](https://xiayihann.github.io/AIInfraGuide/prerequisites/模块一-前置知识/communication/62-ring-allreduce) | 带宽最优原理、2(N-1)/N 通信量推导与数值算例
- **文章：** [6.3 Tree AllReduce 与算法选型](https://xiayihann.github.io/AIInfraGuide/prerequisites/模块一-前置知识/communication/63-tree-allreduce与算法选型) | Tree 原理、Ring vs Tree 对比、分层/双通道变体
- **文章：** [6.4 通信与计算 Overlap](https://xiayihann.github.io/AIInfraGuide/prerequisites/模块一-前置知识/communication/64-通信与计算-overlap) | 依赖分析、分块流水、异步执行
- **文章：** [6.5 NCCL 实战](https://xiayihann.github.io/AIInfraGuide/prerequisites/模块一-前置知识/communication/65-nccl实战) | 库定位、基本用法、环境变量调优、nccl-tests 压测
- **文章：** [6.6 通信视角的并行策略](https://xiayihann.github.io/AIInfraGuide/prerequisites/模块一-前置知识/communication/66-通信视角的并行策略) | TP/PP/DP 通信模式与通信量对比，衔接模块三

原大杂烩已拆分删除；NVLink/InfiniBand 硬件细节不重复(指向 5.7)；每篇 2 张 Mermaid 图配中文导读。

---

## 2026-08-12

### 模块一第 5 章补全：GPU 硬件概论

第 5 章从 2 篇补齐为完整章节(7 篇)：将原大杂烩《GPU基础知识》拆分为 6 篇新文章并大幅扩展（deepseek-v4-flash workflow 写作+审计，规格数字对照官方资料核实）：

- **文章：** [5.2 GPU 架构总览](https://xiayihann.github.io/AIInfraGuide/prerequisites/模块一-前置知识/gpu/52-gpu架构总览) | CPU vs GPU 设计哲学、SM/CUDA Core/Tensor Core 层次、Warp 调度
- **文章：** [5.3 存储层次与 Memory Wall](https://xiayihann.github.io/AIInfraGuide/prerequisites/模块一-前置知识/gpu/53-存储层次与memory-wall) | 寄存器→共享→L1/L2→HBM→主机，带宽/延迟量级，为什么带宽先成瓶颈
- **文章：** [5.4 Tensor Core 与 AI 加速](https://xiayihann.github.io/AIInfraGuide/prerequisites/模块一-前置知识/gpu/54-tensor-core与ai加速) | 4×4 MMA 原理、FP16/BF16/FP8、GEMM 硬件基础
- **文章：** [5.5 主流 AI GPU 规格对比与 Roofline](https://xiayihann.github.io/AIInfraGuide/prerequisites/模块一-前置知识/gpu/55-主流ai-gpu规格对比与roofline) | A100→B200 参数表、Arithmetic Intensity、Roofline Model
- **文章：** [5.6 显存管理基础](https://xiayihann.github.io/AIInfraGuide/prerequisites/模块一-前置知识/gpu/56-显存管理基础) | 训练显存账本（18B/参数）、显存优化策略的硬件视角
- **文章：** [5.7 多卡互联拓扑](https://xiayihann.github.io/AIInfraGuide/prerequisites/模块一-前置知识/gpu/57-多卡互联拓扑) | NVLink/NVSwitch/InfiniBand、topo -m 实战、拓扑→并行策略

原《GPU基础知识》大杂烩已拆分删除；每篇含 2 张 Mermaid 图（配中文导读）、公式三件套；规格数字标注来源口径。

---

## 2026-08-09

---

## 2026-08-09

### 第 12 章新增：TensorRT-LLM 推理引擎

新增完整章节(章首页 + 六篇,由 deepseek-v4-flash agent 写作与审计,事实逐条对照官方仓库 docs/ 核实;原 TensorRT 核心版已整体替换):

- **章首页：** [第 12 章 TensorRT-LLM 推理引擎](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第12章-tensorrt推理引擎/第12章-tensorrt推理引擎)
- **文章：** [12.1 TensorRT-LLM 是什么](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第12章-tensorrt推理引擎/121-tensorrt-llm是什么) | PyTorch 原生架构与 PyExecutor、与 TensorRT/vLLM 的分野、版本节奏
- **文章：** [12.2 从权重到引擎](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第12章-tensorrt推理引擎/122-从权重到引擎模型加载量化与构建) | 模型加载三条路线、ModelOpt 离线量化、1.2 起构建新含义
- **文章：** [12.3 运行时与 LLM API](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第12章-tensorrt推理引擎/123-运行时与llm-api) | PyExecutor 五步循环、trtllm-serve 六端点、In-flight Batching
- **文章：** [12.4 KV Cache 与长上下文](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第12章-tensorrt推理引擎/124-kv-cache与长上下文) | 分页缓存与 radix tree、缓存量化、长序列三件套、PD 解耦
- **文章：** [12.5 并行、量化与高级特性](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第12章-tensorrt推理引擎/125-并行量化与高级特性) | TP/PP/DP/EP/CP/Wide-EP、FP8/FP4 矩阵、投机解码、稀疏注意力
- **文章：** [12.6 生态选型与部署](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第12章-tensorrt推理引擎/126-生态选型与部署) | 与 vLLM 对比、官方部署指南、Triton/Dynamo、Slurm 多节点、模块四总结

---

## 2026-08-09

### 第 3 章新增

- **文章：** [3.2 nano-vllm 源码导读：用 1200 行读懂 vLLM 核心机制](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-深入vllm/32-nano-vllm源码导读) | 逐文件读通 nano-vllm（约 1200 行）的双队列调度、block 记账 KV cache、xxhash 前缀缓存、CUDA graph 与极简张量并行，建立 vLLM 最小心智地图（含 2 张 mermaid 图，README 自报基准 1434 tok/s）

---

## 2026-08-09

### Paper 精读新增

**精读文章配图**(22 张论文原图,均标注图源):DeepSeek-V4 5 张(架构/CSA/效率基准/KV 布局/长上下文)、Kimi K3 5 张(架构/KDA/缩放定律/RL/成本效率)、GLM-5 5 张(训练管线/DSA/基准/长程任务/上下文管理)、Gemma 4 2 张(MTP/视觉分辨率,论文仅此两图)、Kimi K2.5 5 张(Agent Swarm/训练消融/视觉 RL/并行加速/系统组件)。


**2026 年模型技术报告精读系列**(五篇,用 tech-report-writing 产出,高中生版):

- **文章：** [DeepSeek-V4 解读](https://xiayihann.github.io/AIInfraGuide/papers/deepseek-v4-notes) | 百万 token 上下文背后的架构与训练账本：CSA+HCA 混合注意力、mHC、Muon 优化器、V4-Pro 1.6T / V4-Flash 284B
- **文章：** [Kimi K3 解读](https://xiayihann.github.io/AIInfraGuide/papers/kimi-k3-notes) | 2.8T 参数开源前沿模型是怎么造出来的：MoE 配置、47 页训练配方、评测双口径
- **文章：** [GLM-5 解读](https://xiayihann.github.io/AIInfraGuide/papers/glm-5-notes) | 从"一句话编程"到能自主干活 8 小时的 Agent 模型：744B MoE、DSA 稀疏注意力、全异步 Agentic RL
- **文章：** [Gemma 4 解读](https://xiayihann.github.io/AIInfraGuide/papers/gemma-4-notes) | Google 开源多模态家族的架构与取舍：密集/MoE 双线、长上下文四件套、无编码器路线
- **文章：** [Kimi K2.5 解读](https://xiayihann.github.io/AIInfraGuide/papers/kimi-k25-notes) | 会"指挥一群小助手"的多模态 Agent：Agent Swarm 自导向并行、DEP 训练解耦

### 站点更新

**新增 Paper 精读栏目** 📖：论文与技术博客的中文通俗精读（用 tech-report-writing 工作流产出，保留原文全部关键事实，补上直觉与上下文）：

- **文章：** [Assisted Generation 解读：让 1/10 大小的小模型先写草稿，生成延迟最高降 10 倍](https://xiayihann.github.io/AIInfraGuide/papers/assisted-generation-notes) | 用“实习生起草、主编审阅”的框架拆解 HuggingFace 的 Assisted Generation：为什么自回归慢在“搬”不在“算”，三档实测加速（2x/3x/10x）各自的前提，以及与 Batching、张量并行三条路线的账本对比

**阅读体验升级**：正文限宽 75ch（超宽屏不再拉长行）、顶部阅读进度条、返回顶部按钮、阅读时长估算、移动端折叠目录、表格/公式防溢出、中文字体栈优化、打印样式。

### 新增内容

**模块四 · 推理优化** 第 5~11 章全部补齐（共 30 篇子文章），推理模块 11 章全部完成：

- **第 5 章 · Speculative Decoding**：[5.1 核心原理与 Rejection Sampling](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/51-核心原理与rejection-sampling)、[5.2 Draft 模型与 N-gram 方案](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/52-draft模型与n-gram方案)、[5.3 Medusa 与 EAGLE](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/53-self-draft方案-medusa与eagle)、[5.4 收益边界与限制](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/54-收益边界与限制)、[5.5 vLLM 投机解码实战](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/55-vllm投机解码实战)
- **第 6 章 · 分布式推理与大模型部署**：[6.1~6.6](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/61-推理并行策略总览)：推理并行总览、张量并行、流水线并行、数据并行与专家并行（MoE）、Ray 多节点、vLLM 分布式实战
- **第 7 章 · PD 解耦架构**：[7.1~7.6](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第7章-pd解耦架构/71-混合batching互扰分析)：互扰分析、DistServe/Splitwise/TaiChi、KV 传输与 NIXL Connector、Goodput、配比推导、vLLM 解耦实战
- **第 8 章 · 生产级服务特性**：[8.1~8.5](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第8章-生产级服务特性/81-结构化输出)：结构化输出、Tool Calling、Multi-LoRA、多模态 VLM、采样与解码算法
- **第 9 章 · 性能分析与 Benchmark**：[9.1~9.5](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第9章-性能分析与benchmark/91-推理指标体系)：指标体系、压测工具（vllm bench/GenAI-Perf）、性能分析工具、MLPerf 基准、回归门禁
- **第 10 章 · 生产部署与运维**：[10.1~10.4](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第10章-生产部署与运维/101-容器化与kubernetes部署)：K8s 部署、可观测性、扩缩容与负载均衡、容量规划
- **第 11 章 · 选型与端到端实战**：[11.1 决策树](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第11章-推理优化选型与端到端实战/111-优化选型决策树)、[11.2 优化组合](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第11章-推理优化选型与端到端实战/112-优化组合与叠加顺序)、[11.3 端到端实战](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第11章-推理优化选型与端到端实战/113-端到端部署实战)、[11.4 模块总结](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第11章-推理优化选型与端到端实战/114-模块总结与持续学习)——推理模块收官

---

## 2026-08-09

### 新增内容

**模块二 · CUDA 编程与算子优化** Attention 与 AI 编译器两章补齐：

- **文章：** [6.3 FlashAttention-3 详解](https://xiayihann.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/63-flashattention-3详解) | 拆解 FA3 的三大核心设计——Warp Specialization 分工、把 Softmax 藏进 GEMM 流水线、FP8 低精度与数值稳定，讲清 H100 利用率从 35% 到 75% 的路径与工程可用性

- **文章：** [7.2 torch.compile 原理与 Graph Break](https://xiayihann.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/72-torchcompile原理与graphbreak) | TorchDynamo 图捕获与 TorchInductor 代码生成，Graph Break 的成因、诊断（TORCH_LOGS / torch._dynamo.explain）与避免方法，三种编译模式的取舍与性能预期

**模块四 · 推理优化** 第 4 章（量化）系列六篇全部上线：

- **文章：** [4.1 量化基础](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/41-量化基础) | 为什么量化、定点表示与对称/非对称、量化粒度（per-tensor/per-channel/per-group）、PTQ vs QAT
- **文章：** [4.2 W8A8 量化：SmoothQuant](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/42-w8a8-smoothquant) | Activation Outlier 难题与数学等价变换，把量化难度从激活迁移到权重
- **文章：** [4.3 Weight-only INT4：GPTQ 与 AWQ](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/43-weight-only-int4-gptq与awq) | Hessian 驱动的 GPTQ 与激活感知的 AWQ，Marlin Kernel 与 INT4 不一定比 INT8 快的反直觉点
- **文章：** [4.4 KV Cache 量化：KIVI](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/44-kv-cache量化-kivi) | per-channel 量化 + 滑动窗口残差的 2-bit KV 量化：4 倍 KV 压缩、峰值显存省 2.6x、batch 最大 4x，以及 vLLM 原生 FP8 KV Cache 的 2x 捷径
- **文章：** [4.5 FP8 与 NVFP4](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/45-fp8与nvfp4) | E4M3/E5M2 格式、Hopper FP8 Tensor Core 与 FP8 训练，Blackwell 的 NVFP4/MXFP4 低比特浮点
- **文章：** [4.6 量化选型与 vLLM 实战](https://xiayihann.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/46-量化选型与vllm实战) | 症状→诊断→方案决策树、70B 显存账本、vLLM 启用量化命令、可复用评测流程与故障排查清单

---

## 2026-08-08

### 新增内容

**模块二 · CUDA 编程与算子优化** 第 7 章（AI 编译器）首篇子文章上线：

- **文章：** [7.1 Triton 快速上手：Block-level 编程与 Fused Softmax 实战](https://xiayihann.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/71-triton快速上手) | 从 thread-level 切到 block-level 视角，用约 25 行代码实现 Fused Softmax，与 5.2 手写 CUDA 版本逐版对照，讲清编译器替你做了什么、Triton 牺牲什么换取什么

---

## 2026-07-09

### 新增内容

**模块四 · 推理优化** 第 1、2 章文章上线：

- **文章：** [1.1 LLM 推理基础](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第1章-llm推理基础/11-llm推理基础/) | 一篇讲透 LLM 推理的底层逻辑：Prefill/Decode 两阶段、KV Cache 显存账本、TTFT/TPOT 等性能指标，以及用 Roofline 解释为什么 Decode 是 Memory Bound

- **文章：** [2.1 PagedAttention](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/21-pagedattention/) | 深入 vLLM 的核心技术 PagedAttention：借鉴虚拟内存分页思想，用 Block Table 消除 KV Cache 碎片化，并支持前缀共享与 Copy-on-Write

- **文章：** [2.2 Continuous Batching](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/22-continuous-batching/) | 深入 Continuous Batching 的原理：Iteration-level Scheduling 让请求随到随拼、完成即退，把 GPU 利用率从三成拉到八成以上

- **文章：** [2.3 Prefix Cache 与 RadixAttention](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/23-prefix-cache-与-radixattention/) | 深入前缀缓存：vLLM 基于 Hash 的自动前缀缓存如何复用共享前缀的 KV 块，SGLang 的 RadixAttention 如何用 Radix Tree 做更高效的前缀共享

- **文章：** [2.4 Chunked Prefill 与统一调度](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/24-chunked-prefill-与统一调度/) | 深入 Chunked Prefill：把长 Prompt 的 Prefill 切块，消除它对 Decode 请求的干扰；理解 vLLM V1 如何用统一 Token 预算调度器抹平 Prefill/Decode 边界

- **文章：** [2.5 Attention 后端与图优化](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/25-attention-后端与图优化/) | 深入 vLLM 的可插拔 Attention 后端（FlashAttention/FlashInfer/FlashMLA/Triton）与图优化：用 CUDA Graph 和 torch.compile 消除 Decode 阶段的 CPU 启动开销

---

## 2026-07-07

### 新增内容

- **文章：**  [1.1 分布式训练总论：显存账本与五大并行策略全景](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/11-分布式训练总论/) | 手算训练显存账本，总览五大并行策略全景与选择原则

- **文章：**  [1.2 环境搭建与分布式启动](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/12-环境搭建与分布式启动/) | rank/local_rank/world_size、torchrun 启动与会合机制、NCCL 调试、最小 DDP 脚本

---

## 2026-07-06

### 新增内容

更新了分布式训练部分的教程大纲：
| 章节 | 主要内容 |
|------|----------|
| 第 1 章 分布式训练总论 | 为什么需要分布式训练、训练显存账本、五大并行策略全景、环境搭建与 DDP 启动 |
| 第 2 章 集合通信原语 | AllReduce/ReduceScatter/AllGather/All-to-All 语义、Ring 算法、通信量量化分析 |
| 第 3 章 优化器 | SGD/Adam/AdamW 演进、优化器状态显存开销分析、大 Batch 优化器 LAMB/LARS |
| 第 4 章 数据并行 | DataParallel、DistributedDataParallel、FSDP |
| 第 5 章 ZeRO 系列 | ZeRO-1/2/3 切分策略与通信代价、ZeRO-Offload/Infinity 异构内存卸载 |
| 第 6 章 张量并行与序列并行 | Megatron Column/Row Parallel Linear、通信插入位置、序列并行激活显存优化 |
| 第 7 章 流水线并行 | Bubble 问题本质、GPipe、1F1B、Interleaved 调度 |
| 第 8 章 混合精度与显存优化 | FP16/BF16/FP8 混合精度、梯度累积、Activation Checkpointing |
| 第 9 章 长序列训练与上下文并行 | Attention 的 O(s²) 困境、Ring Attention、DeepSpeed-Ulysses、Megatron Context Parallel |
| 第 10 章 MoE 并行 | Router 机制、Expert Parallelism 的 All-to-All 通信、EP×DP×TP×PP 组合与负载均衡 |
| 第 11 章 3D 并行与混合并行策略 | TP×PP×DP×EP×CP 拓扑设计、通信域映射、rank 编排、集群选型 |


---

## 2026-06-25

### 新增内容

- **文章：**  [6.1 FlashAttention V1详解](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/61-flashattention-v1详解) | 本文深入剖析 FlashAttention V1 的核心原理与实现细节，理解如何通过 Tiling + Online Softmax 将 Attention 的额外显存占用从 $O(N^2)$ 降至 $O(N)$，同时大幅减少 HBM 访问量，实现无精度损失的加速

- **文章：**  [6.2 FlashAttention V2详解](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/62-flashattention-v2详解) | 本本文深入剖析 FlashAttention V2 相比 V1 的核心改进：调换内外循环顺序、优化线程块内的工作分配、减少非矩阵乘运算

---

## 2026-06-06

### 新增内容

- **文章：**  [5.2 CUDA Online Softmax 实现优化](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/52-cuda-online-softmax实现) | 本文从算法推导出发，逐步实现并优化 Online Softmax 的 CUDA Kernel，这也是理解 FlashAttention 的核心前置知识

---

## 2026-06-05

### 新增内容

- **文章：**  [5.1 CUDA Softmax 朴素实现优化](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/51-cuda-softmax朴素实现优化) | 本文从朴素实现出发，逐步引入 Safe Softmax、Block 级并行、Warp Shuffle、向量化访存等优化手段

---

## 2026-06-02

### 新增内容

- **文章：** [4.1 CUDA GEMM算子性能优化](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/41-cuda-gemm算子性能优化) |本文从朴素实现出发，逐步Block-Warp-Thread三级Tiling优化、向量化访存、Bank Conflict 消除、双缓冲等优化手段，带你系统掌握 CUDA GEMM 优化的完整方法论

---

## 2026-05-18

### 新增内容

- **文章：** [3.1 CUDA Reduce算子优化](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/31-cuda-reduce算子优化) | Reduce（规约）是 GPU 编程中最基础、也最能体现并行思维的算子之一。本文从最朴素的实现出发，逐步引入 Warp 级原语、向量化访存、多元素处理等优化手段，每一步都有性能对比和原理解析，帮你真正搞懂"怎么写出快的 Kernel"。

---

## 2026-05-16

### 新增内容

- **模块三 · 分布式训练** 新增 [第2章：优化器](/AIInfraGuide/distributed/模块三-分布式训练/第3章-优化器) | 理解 SGD → Momentum → Adam → AdamW 的演进逻辑与内部状态组成，手算优化器显存开销（AdamW 占训练总显存 75%），掌握大 Batch 优化器 LAMB/LARS 的适用场景，为后续 ZeRO 显存优化和混合精度训练打下基础。

### 结构调整

- **模块三 · 分布式训练** 章节重新编号：原第2~7章顺延为第3~8章，新增第2章"优化器"插入总论与数据并行之间
- 同步更新 AI Infra 学习路线、README 中的模块三章节目录

---

## 2026-05-13

### 新增内容

- **文章：** [2.4 同步与原子操作](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/24-同步与原子操作) | 正确的同步机制是编写无 Bug 并行程序的基础。本文详解 CUDA 中的块内同步 `__syncthreads()`、Warp 级同步、Memory Fence，以及原子操作的使用场景、性能代价与优化技巧，帮助你在保证正确性的前提下写出高性能的并行代码。

---

## 2026-05-12

### 新增内容

- **文章：** [2.2 内存访问优化](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/22-内存访问优化) | 内存访问效率是 CUDA Kernel 性能最大的杠杆。本文深入讲解合并访问（Coalesced Access）的原理与判定方法、共享内存 Bank Conflict 的成因与 Padding 解决方案，以及向量化加载（float4/int4）提升带宽利用率的实战技巧。
- **文章：** [2.3 Occupancy 与资源分配](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/23-occupancy与资源分配) | Occupancy 衡量 SM 上实际活跃 Warp 数与理论最大值的比例，是调优 CUDA Kernel 的核心指标之一。本文讲解 Occupancy 的定义、计算方法、三大限制因素（寄存器/共享内存/Block 大小），以及为什么 Occupancy 并非越高越好——真正的目标是在延迟隐藏与资源利用之间找到平衡点。

---

## 2026-05-11

### 新增内容

- **文章：** [2.1 Warp 与执行模型](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/21-warp与执行模型) | 深入理解 GPU 最核心的执行单元——Warp

---
## 2026-05-09

### 新增内容

- **文章：** [CUDA 内存模型](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/13-cuda内存模型) | 本文详解全局内存、共享内存、寄存器、常量内存和统一内存的特性、适用场景及优化技巧
- **文章：** [第一个实用 Kernel：向量加法](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/14-第一个实用kernel) | 通过一个完整的向量加法案例，走通 CUDA 编程的全流程

---
## 2026-05-08

### 新增内容

- **文章：** [1.1 CUDA 开发环境搭建](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/11-cuda开发环境搭建)
- **文章：** [1.2 CUDA 编程模型](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/12-cuda编程模型)

---
## 2026-05-07

### 新增内容

- **文章：** [✅ CUDA编程快速入门指南 🔥](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/cuda编程入门指南/)


---
## 2026-04-28

### 新增内容

- **文章：** [🔥 集群通信网络与NCCL：分布式训练的通信骨架](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/communication/collective-communication-primer/)

---
## 2026-04-27

### 新增内容

- **文章：** [5.1 NVIDIA GPU架构演进：从Volta到Blackwell](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/gpunvidia-gpu-evolution/)

---
## 2026-04-26

### 新增内容

- **文章：** [🔥 GPU基础知识：从硬件架构到AI计算](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/gpu/gpu-basics/)

---
## 2026-04-25

### 新增内容

- **文章：** [🔥 PyTorch框架快速入门篇](/AIInfraGuide/prerequisites/模块一-前置知识/pyroch/pytorch框架入门)

---
## 2026-04-24

### 新增内容

- **文章：** [3.9 Tokenization与词嵌入](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/39-tokenization与词嵌入)


---
## 2026-04-23

### 新增内容

- **文章：** [3.8 从Transformer到LLM自回归生成深入理解](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/38-从transformer到llm自回归生成深入理解)

---

## 2026-04-21

### 新增内容

- **文章：** [3.6 LayerNorm与残差连接深入理解](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/36-layernorm与残差连接深入理解)
- **文章：** [3.7 Transformer Decoder Block完整解析](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/37-transformer-decoder-block完整解析)

---

## 2026-04-20

### 新增内容

- **文章：** [3.4 Transformer前馈网络ffn深入理解](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/34-transformer前馈网络ffn深入理解)
- **文章：** [3.5 Transformer位置编码深入理解](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/35-transformer位置编码深入理解)

---

## 2026-04-18

### 新增内容

- **文章：** [3.3 Self-Attention 机制深入理解](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/33-self-attention机制深入理解)

---

## 2026-04-17

### 新增内容

- **面试专区**：共收录 **180+ 场面试真题**，覆盖 **40+ 家公司**，按梯队分类组织。[在线浏览 →](https://caomaolufei.github.io/AIInfraGuide/interview)
- 涵盖公司包括字节跳动、阿里巴巴、腾讯、百度、快手、美团、蚂蚁、英伟达、MiniMax、蔚来、小鹏、理想等。
- **社区入口**：新增[wx交流群]((/AIInfraGuide/about/))，欢迎大家加入讨论

### 内容优化

- 更新 README，补充项目介绍图片

---

## 2026-04-16

### 重大更新

- **知识库框架全面升级**：细化四大学习模块，调整页面布局结构
  - **[模块一 · 前置知识](/AIInfraGuide/prerequisites)**（6 章）：[编程语言基础](/AIInfraGuide/prerequisites/模块一-前置知识/第1章-编程语言基础)、[数学基础](/AIInfraGuide/prerequisites/模块一-前置知识/第2章-数学基础)、[Transformer 架构详解](/AIInfraGuide/prerequisites/模块一-前置知识/第3章-transformer架构详解)、[PyTorch 框架](/AIInfraGuide/prerequisites/模块一-前置知识/第4章-pytorch框架)、[GPU 硬件概论](/AIInfraGuide/prerequisites/模块一-前置知识/第5章-gpu硬件概论)、[集合通信基础](/AIInfraGuide/prerequisites/模块一-前置知识/第6章-集合通信基础)
  - **[模块二 · CUDA 编程与算子优化](/AIInfraGuide/cuda)**（8 章）：[CUDA 入门](/AIInfraGuide/cuda/模块二-cuda编程与算子优化/第1章-cuda编程入门)、[性能优化基础](/AIInfraGuide/cuda/模块二-cuda编程与算子优化/第2章-cuda性能优化基础)、Reduce 实战、GEMM 实战、Softmax 实战、[Attention 算子](/AIInfraGuide/cuda/模块二-cuda编程与算子优化/第6章-attention算子)、[AI 编译器](/AIInfraGuide/cuda/模块二-cuda编程与算子优化/第7章-ai编译器)、[性能分析工具链](/AIInfraGuide/cuda/模块二-cuda编程与算子优化/第8章-性能分析工具链)
  - **[模块三 · 分布式训练](/AIInfraGuide/distributed)**（7 章）：[分布式训练总论](/AIInfraGuide/distributed/模块三-分布式训练/第1章-分布式训练总论)、[数据并行](/AIInfraGuide/distributed/模块三-分布式训练/第4章-数据并行)、[ZeRO 系列](/AIInfraGuide/distributed/模块三-分布式训练/第5章-zero系列)、[张量并行与序列并行](/AIInfraGuide/distributed/模块三-分布式训练/第6章-张量并行与序列并行)、[流水线并行](/AIInfraGuide/distributed/模块三-分布式训练/第7章-流水线并行)、[3D 并行与混合并行策略](/AIInfraGuide/distributed/模块三-分布式训练/第11章-3d并行与混合并行策略)、[训练框架实战](/AIInfraGuide/distributed/模块三-分布式训练/42-pytorch-数据并行从原理到实战)
  - **[模块四 · 推理优化](/AIInfraGuide/inference)**（11 章）：[推理基础](/AIInfraGuide/inference/模块四-推理优化/第1章-llm推理基础/第1章-llm推理基础)、[推理引擎核心技术](/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/第2章-推理引擎核心技术)、[深入 vLLM](/AIInfraGuide/inference/模块四-推理优化/第3章-深入vllm/第3章-深入vllm)、[量化](/AIInfraGuide/inference/模块四-推理优化/第4章-量化/第4章-量化)、[Speculative Decoding](/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/第5章-speculative-decoding)、[分布式推理](/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/第6章-分布式推理)、[PD 解耦架构](/AIInfraGuide/inference/模块四-推理优化/第7章-pd解耦架构/第7章-pd解耦架构)、[生产级服务特性](/AIInfraGuide/inference/模块四-推理优化/第8章-生产级服务特性/第8章-生产级服务特性)、[性能分析与 Benchmark](/AIInfraGuide/inference/模块四-推理优化/第9章-性能分析与benchmark/第9章-性能分析与benchmark)、[生产部署与运维](/AIInfraGuide/inference/模块四-推理优化/第10章-生产部署与运维/第10章-生产部署与运维)、[选型与端到端实战](/AIInfraGuide/inference/模块四-推理优化/第11章-推理优化选型与端到端实战/第11章-推理优化选型与端到端实战)

### 新增内容

- **Transformer 系列文章**：
  - [Transformer 架构入门](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/transformer架构快速入门)
  - [3.1 AI Infra 工程师为什么必须懂 Transformer](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/31-ai-infra工程师为什么必须懂transformer)
  - [3.2 Transformer 全貌及代码实现](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/32-transformer全貌及代码实现)
- **课程资料**：150+ 篇详细子文章上线，覆盖四大模块全部章节
- **博客文章**：新增[「关于作者」](/AIInfraGuide/about/)介绍页

---

## 2026-04-15

### 项目上线

- AIInfraGuide 知识库正式上线
- 发布 [AI Infra 学习路线](/AIInfraGuide/guides/ai-infra学习路线)总览
- 发布[「从零理解 AI Infra」](/AIInfraGuide/guides/ai-infra学习路线)入门指引

---

## 2026-04-14

### 项目初始化

- 完成网站框架搭建（基于 Astro + Starlight）
- 初始化项目仓库，发布首次 commit
