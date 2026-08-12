---
title: "AIInfraGuide 文章更新日志"
description: "记录 AIInfraGuide 知识库的每一次内容更新，方便读者追踪最新变化"
pubDate: 2026-04-16
tags: ["公告", "更新日志"]
---

本文持续记录 AIInfraGuide 知识库的内容更新，按时间倒序排列，方便大家了解最新动态。

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

- **模块三 · 分布式训练** 新增 [第2章：优化器](/AIInfraGuide/distributed/模块三-分布式训练/第2章-优化器) | 理解 SGD → Momentum → Adam → AdamW 的演进逻辑与内部状态组成，手算优化器显存开销（AdamW 占训练总显存 75%），掌握大 Batch 优化器 LAMB/LARS 的适用场景，为后续 ZeRO 显存优化和混合精度训练打下基础。

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

- **文章：** [3.3 Self-Attention 机制深入理解](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/33-Self-Attention机制深入理解)

---

## 2026-04-17

### 新增内容

- **面试专区**：共收录 **180+ 场面试真题**，覆盖 **40+ 家公司**，按梯队分类组织。[在线浏览 →](https://caomaolufei.github.io/AIInfraGuide/interview)
- 涵盖公司包括字节跳动、阿里巴巴、腾讯、百度、快手、美团、蚂蚁、英伟达、MiniMax、蔚来、小鹏、理想等。
- **社区入口**：新增[wx交流群]((/AIInfraGuide/about/author))，欢迎大家加入讨论

### 内容优化

- 更新 README，补充项目介绍图片

---

## 2026-04-16

### 重大更新

- **知识库框架全面升级**：细化四大学习模块，调整页面布局结构
  - **[模块一 · 前置知识](/AIInfraGuide/prerequisites)**（6 章）：[编程语言基础](/AIInfraGuide/prerequisites/模块一-前置知识/第1章-编程语言基础)、[数学基础](/AIInfraGuide/prerequisites/模块一-前置知识/第2章-数学基础)、[Transformer 架构详解](/AIInfraGuide/prerequisites/模块一-前置知识/第3章-Transformer架构详解)、[PyTorch 框架](/AIInfraGuide/prerequisites/模块一-前置知识/第4章-PyTorch框架)、[GPU 硬件概论](/AIInfraGuide/prerequisites/模块一-前置知识/第5章-GPU硬件概论)、[集合通信基础](/AIInfraGuide/prerequisites/模块一-前置知识/第6章-集合通信基础)
  - **[模块二 · CUDA 编程与算子优化](/AIInfraGuide/cuda)**（8 章）：[CUDA 入门](/AIInfraGuide/cuda/模块二-CUDA编程与算子优化/第1章-CUDA编程入门)、[性能优化基础](/AIInfraGuide/cuda/模块二-CUDA编程与算子优化/第2章-CUDA性能优化基础)、Reduce 实战、GEMM 实战、Softmax 实战、[Attention 算子](/AIInfraGuide/cuda/模块二-CUDA编程与算子优化/第6章-Attention算子)、[AI 编译器](/AIInfraGuide/cuda/模块二-CUDA编程与算子优化/第7章-AI编译器)、[性能分析工具链](/AIInfraGuide/cuda/模块二-CUDA编程与算子优化/第8章-性能分析工具链)
  - **[模块三 · 分布式训练](/AIInfraGuide/distributed)**（7 章）：[分布式训练总论](/AIInfraGuide/distributed/模块三-分布式训练/第1章-分布式训练总论)、[数据并行](/AIInfraGuide/distributed/模块三-分布式训练/第2章-数据并行)、[ZeRO 系列](/AIInfraGuide/distributed/模块三-分布式训练/第3章-ZeRO系列)、[张量并行与序列并行](/AIInfraGuide/distributed/模块三-分布式训练/第4章-张量并行与序列并行)、[流水线并行](/AIInfraGuide/distributed/模块三-分布式训练/第5章-流水线并行)、[3D 并行与混合训练策略](/AIInfraGuide/distributed/模块三-分布式训练/第6章-其他显存优化技术)、[训练框架实战](/AIInfraGuide/distributed/模块三-分布式训练/第7章-训练框架实战)
  - **[模块四 · 推理优化](/AIInfraGuide/inference)**（11 章）：[推理基础](/AIInfraGuide/inference/模块四-推理优化/第1章-llm推理基础)、[推理引擎核心技术](/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术)、[深入 vLLM](/AIInfraGuide/inference/模块四-推理优化/第3章-深入vllm)、[量化](/AIInfraGuide/inference/模块四-推理优化/第4章-量化)、[Speculative Decoding](/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding)、[分布式推理](/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理)、[PD 解耦架构](/AIInfraGuide/inference/模块四-推理优化/第7章-pd解耦架构)、[生产级服务特性](/AIInfraGuide/inference/模块四-推理优化/第8章-生产级服务特性)、[性能分析与 Benchmark](/AIInfraGuide/inference/模块四-推理优化/第9章-性能分析与benchmark)、[生产部署与运维](/AIInfraGuide/inference/模块四-推理优化/第10章-生产部署与运维)、[选型与端到端实战](/AIInfraGuide/inference/模块四-推理优化/第11章-推理优化选型与端到端实战)

### 新增内容

- **Transformer 系列文章**：
  - [Transformer 架构入门](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/Transformer架构快速入门)
  - [3.1 AI Infra 工程师为什么必须懂 Transformer](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/31-ai-infra工程师为什么必须懂transformer)
  - [3.2 Transformer 全貌及代码实现](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/32-transformer全貌及代码实现)
- **课程资料**：150+ 篇详细子文章上线，覆盖四大模块全部章节
- **博客文章**：新增[「关于作者」](/AIInfraGuide/about/author)介绍页

---

## 2026-04-15

### 项目上线

- AIInfraGuide 知识库正式上线
- 发布 [AI Infra 学习路线](/AIInfraGuide/guides/AI Infra学习路线)总览
- 发布[「从零理解 AI Infra」](/AIInfraGuide/blog/ai-infra-introduction)入门指引

---

## 2026-04-14

### 项目初始化

- 完成网站框架搭建（基于 Astro + Starlight）
- 初始化项目仓库，发布首次 commit
