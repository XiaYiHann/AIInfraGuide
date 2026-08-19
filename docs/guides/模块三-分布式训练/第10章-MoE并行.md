---
title: "第10章：MoE 并行"
description: "稀疏专家模型的并行之道——Router 机制、Expert Parallelism（专家并行，把不同专家放不同卡）的 All-to-All（每卡向所有卡各发不同分片的集合通信，MoE 中 dispatch 发 token 到专家卡、combine 发结果回源卡）通信、EP（专家并行）×DP（数据并行，复制模型）×TP（张量并行，切单层矩阵）×PP（流水线并行，切层）多维组合与负载均衡"
pubDate: 2026-04-16
category: "distributed-training"
order: 19
tags: ["MoE", "Expert Parallelism", "All-to-All", "负载均衡", "稀疏模型"]
---

## 📖 本章概述

MoE（Mixture of Experts，混合专家，对每 token 只激活 top-k 个专家以低计算扩参数）是当前千亿/万亿参数模型（DeepSeek-V3、Mixtral 等）的主流架构——用稀疏激活在不显著增加计算量的前提下扩大参数规模。MoE 引入了一种全新的并行维度：**Expert Parallelism（EP，专家并行，把不同专家放到不同 GPU、每卡只存 $E/N$ 个）**，及其标志性的 All-to-All（集合通信，每卡向所有卡各发一片不同数据，MoE 中 dispatch 把 token 发往对应专家所在卡、combine 把结果发回原卡）通信模式。本章独立成章，因为它的通信模式、负载均衡问题与稠密模型的并行有本质区别。

---

## 📑 章节结构

### 1. MoE 模型结构回顾

- 稠密 FFN → 稀疏 MoE：用 $E$ 个 Expert 替换单个 FFN
- Router（Gating Network）：为每个 token 选择 Top-K 个 Expert
- 稀疏激活：参数量 $\times E$，但单 token 计算量只增加 $K$ 倍
- 容量因子（Capacity Factor）与 token 丢弃
- 架构与训练工程的原始论文（noisy top-k、专家坍缩、shrinking batch）：[3.12 参数扩 1000 倍但计算不变：MoE 的诞生（Sparsely-Gated MoE 精读）](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/312-moe-sparsely-gated-moe论文精读)

### 2. Expert Parallelism（EP）

- 核心思想：不同 Expert 放在不同 GPU 上，每卡只存 $\frac{E}{N}$ 个 Expert
- 与 TP 的区别：TP 切单个权重矩阵，EP 切的是“哪些 Expert 在哪”

### 3. All-to-All 通信（本章重点）

- 两次 All-to-All：dispatch（token 发往对应 Expert 所在 GPU）+ combine（结果发回原 GPU）
- 通信量分析：取决于 token 数、Top-K、Expert 分布，数据量≈$B\cdot s\cdot K\cdot h/N$（$B$批 $s$序列 $K$ Top-K $h$隐维 $N$卡，如 4 卡/128 token/Top2/h=4096 时约 4 MB/卡）
- 为什么 All-to-All 是 MoE 训练的主要瓶颈
- 通信优化：分组 All-to-All、计算通信重叠、DeepEP 等高性能通信库

### 4. 负载均衡问题

- Router 倾斜：少数 Expert 被过度选择 → 部分 GPU 过载、部分空闲
- Auxiliary Loss：引导 Router 均匀分发 token 的辅助损失
- DeepSeek 的无辅助损失负载均衡（aux-loss-free，bias 调整）
- Expert 容量与 drop/pad 策略

### 5. EP 与其他并行的组合

- EP × DP：EP 组与 DP 组的正交划分
- EP × TP：Expert 内部再做张量并行（超大 Expert）
- EP × PP：MoE 层与稠密层在流水线中的分配
- 实例：DeepSeek-V3 / Mixtral 的并行配置剖析

### 6. 动手实验

- 实现一个最小 MoE 层 + Top-2 Router，打印 token 路由分布
- 模拟 All-to-All dispatch/combine 过程
- 观察负载不均衡现象，加入 aux loss 后对比

---

## 🎯 本章学习目标

- 能画出 MoE 层的 Router → dispatch → Expert → combine 数据流
- 能解释 Expert Parallelism 的两次 All-to-All 各自传输什么
- 能说明 MoE 负载均衡问题的成因及 aux loss / aux-loss-free 两类解法
- 能分析 EP 与 DP/TP/PP 组合时的通信域划分
