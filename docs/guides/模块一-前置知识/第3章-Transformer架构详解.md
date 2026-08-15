---
title: "第3章：AI Infra工程师学Transformer"
description: "深入理解 Transformer 的每一个组件：Self-Attention、FFN、位置编码、归一化层，以及从 MHA 到 GQA/MLA、从 FFN 到 MoE 的架构演进"
pubDate: 2026-04-16
category: "prerequisites"
order: 3
tags: ["Transformer", "Attention", "RoPE", "MoE", "模型架构"]
---

## 本章简介

Transformer 是大模型时代的核心架构，也是 AI Infra 工程师必须深入理解的对象——你优化的每一个算子、设计的每一种并行策略，最终都作用在 Transformer 的某个组件上。

本章从 **Self-Attention 机制**出发，详解 Q/K/V 投影、Attention 计算全流程及其 O(N²) 复杂度分析。接着覆盖 **FFN 与激活函数**（ReLU/GELU/SwiGLU）、**位置编码**（Sinusoidal/RoPE/ALiBi）、**归一化层**（LayerNorm/RMSNorm、Pre-Norm vs Post-Norm）。

**完整前向过程**部分将逐步追踪从 Token Embedding 到输出 Logits 的完整数据流，标注每一步的输入输出维度，并手算参数量。

最后从分布式与推理视角分析 **架构变种**：MHA → MQA → GQA → MLA 的演进动机，以及 FFN → GLU → MoE 的变化如何影响并行切分和推理优化。
