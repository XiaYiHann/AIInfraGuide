---
title: "第5章：Speculative Decoding"
description: "理解投机解码的核心原理（Draft + Verify）、Self-Draft 方案（Medusa/EAGLE-2）及其收益边界与限制"
pubDate: 2026-04-16
category: "inference-optimization"
order: 34
tags: ["Speculative Decoding", "投机解码", "Medusa", "EAGLE", "N-gram", "Rejection Sampling"]
---

## 本章简介

自回归解码的串行瓶颈是 Decode 阶段效率低的根本原因。Speculative Decoding 通过"先猜后验"打破这一瓶颈。本章原理与 vLLM 中的实际配置并重。

**核心原理**详解 Speculative Sampling 框架：Draft 模型快速猜测多个 Token，Target 模型并行验证。通过 Rejection Sampling 机制保证正确性——数学上证明接受的 Token 严格服从 Target 分布。

**Draft 模型与无模型方案**讨论独立小模型的选择（大小、架构、能力匹配），以及无需额外模型的 N-gram / Suffix Decoding（基于 Prompt 与历史输出做前缀匹配），并分析 Acceptance Rate 对加速比的影响。

**Self-Draft 方案**覆盖 Medusa（多 Decoding Head 预测多 Token）和 EAGLE-2/3（动态 Draft Tree + 置信度校准），以及从 Token 级到 Block 级联合验证的演进。

**收益边界与限制**分析高接受率场景（代码生成，加速明显）vs 低接受率场景（开放对话，收益有限），以及与量化叠加的精度风险、与 Continuous Batching 的调度复杂度。

**在 vLLM 中配置投机解码实战**：介绍 vLLM 对 N-gram、EAGLE、Draft 模型等投机解码方案的配置方式，并对比代码生成 vs 开放对话的 Acceptance Rate 差异。

## 本章小节

- **5.1 核心原理**：Draft + Verify、Rejection Sampling 正确性证明
- **5.2 Draft 模型与 N-gram/Suffix 方案**：模型选择与无模型投机
- **5.3 Self-Draft 方案**：Medusa、EAGLE-2/3、Draft Tree
- **5.4 收益边界与限制**：接受率、精度风险、调度复杂度
- **5.5 vLLM 投机解码实战**：配置方式与接受率实测
