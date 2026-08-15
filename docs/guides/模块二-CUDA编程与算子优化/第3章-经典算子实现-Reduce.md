---
title: "第3章：经典算子实现—Reduce"
description: "通过 Reduce 算子的多个递进优化版本（从朴素实现到 Warp Shuffle、向量化加载），掌握 CUDA 算子逐步优化的方法论"
pubDate: 2026-04-16
category: "cuda-optimization"
order: 12
tags: ["CUDA", "Reduce", "共享内存", "Warp Shuffle", "算子优化"]
---

## 本章简介

Reduce（归约）是最经典的并行算法之一，也是学习 CUDA 优化的最佳入门案例。本章通过 V0→V7 共八个递进版本，体验“分析瓶颈 → 针对优化 → 量化收益”的完整优化循环。

**朴素实现**使用全局内存 + 原子加，最简单但最慢，通过性能分析定位瓶颈。

**共享内存 + 树形归约**在 Block 内使用共享内存进行树形归约，消除 Warp Divergence，处理多 Block 的二次归约问题。

**Warp Shuffle 优化**用 `__shfl_down_sync` 替代共享内存，完全消除 Bank Conflict，实现 Warp 级的高效 Reduce。

**多级归约与大规模数据**介绍 Grid-stride loop 处理超大数组的技巧。

**动手实验**：实现 V0→V7 多个版本的 Reduce Sum，使用 Nsight Compute 对比 throughput，说清每一步优化到底省在哪里。
