---
title: "第4章：经典算子实现—GEMM"
description: "从朴素矩阵乘法到 Block-Warp-Thread三级Tiling优化、向量化访存、Bank Conflict 消除、双缓冲等优化手段，带你系统掌握 CUDA GEMM 优化的完整方法论"
pubDate: 2026-04-16
category: "cuda-optimization"
order: 13
tags: ["CUDA", "GEMM", "矩阵乘法", "Tiling", "Tensor Core", "cuBLAS"]
---

## 本章简介

GEMM（General Matrix Multiply，通用矩阵乘法，$C(M\times N)=A(M\times K)\times B(K\times N)$，如 hidden=4096 时 QKV 为 $4096\times12288$）是深度学习中最核心的算子——线性层、Attention 的 QKV（Query/Key/Value，三投影矩阵合并）投影、FFN（Feed-Forward Network，前馈网络，两层 MLP+SwiGLU）的计算本质上都是 GEMM。本章从零实现高性能 GEMM，理解连接硬件和上层框架的桥梁。

**1. 背景与问题定义** 介绍 GEMM 在深度学习中的地位、问题规模与计算/访存特征，明确优化目标。

**2. 性能分析方法论** 引入算术强度、Roofline 模型与性能评估指标，建立每一步优化的量化分析框架。

**3. 朴素实现：建立 Baseline** 最直观的三重循环 Kernel，暴露全局内存重复访问导致的 Memory-Bound 瓶颈。

**4. Thread Block 级 Tiling 优化** 将矩阵切成 Tile 载入共享内存，从数学上分析全局内存访问的减少量，完成第一轮数据复用。

**5. Thread 级 Tiling 优化** 在寄存器中缓存子块结果，每个线程计算多个输出元素，进一步提升计算密度与数据复用率。

**6. Warp 级分块优化** 按 Warp 组织 Tile 结构，配合线程映射对齐硬件执行单元，减少 Warp 内冗余计算与同步开销。

**7. 向量化访存：float4 优化** 使用 `float4` 合并加载/存储，减少访存指令数并提升有效带宽利用率。

**8. Bank Conflict 消除** 通过 Padding 或地址重排打破 Shared Memory 访问的 Bank 冲突，让共享内存吞吐回归峰值。

**9. 双缓冲与流水线** 在共享内存中开两份 Buffer，让数据加载与计算重叠，隐藏全局内存延迟。

**10. Ampere 异步拷贝流水线** 利用 `cp.async` 将全局内存到共享内存的拷贝异步化，构建多级软件流水线，进一步压榨延迟。

**11. SASS 级优化与寄存器调度** 深入指令级调优：寄存器分配、指令排布与 Bank Conflict，对照 cuBLAS 的 SASS 代码理解极致性能从何而来。

**12. 实战参数选择与调优指南** 总结 Tile 尺寸、线程布局、Block 数量等参数的选择方法论，给出不同矩阵规模与硬件下的调优思路。
