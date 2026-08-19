---
title: "第3章：深入 vLLM 架构与源码"
description: "深入 vLLM 的整体架构、V1 引擎设计、调度器源码与关键配置调优，并横向对比 SGLang、TensorRT-LLM"
pubDate: 2026-04-16
category: "inference-optimization"
order: 32
tags: ["vLLM", "V1引擎", "Scheduler", "源码导读", "SGLang", "TensorRT-LLM"]
updatedDate: 2026-08-19
---

## 本章简介

vLLM 是本模块的实例主线。前两章讲的核心技术，本章落到 vLLM 的真实代码里看它们如何被组织、调度和调优；同时横向对比 SGLang 与 TensorRT-LLM，帮助你做框架选型。

**快速入门**从安装、离线批量推理到 OpenAI 兼容服务部署，用最短路径跑通第一个 vLLM 推理服务。

**整体架构**拆解 vLLM 的分层设计：`LLMEngine` / `AsyncLLM` 入口、`EngineCore` 执行循环、`Scheduler` 调度器、`KVCacheManager` 显存管理、`Worker` / `ModelRunner` 模型执行，理清一个请求从进入到返回的完整数据流。

**nano-vllm 源码导读**先用约 1200 行代码的最小实现建立心智地图：双队列调度、block 记账的 KV cache、xxhash 前缀缓存、CUDA graph 与极简张量并行——这是从“会用 vLLM”到“看懂 vLLM”的捷径。

**V1 引擎深度解析**接着整体架构往下挖：从 `LLM()` 到第一个 step 的冷启动五步序列（加载权重 → profile 测峰值 → 定 KV 池 → 分配 KV → 抓 CUDA Graph）、decode 快的两个机制（CUDA Graph 与默认开启的异步调度）、零开销 Prefix Cache 的块级实现，以及 V1 相对 V0 的改动清单。

**调度器源码导读**深入 Waiting/Running 队列、Token Budget 分配、抢占（Preemption）与重计算（Recompute）机制，看懂 vLLM 每一步 step 到底做了什么。

**关键配置调优**讲清 `gpu-memory-utilization`、`max-num-seqs`、`max-num-batched-tokens`、`block-size` 等参数如何影响吞吐、延迟和显存，给出调参决策思路。

**框架横向对比**用统一维度对照 vLLM（通用、生态最全）、SGLang（RadixAttention、结构化输出、Agent 场景）、TensorRT-LLM（NVIDIA 深度优化、极限延迟），给出选型决策表。

## 本章小节

- **3.1 vLLM 快速入门**：安装、离线批量推理、OpenAI 兼容服务部署
- **3.2 nano-vllm 源码导读**：用约 1200 行最小实现读懂调度、KV cache、前缀缓存、CUDA graph 与张量并行
- **3.3 vLLM 整体架构**：LLMEngine/AsyncLLM、EngineCore、Scheduler、KVCacheManager、Worker/ModelRunner
- **3.4 V1 引擎深度解析**：冷启动五步序列、KV 池定容、CUDA Graph、异步调度、零开销 Prefix Cache、V1 相对 V0 的改进
- **3.5 调度器源码导读**：Token Budget、Waiting/Running 队列、抢占与重计算
- **3.6 关键配置调优**：显存、批大小、Block 等核心参数的调参思路
- **3.7 框架横向对比**：vLLM / SGLang / TensorRT-LLM 选型决策
