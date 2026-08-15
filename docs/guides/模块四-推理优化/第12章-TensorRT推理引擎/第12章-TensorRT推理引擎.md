---
title: "第12章：TensorRT-LLM 推理引擎"
description: "从仓库面貌与 PyTorch 原生架构讲起，覆盖模型加载与量化、PyExecutor 运行时与 trtllm-serve、KV Cache 与长上下文、并行与高级特性，再到生态选型、多节点部署与模块四总结"
pubDate: 2026-08-09
category: "inference-optimization"
order: 41
tags: ["TensorRT-LLM", "推理引擎", "部署", "量化", "KV Cache", "并行", "GPU推理"]
---

## 本章简介

本章围绕 NVIDIA TensorRT-LLM（TRT-LLM）展开：它是什么、模型怎么加载与量化、运行时怎么运转、KV Cache 与长上下文怎么处理、多卡并行与高级特性有哪些、以及它在推理生态里的位置与多节点部署路径。第 1-11 章讲的是推理优化的通用原理，这一章看 NVIDIA 的极致工程实现。

**TensorRT-LLM 是什么**从开源仓库的真实面貌讲起：PyTorch 原生架构与 PyExecutor、与 TensorRT 的分野（1.2 起不再依赖 TensorRT engine 执行）、与 vLLM 的定位差异（Day-0 模型支持与生态绑定）、NGC 容器快速上手与遥测版本政策。

**从权重到引擎**讲清 1.2 移除 `trtllm-build` 后“构建”的新含义：LLM API 直接吃 HuggingFace 权重 + ModelOpt 离线量化产物，与 TensorRT 的 ONNX→plan 流程彻底分道扬镳，并给出硬件边界与三条加载路线的决策树。

**运行时与 LLM API**钻进 PyExecutor 后台循环与 trtllm-serve 服务形态：六个 OpenAI 兼容端点、并行参数与配置优先级（CLI 覆盖 YAML）、多模态与 VisualGen 边界、Slurm 多节点示例，以及 In-flight Batching 调度器的三个参数与一个优先级。

**KV Cache 与长上下文**讲透 block 池与 radix tree 复用、缓存量化（FP8/NVFP4）、cache_salt 安全隔离、长序列三件套（分块、窗口、环形缓冲）与 PD 解耦。

**并行、量化与高级特性**覆盖六种并行策略（TP/PP/DP/EP/CP/Wide-EP）、量化配方与硬件矩阵、投机解码全家桶、guided decoding、稀疏注意力等 TRT-LLM 高阶武器库。

**生态选型与部署**把 TRT-LLM 放回生态：官方部署指南覆盖的模型、快速部署路径、与 vLLM/SGLang 的选型对比、Triton/Dynamo 生产集成、Slurm 两节点实战、压测监控三大 CLI 与 /metrics、版本演进纪律，为模块四收官。

## 本章小节

- **12.1 TensorRT-LLM 是什么**：仓库面貌、版本节奏、PyTorch 原生架构与 PyExecutor、与 TensorRT 的分野、与 vLLM 的定位差异、快速上手与遥测政策
- **12.2 从权重到引擎**：模型加载三种方式、ModelOpt 预量化、量化从构建旗标到离线产物、硬件边界与决策树
- **12.3 运行时与 LLM API**：PyExecutor 五步循环、trtllm-serve 六个端点、并行参数与配置优先级、多模态/VisualGen 边界、多节点示例与 IFB 三个参数
- **12.4 KV Cache 与长上下文**：分页缓存与 radix tree 复用、缓存量化、长序列三件套、PD 解耦
- **12.5 并行、量化与高级特性**：TP/PP/DP/EP/CP/Wide-EP、FP8/FP4 硬件矩阵、投机解码、guided decoding、稀疏注意力
- **12.6 生态选型与部署**：官方部署指南、快速部署、与 vLLM 对比、Triton/Dynamo、Slurm 多节点、压测监控、版本演进与模块四总结
