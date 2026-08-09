---
title: "第12章：TensorRT 推理引擎"
description: "掌握 TensorRT 的定位与两段式设计、核心对象与构建流程、四条模型导入路径与最小实战、精度与量化路线、构建器优化原理与 IPluginV3 插件范式，以及 TRT 生态与选型"
pubDate: 2026-08-09
category: "inference-optimization"
order: 41
tags: ["TensorRT", "推理编译器", "ONNX", "量化", "插件开发", "GPU推理"]
---

## 本章简介

本章围绕 NVIDIA TensorRT（TRT）展开：它是什么、模型怎么进去、engine 怎么构建与运行、精度怎么选、构建器内部在优化什么、以及它在推理生态里的位置。

**TRT 是什么**从开源仓库的真实面貌讲起：TRT 是"模型编译器 + 推理运行时"的两段式设计——ONNX 文件必须先离线编译成序列化 plan 文件，运行时只反序列化并执行，它不解析 ONNX。11.x 大版本对 API 做了一次主动精简，理解这一点才能看懂后续所有代码示例。

**核心对象与构建流程**拆解六大核心对象（Logger/Builder/Network/BuilderConfig/OptimizationProfile/Engine）与运行期三件套（Runtime/Engine/ExecutionContext），建立"构建期 vs 运行期"的心智模型，并给出完整可运行的 Python 构建代码与 trtexec 常用旗标口径。

**导入路径与最小实战**按官方 import_workflows 指南走四条导入路径——ONNX、Torch-TensorRT、Hugging Face Hub、Network Definition API，每条讲清"适合谁、官方命令、最常见的坑"，最后给出 ONNX 到 engine 的最小可运行链路与三级验证闭环。

**精度与量化**讲精度阶梯（FP16/BF16/INT8/FP8）、构建旗标、INT8 无校准的坑、强类型混合精度与 Model Optimizer 校准工具链。

**优化原理与自定义算子**钻进构建器内部：build route 旋钮、tactic 搜索、workspace 规划，以及 11.x 插件新范式 IPluginV3 的注册、序列化与迁移。

**生态与选型**把 TRT 放回推理生态看分工：LLM 走 TensorRT-LLM、非 LLM 走核心 TRT、部署交给 Triton，并给出选型决策树为模块四收尾。

## 本章小节

- **12.1 TensorRT 是什么**：仓库面貌、两段式设计、11.x 大版本跳升、与 cuDNN/CUDA 的分工、安装与配套环境
- **12.2 核心对象与构建流程**：六大核心对象与 trtexec，构建期 vs 运行期心智模型，完整 Python 构建代码
- **12.3 导入路径与最小实战**：ONNX / Torch-TensorRT / HF Hub / Network Definition API 四条路径的选型与全流程，ONNX→engine 最小可运行链路
- **12.4 精度与量化**：FP16/BF16/INT8/FP8 精度阶梯、构建旗标、INT8 无校准的坑、显式量化与 Model Optimizer
- **12.5 优化原理与自定义算子**：构建路由、tactic 搜索、workspace 规划与 IPluginV3 插件新范式
- **12.6 生态与选型**：与 TensorRT-LLM、Triton、ONNX Runtime 的分工与选型决策树
