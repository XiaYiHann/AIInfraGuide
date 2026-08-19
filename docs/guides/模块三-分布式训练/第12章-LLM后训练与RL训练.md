---
title: "第12章：LLM 后训练与 RL 训练"
description: "从 InstructGPT 的 RLHF 三阶段流水线出发，建立 LLM 后训练、偏好优化、GRPO/DAPO 与 RLVR 的训练系统全景"
pubDate: 2026-08-19
category: "distributed-training"
order: 21
tags: ["LLM 后训练", "RLHF", "PPO", "DPO", "GRPO", "RLVR"]
---

## 📖 本章概述

预训练让 LLM（Large Language Model，大型语言模型）学会了从文本中预测下一个 token，但“会续写”不等于“会按人类意图完成任务”。本章把镜头从预训练后的模型行为拉到后训练：先理解 RL（Reinforcement Learning，强化学习）如何把人类示范和偏好变成训练信号，再逐步比较 RLHF（从人类反馈中学习）、PPO（限制策略更新幅度的强化学习算法）、DPO（直接偏好优化）、GRPO（群体相对策略优化）、DAPO（面向大规模训练的策略优化变体）和 RLVR（Reinforcement Learning with Verifiable Rewards，可验证奖励强化学习）等方法的模型成本、数据形态和稳定性边界。

本章的核心问题是两本账：**训练信号从哪里来，以及每轮更新要付出多少模型、显存和采样成本。**12.1 先用 InstructGPT 把 SFT、奖励模型、PPO 和四模型副本账本算清；12.2 再讨论不经过在线 PPO 的直接偏好优化；后续两节把重点推进到数学推理奖励和可验证奖励。

## 📑 章节结构

### 1. 人类偏好如何进入训练回路

- [12.1 RLHF 的四模型成本：InstructGPT 三阶段流水线与 PPO 训练回路（论文精读）](/AIInfraGuide/distributed/模块三-分布式训练/121-rlhf三阶段流水线与instructgpt)：从 next-token 目标错位出发，推导 SFT、Bradley-Terry 奖励模型和 PPO 的 KL 约束，算清 policy/reference/RM/critic 四个模型角色与 on-policy 采样账。
- [12.2 DPO 与偏好优化](/AIInfraGuide/distributed/模块三-分布式训练/122-dpo直接偏好优化)：围绕成对偏好数据，比较直接偏好优化与 PPO-RLHF 的模型、数据和训练成本。

### 2. 从通用偏好到可验证奖励

- [12.3 从 PPO 到 GRPO 再到 DAPO：LLM 后训练算法链与算力/显存账本](/AIInfraGuide/distributed/模块三-分布式训练/123-grpo与dapo后训练算法链)：GRPO 用组内相对优势消掉 critic（G=4 数值走查）、DAPO 四训练技巧、四算法（PPO/DPO/GRPO/DAPO）模型副本与在线采样账本对比。
- [12.4 零数据 RLVR：可验证奖励与 Absolute Zero 自博弈](/AIInfraGuide/distributed/模块三-分布式训练/124-rlvr与absolutezero自博弈)：RLVR 谱系、Absolute Zero 自生成任务循环（生成→程序化验证→reward 的算例）、自博弈训练的额外开销与任务塌缩边界。

## 🎯 本章学习目标

- 能解释 next-token 预训练目标为什么与 helpful、truthful、harmless 的后训练目标存在错位。
- 能画出 RLHF 的 SFT → 奖励模型 → PPO 三阶段数据流，并说明每一阶段的输入、输出与训练信号。
- 能推导 Bradley-Terry 偏好概率和交叉熵损失，算出同一 prompt 的 $K$ 条回答会产生多少个比较。
- 能区分 PPO 的 old policy、冻结 reference、reward model 和 critic，并解释 KL 惩罚与 PPO clip 各自限制什么。
- 能用参数量和精度估算后训练的四模型显存账，说明 on-policy rollout 为什么带来持续的解码和前向成本。
- 能比较 PPO、DPO、GRPO/DAPO 与 RLVR 的奖励来源、数据复用方式和工程取舍。
