---
title: "FlashAttention 精读：把注意力算得快的关键不是少算，而是少搬数据"
description: "拆解 NeurIPS 2022 的 FlashAttention:IO-aware 的精确注意力算法如何用 tiling 把 N×N 注意力矩阵留在片上 SRAM、用在线 softmax 逐块合并、用重计算省掉反向传播的中间矩阵，以及 v2/v3/FA-4 三代如何在 A100/H100/B200 上逼近 GEMM 效率。"
pubDate: 2026-08-15
originalUrl: "https://arxiv.org/abs/2205.14135"
sourceType: "paper"
originalAuthor: "Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré (Stanford University & University at Buffalo)"
tags: ["FlashAttention", "注意力算子", "Kernel 优化", "IO-aware", "SRAM"]
stage: attention
order: 1
prereqs: []
minutes: 50
difficulty: 3
---

> 原文：[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)（Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré,NeurIPS 2022,arXiv:2205.14135 v2,2022-06;本文访问日期 2026-08-15）

注意力慢的根源不在计算量，而在**数据搬运**：标准实现要把 $N \times N$ 的注意力矩阵写进显存再读出来，序列一长，搬运量按长度平方暴涨。FlashAttention 的答案是 IO-aware（感知输入输出）：用 tiling（分块）把注意力矩阵留在 GPU 片上 SRAM（静态随机存取存储器，速度比显存快一个数量级的小块存储）里逐块算完，配合在线 softmax（不用等整行算完就能合并的 softmax 分解）和反向重计算，让显存访问从 $O(N^2)$ 降到 $O(N^2 d^2 / M)$。效果：GPT-2 上注意力计算快 7.6 倍（Fig.1 右），训练 BERT-large 比 MLPerf 1.1 纪录快 15%（Table 1），并把第一个 Transformer 模型带上了 Path-X（序列长 16K,61.4% 准确率，Table 6）。本文先讲直觉与最小例子，再逐项拆公式与算法，最后对照 v2/v3/FA-4 三代演进看它今天在 A100/H100/B200 上的位置。

<!-- more -->

## 📑 目录

- [🗺️ 原文阅读地图](#️-原文阅读地图)
- [0. 读前 3 分钟：先建立两组直觉](#0-读前-3-分钟先建立两组直觉)
- [1. 问题：标准注意力把钱花在了搬运上](#1-问题标准注意力把钱花在了搬运上)
- [2. 核心机制：在线 softmax + tiling（公式逐项拆解）](#2-核心机制在线-softmax--tiling公式逐项拆解)
- [3. 为什么省搬运：IO 复杂度分析](#3-为什么省搬运io-复杂度分析)
- [4. 反向传播：用重计算换显存访问](#4-反向传播用重计算换显存访问)
- [5. 评测：15%、3×、7.6× 这些数字怎么来的](#5-评测15376-这些数字怎么来的)
- [6. 延伸：block-sparse FlashAttention](#6-延伸block-sparse-flashattention)
- [🕰️ 原文时代 vs 当前工程：v2/v3/FA-4 三代演进](#️-原文时代-vs-当前工程v2v3fa-4-三代演进)
- [7. 常见误读与错误做法](#7-常见误读与错误做法)
- [📝 总结](#-总结)
- [🎯 自我检验清单](#-自我检验清单)
- [📚 参考资料](#-参考资料)

## 🗺️ 原文阅读地图

FlashAttention 是一篇“算法 + 分析 + 实验”齐全的短论文，全文 5 节加 5 个附录。下表列出会改变本文主张的原文单元；装饰性内容（相关工作、致谢）不展开。

| 原文单元 | 处理 | 本文位置/省略理由 | 来源锚点 |
| --- | --- | --- | --- |
| §1 Introduction（IO-aware 动机、7.6× 加速、block-sparse 预览） | 精讲 | 第 0、1 节：建立直觉 | §1;Fig.1 |
| §2 Background（2.1 GPU 内存层次与算术强度、2.2 标准注意力 Algorithm 0） | 精讲 | 第 1 节：问题定义 | §2.1;§2.2;Algorithm 0 |
| §3.1 算法（tiling、在线 softmax、重计算、Algorithm 1、Theorem 1） | 精讲 | 第 2、4 节：核心机制，含公式逐项拆解与手算例子 | §3.1;Algorithm 1;Theorem 1 |
| §3.2 IO 复杂度（Theorem 2、Proposition 3 下界、Fig.2） | 精讲 | 第 3 节：为什么省搬运 | §3.2;Theorem 2;Proposition 3;Fig.2 |
| §3.3 block-sparse 扩展（Proposition 4） | 简述 | 第 6 节：只给结论与适用范围 | §3.3;Proposition 4 |
| §4 实验（4.1 训练加速 Table 1-3、4.2 长序列质量 Table 4-6、4.3 基准 Fig.3） | 精讲（4.1、4.2） | 第 5 节：数字逐项核对口径 | Table 1-6;Fig.3 |
| §5 Limitations（CUDA 手写门槛、多 GPU 扩展） | 简述 | 第 7 节与 🕰️ 节：解释适用范围 | §5 |
| 附录 A-E（相关工作、反向算法、证明、扩展细节、完整实验） | 不展开 | 不改变本文机制承诺；证明细节读者可自行查阅 | Appendix A-E |

📌 **本文承诺**：读完后，你应该能用手算一个 $Q$ 块与两个 $K$ 块合并的在线 softmax 过程，解释 $O(N^2 d^2/M)$ 这个复杂度里每一项从哪来，并说清 v2/v3/FA-4 各自解决了 FlashAttention-1 的哪个瓶颈。

## 0. 读前 3 分钟：先建立两组直觉

**第一组直觉：GPU 上有两层“工作台”。** 可以把 GPU 想象成一个厨房：HBM（High Bandwidth Memory，高带宽显存）是大仓库，能装 40-80GB 的食材，但进出仓库要花时间；SRAM 是灶台旁边的小料理台，只有 192KB（每 SM，流处理器组），但拿取几乎不花时间。做菜时，如果你每切一刀都跑一趟仓库，速度就废了。论文 §2.1 给的具体数字是：A100 的 HBM 带宽约 1.5-2.0TB/s，而每 SM 的 SRAM 带宽估计约 19TB/s——差一个数量级。

**第二组直觉：注意力有一个“必须整行看完才能算完”的步骤。** softmax 需要先找到一行的最大值、求和，然后才能归一化。标准做法是先把整个 $N \times N$ 的分数矩阵 $S = QK^\top$ 写进 HBM，再读出来算 softmax，再写回去，再读出来乘 $V$——一进一出三次大搬运。FlashAttention 说：我们一块一块地算，每次只搬运一小块 $K/V$ 到料理台上，用“在线合并”的技巧，算完一块就丢一块，全程不把 $N \times N$ 矩阵落盘。

## 1. 问题：标准注意力把钱花在了搬运上

**标准注意力做什么？** 给定三个矩阵 $Q, K, V \in \mathbb{R}^{N \times d}$（$N$ 是序列长度，$d$ 是每个头的维度），注意力输出是：

$$S = QK^\top \in \mathbb{R}^{N \times N}, \quad P = \mathrm{softmax}(S) \in \mathbb{R}^{N \times N}, \quad O = PV \in \mathbb{R}^{N \times d}$$

其中 softmax 按行作用。**问题**：$S$ 和 $P$ 都是 $N \times N$ 的矩阵，标准实现把它们整个物化（materialize，写进 HBM）。序列长 $N=1024$、$d=64$ 时，$S$ 就有 1024×1024 个元素——$N^2$ 的存储。论文 Algorithm 0 描述的标准实现分三步：读 $Q,K$ 算 $S$ 写回；读 $S$ 算 softmax 写 $P$；读 $P,V$ 算 $O$ 写回。每一步都是一次全矩阵的 HBM 往返。

**为什么这很贵？** 大多数算子（softmax 这种逐元素或归约操作）是 memory-bound（内存受限：时间由访存决定而不是由计算决定，见 §2.1）。注意力里 softmax、mask、dropout 全是这种操作。搬运量是 $O(N^2)$ 级别的，序列翻倍，搬运量翻四倍——这就是“注意力平方级昂贵”的真正来源。错误直觉是“注意力慢是因为 FLOPs 多”，实际上标准的精确注意力 FLOPs 是 $O(N^2 d)$,对 $d=64$ 的常见配置，GPU 算 FLOPs 的能力远大于搬数据的能力，瓶颈在搬运。

**算术强度**（arithmetic intensity，每字节访存对应的算术操作数，§2.1）是判断“计算受限还是内存受限”的标尺：强度高 → compute-bound，强度低 → memory-bound。注意力正是强度低的那类。

## 2. 核心机制：在线 softmax + tiling（公式逐项拆解）

**核心思想一句话：把 $K, V$ 切成块，逐块载入 SRAM，每块算完就把部分结果“合并”进输出，块与块之间只传递三个统计量。** 论文用了两个成熟技巧：tiling（分块）和 recomputation（重计算）。本节先拆“在线 softmax”这个数学基础，再走一遍 Algorithm 1。

### 2.1 在线 softmax：不用整行也能算 softmax

softmax 的数值稳定写法，对一个向量 $x \in \mathbb{R}^B$：

$$m(x) := \max_i x_i, \quad f(x) := \left[e^{x_1 - m(x)}, \ldots, e^{x_B - m(x)}\right], \quad \ell(x) := \sum_i f(x)_i, \quad \mathrm{softmax}(x) := \frac{f(x)}{\ell(x)}$$

- $m(x)$：该块的最大值，用于防止 $e^x$ 溢出；
- $f(x)$：减去最大值后的指数值向量（未归一化）；
- $\ell(x)$：$f(x)$ 的和，即归一化常数；
- 最终 softmax = $f(x)/\ell(x)$。

**关键性质**：把两个向量 $x^{(1)}, x^{(2)}$ 拼接成 $x = [x^{(1)}\ x^{(2)}]$，新的统计量可以从旧统计量合并出来（§3.1 公式）：

$$m(x) = \max(m(x^{(1)}), m(x^{(2)}))$$

$$\ell(x) = e^{m(x^{(1)}) - m(x)} \ell(x^{(1)}) + e^{m(x^{(2)}) - m(x)} \ell(x^{(2)})$$

$$\text{softmax}(x) = \frac{\left[e^{m(x^{(1)}) - m(x)} f(x^{(1)}),\ e^{m(x^{(2)}) - m(x)} f(x^{(2)})\right]}{\ell(x)}$$

直觉：**新最大值更新后，旧块的指数值要按“新老最大值的差”打折**（$e^{m(x^{(1)}) - m(x)}$ 这一项），旧归一化常数同理。只要一直带着 $(m, \ell)$ 两个数，就能一块一块地把 softmax 拼完。

**最小例子（手算）**：设 $x^{(1)} = [1, 2]$，$x^{(2)} = [3, 0]$。

- 第一块：$m = 2$，$f = [e^{-1}, e^0] = [0.368, 1.0]$，$\ell = 1.368$；
- 第二块先算自己的统计：$m^{(2)} = 3$，$f^{(2)} = [e^0, e^{-3}] = [1.0, 0.050]$，$\ell^{(2)} = 1.050$；
- 合并：$m^{\text{new}} = \max(2, 3) = 3$；$\ell^{\text{new}} = e^{2-3} \times 1.368 + e^{3-3} \times 1.050 = 0.368 \times 1.368 + 1.050 = 0.503 + 1.050 = 1.553$；
- 验证整行：$x = [1,2,3,0]$，$m=3$，$f = [e^{-2}, e^{-1}, e^0, e^{-3}] = [0.135, 0.368, 1.0, 0.050]$，$\ell = 1.553$ ✓。

两块各自算、合并，结果与整行一致——**这就是在线 softmax（online softmax）**，也叫代数聚合（algebraic aggregation）。

### 2.2 Algorithm 1：分块注意力完整走查

论文 Algorithm 1 的完整流程（我按原文顺序解释，行号对应原文）：

- **行 1**：设块大小 $B_c = \lceil M/(4d) \rceil$（$K/V$ 块的列数）、$B_r = \min(\lceil M/(4d) \rceil, d)$（$Q$ 块的行数），其中 $M$ 是 SRAM 大小。为什么除 4？因为 SRAM 要同时装下 $Q_i, K_j, V_j, O_i$ 四块，每块各占一份预算（原文行 1 注释）；$B_r$ 额外取 $\min(..., d)$ 是论文对算法正确性证明的约束。
- **行 2**：初始化输出 $O = 0$、归一化常数 $\ell = 0$、最大值 $m = -\infty$（都在 HBM）。
- **行 3-4**：把 $Q$ 切成 $T_r = \lceil N/B_r \rceil$ 块，$K, V$ 切成 $T_c = \lceil N/B_c \rceil$ 块。
- **行 5-7（外层循环，红色箭头，论文 Fig.1 左）**：对每个 $K_j, V_j$ 块：载入 SRAM；然后内层循环遍历所有 $Q_i$ 块。
- **行 9**：在片上算 $S_{ij} = Q_i K_j^\top \in \mathbb{R}^{B_r \times B_c}$——注意这个 $S$ 子块只在 SRAM 里存在，从不写回 HBM。
- **行 10**：算该块的 $\tilde{m}_{ij}$（行最大值）、$\tilde{P}_{ij} = \exp(S_{ij} - \tilde{m}_{ij})$（逐元素）、$\tilde{\ell}_{ij}$（行和）。
- **行 11**：合并统计量：$m_i^{\text{new}} = \max(m_i, \tilde{m}_{ij})$；$\ell_i^{\text{new}} = e^{m_i - m_i^{\text{new}}} \ell_i + e^{\tilde{m}_{ij} - m_i^{\text{new}}} \tilde{\ell}_{ij}$——就是 2.1 节的合并公式。
- **行 12**：更新输出 $O_i \leftarrow \mathrm{diag}(\ell_i^{\text{new}})^{-1}\left(\mathrm{diag}(\ell_i) e^{m_i - m_i^{\text{new}}} O_i + e^{\tilde{m}_{ij} - m_i^{\text{new}}} \tilde{P}_{ij} V_j\right)$。逐项读：旧输出 $O_i$ 按“旧最大值到新最大值的折扣” $e^{m_i - m_i^{\text{new}}}$ 缩放，新贡献 $\tilde{P}_{ij} V_j$ 按“本块最大值到新最大值的折扣”缩放，两者加权相加后除以新的归一化常数 $\ell_i^{\text{new}}$——这就是“按比例把旧输出和新输出拼起来”。
- **行 13**：把 $\ell_i, m_i$ 写回 HBM（每行只有两个数，开销可忽略）。
- **行 16**：返回 $O$。

**为什么这样是对的？** Theorem 1：Algorithm 1 返回 $O = \mathrm{softmax}(QK^\top)V$，FLOPs 仍是 $O(N^2 d)$（没有少算），但额外内存只有 $O(N)$（只存 $O, \ell, m$ 三个 $N \times d$ / $N$ 量级的数组，不存 $N \times N$ 矩阵）。

**最小例子（块级走查）**：取 $N=4, d=1$，$Q = [1, 2, 3, 4]^\top$，$K = [1, 0, 1, 0]^\top$，$V = [2, 5, 2, 5]^\top$，块大小 $B_r = B_c = 2$。直接算第 1 行（$q_1 = 1$）：$S = [1, 0, 1, 0]$，$m = 1$，$f = [e^0, e^{-1}, e^0, e^{-1}]$，$\ell = 2 + 2e^{-1} \approx 2.736$，分子 $= e^0 \cdot 2 + e^{-1} \cdot 5 + e^0 \cdot 2 + e^{-1} \cdot 5 \approx 7.679$，$O_1 = 7.679/2.736 \approx 2.807$。

现在按 Algorithm 1 走：外层 $j=1$（$K_1 = [1, 0]$，$V_1 = [2, 5]$），内层 $i=1$ 处理 $Q_1 = [1, 2]$。$S_{11} = [1, 0; 2, 0]$。第 1 行：$\tilde m = 1$，$\tilde P = [1, e^{-1}]$，$\tilde \ell = 1 + e^{-1} \approx 1.368$，分子 $\tilde P V = 1 \cdot 2 + e^{-1} \cdot 5 \approx 3.839$。合并（从 $m=-\infty$ 开始）：$m^{\text{new}} = 1$，$\ell = 1.368$，$O_1 = 3.839/1.368 \approx 2.807$。

外层 $j=2$（$K_2 = [1, 0]$，$V_2 = [2, 5]$），同一 $Q_1$ 块再算：$S_{12} = [1, 0; 2, 0]$（$K_1 = K_2$ 所以数值相同）。第 1 行新块统计：$\tilde m = 1$，$\tilde \ell = 1.368$，分子 $= 3.839$。合并：$m^{\text{new}} = \max(1, 1) = 1$，$\ell^{\text{new}} = e^{1-1} \cdot 1.368 + e^{1-1} \cdot 1.368 = 2.736$，$O_1 = (e^{1-1} \cdot 3.839 + e^{1-1} \cdot 3.839)/2.736 = 7.679/2.736 \approx 2.807$。

结果与直接整行计算完全一致（$\approx 2.807$）✓。第 2 行（$q_2 = 2$）同理：$\ell = 2 + 2e^{-2} \approx 2.271$，分子 $\approx 5.353$，$O_2 \approx 2.358$——读者可自行验证两个块合并结果相同。

这个例子揭示一个反直觉点：**块之间合并不是简单平均，而是“按 $\ell$ 加权”**——如果两个块的最大值不同，旧输出要先乘 $e^{m_i - m^{\text{new}}}$ 折扣再加。例子里两块完全相同（因为 $K$ 前后两半相同），所以折扣因子都是 $e^0 = 1$，看起来像普通平均；换一组不对称的 $K$ 就能看到折扣的作用。

## 3. 为什么省搬运：IO 复杂度分析

论文 §3.2 的 Theorem 2 给出了两个实现的 HBM 访问次数：

- 标准注意力（Algorithm 0）：$\Theta(Nd + N^2)$ 次 HBM 访问；
- FlashAttention（Algorithm 1）：$\Theta(N^2 d^2 M^{-1})$ 次。

逐项解释：$N$ 是序列长度，$d$ 是头维度（典型 64-128），$M$ 是 SRAM 大小（约 100KB 量级）。$N^2$ 项来自 $N \times N$ 注意力矩阵的读写；$N^2 d^2 / M$ 项来自“把 $K,V$ 切块后，每块都要扫一遍全部 $Q$”——$Q$ 被读 $Nd/M$ 次（每次读 $Nd$ 个元素），所以是 $N^2 d^2 / M$。当 $d^2 \ll M$（对 $d=64$, $d^2 = 4096$;$M \approx 100$KB 即约 25K 个 fp16 元素，$d^2$ 比 $M$ 小一个数量级）时，FlashAttention 的访问量远小于标准实现——论文 Fig.2 显示最多可以少约 **9 倍**。

**下界（Proposition 3）**：对 SRAM 大小的全范围，不存在渐进更好的精确注意力算法——即 $o(N^2 d^2/M)$ 不可能在所有 $M \in [d, Nd]$ 上都成立。直觉：对 $M = \Theta(Nd)$（SRAM 大得能装下整个 $Q$），任何算法都必须至少读一遍 $Q$（$Nd$ 个元素），而 $N^2 d^2 / M = Nd$——下界被“输入至少读一次”钉死。论文说这是流式算法文献中常见的“子范围下界”技巧。

**另一个反直觉点（Fig.2 左）**：FlashAttention 的 FLOPs 其实比标准实现多（反向重计算多算了 $O(N^2d)$），但运行时间更短——因为 HBM 访问才是瓶颈。错误做法是只优化 FLOPs 不看访存（论文 §1 点名批评了许多近似注意力方法：FLOPs 降了但墙钟时间没降）。

## 4. 反向传播：用重计算换显存访问

**问题**：反向传播要算 $dQ, dK, dV$,标准实现需要 $S$ 和 $P$ 这两个 $N \times N$ 中间矩阵。如果 forward 不存它们，backward 就没有原料。

**FlashAttention 的答案（§3.1）**：forward 只存输出 $O$ 和每行的 $(m, \ell)$ 两个统计量；backward 时把 $Q, K, V$ 块重新载入 SRAM,在片上重算 $S$ 和 $P$,然后直接算梯度。这本质是选择性梯度检查点（selective gradient checkpointing）：其他实现用检查点换显存往往要牺牲速度，而这里重计算发生在 SRAM 内，避免的是 HBM 大矩阵读写，所以FLOPs 更多但更快（Fig.2 左）。Appendix B 有完整的 backward 算法，本文不展开。

**代价与边界**：多算的 FLOPs 约等于一次 forward;当 $d^2$ 接近 $M$ 时收益变小；论文 Theorem 2 对 forward 和 backward 都成立（backward 同样是 $\Theta(N^2d^2/M)$）。

## 5. 评测：15%、3×、7.6× 这些数字怎么来的

论文 §4 的实验分三块，数字必须核对口径（全部来自原文表格，我在每处标注锚点）：

### 5.1 训练加速（§4.1）

- **BERT-large**（Table 1）：8×A100 上训练到 MLPerf 目标精度 72.0%（掩码语言建模），比 MLPerf 1.1 的纪录实现快 15%（10 次平均）。
- **GPT-2**（Table 2）：OpenWebText 上端到端训练，相对 HuggingFace 实现最高 3×,相对 Megatron-LM 最高 1.7-1.8×;困惑度与基线一致（不改变模型定义）。
- **Long-Range Arena（LRA）**（Table 3）：序列长 1K-4K,FlashAttention 相对标准注意力最高 2.4×;block-sparse 版在 LRA 上 2.8× 且精度与标准注意力持平。
- **注意力算子本身**（Fig.1 右）：GPT-2 配置下相对 PyTorch 实现 7.6×——这是“算子级”数字，与上面“端到端训练”数字是不同口径，不能混用。

### 5.2 长序列带来更好的模型（§4.2）

- **GPT-2 长上下文**（Table 4）：FlashAttention 把 GPT-2 small 的上下文从 1K 拉到 4K,仍比 Megatron 的 1K 版本快 30%,困惑度还好 0.7——用更长的上下文换到了更好的质量。
- **长文档分类**（Table 5）：预训练 RoBERTa 加长序列，MIMIC-III 上 16K 比 512 高 4.3 点（micro F1），ECtHR 上 8K 比 512 高 8.5 点;开篇摘要里的“6.4 点 lift”是这两组提升的概括。
- **Path-X / Path-256**（Table 6）：第一个在 Path-X（序列 16K）上超过随机水平的 Transformer,准确率 61.4%;block-sparse 版把序列推到 64K,在 Path-256 上 63.1%。此前所有 Transformer 系方法（Linformer、Performer、Reformer 等）都是 ✗（内存爆掉或只有随机水平）。

### 5.3 注意力基准（§4.3）

- **Runtime**（Fig.3 左）：前向+反向，A100 40GB,相对 PyTorch 标准实现最高 3×;近似/稀疏注意力的运行时间随序列线性增长，但短序列（512-1024 之间）仍打不过 FlashAttention——因为访存更少；block-sparse 版在所有序列长度上快于所有已知基线。
- **Memory**（Fig.3 右）：显存占用随序列线性增长，相对精确注意力基线最多省 20×;64K 长度下其他方法基本都 OOM（Linformer 例外），FlashAttention 还比 Linformer 省 2×。

## 6. 延伸：block-sparse FlashAttention

**要解决什么**：精确注意力再快也有 $O(N^2)$ 的下限压力；近似注意力（稀疏/低秩）能降 FLOPs,但常因访存开销不落地。

**做法（§3.3）**：给定块状稀疏掩码 $\tilde{M}$（$B_r \times B_c$ 粒度上决定哪些块跳过），只计算非零块的注意力，其余跳过。IO 复杂度（Proposition 4）为 $\Theta(Nd + N^2 d^2 M^{-1} s)$,其中 $s$ 是非零块比例——比 FlashAttention 多出来的大项直接乘上 $s$。当 $s = N^{-1/2}$（局部注意力常用）时是 $\Theta(N\sqrt N)$,$s = N^{-1} \log N$ 时是 $\Theta(N \log N)$。论文实验用固定 butterfly 稀疏模式。

**边界**：这是近似注意力（丢了部分注意力权重），质量由稀疏模式决定；论文用它做 Path-256（64K 序列）与 LRA 加速，但在主流稠密 Transformer（GPT/BERT 全注意力）里，block-sparse 版本并没有成为默认——今天实际用得多的是精确的 FlashAttention 系列。

## 🕰️ 原文时代 vs 当前工程：v2/v3/FA-4 三代演进

FlashAttention-1（2022 年 5 月提交）在 A100 上只达到理论峰值的 **25-40%**（FA2 论文摘要语），它的后继者一步步逼近 GEMM 效率。三代演进的核心矛盾是：瓶颈从“HBM 访问”转移到“线程分工与片上资源”，再到“非矩阵乘单元”。

| 版本 | 时间与出处 | 解决 FA1 的什么瓶颈 | 关键数字（来源） |
| --- | --- | --- | --- |
| FA2（arXiv:2307.08691,2023-07） | FA1 只到 25-40% 峰值，原因是 thread block/warp 分工不佳（低占用率或不必要的 SRAM 读写） | ①减少非矩阵乘 FLOPs ②单 head 跨 thread block 并行 ③warp 间分工减少共享内存通信 | 相对 FA1 约 **2×**;A100 上达 50-73% 峰值；端到端 GPT 训练 225 TFLOPs/s/GPU（72% MFU）（摘要） |
| FA3（arXiv:2407.08608,2024-07） | FA2 在 H100 只有 35% 利用率，没有用上 Hopper 的新硬件能力 | ①warp specialization + TMA 异步重叠 ②块级 matmul 与 softmax 交错 ③块量化 + 非相干处理用上 FP8 | H100 FP16 相对 FA2 **1.5-2.0×**,达 740 TFLOPs/s（75%）；FP8 接近 1.2 PFLOPs/s;FP8 数值误差比基线低 **2.6×**（摘要） |
| FA-4（MLSys 2026） | Blackwell B200 上 Tensor Core 吞吐翻倍，但共享内存带宽、指数单元等**非 MMA 单元**没跟上（超计算 25-60%） | ①围绕全异步 MMA 重排流水线 ②软件模拟指数与条件 softmax 重缩放 ③利用 TMEM 与 2-CTA MMA 减共享内存流量与原子加 | B200 BF16 相对 cuDNN 9.13 最高 **1.3×**、相对 Triton **2.7×**;达 1613 TFLOPs/s（71%）；CuTe-DSL 实现，编译快 20-30×（摘要） |

**当前工程状态（2026-08 复核）**：FA2/FA3 已深度融入 PyTorch（`torch.nn.functional.scaled_dot_product_attention` 的 flash 后端）、vLLM/SGLang/TensorRT-LLM 等引擎，以及 [pagedattention-notes](/AIInfraGuide/papers/pagedattention-notes) 里讲的分页 KV cache 内核；FA-4 论文明确开源并计划集成 PyTorch 与 Megatron-LM。结论边界：三代演进没有改变 FA1 的核心思想（tiling + 在线 softmax + 重计算），改变的是“在哪种硬件上、瓶颈在哪、怎么把流水线排满”；对做算子优化的读者，FA1 的 IO 分析是地基，FA2-4 是“同一思想在 Ampere/Hopper/Blackwell 上的工程化”。

## 7. 常见误读与错误做法

- **误读 1：“FlashAttention 是近似注意力，精度有损。”** 错。FA1/FA2/FA3 算的都是精确 softmax(QK⊤)V,与标准实现数学上完全一致（Theorem 1）；只有 block-sparse 扩展是近似的。把“FlashAttention 有损”当默认前提，会在对比实验里得出错误结论。
- **误读 2：“FlashAttention 快是因为 FLOPs 少。”** 错。它的 FLOPs 与标准实现同阶（反向还更多），快在 HBM 访问从 $O(N^2)$ 降到 $O(N^2d^2/M)$。用 FLOPs 评估它的收益会完全跑偏（论文 §1 对只看 FLOPs 的近似方法批评正源于此）。
- **误读 3：“序列越长 FlashAttention 收益越大，短序列没用。”** 部分错。§4.3 明确：短序列（≤512）下 FlashAttention 仍快于标准实现与大多数近似方法；越过 512-1024 后近似方法的线性复杂度才开始反超。收益与“访存占比”有关，不是单纯的序列长度函数。
- **错误做法 1（实现层面）**：把 $S = QK^\top$ 物化到显存再分块 softmax——块级优化只发生在计算图层面，数据照样全量搬一次，等于白做。FlashAttention 的关键是整个 kernel 融合（§3.1 implementation details：一次 CUDA kernel 完成 matmul→softmax→matmul），中途不落盘。
- **错误做法 2（工程选型）**：在 decode 阶段（单 query 长 KV）直接照搬 FA1 的前向内核。今天推理侧的 decode 注意力用 FlashDecoding 类方案（把 KV 切块并行归约）或 FA2 的变体；FA1 论文本身面向训练前向/反向。选型前先看 [splitwise-notes](/AIInfraGuide/papers/splitwise-notes) 的 PD 解耦语境，理解 prefill 与 decode 的访存形态差异。

## 📝 总结

1. **注意力慢在搬运不在算**：标准实现把 $N \times N$ 中间矩阵进出 HBM 三次，访问量 $O(N^2)$;FlashAttention 用 tiling 把整个计算压进 SRAM,访问量降到 $O(N^2 d^2 / M)$,对典型 $d, M$ 少约一个数量级（Theorem 2,Fig.2）。
2. **两个技巧缺一不可**：在线 softmax（只带 $(m, \ell)$ 两个统计量逐块合并）解决“softmax 必须看整行”；反向重计算（存 $O$ 与 $(m, \ell)$,backward 在片上重算 $S, P$）解决“不存 $N \times N$ 中间矩阵”。
3. **精确且更快**：FLOPs 不变（反向更多），显存 $O(N)$;换来 GPT-2 注意力 7.6×、BERT 训练 15%、Path-X 首个非随机水平 Transformer;v2/v3/FA-4 在 A100/H100/B200 上把它推到 GEMM 级效率。

## 🎯 自我检验清单

- [ ] 能解释“为什么 $O(N^2)$ 注意力矩阵的物化是瓶颈，而 FLOPs 不是”（用算术强度与 A100 的 HBM/SRAM 带宽数字）。
- [ ] 能手算一次两块的在线 softmax 合并（给出 $x^{(1)}, x^{(2)}$,写出 $m, \ell$ 的合并公式并验证与整行一致）。
- [ ] 能逐项解释 $\Theta(N^2 d^2 M^{-1})$ 中 $N^2$、$d^2$、$M^{-1}$ 的来源，并说明 $d^2 \ll M$ 时为什么省。
- [ ] 能说出 FA1 反向传播存了哪些状态、为什么重计算反而更快。
- [ ] 能区分四个口径的加速数字：7.6×（算子）、3×（GPT-2 端到端 vs HF）、15%（BERT vs MLPerf）、1.5-2.0×（FA3 vs FA2,H100），并指出各自出处。
- [ ] 能指出 FlashAttention 的三代演进各针对哪个瓶颈（HBM 访问→线程分工→非 MMA 单元）。

## 📚 参考资料

- FlashAttention 原文（NeurIPS 2022）：https://arxiv.org/abs/2205.14135 （v2,2022-06-23）
- FlashAttention-2:https://arxiv.org/abs/2307.08691 （v1,2023-07-17）
- FlashAttention-3:https://arxiv.org/abs/2407.08608 （v2,2024-07-12）
- FlashAttention-4（MLSys 2026 论文）：https://proceedings.mlsys.org/paper_files/paper/2026/file/ae8b0b5838ba510daff1198474e7b984-Paper-Conference.pdf
- 官方代码：https://github.com/Dao-AILab/flash-attention
