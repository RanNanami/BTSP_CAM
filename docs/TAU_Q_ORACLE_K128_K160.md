# k=128/160 的离散 CA1 threshold：current-WB replay-oracle 诊断

日期：2026-09-28。**这是实验性 replay oracle，不是 BTSP-CAM 作者实现，也不是 exemplar-free 修复或正式 tau 选择。**正式 reproduction、默认配置、既有 checkpoint 和 coordinate-aligned ridge 均未改动。本轮经用户授权，才为原来没有保存 `W_B` 的 k=128/160 实验重建独立轨迹；因此本报告完成了先前 [`TAU_PROVENANCE_AND_DISCRETE_THRESHOLD_AUDIT.md`](TAU_PROVENANCE_AND_DISCRETE_THRESHOLD_AUDIT.md) 第 5 节因缺 checkpoint 而未运行的实证部分。

## 1. 锁定设置及轨迹验证

- CIFAR-100 B0-Inc5，20 tasks，class order 1993；沿用已缓存的 frozen ViT features、原训练/测试样本顺序、原 random-feature encoder、binary BTSP writer 和 ridge `gamma`。RF seed 与 write seed 均为 **1993**。k=128 对应先前 default-seed paired/path 实验；k=160 对应先前独立 path 实验。除 k 外没有为本轮调整参数。
- 重建脚本：[`experiments/run_tau_q_oracle_k128_k160.py`](../experiments/run_tau_q_oracle_k128_k160.py)。每个 task 在独立目录 `diagnostics/tau_oracle_k128/`、`diagnostics/tau_oracle_k160/` 保存 `WB_task01.npz` 至 `WB_task20.npz`；未覆盖原 checkpoint。
- 每个 task 的 plateau count、gross flips 和 `W_B` density 与归档的 [`k_path_decomposition.csv`](../diagnostics/k_path_decomposition.csv) 一致；k=128 的归档 path accuracy 也与 default-seed paired run 相互一致。重建后，**先**在全部 20 个 task 以 `q=0` 做 current-WB replay，逐 task 的 cumulative、old、current、current-5way accuracy 与归档轨迹在 `1e-12` 容差内一致，然后才放行 `q>0`。

| 轨迹 | Task20 `q=0` final | Task1–20 归档 accuracy | Task1–20 plateau/gross/density | Task20 `W_B` density |
|---|---:|---|---|---:|
| k=128, seeds 1993/1993 | **80.62%** | 全部一致 | 全部一致 | 20.61% |
| k=160, seeds 1993/1993 | **81.16%** | 全部一致 | 全部一致 | 23.50% |

原实验没有保存这两组 `W_B`，故不能与原矩阵逐 bit 比对；上面的写入统计、确定性配置和独立 `q=0` 预测是可用的一致性证据，不能把它表述为“与原 checkpoint 逐 bit 相同”。

## 2. Oracle 和指标定义

对每个 `(k,t,q)`，只读取对应重建的 current `WB_t`。将截至 `t` 的所有历史**训练** sparse codes 用此同一个 `WB_t` 重新编码并重拟合 ridge，随后评估所有已见**测试**类。没有使用 `C/D` coordinate-aligned closure，因为 `q>0` 时 `z=max(0,SW_B^T/k-q/k)` 是非线性的，`C=Σssᵀ`、`D=Σsyᵀ` 不足以精确恢复 ridge 统计量。历史 cache 用于诊断性 replay；这些结果**不是** exemplar-free 算法。

`W_B` 和 sparse `s` 都为 binary，`J=W_Bs` 为整数，`a=J/k`，`z=max(0,(J-q)/k)`。严格按 Eq.(7) 的 `z>0 ⇔ J>q`，所以 `tau=q/k` 时 **`J≤q` 被截断**。预声明级别为 `q=0,1,2,3,4,5,8,12,16,24,32`；另单独检查 `tau=0.5`，对应 k=128 的 `q=64` 与 k=160 的 `q=80`。跨 k 固定 q 只统一截断的 overlap *count*，不使 `J` 分布或正值幅度相同。

`CA1_element_density`、非零样本比例、每样本活跃单元、`J` 直方图、`z` 幅度与 cosine geometry 均在**所有已见测试样本**上测量。CSV 中 `P_J_0`…`P_J_32` 加 `P_J_gt32` 是每个 CA1 元素的完整分箱概率；累计/old/current/current-5way accuracy 由重新拟合的相同 readout 给出。20-task average incremental 是 20 个 cumulative accuracy 的算术均值。

完整逐 task 指标：[k128 CSV](../diagnostics/tau_q_oracle_k128.csv)、[k160 CSV](../diagnostics/tau_q_oracle_k160.csv)；每份均有 **240 行**（20 tasks × 12 级），包含 `J` p50/p90/p95/p99、`mean_z`、p95/p99 `z` 和 same/different cosine/margin。[固定 q 汇总](../diagnostics/tau_q_frontier.csv) 有 24 行。直方图每行求和为 1；对 `q≤32`，`density=P(J>q)` 的最大误差为 `1.1e-16`。

## 3. 固定 q 的 20-task sparsity–accuracy frontier

所有准确率、密度以百分数表示。`q=64/80` 仅是描述性的 `tau=0.5` 对照，**不是**与另一 k 相同 count 的跨 k 比较。

| k=128 q | tau=q/k | 平均增量 | Task5 | Task10 | Task15 | Task20 | Task20 密度 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 86.49 | 90.40 | 86.42 | 83.20 | 80.62 | 100.00 |
| 1 | 0.0078125 | 86.48 | 90.36 | 86.42 | 83.20 | 80.62 | 100.00 |
| 2 | 0.015625 | 86.49 | 90.36 | 86.42 | 83.20 | 80.62 | 100.00 |
| 3 | 0.0234375 | 86.46 | 90.36 | 86.42 | 83.21 | 80.62 | 100.00 |
| 4 | 0.03125 | 86.44 | 90.28 | 86.40 | 83.21 | 80.62 | 100.00 |
| 5 | 0.0390625 | 86.43 | 90.28 | 86.40 | 83.21 | 80.62 | 100.00 |
| 8 | 0.0625 | 86.38 | 90.28 | 86.42 | 83.23 | 80.62 | 100.00 |
| 12 | 0.09375 | 86.22 | 90.12 | 86.38 | 83.24 | 80.59 | 100.00 |
| 16 | 0.125 | 85.94 | 89.76 | 86.36 | 83.21 | 80.57 | 100.00 |
| 24 | 0.1875 | 84.54 | 85.76 | 86.10 | 83.11 | 80.56 | 99.91 |
| 32 | 0.25 | 80.78 | 79.96 | 83.70 | 82.55 | 80.27 | 97.70 |
| 64 | 0.5 | 10.07 | 4.72 | 5.80 | 11.89 | 19.95 | 0.46 |

| k=160 q | tau=q/k | 平均增量 | Task5 | Task10 | Task15 | Task20 | Task20 密度 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 86.90 | 90.48 | 86.34 | 83.65 | 81.16 | 100.00 |
| 1 | 0.00625 | 86.90 | 90.48 | 86.34 | 83.64 | 81.16 | 100.00 |
| 2 | 0.0125 | 86.90 | 90.40 | 86.34 | 83.64 | 81.16 | 100.00 |
| 3 | 0.01875 | 86.89 | 90.40 | 86.34 | 83.64 | 81.17 | 100.00 |
| 4 | 0.025 | 86.87 | 90.40 | 86.34 | 83.64 | 81.17 | 100.00 |
| 5 | 0.03125 | 86.88 | 90.40 | 86.34 | 83.64 | 81.17 | 100.00 |
| 8 | 0.05 | 86.89 | 90.44 | 86.34 | 83.64 | 81.17 | 100.00 |
| 12 | 0.075 | 86.88 | 90.52 | 86.34 | 83.65 | 81.18 | 100.00 |
| 16 | 0.1 | 86.82 | 90.68 | 86.34 | 83.65 | 81.18 | 100.00 |
| 24 | 0.15 | 86.26 | 90.64 | 86.34 | 83.65 | 81.16 | 100.00 |
| 32 | 0.2 | 85.29 | 88.16 | 86.26 | 83.64 | 81.17 | 99.98 |
| 80 | 0.5 | 12.09 | 4.60 | 7.06 | 15.60 | 24.63 | 0.68 |

表中 Task20 的 100.00% 是两位小数舍入；精确数值在 CSV。预先给出的诊断判据是 **Task20 density 相对 q0 至少降低 20%，且 Task20 accuracy 下降不超过 0.5pp**。在预声明 `q≤32` 中，两种 k 都**没有**达到 20% 稀疏化：k128 的 q32 仅降低 **2.30%** 密度、Task20 下降 0.35pp；k160 的 q32 仅降低 **0.024%** 密度、Task20 变化 +0.01pp。`tau=0.5` 虽超过 20% 稀疏化，但 accuracy 分别下降 **60.67pp** 与 **56.53pp**。因此没有满足该双条件的固定 q；这是机制判据，不是选择最优 tau 的测试集规则。

## 4. 为什么同一 q 的功能随 task 和 k 变化

| k | Task | `W_B` density | J mean | J p50 | J p90 | J p99 |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 2.83% | 13.89 | 13 | 25 | 36 |
| 128 | 5 | 9.72% | 29.07 | 29 | 39 | 47 |
| 128 | 10 | 14.78% | 38.27 | 38 | 48 | 56 |
| 128 | 15 | 18.15% | 43.07 | 43 | 52 | 60 |
| 128 | 20 | 20.61% | 46.27 | 46 | 55 | 63 |
| 160 | 1 | 3.51% | 19.44 | 18 | 33 | 47 |
| 160 | 5 | 11.61% | 39.93 | 40 | 51 | 62 |
| 160 | 10 | 17.25% | 51.49 | 51 | 62 | 72 |
| 160 | 15 | 20.90% | 57.31 | 57 | 68 | 76 |
| 160 | 20 | 23.50% | 61.12 | 61 | 71 | 79 |

`q=1` 在数学上已截去 `J=1`，但 Task1 的密度只从 98.39% 降到 97.28%（k128）、从 98.70% 降到 98.32%（k160）；Task20 两者的实测密度均为 100%。这不是“q=1 没有支持集效应”，而是后期几乎没有 `J≤1` 的测试元素。

若把“明显稀疏”具体化为**相对 q0 的元素密度降低至少 20%**，预声明 q 网格中首次达到该线的位置随 task 上移：k128 在 Task1 为 q8、Task5 为 q24、Task10 为 q32、Task15/20 的 q≤32 均达不到；k160 在 Task1 为 q12、Task5 为 q32、Task10/15/20 的 q≤32 均达不到。早期这些点也不能被称为稳定收益：例如 k128 Task5 q24 密度降低 27.38%，accuracy 同时下降 4.64pp；k160 Task5 q32 密度降低 20.26%，accuracy 下降 2.32pp。k160 Task1 q12 曾降低 26.26% 密度且 accuracy 多 0.4pp（500 个测试样本中的 2 个），但这一单 task 小差异既不是稳健收益，也无法跨 task 保持稀疏化。

相同 q 下，k160 的 J 分布整体右移，因此通常比 k128 **更难**稀疏化：Task5 q32 时，k128 密度 30.91%、accuracy 79.96%，k160 密度 79.74%、accuracy 88.16%；Task20 q32 时分别是 97.70% 与 99.98% 密度。相同 q 确保截断 count 的定义一致，却不保证相同的 sparsification strength、正值 `z=(J-q)/k` 尺度或 accuracy cost。

`tau=0.5` 对应的 q64/80 都高于各自 Task20 的 J p99（63/79），所以只留下不到 1% 的 CA1 元素；Task20 accuracy 分别仅 **19.95%**、**24.63%**。这在更大 k 下依然是 **destructive truncation**，而非有效 retrieval competition。

## 5. 判断和边界

1. **重建复现了原结果吗？** 是：k128 Task20 80.62%，k160 Task20 81.16%，tau=0 逐 task 与归档一致；写入统计也逐 task 一致。原矩阵未归档，故不宣称 bytewise 对照。
2. **q=1 已明显 sparsify 吗？** 没有。它只截断稀少的 `J=1`；到 Task20 元素密度没有可测下降。
3. **需要截多少 counts？** 在 20% 密度下降这个描述性尺度上，早期随 task 为 q8→q24→q32（k128）或 q12→q32（k160）；后期 `q≤32` 均不足。更大的固定 q 是否可满足“稀疏且准确”不在本轮预声明网格，不能从这里推断或外推。
4. **有明显 sparsify 且几乎不掉 accuracy 的 q 吗？** 在全 20-task 固定 q、Task20 ≥20% density reduction 与 ≤0.5pp accuracy drop 的事先判据下：**没有**。小 q 保留 accuracy 但近乎不稀疏；更高 q 在早期稀疏时有 accuracy 代价；tau=0.5 严重破坏。
5. **k128 与 k160 相同 q 是否相似？** 截断逻辑相同，但 J 分布不同，密度和准确率代价不相同。固定数值 tau 更不具可比性。
6. **threshold 更像什么？** 在当前重建轨迹和 oracle 下，低 q 主要是**幅度平移/近乎 cosmetic**，中等 q 呈**稀疏度–准确率 tradeoff**，tau=0.5 是**destructive truncation**；未见跨 20 task 稳定的“有效 retrieval competition 且不损分类”证据。个别 cosine margin 增大并不能替代分类准确率的证据（例如 k128 Task5 q32）。

这些数值是在全历史训练 cache 可访问的 current-WB replay oracle 下得到的**测试集机制诊断**。它们不能用于选择正式 tau，不能证明作者使用任一 k 或 tau，也不消除论文未公开 tau 的来源问题。科研数据表规范用于明确每列的统计总体、单位和诊断/模型选择边界。
