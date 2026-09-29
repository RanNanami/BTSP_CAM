# Coordinate-aligned ridge：Task1–5 实验性修复

这是独立的 **experimental repair**，不是 BTSP-CAM 论文原算法。实现位于 `experiments/coordinate_aligned_ridge.py`，运行配置与校验脚本位于 `experiments/configs/coordinate_aligned_ridge_task1_5.yaml` 和 `experiments/run_coordinate_aligned_ridge.py`。正式模型、默认配置、原始 W_B checkpoint 均未修改。本轮仅用原始 R0 sparse encoder、R0 的 Task1–5 W_B 轨迹、相同 class order/训练样本顺序和 gamma；CA1 始终为未阈值化的 `a=W_B s/k`，没有恢复或搜索 tau。逐 task 数值见 `diagnostics/coordinate_aligned_ridge.csv`。

## 原型的状态与数学闭合

对每个新任务的原始二值 exact-k=41 sparse code，原型只把当前批次并入

\[
C_t=\sum_{i\le t}s_i s_i^\top=S_{\le t}^\top S_{\le t},\qquad
D_t=\sum_{i\le t}s_i y_i^\top=S_{\le t}^\top Y_{\le t}.
\]

它在内存中持久保存 `C_t`（float64 CSR）和 `D_t`（float64 dense）以及样本计数；**不保存任何单样本 sparse code、历史图像、ViT feature h 或 CA1 z**。给定当前只读 W_B checkpoint，临时计算

\[
A_t=W_B C_t W_B^\top/k^2,\qquad B_t=W_B D_t/k,
\]

然后调用原有 `StreamingRidge.solve()` 计算 `(A_t+\gamma I)^{-1}B_t`。A/B 和求解器对象仅用于本次读出，不属于原型的 C/D 持久状态。原型不修改或重新训练 W_B。

独立**验证 harness**为构建 current-WB replay oracle，会重新读取先前已有的历史 ViT feature cache、编码 sparse code 并构造历史 z。这个 replay 仅用作逐元素核对，不能与原型的 no-history 状态混为一谈；整个研究工作区原有 cache 也没有因本轮被删除。

## Task1–5 准确率

所有数字为 CIFAR-100 已见类别的累计测试准确率。R0/R1 使用前轮已核验的五任务 ladder；本轮重新计算 aligned 和 current-WB replay oracle，后者逐 task 与前轮 O1 数值完全一致。

| Task | R0 演化 W_B + 旧 A/B | Coordinate-aligned C/D | Current-WB replay oracle | R1 冻结 WB1 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 92.40% | **92.40%** | 92.40% | 92.40% |
| 2 | 51.40% | **89.80%** | 89.80% | 90.60% |
| 3 | 36.87% | **87.07%** | 87.07% | 88.33% |
| 4 | 33.70% | **85.55%** | 85.55% | 86.15% |
| 5 | 33.00% | **85.52%** | 85.52% | 85.92% |

Task5 aligned 的旧类/当前类准确率为 **85.00% / 87.60%**，与 replay oracle 一致。相对 R0，累计准确率恢复 **52.52 个百分点**。这是在不把历史单样本数据存入原型状态的条件下，精确同步 ridge 统计量与当前 W_B 的五任务证明；不是论文算法的准确率，也不代表 Task6–20 已验证。

## 逐层数值核对

浮点差来自 oracle 先以 `z=sW_B^T/k` 做矩阵乘法、aligned 路径先汇总整数计数再除以 k/k²；两者代数相同。所有五个 checkpoint 的逐样本预测完全一致，累计 accuracy 差为 **0**。

| Task | max\|A_aligned−A_oracle\| | max\|B_aligned−B_oracle\| | max\|W_out 差\| | max\|logit 差\| |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 5.68e−14 | 8.53e−14 | 8.63e−12 | 2.07e−12 |
| 2 | 1.99e−13 | 1.14e−13 | 2.01e−11 | 5.50e−12 |
| 3 | 4.55e−13 | 1.14e−13 | 3.68e−11 | 1.14e−11 |
| 4 | 6.82e−13 | 1.14e−13 | 4.50e−11 | 1.34e−11 |
| 5 | 1.59e−12 | 1.14e−13 | 6.46e−11 | 1.92e−11 |

运行时断言使用绝对容差 A≤1e−7、B≤1e−8、权重/logit≤1e−6，且要求完全相同的 argmax；实际误差均远低于界限。额外的小型合成测试直接比较 C/D 推出的 A/B 与 `Z^TZ`、`Z^TY`，也通过。

## C/D 存储与更新成本

C 的形状固定为 8192×8192，共 67,108,864 个位置。CSR `nnz` 统计完整对称矩阵（不只上三角）；密度分母为 8192²。每任务 2500 个样本、每码 41 位，相当于 **4,202,500 次**有序 `s_i s_i^T` 元素贡献。实现使用批量 CSR `S_t^T S_t` 和 CSR 加法，并非 Python 逐样本循环。表中时间为本机单次 wall-clock 测量，不是跨设备 benchmark。

| Task | C 累计 nnz | C 密度 | C float64 CSR 实占 | C 更新耗时 | 当前 W_B 对齐乘法耗时 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,611,052 | 2.40% | 18.47 MiB | 0.137 s | 3.08 s |
| 2 | 2,865,777 | 4.27% | 32.83 MiB | 0.170 s | 5.29 s |
| 3 | 3,893,163 | 5.80% | 44.59 MiB | 0.177 s | 8.10 s |
| 4 | 4,767,945 | 7.10% | 54.60 MiB | 0.199 s | 9.45 s |
| 5 | **5,513,927** | **8.22%** | **63.13 MiB** | **0.195 s** | **11.71 s** |

Task5 D 的形状为 8192×25：float64 **1.5625 MiB**，理论 float32 **0.7813 MiB**。Task5 C 的不同表示估算：dense float64 **512 MiB**，dense float32 **256 MiB**；实际 CSR float64 **63.13 MiB**，假设同样索引布局的 CSR float32 **42.10 MiB**；以两个 int32 坐标估算的 COO float64 **84.14 MiB**，COO float32 **63.10 MiB**。float32 仅做容量估算，**未**声称其能达到本轮 float64 的数值等价。

稀疏 C 的持久占用不等于求解峰值。Task5 中间乘积 `W_B C` 已有 **28,519,966** 非零项，按 float64 CSR 约 **326.4 MiB**；对齐后的 4096×4096 A 已全稠密，其 float64 数组本身 **128 MiB**，尚未计入求解器工作区、W_B 和验证 harness 的 replay 数组。Task5 对齐乘法约 11.71 s、ridge 求解约 0.96 s；这是正确性优先的原型，不是生产级内存/时间优化。后续任务 C 可能继续增密，本轮不能外推至 20 tasks。

## 五个问题的直接回答

1. **C/D 能否复现 replay oracle？** 能。A/B、权重和 logit 在 float64 容差内一致，Task1–5 的预测和准确率完全一致。
2. **是否需要历史 raw exemplar？** 原型更新与再次对齐不需要。它只保留聚合 C/D 和当前 W_B；历史 cache 只由独立 oracle 验证 harness 读取。
3. **Task5 是否恢复到约 85.52%？** 是，aligned 与 replay oracle 均为 **85.52%**，高于 R0 的 33.00%，接近 R1 的 85.92%。
4. **C 的存储与计算成本？** Task5 CSR float64 **63.13 MiB**、551.4 万 nnz；更新当前任务约 0.20 s，对齐乘法约 11.71 s，但临时 `W_B C` 与稠密 A 显著增加峰值内存。dense float64 C 则需要固定 512 MiB。
5. **是否 exemplar-free？** 若定义为“不存历史单样本图像、ViT feature、sparse code 或 CA1 feature”，原型符合；若暗示“无需额外历史信息/近乎常数存储”，则**不符合**。C/D 是从全部过去 sparse codes 累积出的高维聚合统计量，C 具有 O(m²) 最坏存储并可能泄露历史分布信息。本方案必须明确标为 **experimental repair**，不能归入论文原算法。
