# CA1 threshold scale 与 `W_B` drift：Task1–5 诊断

日期：2026-09-23。**全部结果仅为 diagnostic**；没有重训 BTSP memory、没有修改默认配置或正式模型、没有更改 `p/k/f_q/beta/gamma/lambda`，没有运行 Task6–20 或据准确率选择参数。来源为已有 `diagnostics/wb_checkpoints/WB_task01.npz` 至 `WB_task05.npz`、正式 checkpoint 的 ridge 权重，以及相同的 frozen-ViT 特征缓存。五个 `W_B` 快照经逐元素核对与对应正式 checkpoint 完全相同。可重复执行的隔离脚本为 `experiments/run_tau_drift_diagnostics.py`。

## 指标和设计边界

- `tau_sweep.csv` 中 `current_accuracy` 与 `new_accuracy` 都是在**所有已见类 logits** 上对当前任务测试样本求准确率，沿用上一轮诊断口径；Task1 的 `old_accuracy` 不适用，CSV 写 `nan`。`cumulative_accuracy` 使用截至 checkpoint 的所有测试任务。表中准确率为百分数；CSV 为 0–1 比例。
- `ca1_density` 是累计测试集上非零 `z` 元素的比例；`fraction_nonzero` 是至少有一个非零 CA1 槽的**样本**比例；`mean_z/p99_z` 在所有样本与槽上统计。
- **A1/refit_ridge**：每个候选 `τ` 独立运行一次 Eq.(8) 的 streaming sufficient statistics。第 `t` 任务的训练样本用 `WB_task t` 产生 `z_τ`，只加入当任务统计；旧统计不重算，`gamma` 保持原值。这保留了原始持续学习中的特征空间漂移。
- **A2/frozen_ridge**：读取每个正式 task checkpoint 内原先在 `τ=0.5` 下训练的 ridge 权重，**不更新任何统计或权重**，只在评估时变换 `z_τ`。除 `τ=0.5` 外，这是有意的 train/eval 特征分布不匹配探针，不是有效的重新训练方案。
- **B/unthresholded**：`z=a`，以 A1 的相同 streaming 方式重新训练 ridge；与 `τ=0` 数学上相同，单独以 `method=unthresholded` 记录用于核对。
- **C/drift**：只对固定 Task1 训练／测试样本做**五分类**探针，不是 25 类正式 CIL。`frozen` 用 `WB_task1` 上拟合的 Task1 ridge，之后只换测试 `W_B`；`recomputed` 则离线用对应 `W_B` 重新计算 Task1 训练特征和 ridge。后者使用历史 Task1 数据，**只允许作 oracle 诊断，不符合正式 no-replay 更新**。

## A / B：固定记忆下的阈值尺度

每个 checkpoint 的 `W_B` 固定不变；改变的仅是离线特征提取阈值及 A1 中相应的 diagnostic ridge。下表给出完整网格中最能说明问题的 Task1/Task5 切片；全部 85 行见 `diagnostics/tau_sweep.csv`。

| `τ` | Task1 refit 准确率 | Task5 refit 新类 | Task5 refit 旧类 | Task5 refit 累计 | Task5 frozen-ridge 累计 | Task5 CA1 非零密度 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 92.4% | 94.4% | 17.65% | 33.00% | 57.48% | 98.990% |
| 0.01 | 93.2% | 94.2% | 18.40% | 33.56% | 58.36% | 98.990% |
| 0.02 | 94.0% | 94.0% | 18.60% | 33.68% | 59.28% | 98.990% |
| 0.05 | 90.6% | 94.4% | 19.90% | 34.80% | 62.68% | 86.695% |
| 0.1 | 86.4% | 92.4% | 29.05% | 41.72% | 66.36% | 59.034% |
| 0.2 | 67.6% | 80.6% | 47.45% | 54.08% | 64.48% | 12.064% |
| 0.3 | 36.4% | 47.4% | 32.70% | 35.64% | 36.56% | 1.000% |
| 0.5 | 20.2% | 0.0% | 4.95% | 3.96% | 3.96% | 0.000576% |

**解释 A：** 首任务从原阈值 `τ=0.5` 的 20.2% 到 `τ=0` 的 92.4%，或本诊断网格 `τ=0.02` 的 94.0%，证明在相同 `W_B` 下 `a` 的可分类结构没有消失；主要即时损失是 `a→z` 的阈值尺度。Task1 `τ=0.5` 时仅 0.2% 测试样本有任一非零槽；Task5 累计测试集也只有 1.6% 样本有非零槽。`τ=0` 的显式 unthresholded 行与阈值网格中的 `τ=0` 完全一致。

**本次网格中表现较高的范围，不是参数选择：** Task1 的 90% 以上出现在 `τ=0–0.05`；Task5 的 A1 累计准确率相对较高出现在 `τ=0.1–0.2`，本次八点中 `τ=0.2` 为 54.08%，但它的 Task1 仅 67.6%。A2 在 Task5 的八点中 `τ=0.1` 为 66.36%，却是在保持原 `τ=0.5` ridge、只改测试特征的有意不匹配条件下。**不存在由这组结果支持的跨任务“论文最佳 τ”。** 改变阈值可以缓解归零，但没有解决 A1 中的增量旧类损失。

## C：不经阈值的 `W_B` 漂移隔离

固定同一批 Task1 测试样本及其标签，比较 `a_j=WB_task j·s/||s||_1`。准确率都是 Task1 原来 5 类之间分类；与首任务 92.4% 的基线比较。

| 评估 `W_B` | `a_j` 对 `a_1` 均值 cosine | 冻结 Task1 ridge | 用该 `W_B` 离线重建 Task1 ridge |
|---:|---:|---:|---:|
| Task1 | 1.0000 | 92.4% | 92.4% |
| Task2 | 0.9164 | 66.8% | 90.6% |
| Task3 | 0.8761 | 58.4% | 89.2% |
| Task4 | 0.8520 | 56.8% | 86.6% |
| Task5 | 0.8340 | 46.8% | 86.6% |

从 `WB_task1` 到 `WB_task5`，固定 Task1 ridge 的准确率损失为 **45.6 个百分点**（92.4→46.8）。在同一 `WB_task5` 上离线重建 Task1 ridge 可恢复 **39.8 个百分点**（46.8→86.6），即该**五分类探针**损失的 **87.3%**；剩余 **5.8 个百分点／12.7%** 没有通过重拟合恢复。定义为 `(recomputed_WB5 − frozen_WB5)/(frozen_WB1 − frozen_WB5)`。这量化了**固定旧分类器与新记忆特征不匹配的可恢复部分**，不能直接解释为正式 Task5 25 类 CIL 旧类错误中有 87.3% 来自漂移；新类 logits 竞争、旧类统计与不同 task 特征源都未在这个五分类控制实验中出现。

## 两个问题必须分开

**First-task collapse — `a→z` threshold：** 在没有任何跨任务漂移的 Task1，原 `τ=0.5` 已把几乎所有测试特征清零；同一 `W_B` 的 unthresholded ridge 达 92.4%。所以不能以 `W_B` drift 解释首任务的直接崩溃，也不能据此断言 `W_B` 没有分类信息。

**Incremental forgetting — `W_B` drift / ridge mismatch：** 在没有阈值的 Task1 五分类探针中，后续 `W_B` 使旧 ridge 的准确率降到 46.8%，而在同一新 `W_B` 上重建 readout 后达到 86.6%。A1 的 unthresholded Task5 新类仍有 94.4%，但累计只有 33.0%、旧类 17.65%；这与持续写入引起旧统计的特征空间不一致相符，不过 C 探针不完全分解正式 25 类 CIL 的所有遗忘来源。

## 下一步优先级（本轮不执行）

1. **参数来源**：核对作者是否公开 `τ`、`p` 及 threshold 校准协议。这里要解决的是可复现性来源，不能用本次网格的最高准确率反推论文参数。
2. **ridge consolidation**：针对连续变化的 `W_B`，审计论文 Eq.(8) 是否有固定特征源、历史统计重算或其它一致性约束；用不修改正式模型的 oracle／冻结记忆对照细分 CIL 旧类损失。C 部分表明这比立刻重写 CA1 算子更直接。
3. **写入规则**：作为下一层核对 `p` 的来源及 `W_B` 的长期漂移／列占用偏斜。现有 `a` 在当前任务仍有强分类能力，故写入规则不是首任务 collapse 的首要怀疑对象；也不能因本轮结果认定它完全正确。

## 复核

输出为 `diagnostics/tau_sweep.csv`（8 个阈值 × 5 个 checkpoint × 2 种模式，加 5 个显式 unthresholded 行，共 85 行）和 `diagnostics/wb_drift_isolation.csv`（5 个评估记忆 × 2 种 ridge 模式，共 10 行）。`τ=0.5` 的每 task refit 累计准确率与已有 `diagnostics/ca1_readout_ablation.csv` 的正式 thresholded 结果相同；`τ=0` 与显式 unthresholded 行相同。CSV 按一列一变量、一行一观察记录，缺失的 Task1 旧类准确率标记为 `nan`。
