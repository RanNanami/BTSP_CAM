# 多 k writer drift 只读审计

本轮只读取既有 CIFAR-100 B0-Inc5、class-order/RF/writer seed 1993 的结果。没有训练 ViT、`W_B` 或 ridge，没有修改正式源码、配置、旧 checkpoint 或旧结果，也没有 commit/push。主指标始终是 **pre-threshold `a=W_Bs/k`**；没有混入 `tau`。计算脚本现归档于 `experiments/audit/k_writer_drift_readonly.py`（原执行时为 `tmp/` 中的同一脚本），运行约 **43.75 秒**。

## 来源、可用性与相同样本

| k | 完整 Task1–20 `W_B` | Task1 sparse-code 来源 | writer event 来源 |
|---:|---|---|---|
| 41 | `CHECKPOINT_NOT_AVAILABLE`；只有另一组前 5 task checkpoint | — | `k_path_decomposition.csv`, `evolving_aligned` |
| 64 | `CHECKPOINT_NOT_AVAILABLE`；只有 development 前 5 task checkpoint | — | 同上 |
| 96 | 完整；`k_tau_joint_cache/k96/full20/` | `k_tau_joint_cache/k96/full_test_task01.npz` | 同上 |
| 128 | 完整；`tau_oracle_k128/` | 没有单独保存测试码；从冻结 ViT feature cache 在**内存**中确定性重建 | 同上；checkpoint 还含 event 元数据 |
| 160 | 完整；`tau_oracle_k160/` | 同 k128 | 同上；checkpoint 还含 event 元数据 |
| 1024 | 完整；`high_k_mechanism_cache/k1024/` | 同目录 `test_task01.npz` | `high_k_stress_full20.csv`, `FIXED_FBASE` |
| 6144 | 完整；`high_k_mechanism_cache/k6144/` | 同目录 `test_task01.npz` | 同上 |

另外检查的 k768、4096、7168 没有本轮可用的完整 `W_B` checkpoint；有旧汇总日志或 sparse cache 不能代替逐 task 二值矩阵。**没有补训缺失轨迹**。k41/64 的完整 event 汇总和 density 可读，但没有可配对的 full20 writer checkpoint，因此不计算它们的 Hamming、`a` 或 prototype drift；没有将不同 development/前 5 task 分支与 full20 日志拼接。

固定 500 个 Task1 **TEST** 样本（5 类各 100 个）的身份来自 `data/feature_cache_btsp_cam/test_task_01.npz` 中的 CIFAR indices。其有序索引 SHA-256 为 `a1a1988a3a4f7c223afea56cf7e7b3beca82a057ec7549855a53ad97fbfc0fa9`。从这些冻结的 768-D feature 只在内存中应用固定 `R,b`、stable exact top-k，所得 500 条 k96/1024/6144 代码与各自磁盘 sparse cache **逐 bit 一致**，标签/顺序亦相同。k128/160 使用同一批有索引的 feature 行、同一 RF seed 与编码器，仅产生内存中的 `s`，不保存 cache，不运行 ViT，也不训练模型。所有可用 `W_B` 与相应旧日志的 density 逐 task 一致；带元数据的 checkpoint 之 gross/plateau 与旧日志也一致。

## Paper fact 与本地定义

**PAPER FACT:** 论文叙述 CA1 slowly drifting，并报告 prototype similarity `>0.9`。公开文本不足以把其 Figure 4 prototype 的样本集、`a`/`z` 选择、归一化和跨 task 聚合口径唯一对应到本轮数值。因此本轮的 `>0.9` 只能是**同一批 Task1 测试样本、pre-threshold `a` 的本地判据**；不是论文 Figure 4 的精确复现。

对每个类 `c`，计算 `p_c^(t)=mean_{x∈Task1 test,c} a_t(x)`，并取 `cos(p_c^(1),p_c^(t))`；另逐样本取 `cos(a_1(x),a_t(x))`。零向量检查通过。`W_B` Hamming 是两个 checkpoint 的最终位状态差，不能当成期间 gross flips。

## Task1→Task20 结果

| k | active ratio | prototype cosine `a` mean | sample cosine `a` mean | 5 类原型 `>0.90` | Task20 `W_B` density | WB1→20 Hamming / 总位 | 20-task gross flips | plateau slot events / task |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 41 | 0.50% | NA | NA | NA | 0.093686† | NA | 6,315,038† | 15,406.4† |
| 64 | 0.78% | NA | NA | NA | 0.130349† | NA | 10,172,644† | 15,896.4† |
| 96 | 1.17% | **0.914461** | 0.875013 | 5/5 | 0.172041 | 0.167883 | 15,685,598 | 16,339.6 |
| 128 | 1.56% | **0.923963** | 0.896713 | 5/5 | 0.206143 | 0.201681 | 21,311,700 | 16,649.6 |
| 160 | 1.95% | **0.930057** | 0.909534 | 5/5 | 0.235043 | 0.230428 | 26,985,399 | 16,865.4 |
| 1024 | 12.50% | **0.965775** | 0.963692 | 5/5 | 0.460870 | 0.458812 | 177,204,796 | 17,304.3 |
| 6144 | 75.00% | **0.873629** | 0.873572 | 0/5 | 0.499988 | 0.500038 | 374,735,203 | 6,098.9 |

† k41/64 只有旧 full20 event/density CSV，没有可读取的同轨迹 full20 checkpoint。prototype 的 5 类 mean/median/min/max/std 和 `>0.90/0.95/0.99` 比例均在 `k_writer_drift_readonly_summary.csv`；逐类值在 `k_task1_to_task20_prototype_drift.csv`。k96 的 mean `0.9144607454473238` 与独立旧 Figure-4-style audit 的 pre-threshold 值 `0.9144607454473231` 在浮点精度内一致。

状态密度的 1/5/10/15/20 task 快照、各 task 的 `0→1`/`1→0`、净变化和相对于 WB1 的 Hamming 位数均见 `k_writer_state_change.csv`。其中 k96/128/160 的方向性 gross flip 由既存 gross count 与 checkpoint 的 active-bit 净变化严格解出：`LTP=(gross+net)/2`, `LTD=(gross−net)/2`；整数奇偶及非负断言全通过。k1024/6144 的方向性旧日志与同一解法一致。**这不是从 Hamming 猜测 gross flips**：单个位可被多次 XOR。

## 时间轨迹与写入强度

| k | Task2 prototype | Task5 | Task10 | Task15 | Task20 | 早期 Task1→2 损失 | 后续 Task2→20 损失 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 96 | 0.970620 | 0.938468 | 0.922122 | 0.916811 | 0.914461 | 0.029380 | 0.056159 |
| 128 | 0.972496 | 0.943821 | 0.930311 | 0.925578 | 0.923963 | 0.027504 | 0.048532 |
| 160 | 0.973601 | 0.947609 | 0.935586 | 0.931918 | 0.930057 | 0.026399 | 0.043544 |
| 1024 | 0.981366 | 0.969359 | 0.966555 | 0.966046 | 0.965775 | 0.018634 | 0.015591 |
| 6144 | 0.909234 | 0.876069 | 0.873669 | 0.873513 | 0.873629 | 0.090766 | 0.035605 |

低/中 k 的损失早期较快，随后仍继续缓慢下降；k1024 在 Task5 后几乎平台化；k6144 在 Task2 就出现明显跳降，Task10 后约稳定在 0.874。完整 20 点轨迹见 `k_task1_drift_trajectory.csv`。

**DESCRIPTIVE_CORRELATION_ONLY。** 五条有 checkpoint 的轨迹，以 `prototype_drift=1−prototype_cosine_a` 计算 Pearson：与 `WB_1_to_20_fraction_changed` 的 `r=+0.119`，与累计 gross flips 的 `r=+0.483`，与每 task 平均 plateau slot events 的 `r=−0.849`（n=5）。若只看主范围 k96/128/160，随着 Hamming 与 gross flips 增加，prototype drift 反而下降；但 n=3 且 k 同时改变 sparse code、occupancy 和 overlap 尺度，不能将相关解释成写入对稳定性的因果作用。高 k6144 又反转趋势。因此总体判为 **`DRIFT_NON_MONOTONIC_IN_K`**，不能由 bit 修改数量单独解释功能 drift。k1024 在接近 46% 位发生净变化时仍有 0.966 prototype cosine，也提醒高余弦可能部分来自公共分量，不能单凭原型 cosine 断言分类几何或 ridge 坐标一致。

## Plateau 计数口径

旧日志的 `plateau_count` 是每样本在 4096 个 slot 上 `plateau=True` 的数量之和，故 CSV 使用名 `slot_plateau_events`，不是 `samples_with_any_plateau`。虽然能除以每 task 2500 个训练样本得到“每样本平均 slot events”（例如 k96 为 6.53584），旧 full20 日志未保存逐样本 event 数或布尔值，不能回推出：至少一次 plateau 的样本数、正样本的平均 plateau slot 数、正样本比例。`k_plateau_count_semantics.csv` 为这些字段留空并标记 **`PLATEAU_DENOMINATOR_NOT_RECOVERABLE_FROM_EXISTING_LOGS`**。所以此前约 16,340 slot events/task 与论文约 550 的数值 **DENOMINATOR_AMBIGUOUS**，本轮不能证明差异只是计数口径，也不能直接比较。

## 结论与边界

- `LOW_DRIFT_REGIME_ALREADY_EXISTS`：**YES，仅按本地 pre-threshold Task1→20 原型定义。** 主范围 k96/128/160 的均值和每一类都大于 0.90；可用全部轨迹中 k1024 最高（0.965775），主范围中 k160 最高（0.930057）。这不是作者真实 k 的线索，也不是论文 Figure 4 的指标复现。
- `DRIFT_NON_MONOTONIC_IN_K`：k96→1024 随 k 增大而更稳，k6144 则比 k96 更漂。跨 k 写入量、输入码、密度共同变化；只能描述一致性，不能判因果。
- 原型与样本层次不同：k96 prototype 0.914，但同批 500 个 sample mean 只有 0.875。高原型稳定不保证所有旧样本稳定，更不保证历史 ridge sufficient statistics 在新 `W_B` 下仍保持相同坐标。
- 没有读出/accuracy 实验，也没有动 `tau`；`a` 主指标刻意不混入 CA1 threshold。当前结果不可据此反推论文隐藏参数。

## 新文件与轻量核查

新 CSV：`diagnostics/k_writer_drift_readonly_summary.csv`（7 行），`diagnostics/k_task1_to_task20_prototype_drift.csv`（25 行），`diagnostics/k_task1_drift_trajectory.csv`（100 行），`diagnostics/k_writer_state_change.csv`（140 行），`diagnostics/k_plateau_count_semantics.csv`（140 行）。只读脚本归档：`experiments/audit/k_writer_drift_readonly.py`；本轮并未复运行该已归档入口。

仅运行读取/统计有关的检查：脚本编译；每 task checkpoint/log density 对齐；带元数据 checkpoint 的 plateau/gross 对齐；缓存 sparse codes 逐 bit 对齐；exact-k、binary overlap、Task1 self-cosine、flip parity；独立读取新 CSV 核对行数、方向性 flip 恒等式、逐类→汇总算术及 k96 旧审计回归。全部通过；没有运行完整训练测试套件。
