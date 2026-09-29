# Eq.(5) 门控与热点的等预算 CRN 反事实重放（Task1–5）

这是**诊断实验，不是论文实现或正式模型修改**。正式 `src/`、config、`tau/k/p/beta/lambda/gamma`、全部 checkpoint 和正式 ridge 均未改。采用相同的正式 sparse code、标签、样本顺序、eligibility 更新与 XOR `p=0.5`；分支只替换每样本的 plateau 概率。论文定义来自 [BTSP-CAM Eq.(5)–(6)](../../2233_BTSP_CAM_A_Brain_Inspired.pdf)，其中 `F=cos(s,e_y)`、`N=1−F`、`I=exp(−β max rival cosine)`、`G=4N(1−N)F=4F²(1−F)`，`f_actual=f_base G I`。这里的 `f_base=0.005`、其他数值均是当前 reproduction 的原配置；没有据结果选择参数。

## 三种轨迹与公共随机数

1. **OFFICIAL TRAJECTORY / ORIGINAL_REPLAY**：独立使用原 plateau RNG 和 synapse RNG、按原抽样调用顺序重放。Task1–5 每个 `W_B` checkpoint 均逐 bit 相同；12,500 个样本的 plateau/0→1/1→0 计数与前轮正式 replay **零不匹配**，概率最大误差小于 `1e−12`。[原轨迹验证](../diagnostics/original_replay_validation.csv)。
2. **CRN DIAGNOSTIC BASELINE / ACTUAL**：公式仍为 `f_base G I`，但从零 `W_B` 用诊断 seed **20260925** 的另一套公共随机数重放。它**不应**与正式 checkpoint 逐 bit 相同，不能混作正式结果。
3. **COUNTERFACTUAL BRANCH**：与 CRN ACTUAL 共用同一套随机数及初始零 `W_B`。对每个样本，先生成完整的 `U_plateau[4096]` 和 `U_flip[4096,41]`，再交给所有分支；即便某 branch 未选中 slot，对应 flip 随机数也已生成，不会随分支消耗错位。随机数用 float32 Uniform[0,1)；每个 `r` 对应已排序的第 `r` 个 active bit。内存中逐样本生成完整格点，而不是保存 8.4 GB 的全流张量；固定 seed 和样本顺序使每个 `(t,j,r)` 可重复。

Eligibility 只依赖相同的 `s,y` 流、不依赖 `W_B` 或 plateau，因而所有分支共享前轮已与原 replay 对齐的逐样本 `F,I`。top1/top5 为**每个 task 开始前**旧训练码累计频率最高的 82/410 bit。masked 分支仅在 cosine 的 `s` 和各类 eligibility 上屏蔽这些位；**实际写入仍用完整 41 bit**。Task1 尚无历史 bit，HOT 分支与 ACTUAL 完全相同。

分支：`ACTUAL=f_base G I`；`NO_I=f_base G`；`I_ONLY_SHAPING_RATE_MATCHED=mean_t(f_actual)·I/mean_t(I)`；`G_ONLY_SHAPING_RATE_MATCHED=mean_t(f_actual)·G/mean_t(G)`；`HOT5_MASKED_GATING=f_base G_mask5 I_mask5` 及逐 task 乘 `mean_t(f_actual)/mean_t(f_hot5)` 的等预算版本。top1 的 raw/matched 版本是附加敏感性检查。`I_ONLY` 写法等价于先取 `G_ref=mean_t(G)` 再选择 task scalar 消去其幅度；保留的是 **I 的样本间排序**，不是一个新的论文公式。每个 task 的四个 `RATE_MATCHED` 分支与 ACTUAL 的 `mean_f` 最大差 **4.34×10⁻¹⁹**，期望 plateau 数最大差 **3.64×10⁻¹²**；实际数仍可因 Bernoulli 结果相差最多 **122/task**。每个 branch/task 的 signed flip 矩阵与其 CRN `W_B` 前后差逐元素闭合；进一步乘 Task1–4 的固定旧测试码后，事件累计 `Δa` 与终态直接计算在 **32 个 branch×old-task 组合**上逐元素闭合，最大整数误差均为 **0**。

## 写入预算和碰撞分配

下表汇总 Task2–5 的 10,000 个新样本。Q1/Q4 使用**真实历史 collision** 的 task 内四分位（每 task 各 625 张），没有按反事实 `W_B` 重新排序。旧 imprint 为新 task 开始时该 branch 的活跃旧 synapse；“exposure”是选中 slot 与当前 active bit 中仍为 1 的旧 synapse 数；“old 1→0”是这些 synapse 实际被 flip 到 0 的次数。[逐 task 写入预算](../diagnostics/counterfactual_write_budget.csv)、[逐 task×四分位分配](../diagnostics/counterfactual_collision_allocation.csv)。

| CRN branch | 期望 / 实际 plateau（Task2–5） | gross flips | Q4/Q1 实际 plateau/样本 | Q4/Q1 旧 imprint exposure/样本 | Q4/Q1 旧 1→0/样本 |
|---|---:|---:|---:|---:|---:|
| ACTUAL | 61,651 / 61,881 | 1,267,989 | **1.056** | **1.999** | **1.982** |
| NO_I（不等预算） | 75,174 / 75,405 | 1,545,545 | 1.200 | 2.218 | 2.206 |
| I_ONLY_SHAPING_RATE_MATCHED | 61,651 / 61,836 | 1,266,743 | **0.870** | **1.666** | **1.655** |
| G_ONLY_SHAPING_RATE_MATCHED | 61,651 / 61,830 | 1,267,149 | 1.205 | 2.292 | 2.277 |
| HOT5_MASKED_GATING（不等预算） | 56,746 / 56,740 | 1,162,543 | 0.669 | 1.245 | 1.235 |
| HOT5_MASKED_GATING_RATE_MATCHED | 61,651 / 61,755 | 1,264,834 | **0.670** | **1.234** | **1.224** |

ACTUAL 的 Q4/Q1 **期望** plateau 比为 **1.053**，I-only 等预算为 **0.876**，G-only 等预算为 **1.202**，HOT5 等预算为 **0.669**；实际抽样方向相同。Q4 的真实旧 imprint exposure/样本从 ACTUAL **39.09** 降到 I-only **35.67**（−8.8%），HOT5 等预算 **28.65**（−26.7%）；旧 1→0/样本分别为 **19.44、17.70、14.20**。因此 `I` 确实把写入机会从高碰撞样本移开；ACTUAL 中随碰撞升高的 `F/N` 因子部分抵消了这一作用。屏蔽热位的**门控视野**即使保留全 41 bit 写入，也能强烈改变 allocation。旧 imprint exposure 在 Q4/Q1 仍大于 1，说明碰撞样本即便少获得 plateau，单次写入仍更可能遇到旧 synapse。

## 旧任务功能漂移

每个 branch 的 Task1–4 测试码各 500 张，比较该 branch 在**任务学习完成时**的 `a_i=W_B^{(i)}s/41` 与 Task5 的 `a_5`。cosine/relative L2 是逐样本指标；`ρ*` 是对该来源 500×4096 矩阵拟合的**单一**标量，residual ratio 为 `||a_5−ρ*a_i||_F/||a_5||_F`。gross functional movement 对每次真实 CRN flip 累加 `old_query_fanout(bit)/41`，不能相互抵消；net L1/L2 在事件相加后计算。parallel/orthogonal 分量以该 branch 自己的学习时 `a_i` 为基准。所有 source×branch 数值、两端 same/different cosine 与 margin 见[功能漂移表](../diagnostics/counterfactual_functional_drift.csv)。

| CRN branch | Task1 old mean cosine | Task1 global-ρ residual ratio | Task1 gross movement | 四个旧 task 平均 residual ratio | 当前任务 margin 均值（Task1–5） |
|---|---:|---:|---:|---:|---:|
| ACTUAL | 0.83628 | 0.53145 | 314,724 | 0.37975 | 0.04211 |
| I_ONLY_SHAPING_RATE_MATCHED | **0.82983** | **0.54244** | 307,220 | **0.38446** | 0.04092 |
| G_ONLY_SHAPING_RATE_MATCHED | 0.83663 | 0.52992 | 317,496 | 0.37961 | 0.04173 |
| HOT5_MASKED_GATING（不等预算） | 0.84194 | 0.52200 | 281,148 | 0.37507 | 0.04361 |
| HOT5_MASKED_GATING_RATE_MATCHED | **0.83710** | **0.53021** | 305,917 | **0.37668** | 0.04258 |

**I-only 等预算没有在此 CRN 轨迹中更保护旧 `a`。** 它的 Q4 暴露与旧 1→0 都降低，但 Task1 cosine 比 ACTUAL **低 0.00645**、global-ρ residual ratio **高 0.01099**；四个旧 task 的平均 residual ratio **0.38446 vs 0.37975**。gross movement 小 2.4%，但方向抵消、memory 状态及 class 几何使 gross 减少不等于净漂移改善。I-only 的 Task1 学习时 `W_B` 与 ACTUAL 也不同，所以跨 branch 的“学习→Task5”漂移不是保持初始 Task1 memory 不变的纯后续干预。

**HOT5 等预算有小幅、非全面的旧几何改善。** Task1 cosine **+0.00081**、global-ρ residual ratio **−0.00123**，四个旧 task 平均 residual ratio **−0.00307**；但 Task1 终态 class margin **0.01954 vs ACTUAL 0.01976**，并非每项都改善。raw HOT5 的更大 gross/drift 下降同时少了约 **8.0%** 的期望写入预算，不能将 raw 与 ACTUAL 的全部差归于分配。HOT5 在 Task1 与 ACTUAL 具有**同一 CRN `W_B^{(1)}`**，因此该比较不存在 I-only 的首任务基线差异。

`NO_I` 虽然多写，Task1 的学习→终态 cosine **0.85047**、ρ residual ratio **0.50920** 看似比 ACTUAL 好；但其学习时/终态 Task1 class margin **0.07334/0.01620**，均低于 ACTUAL 的 **0.08426/0.01976**。相对于自身基线“漂得少”不能单独定义为“类信息保护得好”；本轮不以某一指标选择分支。

CRN ACTUAL 的 Task1 gross 功能移动中，按新 task 开始前 top5 bit 归组的份额为 **77.95%**，与正式轨迹前轮约 **78.0%** 一致。HOT5 等预算仍有 **76.22%** 的 gross 移动发生在这些高 fanout 位；mask 改的是 gate，不是写入内容。前者是**固定实际轨迹内的 write-content/fanout 归因**，后者是**更改门控统计后的干预差**，两者有交互，绝不能相加或解释为“78% 与 2.8% 的因果分摊”。

## 当前任务写入能力和限制

本轮只在每个 task 完成后用该 branch 的 `W_B` 对**当前 task 500 张测试码**计算未阈值化的 `a` 类内/类间 cosine margin；没有使用正式 `tau=0.5`，也没有重训正式 ridge。允许但非必需的当前任务 5-way ridge probe **未运行**，CSV 明确标记 `NOT_RUN_OPTIONAL`。[当前任务几何](../diagnostics/counterfactual_plasticity.csv)。

ACTUAL 的 Task1–5 margin 为 **0.08426、0.04370、0.03351、0.02716、0.02195**；I-only 等预算为 **0.08329、0.04166、0.03213、0.02627、0.02124**（五任务均值约 −2.8%，仍全部大于零）；HOT5 等预算为 **0.08426、0.04437、0.03408、0.02793、0.02226**（均值约 +1.1%）。这说明两个干预在这项几何指标上都**没有使当前任务信号明显消失**，但 margin 不能替代分类 accuracy。NO_I 多写约 22%，当前 margin 均值却从 ACTUAL **0.04211** 降到 **0.03613**，说明“多写”也不是单调的 plasticity 收益。

## 对七个问题的判断

1. **`I` 是否减少高碰撞写入？是。** NO_I 的 Q4/Q1 plateau 比 1.200，ACTUAL 为 1.056；等预算 I-only 进一步降至 0.870。原始无 I 分支总写入更多，只有等预算分支可用于纯分配比较。
2. **为什么 ACTUAL 没充分减少？** 真实编码中高碰撞同时抬高 `F`，令 `G=4F²(1−F)` 增大；其正向作用部分抵消 I。实际 Q4/Q1 期望写入仍 1.053，而 G-only 等预算为 1.202。
3. **等预算 I-only 更保护旧表示吗？本次没有。** 它降低 Q4 旧 imprint 暴露和 1→0，但四个旧任务的非标量 residual 略升；不能把 allocation 改善自动当作 representation 保护。
4. **hot5-masked gate 改变 drift 吗？会，但终态幅度小。** 等预算 Task1 residual ratio −0.00123，四旧任务均值 −0.00307；一些 margin 指标不一致。raw 版本降得更多，但少写，不能作等预算结论。
5. **热点主要路径更像哪一条？就本轮的旧 `a` 功能漂移，A：write-content/fanout 证据更强。** 动态 top5 位仍贡献约 76%–78% gross movement；B：gating-state contamination 明显影响 plateau allocation，但只带来小的终态 drift 变化。两路径相互作用，不能给加和百分比。
6. **减少 interference 会明显损害当前任务写入吗？未见明显消失。** I-only 当前任务未阈值化 margin 平均低约 2.8%，仍为正；未运行 5-way probe，故不声称 accuracy 保持。
7. **是否出现清晰 stability–plasticity tradeoff？没有简单的单调交换。** I-only 分配更避碰撞，却没有改善旧表示且当前 margin 略低；HOT5 等预算当前 margin 略高、旧 residual 略低，但旧 margin 不一致。一个固定 CRN seed 的小效应不能当作一般性或论文性能结论。
