# Eligibility 热点、旧任务漂移与类别级链条审计（仅 Task1–5）

本轮是**只读诊断**：使用 seed=1993 的正式 ViT cache、原始 `R,b,k=41`、Task1–5 正式 checkpoint；没有改正式模型或任何参数，没有重新训练 `W_B` 或正式 ridge。Eligibility 与 XOR 在诊断脚本内按正式随机流确定性重放；Task1–5 的 `W_B` 均与 checkpoint **逐 bit 相同**。对 10 个 `old_task i → new_task j`（`i<j≤5`），真实 flip 对旧测试码的整数 overlap 增量与 `WB_j−WB_i` 直接计算**逐元素闭合，max_abs_error=0**。Eq.(5) 的形式及名词沿用 [BTSP-CAM 论文](../../2233_BTSP_CAM_A_Brain_Inspired.pdf)；以下数值只描述当前 reproduction。

数据口径：A 部分为 Task2–5 各 2,500 个训练流样本；碰撞四分位在**每个 task 内**按其对历史训练码的 mean collision 稳定排序，再合并成每组 2,500 张。每个 task 开始前，用全部历史训练 sparse code 频率确定 top 1%（82 bit）与 top 5%（410 bit）；它们是嵌套组。每张样本的 41 个活动位、更新后的当前类/rival eligibility、历史全局频率、当前类此前频率与旧任务频率在 [逐 bit 表](../diagnostics/familiarity_bit_attribution.csv)，逐样本指标和四分位结果在 [逐样本表](../diagnostics/familiarity_sample_diagnostics.csv)与[分位表](../diagnostics/familiarity_by_collision_quartile.csv)。`current_class_frequency_before` 在该类首样本时为空；这些历史频率没有未来样本泄漏。Eq.(5) 的 `F` 与 rival cosine 对前轮确定性 replay 的最大误差为 **6.33×10⁻¹⁵**。

## 1. `F` 为什么随 collision 增加？

同一批高频位同时抬高历史碰撞和当前类 trace 的内积。Q1→Q4：活动位中历史 top 5% 的占比 **30.04%→62.23%**，top 1% **8.36%→27.77%**；`s·e_y` 总 numerator **92.22→110.83**，其中 top 5% 部分 **40.10→84.94**，非 top 5% 反而 **52.12→25.89**。top 5% 的 numerator 份额 **43.10%→76.44%**。`||e_y||₂` 平均 **40.00→41.61**，其中 top 5% 子空间范数 **30.38→34.74**；分母增大没有抵消 numerator 的集中增长，`F` **0.3645→0.4183**。范数不是可加贡献，故不能把子空间范数之和当总范数。

逐样本历史 mean collision 与 top 5% 活动位占比的 Pearson/Spearman 为 **+0.876/+0.863**（class 内去均值后）；与 `F` 为 **+0.254/+0.283**。这些是当前轨迹里的共享 bit 证据，而不是“热点独立造成 `F` 增加”的因果干预。[全部相关性](../diagnostics/familiarity_associations.csv)。

## 2. `F/N` 对 `I` 的抵消主要来自热点共享吗？

**在离线 cosine 归因口径下，是主要解释。** Q1→Q4 的原始 `I` **0.8749→0.7665**，原始 `F` 上升；仅在现有 `s` 和 eligibility 上屏蔽 top 5% 维度后，`F_no_top5` 却 **0.3825→0.2892**，`I_no_top5` **0.9059→0.9004**，诊断 `f_eff_no_top5` **0.001608→0.001075**（−33.1%），与真实 `f_eff` 的 +5.3% 方向相反。只屏蔽 top 1% 时，诊断 `f_eff` 从 **0.001535→0.001332**（−13.2%）。去热点不是合法的正式训练轨迹：trace、plateau 抽样和 `W_B` 完全没有重新演化，不能把诊断差值解释为执行“去热点模型”的效果。

## 3. 同一 class 内是否仍有门控抵消？

**仍有。** 在每个 class 内中心化后，历史 mean collision 对 `F`、`I`、`G_FN=4N(1−N)F`、真实 `f_eff` 的 Pearson 分别为 **+0.254、−0.844、+0.272、+0.091**；未中心化分别为 **+0.253、−0.826、+0.271、+0.073**。因此不是仅由高碰撞 class 与低碰撞 class 的组成差异造成。class 固定效应只是观察性控制，仍可能有流顺序、样本难度和 trace 状态混杂。

## 4. Task1 的 feature drift 能代表其他旧 task 吗？

**现象方向可代表，幅度不能直接代表。** 固定各来源 task 全部 500 张测试 sparse code，对其学习时 `WB_i` 与 `WB_5` 计算阈值前 `a=WB·s/41`；表中 cosine 与相对 L2 为**逐样本平均**，best-ρ residual 是每样本最小二乘标量拟合后的残差除以终态范数。原始余弦 margin 只在来源 task 自身 5 个类别、500 张 query 上计算。[完整 source×eval 矩阵](../diagnostics/all_old_task_feature_drift.csv)。

| 来源 | 到 WB5 的 mean cosine | mean relative L2 | mean best-ρ residual ratio | 来源 margin → WB5 margin |
|---:|---:|---:|---:|---:|
| Task1 | 0.8340 | 1.3397 | 0.5457 | 0.08384 → 0.01964（−76.6%） |
| Task2 | 0.8916 | 0.8887 | 0.4491 | 0.04314 → 0.01657（−61.6%） |
| Task3 | 0.9385 | 0.5210 | 0.3429 | 0.03341 → 0.01894（−43.3%） |
| Task4 | 0.9749 | 0.2613 | 0.2208 | 0.02664 → 0.02156（−19.1%） |

四个旧 task 的 same-class 与 different-class cosine 均升高，但后者上升更快，margin 均压缩。Task1 是最长暴露、最低初始 occupancy 的 cohort，不能拿它的 76.6% margin 损失当所有旧类的典型值。

## 5. Memory age 与 drift 是否有系统关系？

**在每个固定来源 task 内，随 checkpoint 变晚，cosine 逐步下降、相对 L2 和非标量 residual 上升，margin 逐步下降**；Task1 的 mean cosine 依次为 `1.000→0.916→0.876→0.852→0.834`。Task2/3/4 的每步轨迹在完整矩阵中也单调。不同 cohort 的**一步** cosine 分别为 Task1→2 **0.916**、Task2→3 **0.941**、Task3→4 **0.963**、Task4→5 **0.975**，说明 age 之外还有学习阶段/基线密度差异；不能由此单独估计“年龄因果效应”。

Task2–5 共 **1,267,909** 次真实 flip。每个新 task 的事件按方向×碰撞四分位×task-start 热度，先整数聚合 `ΔW` 再乘全部较早 task 的固定 query；全部 **430 行**分组结果在 [all_task_flip_to_drift.csv](../diagnostics/all_task_flip_to_drift.csv)。`gross_query_weighted_movement=Σ_flip query_fanout(bit)/41` 不允许方向抵消；`net_feature_l1/l2` 对组内事件先相加，不能跨组直接加范数。parallel/orthogonal 分量以各来源 `a_i` 为参考。每个 source→new checkpoint 的联合分组之和以及连续事件累积均对直接 checkpoint 差闭合，最大整数误差 **0**。因此不是只发生在 Task1 probe 上的代数效应。

## 6. 哪些 class 更容易漂；能由热点/碰撞解释吗？

按各类自身学习 task 到 WB5 的**阈值前**样本 best-ρ residual，四个旧 cohort 内最大者分别为增量 class **2（0.614）**、**8（0.495）**、**13（0.357）**、**16（0.248）**；不同 cohort 不能不校正 age 就直接排名。25 类的热点占比、与其他类测试码的 mean `J` 和 `P(J≥5)`、有效支持 `exp(H)`、bit entropy、imprint column 数、后续暴露/flip、feature 几何、正式 ridge 只读 accuracy 均在 [class_level_drift_chain.csv](../diagnostics/class_level_drift_chain.csv)。这里 class 的 collision 是其 **100 张测试码 × 其他 24 类的 2,400 张测试码**；它不同于前轮全训练码跨 task 的 4.741% 高尾口径。top1/top5 依赖度使用全 Task1–5 训练频率确定的**回顾性固定集合**，不用于在线决策。

类别级观察链**没有形成稳定的正向预测关系**。在 20 个旧类上，top5 占比与 mean `J` 的 Pearson/Spearman **+0.912/+0.901**，但与样本非标量 residual 的原始 Pearson 只有 **+0.190**；控制学习 task 后为 **−0.574**。`P(J≥5)` 对后续 imprint-bit exposure rate、非标量 residual 的原始 Pearson 分别为 **−0.411、+0.273**，比 mean `J` 的绝对值 **0.343、0.133** 更大；控制学习 task 后尾部系数分别 **−0.484、−0.730**，mean `J` 分别 **−0.375、−0.621**。因此高尾在本小样本中**数值上更相关**，但方向混杂且不能声称它更好地预测有害 drift。全链 Pearson/Spearman、原始与学习 task 固定效应都见 [相关性表](../diagnostics/class_level_chain_correlations.csv)。

“旧 imprint bit”定义为该 class 训练样本曾触发 0→1 的**column 集合**；后续暴露指新训练 code 再命中这些 column，后续 1→0/0→1 为发生在这些 column 上的 flip。一个 column 可同时归属多类，不能把这些计数加成全局 flip 数，也不能认定某个 1→0 必定抹去同一 class 的具体旧 synapse。`later_01_query_weighted` 则以该类全部 100 张测试码 fanout 对实际后续 0→1 事件计权，仍是实际轨迹的功能暴露，不是删除事件后的模型反事实。

正式 readout 的类别级 accuracy **无法可靠排序遗忘**：在学习时 Task1 为 20.2%，Task2–4 的当前类几乎全为 0%；Task5 全 25 类累计仅 **3.96%**，测试 `z` 非零密度 **5.76×10⁻⁶**。Task5 的正确预测几乎全来自增量 class 0（96/100）；其他 24 类几乎全为 0。这是阈值 collapse 与 argmax 默认类造成的地板效应，不是各类都“零遗忘”。CSV 仍如实给出 checkpoint 原 ridge 在学习时、Task5 的只读 eval 及 accuracy drop；没有重新拟合。

## 7. Prototype stability 会掩盖样本几何损伤吗？

**会掩盖一部分，但不能据此否认 prototype 也在变。** 旧 20 类平均 `a` prototype cosine **0.9649**，而配对测试样本 cosine **0.9098**。例如 class 5 的 prototype cosine **0.9664**，样本 cosine **0.8982**、样本 best-ρ residual ratio **0.4374**；class 10 分别 **0.9873、0.9454、0.3237**。共同方向保留可使 prototype 看似稳定，个体特征仍有较大非标量变化。这里测的是阈值前 `a`，不能和论文 Appendix F.2 的理想化阈值后 `z` 稳定性证明直接等同。

## 8. Eq.(5) 是否表现为 `I` 识别风险、`F/N` 补偿风险？

**是，就当前真实编码统计和现有训练轨迹而言。** Q1→Q4：`F` **0.3645→0.4183**，`N=1−F` **0.6355→0.5817**（靠近 0.5），`G_FN` **0.3331→0.4002**（+20.1%），`I` **0.8749→0.7665**（−12.4%），最终算术平均 `f_eff` **0.001455→0.001532**（+5.3%）。`F` 及其确定性变换 `N` 不能视为两个独立随机原因。

对数恒等式 `log f_eff = log f_base + log[4N(1−N)] + log F + log I` 只在正值样本计算：Q1 **2,497** 个、Q4 **2,498** 个。组内**平均 log** 的 Q4−Q1 分别为 `Δlog novelty prior=+0.055916`、`Δlog F=+0.145254`、`Δlog I=−0.132731`，精确相加 `Δlog f_eff=+0.068439`（误差约数值舍入）。这相当于几何平均比约 **+7.1%**；它不必等于上面的**算术平均** +5.3%。`I` 的抑制在乘法结构中被 F 与 novelty prior 的两项增益超过。该结论仅针对现有 `F,I` 的观察分布，不提出新 gating、参数选择或 accuracy 优化。
