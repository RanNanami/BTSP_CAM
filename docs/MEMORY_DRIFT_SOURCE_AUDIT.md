# BTSP 二值记忆漂移来源审计（Task1–5）

审计日期：2026-09-23。本报告只讨论 `W_B` 的写入和演化；没有修改模型、配置或参数，没有训练新轨迹。为恢复 checkpoint 未保存的逐样本 plateau/flip 事件，用原训练特征缓存、seed 1993 和原写入顺序在内存中确定性重放 Task1–5。**五个 task 末的 `W_B` 与原诊断 checkpoint 和正式 checkpoint 均逐 bit 相同**；各 task 的 0→1、1→0 总数也与既有 `btsp_learning_dynamics.csv` 完全相同。因此下述“实际事件”是原轨迹的恢复，不是另一次参数实验。

## 机制事实与审计边界

论文第 5 页 Eq.(5) 用类别 eligibility 的 familiarity/novelty 和最大 rival cosine 构成单个样本级 `f_eff=f_base·4N(1−N)·F·I`，其中 `I=exp(−β·max rival cosine)`。然后所有 4096 个 slot 各自以同一 `f_eff` 抽取 plateau。Eq.(6) 仅对 plateau slot 与当前稀疏码的 41 个 active bit 抽取 Bernoulli(`p`) XOR flip：原 bit 为 0 时是 0→1，原 bit 为 1 时是 1→0。本地 `eligibility.py`、`plateau.py`、`binary_memory.py` 和 `model.py` 保持这一顺序。**gating 是全局概率缩放，不是逐 slot 的 collision detector；LTD 由 bit 现值决定，不检查该 bit 属于哪个旧类。**

`W_B` 只有二值矩阵，没有持久的 class/slot 所有权。为避免把某行凭空归属某类，本审计定义“旧 task 印迹”为 task `i` 中真实发生过 0→1、且在 task `i` 末仍为 1 的 bit 集合。`old_task_slot=1` 表示该 row 至少有一个这样的 bit。这是**写入来源代理**，不是唯一的语义记忆标记；同一 row/bit 以后可以被多个 task 重用。

[memory_interference.csv](../diagnostics/memory_interference.csv) 对所有 `old_task<new_task≤5` 的每个 slot 给出一行，共 10×4096＝40,960 行。主要字段定义如下：

| 字段 | 定义 |
|---|---|
| `old_activation` / `new_activation` | 当前新 task 写入前 `W_B^(j−1)` 与写入后 `W_B^j` 的该 row 活跃 bit 比例；分母 8192。 |
| `flip_01` / `flip_10` | 同两个 checkpoint 之间该 row 的**净** 0→1 / 1→0 bit 数。同一 `new_task` 的净变化会随不同 `old_task` 在 CSV 重复，汇总时不可重复相加。 |
| `collision_count` | 新 task 的 2500 个 sparse cue 中，与该 slot 的旧 task 保留印迹至少共享一个 active bit 的样本数；这是**潜在碰撞**，不要求抽到 plateau。 |
| `class_overlap` | 新 task 稀疏 cue 与旧印迹在该 slot 的平均共享 bit 数 / 41。`old_task` 与 `new_task` 各包含 5 类；这是**任务组级、slot 级重叠代理**，不是某两类 prototype 的余弦。 |
| `plateau_collision_count` | 上述潜在碰撞中，确实抽中该 slot plateau 的事件数。 |
| `gross_flip_01/10` | 重放恢复的当前 task 该 row **全部实际 flip 事件**，包括后来又翻回的 bit。 |
| `old_imprint_active_before/after`、`old_imprint_lost` | 旧印迹中，在当前新 task 前/后仍活跃的 bit 数，以及本 task 真正从 1 变 0 的旧印迹 bit 数。 |
| `expected_plateau_collisions_gated/no_I` | 固定同一 eligibility 和 cue 序列，分别按原 `f_eff` 和解析地去掉 `I` 后的 `f_eff/I` 对潜在碰撞求概率和。后者**不是**一次无 gating 的训练运行或实际观测。 |

## 1. 旧 task 的 slot 被修改多少？

每个旧 task 留下印迹的 row 数为 **3984–4024 / 4096**。换言之，原写入不是给每个 task 分配一小块互不重叠的专属 row，几乎所有 row 都承载过旧 task 的至少一个保留 LTP bit。在任一后续 task，**97.3%–98.1%** 的这类旧 row 至少有一个净 bit 变化；平均每个旧 row 约 **70–73 / 8192** 个 bit 净变化。整个矩阵每个新 task 的净变化约为 **0.85%–0.88%** 的 bit。**“多数 row 被碰过”不等于“多数旧 bit 丢失”。**

按旧印迹 bit 而非 row 计，后续单个 task 造成的活跃旧 bit 损失约 **4.5%–6.0%**（具体取决于旧/新 task 对）。Task1 末的 307,268 个印迹 bit 到 Task5 仍活跃 257,402 个，**累计净损失 16.23%**。其他起点到 Task5 的累计净损失：Task2 为 13.27%，Task3 为 9.46%，Task4 为 4.54%。这些是 bit 保留率，不是类别信息保留率；新写入可能在别处产生补偿，单凭此数不能判定旧类识别能力。

## 2. 主要是 0→1 还是 1→0？

| 新 task | 实际 gross 0→1 | 实际 gross 1→0 | gross LTD 占全部 flip | checkpoint 净 0→1 | checkpoint 净 1→0 |
|---:|---:|---:|---:|---:|---:|
| 2 | 279,988 | 30,781 | 9.90% | 267,651 | 18,444 |
| 3 | 272,846 | 41,732 | 13.27% | 260,748 | 29,634 |
| 4 | 267,950 | 50,872 | 15.96% | 255,624 | 38,546 |
| 5 | 262,229 | 61,511 | 19.00% | 248,791 | 48,073 |

**0→1 始终占主导**，推动矩阵占用率从 Task1 的 0.916% 增至 Task5 的 3.592%。但随着已有 1 bit 增多，1→0 事件占比持续上升；Task5 的 gross LTD 已有 61,511 次。gross 与净数不同，说明单 task 内有 bit 被多次翻转；不能仅用两张 checkpoint 估算全部 LTP/LTD 动力学。

## 3. 修改是否集中在 collision slot？

按照 `class_overlap` 将旧印迹 row 分为上、下四分位，并以该组写入前仍活跃的旧印迹 bit 数作分母，比较其本 task 损失率。10 个旧/新 task 对的“高 overlap / 低 overlap”损失率比值为 **1.00–1.26，中位数 1.13**。例如 Task1→Task2 为 **6.02% / 5.99%**，Task1→Task5 为 **5.08% / 4.56%**，Task4→Task5 为 **4.73% / 4.17%**。高 overlap 有轻到中等的损失富集，但**没有强烈集中在少数 collision row 的证据**。这也符合实现：plateau 对各 row 独立同概率抽取，只有当前 cue 的 active bit 决定该 row 中哪些突触可翻转。

`collision_count`、`plateau_collision_count`、`old_imprint_lost` 是三个不同层级：潜在 cue 重叠、实际 plateau 暴露、最终旧 bit 损失。不能将前两者直接等同为 1→0，更不能把任意活跃 row 直接称作某个旧类的专属 slot。

## 4. Collision-aware gating 是否真的减少旧 memory interference？

**机制与概率层面：是。结果因果层面：本轮不能证明。** 对原轨迹的每个样本，`I≤1` 确实降低所有 slot 的 plateau 概率。固定原 eligibility/cue 序列，解析地比较 `f_eff` 与 `f_eff/I`，Task1–5 的期望 plateau 总数分别减少 **16.3%、17.5%、17.9%、17.9%、18.6%**。针对 Task1 印迹与 Task5 cue 的潜在碰撞，原 gating 的期望 plateau-collision 数为 **13,478**，解析去除 `I` 则为 **16,614**（约少 18.9%）；原轨迹实际发生 **13,518** 次，接近原概率期望。

这些数字说明 **gating 确实执行并降低了预期碰撞写入机会**，但 `f_eff/I` 不是独立运行，不能推出“最终旧 bit 损失减少 18.9%”或“旧 memory 更稳定 18.9%”。而且 `I` 来自类别 trace 的最大 rival cosine，**并不读取具体 `W_B` row 的旧印迹**；它降低全部 row 的采样率，而不是针对 CSV 中高 `class_overlap` 的 row 额外抑制。现有单轨迹没有无 gating 的最终 `W_B` 对照，故对最终 interference 的因果改善标为 **NOT ESTABLISHED**。

## 5. LTD 是否承担稳定旧 memory 的作用？

**LTD 是旧 bit 丢失的直接机制之一，不是旧 bit 的保护操作。** 代码中的 XOR 在已活跃 bit 上产生 1→0，并没有旧/新类所有权判断。实测旧印迹每后续 task 损失 4.5%–6.0%，Task1 到 Task5 累计净损失 16.23%；同时高重叠 row 的旧 bit 损失率略高。这与论文第 5、16 页所述“遇到共享 bit 时通过 LTD 剪除重叠”的局部机制方向一致。

但“剪除共享 bit”与“保持旧记忆”不是同一可观测命题：前者可能减小跨类共占，后者要求旧类信息不被过度损坏。本轮既没有逐类 bit 所有权，也没有关闭 LTD 后的同轨迹终态对照；因此**不能仅凭本 CSV 证明 LTD 对旧 task 具有净稳定收益**。目前可确定的漂移来源是广泛的随机 plateau/活跃 cue 位 XOR 写入：LTP 主导总占用增长，LTD 则持续修剪已有 bit；gating 降低的是写入机会的期望数量，而不是锁定旧 slot。

审计脚本 `tmp/memory_audit_compute.py` 只读原缓存和 checkpoint、在内存中重放并断言逐 bit 一致；输出仅为上述 CSV。未更改正式模型或参数。
