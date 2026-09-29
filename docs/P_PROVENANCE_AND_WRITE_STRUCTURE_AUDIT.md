# BTSP-CAM 的 XOR 概率 p：来源与写入结构审计

## 范围与结论

**P_VALUE_NOT_SPECIFIED。** BTSP-CAM 给出了 p 的定义、概率范围和理论条件，但在已提供的全文（主文、表格、Appendix B/D/E/F）未公开具体数值。当前 formal YAML 的 p=0.5 是受 simpleBTSP 启发的**本地选择**，不能称为作者实验参数。这里所有运行均是独立诊断：CIFAR100 B0-Inc5、class-order seed 1993、同一 cached frozen ViT、同一 RF encoder、k=128（条件核查 k=160）、固定 λ=0.01、τ=0、experimental coordinate-aligned ridge；没有编辑 formal config、模型源码或正式 checkpoint。对照只改变 p，以及 equal-budget 组由公式预先决定的 f_base。

Task1–10 的判定是 **P_CHANGES_MEMORY_STRUCTURE, BUT SMALL CLASSIFICATION EFFECT**：固定 f_base 时 p 强烈控制 flip 数与 memory density；保持 f_base·p 近似不变后，这些总量和热点几乎对齐，但 a 的余弦 margin 仍有系统差异。准确率差则远小于结构差，不支持把 p 单独列为剩余论文准确率差距的主要来源。等预算并非严格匹配每个 slot/样本的写入位置。

## 1. 来源证据与不能混同的 p

页码为 PDF 印刷页码。本轮对 [BTSP-CAM PDF](../../2233_BTSP_CAM_A_Brain_Inspired.pdf) 逐页检索 p/probability/Bernoulli/synapse/LTP/LTD/XOR/hyperparameter 并目检第 5、14–15 页；对 [simpleBTSP PDF](../../s41467-024-55563-6.pdf) 的正文与 Methods、[upstream 代码](../upstream/simpleBTSP/main_BTSP_feedforward_model.py)作对照。

| 来源 | 短证据/定义 | 判定 |
|---|---|---|
| BTSP-CAM 第 4 页 Eq.(2) | active presynaptic bit、plateau 后，0→1 与 1→0 各以 p 发生。 | **PAPER_SPECIFIED** 机制；未指定数值。 |
| 第 5 页 §4.3 Eq.(6)，Algorithm 1 行 8–11 | `u_ji ~ Bernoulli(p)`，对每个已选 slot j 和 active input i 独立采样，`w_ji ← w_ji XOR u_ji`。 | **PAPER_SPECIFIED**：p 是条件于 plateau 的逐突触翻转概率，不是 plateau 概率。 |
| 第 12 页 Appendix B 术语表；第 14 页 Eq.(18) | p 为 “Probability of synaptic bit-flip”；`p∈(0,1]`，未翻转概率 1−p。 | **PAPER_SPECIFIED** 语义/范围；**P_VALUE_NOT_SPECIFIED**。 |
| 第 14–15 页 Appendix D.1 | 新的全零 slot 单次 plateau 中 `K_active~Binomial(k,p)`；`p>τ` 且 k 增大时越过检索阈值的概率趋近 1。 | **DERIVED** 的充分条件，非 p=0.5 的出处；只针对全零 slot 的单次写入，不保证持续重写下的类别保持。 |
| 第 15–16 页 Appendix D.2/E | 对反复碰撞的同一 synapse，转移矩阵非对角元为 p；旧 1 经一次相关更新以 p 变 0。 | **PAPER_SPECIFIED** 的简化理论；没有给实验用 p。 |
| 第 17–18 页 Appendix F | 讨论阈值、prototype drift 与 ridge margin。 | **NOT_FOUND**：没有补充 p 的数值或选择过程。 |
| 第 8 页 partial-cue 实验的 `p=0.6` | 60% input bits 被 drop。 | 是遮蔽比例，**不是** Eq.(6) 的 p。 |
| simpleBTSP 第 3–4 页、16 页 Methods | 简化塑性规则中的 0.5 来自把 plateau 前后约 10 秒窗口拆成等长 LTP/LTD 时间段，并假定输入与 plateau 相对时刻独立。 | simpleBTSP 的**建模近似**，不是 BTSP-CAM 的参数声明，也非直接测得的逐突触概率。 |
| simpleBTSP upstream 第 34、102、114–126 行 | 变量 `p=0.005` 是 input-bit density；代码用互斥的 slot-level LTP/LTD plateau 事件，各约 `f_q/2`，不是 BTSP-CAM 的 per-(slot,bit) Bernoulli(p) 抽样。 | 同名 `p` **绝不可继承**；0.5 仅可作为概念启发。 |
| formal YAML `flip_probability=0.5` | 标记 `PLAUSIBLE_INHERITANCE`。 | 在严格四分类中应归 **LOCAL_CHOICE (simpleBTSP-motivated)**，不是 PAPER_SPECIFIED / DERIVED / 已证实的逐位 SIMPLEBTSP_INHERITED。 |

Eq.(18) 给出相同 p 的 LTP/LTD 转移；Proposition 4.1 单写 `p_LTD>0` 是条件，不提供一个独立已公开的 LTD 数值。Appendix D.1 中的 τ 指 CA1 normalized overlap retrieval threshold；`p>τ` 是概率集中条件，**不能**将 τ 当成 f_base、也不能从该不等式算出作者的 p。文中 Eq.(19) 的“≥τk”与 `1−F(floor(τk))` 在整数边界有一位差异；本审计只使用无歧义的渐近结论，不用它决定数值。

## 2. 数学作用与实验设定

给定一个 plateau slot 和 k 个 active bits，独立 `u_i~Bernoulli(p)`，故 `E[gross flips | plateau]=kp`，k=128 时分别为 12.8、32、64、96、128。每次真实 flip 的方向由旧值决定：0→1 是 LTP，1→0 是 LTD；因此 `E[net added]` 不能仅由 kp 决定。若同一 synapse 从 0 开始有 n 次被选中机会，它最终为 1 的概率是 `[1−(1−2p)^n]/2`；相同预期 flip 总数但不同 p、不同 plateau 次数，仍可能有不同 slot 分布、奇偶历史和几何。

`f_base` 乘 Eq.(5) 的 F/N/I 后决定 **slot plateau 频率**；p 决定 **已选 slot 内每个 active bit 的 flip 频率**。固定 f_base 时，p 不改变 F/N/I 或 plateau 抽样（本地 trace/gating 不读 W_B），所以 plateau count 完全一致。equal-budget 组按 `f_base(p)=0.005·0.5/p` 预设，只维持一阶 `f_base·p=0.0025`；实际 gross flips 必须实测，不能假定精确相等。相同 encoder、训练顺序以及从同一 seed 重启的 plateau/synapse RNG；不同 plateau 个数后两组随机消耗路径自然分叉。

实验在[独立预声明配置](../experiments/configs/p_write_structure.yaml)和[runner](../experiments/run_p_write_structure.py)中。基线 k=128、p=.5 的 Task1–10 W_B 与历史 checkpoint **逐 bit 相同**，逐 task accuracy 与历史 k-path CSV 相同；实验性 trace 加速器仅在极小范数时直接回退原向量范数以保持该等价。runner 为配对效率预编码并在进程内保留 sparse codes；**runner 本身不主张满足严格 exemplar-free 存储边界**，但每个 task 的 aligned C/D 更新只读取当前训练码，不 replay 旧训练码。测试码只用于评估与固定 Task1 probe。

“destructive ratio”在 CSV 中按请求定义为 `flips_10/gross_flips`；这是 **LTD 比例**，不能凭名字推断这些 LTD 都损害旧类。逐事件 Task1 probe gross functional movement 将每个真实翻转对固定 500 个 Task1 测试码的 `a` 绝对变化求和，允许同一突触多次翻转的贡献累加；task-end net movement 用前后两个 W_B 的 `mean|Δa|`。两者之差体现路径内抵消。Task1 probe 在 Task1 写入期间不是“旧类”，旧 query 解释从 Task2 开始。

## 3. Task1–10：固定 f_base

[完整逐 task 数据](../diagnostics/p_fixed_fbase_task10.csv)。以下为 Task10，accuracy/density 按百分数，margin 为同类与异类平均余弦之差。

| p | plateau | gross flips | LTD/gross | W_B density | top1% column share | a margin | cumulative acc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| .10 | 17,391 | 222,273 | 18.44% | 5.047% | 8.003% | .006537 | 86.24% |
| .25 | 17,391 | 555,880 | 29.55% | 9.803% | 5.022% | .003737 | 86.60% |
| .50 | 17,391 | 1,112,277 | 37.19% | 14.779% | 3.427% | .002581 | 86.42% |
| .75 | 17,391 | 1,668,250 | 40.80% | 18.117% | 2.822% | .002109 | 86.34% |
| 1.00 | 17,391 | 2,226,048 | 42.87% | 20.599% | 2.481% | .001829 | 86.14% |

在 p=.5，实测 `gross/plateau=63.96`，与 `kp=64` 一致；p=1 时恰为 128。p 从 .1 到 1 使 gross flips 约 10 倍，Task10 密度提高 15.55 个百分点，LTD 占比上升约 24.4 个百分点，a margin 下降约 72%；但五个 accuracy 只落在 86.14–86.60% 的 0.46pp 区间，**不是**单调的“越写越好/越坏”。top1% share 的下降是随整体 occupancy 上升的相对集中度变化，不代表热点的绝对写入减少。全部 p 的 Task10 F/N/I 与 f_eff 完全相同，符合独立门控路径。

## 4. Task1–10：equal-budget

[完整逐 task 数据](../diagnostics/p_equal_budget_task10.csv)。p=.5 行为独立重跑并与 fixed-f_base 行逐值一致，不是事后复制。

| p | f_base | plateau | gross flips | LTD/gross | W_B density | top1% column share | a margin | cumulative acc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| .10 | .025000 | 85,552 | 1,096,775 | 37.19% | 14.732% | 3.434% | .001804 | 85.92% |
| .25 | .010000 | 34,404 | 1,101,331 | 37.22% | 14.758% | 3.427% | .002087 | 86.26% |
| .50 | .005000 | 17,391 | 1,112,277 | 37.19% | 14.779% | 3.427% | .002581 | 86.42% |
| .75 | .003333 | 11,688 | 1,122,041 | 37.20% | 14.771% | 3.433% | .003104 | 86.62% |
| 1.00 | .002500 | 8,766 | 1,122,048 | 37.25% | 14.762% | 3.429% | .003630 | 86.60% |

总 flip 数相对 baseline 落在约 −1.39% 至 +0.88%；Task10 density 最大差仅约 0.047 个百分点，top1% column share 最大差约 0.007 个百分点，LTD 占比几乎相同。然而 a margin 在 p=.1 与 p=1 之间约翻倍，p=1 相对 baseline 高约 40.6%；旧类 margin 也有同向变化。这是**写入结构**留下的诊断证据，可能与“更多 slot 各翻少量 bits”对比“更少 slot 各翻全部 bits”的分配与 XOR parity 有关；数据没有分离每个微观机制的因果比例。

准确率的效应较弱且非单一方向：Task10 p=.1 比 baseline 低 0.50pp，p=1 高 0.18pp；Task1–10 平均累计准确率分别为 89.48%、90.65%、baseline 90.06%。固定 p=1 比 baseline 增加的几何 margin 并没有转化为同等级的分类增益。

在 Task10，fixed-f_base p=.1→1 的逐事件 Task1-probe gross functional movement 从 .02029 增至 .20206；equal-budget 则约 .09967–.10244，与 baseline .10107 接近。对应 task-end `mean|Δa|` 在 equal-budget 约 .02016–.02099，说明路径内抵消明显且净漂移差异小。p=1 相对 baseline 的 gross movement 约 +1.35%、net `mean|Δa|` 约 −3.35%，均不到 15% 的预声明漂移门槛。它也未显示出大幅新增的 LTD 比例；不能把 margin 变化直接称作 catastrophic overwrite。

## 5. 条件 full20 与 k=160 核查

预声明门槛是 Task10 cumulative accuracy 差 ≥0.5pp、a margin 差 ≥15%、equal-budget density 差 ≥2pp 或旧 query functional drift 差 ≥15%。equal-budget p=.1 的 Task10 差约 −0.50pp，p=1 的 a margin 差 +40.6%，故触发 full20。full20 的 p=1 在**看 test 结果前**已在 YAML 中选为最大 XOR 概率的结构对照，不因其 accuracy 更高而被选中。p=.5 full20 使用此前已逐 bit 验证的 λ 固定 .01 / k-path baseline，而不是不必要地重复训练；[汇总 CSV](../diagnostics/p_full20.csv)中的 `source` 明确区分归档基线与新运行，归档没有的逐 flip 字段留空，不以零填充。

| k | p / equal-budget f_base | Task20 cumulative | Task20 old | Task20 current | full20 avg incremental | Task20 a margin | Task20 density |
|---:|---|---:|---:|---:|---:|---:|---:|
| 128 | .5 / .005 | 80.62% | 80.86% | 76.0% | 86.49% | .001438 | 20.614% |
| 128 | 1 / .0025 | 80.74% | 80.97% | 76.4% | 86.86% | .001716 | 20.602% |
| 160 | .5 / .005 | 81.16% | 81.31% | 78.4% | 86.90% | .001092 | 23.504% |
| 160 | 1 / .0025 | 81.21% | 81.52% | 75.4% | 87.20% | .001288 | 23.487% |

k=128 Task20 accuracy 只差 +0.12pp，但 a margin +19.4%，满足预声明的机制显著性门槛，因此按原计划检查**同一对 p** 在 k=160 的表现，不重新选择对照值。k=160 同样显示几何 margin +18.0%，而 Task20 accuracy 仅 +0.05pp，WB density −0.018 个百分点；即几何作用跨 k 出现，但**几乎未转化为最终 100-class 性能**。k=160 的 Task20 old accuracy 高约 0.21pp、current accuracy 低 3.0pp，表明对当前类的影响方向也未稳定，不能仅看 cumulative +0.05pp 声称整体改进。两种 k 的 Task1–20 平均累计 accuracy 分别约 +0.37pp、+0.30pp，仍远小于先前由 k/坐标对齐造成的数个百分点差异。

## 6. 解释边界与优先级

1. **论文是否公开 p？** 没有；`P_VALUE_NOT_SPECIFIED`。p=.5 有 simpleBTSP 的建模启发，但本地逐位 p=.5 不是已证实作者参数。
2. **p 控制 quantity 还是 structure？** 两者都有。固定 f_base 下 quantity 主导密度和翻转；等预算消去总量后仍有几何结构差异，故不能写成 `P_EFFECT_MAINLY_WRITE_BUDGET` 的充分结论。
3. **是否明显影响 catastrophic overwrite？** 当前等预算旧 probe 净漂移、LTD 比例及整体 occupancy 没有相称的巨大变化，未见 p=1 导致明显更强的灾难性覆盖；结构 margin 与分类需要分别报告。
4. **是否明显影响 classification？** Task10 等预算极值跨度约 0.70pp，p=1 相对 .5 +0.18pp；k=128 Task20 +0.12pp。单次 seed/固定 class order 下不能将微小差异视为稳健可迁移增益。
5. **strict reproduction 优先级？** p 仍是未公开参数，但 k=128/160 的等预算 full20 分别只有 +0.12/+0.05pp 最终差异，不支持其单独解释剩余多百分点差距；应降低 p 的优先级，下一项查 γ，而不是据论文 accuracy 反选 p。
