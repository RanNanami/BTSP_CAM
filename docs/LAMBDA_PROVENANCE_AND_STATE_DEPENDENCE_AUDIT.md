# Eligibility trace λ 的来源、语义与功能审计

## 1. Executive conclusion

**最终分类：C — `PAPER_STATE_DEPENDENCE_UNSPECIFIED`。** [BTSP-CAM 原文](../../2233_BTSP_CAM_A_Brain_Inspired.pdf)第 5 页 Eq.(4) 明确给出逐类、逐样本衰减的递推和 Algorithm 1 的执行顺序，却只把 `λ_{c,t}` 称作“state-dependent decay”，没有公开函数、数值、状态变量或计算步骤。本地 `trace_decay=0.01` 是 **LOCAL IMPLEMENTATION / LOCAL_CHOICE**，不是作者参数。当前代码的“先对所有类别 decay、再给当前类加 `ηs`、再算 F/N/I”顺序符合 Eq.(4) 与 Algorithm 1；缺口是 λ 的值/状态规则，而不是已确认的更新顺序错误。simpleBTSP 没有可直接继承的类别 eligibility λ。

本轮没有更改 `src/`、正式 config、正式 checkpoint 或原有实验文件。所有新增规则仅在独立的 [实验脚本](../experiments/run_lambda_state_dependence.py)与[预声明配置](../experiments/configs/lambda_state_dependence.yaml)中运行，使用固定 ViT cache、CIFAR100 B0-Inc5、class-order seed 1993、相同 sparse/RNG 流，`k=128`、`z=a`、`τ=0` 和 experimental coordinate-aligned ridge。它们是 **DIAGNOSTIC COUNTERFACTUAL**，不是 strict paper reproduction，也不依据论文准确率选择 λ。CSV 中准确率是 0–1 比例，密度亦为 0–1 比例。

## 2. Paper evidence

以下页码是 PDF 页序和印刷页码。针对全文逐页检索 `lambda/λ/decay/eligibility/state-dependent/class-dependent/forgetting/trace`，并检查 Algorithm 1、Appendix、术语表及 ablation；短引文用于定位，不以实验结果解释作者意图。

| 页 / 位置 | 原文短引文或公式 | λ 的数学角色 | 数值 / 函数 / 状态依赖 |
|---|---|---|---|
| 5，§4.2 Eq.(4) | `e_{c,t}=(1−λ_{c,t})e_{c,t−1}+η·I(c=y_t)s_t` | 每个类别旧 trace 的保留因子 `1−λ_{c,t}`。 | 未给数值；未给计算函数。 |
| 5，Eq.(4) 后 | “`λ_{c,t}` defines a state-dependent decay” | 声称状态依赖。 | **STATE-DEPENDENT RULE NOT SPECIFIED**；未列状态变量或函数。 |
| 5，Eq.(4) 后 | “accumulate evidence for the current class while slowly forgetting inactive ones” | 定性说明当前类累积、非当前类遗忘。 | 未定义“slowly”的时间单位/衰减率。 |
| 5，Algorithm 1 行 2、6–8 | hyperparameters 含 `{λ_{c,t}}`；先 “Update traces ... via Eq. (4)”，再求 `f_q^eff`。 | 把 λ 放在当前样本的 gating 前。 | 未提供任何 λ 更新伪代码。 |
| 12，Appendix B symbol glossary | eligibility trace `e_c` 被解释为类别相关的 recent-input 表示。 | 补充对象语义。 | 没有补 λ 值/规则。 |
| 7，§6.1 Table 2 | “Effect of trace-gated plasticity” | 比较是否使用 trace gating。 | 不是 λ 数值、固定/状态依赖规则的 ablation。 |

全文及 appendix 中没有发现 λ 对 `y_t`、trace norm、F、N、I、task index、class age、sample count、物理时间、plateau 或 `W_B` 状态的**明确定义依赖**。下标 `c,t` 允许对类别和流步不同，并非已给出具体函数；不能从下标推出作者实际用了按类别年龄的规则。`f_q^eff` 的 Eq.(5) 确实依赖 F/N/I，但**不能反推** λ 也依赖这些量。

## 3. Eq.(4) semantics

对每个流样本 `(s_t,y_t)`，Eq.(4) 的同步旧态解释为

```text
for each class c:
    e_c,new = (1 - λ_c,t) e_c,old + η · 1[c=y_t] · s_t
```

因此旧项对**所有**类别每样本都乘保留系数（若某类别的 λ 恰为 0，则其数值不变）；当前类**先 decay 后 add**；非当前类只有 decay。`λ_{c,t}` 是作用于类别向量 `e_c` 的标量系数，既有 `c` 又有 `t` 下标；论文没有把它写成 per-bit vector，也没有说它必是所有类相同的全局常数。它可以数学上退化成全局固定数，但“state-dependent”文字不支持把固定 0.01 冒充完整指定。Eq.(4) 没有显式给 `λ∈[0,1]` 的约束；这个范围是本地为了非负保留因子而作的实现约束，不应归为论文事实。

## 4. Algorithm 1 ordering

| 顺序 | 论文第 5 页行号 | 本地对应 | 判断 |
|---|---:|---|---|
| 取 `(x,y_t)` 并生成 `s` | 4–5 | `model.py` 样本循环 / encoder | 同序。 |
| 用 Eq.(4) 更新全部类别 trace | 6 | `model.py:_learn_code` 调 `traces.update`；`eligibility.py` 先全体乘 `1−decay`，再当前类加 `ηs` | **同序**；但固定 λ 是本地补全。 |
| 从**更新后**的 trace 算 F/N/I、`f_q^eff` | 7 | `traces.signals` 后 `plateau_probability` | **同序**。 |
| 采 plateau、采 synapse Bernoulli、XOR 写 `W_B` | 8–11 | `sample_slot_plateaus`、`memory.update` | 同序。 |
| 任务写完后构建 CA1、ridge | 14–15 | formal `learn_features` Phase 2 | 同序；本轮诊断 readout 另用 coordinate-aligned τ=0。 |

没有证据支持把 F/N/I 改为更新前 trace。本地若沿用固定 `λ=0.01`，与论文“state-dependent”描述**规格不完整**，但**不是**已证实的 `IMPLEMENTATION SEMANTICS MISMATCH`（顺序和代数递推未错）。

## 5. simpleBTSP inheritance audit

| 来源 | 核对结果 | 与 `λ_{c,t}` 的关系 |
|---|---|---|
| [simpleBTSP 正文](../../s41467-024-55563-6.pdf)第 3–4 页 | 二值突触、plateau 前后约 −6 至 +4 秒的塑性窗、概率性 LTP/LTD；按输入和 plateau 修改权重。 | 没有类别 `e_c` 或 Eq.(4) 式每样本 decay。 |
| 正文第 16 页 Methods | 引用另一连续权重 BTSP 模型的 `τ_ET=1000 ms`、`τ_IS=500 ms`。 | 是生物塑性仿真时间常数，**不是**类别 eligibility `λ`。 |
| [simpleBTSP supplement](../../supplement.pdf)第 2 页及其他相关段 | plateau 的时间窗/二值模型补充；第 8 页的“weight decay processes”是 LTD 概念性讨论。 | 没有 `λ_{c,t}` 的公式或数值。 |
| [本地 upstream simpleBTSP](../upstream/simpleBTSP) | 搜索 `eligib/lambda/decay/trace`；Python 语法 `lambda` 和 Poisson 均值变量 `lambdas` 与此无关。 | 没有类别 trace 更新可映射。 |

**NO DIRECT SIMPLEBTSP LAMBDA INHERITANCE FOUND.** 不把 BTSP 时窗、plateau probability、input sparsity、firing threshold 或 Poisson `lambdas` 混为 eligibility λ。

## 6. Local implementation audit

| 对象 | 代码事实 | 归类 |
|---|---|---|
| `trace_decay=0.01` | [formal YAML](../configs/btsp_cam_cifar100.yaml)第 60–63 行明确标 `LOCAL_CHOICE`，来源是本地 smoke configuration。 | **LOCAL IMPLEMENTATION**，不是论文数值。 |
| `ClassEligibilityTraces.decay` | [eligibility.py](../src/btsp_cam/eligibility.py)第 25–59 行接收一个固定标量或固定 per-class 数组，broadcast 后不随样本状态更新；范围检验 `[0,1]`。 | Eq.(4) 的一个特例；state-dependent rule 未实现。 |
| 训练顺序 | [model.py](../src/btsp_cam/model.py)第 140–166 行先 update，再 signals，再 plateau，再 XOR；所有类别随每样本衰减。 | **PAPER SPECIFIED** 顺序的本地实现。 |
| `η=1` | formal YAML 第 56–59 行标 `DERIVED`；`eligibility.py` 当前类加 `ηs`。 | 数值不是论文指定；在下述条件下是纯尺度 gauge。 |
| `W_B` 写入 | [binary_memory.py](../src/btsp_cam/binary_memory.py)第 39–69 行保留原 0↔1 XOR writer；本轮未改。 | 与 λ 审计相邻但不作为 λ 来源。 |

## 7. eta/lambda scale analysis

设 `e_{c,0}=0`、所有类别使用同一个 `η>0`。若固定一条 `λ_{c,t}` 序列，则递推展开为

```text
e_c,t(η) = η Σ_{j≤t, y_j=c} [ Π_{r=j+1}^t (1−λ_c,r) ] s_j.
```

于是对每个类别和时间 `e_c,t(η)=ηe_c,t(1)`，非零 trace 的 `cos(s,e_c)` 不变；零 trace 按本地定义 cosine=0，仍不变。**类别历史长度不同并不破坏该结论**。即使 λ 随类别和时间变化，只要所用 λ 序列在 η 改变时保持相同，仍完全约掉。如果 λ 是仅由 cosine/F/N/I、类别计数、年龄等**尺度不变状态**确定的函数，则归纳可得不同 η 下 λ 序列仍相同，也继续约掉。反之，若 λ 依赖 `||e||`、某 bit 的绝对幅值、η 本身等尺度敏感量，则不同 η 可改变 λ 序列，η 不再保证可约。论文没有给 λ 函数，故不能对作者真实状态依赖作无条件尺度无关断言。有限精度下极小 trace 还可能下溢，属数值例外。

## 8. F/N/I sensitivity

令更新前当前类 trace 为 `e`、本次 `s` 的 norm² 为 `k`，当前类保留系数 `α=1−λ_y,t`。更新后

```text
F(α) = cos(s, αe + ηs).
```

在 `η>0`、`α≥0`、非负 sparse trace 的本地设置中，`dF/dα = αη[(s·e)²−||s||²||e||²] / (||s||·||αe+ηs||³) ≤ 0`（常数因子写法以 η 的缩放参数化为准）。故**本次当前类更强 decay 通常提高、而非降低 F**，因为新的 `s` 权重相对旧历史增大；完全平行时不变。`N=1−F` 因而通常下降。对于本次非当前类，纯粹乘正标量不会改变 cosine，所以只改变其 λ 不会立即改变 `I=exp(−β max rival cosine)`；乘成零或数值下溢是例外。后续流中不同历史分量权重被重新混合，rival trace 的**方向**可变，`I` 可以升也可以降。

`G_FN=4N(1−N)F=4F²(1−F)` 在 `F=2/3` 最大，因而 F 上升对 `G_FN` 的作用先正后负；`f_eff=f_base·G_FN·I` 还受 I 影响。**最终 write probability 对 λ 为 NON-MONOTONIC。** 本地 formal k=41 checkpoint 的只读核查：task1/5/10/20 的已见类 trace 平均 norm 分别约 `42.28/8.13/4.62/2.04`；Task5/10/20 旧类均值约 `1.26×10⁻¹⁰/5.42×10⁻¹¹/2.55×10⁻¹¹`。这反映本地固定 0.01 连续衰减：在新 task 2,500 个样本后，旧 trace 保留幅度约 `0.99^2500≈1.2×10⁻¹¹`；**不能据此判断论文作者的 λ**。

用未改动的 k41 checkpoint 和每类测试码作[逐类别只读快照](../diagnostics/lambda_formal_k41_trace_snapshot.csv)（脚本见[此处](../experiments/audit_formal_lambda_traces.py)）：Task5 旧类（20 类）平均 trace age 约 6,255 个样本、norm 中位数 `2.71×10⁻²¹`，自己的类别 cosine 均值 `.1770`、最大 rival cosine 均值 `.1243`；Task10 旧类（45 类）为 `.1794/.1352`。Task20 的 95 个旧类中有 **25 个 trace 的直接 L2 norm 在 float64 下为零**，其 own-class cosine 被本地 zero-denominator 约定置零；其余 70 类为 `.1870/.1463`。因此 Task20 对全部旧类直接平均的 cosine 不能解读为真实类别几何：它同时受到固定 .01 长程衰减与数值下溢影响。这是本地实现/精度现象，**不是论文 λ 的证据**。下方 k128 CSV 还逐 task 记录旧/新类 trace norm、非零密度与样本年龄，以及来自训练流的当前类 F / 最大 rival cosine 所构成的 I。

## 9. Diagnostic lambda families, Task1–10

七个规则在[预声明 YAML](../experiments/configs/lambda_state_dependence.yaml)中固定：`L0=(0,0)`、`L_FIXED_SMALL=(.001,.001)`、`L_FIXED_01=(.01,.01)`、`L_FIXED_05=(.05,.05)`、`L_FIXED_10=(.10,.10)`、`L_CURRENT_PROTECTED=(0,.01)`、`L_CURRENT_FAST=(.01,.05)`；括号依次是当前类/非当前类的 λ。没有 `L_AGE_DEPENDENT`：论文没有把“state-dependent”明确与 class age 连接，依请求不发明 age 规则。相同 encoder、数据顺序和独立 writer RNG seed；只改变 λ。基线 `L_FIXED_01` 的 Task1–10 `W_B` 与已存 k128 checkpoint **逐 bit 一致**，逐 task 准确率与既有 k-path CSV 相等。

为使全部 family 使用逐 bit 相同的 sparse 输入，诊断 runner 在单次进程内预编码并暂存训练/测试 sparse codes；这是实验工程缓存，**不主张该 runner 自身满足严格 exemplar-free 存储边界**。每个 task 的 coordinate-aligned `C/D` 更新只读取本 task 训练码，后续 ridge 对齐只读取聚合 `C/D` 和当前 `W_B`，没有 replay 旧训练码。测试码仅用于评估/geometry。

为减少重复 cosine 计算，独立 runner 缓存 trace 平方范数。初次 full20 校验在 Task16 发现该缓存反复乘衰减系数后进入次正规数/下溢，导致其 `W_B` 与原 checkpoint 不一致；**这批未通过的结果未进入最终表**。runner 随后仅对极小已见类 trace 从原始向量直接重算范数，保持与 formal `np.linalg.norm` 语义一致，再从 Task1 重新运行。最终 fixed-.01 基线每 task 的 `W_B` 逐 bit 校验和既有 k-path 准确率校验是 full20 结果的前提。没有改 formal trace 类或任何 checkpoint。

最终 CSV 的 task-end trace 描述字段又通过[独立只读 eligibility 重放](../experiments/recompute_lambda_trace_descriptives.py)从 **trace 向量本身**重算（`trace_norm_source=direct_vector_float64`）；加速缓存的平方范数不再用于这些字段。该重放对准确率、plateau、gross flips、W_B 密度等所有非 trace 字段做逐行不变校验。两个 CSV 的唯一键、行数与 k128 前 10 task 的 paired equality 均已核对。

| 规则 | Task1–10 平均累计 acc | Task10 累计 acc | Task10 F | Task10 I | Task10 `G_FN` | Task10 gross flips | Task10 `W_B` 密度 |
|---|---:|---:|---:|---:|---:|---:|---:|
| L0 | 90.184% | 86.48% | .4089 | .7171 | .3834 | 906,312 | 12.867% |
| L_FIXED_SMALL | 90.189% | 86.32% | .4146 | .7178 | .3901 | 922,646 | 13.044% |
| L_FIXED_01 | 90.065% | 86.42% | .4751 | .7318 | .4584 | 1,112,277 | 14.779% |
| L_FIXED_05 | 90.033% | 86.44% | .6675 | .7937 | .5633 | 1,486,517 | 17.419% |
| L_FIXED_10 | 89.963% | 86.18% | .7879 | .8322 | .4769 | 1,316,492 | 16.392% |
| L_CURRENT_PROTECTED | 90.103% | 86.02% | .4619 | .7287 | .4441 | 1,072,144 | 14.435% |
| L_CURRENT_FAST | 89.897% | 86.44% | .6356 | .7882 | .5589 | 1,466,782 | 17.248% |

完整逐 task 的 trace norm/density/age、F/N/I/`G_FN`/`f_eff`、plateau、gross flip、`W_B` 密度、sparse 与 `a` 的全体/旧/当前类 cosine 和 margin、四种准确率见 [Task10 CSV](../diagnostics/lambda_family_task10.csv)。Task10 的 `L_CURRENT_FAST` 相比固定 .01：`F +.1605`、该任务 gross flips `+31.9%`、`W_B` density `+2.47` 个百分点，已满足实验前声明的机制触发阈值（F绝对变化≥.05，或 gross flips相对变化≥20%，或 density绝对变化≥.02）。这只授权一个预先固定的跨 k full20 **机制验证**，不构成 accuracy 驱动的选型。Task10 累计准确率只差 `+0.02` 个百分点；不同 λ 在这一端点的分类效果很小，但写入动力学显著不同。

## 10. Full20 result if justified

跨 k 的完整运行按上述机制触发条件进行，使用同一个预声明 `L_CURRENT_FAST`，与 `L_FIXED_01` 分别在 k128/k160 成对比较。结果见 [full20 CSV](../diagnostics/lambda_family_full20.csv)。

| k | 规则 | 20-task 平均累计 acc | Task20 累计 acc | Task20 旧类 acc | 当前类 acc | 当前 5-way | Task20 W_B 密度 | 20-task gross flips |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 128 | L_FIXED_01 | 86.487% | 80.62% | 80.863% | 76.0% | 92.2% | 20.614% | 21,311,700 |
| 128 | L_CURRENT_FAST | 86.365% | 80.69% | 80.937% | 76.0% | 90.8% | 23.543% | 28,850,575 |
| 160 | L_FIXED_01 | 86.901% | 81.16% | 81.305% | 78.4% | 92.6% | 23.504% | 26,985,399 |
| 160 | L_CURRENT_FAST | 86.896% | 81.47% | 81.695% | 77.2% | 92.4% | 26.273% | 35,600,578 |

两个 fixed-.01 基线的 **每 task `W_B` 与各自已存轨迹逐 bit 一致**，逐 task cumulative accuracy 与既有 k-path 结果一致。k128 `L_CURRENT_FAST` 的前 10 task 与重新计算后的 Task10 CSV 在 accuracy 和 gross flips 上逐 task 相等。完整 CSV 每个 k×规则各 20 行，无缺 task/重复键。统计口径：平均累计 acc 是 20 个 task 的 cumulative accuracy 算术均值；Task20 旧类为 95 类、当前类为 5 类。

数值边界：Task20 `L_CURRENT_FAST` 的旧类 trace **直接范数中位数为 0**（k128、k160 均如此），固定 .01 对照中位数约 `6.92×10⁻¹⁰⁸ / 7.91×10⁻¹⁰⁸`。在当前 float64 实现下，部分早期旧类的 cosine 已按 zero-trace 约定置零；因此 `I` 上升包含 trace 数值消失的效应，不能单纯解读为生物学上更好的 interference 抑制，更不能据此推断作者的 state rule。

固定 .01 → 预声明状态规则的 Task20 差值：k128 **+0.07pp**，k160 **+0.31pp**；20-task 平均分别 **−0.122pp**、**−0.006pp**。相反，20-task gross flips 分别增加约 **35.4% / 31.9%**，Task20 W_B 密度提高 **2.93pp / 2.77pp**。Task20 `a` 全类别 cosine margin 在 k128 为 `.001438→.001206`、k160 为 `.001092→.000937`；更密集的写入没有提高这个几何指标。`L_CURRENT_FAST` 在 k128 Task20 的 F/I/`G_FN` 为 `.621/.788/.556`，基线 `.448/.724/.432`，对应平均 plateau/sample `8.94` vs `6.33`。k160 也显示相同方向的机制变化。旧类准确率小幅增加，当前类准确率持平或下降，因此最终 +0.07/+0.31pp 不应解释为全面表征改善。两个 k 的机制效应都大，分类效应都小；没有证据支持“强烈 k 依赖的 λ 分类收益”。

本轮**未用 test accuracy 选 λ**：full20 只运行预先固定的同一个状态规则；这些 test 结果是诊断性、单 class-order/单随机流比较，不能作为最终 λ 参数选择。Task1–10 的七族分类范围也很窄（Task10 约 86.02–86.48%），而 F、plateau、gross flips、occupancy 的差异显著。因此应降低“当前简单 λ 规则缺口解释剩余 accuracy gap”的优先级；但论文未公开的真正 state function 仍是未决项，不能由这组有限反事实排除。

## 11. Strict reproduction implications

`λ_{c,t}` 的**角色和 Eq.(4) 顺序**可严格按论文实现；其**状态函数/数值**不能由公开 PDF + simpleBTSP 正文、supplement、upstream 唯一恢复。当前 formal 固定 .01 是为可运行实验作的本地补全。即便某一实验规则改善 accuracy，也只能称 **PLAUSIBLE MISSING IMPLEMENTATION DETAIL**；不能称 `AUTHOR RULE RECOVERED`，也不能用与论文数字接近作为 provenance 证据。coordinate-aligned、τ=0 readout 亦为实验 repair，不是作者实现。

## 12. Final classification

**C — PAPER_STATE_DEPENDENCE_UNSPECIFIED。** `A` 不成立：函数和值均未完整给出；`B` 不成立：连 λ 数值也未给；`D` 不成立：当前更新顺序符合公式与算法，固定 λ 仅是未指定规则的本地特例；`E` 不适用：公开文本足以判断该具体缺口。

**当前 λ 实现是否可以被称为作者参数？NO。**
