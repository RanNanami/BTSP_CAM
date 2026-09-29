# Sparse collision 到 BTSP 写入干扰：Task1–5 诊断

本轮只读取正式 ViT 缓存、固定 `R,b`、原 checkpoint 和配置；在内存中对原随机数流做确定性 replay。**没有修改模型、参数、训练数据、正式 W_B 或 checkpoint，也没有重新训练 W_B。** 输出表位于 `diagnostics/`，脚本位于 `tmp/collision_to_interference_a.py`、`tmp/collision_to_interference_b.py`。论文事实依据为 [BTSP-CAM 原文](../../2233_BTSP_CAM_A_Brain_Inspired.pdf)第 4–5 页 Eq.(3)–(6)、第 13–14 页 Appendix C；已有精确碰撞普查见 [前轮审计](APPENDIX_C_COLLISION_ASSUMPTION_TEST.md)。

## 口径与复核

- 正式 Task1–5 训练缓存：12,500 张、25 类、每类 500 张；`m=8192,k=41`。`R,b` 与正式 Task5 checkpoint 逐元素一致，编码调用未改的 `RandomSparseEncoder.encode`。A 部分为配对分析临时按 class 排序；B 部分保留原训练流顺序。
- 三类配对均值使用前轮 **78,118,750 对精确普查**作独立校验。same-class 采用同类中两个**不相同**样本，different-class 的两端来自不同类，cross-task 又来自不同 task。`uniform_mean=k²/m=0.2052001953125`。
- B 部分逐样本遵循原顺序：eligibility 更新、计算 `F,N,I`、采样 plateau、对当前 active bit 做 Bernoulli(`p=0.5`) XOR。每个 task 末的 replay `W_B` 与**正式 checkpoint 及先前诊断 checkpoint 逐 bit 相同**；各 task gross 0→1 / 1→0 总数也与原动力学 CSV 相同。因此记录的是原轨迹事件的恢复，而不是新的训练对照。
- “旧 imprint”是此前 task 中实际 0→1 写入、在该 task 末仍为 1 的 bit 的并集；每次暴露还要求**本样本更新前**该 bit 仍为 1。它是写入来源代理，不是独占的类别/slot 所有权。`old_imprint_potential_pre_gate` 在全部 slot 计数；`old_imprint_exposure` 只在本样本抽中的 plateau slot 计数。`gross_10_on_old_imprint` 与同一步的 `old_imprint_bits_lost` 按定义相同，不能当两个独立证据。
- `hot_bit_score_sum=Σ_{i∈s}p_i^{old}`，恰等于与历史 sparse codes 的平均 overlap `old_code_collision_mean`；`hot_bit_score_mean` 再除以 41。三者只是尺度不同，不能把相同相关性当独立验证。Task1 无历史 task，相关分析限 Task2–5 的 10,000 个样本。

## A. 均值 excess 的精确拆解

`diagnostics/collision_mean_decomposition.csv` 包含 25 个同类行、50 个同 task 异类行、250 个跨 task 异类行，以及各组汇总行。每个 class 的每位边际概率 `p_i^c`、每 task 和整体边际概率都计算自正式编码，保存在 [逐 bit 表](../diagnostics/collision_bit_contribution.csv) 的 `p_class_*`、`p_task_*`、`p_overall` 列。对不同来源 A/B 的完整笛卡儿样本对，`E[J]=Σ_i p_i^A p_i^B` 是**线性计数恒等式**，不假设同一样本内各 bit 独立。

| 配对 | Uniform baseline | Marginal-only `E[J]` | 真实 `E[J]` | Marginals 解释的超额 | 剩余均值项 |
|---|---:|---:|---:|---:|---:|
| Same class | 0.205200 | 3.436779 | 3.361502 | 3.231579 | −0.075277 |
| Different class, same task | 0.205200 | 1.593486 | 1.593486 | 1.388286 | 0 |
| Different class, cross task | 0.205200 | 1.580174 | 1.580174 | **1.374973（100%）** | **0** |

对最相关的跨 task 异类组，**7.70× 的平均碰撞全部由 class-conditional bit marginals 的线性期望解释**：`0.205200 + 1.374973 + 0 = 1.580174`。这并不是说高阶依赖不存在；均值在数学上无法识别它。Same-class 的 `−0.075277` 也不是负的“高阶结构”：同类 500 张不放回配对需扣掉对角线，精确校正为 `(500·Σp_i²−41)/499 − Σp_i²`。因此不能拿 same-class 的 102.4% 比例解释成物理上的过度归因。

## A. 哪些 bit 贡献均值？

每个 bit 的组均值贡献 `c_i`、排序名次和累计值在 [逐 bit 表](../diagnostics/collision_bit_contribution.csv)。跨 task 异类的 marginal collision：排名前 1、10、41、82、128、256、512、1024 位分别贡献总均值的 **4.58%、23.84%、45.10%、58.21%、67.52%、81.51%、91.48%、97.26%**。累计达到 50%、75%、90% 分别只需 **54、182、452** 个 bit。这说明少量热点很重要，但尾部至 452 位才累计到 90%，并非单一 bit 独占。

按**全体样本激活频率**预先定义的 top 82 位对跨 task 异类均值贡献 `0.91966344`；扣除均匀模型对此 82 位的 `0.002054` 后，占总 excess `1.37497334` 的 **66.7365%**，与前轮 66.74% 完全一致。这里的“top 82 按全体频率”和“top 82 按跨组贡献”排序标准不同，后者贡献总均值 58.214%；不能混写为同一个统计量。

按题定 `q_i=p_i/41`，整体 `N_eff=1/Σq_i²≈1014.75`，仅为 nominal 8192 的 **12.39%**（约缩小 8.07 倍）；每 task 范围约 **722.72–927.15**。这个有效支持量反映 bit 使用集中度，不等于真实非零 bit 数。

## A. 保留 degree 的随机化尾部对照

为隔离高阶结构，做三份确定性 double-edge-swap 随机化（seed 9101/9102/9103）。每份接受 **2,562,500 次交换，即原矩阵 512,500 条 1-edge 的 5 倍**。交换仅在同一 class 的 500 行内进行，因此**同时精确保留每个 sample 的 41 个 active bit、每个 bit 的全局频次、每个 class/task 的 bit 频次**；这比只保留全局列总数更严格，避免把 class-conditional 边际差异误判为高阶结构。所有 degree 用断言复核。每份对每种配对类型独立有放回抽 **150,000 对**；真实值来自精确 census，均匀值为超几何解析式。抽样行的 CSV 给出均值及 `P(J≥1…5)` 的近似 95% CI（均值按抽样方差；比例按 binomial SE），CI 是**固定随机化矩阵条件下的 pair-sampling 误差**，不包含交换链未充分混合的系统误差。

| 跨 task 异类 | mean J | P(J≥1) | P(J≥2) | P(J≥3) | P(J≥4) | P(J≥5) |
|---|---:|---:|---:|---:|---:|---:|
| Uniform `k`-subset | 0.205200 | 0.186346 | 0.017717 | 0.001087 | 0.0000478 | 0.00000160 |
| Real（精确） | **1.580174** | **0.738033** | **0.428618** | **0.215737** | **0.101550** | **0.047411** |
| Degree-preserved 3 replicas 范围 | 1.576633–1.581493 | 0.793160–0.794873 | 0.467120–0.469720 | 0.209767–0.211107 | 0.075413–0.075893 | 0.022240–0.022873 |

保度后**均值如预期维持**，但分布形状显著不同：真实 `P(J≥5)=0.047411`，约为对照的 **2.07–2.13 倍**；`P(J≥4)` 也较高，而 `P(J≥1–2)` 反而较低。这是比保度零模型**更分散、两端更重**的 pairwise overlap 分布，说明编码还保有 class 内样本相关性/共激活结构；不能把它当作额外的平均 `J` 贡献。三份对照方向一致，差距远大于 CSV 中 150,000 对的抽样 CI；但 5× edge 交换不是严格的马尔可夫链混合证明，所以这里只下诊断结论。同 task 异类亦见同向 `P(J≥5)`：真实 0.049822，对照 0.023867–0.024313。[对照全表](../diagnostics/collision_degree_preserved_control.csv)

## B. 碰撞是否传到写入？

[逐样本 replay 表](../diagnostics/sample_collision_write_link.csv) 有 12,500 行，每行记录更新前 active 41 位、历史均值/最大碰撞、eligibility rival cosine、`I,F,N,f_eff`、实际 plateau、旧 imprint 潜在/实际暴露与 gross flip。下表按**各 task 内**历史平均碰撞排序后各分四等份，再把 Task2–5 的同一等份合并（各 2,500 张）；因此不把 task 编号带来的历史库大小差异误作样本碰撞效应。

| 碰撞分位 | 历史 mean J | I | f_eff | plateau / 样本 | 旧 imprint 潜在暴露 | 实际暴露 | 旧 imprint 1→0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 低 | 0.978 | 0.8749 | 0.001455 | 5.952 | 13,372 | 19.898 | 9.948 |
| Q2 | 1.387 | 0.8410 | 0.001500 | 6.058 | 17,364 | 25.932 | 12.905 |
| Q3 | 1.742 | 0.8116 | 0.001534 | 6.406 | 20,553 | 32.612 | 16.362 |
| Q4 高 | 2.354 | 0.7665 | 0.001532 | 6.342 | 25,939 | 39.720 | 19.865 |

Q4 相比 Q1，真实 plateau 旧 bit 暴露约 **2.00×**、old-imprint 1→0 约 **2.00×**。因此碰撞并未停在 encoder 统计层，确实沿当前轨迹传导到旧 bit 写入机会及 LTD。此处是**观测关联**，不等于“单独改变 sparse collision 就会导致这些差值”的因果实验；当前 `W_B` 状态及类别组成也参与。

`I=exp(−β·max rival eligibility cosine)` 来自论文 Eq.(5)；它读取**类别 trace**，不读取 `W_B` 的实际 row/bit。Task2–5 pooled 的 Pearson/Spearman：历史 mean J 对 `I` 为 **−0.826/−0.828**，对更新前 old-imprint 潜在暴露为 **+0.483/+0.460**，对实际 old-imprint 暴露为 **+0.303/+0.317**，对 old-imprint 1→0 为 **+0.295/+0.307**；潜在暴露对 `I` 为 **−0.596/−0.565**，实际暴露对 `I` 为 **−0.359/−0.364**。逐 task 的“mean J vs I” Pearson 在 **−0.883 至 −0.813**；“潜在暴露 vs I”在 **−0.856 至 −0.807**。故 `I` **不是基本无关**：它对代码碰撞和潜在旧 bit 暴露有强反向排序能力。但它只是样本级、非 slot/bit 级代理，不能精确识别某个 plateau row 的旧 imprint。

更重要的是完整门控的净效果：历史 mean J 与 `f_eff` pooled Pearson/Spearman 只有 **+0.073/+0.089**，与 plateau count 为 **+0.052/+0.059**。Q4 的 `I` 低于 Q1，但 `f_eff`、plateau count 未同步降低，实际旧 bit 暴露反而翻倍。原因从乘积式可见：`F,N` 也随样本改变，可能抵消 `I`；**不能因为 `I≤1` 就判定实际高碰撞样本写得更少**。这不否认前轮解析去掉 `I` 后预期 plateau 增多约 16%–19%，但该反事实不证明当前门控已足以抵消高碰撞样本的旧 bit 暴露。定位为：`I` 对碰撞有意义但粗粒度；**完整 gating 到实际写入的对准不足**，不是 `I` 本身与碰撞无关。

## B. XOR 条件概率核验

Task2–5 有 **295,404 次**“旧 imprint bit 在抽中 plateau 的当前 active 位、且更新前为 1”的暴露，其中 **147,701 次**本步 1→0。观测率 **0.4999966**，与代码/配置的 Bernoulli(`p=0.5`) 几乎一致（描述性 binomial SE≈0.000920；`z≈−0.0037`）。各次写入前按实际 `W_B` 状态截取，没有把任务起点 snapshot 当成每步状态。这个核验支持**普通 XOR 的 1→0 条件机制**，未发现额外的旧 bit 定向异常。暴露次数本身由 history、plateau 和 `W_B` 共同决定，所以不把全轨迹事件误称为独立随机试验。

## 六项最终判断及定位

1. **7.70× 平均 collision：** 跨 task 异类真实 `1.580174 = 0.205200 uniform + 1.374973 marginal imbalance + 0 mean residual`；按这个明确的 excess denominator，边际频率解释 **100%**。same-class 的有限总体对角线校正另计。
2. **保度后的 tail：** 真实 `P(J≥5)=4.7411%`，对照约 `2.22%–2.29%`，有明显残余高尾；它改变**分布**而不增加均值。
3. **真实写入：** 高碰撞四分位的旧 imprint 实际暴露及 1→0 均约为低四分位 **2×**。已有 [feature-drift 审计](../diagnostics/memory_drift.csv) 显示 Task1 查询在 `WB_task1→WB_task5` 的未阈值 overlap cosine 约 0.834，但本次逐样本关联**没有识别**每次旧 bit 1→0 对最终 feature drift 的因果份额；不能把 2× 直接解读为 drift 份额。
4. **`I` 相关性：** 对历史 mean collision 强反相关（Pearson −0.826；每 task −0.813 至 −0.883），对更新前旧 imprint 潜在暴露也强反相关；对实际 plateau 暴露较弱（−0.359）。`hot_bit_score` 与 mean collision 代数相同，不构成重复证据。
5. **gating 定位：** `I` 对碰撞是**有效但粗粒度的代理**；完整 `f_eff` 和 plateau count 与碰撞几乎不呈负相关，高碰撞样本仍有更多旧 bit 暴露。因此结果更接近“**粗糙相关、未充分对准最终写入**”，不是“基本无关”，也不能说“正确针对且充分解决”。
6. **XOR：** 暴露旧 bit 的实际 1→0 为 49.9997%，符合原 Bernoulli(`p=0.5`)；没有在此观察到 implementation anomaly。

综合归类：**A encoder marginal imbalance 是均值超额的直接统计来源；B encoder higher-order co-activation 表现在重尾而非均值；C gating proxy mismatch 只适用于从 `I` 到实际 `f_eff`/slot 暴露的不足，不能说 `I` 完全不识别碰撞；D ordinary XOR consequence 是暴露后旧 bit 丢失的直接机制；E implementation anomaly 没有证据。** 这轮仅诊断，不提出参数选择或模型修复。
