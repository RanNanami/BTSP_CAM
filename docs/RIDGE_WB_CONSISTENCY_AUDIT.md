# Ridge consolidation 与动态 `W_B` 的时间一致性审计

审计日期：2026-09-23。本轮仅阅读 BTSP-CAM 论文 PDF、附带的 simpleBTSP 源码、当前实现及既有诊断报告；**没有修改代码或运行实验**。论文页码均指 `2233_BTSP_CAM_A_Brain_Inspired.pdf` 的 PDF 页码。全文检索了 ridge、consolidation、sufficient statistics、`A_t`、`B_t`、`Z_t`、feature、memory update、drift、stability、plasticity、fixed memory、freeze、replay、exemplar 等词，并核对了第 5、6、8、17–18 页的公式与排版。

## 结论先行

论文**明确规定**：先处理完当前 task 的 BTSP 写入，再以当时的 `W_B` 构造当前 task 的 `Z_t`，于 task 结束时将 `Z_t^T Z_t` 和 `Z_t^T Y_t` 加入历史统计并重新求 ridge。论文也**明确承认**后续写入会使 CA1 表示漂移；它用“缓慢漂移”、prototype 稳定性和理想化 margin 分析论证影响有限。

但论文**没有说明**如何让早期 task 保存的统计量重新对应后期 `W_B`。Eq.(8) 的逐 task 累加不能自动等同于“用当前 `W_B` 重新表示全部已见数据”后的 ridge 统计。这个时间一致性问题在所审论文中是 **NOT DISCUSSED**，不是“作者假设 `W_B` 完全不改变旧特征”：作者实际上承认漂移，只是没有给出旧统计与当前特征之间的精确连接。

## A. 论文明确说明了什么

| 问题 | 论文证据 | 判断 |
|---|---|---|
| Ridge 统计何时计算？ | 第 5 页 Algorithm 1 第 4–12 行先对当前 `D_t` 的每个样本更新 `W_B`；第 13–15 行在 **Phase 2: Systems Consolidation** 中构造 `Z_t`、更新 `(A_t,B_t)`、计算 `W_out^(t)`。第 6 页 §4.5 又写明 “at the end of task t”。 | **每个 task 结束时**，在当 task 全部写入之后；不是最终所有任务完成后一次性拟合，也不是 Algorithm 1 所规定的逐样本在线 ridge 更新。 |
| Ridge 使用更新前还是更新后的 `W_B`？ | Algorithm 1 第 11 行已更新权重，第 12 行结束当前 task 循环，第 14 行才按 Eq.(7) 构造 `Z_t`。 | **更新后的当 task `W_B`**；这是算法顺序的直接推论。论文没有另写“用更新前快照”的路径。 |
| Eq.(8) 具体保存什么？ | 第 6 页 Eq.(8)：`A_t=A_{t-1}+Z_t^T Z_t`，`B_t=B_{t-1}+Z_t^T Y_t`，`W_out^(t)=(A_t+γI)^{-1}B_t`。 | 保存聚合的二阶与标签交叉统计，随后重求线性权重；“重新求 ridge”不等于“重新计算历史任务的 `Z`”。 |
| 是否保存/重算历史样本？ | 第 6 页 §4.5 明言只用 **current-task data** 更新充分统计，**without storing any past exemplars**；第 3 页 §3.1 将历史任务样本 `D_{1:t-1}` 禁止保留作为严格 exemplar-free 约束。 | 论文描述的算法**不保存或回放历史图像**。它也**没有指定**保存每个历史样本的 sparse code `s`、CA1 特征 `z`，或在新 `W_B` 下重算旧 `z` 的步骤。后一项属 **NOT DISCUSSED**，不能把沉默解释为暗中重算。 |
| task 结束后 `W_B` 是否继续变化？ | 第 5 页 Algorithm 1 以 `D_t` 和现有 `W_B` 为输入，第 4–11 行写入；第 4–5 页 Eq.(2),(6) 定义后续 plateau bit flips；没有 task 结束永久冻结 `W_B` 的步骤。第 17 页 Appendix F.2 明说 subsequent tasks 的 LTD 更新会使旧表示退化。 | 连续任务执行时 **`W_B` 继续变化**。论文冻结的是第 3 页的 PTM backbone，不是整个增量过程中的 `W_B`。 |
| 作者有没有考虑 drift？ | 第 5 页 Proposition 4.2 假定 CA1 competition、每 task 重估 ridge、类几何仍分离；第 6 页称 ridge 将“potential geometric drift”转为较小 margin 收缩；第 8 页 §6.3 称 CA1 空间 “slowly drifting” 并报告 prototype 相似度 >0.9；第 17–18 页 Appendix F.2 建模 `\tilde z_c=ρz_c+ε`。 | **有定性与理想化 margin 讨论**，不能写成“论文完全没提漂移”。这些段落没有展示 Eq.(8) 的跨版本统计如何精确匹配当前 `W_B`。 |

**Algorithm 1 实际只命名两个 Phase。** 用户所说的“Phase 3: ridge”在逻辑上是 CA1 后的第三个操作，但论文将 CA1 构造和 ridge 更新都放在 Phase 2，第 14–15 行。

## B. 论文没有说明的时间一致性问题

令 `M_t=W_B^(t)` 为第 `t` 个 task 写入完成后的记忆，`S_i` 为第 `i` 个 task 的 sparse-code 矩阵，并记 `Z_{i|j}=f(M_j,S_i)`，其中 `f` 是论文 Eq.(7) 的 CA1 映射。

按 Algorithm 1 + Eq.(8) 逐 task 执行，历史统计为：

```text
A_t = Σ_{i=1}^t Z_{i|i}ᵀ Z_{i|i}
B_t = Σ_{i=1}^t Z_{i|i}ᵀ Y_i
```

而如果目标是以**当前记忆 `M_t`** 对全部已见训练样本定义一个统一的 ridge 特征空间，则相应统计应为：

```text
A_t(current) = Σ_{i=1}^t Z_{i|t}ᵀ Z_{i|t}
B_t(current) = Σ_{i=1}^t Z_{i|t}ᵀ Y_i
```

例如 task 2 结束时，Eq.(8) 留下 `Z_{1|1}` 的旧贡献，加上 `Z_{2|2}` 的新贡献；旧类推理却使用 `Z_{1|2}`。两组统计**一般不相等**。准确地说，只有旧特征不变，或尽管特征变化但两组 Gram／交叉标签统计恰好相同，或存在论文说明的等价变换与同步权重处理时，才能推出一致；论文没有给出这些条件或操作。这是由公式直接导出的**时间一致性疑点**，不是本轮新增实验结论。

| 具体缺口 | 审计状态 |
|---|---|
| 随 `W_B` 更新重算历史 `Z_i`，并更新历史 `A/B` 贡献 | **NOT DISCUSSED**；Algorithm 1 和 §4.5 只处理当前 task。 |
| 保存历史 `s`、`z` 或历史样本以便重算 | **NOT DISCUSSED**；论文明确不存过去 exemplars。Eq.(8) 的聚合统计本身不保留每样本 `s/z`。 |
| 对 `W_B` 漂移推导 `A_t,B_t` 的精确变换/误差界 | **NOT DISCUSSED**；没有 `\|A_t-A_t(current)\|` 或相应 `B` 的界。 |
| 假设 `W_B` 更新完全不改变旧特征 | **不能归因给作者**；§6.3 和 Appendix F.2 明确承认 CA1/旧表示漂移。 |
| 假设 drift 足够小，不会严重损坏 readout | **论文定性主张/理想化条件**；§6.3 的 prototype 稳定性、Proposition 4.2 的类分离条件及 Appendix F.2 的 `ρ,ε` 模型支持其叙述，但没有证明混合版本充分统计等于当前版本统计。 |

第 17 页 Appendix F.2 把退化后的表示写成 `\tilde z_c=ρz_c+ε`，并在 Eq.(35) 用 `\tilde Z^T\tilde Z` 写出理想化 ridge 表达式。**这不是 Eq.(8) 的实际累计矩阵等于 `\tilde Z^T\tilde Z` 的证明**：附录没有说明在不回放历史数据时，所有历史样本的 `\tilde Z` 如何获得，也没有把该式与 `Σ_i Z_{i|i}^T Z_{i|i}` 联系起来。第 8 页 “ridge re-fits ... on top of the slowly drifting CA1 space” 描述了作者的稳定性解释，不是一个旧统计刷新算法。

### exemplar-free / no-replay 的边界

- **论文事实**：只累计 `A/B` 且不存历史 exemplars，按论文给出的操作并不违反其 no-replay 声明；“可能存在旧统计与新特征不一致”与“偷偷 replay”是两个不同问题。
- **未指定的替代路径**：若在新 `W_B` 下逐样本重算历史 `Z_i`，必须重新取得历史输入/代码或另有未给出的等价方法。论文没有宣称这样做，不能把它补进 Eq.(8)；若保存并重用旧样本/逐样本代码作为训练输入，还必须重新审视与严格 no-replay 声明的兼容性。
- **当前工程实现的区别**：`src/btsp_cam/model.py:198-211,293-341` 的模型状态只累加统计，不在 checkpoint 保存逐样本 `s/z`；`model.py:250-253` 在推理时使用**当前** `W_B`。项目另有逐 task frozen-ViT 特征缓存（`src/btsp_cam/feature_cache.py`、`experiments/run_btsp_cam_cifar100.py`），它是实验运行/评估基础设施，确实在磁盘保留逐样本 `h`；正式 `learn_task` 没有读旧 task 缓存来更新 ridge。因此只能说**正式更新路径没有历史训练样本 replay**，不能笼统说整个工作目录物理上不保留历史特征。上一轮离线重建历史 ridge 也仅是 oracle 诊断，不是论文算法。 

## simpleBTSP 的继承边界

| 问题 | 附带 simpleBTSP 源码 | 对本问题的意义 |
|---|---|---|
| 是否逐样本写入？ | `upstream/simpleBTSP/main_BTSP_feedforward_model.py:113-128` 在 `for idx in range(M)` 中连续更新二值 `W`；memory 在写入期可以变化。 | simpleBTSP **不是**“整个学习过程记忆静止”。 |
| recall 何时进行？ | 同脚本第 132–169 行在完整 `M` 个 pattern 写入并固定该轮 `W` 后，才构造原/扰动 query 并计算二值 recall；feedback 脚本第 59–113 行先建好 `W_feed/W_back`，第 115–151 行后进行 recall。 | 其展示的检索实验是**写入结束后的静态 memory recall**。 |
| 是否有 task-wise ridge/A/B？ | 两个脚本均没有 Eq.(8) 式的跨 task 聚合 ridge 统计；recall 用当前固定权重与阈值。 | simpleBTSP 不存在“旧 ridge 统计由旧 `W` 生成，而推理用新 `W`”这一**特定**矛盾；但不能推论其顺序写入绝无旧记忆退化。 |
| BTSP-CAM 改变了什么？ | BTSP-CAM 将写入跨 class-incremental tasks 持续进行，且每 task 保存闭式 readout 的旧 `A/B`。 | 旧统计与动态 `W_B` 的时间一致性是 BTSP-CAM 的新接口问题，**不能**从 simpleBTSP 的静态 recall 实验直接获得保证。 |

## C. 既有实验提示了什么（本轮未重跑）

上一轮 `docs/TAU_DRIFT_ANALYSIS.md` 的**未阈值化、固定 Task1 五分类探针**中，使用 Task1 ridge 但将评估记忆从 `WB_task1` 换到 `WB_task5`，准确率由 92.4% 降到 46.8%；在相同 `WB_task5` 上用 Task1 历史训练数据离线重建 ridge 后为 86.6%。固定 Task1 测试样本的 `a` 对 `a_1` 均值 cosine 约 0.834。这个对照与 readout–feature 不匹配相符，但只分解五分类 probe，**不证明**论文作者的实际实现等于本地实现，也不能把恢复比例直接套到正式 25 类 CIL 旧类错误。

首任务 `τ=0.5` 的近零表示是另一问题；本审计讨论的是即使 `a` 保留分类信息，后续动态 `W_B` 仍会改变历史统计所依据的映射。两者不可混为一个原因。

## 最终分类判断

### A. 论文明确说明

1. 当 task 写入完成后构造当前任务 `Z_t`；task 结束时仅用当前数据更新 `A_t,B_t`，并重新求 ridge。
2. 不保存过去 exemplars；frozen PTM 固定，但 `W_B` 仍由后续任务更新。
3. 作者承认 CA1 drift，并在类几何保持分离等条件下定性/理想化论证 margin 稳定。

### B. 论文没有说明

1. 没有历史 `Z_i` 重新计算、历史 `A/B` 修正或跨版本统计等价变换的步骤。
2. 没有证明 Eq.(8) 混合不同 `W_B` 版本的统计与当前 `W_B` 下所有已见样本的 ridge 统计一致。
3. 没有提供不存历史逐样本数据却又可精确刷新旧统计的机制。**这不等于证明论文结果错误**，但留下了可复现时必须显式标注的时间一致性缺口。

### C. 根据既有实验推断

本地任务中动态 `W_B` 造成的旧特征变化已足以损害固定 readout，离线重估能恢复大部分 Task1 五分类准确率。该观察支持“ridge–feature mismatch 是本地增量遗忘的一项重要来源”，而不是论文 Eq.(8) 解决了该问题的证据。
