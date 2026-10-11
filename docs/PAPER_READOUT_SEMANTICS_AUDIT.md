# BTSP-CAM 论文 readout / ridge 时间一致性语义审计

审计日期：2026-09-28。主要且优先来源：项目研究时提供的 BTSP-CAM 原论文 PDF（未随仓库分发）；页码均为 PDF 内印刷页码（与 PDF 页序相同）。本轮只阅读材料、逐页核对公式和文字；未训练模型、调参或修改代码。**论文事实、形式推导和实现未知项分别标记；不以本地 accuracy 判断作者实际实现。**

## 1. Executive conclusion

**分类：C — `PAPER_IS_INTERNALLY_AMBIGUOUS`。** Eq.(8)、§4.5 的“仅用当前任务数据”和 Algorithm 1 的执行顺序明确给出历史各任务特征 Gram/交叉项的累积（A 语义）。另一方面，§6.3 声称 ridge 在“缓慢漂移的 CA1 空间上 re-fits”，Appendix F.2 又直接以漂移后的 `\tilde Z` 写 ridge Gram；它们描述的是当前漂移表示下的 readout，但没有说明怎样由 Eq.(8) 中旧坐标的统计量得到该 Gram。后两处**不是**历史样本重编码或 coordinate-aligned 更新的明确算法声明，因此不能判 B；但也不能说正文与理论全都无条件支持 A。此处是 **THEORY / ALGORITHM INTERFACE GAP**，不是对实际作者代码的结论。

## 2. Exact evidence table

“支持”仅评价该段**字面**对 A（旧 `Z_i` 统计永久累积）/ B（当前 `W_B` 统一空间重拟合）的支持强度；`B 倾向`不等于证明 B。短引文保持原文英文，略去公式的重复抄写。

| 页 | Section / 位置 | 原文短引文 | 严格字面含义 | 支持 |
|---|---|---|---|---|
| 1 | Abstract | “closed-form ridge readout” / “after each task” | 每任务后有解析 readout；不指定旧特征是否重算。 | neither |
| 2 | Introduction | “logits are consolidated from the evolving CA1 representation” | CA1 会演化；没有给统计量的时间坐标定义。 | neither（B 倾向措辞） |
| 2 | §2.1 | “replay-free heads ... updated with sufficient statistics” | 将方法置于解析/统计量更新语境；未独立定义本方法。 | neither |
| 3 | §3.1 | “restricted to access data exclusively from the current dataset” | 训练阶段只能访问当前 `D_t`。 | A；反对原始历史数据 replay 版 B |
| 3 | §3.1 | “storage of historical samples ... is prohibited” | 禁存历史样本；并未禁止某种聚合统计，但没有给出这种统计。 | A；不排除抽象的无样本对齐 |
| 4 | §4 Methodology | “resulting CA1 representation is then consolidated” | memory 写入之后 readout；未说所有历史 CA1 特征重算。 | neither |
| 5 | Algorithm 1，行 4–15 | “Construct CA1 features `Z_t` via Eq. (7)” / “Update `(A_t,B_t)`” | 当前 `D_t` 的写入循环结束后构建 `Z_t`，然后更新统计量。 | A |
| 5 | Proposition 4.2 | “ridge readout is re-estimated at the end of each task” | 权重每任务重新估计；不指定设计矩阵是否为当前 `W_B` 下的全部历史样本。 | neither |
| 5–6 | Proposition 4.2 intuition | “CA1 layer provides a stable basis” / “aligns this basis to the labels” | 论证漂移较小时 readout 可保持 margin；未定义旧 `A/B` 的坐标变换。 | neither |
| 6 | §4.5，Eq.(7) | `a(x)=W_Bs/\|s\|_1`, `z(x)=max(0,a(x)−τ)` | CA1 特征依赖当前 memory；memory 变动可令同一输入特征改变。 | neither |
| 6 | §4.5，Eq.(8) | `A_t=A_{t−1}+Z_t^T Z_t`；`B_t=B_{t−1}+Z_t^T Y_t` | 明确在旧统计量上加当前批 Gram/交叉项；未给旧项重算/运输。 | A |
| 6 | §4.5，Eq.(8) 后 | “using only current-task data” / “without storing any past exemplars” | 对 `A/B` 的任务末更新只用当前任务数据。 | A |
| 6 | §4.5，Eq.(9) | `z(x)W_out^T` | 推断用当时的 CA1 特征和 readout；没有定义历史训练坐标的修正。 | neither |
| 8 | §6.3，Figure 4 | “prototype similarity remaining above 0.9” | 声称原型几何稳定；稳定不等于统计量严格同坐标。 | neither |
| 8 | §6.3，Table 4 讨论 | “re-fits ... on top of the slowly drifting CA1 space” | 比 Eq.(8) 更强的自然读法是 readout 适配漂移后的空间，但也可能仅指重解闭式权重；未说历史重编码。 | B 倾向，非 B 证明 |
| 9 | Impact Statement | “strict exemplar-free setting” / “historical data buffers” | 不保留历史数据缓冲区；未给 current-space alignment 的其他路径。 | A；不排除聚合统计 |
| 12 | Appendix B，glossary | “`z(x)` Thresholded memory output” / “`Wout` Linear readout weights” | 补充符号，没有 sparse-space `C/D`、transport 或 task snapshot 定义。 | neither |
| 12 | Appendix B.1 | “does not model this macroscopic task process directly” | 附录理论本身不直接建模完整跨任务训练过程。 | neither；限制 F.2 的证明范围 |
| 17 | Appendix F.2，Eq.(35) 前 | “actual representation `\tilde z_c` degrades” / “`\tilde z_c=ρz_c+ε`” | 用退化后的类别潜在表示建模旧类 drift。 | neither |
| 17 | Appendix F.2，Eq.(35) 前 | `w_c=(\tilde Z^T\tilde Z+γI)^{−1}\tilde z_c` | 理论 ridge Gram 与目标信号都采用带波浪号的漂移表示；未推出 Eq.(8) 旧 `A/B` 等于该 Gram。 | B 倾向，非 B 算法证明 |
| 18 | Appendix F.2，Eq.(37)–(38) | `ρ(T)≈1−n_c(T)/k`；`Δ(T)≈Δ(0)−κn_c(T)` | 在小退化近似下讨论 margin 收缩，不是 `A/B` 时间一致性等式。 | neither |

正文中 `re-fits` 跨行排版为 `re-`/`fits`，已按第 8 页图像核对，不能仅凭文本检索缺失判定不存在。第 7–8 页关于 old-class accuracy、prototype stability、memory update 的陈述是实验解释或几何主张，未给历史设计矩阵的重算算法。

## 3. Eq.(8) reconstruction

**论文符号。** §3.1 定义 `D_t={(x_i,y_i)}_{i=1}^{N_t}` 为当前训练任务；§4.5 Eq.(7) 定义由 sparse code `s` 和 memory `W_B` 生成每个样本的 `z(x)`。Algorithm 1 行 14 称任务特征矩阵为 `Z_t`；Eq.(8) 中 `Y_t` 是该任务标签矩阵（论文未详细列出其列扩展/one-hot 实现）。`A_t` 是 readout 特征 Gram 的历史累积，`B_t` 是特征–标签交叉统计；`W_out^(t)=(A_t+γI)^{-1}B_t`。单样本写法与第 6 页 Eq.(9) 的转置布局可互相换算，不影响时间语义。

**原文明确：**

```text
A_t = A_{t−1} + Z_t^T Z_t
B_t = B_{t−1} + Z_t^T Y_t
W_out^(t) = (A_t + γI)^−1 B_t
```

结合 Eq.(8) 后“using only current-task data”，`Z_t` 是**当前任务数据**的 CA1 特征，不是 `D_1∪…∪D_t` 的特征矩阵。结合 Algorithm 1 行 4–12 先写完 memory、行 14 后构建特征，可**推导**该 `Z_t` 使用任务末的 `W_B^(t)`，但论文没有显式写 `Z_{t|t}` 下标。未发现定义 `Z_t=all-seen samples under current W_B`。由递推式展开得到 `A_t=Σ_{i≤t}Z_i^T Z_i`、`B_t=Σ_{i≤t}Z_i^T Y_i`；这是 Eq.(8) 的代数结果，而 `Z_i` 在后续任务变更 `W_B` 时并无更新项。

## 4. Algorithm 1 execution trace

| 顺序 | Algorithm 1 行号 | 字面操作及时间语义 |
|---|---:|---|
| 1 | 1–3 | 输入当前 `D_t`、`W_B`、eligibility；进入 memory allocation。 |
| 2 | 4–11 | 遍历 `D_t`：编码 `s`、更新 trace/plateau、对 `W_B` 做 XOR bit flip。 |
| 3 | 12–14 | 当前任务写入循环结束，依 Eq.(7) 构建 `Z_t`；因此是写入**后**的 `W_B`（从顺序推导）。 |
| 4 | 15 | 依 Eq.(8) 更新 `A_t/B_t` 并解 `W_out^(t)`。 |
| 5 | 下一任务 | `W_B` 继续作为可更新 memory 输入；Algorithm 1 无任务末永久冻结指令。 |

`D_1…D_{t−1}` 重新访问、历史 `Z_i` 重算、使旧 `Z_i` 进入当前 `W_B` 坐标系的统计量、`C=Σss^T`/`D=ΣsY^T`、feature transport/transform、readout correction、prototype remapping：**Algorithm 1 中未发现**。本文其他公式/附录中也未找到这些 readout 对齐步骤。该“未发现”是对公开描述的审计，不是证明作者私有实现没有它们。

## 5. Text wording audit

第 5 页 `re-estimated at the end of each task` 的上下文是 Proposition 4.2 的**条件**，紧随的结论是 memory 中类别仍几何分离时 margin 收缩受限。它并未写“重新生成历史 `Z`”。Eq.(8) 已包含每任务重解 `W_out`，所以 `re-estimated` 可与 A 完全相容。

第 8 页 `re-fits ... on top of the slowly drifting CA1 space` 更有张力。相邻文字先说 Figure 4 的 prototype 稳定，再说 Table 4 将结构稳定关联至决策边界；后句是移除 CA1/consolidation 后 old-class accuracy 降低。就自然语义而言它可指 readout 与**目前**漂移的 CA1 space 配合；但上下文未交代重拟合所用数据、历史特征、Gram 或新统计量，因此不能仅由 `re-fits` 推出 B。第 4 页 `consolidated`、第 6 页 `closed form` 同样说明重解/整合，不说明 history realignment。

## 6. Proposition 4.2 / Appendix F.2 audit

Proposition 4.2 明确把“CA1 competition + 每任务 readout re-estimated + `W_B` 内类别保持分离”作为 margin 论断的条件。Appendix F.2 的 `z_c` 是“class c 的 ideal latent representation”，不是完整样本级 `Z_i` 的时间索引；因此它至多是类别潜在向量/原型式抽象，论文并未定义为历史训练样本逐个重编码的矩阵。`ρ∈(0,1]` 是标量 retention rate，Eq.(37) 令其随 task/类别复用 slot 数变化；不是一般线性 feature transport。`ε` 仅称 sparse noise；此处未给零均值、独立性、具体方差或逐样本变换规则。推导另假设类别 manifold 近似正交，并把 Gram 近似为对角块；负类干扰项 `δ` 由 Theorem E.2 的 overlap 上界控制。

F.2 写 `w_c=(\tilde Z^T\tilde Z+γI)^{-1}\tilde z_c`（第 17 页），其 Gram 和被评分的类别信号都使用漂移后的表示，接着在小 `n_c(T)≪k` 近似下得到 Eq.(35)–(38) 的 margin 收缩。**这不是完整多类、跨任务 Eq.(8) ridge 统计量的推导**。尤其未证明

```text
Σ_i Z(D_i; W_B^(i))^T Z(D_i; W_B^(i))
    = Σ_i Z(D_i; W_B^(t))^T Z(D_i; W_B^(t))
```

或对应的 `B` 等式/误差界。`ρz_c+ε` 的小漂移假设也不自动给出此等式。因此：**historical `A/B` 在 `W_B` 漂移后仍有效 — NOT PROVED**；理论公式使用同一漂移表示形成 readout 与评分信号，却没有说明 Eq.(8) 如何产生它，标记为 **THEORY / ALGORITHM INTERFACE GAP**。第 12 页 Appendix B.1 还明确声明该分析不直接建模宏观任务过程，应避免把局部 margin 模型解释为完整 continual ridge 实现证明。

## 7. Coordinate-aligned equivalence check

本地 experimental repair 的数学对象（仅用于形式比较，不是论文声称）为历史 sparse-code 聚合量 `C_t=Σ_{i≤t}S_i^TS_i`、`D_t=Σ_{i≤t}S_i^TY_i`。当且仅当特征采用**线性、无阈值** `Z_i(W)=S_iW^T/k` 时，可代数得到

```text
A_current(W) = W C_t W^T / k²
B_current(W) = W D_t / k.
```

相反，论文 Eq.(7) 指定 `Z=ReLU(SW^T/k−τ)`，一般不能从上述 `C/D` 精确重构其 `Z^TZ` 和 `Z^TY`。即使在 `τ=0` 的线性特例，Eq.(8) 的逐任务旧 `Z_i(W_i)` 累加也**不会自动**变成 `W_t C_t W_t^T/k²`；需要额外定义并维护 `C/D` 或等价历史对齐机制。Eq.(8)、附录推导与符号表均没有这些对象：**NO EXPLICIT EQUIVALENCE FOUND**。这只比较公式，不据此猜测作者代码。

## 8. Hidden-alignment mechanism search

| 待排查机制 | 公开文本结果 |
|---|---|
| readout 训练后永久冻结 `W_B` | **NONE FOUND**；Algorithm 1 按任务继续写 memory（第 5 页）。 |
| 每任务重新初始化 ridge、仅新任务拟合 | **NONE FOUND**；Eq.(8) 明确保留 `A_{t−1}/B_{t−1}`（第 6 页）。 |
| 在当前 `W_B` 下重算旧类 prototype / 历史 `Z` | **NONE FOUND**；第 8 页 prototype stability 是分析结果，不是重算步骤。 |
| 保存历史 sparse samples 或 CA1 features | **NONE FOUND**；第 3、6、9 页强调仅当前数据/无历史 exemplars。 |
| 保存 sparse-space `C/D` 或等价聚合 | **NONE FOUND**；仅显式定义 CA1-space `A/B`（第 6 页）。 |
| `T_{t−1→t}`、旧权重投影、classifier correction | **NONE FOUND**；Eq.(8) 仅重解 ridge。 |
| replay 历史图像/数据 | **NONE FOUND**；与 §3.1 strict exemplar-free 冲突。 |
| classifier 仅做当前任务分类 | **NONE FOUND**；§3.1 要求 unified seen-class 预测，第 6 页 Eq.(9) 为统一 logits。 |
| inference 选择 task-specific `W_B` snapshot | **NONE FOUND**；Eq.(7)/(9) 没有 snapshot 选择规则。 |

整体结论：**NONE FOUND**，指论文正文、算法、符号表与附录未描述可使 A 精确变成 B 的隐藏对齐机制；不代表可证明作者实现中不存在。

## 9. Internal consistency assessment

1. **论文明确算法 A：** 当前任务 memory 写完后得到 `Z_t`，只用 `D_t` 增量更新 `A/B`，每任务重解 ridge。
2. **更强的文字/理论语义：** 第 8 页说 readout 在 slowly drifting CA1 space 上 `re-fits`；第 17 页以 `\tilde Z^T\tilde Z` 建 ridge Gram。这与“当前空间重新拟合”的数学叙述相似，但并非执行步骤。
3. **不能唯一调和之处：** 对漂移后的旧类，Eq.(8) 保留旧坐标 Gram；F.2 使用漂移表示的 Gram，却没有 `A/B` transport、重算或等价 sparse-space 聚合。F.2 也可能只是忽略该时间差的理想化分析；论文未说明。
4. **限制：** 文中没有足够证据判断作者真实程序是否用了 B。论文第 1 页声明[代码地址](https://github.com/ericzhengz/BTSP-CAM)；本轮对该地址的只读访问返回 404，**author code could not be inspected**。这既不证明代码不存在，也不证明实际实现等于论文伪代码或含有 hidden alignment。

## 10. Final classification

**C — `PAPER_IS_INTERNALLY_AMBIGUOUS`。** Eq.(8)/Algorithm 1 与“only current-task data”是 A 的明确算法证据；“re-fits ... slowly drifting CA1 space”和 F.2 漂移后的 Gram 引出 B 式理论/叙述张力；但**没有**历史重编码、对齐统计或等价变换的公开规则，故不能判 `PAPER_EXPLICITLY_MATCHES_CURRENT_SPACE_REFIT`。对“论文公开文本是否支持作者实际采用了与 coordinate-aligned ridge 相同或等价的方法”的严格回答是：**AMBIGUOUS**——可以识别语义张力，不能将该方法归于作者实际实现。
