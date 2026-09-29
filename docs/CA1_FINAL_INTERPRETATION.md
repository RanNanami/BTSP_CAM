# CA1 公式来源：最终解释（本轮只审计，不修模型）

审计日期：2026-09-23。范围为附带的 BTSP-CAM 18 页 PDF（含附录）、官方 simpleBTSP 源码、当前实现及已完成的 Task 1–5 诊断。**本轮没有运行新实验，没有改代码或任何模型参数。** 详细逐项核对见 [论文公式审计](CA1_PAPER_FORMULA_AUDIT.md) 和 [引用文献审计](CA1_REFERENCE_AUDIT.md)。

## 一句话判断

**当前 CA1 collapse 有充分证据表明发生在论文明确写出的 `a → max(0,a−τ)`；没有发现当前代码误读了 Eq.(7)，也没有证据证明论文漏写了一层 normalization。最强的来源问题是：论文指定了 `τ` 的作用，却没有公开其数值或选择规则；当前 `τ=0.5` 是本地选择，且与本地 `p=0.5` 不满足论文附录的 `p>τ` 单次检索充分条件。** 这足以说明本地配置不能声称为论文参数复现，但尚不能证明论文的所有结果“不可复现”或作者私有实现具体用了什么阈值。

## 四种可能性分级

| 假说 | 论文事实 | 代码事实 | 实验观察／当前判断 |
|---|---|---|---|
| **A. 代码 CA1 公式错误** | Eq.(7)／Eq.(31) 明写 `a=W_B s/||s||_1`，`z=max(0,a−τ)`。 | `ca1.py` 求活跃列和、除以实际 `sum(s)`、再逐槽减 `τ` 截零；`W_B` 为 `4096×8192` 二值，`s` 为 exact-41 二值。 | **未发现公式级错误。** 不能把性能异常本身当作公式写错的证据。 |
| **B. 论文缺失额外 normalization / CA1 top-k** | 唯一明确的归一化就是 `||s||_1`；top-k 只在 Eq.(3) 的 sparse encoder。 | 代码也没有 row/column normalization、第二次 top-k、softmax、lateral inhibition。 | **无正面证据。** 假设隐藏归一化是推测，不应作为下一次修改的依据。 |
| **C. `τ` 未公开，当前取值不受论文支持** | Algorithm 1 与 Table 6 指明 `τ` 是检索阈值；正文、实现细节和附录未给数值/选取协议。Appendix D.1 仅给 `p>τ` 的理想化充分条件。 | `configs/btsp_cam_cifar100.yaml` 将 `τ=0.5` 标为 LOCAL_CHOICE，`p=0.5`。 | **最有证据支持的来源差距。** 当 `p=τ` 时不能使用论文的单次可靠检索保证；现有诊断中测试 `a` 的高分位仍低于 0.5。 |
| **D. 本地 `W_B` scale 与作者实际实现不同** | 论文给出 Eq.(6) 的翻转规则与 `W_B` 维数，但未公开 `p`、`τ` 等所有实现数值；Appendix F.1 的 `w_j≈s*` 是理想假设。 | 本地 `p=0.5` 被配置标作“simpleBTSP update probability”，但提供的 upstream 代码 `p_w=0.6` 是随机**连接 mask** 概率，LTP/LTD 示例为互斥 set/reset，不是 `p=0.5` 的逐突触 XOR；该配置来源标注不能由 upstream 验证。 | **可能的参数／写入尺度差异，但没有作者 `W_B` 或真实参数作对照，不能判定 D 为事实。** 本地 `a` 确实有可读类别信号，只是绝大多数测试激活小于本地阈值。 |

## 证据链：严格区分四类陈述

### 论文事实

1. 第 6 页 Eq.(7) 和第 17 页 Eq.(31) 对 `a`、`z` 给出相同的归一化 overlap + 减阈值算子；第 12 页 Table 6 解释 `τ`，但未列数值。第 8 页“CA1 competition”文字及消融没有增加一个不同的可执行算子。
2. 第 14–15 页 Appendix D.1 从全零槽、一次 plateau 推出 `K_active~Binomial(k,p)`，并称 `p>τ` 时，随着 `k` 增大，一次写入可高概率越过 `τk`。这不是对 `p=τ` 的成功保证。
3. 第 17 页 Appendix F.1 的高 overlap 论证以 `w_j≈s*` 为理想情况。它不是对 Eq.(6) 在任意参数、任意真实测试样本上的数值预测。

### 代码事实

1. `src/btsp_cam/ca1.py:20-29` 与 Eq.(7) 代数一致；`src/btsp_cam/sparse_encoder.py:27,40-42` 给 exact-k 0/1 代码；`src/btsp_cam/binary_memory.py:30-35` 给 `S×m` 二值矩阵。
2. `configs/btsp_cam_cifar100.yaml:52-67` 中 `p=0.5` 与 `τ=0.5` 并非从 BTSP-CAM PDF 中抄取。尤其 `p` 的“simpleBTSP update probability”来源标签与所附 upstream 脚本内容不符；它不改变当前 CA1 公式匹配结论，但降低参数溯源可信度。
3. simpleBTSP 的 `main_BTSP_feedforward_model.py:95,165-169` 使用 raw-count firing threshold 与二值 recall；BTSP-CAM 改为归一化后的连续 `z`，再交给 ridge。旧阈值系数不能无变换地继承为新 `τ`。

### 既有实验观察（引用上一轮结果；本轮未重跑）

1. `docs/CA1_DIAGNOSIS_ROUND1.md`：Task 1 current-test `a` 的 P99 为 0.2195，Task 5 为 0.2927；`a>0.5` 极罕见（Task 1 为 `4.88e-7`，Task 5 为 `9.77e-7`）。
2. 同一诊断：Task 1 thresholded cumulative 20.20%，unthresholded overlap cumulative 92.40%；Task 5 current-task thresholded 0.00%，unthresholded 94.40%。这支持**本地 threshold 是当前任务表示归零的直接原因**，不反推出论文应使用哪个 `τ`。
3. 同一诊断：固定 Task 1 测试代码改用 Task 5 的 `W_B` 后，pre-threshold cosine 均值降至 0.834；unthresholded Task 5 cumulative 仅 33.00%。这表明 continual readout 另有 drift 问题，但它不能解释 Task 1 已出现的 threshold collapse，也不构成 CA1 公式错误证据。

### 推测与界限

- **推测**：本地 `p=τ=0.5` 使单次新槽写入的期望重合率位于阈值附近；若测试代码只与写入代码部分重叠，实际 `a` 可能更低。前半句可由附录的 Binomial 模型推导，后半句是对测试泛化的机制推断，**不是作者参数的事实**。
- **不能断言**作者使用另一层 normalization、另一套 winner 规则或其它 `W_B` 初始化，因为被审材料无此定义。
- **不能断言**“论文所有公开信息都不足以复现”：本轮审计的结论限于这份 PDF、指定 simpleBTSP upstream 与核对过的引用原文；未获得可核实的 BTSP-CAM 作者实现、配置或作者回复。更准确的表述是**仅凭本 PDF 与 simpleBTSP 代码，无法唯一确定本实验的 CA1 阈值及相关尺度参数**。

## 下一步只建议验证，不在本轮执行

1. **优先做来源核实**：查找作者公开的 BTSP-CAM 配置／代码或向作者确认 CIFAR-100 BTSP-only 的 `τ`、`p`、是否固定、阈值的校准依据以及训练／测试时 CA1 是否严格按 Eq.(7)。这是判别“公开披露不足”与“本地参数选错”的直接证据，不需要反推 accuracy。
2. **做公式级静态对照**：用手算小型二值 `W_B,s` 对照 Eq.(7)、Eq.(31) 和当前函数，核对 exact-k、边界 `a=τ`、除数及维度。这是正确性验证，不改变实现。
3. **做写入尺度验证实验（另起实验轮次）**：固定现有写入协议及参数，记录全零槽初次 plateau 后同一代码的 overlap 分布，再记录同类未见测试代码的对应分布；与论文 Appendix D.1 的 `Binomial(k,p)` 理想情况和 Appendix F.1 的 `w≈s*` 假设分别比较。只检验假设，不以 accuracy 选择 `τ`。
4. **单独验证 continual drift（另起实验轮次）**：固定若干旧样本，比较不同 task 的 `a`／`z` 与旧 ridge 充分统计所依据的特征空间；不要将这一问题与首任务阈值归零混为一谈。

## 本轮停止判断

**优先检查 CA1 阈值／写入尺度的参数来源，而非改写 CA1 公式或加入未证实的操作。** 当前崩溃不是一个已识别的 Eq.(7) 实现错误；它更符合“论文给了阈值化算子，但没有给足数值配置，本地 `τ=0.5` 与本地 `W_B` overlap 尺度不匹配”。来源确认前不据此改 `τ/k/p/f_q` 等参数。
