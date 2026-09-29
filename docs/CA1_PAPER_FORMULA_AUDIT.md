# CA1 论文公式审计（只读，2026-09-23）

## 范围与证据标记

审计对象是项目所附的 18 页 `2233_BTSP_CAM_A_Brain_Inspired.pdf`（主文及同一 PDF 的 supplementary material）、`upstream/simpleBTSP` 和当前实现。逐页检查了正文、算法、表格、附录与参考文献；检索了 CA1、competition、threshold、tau/τ、`a(x)`、`z(x)`、`W_B s`、`||s||_1`、normalization、sparsity、ReLU、winner、top-k、activation，并检查是否存在 lateral inhibition、softmax、额外行/列归一化等 CA1 操作。这里的 **NOT_FOUND** 仅指所审 PDF／指定源码内未找到，不能外推为所有未审的公开材料均不存在。

- **PAPER_SPECIFIED**：BTSP-CAM PDF 明确写出。
- **DERIVED**：可由论文公式与约束直接推出；不是论文另给的数值。
- **NOT_FOUND**：在指定 PDF 中没有定义或数值。
- **CODE_FACT**：当前实现的事实，不冒充论文事实。

## Eq.(7) 到底规定了什么

| 审计项 | 状态 | 证据与边界 |
|---|---|---|
| `a(x)=W_B s / ||s||_1` | PAPER_SPECIFIED | 第 6 页 §4.5 Eq.(7)；第 17 页 Appendix F.1 Eq.(31) 又按槽写为 `a_j=w_j^T s/||s||_1`，称为 normalized overlap，且 `a∈[0,1]^S`。 |
| `z(x)=max(0,a(x)-τ)` | PAPER_SPECIFIED | 第 6 页 Eq.(7)；第 17 页 Appendix F.1 重述；这是逐槽的减阈值截断，输出为非负**连续值**，不是只保留二进制 firing。 |
| `τ` 是 CA1 retrieval threshold | PAPER_SPECIFIED | 第 5 页 Algorithm 1 第 2 行列为超参数；第 12 页 Appendix B Table 6 明确释义；第 17 页 Appendix F.1 说非零 `z_j` 表示检索成功。 |
| `τ` 的数值、搜索范围、估计规则、是否逐任务/逐槽变化 | NOT_FOUND | 第 6–7 页 §5.1 只给 RF 维数 8192 和 4096 个 memory slots；Table 6 只定义名称。正文、图表及附录没有 CA1 `τ` 的数值或校准协议。Eq.(7) 写作单一 `τ`，但没有明确说明实现时是否固定。 |
| 对 `a` 再做行归一化、列归一化、标准化或重缩放 | NOT_FOUND | Eq.(7)、Eq.(31) 的唯一显式归一化是除以输入代码的 `||s||_1`；不能从 “normalizes” 一词推导出其它操作。 |
| 在 CA1 再做 top-k / winner selection / lateral inhibition / softmax / cosine normalization | NOT_FOUND | 第 4 页图 1 的 `k-WTA` 位于 frozen feature → sparse code；第 4 页 Eq.(3) 和第 13 页 Eq.(10) 也是编码层。第 6 页 Eq.(7) 与第 17 页 Appendix F.1 没有第二次 top-k。 |
| CA1 “competition” 的算子 | PAPER_SPECIFIED | 第 6 页 §4.5 把 Eq.(7) 称为 CA1-like competitive layer；第 8 页 §6.3/Table 4 用文字说 competition “normalizes and sparsifies projected codes”，但未另外给出算子。按可执行数学式，competition 是归一化 overlap 加逐槽 threshold。 |
| `W_B∈{0,1}^{S×m}`，`s∈{0,1}^m` 且 `||s||_0=k` | PAPER_SPECIFIED | 第 12 页 Appendix B Table 6；第 4 页 Eq.(3) 给 binary top-k；第 13 页 Eq.(10) 重述。 |
| 对 exact-k 二值 `s`，`||s||_1=k` | DERIVED | 由二值与恰好 `k` 个 1 推出。论文的除数写的是 `||s||_1`，代码用实际活跃位数；在此设置下等于 41，而不是 `sqrt(k)` 或 `m`。 |

### 同一符号 `τ` 的注意事项

第 13–14 页 Appendix C.2 的 Corollary C.2 还把 `τ` 用作**稀疏码两两碰撞容忍阈值**（Eq.(14)–(17)），其对象是未归一化的 `s_a^T s_b`；那里没有给数值，也不能把它当成 CA1 Eq.(7) 的数值来源。第 14–15 页 Appendix D.1 和第 17 页 Appendix F.1 则明确讨论 CA1 检索阈值。

## 论文内部的尺度约束

1. **PAPER_SPECIFIED**：第 14–15 页 Appendix D.1 Eq.(19)–(20) 从初始全零槽、一次 plateau、每个活跃突触以概率 `p` 写入出发，得到成功写入数 `K_active ~ Binomial(k,p)`；论文明确说当 **`p>τ`** 且 `k` 足够大时，单次检索成功概率趋近 1。这是该理想化场景的**充分条件**，不是任意数据分布下的必要条件或通用调参法。
2. **CODE_FACT**：当前配置 `p=0.5`、`τ=0.5`，所以**不满足严格的 `p>τ`**。此事实只能说明不能援引论文的这一可靠性保证来支持本地参数；不能证明论文作者实际使用了什么 `p` 或 `τ`。
3. **PAPER_SPECIFIED / IDEAL ASSUMPTION**：第 17 页 Appendix F.1 Eq.(32)–(33) 在理想 `w_j≈s*` 下令正槽 overlap 约为 1、非相关槽 overlap 远小于 `τ`。它没有证明按 Eq.(6) 的随机单次写入必达到 `w_j≈s*`，也没有给出符合这一区间的数值阈值。

## 当前代码对公式的逐项映射

| 项目 | 当前实现 | 对照判断 |
|---|---|
| 投影后代码 | `src/btsp_cam/sparse_encoder.py:17-18,27,35-42` 生成高斯 `R,b`，稳定排序选恰好 `k` 位，`uint8` 0/1 | Eq.(3) 的二值 exact-k 部分一致；并列分数处理是代码本地选择。 |
| 矩阵形状与取值 | `src/btsp_cam/binary_memory.py:30-35` 建立 `(slots,code_dim)` 的 `uint8` 零矩阵；配置为 `4096×8192`；更新后保持二值 | 与 Table 6 的 `S×m` 及 §5.1 的维数一致。 |
| overlap | `src/btsp_cam/ca1.py:20-29` 对每个样本求当前活跃列的逐行和，再除以该样本的 `active=sum(s)` | 代数上等于 `W_B s / ||s||_1`；`k=41` 时分母为 41。没有 `sqrt(k)`、`m`、row norm 或 column norm。 |
| threshold | `src/btsp_cam/ca1.py:29` 计算 `max(0,activation-tau)`；`model.py:170-175` 调用 | 与 Eq.(7) 一致。`τ=0.5` 来自配置文件 `configs/btsp_cam_cifar100.yaml:64-67` 的 **LOCAL_CHOICE**，不是论文指定值。 |
| 每 task 的调用阶段 | `src/btsp_cam/model.py:198-211` 先处理当任务所有写入，随后以此刻 `W_B` 构造本任务 `Z_t`，更新 ridge | 与 Algorithm 1 的两阶段顺序一致。旧任务统计在以后任务写入后不重算；Eq.(8) 同样只保存充分统计量，但论文没有提供补偿 `W_B` 变化的进一步公式。此点与本轮首任务 threshold collapse 不是同一问题。 |

**公式一致性结论：** 在被审计的 `s → a → z` 路径上，未发现把 Eq.(7) 写错的代码证据。缺的是 `τ` 的公开数值/选择协议，而不是论文遗漏了第二个明示的 CA1 归一化式。不能仅凭当前 `a` 的分布推定论文暗中用了行归一化或 top-k。

## simpleBTSP 继承关系（源码级）

| mechanism | simpleBTSP（附带 upstream 代码） | BTSP-CAM（PDF／当前实现） | source |
|---|---|---|---|
| binary synapse | `W` 为二值矩阵，并乘随机连接 mask `W_mask` | `W_B∈{0,1}^{S×m}`；当前 `uint8` 二值矩阵 | upstream `main_BTSP_feedforward_model.py:109,123-128`；PDF 第 3–5、12 页；`binary_memory.py:30-35` |
| 写入及 LTP/LTD | upstream 用互斥的 LTP、LTD plateau 事件，并对活跃输入位置执行 set/reset | Eq.(6) 以每突触 Bernoulli(`p`) 对 0/1 进行 XOR 翻转；当前实现逐活跃位独立翻转 | upstream `main_BTSP_feedforward_model.py:114-126`；PDF 第 4–5 页 Eq.(2),(6)，第 14 页 Eq.(18)；`binary_memory.py:54-67` |
| plateau | 随机 plateau；示例 `fq=0.005`，LTP/LTD 分支各半 | 按类别 eligibility、F/N/I 自适应 `f_q^eff`，再逐槽采样 plateau | upstream `main_BTSP_feedforward_model.py:25,106-119`；PDF 第 5 页 Eq.(4)–(6) |
| memory recall / CA1 | 对 `W^T x` 的**原始计数**作阈值判断，返回二值 firing；阈值系数在代码注释中标为经 grid search 优化 | 对 `W_B s/||s||_1` 作 `max(0,a-τ)`，返回连续非负 `z`，再用 ridge 分类 | upstream `main_BTSP_feedforward_model.py:95,165-169`；PDF 第 6 页 Eq.(7)–(8) |
| 其它 CA1 competition | upstream 没有作为另一个归一化/选择层出现 | 第 6 页 Eq.(7) 把归一化 overlap 与减阈值叫作 CA1-like competition；未另给 second winner/top-k | upstream 同上；PDF 第 4 页图 1、第 6 页 Eq.(7)、第 17 页 Eq.(31) |

所以：**阈值化检索的思想继承自 simpleBTSP；`τ=0.5` 并非继承值。** BTSP-CAM 将原来的 raw-count 二值 recall 改成按输入活跃数归一化的连续阈值特征，并增加 ridge。若把 “CA1 competition” 理解成一个比 Eq.(7) 更多的抑制／winner 机制，那是未获该 PDF 支持的扩展。
