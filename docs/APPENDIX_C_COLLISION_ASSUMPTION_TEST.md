# Appendix C 随机支持集假设与真实 sparse code 的只读检验

**结论：`STRONGLY_VIOLATED`（当前 Task1–5 编码相对于 independent uniform k-subsets 模型）。** 这是**理论模型与当前编码数据之间的 model-data gap**，不等于“论文理论错误”。Appendix C 在其明确的独立、均匀、固定 `k` 支持集前提下，超几何分布和期望值的概率计算成立；真实 frozen-ViT + 固定 Gaussian 投影 + exact top-k 的 winner 频率显著不均，导致不同类跨 task 的平均碰撞数是均匀模型的 7.70 倍。

本测试的前置门槛已通过：[权重来源审计](VIT_WEIGHT_PROVENANCE_AUDIT.md) 判定 `COMPATIBLE`（同源但旧 NPZ 与当前 safetensors 未作逐张量比较），[CIFAR 预处理审计](CIFAR100_PREPROCESS_PROVENANCE_AUDIT.md) 判定相对于 PILOT 非 CODA `cifar224` 路径 `MATCH`；上一轮 [输出点审计](VIT_FEATURE_OUTPUT_AUDIT.md) 判定 `EQUIVALENT_WITH_VERSION_DIFFERENCE`。没有改变模型、`R,b,k`、训练缓存或 W_B，也没有重新训练。

## 理论事实和内部推导范围

BTSP-CAM PDF 第 13–14 页 Appendix C.1–C.2 的 Eq.(10) 定义固定投影和 top-k，随后**额外假设**两个不相关样本的支持集是独立、均匀的 `k`-subset。固定第一个集合后，第二个在 `m` 位中抽 `k` 位，其中第一个集合的 `k` 位为“成功”位，因此交集大小 `J` 的精确分布为

`P(J=j) = C(k,j) C(m-k,k-j) / C(m,k)`，即 `Hypergeometric(m,k,k)`；`E[J]=k²/m`，`Var(J)=k(k/m)(1-k/m)(m-k)/(m-1)`。在 `m=8192,k=41` 下，理论平均 `0.2052001953`、方差 `0.2031761313`。[论文原件](../../2233_BTSP_CAM_A_Brain_Inspired.pdf)

Appendix C 的 Eq.(11)/(13) 写出 `exp[-m D_KL(t/m || (k/m)²)]` 的上界；论文从“超几何近似 `Binomial(k,k/m)`”直接跳到这个 **`m`-trial** 表达，中间缺一项说明。该上界本身可以独立成立：对 `λ>0`，超几何 `J` 的矩母函数是 `m` 个权重（`k` 个为 `e^λ`，其余为 `1`）的第 `k` 个初等对称平均；Maclaurin 不等式使之不大于 `(1-p+p e^λ)^k`，其中 `p=k/m`。再令 `c=e^λ-1`，对数凹性给出 `k log(1+pc) ≤ m log(1+p²c)`，即被 `Binomial(m,p²)` 的矩母函数界住；Chernoff 优化便得到 Eq.(11)（`t>E[J]`）。因此这里指出的是**论文展示的推导步骤省略**，不是在所述概率模型下发现反例。`t=1…5` 的精确理论尾概率分别为 `0.18634642, 0.01771719, 0.00108709, 0.0000478399, 0.00000160248`；Eq.(11) 对应的上界分别为 `0.454295, 0.0633405, 0.00523273, 0.000307716, 0.0000140535`，均大于精确尾概率。C.2 的 union-bound 容量推论仍**以这个随机支持集模型为前提**，不能自动推广到下述真实编码。

## 测试协议与可重复性

- 数据：正式 `data/feature_cache_btsp_cam/train_task_01.npz` 至 `train_task_05.npz`，共 25 类、每类 500 个样本、每 task 2,500 个，合计 12,500 个缓存 ViT 特征。训练缓存代表既有实验每图一次的固定增强；没有重新采样或重跑 ViT。
- 编码：读取 `results/btsp_cam_cifar100/checkpoint_task5.npz` 中的正式 `projection` 与 `bias`；它们与 seed=1993 推导的固定 encoder 数组逐位一致。调用**原** `RandomSparseEncoder.encode`，每行恰有 `k=41` 个 1。逐 task 的全部 8,192 个 bit 频次与现有 `diagnostics/sparse_bit_frequency.csv` 逐项一致，且重复得到 1,227 个 zero-winner bit。
- 配对：**精确枚举全部不重复样本对**，没有 Monte Carlo 或 pair sampling。组别分别为：same-class 3,118,750 对；different-class same-task 12,500,000 对；different-class cross-task 62,500,000 对；合计 78,118,750 对。最后一组是 Appendix C“unrelated inputs”最接近的可用代理，但不同 CIFAR 类也不保证严格语义无关或独立。
- hotspot 反事实：从上述原始编码的全体 Task1–5 训练样本频次中取最高的 **82/8192 位**（平频按 bit ID 稳定排序），仅在计算 `J` 时忽略这些位；**不重新 top-k、不训练新编码器、不改变原样本和正式模型**。这一选择使用全部五 task，属于事后归因诊断，不是可部署的在线方法。删位后每码有效 `k` 不再固定为 41，所以 CSV 中该模式不套用原来的超几何理论列。
- 输出：[完整分布](../diagnostics/sparse_collision_distribution.csv) 含三个组、原始/删位两种模式、`J=0…41` 的完整计数与 PMF、理论 PMF/比值、均值/方差及 `P(J=0),P(J≥1…5)`；[task 对照](../diagnostics/collision_by_task_pair.csv) 含 10 个跨 task 对、两种模式的摘要。两份 CSV 的 count 与解析概率检查通过；跨 task 均值还用现有逐 task bit 频次的点积作独立计算，最大误差为 `0.0`。

## 主要数值

| Population | pairs | mean `J` | variance | `P(J=0)` | `P(J≥1)` | `P(J≥2)` | `P(J≥3)` | `P(J≥4)` | `P(J≥5)` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Uniform independent theory | — | 0.205200 | 0.203176 | 0.813654 | 0.186346 | 0.017717 | 0.001087 | 0.00004784 | 0.000001602 |
| Same class, original | 3,118,750 | 3.361502 | 7.394447 | 0.099024 | 0.900976 | 0.728754 | 0.544794 | 0.388273 | 0.268535 |
| Different class, same task, original | 12,500,000 | 1.593486 | 2.474141 | 0.262459 | 0.737541 | 0.430194 | 0.218714 | 0.104751 | 0.049822 |
| Different class, cross task, original | 62,500,000 | 1.580174 | 2.402517 | 0.261967 | 0.738033 | 0.428618 | 0.215737 | 0.101550 | 0.047411 |
| Different class, cross task, ignore top 82 | 62,500,000 | 0.660510 | 0.827249 | 0.547491 | 0.452509 | 0.146298 | 0.042141 | 0.012555 | 0.004179 |

同 task 不同类均值是理论的 **7.77×**；跨 task 不同类是 **7.70×**。跨 task 的 `P(J≥1)` 是理论的 **3.96×**，`P(J≥2)` **24.19×**，`P(J≥3)` **198.45×**，`P(J≥5)` **29,585.81×**。same-class 的高碰撞预期包含真实类别相似性，不宜把其 16.38× 比值作为“无关样本模型失效”的首要证据；结论主要依托不同类两组。

## 高频 bit 对 excess collision 的归因

最高频 82 位占原始编码全部激活的 `19.7497%`，显著大于其 `82/8192≈1.00%` 的维度占比。对跨 task 不同类配对，忽略这 82 位后平均 `J` 从 `1.580174` 降至 `0.660510`，差 `0.919663`，即**原总碰撞的 58.20%**。均匀模型对这 82 位的原本期望贡献为 `82(41/8192)²=0.002054`，所以它们贡献的超额均值为 `0.917609`；全部超额是 `1.580174−0.205200=1.374973`，该位组解释 **66.74% 的均值超额**。这是一项线性 overlap 的精确分解，不等于认定其余 33.26% 全是另一单一原因。删位后 `P(J≥3)` 从 `0.215737` 到 `0.042141`，下降 `80.47%`；这个非线性尾概率变化不可像均值那样分解为独立贡献。

## Task-pair evolution

十个不同类跨 task 对的原始 mean `J` 范围为 **1.410973（Task4–5）至 1.761090（Task1–2）**，均明显超过均匀理论 `0.205200`。固定 Task1 与后续任务的均值依次为 Task2 `1.761090`、Task3 `1.591025`、Task4 `1.485817`、Task5 `1.616631`：**没有随新 task 序号单调增长**。存在 task 配对差异，但从这些数字不能推出时间导致的 collision growth；encoder 与 `R,b` 固定，配对差异还受任务所含类别组成影响。逐对原始与删位的 tail 统计见 `collision_by_task_pair.csv`。

## 判断边界

1. **Appendix C 内部概率计算：** 对独立均匀固定 `k`-subset，超几何 PMF、均值、方差成立；Eq.(11) 可用额外矩母函数比较论证为上界，但论文省略了从 `k`-trial 到 `m`-trial 的说明。并集上界的适用前提不因真实编码而自动得到满足。
2. **真实编码与假设：** 在已核对的当前 ViT 权重来源、PILOT 图像路径及正式 encoder 下，跨 task 不同类 `J` 的均值及高阶尾概率强烈高于均匀模型，故判 `STRONGLY_VIOLATED`。精确配对 census 中的 pair 不是独立统计样本，本报告只给描述性比值，不做伪独立显著性检验。
3. **热点解释力：** 82 个高频位解释约 66.74% 的跨 task 不同类**均值超额**，但删去后仍有平均 `0.660510` 的 overlap；固定投影在真实 ViT 特征上的其余不均衡与类别结构也可能参与。
4. **时间演化：** 未见单调增长。这里观察的是冻结 encoder 的各 task 类别配对分布，不是 W_B memory 随任务写入造成的漂移。

本报告只诊断 Eq.(10) 的稀疏编码碰撞与 Appendix C 的模型假设，不评价或调整 W_B、CA1 或分类器。
