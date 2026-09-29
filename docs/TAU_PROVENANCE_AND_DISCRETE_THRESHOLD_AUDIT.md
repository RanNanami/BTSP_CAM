# BTSP-CAM τ 来源与离散 overlap 阈值审计

日期：2026-09-28。范围：来源核对、精确离散推导、既有文件可用性检查。未改正式 reproduction、代码或默认配置；未训练/重训 `W_B`，未以论文 accuracy 反推 τ。**k=128/160 的 current-WB replay oracle 未运行**，原因及不能回答的实证问题见第 5 节。

## 1. 来源与术语边界

首要来源：[BTSP-CAM 原论文 PDF](C:/Users/苒苒/Documents/学委/类脑/BTSP/2233_BTSP_CAM_A_Brain_Inspired.pdf)（18 页，包含 Appendix A–F）；继承来源：[simpleBTSP 原论文 PDF](C:/Users/苒苒/Documents/学委/类脑/BTSP/s41467-024-55563-6.pdf)（*Nature Communications*, 2025，18 页）、[其本地 Supplement](C:/Users/苒苒/Documents/学委/类脑/BTSP/supplement.pdf)（12 页）和[本地上游代码](C:/Users/苒苒/Documents/学委/类脑/BTSP/BTSP-CAM-reproduction/upstream/simpleBTSP)。[simpleBTSP 期刊原文](https://www.nature.com/articles/s41467-024-55563-6)可用于核对原论文；未把其他论文的 mV、Ca²⁺ 阈值或稀疏比例换算成 BTSP-CAM 的无量纲 `τ`。

| 来源与页码/section | 必要短引文/公式 | 参数所指 | 生物实验值？ | grid-search/model 参数？ | 具体数值公开？ |
|---|---|---|---|---|---|
| BTSP-CAM p.5 Algorithm 1，第 2 行 | `..., {λc,t}, τ, γ` | `τ` 被列为算法超参数 | 否 | 是，但此处无选择规则 | **否** |
| BTSP-CAM p.6 §4.5 Eq.(7) | `z(x)=max(0,a(x)−τ)` | CA1 **归一化 overlap** 的检索阈值 | 未声称 | 模型参数 | **否** |
| BTSP-CAM p.12 Appendix B，Table 6 | “Retrieval threshold for CA1 competition layer” | 同上 | 未声称 | 模型参数 | **否** |
| BTSP-CAM p.13 Appendix C，Cor. C.2 Eq.(14) | “maximum pairwise collision ... threshold τ” | 两个稀疏码的**整数 overlap count** 容忍阈值；不是 Eq.(7) 中 `a∈[0,1]` 的数值设定 | 否 | 理论容忍参数 | **否** |
| BTSP-CAM pp.14–15 Appendix D.1，Eqs.(19)–(20) | “activation ... threshold τk”; `p>τ` | one-shot 写入后的 overlap count `J` 与 `kτ` 比较；与 Eq.(7) 同尺度 | 否 | 理论符号/条件 | **否** |
| BTSP-CAM p.17 Appendix F.1，Eqs.(31)–(34) | `z=max(0,a−τ)`; “trade-off between sensitivity and specificity” | recall 的归一化 overlap 阈值 | 未给映射 | 模型参数 | **否** |
| simpleBTSP p.4，memory-trace setup | `f_p=0.005`; “fraction of 1’s in ... input pattern” | CA3/input 码中 active bits 的比例 | 基于文中引用的生物稀疏率估计 | 模型输入密度 | **是，0.005；不是 τ** |
| simpleBTSP pp.5–6，Fig. 2 与正文 | `f_q=0.005`; “probability ... of a plateau potential” | 学习期间 memory/CA1 神经元获得 plateau 的概率 | 由实验 plateau 发生率和塑性窗口推算 | 模型 gating 概率 | **是，0.005；不是 τ** |
| simpleBTSP p.4，memory-trace setup | “not aware of reliable experimental data on the firing threshold” | CA1 memory neuron 的 weighted-input firing threshold | **明确说无可靠值** | 随后用 grid search 定 | 未给跨模型统一固定值 |
| simpleBTSP p.16，Methods / Fig. 2 | “selected the optimal `v_th` by minimizing ... HD ratio” | partial-cue recall 的 neuron threshold | 否 | **是**；33% masking、30,000 patterns 的准则 | 依实验条件求得，不是 0.005 |
| simpleBTSP p.16，Methods / Fig. 5 | “various thresholds within the range [1, 100]” | input/memory neurons 的 firing thresholds | 否 | **是**；按 HD 最小并拟合函数 | 给搜索范围/拟合系数，不是 BTSP-CAM τ |
| simpleBTSP Supplement p.6，Fig. S1 | “core BTSP, with `f_q=0.0025`” | core BTSP 变体的 plateau 概率；非 firing threshold | 否 | 模型 gating 概率 | **是，0.0025；不是 τ** |
| simpleBTSP 本地代码 `main_BTSP_feedforward_model.py` 行 25、34、95 | `fq=0.005`; `fp=p=0.005`; `vth...0.45` / “optimized by grid search” | plateau、input density 与 recall threshold 为不同变量；此处代码变量 `p` 是 `fp` 的别名，不能当 BTSP-CAM bit-flip `p` | 否 | `vth` 为任务相关选择 | 代码给该实验的设定，**不**给 BTSP-CAM τ |

**符号陷阱。** BTSP-CAM Appendix C 的 Eq.(14) 复用 `τ` 表示稀疏码两两交集 `s_aᵀs_b` 的 count tolerance（量纲是 bit count）；Eq.(7) 则用 `τ` 截断 `W_Bs/k`（无量纲，0–1）。不能把 Appendix C 中的 `τ` 当作公布的 CA1 数值。Appendix D.1 虽把 `τk` 与 count 对齐，也没有给出 `τ` 的具体设置；Eq.(19) 的 `≥τk`/CDF 下标在**整数边界**与 Eq.(7) 的严格 `>` 支持条件存在表述差异，本报告的离散表一律以 Eq.(7) 的 `z>0` 定义为准。

BTSP-CAM 本地 PDF 已全文检索 `τ/tau/threshold/grid search/hyperparameter/validation/sparsity/activation/overlap`，并逐页核对 Eq.(7)、Table 6、Appendix C、D.1、F.1；**未找到 CA1 τ 数值、τ 的 validation/grid-search 规则或生物测量到 τ 的映射**。simpleBTSP 的 `f_p`、`f_q` 与 `v_th` 在原文和 Methods 中有明确定义，但没有论证 `v_th` 与 BTSP-CAM 归一化 `τ` 的数值恒等。

公开版本核查：[OpenReview 论文入口](https://openreview.net/forum?id=2HlW5mj6Ch)及其 PDF 入口在本环境被浏览器验证页拦截，revision / 额外 supplement **不可检查**；不能据此断言它们不存在。BTSP-CAM 论文 p.1 声明的 [GitHub 地址](https://github.com/ericzhengz/BTSP-CAM) 当前只读访问返回 404，**author code could not be inspected**；不能推断作者无代码或其代码内容。项目中的 `supplement.pdf` 是 **simpleBTSP** 的 Supplement，不是 BTSP-CAM 的附加版本。

**问题“0.005 是否有证据是 BTSP-CAM τ 的生物来源？”— NO。** 两个有来源的 0.005 属于 `f_p`（输入稀疏率）与 `f_q`（plateau gating 概率）；原文反而说明 CA1 firing threshold 无可靠实验值，并用 recall 准则选择。无证据把任一 0.005 数值迁移到 BTSP-CAM Eq.(7)。注意 `f_q≈0.005` 是 simpleBTSP 根据约 `0.0005/s` plateau rate × `10 s` 塑性窗口构造的有效概率，不是归一化 overlap 门槛。

## 2. τ 的精确离散语义

对 binary `W_B` 与 exact-k binary `s`，每个 memory slot 的交集数 `J=w_jᵀs` 是整数 `0…k`，`a_j=J/k`。由 Eq.(7)：

```text
z_j = max(0, J/k − τ)
z_j > 0  ⇔  J > kτ
被截掉的整数 J: 0 ≤ J ≤ floor(kτ)  （0 ≤ τ ≤ 1；tau=0 时 J=0 本已为零）
```

下表每格为 `kτ / 截掉的最大 J=q / 是否改变正值 support`。最后一项“是”表示**数学上可能**截掉 `J≥1`，不保证给定 checkpoint 确有该 count。

| τ | k=41 | k=128 | k=160 |
|---:|---:|---:|---:|
| 0 | 0 / 0 / 否 | 0 / 0 / 否 | 0 / 0 / 否 |
| 0.001 | 0.041 / 0 / 否 | 0.128 / 0 / 否 | 0.160 / 0 / 否 |
| 0.0025 | 0.1025 / 0 / 否 | 0.320 / 0 / 否 | 0.400 / 0 / 否 |
| 0.005 | 0.205 / 0 / 否 | 0.640 / 0 / 否 | 0.800 / 0 / 否 |
| 0.00625 | 0.25625 / 0 / 否 | 0.800 / 0 / 否 | 1 / 1 / 是 |
| 0.0078125 | 0.3203125 / 0 / 否 | 1 / 1 / 是 | 1.25 / 1 / 是 |
| 0.01 | 0.410 / 0 / 否 | 1.28 / 1 / 是 | 1.60 / 1 / 是 |
| 0.02 | 0.820 / 0 / 否 | 2.56 / 2 / 是 | 3.20 / 3 / 是 |
| 0.05 | 2.05 / 2 / 是 | 6.40 / 6 / 是 | 8 / 8 / 是 |
| 0.1 | 4.10 / 4 / 是 | 12.8 / 12 / 是 | 16 / 16 / 是 |
| 0.2 | 8.20 / 8 / 是 | 25.6 / 25 / 是 | 32 / 32 / 是 |
| 0.5 | 20.5 / 20 / 是 | 64 / 64 / 是 | 80 / 80 / 是 |

因此 `τ=0.005` 对 `k=41/128/160` 的**support 均与 τ=0 完全一样**：所有 `J≥1` 仍然非零；只把每个正值从 `J/k` 减去 0.005。它可能改变幅值与重新拟合的 ridge 表现，但不能产生 binary-support sparsification。最小能**在存在 J=1 时**改变正值 support 的正阈值是 `1/k`：k=41 为 `0.0243902439`，k=128 为 `0.0078125`，k=160 为 `0.00625`。真正对实测矩阵产生非零密度下降的门槛仍须看其 `J` 直方图；若 J=1 缺失，实际最小值更高。

## 3. 用 overlap-count q 比较不同 k

定义离散级别 `q=floor(kτ)`；在整数边界可直接取 `τ=q/k`，那么 `J≤q` 被截断、`J≥q+1` 保留。跨 k 比较“截掉几个 synaptic overlaps”应固定 `q` 而非固定 `τ`。同一 `τ=0.02` 在 k=128 截掉 `J≤2`，在 k=160 截掉 `J≤3`。固定 q 只让 **support 截断级别**等价，不能保证 `z=(J−q)/k` 的幅值、`J` 分布或 accuracy 等价。

| q | τ=q/128 | τ=q/160 | 截断的正 overlap counts |
|---:|---:|---:|---|
| 0 | 0 | 0 | 无 |
| 1 | 0.0078125 | 0.00625 | J=1 |
| 2 | 0.015625 | 0.0125 | J=1,2 |
| 3 | 0.0234375 | 0.01875 | J=1…3 |
| 4 | 0.03125 | 0.025 | J=1…4 |
| 5 | 0.0390625 | 0.03125 | J=1…5 |
| 8 | 0.0625 | 0.05 | J=1…8 |
| 12 | 0.09375 | 0.075 | J=1…12 |

这只是**预先定义的 diagnostic 级别**，不表示建议修改正式 τ 或在测试集上选最优 q。

## 4. 生物学边界

`f_p` 是输入/CA3 活跃比例；`f_q` 是学习期间 plateau-gated 写入机会；`v_th` 是 simpleBTSP memory neuron 的 weighted-sum firing threshold；BTSP-CAM `τ` 是 `J/k` 上的无量纲 cutoff。膜电位阈值（mV）、Ca²⁺ signal 检测阈值、place-field/active-cell sparsity、plateau rate 均不是同一量纲或同一事件。simpleBTSP p.4 明确承认缺少可靠 CA1 firing-threshold 实验值，p.16 的 `v_th` 由 partial-cue recall/HD 准则确定，而 BTSP-CAM 没有从该 `v_th` 经 normalization 映射到其 `τ` 的公式。**NO DIRECT BIOLOGICAL MAPPING FOUND。**

## 5. k=128/160 current-WB replay oracle：未运行及原因

用户指定的 oracle 必须输入各 k 的**已存在 `W_B` 轨迹**，再用历史训练 cache 在每个 current `W_B` 下重编码 `z_τ`、重新解 ridge；`τ>0` 时 sparse `C/D` 的线性闭合不再成立，不能拿 coordinate-aligned 解代替 replay oracle。本轮核查了项目的 `.npz/.npy/.pt/.pth/.ckpt` 文件及实验脚本：`results/btsp_cam_cifar100/checkpoint_task*.npz` 和 `diagnostics/wb_checkpoints/WB_task*.npz` 是正式 `k=41`；[`run_k_robustness_and_selection.py`](C:/Users/苒苒/Documents/学委/类脑/BTSP/BTSP-CAM-reproduction/experiments/run_k_robustness_and_selection.py) 的 k=128/160 分支保存逐任务 CSV 指标，**不保存 `W_B` checkpoint**。在项目目录未找到可读取的 k=128 或 k=160 `W_B` 矩阵；旁边 `simpleBTSP` 项目的 Split-MNIST 状态不属于该 CIFAR100 轨迹。

因此，严格按“使用各自已有轨迹、只读 replay、不重新训练 W_B”的边界，**不能计算**指定 q 扫描的 cumulative/old/current accuracy、CA1 density、sample nonzero fraction、`J` 分布、`z` 幅值或 same/different margin。已有 [`K_ROBUSTNESS_AND_SELECTION.md`](C:/Users/苒苒/Documents/学委/类脑/BTSP/BTSP-CAM-reproduction/docs/K_ROBUSTNESS_AND_SELECTION.md) 的 `τ=0` k=128/160 accuracy 不能推算正 τ oracle。为了不制造数据，本轮没有输出虚构的 replay 表，也没有由别的 k 外推结论。

**实证问题 7 的状态：UNRESOLVED。** 尚不能说 positive threshold 在 k=128/160 的 current-WB oracle 下既明显 sparsify 又不损害 classification，也不能说它一定损害。数学上最早可能改变 support 的 τ 分别为 `0.0078125` 和 `0.00625`；实际密度下降和功能收益仍须有对应 `W_B` 轨迹才能测量。

## 6. 逐项回答

1. **BTSP-CAM 是否公开 τ 数值？** 本地原论文及附录未找到；可访问的作者代码也无法核对。不能把当前 reproduction 的 τ 当论文值。
2. **τ 是否有明确生物来源？** 未找到。
3. **simpleBTSP 的 0.005 是什么？** 分别可指 `f_p` 输入稀疏率和 `f_q` plateau 概率；**不是** firing threshold `v_th`，也没有证据等于 BTSP-CAM τ。
4. **τ=0.005 的作用？** k=41/128/160 的 `kτ` 分别是 `0.205/0.64/0.8`，均只截掉已为零的 `J=0`，不改变 support，但改变正值幅度。
5. **为何比较 q？** support 由整数 `J>kτ` 决定；相同 τ 跨 k 截掉不同数量的 overlap counts。
6. **真正开始 sparsify 的最小 τ？** 若存在 `J=1`：k=128 为 `1/128=0.0078125`；k=160 为 `1/160=0.00625`。这是数学候选，不是已测实证门槛。
7. **positive τ 能否稀疏化而不损害分类？** **未能验证**：指定 k 的已保存 `W_B` 轨迹缺失，不能遵守只读约束完成 oracle。
8. **真实 CA1 数据到模型 τ 的直接映射？** **NO DIRECT BIOLOGICAL MAPPING FOUND。**
