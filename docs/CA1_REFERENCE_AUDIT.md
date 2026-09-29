# CA1 相关引用文献审计（2026-09-23）

## 审计原则

以 BTSP-CAM PDF 第 9–11 页 References 为清单，筛选与 CA1、海马检索、pattern separation、稀疏记忆、plateau／BTSP 直接相关的条目，并核对可公开阅读的原始论文或出版社页面。**BTSP-CAM 在某处引用一篇文献，不等于那篇文献给出了 Eq.(7) 或 `τ`。** 下表的“位置”指 BTSP-CAM PDF 页码；“机制”只列与当前 CA1 公式审计有关的内容。

| 被引论文（作者，年） | BTSP-CAM 引用位置 | 该来源实际支持的相关机制 | 是否给 BTSP-CAM Eq.(7) / `τ` |
|---|---|---|---|
| [*A simple model for Behavioral Time Scale Synaptic Plasticity (BTSP) provides content addressable memory with binary synapses and one-shot learning*](https://www.nature.com/articles/s41467-024-55563-6)（Yujie Wu, Wolfgang Maass，2025） | 第 2–4、7 页，simpleBTSP 的直接祖先 | CA3→CA1 二值突触、plateau-gated 写入、以输入加权和与 firing threshold 比较的二值 memory trace、部分线索检索；其文中明确说 CA1 firing threshold 缺乏可靠生物学数值，实验采用 grid search 选阈值。 | **没有**给 BTSP-CAM 归一化连续 `z=max(0,a-τ)` 的数值 `τ`；也不支持把其 raw-count 阈值系数直接当作新模型 `τ`。 |
| [*Behavioral time scale synaptic plasticity underlies CA1 place fields*](https://pmc.ncbi.nlm.nih.gov/articles/PMC7289271/)（Katie C. Bittner, Aaron D. Milstein, Christine Grienberger, Sandro Romani, Jeffrey C. Magee，2017） | 第 2–3 页 | CA1 plateau 相关的秒级 BTSP 与单次经历形成 place field，是生物学写入动机。 | **没有**提供二值 CAM 的 overlap 归一化、winner、具体阈值。 |
| [*Bidirectional synaptic plasticity rapidly modifies hippocampal representations*](https://elifesciences.org/articles/73046)（Aaron D. Milstein, Yiding Li, Katie C. Bittner, Christine Grienberger, Ivan Soltesz, Jeffrey C. Magee, Sandro Romani，2021） | 第 2–3 页 | plateau 诱发的、依原有突触权重而定的 LTP/LTD，可重塑 CA1 表示。 | **没有**规定 BTSP-CAM 的 Eq.(7) 或 `τ`。 |
| [*Behavioral Time Scale Plasticity of Place Fields: Mathematical Analysis*](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2021.640235/full)（Ian Cone, Harel Z. Shouval，2021） | 第 2 页 | 秒级 eligibility traces 与 plateau instructive signal 的数学模型；对应写入侧的时间桥接。 | **没有**规定二值 CAM 的 CA1 readout／阈值。 |
| [*Pattern separation in the hippocampus*](https://pmc.ncbi.nlm.nih.gov/articles/PMC3183227/)（Michael A. Yassa, Craig E. L. Stark，2011） | 第 7 页 §6.2，解释 LTD 的几何 repulsion | 海马 pattern separation 的概念与证据；该综述特别把强分离与 DG/CA3 联系起来，而非给 CA1 一个固定 top-k competition 算子。 | **没有**给 Eq.(7) 或阈值。不可把概念性“pattern separation”解释成代码必须加入 CA1 lateral inhibition。 |
| [*Neural networks and physical systems with emergent collective computational abilities*](https://pmc.ncbi.nlm.nih.gov/articles/PMC346238/)（John J. Hopfield，1982） | 第 8 页 §6.4，partial-cue / attractor 类比 | 联想记忆及 attractor 检索的理论背景。 | **没有**给 BTSP-CAM 的逐槽归一化 overlap、CA1 threshold 或 ridge。 |
| [*Sparse Distributed Memory*](https://mitpress.mit.edu/9780262514699/sparse-distributed-memory/)（Pentti Kanerva，1988） | 第 4 页 §4.1 sparse-code/CAM 几何；第 8 页 §6.2 repulsion 讨论 | 稀疏分布式记忆／内容寻址的背景。此项核对到出版社图书信息，而非逐章重审该书。 | **没有可据以移植**到 Eq.(7) 的 `τ` 数值、行/列归一化或 CA1 winner 规则。 |
| *Why there are complementary learning systems in the hippocampus and neocortex*（James L. McClelland, Bruce L. McNaughton, Randall C. O’Reilly，1995）及 *Complementary learning systems*（Randall C. O’Reilly, Rajan Bhattacharyya, Michael D. Howard, Nicholas Ketz，2014） | 第 2–3 页，海马快速学习／皮层巩固背景 | 支持双系统的总体动机；与 Eq.(7) 的逐槽数值定义没有直接承继关系。此处仅核对 BTSP-CAM 的引用上下文与 References，未对原文逐节展开审计。 | **不能**作为 Eq.(7) 细节或 `τ` 数值的来源。 |

## 可追溯的机制谱系

`CA1 plateau / 秒级 BTSP`（Bittner、Milstein、Cone） → `二值突触、随机写入、阈值触发二值 CA1 firing / CAM`（Wu & Maass） → `随机稀疏编码 + 按 ||s||₁ 归一化的连续 CA1 overlap + 逐槽减 τ + streaming ridge`（BTSP-CAM 自身第 4–6、17 页）。

尤其值得注意：Wu & Maass 原文把 CA1 memory neurons 建模为**几乎没有横向连接的独立阈值神经元**，并指出对大规模 CA1 假定 winner-take-all 运算有问题。这支持“阈值 firing 是先例”，但**不支持**为修复当前输出稀疏度而在 CA1 擅自补一个 lateral-inhibition/top-k 竞争层。

这是**论文谱系上的机制对应**，不是说前两组文献严格推导了 BTSP-CAM Eq.(7)。特别是 simpleBTSP 的生物／计算模型把 firing threshold 视作需要针对其任务设定的阈值；没有从这些引用中发现可直接继承的 `τ=0.5` 或额外 CA1 正规化。论文第 4 页 Eq.(3) 的 **k-WTA 是编码阶段**，不应误归因于上述生物参考文献作为 CA1 第二次 winner selection。

## 与本次问题无关、未作为 CA1 公式证据的引用

Johnson–Lindenstrauss 被引于第 4 页随机投影，Hoerl–Kennard 被引于第 6 页 ridge，PILOT／RanPAC／SimpleCIL 等提供实验框架或 CIL 基线。这些不能填补 CA1 阈值信息。第 4 页 Jiang et al. 与 Li et al. 出现在稀疏编码的文献串中，但 BTSP-CAM 没有从它们引用一个 CA1 归一化或抑制公式，故不把它们列为 `τ` 来源。
