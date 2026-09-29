# BTSP-CAM baseline freeze 前公开协议对齐审计

**总判定：`PUBLIC_PROTOCOL_ALIGNED_EXCEPT_DECLARED_GAPS`。** 本轮逐项对照论文主文、Appendix、固定的 PILOT 源码及正式 reproduction 路径，未发现新的、可由公开论文明确条文判定的 `MISMATCH`。这不表示已经完全复现作者实现或实验数值；下列 `PAPER_UNSPECIFIED`、`PAPER_AMBIGUOUS`、`NEEDS_REVIEW` 边界仍然存在。审计只读，不调参、不训练、不改变正式模型。

## 证据范围与判定口径

- 论文：[BTSP-CAM 原件](../../2233_BTSP_CAM_A_Brain_Inspired.pdf)，尤其 PDF 第 3–7 页的 §3、§4、Algorithm 1、Eq.(3)–(9)、§5.1，以及第 12–18 页 Appendix。页码指 PDF 物理页。
- PILOT 固定提交 `7a6e904c5bc5cb7a4e1823b3434020be27469b63`：`upstream/LAMDA-PILOT/utils/data.py`、`utils/data_manager.py`、`utils/inc_net.py`。
- 正式路径：`configs/btsp_cam_cifar100.yaml`、`src/btsp_cam/*.py`、`experiments/run_btsp_cam_cifar100.py`；实验路径单独放在 `experiments/` 及其配置中。
- `PASS` 仅表示可核实的公式或公开协议对齐，不涵盖参数真实性、数值复刻或作者未公开的实现细节。`NEEDS_REVIEW` 表示现有证据不足以升级为精确一致，也不等于已发现冲突。

| component | paper requirement | current implementation | evidence | status |
|---|---|---|---|---|
| A. 数据集与任务 | CIFAR-100、B0-Inc5、100 类、20 个互斥任务、每任务 5 类 | formal YAML 固定 `cifar224/init_cls=5/increment=5/tasks=20`；runner 校验 50,000/10,000 样本及 20×5 任务 | 论文 §3.1、§5.1；`configs/btsp_cam_cifar100.yaml:5–12`；`experiments/run_btsp_cam_cifar100.py:208–217`；`tests/test_pilot_protocol.py` | PASS |
| A. 类顺序 | 按 seed 1993 打乱；与 PILOT remapped label 语义一致 | `RandomState(1993).permutation(100)`，原标签映射到顺序索引；已对 pinned PILOT `np.random.seed(seed); np.random.permutation` 测试 | 论文 §5.1；`src/btsp_cam/protocol.py:8–36`；`upstream/LAMDA-PILOT/utils/data_manager.py:139–167,213`；`tests/test_pilot_protocol.py` | PASS |
| A. 增量训练、测试与指标 | task t 仅以 `D_t` 更新；无旧样本 replay；推理无需 task ID；在全部已见类上累计评估；`A_B` 为末阶段准确率，`Ā` 为各阶段均值 | formal runner 每轮只取当前 task 训练 feature；调用 `learn_task` 一次；测试拼接全部已见任务；`predict` 仅屏蔽未见类，不接收样本 task ID；均值对各轮 cumulative accuracy 求算术平均 | 论文 §3.1、§5.1；Algorithm 1；`experiments/run_btsp_cam_cifar100.py:267–352`；`src/btsp_cam/model.py:177–290`；`src/btsp_cam/metrics.py:16–17` | PASS |
| A. exemplar-free 的磁盘边界 | 算法不保留/回放旧 exemplar | 模型 checkpoint 不含图像、历史 `h/s/z`；formal 训练在 task t 不读取旧 task 的训练 cache。不过工程性磁盘 cache 保留逐样本 frozen `h`，供重跑/实验诊断使用；若把“不保存”解释成文件系统上绝不留历史逐样本 feature，则不能声称满足这一更严口径 | 论文 §3.1；`src/btsp_cam/model.py:177–211,337–350`；`experiments/run_btsp_cam_cifar100.py:267–307`；`src/btsp_cam/feature_cache.py`；formal YAML `engineering_choices` | NEEDS_REVIEW |
| B. backbone | 冻结 ImageNet-21K 预训练 ViT-B/16，768-D CLS feature | timm 1.0.29 的 `vit_base_patch16_224.augreg_in21k`，`pretrained=True,num_classes=0`、`eval()`、参数 `requires_grad=False`、`inference_mode()`；末层 norm 后 CLS、head 前 768-D | 论文 §3.1、§5.1；`src/btsp_cam/backbone.py:19–72`；`upstream/LAMDA-PILOT/utils/inc_net.py:17–20`；`docs/VIT_FEATURE_OUTPUT_AUDIT.md` | PASS |
| B. 权重文件的精确同一性 | PILOT 使用旧 timm `vit_base_patch16_224_in21k` 的 ImageNet-21K AugReg 来源 | 新旧别名指向同一 Google AugReg NPZ 来源，但当前本机加载 HF safetensors；未将旧 timm 的 NPZ 转换结果与本机权重逐张量比较，故只证明来源/语义兼容，不证明字节级参数同一 | `docs/VIT_WEIGHT_PROVENANCE_AUDIT.md`；`src/btsp_cam/backbone.py:19–40` | NEEDS_REVIEW |
| B. PILOT cifar224 预处理 | non-CODA PILOT train 随机裁剪/翻转/ToTensor；test bicubic Resize256/CenterCrop224/ToTensor | 两套 transform 与 pinned `utils/data.py` 对齐；无额外 `Normalize`；训练增强只用于 train cache，test 确定性；同一冻结 ViT 用于两者 | `src/btsp_cam/cifar100.py:12–35`；`upstream/LAMDA-PILOT/utils/data.py:95–116,121–145`；`docs/CIFAR100_PREPROCESS_PROVENANCE_AUDIT.md` | PASS |
| C. Eq.(3) 随机投影 | `u=Rh+b`，`R,b` 标准正态，固定；`m=8192`；exact top-k 得二值 `s` 且每样本恰好 k 个 1 | 一次初始化 `R∈R^(8192×768)` 与 `b∈R^8192`；`scores=x@R.T+b`；稳定降序取 exact k，输出 uint8 二值；train/test 共用实例，无额外 normalization 或 learned projection | 论文 Eq.(3)、Appendix C；`src/btsp_cam/sparse_encoder.py:9–46`；`src/btsp_cam/model.py:75–88,171–174,202–208` | PASS |
| C. Top-k 平分处理 | 论文未规定 score 精确相等时的 tie-breaking | 本地 stable sort、低索引优先，属于确定性工程选择；Gaussian 连续分布下理论概率为零，但有限精度时并非绝无可能 | `src/btsp_cam/sparse_encoder.py:36–43` | PAPER_UNSPECIFIED |
| D. Eq.(4) eligibility | 每类独立 trace；每个样本先对所有类施加 `(1−λ_c,t)`，仅真类加 `ηs`，之后计算 gate | `values *= 1−decay[:,None]`，`values[label] += eta*s`；`_learn_code` 先 `traces.update` 后 `signals`，非当前类没有标签加项；正全局 η 对初始零 trace 的 cosine 门控是尺度不变的 | 论文 Eq.(4)、Algorithm 1 第 6–7 行；`src/btsp_cam/eligibility.py:29–88`；`src/btsp_cam/model.py:140–166` | PASS |
| D. state-dependent λ 规则 | 符号 `λ_c,t` 为 state-dependent，但未给可唯一恢复的函数/数值 | formal YAML 使用固定 `0.01`，明确标记 `LOCAL_CHOICE`；不能冒充作者规则 | 论文 Eq.(4)；`configs/btsp_cam_cifar100.yaml:57–61` | PAPER_UNSPECIFIED |
| E. Eq.(5) F/N/I | `F=cos(s,e_y)`；`N=1−F`；`I=exp[-β max_{c≠y}cos(s,e_c)]`；`f_eff=f_base·4N(1−N)·F·I` | 顺序、正负号、真类/rival 下标及全部因子相同。零 trace cosine 按 0 处理；未见类的零 trace 不会高于非负 trace/codes 的正相似度 | 论文 Eq.(5)、Algorithm 1 第 7–8 行；`src/btsp_cam/eligibility.py:18–88`；`src/btsp_cam/plateau.py:10–47` | PASS |
| E. Plateau 粒度 | 每个 slot 独立 Bernoulli(`f_eff`) | `rng.random(slots)<probability` 一次产生逐 slot 独立 draw；没有把单个布尔值广播到全部 slot | 论文 Algorithm 1 第 8 行；`src/btsp_cam/plateau.py:44–51`；`src/btsp_cam/model.py:148–151` | PASS |
| F. Eq.(6) binary memory | `W_B` 为 4096×8192 二值；plateau slot 与 active presynaptic bit 交集内，每个 synapse 独立 Bernoulli(p)，XOR 翻转 | 零初始化 uint8 矩阵；`rng.random((selected.size,active.size))`；`weights[np.ix_(selected,active)] ^= flips`，同一规则产生 0→1 和 1→0；无 connectivity mask | 论文 Eq.(6)、Algorithm 1 第 9–11 行；`src/btsp_cam/binary_memory.py:35–74`；formal YAML `memory_slots=4096/code_dim=8192` | PASS |
| F. simpleBTSP 继承边界 | BTSP-CAM Eq.(6) 是 XOR，不是 simpleBTSP set/reset | `official_simplebtsp_step` 仅作参考函数，formal `_learn_code` 调用 `BinaryMemory.update` 的 XOR；没有把 simpleBTSP connectivity mask 带入 CAM | `src/btsp_cam/binary_memory.py:8–74`；`src/btsp_cam/model.py:151–154` | PASS |
| G. Eq.(7) CA1 | `a=W_B s/||s||_1`；`z=max(0,a−τ)`，未写额外 top-k、lateral inhibition 或行/列归一化 | 按每样本 active bit 数求列和并相除，再逐元素 ReLU 减 tau；没有额外竞争/归一化 | 论文 Eq.(7)；`src/btsp_cam/ca1.py:8–32` | PASS |
| H. Eq.(8) ridge 明确部分 | 当前 task `A+=Z_t^T Z_t`、`B+=Z_t^T Y_t`，`W_out=(A+γI)^−1B`；one-hot seen-class label | `A:4096×4096`、`B/W_out:4096×C`；新类扩 B 列；`z.T@z`、`z.T@one_hot`；Cholesky solve；logits `z@W_out` 为 `N×C`。论文 Eq.(9) 印作 `z W_out^T`，与 Eq.(8) 的列方向不相容；代码遵循 Eq.(8) 自洽维度 | 论文 Eq.(8)–(9)、Algorithm 1 第 14–15 行；`src/btsp_cam/ridge.py:9–82`；`src/btsp_cam/model.py:202–211,248–265` | PASS |
| H. evolving `W_B` 与旧 ridge 坐标 | 公开 Eq.(8) 未解释旧 `Z_i` 是否/如何随更新后的 `W_B` 重算 | formal 保留各 task 当时的 A/B；coordinate-aligned C/D 是另一个实验性修复，不是论文实现声明；此项不作 FAIL | 论文 Eq.(8)、Algorithm 1、Appendix F.2；`docs/PAPER_READOUT_SEMANTICS_AUDIT.md`；`experiments/coordinate_aligned_ridge.py` | PAPER_AMBIGUOUS |
| I. RNG 与评估只读性 | class shuffle 可复现；投影/写入 RNG 不应因评估重置；test 不得更新学习状态 | class literal seed1993；SeedSequence 分出 train order、augmentation、projection、plateau、synapse 流；encoder 固定矩阵；`evaluate→predict→ca1` 不调用 trace/update/ridge；模型 checkpoint 保存各 RNG state 以续跑 | `src/btsp_cam/rng.py:21–34`；`src/btsp_cam/model.py:68–100,248–290,309–415`；`experiments/run_btsp_cam_cifar100.py:125–139,301–308` | PASS |
| I. RF/writer 随机数的作者精确值 | 论文明确 class-order seed1993，未逐一公开 NumPy RNG 算法/各子流 seed 与逐样本顺序 | 正式实现从 1993 确定性派生独立 RF/plateau/synapse streams；可复现，不可据此声称随机轨迹与作者逐 bit 相同 | 论文 §5.1；`src/btsp_cam/rng.py:21–34`；formal runner `make_loader` | PAPER_UNSPECIFIED |
| J. 正式与实验路径隔离 | 正式 Eq.(3)–(8) 不能被实验 repair/筛参结果暗中替换 | formal YAML 保留 `k=41,τ=0.5` 与普通 `StreamingRidge`；`coordinate_aligned_ridge` 独立位于 `experiments/`；tau=0、k 搜索使用实验配置；正式 checkpoint 的配置元数据仍为 formal 值 | `configs/btsp_cam_cifar100.yaml:28–74`；`src/btsp_cam/model.py:86–99,202–211`；`experiments/coordinate_aligned_ridge.py`；formal checkpoint metadata | PASS |
| 参数数值来源 | 论文给符号或定性约束，却未公开唯一的 `k,f_base/f_q_base,p,β,λ_c,t,τ,γ` 配置 | formal YAML 分别标记 `PLAUSIBLE_INHERITANCE`、`DERIVED` 或 `LOCAL_CHOICE`；它们不是作者参数证明；实验 k/τ 的 performance 不能反推作者值 | 论文 Eq.(3)–(8)、§5.1、Appendix；`configs/btsp_cam_cifar100.yaml:28–74`；既有参数来源审计 | PAPER_UNSPECIFIED |

## 保留边界，不归入新增 mismatch

1. **Paper-specified and aligned：** 已公开的任务拆分、PILOT 类顺序/预处理、冻结 CLS backbone 语义、Eq.(3)–(8) 形式与 Algorithm 1 的写入顺序，在正式代码中有对应实现。这个结论是公开协议层面的，不是作者实验的数值等价证明。
2. **Paper-unspecified parameters：** `k=41`、`f_base=0.005`、`p=0.5` 等只能保持各自的本地来源标签；`β=1`、固定 `λ=0.01`、`τ=0.5`、`γ=0.1` 是 local choices。固定 λ 不能代表论文 state-dependent λ 规则。不得依据与论文 accuracy 的接近程度更改来源判定。
3. **Paper-ambiguous readout semantics：** Eq.(8) 的累计 A/B 与持续变化的 `W_B` 存在历史 feature-space 坐标问题；论文没有证明或规定对齐操作。Eq.(9) 的转置写法也与 Eq.(8) 导出的矩阵方向不一致；当前 solve/logit 实现遵循 Eq.(8) 的自洽维度。
4. **Experimental repairs / diagnostics：** `coordinate_aligned_ridge`、`tau=0` 与 k 搜索都不属于 formal YAML/default，也不能被称为作者实现。
5. **待复核证据：** 旧 timm Google NPZ 与当前 HF safetensors 尚未逐张量比对；formal 运行保留磁盘逐样本 frozen-feature cache，但学习器不读历史训练 cache、checkpoint 不保存样本。论文没有明确约束离线工程 cache 的物理存在，严格的文件系统级 exemplar-free 主张须另行定义。以上是 `NEEDS_REVIEW`，不是已证实的新增公开协议冲突。

## 完整性与测试记录

审计后使用系统 Python 3.14.5，在当前命令进程中将项目 `src` 与 `experiments` 加入 `PYTHONPATH`，执行 `python -m pytest -q`：**39 passed，3 subtests passed，7.06 s**。最初未设置导入路径的一次尝试在 collection 阶段报 `ModuleNotFoundError: btsp_cam`，没有运行测试；设置路径后完整套件通过，没有为此修改代码或配置。

本轮审计开始前与测试完成后，用 SHA256 对同一组受保护文件逐个比较：**正式 YAML 1/1、`src/btsp_cam/*.py` 15/15、正式 `checkpoint_task*.npz` 20/20 完全一致**。正式 YAML 的本轮前后 SHA256 均为 `B34947EF0D8C705079C426E71A009D6C49FDA6B5AB573772EE5A001E57CFD1FD`。这也是既有冻结记录所述的原 YAML 内容；项目在审计开始时尚无 Git 历史，因此本轮前后哈希只能证明本轮未改动，不能从 Git 证明更早时点的状态。两张用于目视核对 PDF 公式的临时渲染图已从 `tmp/pdfs/` 精确删除；未写入 feature cache、CIFAR 数据或正式 checkpoint。

**Git 交付：** 审计时项目目录原本没有 `.git`，故当时无法执行项目级 status/diff/commit。用户随后提供空的 `RanNanami/BTSP_CAM` 远端并确认建立本地初始历史。现已在本项目初始化 `main`、设置 `origin`（仅配置远端，不推送），先以 `d2db2bbe3e75cc10fcb05e77595d52b3e810f064` 提交 124 个筛选后的项目基线文件：源码、配置、实验脚本、测试及既有文档；`.gitignore` 排除数据、cache、checkpoints、大型 diagnostics、临时文件和独立 upstream 仓库。**本审计文档单独提交**，不与基线混入。上述 Git 操作未改变模型行为、正式 YAML、核心源码或正式 checkpoint；未 push。
