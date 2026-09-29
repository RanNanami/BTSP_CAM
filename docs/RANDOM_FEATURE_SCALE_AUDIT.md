# Eq.(3) 随机特征尺度与 backbone normalization 来源审计

审计范围：本地 `2233_BTSP_CAM_A_Brain_Inspired.pdf` 全文，重点为 PDF 第 3–4、6–8、12–14 页（Eq.(3)、Eq.(10)、Appendix B/C），以及项目内固定版本的 PILOT 源码和当前 frozen ViT 提取路径。本文只记录来源，不把任何离线反事实当作论文隐藏实现。

| 审计项 | 标记 | 证据与边界 |
| --- | --- | --- |
| `h=ϕ(x)∈R^d` 是 frozen PTM 的连续特征 | PAPER_SPECIFIED | 论文第 3 页 ``3.1`、第 4 页 `4.1`、第 12 页 Table 6。没有在 `ϕ` 定义中附带单位范数条件。 |
| `h←h/‖h‖₂` 或其他输出 feature normalization | NOT_FOUND | 第 4 页 Eq.(3)、第 13 页 Eq.(10)、Appendix B/C 与第 6–7 页实验细节均未规定。全文出现的 “normalization” 指 CA1 的 `W_B s/‖s‖₁` 或类重叠归一化，不是输入 `h`。 |
| `‖h‖₂=1`、有界范数、各维零均值、各向同性或全局中心化 | NOT_FOUND | 第 12–14 页 Appendix B/C 对 `h` 仅写 `R^d`；没有这些条件。不能因为引用 Johnson–Lindenstrauss 就补出这些前处理。 |
| Eq.(3) 投影与稀疏化 | PAPER_SPECIFIED | 第 4 页：`u=Rh+b`，`s=I(u∈Top-k(u))`，`R∈R^{m×d}`、`b∈R^m` 冻结。第 13 页 Eq.(10) 用 `W_in` 重述同一映射。 |
| `R` 与 `b` 的初始化分布 | PAPER_SPECIFIED | 第 4 页写两者从 standard normal draws；第 13 页写 `N(0,1)`。按元素标准正态理解，无额外缩放。 |
| `1/√d`、`1/√m` 或按样本范数缩放 `Rh` | NOT_FOUND | Eq.(3)/(10) 未写，实验细节和 Appendix C 也未补充。把它当成实现会改变 `b` 相对于 `Rh` 的尺度。 |
| `b∼N(0,1)` 与 `Rh` 必须同尺度、为何这样选 | NOT_FOUND | 论文给出 `b` 的分布但没有解释其理论角色或尺度匹配条件。它是公式中的固定加性偏移，这是从公式可直接读出的数学作用，标记 DERIVED；不能说作者明确以此消除 winner monopoly。 |
| Appendix C 碰撞模型 | PAPER_SPECIFIED | 第 12 页 `B.1` 与第 13 页 `C.2`：对于“不相关”概念，**编码后** active index sets 被**建模为独立、均匀的 k-subsets**；`k≪m` 且写有 `k=o(√m)` 的稀疏假设。Lemma C.1 的超几何重叠以此为前提。 |
| 上述均匀独立 active-set 模型由固定 Gaussian `R,b` 加 exact top-k 推出 | NOT_FOUND | 附录没有证明这一连接。它是局部 Hamming-code 统计模型，不是对任意 frozen PTM 特征分布的定理。 |
| Appendix C 隐含 unit-norm 或 centered `h` | NOT_FOUND | 文字没有这样的前提；**均匀独立 active sets** 是明确的另外一种假设。unit-norm 既非文中条件，也不足以单独保证固定 `R,b` 下的均匀 winner 频率。 |
| PILOT 的 CIFAR224 图像变换 | PILOT_BEHAVIOR | 固定 PILOT `utils/data.py:96–139`：非 CODA 分支训练为 crop/flip/ToTensor，测试为 resize/crop/ToTensor，`common_trsf` 为空；没有图像 mean/std normalize。图像输入归一化与 `h` 的 L2 归一化是不同问题。 |
| PILOT frozen ViT 输出后的 L2 / 全局中心化 | PILOT_BEHAVIOR | `utils/inc_net.py:12–20` 使用 `timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0).eval()`；此路径没有显式 `F.normalize` 或减全局特征均值。项目 `src/btsp_cam/backbone.py` 直接取 `self.model(images)`。 |
| ViT 内部 LayerNorm、`pre_logits` 与输出点 | PILOT_BEHAVIOR | 当前项目对应的 timm 1.0.29 alias `vit_base_patch16_224.augreg_in21k`，本机实例化核对：`norm=LayerNorm`、`global_pool=token`、`fc_norm=Identity`、`head=Identity`、无独立 `pre_logits` 属性，输出 768-D。该 **内部网络 LayerNorm** 不等于输出向量 `h/‖h‖₂`，也不保证跨样本均值为零；它是当前别名的代码行为，不是论文指定的额外后处理。 |

## Appendix C 的理论适用范围

第 12 页 `B.1` 和第 13 页 `C.2` 明确先把无关样本的编码支持集当作独立、均匀的 `k`-subsets，再得到重叠超几何模型与 `E[s_a^⊤s_b]=k²/m`。这是 **PAPER_SPECIFIED 的建模假设**。从 `R_i∼N(0,I)` 可推得：对一个固定 `h`，不同投影行的 `R_i h` 边际同分布；但同一组 `R,b` 一旦冻结，跨样本共享的投影行与非零特征均值会造成持久的 bit-specific 偏移。论文没有证明编码后支持集对固定实现仍独立均匀。因此 Appendix C 不应被读作“任意 raw ViT `h` 都自动满足均匀碰撞模型”。

从第 4 页 Eq.(3) 与第 13 页 Eq.(10) **DERIVED**：若仅把 `R` 乘正数而保持 `b` 不动，会改变 `Rh`/`b` 的相对尺度和 top-k；若 `R`、`b` 同乘相同正数，top-k 排名不变。论文没有选择前一种缩放，也没有阐明 `b` 与实际 PTM feature norm 如何匹配。

## 审计结论

论文**没有明确 feature L2 normalization 或 centering**，也没有明确或暗含 `‖h‖=1` 的数学前提。它**明确**规定了标准正态 `R,b` 和一个编码后的均匀独立支持集近似，但未证明真实 frozen ViT 特征经固定投影后满足该近似。PILOT 路径同样不提供输出 feature L2 后处理；当前 timm ViT 内部有 LayerNorm，但不等价于 L2/unit norm。于是，当前 raw feature 与 `b` 的尺度关系是需实测的开放实现问题，而非可由论文补全的隐藏参数。
