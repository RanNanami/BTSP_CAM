# ViT feature `h` 输出点审计（只读）

**判定：B. EQUIVALENT_WITH_VERSION_DIFFERENCE。** 对论文所称的 frozen ViT-B/16 ImageNet-21K 和 PILOT 中相应的 `in21k` 分支，当前 reproduction 的 `h` 都是 **transformer 最终 LayerNorm 后的 CLS token、分类 head 前的 768D 向量**。两边都通过 `backbone(x)` / `model(x)` 取得该向量；timm 版本和模型注册名不同，但这个输出点的运算语义相同。此判定限于 **feature output point**，并非声称论文作者未公开的完整运行脚本、所有权重字节或整套实验完全一致。

审计没有修改正式模型、参数或缓存，没有重新训练；仅对已有缓存中的两张固定 CIFAR-100 测试图做了只读 forward。

## 证据边界：论文与 PILOT

- 论文 PDF 第 4 页 §4.1 将 `h=φ(x)` 称为 backbone feature；第 6–7 页 §5.1 指定共享 frozen ViT-B/16、ImageNet-21K 预训练、PyTorch/PILOT，但**没有逐个写明 CLS、pool、`forward_head` 或 `pre_logits` 接口**。因此精确输出点需要由其引用的 PILOT/timm 路径判定，不能仅由论文文字推出。[论文原件](../../2233_BTSP_CAM_A_Brain_Inspired.pdf)
- 本地 PILOT `upstream/LAMDA-PILOT` 的 HEAD 已核对为 `7a6e904c5bc5cb7a4e1823b3434020be27469b63`。PILOT README 第 139 行列出 timm 0.6.12。
- PILOT `utils/inc_net.py:10–20`：`pretrained_vit_b16_224_in21k` / `vit_base_patch16_224_in21k` 分支调用 `timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)`，置 `out_dim=768`、`eval()`。`num_classes=0` 使 timm 的原 21K 分类 head 成为 `Identity`。
- `utils/inc_net.py:289–309` 的 `BaseNet.extract_vector`（vit 分支）和 `BaseNet.forward` 都调用 `self.backbone(x)`；后者才交给额外的 `self.fc`。`utils/inc_net.py:606–614` 的 `SimpleVitNet.extract_vector` 也直接返回 `self.backbone(x)`；其 `forward` 先取 backbone 输出，然后才可选执行 RanPAC 的 `relu(x @ W_rand)` 并送入 classifier。
- 消费端交叉验证：`models/simplecil.py:27–49` 的 prototype 使用 `model.backbone(data)`；`models/ranpac.py:31–50` 先用 `model.extract_vector(data)` 取得 768D embedding，然后才做自身的随机映射。因而这里的基础 `h` **不是** RanPAC 映射后的特征，也不是额外分类器 logits。
- **配置限制：** PILOT 自带 `exps/simplecil.json:11` 的 CIFAR 默认配置实际上写的是 `pretrained_vit_b16_224`（非 `in21k`），且是 10+10；不可把该默认 JSON 当作论文 0+5、ImageNet-21K 实验的逐字运行配置。这里比较的是 PILOT 已实现的 `in21k` 分支与论文明确指定的 backbone；作者是否使用了另一个未公开脚本无法由仓库证明。

## timm 0.6.12 调用链

在 [timm v0.6.12 原始实现](https://github.com/huggingface/pytorch-image-models/blob/v0.6.12/timm/models/vision_transformer.py#L339-L437) 中，此模型的默认值为 `global_pool='token'`、`embed_dim=768`、`class_token=True`、`pre_norm=False`、`fc_norm=None`。`use_fc_norm = (global_pool == 'avg')`，所以本分支 `norm=LayerNorm`、`fc_norm=Identity`、`head=Identity`（PILOT 传入 `num_classes=0`）。[对应模型工厂](https://github.com/huggingface/pytorch-image-models/blob/v0.6.12/timm/models/vision_transformer.py#L943-L951)还明确注明 **无独立的 representation / pre-logits layer**；`pre_logits=True` 是 `forward_head` 的返回位置标志，不是另一个投影层。

[timm v0.6.12 `forward_features` / `forward_head` / `forward`](https://github.com/huggingface/pytorch-image-models/blob/v0.6.12/timm/models/vision_transformer.py#L493-L512) 的实际链为：

`x → patch_embed → pos_embed → norm_pre(Identity) → transformer blocks → norm(LayerNorm) → [B,197,768] → x[:,0] (CLS) → fc_norm(Identity) → head(Identity) → [B,768]`。

所以 `forward_features(x)` **本身不是**最终的 768D `h`；它仍是 token 序列。最终 `h = forward(x) = forward_head(forward_features(x), pre_logits=True) = norm(blocks(...))[:,0]`。它经过 transformer 最终 LayerNorm，但没有额外 L2 normalization、mean pooling、fc-norm、representation layer 或 21K 分类 head。

## 当前 timm 1.0.29 / reproduction 对照

当前 `src/btsp_cam/backbone.py:19,36–47,51–64` 使用 `vit_base_patch16_224.augreg_in21k`、`pretrained=True, num_classes=0`、`eval()`，并直接把 `self.model(images)` 的 `[B,768]` 输出写入缓存。没有手动调用 `forward_features` 后另取 patch 均值，也没有在缓存前取 pre-LayerNorm token。

| 环节 | PILOT / timm 0.6.12 | 当前 / timm 1.0.29 | 结论 |
|---|---|---|---|
| 模型注册名 | `vit_base_patch16_224_in21k` | `vit_base_patch16_224.augreg_in21k` | 新版[显式旧名映射](https://github.com/huggingface/pytorch-image-models/blob/v1.0.29/timm/models/vision_transformer.py#L4685-L4691)；两个配置指向相同 [旧版 URL](https://github.com/huggingface/pytorch-image-models/blob/v0.6.12/timm/models/vision_transformer.py#L118-L120) / [新版 URL](https://github.com/huggingface/pytorch-image-models/blob/v1.0.29/timm/models/vision_transformer.py#L1951-L1954) `B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz` |
| `forward_features` | 最终 `norm` 后 `[B,197,768]` | 最终 `norm` 后 `[B,197,768]` | 输出 token 序列语义一致；新版多了默认 `patch_drop=Identity` 分支 |
| pooling | `global_pool='token'`，取 `x[:,0]` | `global_pool='token'`，经 `pool()` 取 `x[:,0]` | 都是最终归一化的 CLS，不是 mean pooled patch tokens |
| 最终 norm | `norm=LayerNorm`；`fc_norm=Identity` | 实例化检查：`norm=LayerNorm`；`fc_norm=Identity` | 无第二次特征 norm；内部 LayerNorm 不等于输出 L2 单位化 |
| `pre_logits` | `forward_head(..., pre_logits=True)` 在 head 前返回 CLS；无独立 representation layer | 同样在 head 前返回 CLS；实例无独立 `pre_logits` 模块 | 含义一致 |
| classifier | `num_classes=0 → head=Identity` | `num_classes=0 → head=Identity`，`head_drop=Dropout(p=0)` 且 eval | 两者 `forward(x)` 均返回 768D 特征而非分类 logits |

新版调用链见 [timm v1.0.29 `forward_features` 至 `forward`](https://github.com/huggingface/pytorch-image-models/blob/v1.0.29/timm/models/vision_transformer.py#L1154-L1220)；默认 token pooling、final norm 与 head 构造见[构造函数](https://github.com/huggingface/pytorch-image-models/blob/v1.0.29/timm/models/vision_transformer.py#L673-L705)及[模块创建](https://github.com/huggingface/pytorch-image-models/blob/v1.0.29/timm/models/vision_transformer.py#L829-L853)。本机实例化也直接核对到 `global_pool=token`、`norm=LayerNorm`、`fc_norm=Identity`、`head=Identity`、`head_drop=Dropout(p=0)`、`pre_logits` 模块不存在。

## 已有 cache 的静态验证

数据来自 `data/feature_cache_btsp_cam/test_task_01.npz` 的前两条固定索引 `8, 27`；元数据注明 timm `1.0.29`、上述新版模型名、768D、PILOT 确定性测试变换。用现有 CIFAR-100 文件及现有 `PilotFrozenViT` 只读重算，未写缓存。对照表中的差异是与正式 cache `h` 的最大绝对差，cosine 按两样本顺序列出：

| 当前模型输出点 | shape | max abs diff vs cache | cosine vs cache | 两样本向量范数 |
|---|---:|---:|---:|---:|
| `model(x)` | `[2,768]` | `1.3113e-5` | `≈1.000000, ≈1.000000` | `43.69669, 43.14605` |
| `forward_features(x)[:,0]` | `[2,768]` | `1.3113e-5` | `≈1.000000, ≈1.000000` | `43.69669, 43.14605` |
| `forward_head(forward_features(x), pre_logits=True)` | `[2,768]` | `1.3113e-5` | `≈1.000000, ≈1.000000` | `43.69669, 43.14605` |
| 最终 LayerNorm **之前**的 CLS | `[2,768]` | `20.1970` | `0.5565, 0.5333` | `28.00275, 27.52256` |
| 最终 LayerNorm 后的 **patch-token 均值** | `[2,768]` | `19.6354` | `0.4213, 0.4681` | `53.38283, 51.51360` |

缓存前两条的范数为 `43.69670, 43.14606`。小的 `1.31e-5` 差异与重算设备/运算数值误差相容；它与错误输出点的约 20 的量级差异清楚分开。测试只覆盖两条确定性样本，足以核对 cache 的输出路径，但不构成所有缓存条目或 timm 两版参数张量逐元素相同的证明。

## 最终判断与停止边界

**B. EQUIVALENT_WITH_VERSION_DIFFERENCE（针对 `h` 输出点）。** 当前 `h` 并非误取了未归一化 token、patch 平均、21K logits 或另一 `pre_logits` representation。此轮没有发现“取错 ViT feature output point”的证据，因此现有 winner imbalance 不能据此归因为输出点错误；同时，本审计也**不**证明论文作者未公开实现中的全部预处理或实际权重加载路线与本地完全相同。没有执行后续 collision 实验，也未据此调整 encoder、normalization 或任何模型参数。
