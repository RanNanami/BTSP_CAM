# ViT-B/16 ImageNet-21K 权重来源审计

**门槛结论：COMPATIBLE；四类细分为 `SAME_SOURCE_DIFFERENT_CONVERSION`，而不是已证明 `EXACT_SAME_WEIGHTS`。** 两个 timm 版本的模型配置指向同一 Google AugReg JAX/NPZ 文件；本机 timm 1.0.29 实际走 Hugging Face Hub 的 PyTorch safetensors 移植件。本轮逐张量核对了本机模型与该 safetensors 的全部非分类头参数，但未取得旧版 Google NPZ 与本地 safetensors 作逐张量比较，因此不把“同来源”升级成“旧新版参数逐位相同”。这满足继续进行只读 collision 诊断的**权重来源兼容**门槛，不是论文作者完整实现的字节级复刻证明。

## 论文、PILOT 与版本配置

- BTSP-CAM 论文 PDF 第 6–7 页 §5.1 仅称共享 frozen ViT-B/16、ImageNet-21K 预训练、在 PILOT 上实现；没有给出权重 SHA256、文件名或下载代码。[论文原件](../../2233_BTSP_CAM_A_Brain_Inspired.pdf)
- 固定的 PILOT 仓库 `upstream/LAMDA-PILOT` HEAD 为 `7a6e904c5bc5cb7a4e1823b3434020be27469b63`。`utils/inc_net.py:17–20` 的指定 `in21k` 分支调用 `timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)`；`README.md:139` 列出 timm 0.6.12。注意仓库自带 `exps/simplecil.json:11` 的 CIFAR 示例是非 `in21k` 名称，不可冒充论文实际启动配置。
- [timm v0.6.12 配置](https://github.com/huggingface/pytorch-image-models/blob/v0.6.12/timm/models/vision_transformer.py#L105-L120) 明确注释权重源为 Google JAX ImageNet-21K，并将 `vit_base_patch16_224_in21k` 指向 `https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz`。[_create_vision_transformer](https://github.com/huggingface/pytorch-image-models/blob/v0.6.12/timm/models/vision_transformer.py#L729-L739) 根据 `.npz` 启用自定义加载；[_load_weights](https://github.com/huggingface/pytorch-image-models/blob/v0.6.12/timm/models/vision_transformer.py#L563-L579) 从 Google/Flax NPZ 转换到 PyTorch 参数。
- [timm v1.0.29 配置](https://github.com/huggingface/pytorch-image-models/blob/v1.0.29/timm/models/vision_transformer.py#L1934-L1954) 把 `vit_base_patch16_224.augreg_in21k` 指向**完全相同的原始 NPZ URL**，并增加 `hf_hub_id=timm/vit_base_patch16_224.augreg_in21k`。[旧名映射](https://github.com/huggingface/pytorch-image-models/blob/v1.0.29/timm/models/vision_transformer.py#L4685-L4691) 明确把旧 `vit_base_patch16_224_in21k` 注册名映射到该新名。[timm 维护者的模型卡](https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k) 明称其为由原作者 JAX ImageNet-21K AugReg 权重移植到 PyTorch 的模型。

## 所需元数据逐项核对

| 项目 | timm 0.6.12 / PILOT `in21k` | 当前 timm 1.0.29 |
|---|---|---|
| 名称 | `vit_base_patch16_224_in21k` | `vit_base_patch16_224.augreg_in21k` |
| 原始 URL/文件名 | 上述 Google `B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz` | `pretrained_cfg.url` 中**相同** URL；实际本机优先通过 `pretrained_cfg.hf_hub_id` 取得 safetensors |
| 格式与来源 | Google JAX/Flax NPZ；timm 自定义转换 | timm/HF Hub PyTorch safetensors；模型卡标注从 JAX 权重移植 |
| 训练 recipe | 文件名表明 ImageNet-21K、300 epochs、`lr_0.001`、`aug_medium1`、`wd_0.1`、`do_0.0`、`sd_0.0`；这是**文件名/配置证据**，非本轮重新核验训练日志 | 同一源文件名与模型卡的 AugReg 描述 |
| image size / patch size | 224×224 / 16×16 | 224×224 / 16×16（实例运行时检查） |
| checkpoint 类别数 | 配置 `num_classes=21843` | 配置 `num_classes=21843`，本机 safetensors `head.weight` 形状 `[21843,768]` |
| 实际 feature model 类别数 | PILOT 传 `num_classes=0`，移除 head | reproduction 传 `num_classes=0`，移除 head；实例 `num_classes=0` |
| representation layer | [v0.6.12 工厂说明](https://github.com/huggingface/pytorch-image-models/blob/v0.6.12/timm/models/vision_transformer.py#L943-L951) 指定此 B/16 21K 权重无独立 representation/pre-logits layer | 当前实例不存在 `pre_logits` 模块，`num_features=768` |

`representation_size` 因此不是另一个被选中的有参层：两边均没有需要指定或加载的独立 representation layer。`pretrained_cfg` 的 `mean/std=(0.5,0.5,0.5)` 属于建议输入数据配置，不是权重参数，也不会在 `model.forward` 中自动执行；预处理实况见单独报告。

本机当前模型的完整 `pretrained_cfg` 值（仅重排字段便于阅读）：`url` 为上述 Google NPZ；`hf_hub_id='timm/vit_base_patch16_224.augreg_in21k'`；`architecture='vit_base_patch16_224'`；`tag='augreg_in21k'`；`custom_load=True`；`input_size=(3,224,224)`；`fixed_input_size=True`；`interpolation='bicubic'`；`crop_pct=0.9`；`crop_mode='center'`；`mean=(0.5,0.5,0.5)`；`std=(0.5,0.5,0.5)`；`num_classes=21843`；`pool_size=None`；`first_conv='patch_embed.proj'`；`classifier='head'`；`license='apache-2.0'`。旧版 [默认 `_cfg`](https://github.com/huggingface/pytorch-image-models/blob/v0.6.12/timm/models/vision_transformer.py#L37-L45) 给出同一 NPZ URL、`input_size=(3,224,224)`、`crop_pct=0.9`、bicubic、固定输入、Inception mean/std `(0.5,0.5,0.5)`、`first_conv='patch_embed.proj'`、`classifier='head'`，此模型覆盖 `num_classes=21843`；旧版没有新版的 HF repo、tag、license 等附加注册字段。

## 当前实际加载路径及本机文件证据

`src/btsp_cam/backbone.py:19,36–47` 调用 timm 1.0.29 模型 `pretrained=True, num_classes=0`。本机安装的 timm `_builder._resolve_pretrained_source` 在有 HF Hub ID 且客户端可用时优先选择 `hf-hub`，`load_pretrained` 再加载其 state_dict、删除与 `num_classes=0` 不符的 head。运行时核对与本机缓存一致：

- HF repo：`timm/vit_base_patch16_224.augreg_in21k`；缓存 snapshot：`bb70896cd6bffdd579c3d4f7284f39d0c4ab46c3`。
- 实际文件：`model.safetensors`，410,397,786 bytes；SHA256 `2D31BB5597C720912F8EA5E4F1DE73909A6911798D78AE5D1B9D63A955FED049`。文件自身无 `__metadata__` 项；不能从 safetensors header 独立证明 Google 原始 NPZ 的哈希。
- 文件有 152 个 tensor，其中 `head.weight=[21843,768]`、`head.bias` 是将被移除的分类头；另外 150 个键与实例 `state_dict` 的 150 个键**完全相同**，逐张量最大绝对差 `0.0`，不匹配数 `0`。实例 85,798,656 个参数、patch embedding 权重 `[768,3,16,16]`。

**分类说明：** `EXACT_SAME_WEIGHTS` 要求旧 Google NPZ 经 v0.6.12 转换后与当前所有参数数值逐项一致，本轮没有该直接比较。`DIFFERENT_PRETRAINED_WEIGHTS` 与相同源 URL、别名映射、模型卡和当前加载检查相矛盾。可证的较强结论是**同一上游权重来源、不同存储/转换路径**；极小的移植版本数值差异仍未排除。
