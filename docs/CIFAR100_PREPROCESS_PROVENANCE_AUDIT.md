# CIFAR-100 输入预处理来源审计

**门槛结论：MATCH（对 PILOT 固定提交的非 CODA `cifar224` 路径）。** 论文只给出 CIFAR-100 / PILOT / frozen ViT-B/16 ImageNet-21K，没有公布逐步图像 transform。PILOT 所选择的 `cifar224` 非 CODA 路径训练时为 `RandomResizedCrop(224, scale=(0.05,1.0), ratio=(3/4,4/3)) → RandomHorizontalFlip(0.5) → ToTensor`，测试时为 `Resize(256, bicubic) → CenterCrop(224) → ToTensor`，没有 `Normalize`。当前 reproduction 的显式管线与此一致；两张缓存对应图的 patch embedding 输入钩子证明实际是原始 `[0,1]` 像素，不是 ImageNet mean/std 标准化后的 tensor。

## 论文公开范围

BTSP-CAM PDF 第 6–7 页 §5.1 指定 CIFAR-100、PILOT 与 frozen ImageNet-21K ViT-B/16，但未列出 Resize、crop、ToTensor、Normalize 或 mean/std。全文关键词检索的 `normalize/normalization` 命中来自其他理论与表示段落，不是 CIFAR 输入图像 transform；`Resize`/`Crop`/`mean/std` 未命中。因此**论文独立没有指定**这些预处理细节，以下 `MATCH` 是对 pinned PILOT 实际代码路径的判断。[论文原件](../../2233_BTSP_CAM_A_Brain_Inspired.pdf)

## 从 CIFAR 原始图到 ViT patch embedding 的源码链

| 环节 | PILOT commit `7a6e904c5bc5cb7a4e1823b3434020be27469b63` | 当前 reproduction |
|---|---|---|
| CIFAR 数据源 | `utils/data.py:121–145` 的 `iCIFAR224` 用 `torchvision.datasets.CIFAR100`，取其原始 uint8 图像 | `src/btsp_cam/cifar100.py:59–67` 的 `torchvision.datasets.CIFAR100` |
| 分支选择 | `utils/data.py:127–136`：仅 `model_name == 'coda_prompt'` 走另一个 transform；普通 BTSP-only / frozen ViT 对照走 `build_transform`，`common_trsf=[]` | `build_pilot_cifar224_transform(train)`，无 CODA 分支 |
| TRAIN | `utils/data.py:95–107`：`RandomResizedCrop(224, scale=(0.05,1.0), ratio=(3/4,4/3))`、`RandomHorizontalFlip(p=0.5)`、`ToTensor()` | `src/btsp_cam/cifar100.py:17–25` 完全同序同参数 |
| TEST | `utils/data.py:109–118`：`Resize(256, interpolation=3)`、`CenterCrop(224)`、`ToTensor()`；torchvision 常量 `3` 对应 bicubic | `src/btsp_cam/cifar100.py:27–32`：`Resize(256, InterpolationMode.BICUBIC)`、`CenterCrop(224)`、`ToTensor()` |
| Compose / wrapper | `utils/data_manager.py:40–56,140–156` 用 `[*train/test_trsf, *common_trsf]` 构成 `transforms.Compose`；`DummyDataset.__getitem__` 在 `utils/data_manager.py:191–208` 对 `Image.fromarray` 施加它 | `IndexedRemappedSubset.__getitem__` 于 `src/btsp_cam/cifar100.py:53–56` 取 torchvision 已变换图；`src/btsp_cam/backbone.py:51–65` 的 `extract_loader` 直接执行 `self.model(images.to(device))` |
| cache 管线 | PILOT 此分支无本项目缓存 | `experiments/run_btsp_cam_cifar100.py:84–112,137–161` 把已定义的 dataset/loader 特征缓存；`src/btsp_cam/feature_cache.py:76–84` 缓存/提取未添加 transform；test cache 元数据注明 `PILOT_RESIZE_CENTER_CROP_TO_TENSOR` |

### “隐藏 normalization”逐处排除

- `iCIFAR224` 的 `common_trsf=[]`，因此没有在 train/test transform 后追加 `Normalize`。PILOT 的**其他 dataset**（如 32×32 CIFAR-100、CUB 等）或 CODA 分支确有 `Normalize`，不能把它们误合并到 `cifar224` 非 CODA 路径。
- PILOT `utils/inc_net.py:17–20,606–614` 在 backbone 创建/forward 间没有 processor、`resolve_data_config`、`create_transform` 或显式图像标准化。
- 当前 `src/btsp_cam/cifar100.py` 的 `Compose` 和 `src/btsp_cam/backbone.py:59` 之间没有别的 tensor 转换。当前 timm 的 `forward_features` 从 `patch_embed(x)` 开始；`pretrained_cfg.mean/std=(0.5,0.5,0.5)` 是**若调用** `timm.data.resolve_model_data_config` / `create_transform` 才会用的建议配置，不会在 `model.forward` 中自动应用。当前及上述 PILOT 路径均没有这些调用。[当前 timm 模型调用链](https://github.com/huggingface/pytorch-image-models/blob/v1.0.29/timm/models/vision_transformer.py#L1154-L1220)。HF 模型卡的**示例代码**主动调用 `create_transform`，不能当作 PILOT 真正执行了该示例的证据。[模型卡](https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k)
- `ToTensor()` 对 PIL uint8 RGB 图的结果为 `[0,1]`、像素值按 `1/255` 网格量化；本轮钩子复验符合这一机制。

## 已有 test cache 对应样本的只读实测

从 `data/feature_cache_btsp_cam/test_task_01.npz` 的原始 CIFAR 索引 `8,27` 取测试图，按当前正式 test transform 走一次 `PilotFrozenViT(pretrained=True)`；`patch_embed` 的 forward-pre-hook 捕获真正的第一层输入，未写入缓存：

| 指标 | 两样本合并的 patch_embed 输入 |
|---|---:|
| shape | `[2,3,224,224]` |
| min | `0.01960784` |
| max | `0.95686275` |
| mean | `0.64827466` |
| population std | `0.24040473` |
| 与 DataLoader 图像 tensor 逐元素一致 | `True` |
| `x*255` 到最近整数的最大偏差 | `0.0` |
| 重算 768D feature vs 正式 cache 最大绝对差 | `1.3113e-5` |

单样本 `(min,max,mean,std)` 分别为 `(0.02352941,0.95686275,0.62584919,0.27826238)` 与 `(0.01960784,0.94901961,0.67070025,0.19275141)`。即实际输入符合 **A. raw `[0,1]` ToTensor**，不符合经过 ImageNet mean/std 的输入。TRAIN 路径由源码核对，未声称复原了缓存训练图的随机 crop RNG。

额外用当前环境直接调用 pinned PILOT 的 `build_transform(False, {'model_name':'simplecil'})`，与当前 `build_pilot_cifar224_transform(False)` 处理同一原始 CIFAR 测试图（索引 8）：两个 `[3,224,224]` tensor **逐位相同**、最大绝对差 `0.0`。打印的 PILOT 训练 transform 也确认为 bilinear `RandomResizedCrop`、`RandomHorizontalFlip(0.5)`、`ToTensor`，无标准化。这项运行时交叉检查不替代作者原始 torchvision 版本的逐像素证明，但支持当前实现与 pinned PILOT 源码定义的等价性。

## 门槛与限制

与 PILOT `cifar224` 非 CODA 路径**没有预处理 mismatch**，所以可以继续只读 Appendix C collision 检验。此结论不替代论文未公开的作者实际运行配置；尤其 PILOT 仓库自带 `exps/simplecil.json` 的默认 CIFAR 配置并非论文 0+5/ImageNet-21K 组合。没有修改正式模型或应用任何 normalization 修复。
