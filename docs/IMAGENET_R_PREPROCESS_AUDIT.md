# ImageNet-R preprocessing audit

Pinned LAMDA-PILOT commit: `7a6e904c5bc5cb7a4e1823b3434020be27469b63`. `upstream/LAMDA-PILOT/utils/data.py:208-222` constructs non-CODA `iImageNetR` transforms through `build_transform` at lines 95-120. `DataManager.get_dataset` composes these with an empty `common_trsf` (`utils/data_manager.py:33-57`).

| Operation | Train | Test |
| --- | --- | --- |
| Resize/crop | `RandomResizedCrop(224, scale=(0.05,1.0), ratio=(3/4,4/3))`; default PIL bilinear | `Resize(256, interpolation=3)` (bicubic), then `CenterCrop(224)` |
| Horizontal flip | `RandomHorizontalFlip(p=0.5)` | None |
| Tensor | `ToTensor()`; nominal float32 [0,1] | Same |
| Normalize | None | None |

`PilotFrozenViT` in `src/btsp_cam/backbone.py` is reused without an ImageNet-R-specific forward path. It calls timm's model with `num_classes=0`, final LayerNorm CLS 768-D output. PILOT's older name is `vit_base_patch16_224_in21k`; the project uses the modern equivalent `vit_base_patch16_224.augreg_in21k` (see `docs/CIFAR100_PROTOCOL.md`).

The experimental cache stores one seeded train augmentation per image and task and deterministic test features. This is the existing CIFAR/CUB engineering policy, **LOCAL_OPERATIONALIZATION**, not an explicitly documented BTSP-CAM ImageNet-R augmentation policy. No normalization is added.
