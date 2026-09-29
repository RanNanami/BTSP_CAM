# CIFAR-100 B0-Inc5 protocol

## Evidence levels

### PAPER_SPECIFIED

- CIFAR-100, frozen ViT-B/16 pretrained on ImageNet-21K.
- `B0-Inc5`, shuffled class order with seed 1993.
- Exemplar-free; train task `t` can access only `D_t`; cumulative seen-class evaluation without task ID.
- RF dimension 8192 and 4096 memory slots.

### PILOT_PROTOCOL

Pinned reference: official PILOT commit `7a6e904c5bc5cb7a4e1823b3434020be27469b63`.

- Operational B0-Inc5: `init_cls=5`, `increment=5`, producing 20 real stages. `DataManager` cannot have an empty first stage (`utils/data_manager.py:9-20`).
- Dataset identifier: `cifar224`; memory size: 0; shuffle: true; seed: 1993. Official EASE/MOS configs independently use the same 5/5/0/1993 combination.
- Class order: PILOT calls `np.random.seed(seed)` then `np.random.permutation(100)` (`utils/data_manager.py:156-157`). Runtime matches it with a private legacy `RandomState` so unrelated global calls cannot perturb it.
- Original labels are remapped to incremental indices through `order.index(label)` (`utils/data_manager.py:184-185`). Task 1 labels therefore become 0-4, task 2 become 5-9, and so on.
- Training set at stage `t`: only the new incremental label interval. Test set: labels 0 through the latest seen class (`models/simplecil.py:59-67`).

## Exact class order

```text
[68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50, 28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96, 98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69, 36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33]
```

`tests/test_pilot_protocol.py` creates an actual official `DataManager` against a fake 100-class dataset and performs an elementwise parity assertion. It also asserts 20 disjoint tasks of five classes.

## Actual cifar224 transforms

From `utils/data.py:96-116,121-146` for a non-CODA method:

**Train**

1. `RandomResizedCrop(224, scale=(0.05,1.0), ratio=(3/4,4/3))`
2. `RandomHorizontalFlip(p=0.5)`
3. `ToTensor()`

**Test**

1. `Resize(256, interpolation=BICUBIC)`
2. `CenterCrop(224)`
3. `ToTensor()`

There is **no normalization** in `iCIFAR224.common_trsf`. Tensor shape is `[3,224,224]`, dtype float32, nominal range `[0,1]`. This is PILOT behavior even though timm's checkpoint metadata advertises mean/std `(0.5,0.5,0.5)`.

## Backbone provenance

- PILOT code: `timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)` and manually records output dimension 768 (`utils/inc_net.py:17-20`).
- PILOT dependency documentation pins timm 0.6.12.
- In timm 0.6.12 this name resolves to Google's `B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz`.
- Current timm 1.0.29 renamed the same URL/checkpoint to `vit_base_patch16_224.augreg_in21k`. The project uses this explicit modern alias because timm 0.6.12 cannot import under Python 3.14.
- Model is in `eval()` mode, every parameter has `requires_grad=False`, input resolution is 224, and output dimension is 768. Repeated extraction of one identical eval tensor was bit-identical (`max_abs_diff=0`).

Evidence status: architecture is `PAPER_SPECIFIED`; concrete timm constructor/checkpoint and transforms are `PILOT_PROTOCOL`; modern alias substitution is `PILOT_PROTOCOL_INFERENCE`, not paper-specified.

## RNG provenance

- class order: literal PILOT seed 1993 and legacy MT19937 API;
- train sample order: named seed derived from a `SeedSequence(1993)`;
- augmentation: separate named torch seed per task;
- projection R/b, plateau q, and synapse u: three separate named NumPy generators.

No component relies on an unrecorded global NumPy RNG. DataLoader workers are set to zero in protocol smoke so transform randomness stays in the explicitly seeded main torch stream.
