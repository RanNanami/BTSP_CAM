# CUB-200 frozen-baseline protocol audit

Status: **protocol identified; official-split data now prepared and integrity-checked**. The initial read-only audit preceded CUB code or training; the data-availability addendum below records the subsequent acquisition. The pinned `upstream/LAMDA-PILOT` checkout is commit `7a6e904c5bc5cb7a4e1823b3434020be27469b63`. No CIFAR configuration, checkpoint, or result was changed.

## Paper evidence

The local BTSP-CAM PDF printed p.6, §5.1 names **CUB-200** among its benchmarks, specifies a common frozen ImageNet-21K-pretrained ViT-B/16, class-order shuffle seed **1993**, RF dimension **8192** and **4096** memory slots. Table 1 on printed p.6 labels the CUB column **B0 Inc10** and reports BTSP-only **Abar 85.15% / final 72.91%**. Printed p.7, §5.1 defines Abar as the arithmetic mean of incremental-stage top-1 accuracies. The PDF does not state a CUB-specific k, tau, f_base, beta, lambda, eta, p, gamma, image-list hash, or sample count. These numbers are a reference only, not a tuning target. Source: `../2233_BTSP_CAM_A_Brain_Inspired.pdf`, printed pp.6–7.

## Pinned PILOT implementation

| Item | Evidence and status |
|---|---|
| Dataset identifier | `cub` maps to `CUB()` in `upstream/LAMDA-PILOT/utils/data_manager.py:218-235` (**PILOT_SPECIFIED**). |
| Data split and format | `CUB.download_data()` loads `torchvision.datasets.ImageFolder('./data/cub/train/')` and `ImageFolder('./data/cub/test/')`, with `use_path=True`; sorted folder names become original integer labels and sorted image paths determine within-class base order (`upstream/LAMDA-PILOT/utils/data.py:260-278`). PILOT does **not** construct a split from official CUB metadata, and does **not** download CUB. Exact train/test image lists and counts are determined by the supplied folders, not by these source lines. (**PILOT_SPECIFIED** layout; **PAPER_UNSPECIFIED** image-list identity.) |
| CIL split | `DataManager('cub', shuffle=True, seed=1993, init_cls=10, increment=10, ...)` yields 20 tasks of 10 classes for 200 class folders (`upstream/LAMDA-PILOT/utils/data_manager.py:10-24,139-165`). `init_cls=10`, `increment=10`, `shuffle=True` are the local explicit mapping from paper B0-Inc10 and its seed, not a checked-in CUB-specific PILOT experiment JSON. (**PAPER_SPECIFIED** task shape/seed; **LOCAL_CHOICE** concrete PILOT launch arguments.) |
| Class order/remapping | `np.random.seed(1993); np.random.permutation(200).tolist()` then `order.index(original_label)` for both splits (`upstream/LAMDA-PILOT/utils/data_manager.py:153-165,208-210`). The local no-global-state equivalent is `np.random.RandomState(1993).permutation(200)`. (**PILOT_SPECIFIED**.) |
| Selection order | `get_dataset(indices, source, mode)` concatenates samples class-by-class in requested remapped-class order, selecting each label interval from its split (`upstream/LAMDA-PILOT/utils/data_manager.py:32-75,167-170`). A training DataLoader shuffle/augmentation seed remains an explicit **LOCAL_CHOICE**, kept equal to the CIFAR frozen pipeline's RNG provenance. |
| Train transform | PILOT `CUB.train_trsf = build_transform(True,None)`: `RandomResizedCrop(224, scale=(.05,1), ratio=(.75,1.333…))`, horizontal flip p=.5, `ToTensor()`; `common_trsf=[]` (`upstream/LAMDA-PILOT/utils/data.py:95-110,260-267`). **No Normalize**. (**PILOT_SPECIFIED**.) |
| Test transform | `Resize(256, interpolation=3/bicubic)`, `CenterCrop(224)`, `ToTensor()`, no Normalize (`upstream/LAMDA-PILOT/utils/data.py:112-123,260-267`). PILOT `DummyDataset` opens file paths as RGB PIL images (`upstream/LAMDA-PILOT/utils/data_manager.py:191-208,245-254`). (**PILOT_SPECIFIED**.) |
| Backbone feature | Pinned PILOT `utils/inc_net.py:10-20,289-309` has the `vit_base_patch16_224_in21k` frozen `num_classes=0` branch and returns `backbone(x)` for 768-D features; current `src/btsp_cam/backbone.py:22-66` uses timm 1.0.29's `vit_base_patch16_224.augreg_in21k` alias and `model(x)`. The established feature-output audit identifies the final LayerNorm CLS token before classification head. The existing CIFAR checkpoint/feature path is reused, with no CUB-specific normalization. (**PILOT_SPECIFIED** output semantics; **LOCAL_CHOICE** timm-version alias.) |
| Sample counts | Caltech's official CUB-200-2011 record states 200 categories / 11,788 images. The downloaded original metadata and constructed ImageFolder tree both measure 5,994 train / 5,794 test. The pinned PILOT `ImageFolder` loader itself does **not** enforce these counts; the separately distributed PILOT preprocessed folder's image-list identity remains unverified. ([Caltech dataset record](https://www.vision.caltech.edu/datasets/cub_200_2011/), [CaltechDATA archive](https://data.caltech.edu/records/65de6-vp158)). |

## Dataset availability and protocol boundary

The initial search found no CUB data in the project, parent data root, pinned PILOT default root, or Downloads. PILOT's README points to manually supplied preprocessed folders (`upstream/LAMDA-PILOT/README.md:170-186`); the loader itself does not auto-download. The [CaltechDATA](https://data.caltech.edu/records/65de6-vp158) signed download returned **403 Forbidden** on this host, and the older Caltech Vision URL returned **404**. A [GitHub-hosted mirror of the original `.tgz`](https://github.com/vignagajan/CUB-200-2011) was downloaded instead. Its MD5 is **`97eceeb196236b17998738112f37df78`**, exactly matching CaltechDATA's published MD5, so the archive bytes match the original release. The mirror is the download transport, not a third-party resplit.

`scripts/prepare_cub200_pilot_split.py` uses original `images.txt`, `image_class_labels.txt`, `classes.txt`, and `train_test_split.txt` (`1=train`, `0=test`) to construct `data/cub`. It does not random-split, move, or modify the source. All 11,788 targets were hardlinked; an idempotency rerun skipped all 11,788 existing identical files. `diagnostics/cub200_dataset_integrity.json` records PASS: 200 matching class directories, 11,788 metadata IDs and image files, 5,994 train, 5,794 test, no missing/extra files, all images open, no train/test path overlap, and explicit ImageFolder-to-incremental label mapping. `diagnostics/cub200_class_counts.csv` records unequal per-class counts. **DATA_READY=YES for the official-split ImageFolder protocol.**

This is the official split requested for this transfer diagnostic and is compatible with pinned PILOT's loader/transform behavior. The BTSP-CAM paper and pinned PILOT code do not publish an image-list checksum for the separately offered PILOT preprocessed archive; exact identity to that archive remains **PAPER_UNSPECIFIED**, not silently asserted.

Required tree (class folder names should be identical and sortable in both splits):

```text
data/cub/
  train/<200 class-name directories>/<image files>
  test/<200 matching class-name directories>/<image files>
```

The CUB runner independently inventories all supplied image bytes and paths, verifies 200 matching classes and disjoint train/test content, and records a content-based dataset digest before smoke/Full. If a separate PILOT preprocessed archive is later found with a different split, that difference must be reported rather than silently repaired or used to tune toward paper numbers.

## Frozen local working configuration and reuse

The CUB run must lock to `experiments/configs/ablation_baseline_frozen.yaml`: m=8192, slots=4096, k=96, q=4, tau=4/96, f_base=.005, beta=1, lambda=.01, eta=1 (from formal config), p=.5, gamma=.1, RF/writer seed 1993. Initial W_B is all zero (`src/btsp_cam/binary_memory.py:29-38`); Eq.(4)–(6) uses the unmodified core implementation. This is a **LOCAL_CHOICE working baseline**, not an author configuration. The `CURRENT_WB_REPLAY_REFIT` readout reconstructs all seen historical *training sparse-code* CA1 features at the current W_B and solves gamma=.1 ridge; it is a coordinate-consistent local working readout, **not** claimed paper-exact/exemplar-free Eq.(8).

Reusable: `BTSPCAM` writer, sparse encoder, RNG provenance, frozen `PilotFrozenViT`, class-order/remap semantics, CA1/ridge math, cache validation. Minimum additions: CUB `ImageFolder` inventory/loader, strict PILOT transform and remap adapter, variable per-class/task sample counts, independent CUB config/cache/output paths and full-task evaluator. No CIFAR defaults or checkpoint paths need changing.
