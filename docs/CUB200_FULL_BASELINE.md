# CUB-200 frozen working Full baseline

**Status: DATA_INTEGRITY_PASS; REAL_SMOKE_PASS; FULL_20_TASK_COMPLETED.** This is a frozen cross-dataset transfer diagnostic, not a paper-exact or tuned implementation. Protocol details and source-line evidence are in `docs/CUB200_PROTOCOL_AUDIT.md`.

## Data source, official split, and integrity

No CUB data were present in the initially searched project, parent data, PILOT, or Downloads roots. The [CaltechDATA original release](https://data.caltech.edu/records/65de6-vp158) returned HTTP 403 through its signed download redirect; the old Vision URL returned 404. The original archive was obtained through a [GitHub-hosted mirror](https://github.com/vignagajan/CUB-200-2011), **not** a third-party resplit. Its MD5 `97eceeb196236b17998738112f37df78` exactly matches CaltechDATA's published MD5. The source is retained at `data/CUB_200_2011.tgz` and extracted under `data/CUB_200_2011/`.

`scripts/prepare_cub200_pilot_split.py` used original `images.txt`, `image_class_labels.txt`, `classes.txt`, and `train_test_split.txt`: flag 1 went to train and 0 to test. It hardlinked (never moved) all 11,788 images into PILOT's `data/cub/{train,test}/<class>/` ImageFolder layout, preserving all 200 original class names. A second run skipped all 11,788 identical targets, demonstrating idempotency. The script checks metadata ID uniqueness/equality, raw and output no-missing/no-extra files, all images opening, class directory identity, and split disjointness. `diagnostics/cub200_dataset_integrity.json` reports PASS: **5,994 train / 5,794 test**. `diagnostics/cub200_class_counts.csv` records each class's true count. The runner independently hashed all image bytes and paths to SHA-256 `d4b321f10635f985fa79fd0d333941526c530e362ba7d3690e7fce76b9c85af7`.

This is the **official CUB split** requested for the transfer diagnostic. Pinned PILOT's loader and transforms consume it unchanged, but the paper and pinned PILOT code do not publish a checksum for PILOT's separately distributed preprocessed image lists. Identity to that separate archive is unverified, and Table 1 is a reference rather than a claim of identical image lists or author implementation.

## Frozen protocol and run contract

- Dataset: pinned PILOT `cub` ImageFolder, `DataManager('cub', True, 1993, 10, 10, ...)`, 200 classes, 20 B0-Inc10 tasks. Official split supplied through the folder structure; PILOT class order and remapping were checked against seed 1993.
- Frozen ViT-B/16 ImageNet-21K, final LayerNorm CLS 768-D. PILOT CUB train transform is random-resized crop 224 (scale .05–1), horizontal flip, ToTensor; test is bicubic resize 256, center crop 224, ToTensor; no Normalize. The experimental cache keeps one seeded training augmentation per image, as in the frozen CIFAR working path.
- Working parameters: m=8192, slots=4096, k=96, q=4, tau=4/96, f_base=.005, beta=1, lambda=.01, eta=1, p=.5, gamma=.1, RF and writer seed=1993. W_B starts at zero. These local values were transferred without CUB retuning.
- Readout: **CURRENT_WB_REPLAY_REFIT**, using all seen **training sparse codes** with current W_B at each task. This is a local coordinate-consistent working readout requiring historical sparse-code replay; it is **not** asserted to be the paper's author implementation or exemplar-free Eq.(8).
- Only Full was run: no component ablation, threshold sweep, alternate readout, or post-result parameter change.

## Reproduction commands and smoke gate

Run from the project root. Preparation is idempotent and requires the published original-archive MD5. Inventory must pass before training; Full requires a smoke marker for the same dataset digest.

```powershell
python scripts/prepare_cub200_pilot_split.py --source data/CUB_200_2011 --archive data/CUB_200_2011.tgz --output data/cub --diagnostics diagnostics
$env:PYTHONPATH='src;experiments;upstream/LAMDA-PILOT'
python experiments/run_btsp_cam_cub200.py --phase inventory
python experiments/run_btsp_cam_cub200.py --phase smoke
python experiments/run_btsp_cam_cub200.py --phase full
```

The real smoke trained Task1–2 in an isolated output directory and passed exact-96 codes, finite ViT features, class remapping, seen-test sizes, nonzero CA1, writer updates, ridge solve, prediction range, and train-only replay checks. `diagnostics/cub200_smoke_task1_2.json` records 10/20 seen classes, cumulative test sizes 297/593, nonzero CA1 density and successful solves. Full started a **fresh W_B trajectory from Task1**; its Task1–2 accuracies exactly matched smoke. Both phases reused validated feature/sparse caches but not smoke writer state.

Full outputs are `diagnostics/cub200_full_taskwise.csv`, `diagnostics/cub200_full_geometry.csv`, `diagnostics/cub200_full_writer.csv`, and `diagnostics/cub200_full_summary.json`. Feature caches contain dataset digest/count, split, class order, task classes, backbone model identifier/timm version/output point, PILOT transform provenance and augmentation/order seeds. Sparse caches add projection seed, k and exact-k density validation. The model identifier tracks the timm checkpoint family, but the feature cache does **not** include a cryptographic hash of the ViT weight file. Existing CIFAR caches/checkpoints were not read or rewritten.

## Full 20-task result

| Task | Seen classes | Cumulative test n | Cumulative accuracy |
|---:|---:|---:|---:|
| 1 | 10 | 297 | 97.31% |
| 2 | 20 | 593 | 96.12% |
| 3 | 30 | 866 | 95.61% |
| 4 | 40 | 1,146 | 93.89% |
| 5 | 50 | 1,422 | 93.88% |
| 6 | 60 | 1,707 | 91.27% |
| 7 | 70 | 2,000 | 88.70% |
| 8 | 80 | 2,300 | 89.17% |
| 9 | 90 | 2,596 | 88.75% |
| 10 | 100 | 2,885 | 87.42% |
| 11 | 110 | 3,181 | 87.33% |
| 12 | 120 | 3,465 | 85.86% |
| 13 | 130 | 3,755 | 84.42% |
| 14 | 140 | 4,045 | 84.05% |
| 15 | 150 | 4,323 | 82.88% |
| 16 | 160 | 4,615 | 82.86% |
| 17 | 170 | 4,913 | 82.98% |
| 18 | 180 | 5,211 | 82.50% |
| 19 | 190 | 5,498 | 81.59% |
| 20 | 200 | 5,794 | 80.60% |

The unrounded arithmetic mean of the 20 cumulative accuracies is **87.8606%**. Task20 final is **80.6006%**; old-class accuracy **80.9021%**, current-class all-seen accuracy **75.0000%**, and restricted current 10-way accuracy **94.5946%**. Final W_B density is **5.2664%**. Final CA1 element density is **99.7004%** and `P(a≤tau)` is **0.2996%** over all seen test samples. This large CA1 density is an observation, not a trigger to change tau. Final ridge condition number is **1.6956×10^8**, solve status `ok`. Task20 plateau/LTP/LTD event counts are 2,267 / 88,036 / 20,924. Full runtime was **1,345.99 s (22.43 min)** on CPU-only PyTorch; peak process RAM **2,176.86 MiB**; no CUDA VRAM was available. The summary JSON agrees with the 20-row accuracy mean and final row.

## Paper reference, not a tuning target

| Measure | Local frozen CUB | Paper Table 1 BTSP-only | Local minus paper |
|---|---:|---:|---:|
| 20-task average incremental accuracy | 87.8606% | 85.15% | +2.7106 pp |
| Task20 final accuracy | 80.6006% | 72.91% | +7.6906 pp |

The paper values were rechecked in the local PDF Table 1, printed p.6, and **were not used to select or adjust any parameter**. The PDF did not disclose CUB-specific hyperparameters or exact PILOT image-list identity. The positive local deltas must not be interpreted as a paper-exact reproduction: k=96, tau=4/96, and replay-refit readout are declared local working choices. No ViT→ridge or sparse→ridge reference was run; they were optional, and the single Full trajectory already answered the transfer-health question without expanding the experiment.

The complete test suite passed **50 tests and 3 subtests** (including CUB official split/integrity/idempotency, task and exact-k, no-test-leakage, frozen-lock, and CIFAR regression checks). No formal CIFAR source/config/results/checkpoints or pre-existing ablations were changed. No commit or push was made.
