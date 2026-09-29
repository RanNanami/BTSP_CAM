# Extended train-only k search before baseline freeze

Status: complete, 2026-09-29. This is an **experimental repaired pipeline**, not the published/default BTSP-CAM implementation and not evidence of the authors' undisclosed `k`.

## Locked protocol and selection boundary

- CIFAR-100 B0-Inc5; fixed 20-task class order, seed 1993; frozen ViT feature cache and existing sample order. RF and writer seeds are both 1993. `m=8192`, 4096 binary memory slots, and all non-`k` baseline settings are unchanged: `p=0.5`, `f_base=0.005`, `beta=1`, `lambda=0.01`, `gamma=0.1`.
- Readout is the independent **coordinate-aligned ridge** repair with `z=a=W_B s/k` and `tau=0`. The formal default implementation, configuration and checkpoints were not changed.
- The grid was fixed before running: `k=[128,160,192,224,256,320,384,512,768,1024]`. Input sparsity is `k/8192`, spanning 1.5625% to 12.5%. There was no adaptive grid expansion or early stopping.
- Each class in Tasks1–5 was split with seed 20260928 into 400 development-training and 100 validation examples. Validation examples did **not** write eligibility or `W_B`, nor enter `C/D` or ridge statistics. The primary metric was the mean of five cumulative **training-side validation** accuracies. A candidate within 0.10 percentage points of the maximum ties, and the smallest tied `k` wins.
- Both control candidates reproduced every archived Task1–5 validation accuracy exactly: `k=128` mean 80.7807% and `k=160` mean 81.5847%. New candidates were accepted only after that check.

## Task1–5 validation curve

All percentages are percentage units, not proportions. Geometry and writer columns are Task5 development/validation diagnostics; the primary selection metric is **only** the mean-validation column.

| k | k/m | Mean validation | Task5 validation | Sparse cosine margin | Top-1% bit share | Entropy effective support | Task5 W_B density | Gross flips / dev sample |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1.5625% | 80.7807% | 75.92% | 0.05716 | 13.53% | 2,906 | 8.39% | 438 |
| 160 | 1.9531% | 81.5847% | 75.84% | 0.06012 | 12.42% | 3,109 | 10.08% | 555 |
| 192 | 2.3438% | 82.3787% | 77.36% | 0.06222 | 11.56% | 3,282 | 11.66% | 672 |
| 224 | 2.7344% | 82.3513% | 77.64% | 0.06395 | 10.85% | 3,436 | 13.15% | 790 |
| 256 | 3.1250% | 82.0467% | 77.80% | 0.06550 | 10.26% | 3,575 | 14.56% | 909 |
| 320 | 3.9063% | 82.6580% | 78.04% | 0.06786 | 9.32% | 3,815 | 17.13% | 1,146 |
| 384 | 4.6875% | 82.5107% | 78.12% | 0.06962 | 8.57% | 4,022 | 19.43% | 1,384 |
| 512 | 6.2500% | 83.1827% | 78.08% | 0.07211 | 7.45% | 4,365 | 23.40% | 1,856 |
| 768 | 9.3750% | 83.4420% | 77.96% | 0.07458 | 6.03% | 4,883 | 29.47% | 2,774 |
| 1024 | 12.5000% | **83.8600%** | 79.20% | 0.07519 | 5.12% | 5,277 | 33.84% | 3,643 |

The predeclared rule selected **`k=1024` before any new full20 test run**. Its mean validation accuracy exceeds `k=768` by **0.418 percentage points**, exceeding the 0.10pp tie tolerance. `k=160` was the prior grid boundary, not the optimum in this extended grid; `k=1024` exceeds it by 2.275pp on the selection metric. The curve has small local reversals (224/256 and 384), so it is not pointwise monotone. Nevertheless its final boundary step is positive and above tolerance: classification **`STILL_INCREASING_AT_BOUNDARY`**. The global optimum remains unresolved; `1024` is only the new boundary. We stop here as predeclared, without extending the search.

## Fresh full20: selected candidate and predeclared neighbor

The development models were discarded. Both runs retrained from Task1 using the full training set; each task's test split was evaluated once after its update. The test results below were not used to reselect `k`.

| k / role | Task5 | Task10 | Task15 | Task20 100-class | Average incremental | Task20 old | Task20 current | Task20 current 5-way |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 768 / neighbor | 92.92% | 88.66% | 86.09% | 83.92% | 89.39% | 84.11% | 80.40% | 95.40% |
| **1024 / train-selected** | **93.84%** | **89.14%** | **86.39%** | **84.34%** | **89.55%** | **84.52%** | **81.00%** | **94.00%** |

At Task20, selected minus neighbor is +0.42pp cumulative and +0.16pp average incremental. This is a modest test difference, consistent with a broad high-`k` region, **not** proof of a bracketed peak: no value above 1024 was tested, and the train-only curve was still rising at the boundary. The 5-way current-task comparison also changes sign (94.0% versus 95.4%), so the selected `k` does not dominate every readout. Relative to the previously recorded default-seed `k=160` path (86.90% average / 81.16% final), the locked `k=1024` run is higher by 2.65pp / 3.18pp. This comparison was not a selection criterion.

## Representation and write-budget interpretation

From `k=160` to `k=1024` on Task5 validation, sparse same/different cosine changes from 0.1444/0.0843 to 0.3241/0.2489: both rise, but their margin rises from **0.0601 to 0.0752**. Entropy effective support rises 3,109→5,277 bits; top-1% bit share falls 12.42%→5.12%; never-winning RF bits fall 253→0. This is compatible with less winner monopoly and better sparse class separation under the larger exact-`k` code. Raw overlap counts `J_same/J_diff` rise 23.10/13.48→331.89/254.90, mechanically expected when `k` rises; raw `J` growth alone is not collision harm or benefit.

The memory/write path changes concurrently. At Task5 validation, `W_B` density rises 10.08%→33.84%, cumulative gross flips per development sample 555→3,643, and flips per plateau 79.9→511.8. At Task20 full training, `W_B` density is 43.70% for `k=768` and 46.09% for `k=1024`; gross flips per sample are 2,696 and 3,544, respectively. There was **no equal-write-budget intervention** in this round, so these correlated changes cannot be assigned independent causal shares. Earlier sparse-only and frozen-memory path controls up to `k=160` support a representation-side contribution, but cannot quantitatively extrapolate that decomposition to 1024.

The raw unthresholded `a` cosine margin does **not** improve with larger `k`: on Task5 validation it falls 0.00250→0.000196 from 160→1024; Task20 full-test margins are 0.000177 (768) and 0.000130 (1024). A growing shared positive overlap component raises both same- and different-class cosine toward 1. Thus the classification gain is **not** explained by a larger raw `a` cosine margin; the high-dimensional ridge can still exploit small absolute feature differences. The experimental `a` is fully element-dense at Task5 for this range of `k`. We did not compute a random-feature condition number in this round, so do not claim improved conditioning.

## Reproducibility and limits

- Protocol: `experiments/configs/k_extended_search.yaml`; runner: `experiments/run_k_extended_search.py`. The selection lock `diagnostics/k_extended_selection_lock.json` was written from the validation CSV before either new full20 run and records `full20_k=[768,1024]` and `selection_used_test_accuracy=false`.
- `diagnostics/k_extended_validation.csv` has exactly 50 unique `(k,task)` rows, five for every locked grid value. `diagnostics/k_extended_full20.csv` has exactly 40 unique `(k,task)` rows, 20 for each locked full20 candidate. Every full20 row records one test evaluation at that task. Archived 128/160 controls match all five cumulative/current/5-way/average validation fields to numerical tolerance 1e-12.
- In both CSVs, Task1 `old_accuracy=nan` is intentional: no old classes exist yet. Other requested numeric metrics are populated.
- This is one fixed RF/writer seed pair and one stratified validation split, not a multi-seed uncertainty estimate for high `k`. The tie rule is a deterministic selection convention, not a statistical confidence interval. The prior nine-pair result only established `k=128 > k=41` under that smaller comparison.
- SHA-256 hashes of the formal YAML and all 15 core `src/btsp_cam/*.py` files matched their pre-experiment values. No formal config, model source, or checkpoint was overwritten.
- **No author-parameter inference:** under this reconstructed/repaired pipeline, train-side validation selected `k=1024` from the predeclared grid. It does not show that the authors used `k=1024`, nor that this is a globally optimal `k`.
