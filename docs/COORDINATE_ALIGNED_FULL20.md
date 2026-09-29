# Coordinate-aligned ridge: CIFAR-100 full 20-task scaling test

## Scope and provenance

This is an **experimental repair**, not the BTSP-CAM paper algorithm or a change to the formal reproduction. The run uses the original frozen ViT feature caches, raw Gaussian sparse encoder and seed 1993, exact `k=41`, CIFAR-100 B0-Inc5 class order, unchanged BTSP write trajectory, `4096 × 8192` binary memory, and the existing ridge solver and `gamma=0.1`. All three readouts use the unthresholded feature `a = W_B s / ||s||_1`; the formal `tau=0.5` path remains untouched. No hyperparameter was tuned.

The formal `results/btsp_cam_cifar100/checkpoint_task1.npz` through `checkpoint_task20.npz` supply read-only `W_B` snapshots. Thus the evolving branches follow the already completed, unchanged formal BTSP writes rather than retraining them. Branch A accumulates each task's `a(W_B_task_t,s)` into historical ridge `A/B`. Branch B updates only current-task `C=Σssᵀ` and `D=Σsyᵀ`, then reconstructs `A=W_B C W_Bᵀ/k²`, `B=W_B D/k` under the current `W_B`. Branch C fixes `W_B_task1` after Task1 and accumulates ordinary ridge `A/B`. Branch B's persistent statistical state contains no individual images, ViT embeddings, sparse codes, or CA1 features. Historical feature-cache reads are confined to seen-class *evaluation* and the five independent replay-oracle checks, not to Branch B's update.

The isolated runner/config are `experiments/run_coordinate_aligned_full20.py` and `experiments/configs/coordinate_aligned_full20.yaml`. Raw outputs are `diagnostics/coordinate_aligned_full20.csv` (one row per branch/task) and `diagnostics/coordinate_aligned_scaling.csv` (one row per task). Accuracies and cosine metrics are fractions, not percentages; storage uses MiB (2²⁰ bytes). Old-class metrics at Task1 are undefined (`nan`). Test data for each source task are read and discarded during each cumulative evaluation. Cosines are exact means over distinct pairs of L2-normalized `a` vectors; zero vectors contribute zero. `old` means all previously seen classes, `current` means the newest five.

## Results

| Task | Official-style evolving / old A,B | Coordinate-aligned | Frozen WB1 |
|---:|---:|---:|---:|
| 1 | 92.40% | 92.40% | 92.40% |
| 5 | 33.00% | 85.52% | 85.92% |
| 10 | 28.40% | 80.38% | 79.96% |
| 15 | 24.17% | 76.13% | 75.64% |
| 20 | **23.14%** | **73.30%** | **72.62%** |

At Task20, coordinate alignment gains **50.16 percentage points** over the unthresholded evolving/old-statistics baseline and **0.68 points** over frozen WB1. The average of cumulative accuracies over all 20 tasks (average incremental accuracy) is **32.26%**, **80.68%**, and **80.46%**, respectively. These are within-protocol comparisons, not paper-accuracy claims.

| Task20 metric | Evolving / old A,B | Aligned | Frozen WB1 |
|---|---:|---:|---:|
| Old 95-class accuracy | 19.56% | 73.49% | 72.80% |
| Current-task accuracy among 100 classes | 91.20% | 69.60% | 69.20% |
| Current-task five-way accuracy | 92.80% | 89.00% | 90.00% |

The old-statistics baseline has high newest-task accuracy but severe old-class failure, consistent with readout/feature-coordinate mismatch. Aligned and frozen WB1 remain close in overall accuracy; the small 0.68-point advantage of aligned WB at Task20 does not establish a general causal benefit from continual writes. Current-task five-way accuracy remains high in all branches and is actually one point lower for aligned than frozen WB1 at Task20.

## Current-WB oracle closure

At Tasks 1, 5, 10, 15, and 20 an independent diagnostic reread all historical **training** feature caches, re-encoded every sample under the checkpoint's current `W_B`, updated ordinary ridge statistics, and compared to Branch B. This is a replay oracle and is **not** the exemplar-free update path. At every checkpoint, oracle and aligned predictions match for every seen test sample and cumulative accuracy matches exactly. The largest observed absolute differences across these checkpoints are `A ≤ 6.37e−12`, `B ≤ 1.42e−13`, ridge weights `≤ 6.84e−11`, and logits `≤ 2.61e−11`. Task20 oracle and aligned accuracies are both **73.30%**. The first five tasks also reproduce the archived R0/R1/aligned accuracy values exactly.

The sparse-dense overlap accelerator uses float32 only to sum at most 41 binary ones—an exact integer operation in that range—and converts to float64 *before* division by 41. A direct Task1 check against the original overlap implementation was exactly equal. `C`, `D`, `A`, `B`, and ridge fitting remain float64; no float32 statistical approximation was needed.

## Geometry and memory evolution

| Task | Evolving WB density | Evolving old-class cosine margin | Frozen-WB1 old-class margin |
|---:|---:|---:|---:|
| 5 | 3.59% | 0.01927 | 0.04569 |
| 10 | 6.04% | 0.01004 | 0.04097 |
| 15 | 7.88% | 0.00720 | 0.03962 |
| 20 | 9.37% | 0.00565 | 0.03850 |

The evolving memory's old-class same/different cosines rise to **0.9454/0.9398** by Task20, leaving a much smaller raw cosine margin than fixed WB1 (**0.6913/0.6528**). Thus there is measurable geometric compression/common-component growth. However, the synchronized readout still reaches 73.30%, slightly above frozen WB1's 72.62%; this particular cosine metric does **not** demonstrate a corresponding catastrophic loss of usable class information. The dominant *accuracy* failure of the unaligned branch is temporal inconsistency of old `A/B`, while the evolving memory does exhibit a secondary, descriptive cosine-geometry degradation. We cannot assign a strict causal percentage to either from this comparison.

## C/D storage and computation

| Task | C nnz | C density | CSR float64 actual | D float64 actual | C/D update | `W C Wᵀ` alignment | Aligned ridge solve |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,611,052 | 2.40% | 18.47 MiB | 0.31 MiB | 0.13 s | 2.70 s | 0.91 s |
| 5 | 5,513,927 | 8.22% | 63.13 MiB | 1.56 MiB | 0.16 s | 9.13 s | 0.93 s |
| 10 | 8,181,303 | 12.19% | 93.66 MiB | 3.13 MiB | 0.17 s | 16.47 s | 0.88 s |
| 15 | 10,186,976 | 15.18% | 116.61 MiB | 4.69 MiB | 0.19 s | 23.12 s | 0.89 s |
| 20 | 11,728,640 | 17.48% | **134.25 MiB** | **6.25 MiB** | **0.19 s** | **28.35 s** | **0.89 s** |

Each task contributes 2,500 training samples × 41² = **4,202,500 ordered pair contributions** before deduplication. At Task20 the observed float64 CSR `C` is **134.25 MiB**, versus 256 MiB for dense float32 and 512 MiB for dense float64; neither crossing threshold occurs. A hypothetical CSR float32 copy would be 89.51 MiB, but was not used because float64 fit. `D` adds 6.25 MiB; combined persistent `C/D` storage is **140.50 MiB**, excluding the separate `W_B` checkpoint and transient ridge matrices. The measured process peak working set was **1,617.57 MiB**; this is a process high-water mark, not `C/D` storage alone.

Across 20 tasks, C/D updates took **3.41 s**, alignments **333.91 s**, and aligned ridge solves **18.08 s**. Full three-branch execution including repeated cumulative test encoding/evaluation and five replay-oracle checks took **1,024.41 s** (~17.1 min). Task20 alone took 462.14 s, but that includes the 50,000-sample replay-oracle encoding/fitting; its own alignment and solve were 28.35 s and 0.89 s. Replay-oracle time was not separately instrumented, so the Task20 total must not be presented as an online per-task cost. The principal growing online computation is `W C Wᵀ`; the direct C/D update stayed below 0.2 s per task on this machine. The 20-task experiment is feasible here, but this is not evidence of production-scale performance beyond `m=8192`, `slots=4096`.

## Interpretation

The full run confirms the five-task mathematical closure at 100 classes. Historical raw exemplars are **not required for the aligned update**, but the method retains a high-dimensional aggregate second-moment matrix; “no historical samples” is not “constant memory.” The independent oracle uses replay strictly for verification. The untouched formal thresholded reproduction is not compared here, because `tau=0.5` is intentionally excluded to isolate coordinate alignment. No threshold or other model parameter was changed.
