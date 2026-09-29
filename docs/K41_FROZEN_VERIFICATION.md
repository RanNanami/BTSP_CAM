# k=41 frozen repaired-baseline verification

Status: **EXACT_REPRODUCTION**. This is a fresh Task1–20 training run, not a replay of historical `W_B` checkpoints and not a parameter search. The experimental coordinate-aligned readout is distinct from the untouched formal paper reproduction.

## Locked protocol

The run reused the same `load_split`, cached frozen ViT features, class/sample order, model construction, BTSP writer, coordinate-aligned ridge solve, and `score_sources` evaluation functions as the latest `k_extended` full20 experiment. It constructed a fresh model at Task1 and trained all 20 tasks in order. CIFAR-100 B0-Inc5; class-order, RF, and writer seeds all 1993; `m=8192`, 4096 memory slots, `k=41`, `p=0.5`, `f_base=0.005`, `beta=1`, fixed `lambda=0.01`, `gamma=0.1`, and `tau=0` were asserted before training. No normalized gamma, alternative decay/write rule, budget rescaling, threshold sweep, or test-driven selection was used.

For each task, the fresh model encoded that task's 2,500 cached training features, updated the original BTSP writer and the experimental `C/D` sufficient statistics, solved the current-`W_B` aligned ridge, then evaluated the seen cached test tasks once. The run did not load old `W_B` checkpoints for training or evaluation. The 20-row flat data table is `diagnostics/k41_frozen_verification.csv`; values are fractions in the CSV and percentages below. `old_accuracy=nan` at Task1 is intentional because there are no old classes.

## Results

| Task | Cumulative | Old | Current | Current 5-way | W_B density | Gross flips, task |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 92.40% | n.a. | 92.40% | 92.40% | 0.9157% | 343,528 |
| 5 | 85.52% | 85.00% | 87.60% | 96.20% | 3.5923% | 323,740 |
| 10 | 80.38% | 80.3778% | 80.40% | 93.60% | 6.0391% | 335,224 |
| 15 | 76.1333% | 76.9429% | 64.80% | 89.20% | 7.8835% | 300,528 |
| 20 | **73.30%** | **73.4947%** | **69.60%** | **89.00%** | **9.3686%** | **300,668** |

The Task1–20 mean cumulative (average incremental) accuracy is **80.6848031372%**. The Task20 cumulative accuracy is **73.30%**; old/current/current-5-way are **73.4947% / 69.60% / 89.00%**.

## Historical parity and classification

The historical `k41 / evolving_aligned` trajectory in `diagnostics/k_path_decomposition.csv` has 20 task rows. The fresh run matches it at **every task exactly** for cumulative, old, current, and 5-way accuracy, `W_B` density, and task gross flips: maximum absolute difference is **0** for each field. Its 20 cumulative accuracies also match the earlier `coordinate_aligned_ridge` branch in `diagnostics/coordinate_aligned_full20.csv` exactly.

| Comparison | Fresh run | Historical archive | Delta |
|---|---:|---:|---:|
| Average incremental | 80.6848031372% | 80.6848031372% | **0.0000pp** |
| Task20 cumulative | 73.30% | 73.30% | **0.0000pp** |

The user-facing historical shorthand “80.68%” was rounded. Subtracting that rounded number gives +0.004803pp, a display-rounding difference, **not** a run mismatch. Against the unrounded archived trajectory, `delta_avg=0` and `delta_final=0`. This meets **EXACT_REPRODUCTION**, stronger than the specified 0.05pp numerical-equivalence tolerance. No discrepancy investigation or baseline replacement is needed.

## Integrity and scope

- The output CSV has exactly one row for each Task1–20; task, protocol, accuracy, density and flip columns are populated. Only Task1 old-class accuracy is not applicable.
- Before and after the run, SHA-256 matched for **41 protected existing files**: the formal default YAML, two experiment YAMLs, 15 core `src/btsp_cam/*.py` files, all 20 formal `checkpoint_task*.npz` files, and the archived k-path/full20 selection artifacts. There were **zero changed hashes**. No source, config, or checkpoint was modified or overwritten.
- This confirms reproducibility of the present **repaired** k=41 pipeline under the current code/cache. It does not resolve the paper's undisclosed parameters or make coordinate-aligned ridge/`tau=0` the author's implementation. The formal reproduction remains separate and untouched.
