# CUB-200 readout causal decomposition

## 1. Question and result

On the **same CUB writer trajectory**, how much does changing only the cross-task maintenance of the ridge readout change performance? The local current-`W_B` replay-refit readout (R0) reaches **87.8606% average incremental / 80.6006% final**. Literal Eq.(8) old-`A/B` accumulation (R1) reaches **54.8096% / 33.3276%**. A stable-coordinate control using the Task1 memory at every later task (R3) reaches **84.9342% / 77.2006%**. The R0–R1 gap is **33.0510 / 47.2730 percentage points**. R3–R1 recovery is **30.1247 / 43.8730 pp**. In this fixed local pipeline, the evidence fits **Case A: `STRONG_TEMPORAL_COORDINATE_MISMATCH_EVIDENCE`**.

These are interventions on a local experimental readout, not claims about the authors' undisclosed implementation. R3 is a mechanism control with a different feature map; it is not a fair alternative BTSP method comparison. No parameter was selected by these results.

## 2. Frozen protocol

The run used the existing CUB-200 B0-Inc10 official split, PILOT seed-1993 class order and seeded train order, frozen ViT caches and exact-96 sparse caches, with no new ViT forward pass. Frozen values from `configs/btsp_cam_cub200_frozen.yaml` were `m=8192`, `slots=4096`, `k=96`, `q=4`, `tau=4/96`, `f_base=.005`, `beta=1`, `lambda=.01`, `eta=1`, `p=.5`, `gamma=.1`, and RF/writer seed 1993. All branches use the same sparse train/test rows and labels, class mask, `z=ReLU(W_B s/96 - 4/96)`, ridge solver, and all-seen-class evaluation. The code asserts the frozen configuration before running. No CIFAR, frozen, or CUB configuration or formal `src/` file was edited.

## 3. Same-writer guarantee and reconstruction gate

The existing CUB Full run did not retain every task's `W_B`. `experiments/audit/cub200_readout_decomposition.py` therefore reconstructed **one** writer trajectory from fresh `W_B=0` using only the existing Task1–20 **train sparse caches**, the frozen model implementation, and the original row order/RNG streams. Before comparing readouts, it evaluated R0 at all 20 tasks. Each cumulative accuracy matched `diagnostics/cub200_full_taskwise.csv` with **maximum absolute error 0**, including Task20 `0.8060062133241284`. The arithmetic mean was `0.8786059546186593`. Only after this gate passed did branch evaluation begin.

Every reconstructed memory was stored as a bit-packed, losslessly restored checkpoint under `diagnostics/cub200_readout_wb_checkpoints/WB_task01.npz` … `WB_task20.npz`; each has a SHA-256 of the unpacked binary `W_B`. R0, R1, and R2 load the **same** checkpoint for each task and assert the hash equality. Task20 hash `bf4b2da7528e24035623627797d0a46d9c3ed9ecdd0fdc7692107131c7876a8e` also matches the *independent prior leakage-audit replay* Task20 `W_B` hash. R3 intentionally always loads WB1 (density `0.37436%`), whereas R0/R1 Task20 use WB20 (density `5.26641%`). Neither R1 nor R3 trains a second writer.

The checks and hashes are in `diagnostics/cub200_readout_writer_verification.json` and `diagnostics/cub200_readout_decomposition_summary.json`. The previous leakage audit's `TEST_LEAKAGE=NO_EVIDENCE` stands; no test/future sample entered this writer replay.

## 4. Branch definitions and fit boundaries

| Branch | Feature map for task *t* | New `A/B` contribution at task *t* | Historical training sparse codes used at task *t*? | Status |
|---|---|---|---|---|
| R0 `CURRENT_WB_REPLAY_REFIT` | Current `W_B,t` for every seen source task | Reset, then `Σ_{i≤t} Z_{i|t}ᵀZ_{i|t}` and `Σ_{i≤t} Z_{i|t}ᵀY_i` | **YES** | Local experimental replay readout, not paper-confirmed |
| R1 `LITERAL_STREAMING_EQ8` | Current `W_B,t` for only new train, and all seen test | Preserve old `A/B`; add only `Z_{t|t}ᵀZ_{t|t}` / `Z_{t|t}ᵀY_t` | **NO** | Literal interpretation of public Eq.(8), not claimed as author code |
| R2 `ALIGNED_REBUILD_CHECK` | Current `W_B,t` for every seen train source | Reset, then update **source task by source task** | **YES** | Numerical check only, not an independent performance method |
| R3 `STABLE_WB1_STREAMING_CONTROL` | Freeze `W_B,1` for all later readout features | Preserve old `A/B`; add only current task under fixed WB1 | **NO** | Stable-coordinate control, not paper BTSP-CAM |

For every fit call, the audit checks training sample IDs against the prior forensic provenance ledger. R0/R2 accept only Tasks1–t train IDs; R1/R3 add only Task-t train IDs, never historical rows to their *new* sufficient-statistic update. All branches reject test IDs and future-task train IDs. The experimental driver preloads cached rows for diagnostics; **R1's `STRICT_EXEMPLAR_FREE=YES` describes its readout update rule**, not physical deletion of old cache files from the research workspace. R0/R2 are not strict exemplar-free because historical sparse samples are replayed.

R2 matched R0 at **every task**: `A`, `B`, ridge weights, and evaluation logits had maximum absolute difference **0**, and all accuracies were exactly equal. This confirms that source-task-wise aligned rebuild and the local R0 fit follow identical math; it is not evidence that the paper specifies replay.

## 5. Taskwise performance

Accuracies below are percentages; full precision and current/old/current-10-way, ridge condition, CA1 density, zero-row rate, mean feature norm, and memory density for all 60 branch-task rows are in `diagnostics/cub200_readout_decomposition_taskwise.csv`.

| Checkpoint | R0 cumulative | R1 cumulative | R3 cumulative | R0−R1 gap |
|---:|---:|---:|---:|---:|
| Task1 | 97.3064 | 97.3064 | 97.3064 | 0.0000 pp |
| Task5 | 93.8819 | 71.9409 | 91.3502 | 21.9409 pp |
| Task10 | 87.4177 | 50.5719 | 84.3674 | 36.8458 pp |
| Task15 | 82.8823 | 41.7765 | 79.7132 | 41.1057 pp |
| Task20 | **80.6006** | **33.3276** | **77.2006** | **47.2730 pp** |
| Average incremental | **87.8606** | **54.8096** | **84.9342** | **33.0510 pp** |

Task20 R0 old/current/restricted-current-10-way accuracies are **80.9021 / 75.0000 / 94.5946%**. R1's are **29.8654 / 97.6351 / 97.6351%**: the literal readout becomes highly biased toward recent task coordinates while retaining strong immediate current-task separability. R3's are **77.4645 / 72.2973 / 94.9324%**. R0 and R1 use identical Task20 CA1 test density (`99.7004%`) and mean feature norm (`10.3451`), so this comparison does not confound their feature extraction at a given checkpoint. R3 intentionally differs: WB1 CA1 density is `14.5316%` and mean norm is `0.9207` on Task20 seen test.

## 6. Source-task retention

The complete triangular matrix `Acc(source task i | checkpoint t)` has 210 source-checkpoint combinations per branch in `diagnostics/cub200_readout_source_task_matrix.csv`. Task20 makes the recency pattern unambiguous:

| Task20 source group | R0 mean source accuracy | R1 mean source accuracy | R3 mean source accuracy |
|---|---:|---:|---:|
| Oldest Task1–5 | 81.82% | **10.01%** | 79.97% |
| Newest Task16–20 | 81.97% | **68.53%** | 78.39% |
| Task1 alone | 77.78% | **8.08%** | 80.47% |
| Task20 alone | 75.00% | **97.64%** | 72.30% |

The group means are *unweighted means of five source-task accuracies*, not pooled image accuracies. The full per-source counts, correct predictions, and rates are in the CSV. R1 does not simply lose all classes evenly: older sources are disproportionately damaged while the just-learned task scores exceptionally high. That pattern is consistent with stale historical ridge statistics in a moving feature space; the fixed-map R3 control largely removes it.

## 7. Direct feature-coordinate drift

For each source task and split, the audit fixes the **first 100 cache rows by source ID/order** at the time that source is learned and recomputes their thresholded CA1 features at every later `W_B,t`. It records mean row-wise cosine, mean relative L2 drift `||z_t-z_i||/(||z_i||+1e-12)`, support XOR fraction, and ratio of mean row norms. Train and test probes are separate; test probes never enter any fit. The sampling rule and digest of the 100 sample IDs are written on every row of `diagnostics/cub200_readout_feature_drift.csv` (420 rows). Values below are Task20 **test** probes:

| Source | Mean cosine to learning-time feature | Mean relative L2 drift | Support-change rate | Norm ratio |
|---:|---:|---:|---:|---:|
| Task1 | 0.4028 | 5.6577 | 82.27% | 5.5568 |
| Task5 | 0.7595 | 2.1894 | 28.71% | 2.7910 |
| Task10 | 0.9070 | 0.8354 | 7.67% | 1.6212 |
| Task15 | 0.9701 | 0.3410 | 1.21% | 1.2073 |
| Task20 | ≈1 | 0 | 0 | ≈1 |

At Task20, oldest Task1–5 test probes have mean cosine `0.5920` and support-change rate `56.05%`; newest Task16–20 have mean cosine `0.9898`. Training-probe results show the same direction. These are descriptive representation changes, not a proof that each changed unit individually caused a particular prediction. Floating-point cosine values for self-comparisons can exceed 1 by ~`3e-15` in the raw CSV; interpret those as 1.

## 8. Direct sufficient-statistic coordinate mismatch

At each task, the audit compares literal mixed-time `A_stream,B_stream` with aligned current-memory `A_align,B_align` by exact Frobenius norms. `A_align,B_align` are the same arrays independently rebuilt by R2; no approximation or sampled submatrix is used. `diagnostics/cub200_readout_coordinate_mismatch.csv` stores the 20 task rows.

| Task | `||A_stream−A_align||F / ||A_align||F` | `||B_stream−B_align||F / ||B_align||F` | R0−R1 accuracy gap |
|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 pp |
| 5 | 0.5854 | 0.5206 | 21.94 pp |
| 10 | 0.6213 | 0.5702 | 36.85 pp |
| 15 | 0.6053 | 0.5623 | 41.11 pp |
| 20 | **0.5847** | **0.5439** | **47.27 pp** |

The Pearson correlation across 20 tasks between relative `A` mismatch and the R0−R1 accuracy gap is `r=0.7389`. This is **descriptive correlation**, not an additional causal proof: task index, memory occupancy, and the performance gap co-evolve. The principal intervention is the same-trajectory R0/R1 readout comparison; R3 is the stable-coordinate mechanism control.

## 9. Read-only CIFAR comparison

The existing CIFAR forensic report `docs/PAPER_STYLE_ABLATION_FORENSIC_AUDIT.md` records the frozen Full readout crosscheck. No CIFAR experiment was rerun here.

| Dataset | Replay-refit Abar / Final | Literal streaming Abar / Final | Replay−streaming gap Abar / Final |
|---|---:|---:|---:|
| CIFAR-100, existing read-only result | 85.553 / 79.180% | 31.934 / 20.630% | **53.619 / 58.550 pp** |
| CUB-200, this experiment | 87.861 / 80.601% | 54.810 / 33.328% | **33.051 / 47.273 pp** |

Both local pipelines are sensitive to temporal readout coordinates; the CUB gap is smaller. Two datasets do not justify a universal claim about other backbones, seeds, splits, or tasks.

## 10. Interpretation

**Case A — strong temporal coordinate-mismatch evidence.** With the *same* evolving `W_B,t`, replacing current-coordinate historical replay (R0) by stale historical sufficient statistics (R1) reduces CUB final accuracy by **47.273 pp**. R2 proves that the aligned rebuild is numerically the same as R0, while R3 shows that ordinary streaming ridge works much better once the feature map is held stable: its final accuracy is **43.873 pp above R1** and only 3.400 pp below R0. Task20 old-source collapse, direct feature drift, and substantial `A/B` mismatches all align with this reading. The conclusion is about this frozen local writer/readout system, not an estimate of what the authors implemented.

## 11. Limitations and protocol classification

- **R0:** historical train sparse-code replay **YES**; strict exemplar-free **NO**; local experimental readout, not paper-confirmed.
- **R1:** no historical train code is read for a new sufficient-statistic update; strict exemplar-free **YES at the readout algorithm level**. The research driver still has historical cache files on disk to support R0/R2 and drift diagnostics. This is a literal interpretation of public Eq.(8), not the author's verified code.
- **R3:** stable WB1 map and streaming ridge; no historical train replay in the update, but memory freezing intentionally departs from evolving BTSP-CAM.
- The first-100-row drift subset is deterministic and not selected by outcome. It is not a census of every train/test feature change.
- The `W_B` trajectory is replayed from sparse caches and verified by 20 accuracies plus independent Task20 state hash; a cryptographic per-task hash from the original Full run was not available.
- The earlier forensic audit found `TEST_LEAKAGE=NO_EVIDENCE`, not mathematical impossibility. The official CUB split also contains a few visually near-duplicate cross-split image variants, reported separately there.
- This experiment did **not** vary tau, gamma, writer parameters, backbone, encoder, class order, or dataset, and did not use paper numbers to pick a branch.

## 12. Exact commands, tests, runtime, and git state

From the repository root in PowerShell:

```powershell
$env:PYTHONPATH='src;experiments;upstream/LAMDA-PILOT'
python experiments/audit/cub200_readout_decomposition.py
python -m pytest -q tests experiments/audit/test_cub200_readout_decomposition.py
python -m pytest -q tests
```

The run took **434.48 s (7.24 min)** including writer reconstruction, 20-task R0 gate, all readout branches, R2 equivalence, source-task matrix, feature drift, and coordinate mismatch. Combined suite: **63 passed, 3 subtests passed**; existing `tests/` alone: **50 passed, 3 subtests passed**. The audit tests additionally rerun actual Task1–2 cached data with R0/R1 execution order reversed and verify identical branch logits plus unchanged old B columns. The integration script also asserts frozen parameter equality, preflight accuracy equality, same `W_B` hash, R2 array/logit/accuracy equality, and train-only/future-task identity boundaries before writing final CSVs.

`git diff --name-only` is empty. `git status --short` lists new/untracked audit code and this report alongside pre-existing untracked CUB/experiment files; diagnostic outputs are ignored by the repository's Git rules. No tracked source or config was changed, and no commit or push was made.

## 13. Paper reference only

BTSP-CAM Table 1 reports CUB BTSP-only **85.15% average incremental / 72.91% final**. These values were not used for any parameter, branch, stopping, or interpretation choice. Numerical closeness to either local branch does not identify the authors' readout implementation. In particular, **87.86% / 80.60% cannot be called paper-exact** because R0 replays historical training sparse codes and the local working configuration is not author-confirmed.
