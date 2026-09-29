# Current-WB replay-oracle audit of CA1 threshold, Task1–20

## Scope and mathematical boundary

This is a **mechanism diagnostic**, not an exemplar-free BTSP-CAM variant and not a parameter-selection exercise. It leaves the formal reproduction, the `coordinate_aligned_ridge` implementation, all checkpoint weights, the backbone, sparse encoder, class order, `k`, and ridge `gamma` unchanged. For each checkpoint `W_B,t` and each predeclared `tau` in `{0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50}`, the diagnostic replays **all seen training feature caches**, forms `a = W_B,t s / 41`, applies `z = max(0, a − tau)`, and refits the unchanged float64 ridge in one current-WB coordinate system. It then evaluates all seen test classes. The results at `tau>0` do **not** use C/D closure: `C=Σssᵀ` and `D=Σsyᵀ` are insufficient to reconstruct the ReLU feature Gram matrix exactly.

The runner is `experiments/run_aligned_threshold_oracle_full20.py`; its isolated config is `experiments/configs/aligned_threshold_oracle_full20.yaml`. The complete observations are `diagnostics/aligned_tau_sweep_full20.csv` (280 task/threshold rows) and `diagnostics/a_scale_evolution_full20.csv` (80 task/split rows). Values are proportions unless stated otherwise. CA1 density, sample-nonzero fraction, `mean_z`, and `p99_z` are measured over **all seen test samples** at the current checkpoint; geometry is the exact same/different-class pairwise cosine over those samples, with zero vectors mapped to zero. The scale table's quantiles are element-wise across slots. Task1 old-class quantities are undefined. The diagnostic temporarily retains seen sparse codes/counts to avoid repeated ViT projection, but never supplies future-task data to an earlier ridge fit and does not write historical samples into the experimental coordinate-aligned state.

## Fixed-threshold results

| Fixed tau | Average incremental | Task5 | Task10 | Task15 | Task20 | Task20 CA1 density |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 80.68% | 85.52% | 80.38% | 76.13% | **73.30%** | 99.993% |
| 0.005 | 80.72% | 85.64% | 80.30% | 76.13% | 73.30% | 99.993% |
| 0.01 | 80.73% | 85.60% | 80.34% | 76.12% | 73.29% | 99.993% |
| 0.02 | 80.72% | 85.64% | 80.34% | 76.11% | 73.31% | 99.993% |
| 0.03 | 80.50% | 85.44% | 80.32% | 76.13% | 73.30% | 99.937% |
| 0.05 | 79.56% | 84.44% | 80.24% | 76.05% | **73.32%** | 99.699% |
| 0.075 | 78.21% | 81.44% | 79.78% | 76.07% | 73.29% | 98.981% |
| 0.10 | 76.76% | 78.48% | 78.28% | 75.39% | 73.11% | 97.288% |
| 0.15 | 73.77% | 74.72% | 75.10% | 73.56% | 72.08% | 88.491% |
| 0.20 | 69.56% | 70.04% | 71.46% | 70.87% | 70.01% | 70.353% |
| 0.25 | 63.01% | 60.20% | 67.42% | 66.76% | 67.75% | 46.483% |
| 0.30 | 52.48% | 44.68% | 56.90% | 60.71% | 63.04% | 24.839% |
| 0.40 | 25.27% | 13.28% | 24.20% | 33.01% | 39.89% | 3.607% |
| 0.50 | 7.44% | 4.36% | 5.02% | 7.53% | **10.72%** | **0.206%** |

“Average incremental” is the unweighted mean of each task's cumulative seen-class accuracy. The Task20 **best positive tau in this tested grid** is 0.05 at 73.32%, only **+0.02 percentage points** above `tau=0`; that is a descriptive test-set observation, **not a chosen parameter**. At Task20, `tau=0.05` changes old accuracy from 73.4947% to 73.5263% (+0.0316 points) while current-task 100-way accuracy falls from 69.6% to 69.4%. The difference amounts to two additional correct predictions among 10,000 seen test samples, so it is not a substantive gain. Across Tasks2–20, `tau=0.05` lowers mean old-class accuracy by about **1.02 points**. At `tau=0.10` or larger, old-class accuracy is never above `tau=0` in any Task2–20 checkpoint.

No fixed positive tau is **consistently** superior across all 20 tasks. `tau=0.01` has the highest average incremental accuracy, 80.7316% versus 80.6848% for zero (+0.0468 points), but wins only 9 of 20 tasks and loses 0.01 point at Task20. `tau=0.005` wins 11 of 20; every `tau≥0.075` wins zero tasks. These tiny low-tau differences must not be used to select a final threshold from the test set.

## Activation scale and threshold stability

| Checkpoint | WB density | Train-current mean a | Test-current mean a | Test-old mean a | Test-all mean a | Test-all p99 a |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.916% | 0.0940 | 0.0583 | n.a. | 0.0583 | 0.2195 |
| 5 | 3.592% | 0.2040 | 0.1297 | 0.1305 | 0.1303 | 0.3171 |
| 10 | 6.039% | 0.2863 | 0.1935 | 0.1891 | 0.1896 | 0.3902 |
| 15 | 7.884% | 0.3224 | 0.2156 | 0.2260 | 0.2253 | 0.4146 |
| 20 | 9.369% | 0.3517 | 0.2591 | 0.2521 | 0.2525 | 0.4390 |

Test-all mean overlap rises monotonically from **0.0583** to **0.2525** as the memory fills; its Pearson correlation with WB density across the 20 checkpoints is **0.9964**, a descriptive association, not a causal estimate. The current training samples also have materially larger overlap than current test samples at every listed checkpoint. At Task20, their means are **0.3517 versus 0.2591**; this train/test scale gap is relevant when considering an absolute threshold. The CSV retains mean, standard deviation, p50, p75, p90, p95, p99, and p99.9 separately for train-current, test-current, test-old, and test-all.

The same fixed tau therefore has a changing *sparsification strength*: `tau=0.10` leaves **15.56%** of Task1 test elements active but **97.29%** at Task20; `tau=0.20` leaves **1.65%** then **70.35%**. This is clear fixed-threshold **scale instability**. It does not imply a monotonic accuracy-optimal tau: the per-task highest accuracy in this grid moves among zero and small positive values, with differences often only one or a few test predictions.

Because `a` is an integer overlap divided by 41, its smallest positive value is `1/41≈0.02439`. Any `0<tau<1/41` leaves the support of `a` unchanged: `tau=0.005`, `0.01`, and `0.02` have **exactly the same element density** as `tau=0` at every checkpoint. Their small accuracy changes come from subtracting a constant from nonzero magnitudes, not from making the representation sparser. At Task1, `tau=0.02` raises accuracy from 92.4% to 94.0% without changing density; that isolated early gain does not persist as a stable 20-task benefit.

The older Task1–5 threshold diagnostic used *historical per-task WB features* when accumulating ridge statistics, so its Task5 `tau=0.20` accuracy of 54.08% was measured **with temporal feature-space mismatch** (its `tau=0` baseline was 33.00%). In the present unified current-WB oracle, the same Task5 `tau=0.20` reaches 70.04% but is **15.48 points below** its own `tau=0` baseline of 85.52%. The earlier relative improvement therefore cannot be interpreted as evidence that a large threshold helps once ridge coordinates are aligned.

## What threshold contributes here

At Task20, `tau=0.20` reduces CA1 element density from 99.993% to 70.353%, but lowers cumulative accuracy **3.29 points** and old accuracy **3.38 points**. `tau=0.30` cuts density to 24.839% and accuracy to 63.04% (−10.26 points). Interestingly, the all-class cosine margin can increase as tau rises (Task20: 0.00561 at zero versus 0.03792 at 0.20), while classification falls; this one scalar geometry metric is not a sufficient readout-quality criterion.

`tau=0.50` remains a destructive truncation even with a perfectly current-WB ridge: Task20 cumulative **10.72%**, old **10.75%**, current **10.20%**, and current five-way **37.20%**. Only **0.206%** of test CA1 elements are active; 49.67% of samples have at least one active element, but nearly every element is zero and test `p99(z)=0`. Task1 is even more extreme: 20.2% five-way accuracy, 0.0000488% active elements, and only 0.2% of samples have any active element. Thus temporal coordinate alignment does not rescue this threshold scale.

**Conclusion:** within this unchanged representation and tested grid, tiny positive thresholds can slightly rescale features and fluctuate around zero-threshold accuracy; thresholds that produce meaningful sparsification mostly reduce classification accuracy; `tau=0.5` causes severe information loss. There is no demonstrated fixed positive threshold with both stable 20-task accuracy improvement and meaningful sparsification. This diagnostic does not infer the paper authors' unpublished tau or authorize choosing one by test accuracy. Any subsequent calibration rule would need to use only available current-training statistics and be evaluated separately.
