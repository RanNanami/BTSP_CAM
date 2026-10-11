# Figure-4-style CA1 prototype drift audit

Status: **FIGURE4_STYLE_LOCAL_OPERATIONALIZATION; NOT PAPER_EXACT_REPRODUCTION.** This audit is read-only. It did not train a model, update W_B, fit or modify a ridge readout, run ViT, change parameters, or overwrite an existing cache/checkpoint/result.

## Paper boundary

- **Paper claim:** `prototype similarity > 0.9`.
- **Paper wording:** `slowly drifting CA1 space`.
- **Exact paper metric formula:** `NOT FULLY SPECIFIED` in the public text available to this project.
- The public description does not fully identify whether Figure 4 uses train or test samples, raw class means or means of sample-normalized features, pre-threshold `a` or thresholded `z`, or its precise checkpoint aggregation rule.
- **Primary local operationalization:** for every class, use exactly the same fixed TEST sample IDs; compute the raw class mean of thresholded Eq. (7) CA1 features at the class-introduction checkpoint and every later checkpoint; report their cosine.
- Sensitivity versions are reported without selecting the favorable one: pre-threshold `a` class means and means of individually L2-normalized nonzero `z` samples.

Both datasets use their frozen k=96 trajectories with `q=4`, `tau=4/96`. CIFAR-100 uses five classes per task and 10,000 fixed test samples. CUB-200 uses ten classes per task and 5,794 fixed test samples. Each class/sample-ID digest stays constant across all compared checkpoints. Every nonzero introduction-checkpoint `i→i` prototype cosine is 1 within `1e-12`.

## Primary results

| Dataset / comparison | Raw-mean z prototype cosine | Sample cosine | Prototype − sample | Sample-normalized-mean prototype | Pre-threshold a prototype | Fraction z prototype >0.90 |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR Task1 classes → Task20 | 0.8285 | 0.7428 | 0.0858 | 0.8194 | 0.9145 | 0/5 |
| CUB Task1 classes → Task20 | 0.4418 | 0.4105 | 0.0313 | 0.4427 | 0.6034 | 0/10 |
| CUB all old classes at Task20 | 0.8643 | 0.8387 | 0.0256 | — | — | mixed exposure |

For CUB Task1→20, the ten class prototype cosines have mean **0.441813**, median **0.440773**, minimum **0.407362**, maximum **0.480163**, and population standard deviation **0.023915**. Fractions above 0.90, 0.95 and 0.99 are all **0%**. The result is insensitive to the alternative “normalize each sample, then average” definition: its mean is **0.442657**. Thresholding amplifies the apparent drift, but does not create the whole discrepancy: the pre-threshold `a` prototype mean is only **0.603369**, still far below 0.9.

The all-297-sample CUB Task1→20 sample cosine is **0.410541**. The existing independent `FIRST_100_CACHE_ROWS_FIXED_AT_SOURCE_TASK` diagnostic is reproduced at **0.402774** (archived value 0.402773642; numerical difference below `2e-7`). The two values use different fixed subsets and therefore should not be forced to match exactly.

## Exposure duration

Task20 summaries preserve exposure duration. Task16–20 are not pooled because they have 4, 3, 2, 1 and 0 subsequent writer updates.

| CUB source tasks → Task20 | z prototype mean | Sample mean | Fraction class prototypes >0.90 |
|---|---:|---:|---:|
| Task1–5 | 0.6349 | 0.5917 | 0% |
| Task6–10 | 0.8881 | 0.8543 | 40% |
| Task11–15 | 0.9676 | 0.9515 | 100% |
| Task16 | 0.9868 | 0.9778 | 100% |
| Task17 | 0.9902 | 0.9842 | 100% |
| Task18 | 0.9944 | 0.9905 | 100% |
| Task19 | 0.9974 | 0.9956 | 100% |
| Task20 (`i→i`) | 1.0000 | 1.0000 | 100% |

This explains why a checkpoint-wide average can appear high even when the oldest classes drift strongly: recent classes dominate with short exposure. At CUB Task20, averaging all 190 old classes gives 0.8643, whereas the ten Task1 classes give 0.4418. The exact Figure 4 aggregation rule is not public enough to decide whether this is the same quantity as the paper's reported `>0.9`.

For CIFAR, Task1→20 is also below 0.9 at 0.8285, although its pre-threshold `a` prototype remains above 0.9 at 0.9145. CIFAR Task1–5 pooled at Task20 is 0.9330, again showing that pooling different exposure durations can conceal the oldest-class drift.

## Relation to readout failure

The existing frozen CUB decomposition is loaded without refitting. At Task20:

- R0 current-W_B historical-TRAIN replay/refit: **80.6006%**.
- R1 literal streaming Eq. (8): **33.3276%**.
- R0−R1 gap: **47.2730 pp**.
- Mean prototype/sample similarity over all old classes: **0.8643 / 0.8387**.
- Task1-only prototype/sample similarity: **0.4418 / 0.4105**.

Across Tasks2–20, the Pearson correlations between the readout gap and old-class prototype/sample similarity are **−0.2455** and **−0.2817**. These are `DESCRIPTIVE_CORRELATION_ONLY`; checkpoints are ordered, dependent observations and this statistic does not establish causation.

The requested logical classification is therefore:

**P2 — LOCAL_WRITER_DRIFT_REGIME_STRONGER_THAN_PAPER_REPORTED.**

Under the primary local definition, neither the oldest CUB nor oldest CIFAR prototypes remain above 0.9. The local frozen trajectory therefore differs materially from the drift regime described by the paper on this metric. Hidden hyperparameters, writer implementation details, feature choice, or Figure 4's undisclosed aggregation could explain the difference; this audit does not select among them.

Because the local result is P2 rather than P1, it does **not** empirically test the implication “prototype >0.9 ⇒ temporal coordinate consistency.” The answer for this trajectory is `NOT_DETERMINED`, not YES. Separately, class-mean stability would not mathematically make historical ridge `A/B` sufficient statistics invariant: means do not determine within-class covariance or the full feature-coordinate transform. That mathematical limitation is context, not a P1 result from the present data.

## Outputs and verification

- `diagnostics/figure4_style_prototype_drift_classwise.csv`: 3,150 class/checkpoint records, fixed sample-ID digests and all three prototype definitions.
- `diagnostics/figure4_style_prototype_drift_summary.csv`: 438 source-task/checkpoint and exposure-group summaries.
- `diagnostics/figure4_style_prototype_vs_sample.csv`: 420 matched source/checkpoint comparisons.
- `diagnostics/figure4_style_prototype_readout_relation.csv`: 20 CUB checkpoint comparisons with archived R0/R1 accuracies and descriptive correlations.
- `diagnostics/figure4_style_prototype_drift_summary.json`: paper boundary, key results, checkpoint hashes, row counts and interpretation.

Runtime for the read-only metric computation was **40.34 seconds**. Tests independently recompute summary arithmetic, fixed-ID invariance, `i→i` identities, the CUB first-100 sample drift, archived R0/R1 linkage and descriptive correlations. No commit or push was performed.

## Final classification

- `PAPER_METRIC_EXACTLY_REPRODUCIBLE = NO`
- `INTERPRETATION_CASE = P2_LOCAL_WRITER_DRIFT_REGIME_STRONGER_THAN_PAPER_REPORTED`
- `DOES_OUR_PROTOTYPE_LOOK_SLOWLY_DRIFTING = NO` for the oldest-class primary metric; recent classes are exposure-duration dependent.
- `DOES_PROTOTYPE_STABILITY_GUARANTEE_READOUT_COORDINATE_CONSISTENCY = NOT_DETERMINED` by this P2 trajectory.
