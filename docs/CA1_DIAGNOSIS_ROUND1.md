# CA1 collapse diagnosis, round 1

## Scope

This round uses the exact formal CIFAR-100 B0-Inc5 feature caches, PILOT class
order, seed 1993, frozen ViT-B/16 ImageNet-21K representation, streaming/no-replay
protocol, BTSP update order, and configured `k=41`, `tau=0.5`. Only Tasks 1-5
were run. No model or protocol hyperparameter was changed.

The diagnostic separates

`s -> a = W_B s / k -> z = max(0, a - tau)`

and saves the binary memory after each task. The unthresholded readout exists
only in the diagnostic runner and is not saved as a formal model.

## Result

`W_B` learns a highly readable current-task overlap representation. The fixed
threshold removes almost all test activation. Separately, continuing to update
`W_B` changes old-task overlap features enough that ridge statistics formed in
earlier feature spaces no longer match current inference features.

The primary immediate collapse is therefore `a -> z`. Continual representation
drift is a second, independent failure visible after Task 1. The evidence does
not identify the stochastic BTSP write rule as the primary first-task failure.

## 1. Pre-threshold overlap distribution

The statistics below flatten the `[samples, 4096 slots]` overlap matrix. Values
are multiples of `1/41` before float conversion.

| Task | Split | Mean `a` | P99 `a` | Max `a` | Fraction `a > 0` | Fraction `a > 0.5` |
|---:|---|---:|---:|---:|---:|---:|
| 1 | train | 0.0940 | 0.3171 | 0.8049 | 90.35% | 8.37e-4 |
| 1 | current test | 0.0583 | 0.2195 | 0.5122 | 79.92% | 4.88e-7 |
| 3 | current test | 0.0959 | 0.2683 | 0.5366 | 96.18% | 1.46e-6 |
| 5 | train | 0.2040 | 0.4146 | 0.7805 | 99.92% | 1.33e-3 |
| 5 | current test | 0.1297 | 0.2927 | 0.5610 | 99.05% | 9.77e-7 |
| 5 | old test | 0.1305 | 0.3171 | 0.6098 | 98.98% | 6.96e-6 |

The overlap is not globally zero: most test entries are positive by Task 5,
and the unthresholded readout below shows that their relative pattern is
class-informative. The scale is below the threshold. Even the current-test P99
remains between 0.22 and 0.29 across Tasks 1-5, far below 0.5. At Task 1 only
one of 2,048,000 current-test slot activations exceeds the threshold; at Task 5
only two do.

There is also a train/test tail gap. Training fractions above 0.5 are roughly
7.5e-4 to 1.33e-3, while current-test fractions are zero to 1.46e-6. The
threshold selects rare training-code overlap events that almost never
generalize to unseen samples.

## 2. Thresholded and unthresholded readout

Both readouts use task-wise streaming sufficient statistics. Both train on the
current task representation produced by the current post-update `W_B` and
evaluate cumulative samples with the latest `W_B`. The only difference is
whether the feature is `z=max(0,a-tau)` or `a`.

| Task | Method | Current accuracy | Old accuracy | Cumulative accuracy |
|---:|---|---:|---:|---:|
| 1 | thresholded CA1 | 20.20% | n.a. | 20.20% |
| 1 | unthresholded overlap | 92.40% | n.a. | 92.40% |
| 2 | thresholded CA1 | 0.00% | 20.00% | 10.00% |
| 2 | unthresholded overlap | 89.40% | 13.40% | 51.40% |
| 3 | unthresholded overlap | 91.20% | 9.70% | 36.87% |
| 4 | unthresholded overlap | 92.20% | 14.20% | 33.70% |
| 5 | thresholded CA1 | 0.00% | 4.95% | 3.96% |
| 5 | unthresholded overlap | 94.40% | 17.65% | 33.00% |

Removing the threshold restores Task-1 cumulative accuracy from 20.20% to
92.40%. At Task 5 it restores current-task accuracy from 0.00% to 94.40% and
cumulative accuracy from 3.96% to 33.00%. Therefore `W_B` contains strong
task-local discriminative information, and thresholding is the direct cause of
the near-zero current-task CA1 output.

The 33.00% cumulative result is still far below the 94.40% current result.
Thresholding is not the only continual-learning problem: the old-class overlap
readout is inconsistent with the ridge statistics retained from earlier memory
states.

## 3. Geometry before and after thresholding

Current-test geometry gives the cleanest localization because it excludes
training-code memorization:

| Task | Sparse margin | Overlap margin | Thresholded margin | Thresholded zero rows |
|---:|---:|---:|---:|---:|
| 1 | 0.07335 | 0.08384 | 0.00000 | 99.8% |
| 2 | 0.05617 | 0.04314 | 0.00000 | 100.0% |
| 3 | 0.06065 | 0.03341 | 0.00000 | 99.4% |
| 4 | 0.07155 | 0.02664 | 0.00000 | 99.8% |
| 5 | 0.07022 | 0.02148 | 0.00000 | 99.6% |

At Task 1, `s -> a` preserves and slightly increases the cosine margin on test
samples. At Task 5, `a` still has a positive margin and supports 94.40% current
accuracy. The measured test geometry becomes exactly zero at `a -> z` because
almost every row is zero.

The current-training overlap margin decreases from 0.04394 at Task 1 to
0.00973 at Task 5, so the evolving binary memory also compresses class geometry
over time. This is measurable but not enough to prevent strong current-task
unthresholded classification.

## 4. Memory drift

Task-1 test codes were held fixed and re-evaluated with each saved memory:

| Eval memory | Mean cosine of `a` vs `WB_task1` | Mean absolute `a` change | Thresholded support Jaccard |
|---:|---:|---:|---:|
| Task 1 | 1.000 | 0.0000 | 1.000 |
| Task 2 | 0.916 | 0.0283 | 0.000 |
| Task 3 | 0.876 | 0.0485 | 0.000 |
| Task 4 | 0.852 | 0.0650 | 0.0526 |
| Task 5 | 0.834 | 0.0804 | 0.000 |

The pre-threshold overlap representation drifts monotonically. By Task 5, the
mean absolute change is 0.0804, equivalent to about 3.3 overlap-count steps of
size `1/41`. This is substantial relative to the Task-1 test mean `a=0.0583`.

Ordinary cosine for `z` is mostly undefined rather than merely small: under
`WB_task1`, only 0.2% of Task-1 test samples have a nonzero `z`, and only one
slot activation exists across the whole split. The CSV therefore reports the
fraction of samples with valid nonzero-vector cosine. Thresholded-support
overlap is essentially zero under later memories. This confirms severe feature
space inconsistency, while also showing that threshold collapse precedes any
meaningful `z`-drift measurement.

## 5. Binary-memory update statistics

Memory occupancy rises from 0.9157% after Task 1 to 3.5923% after Task 5. It is
globally sparse, not saturated.

Rows are not dominated by a tiny set of slots. At Task 5, row occupancy has
median 3.5767%, P99 5.2368%, and maximum 6.1035%. Per-slot plateau counts have
median 4, P99 9, and maximum 13. Across the 25 classes, mean plateau counts per
sample range from 5.75 to 6.91, and mean synapse flips per sample range from
117.66 to 140.99. No class is starved and no small slot subset receives orders
of magnitude more plateau events.

Columns are much more skewed. At Task 5, column occupancy has median 0.7324%,
P99 35.72%, and maximum 49.76%, against a 3.5923% mean. This indicates that a
small subset of sparse-code dimensions is written across many memory rows.
The present experiment does not distinguish whether that skew comes from the
input code activation-frequency distribution or from the binary update rule;
it is a secondary memory diagnostic, not evidence that slot overuse causes the
first-task collapse.

## Answers

1. **Does pre-threshold `a` contain signal?** Yes. It has positive class margin
   on every current test split and produces 89.4%-94.4% current-task accuracy.
2. **Is thresholding the main immediate information loss?** Yes. Current-test
   P99 overlap is far below 0.5, and 99.4%-100% of thresholded test rows are zero.
3. **Does unthresholded ridge recover accuracy?** Yes. Task-1 cumulative rises
   to 92.4%; Task-5 current rises to 94.4%. Task-5 cumulative only reaches 33.0%
   because old classes remain poor.
4. **Is `W_B` drift serious?** Yes for continual consistency. Fixed Task-1
   overlap cosine falls to 0.834 and old-class unthresholded accuracy remains
   9.7%-17.65% after later tasks.
5. **Where does class geometry disappear?** The catastrophic test collapse is
   `a -> z`. The `s -> a` step progressively compresses margin but retains a
   highly readable current-task signal.
6. **What should be checked next?** First audit the CA1 formula's intended
   normalization, threshold semantics, and scale against the paper because it
   causes the first-task collapse. Then isolate continual representation drift
   by comparing frozen-memory and evolving-memory readouts. The BTSP update is
   not the first target because it already produces strong current-task `a`,
   though its column imbalance merits a later source-versus-update audit.

## Artifact definitions

- `ca1_overlap_distribution.csv`: flattened slot-activation distribution for
  current train, current test, and old test.
- `ca1_readout_ablation.csv`: thresholded and unthresholded streaming readouts.
- `memory_drift.csv`: fixed Task-1 samples under `WB_task01` through `WB_task05`.
  `nonzero_overlap` is global Jaccard overlap of active `z` entries;
  `activity_difference` is mean per-sample absolute density change.
- `memory_geometry_before_after_ca1.csv`: exact pairwise cosine means for
  current train, current test, and old test at `s`, `a`, and `z`.
- `memory_statistics.csv`: task-level row/column/slot distributions and
  class-level write/plateau means. A write is an actual 0-to-1 or 1-to-0
  synapse flip.
- `wb_checkpoints/WB_task01.npz` through `WB_task05.npz`: binary `W_B` snapshots
  plus task, seed, `k`, and `tau` metadata.

