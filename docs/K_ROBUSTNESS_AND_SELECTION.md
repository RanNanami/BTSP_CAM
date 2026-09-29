# K robustness, path decomposition, and train-only selection

Status: complete. These are independent experimental results, not a claim
about the authors' undisclosed `k` or a change to the formal reproduction.

## Locked protocol

- CIFAR-100 B0-Inc5, 20 tasks, class-order seed 1993, frozen cached ViT
  features, unchanged train/test protocol and training order.
- Experimental coordinate-aligned ridge with `z=a=W_B s/k`, `tau=0`.
  All other hyperparameters and formal writer/trace implementation are held
  fixed. The formal default remains untouched.
- RF seeds `[1993, 3001, 7001]`; independent writer RNG seeds
  `[1993, 4001, 8001]`. Each of nine crossings uses both `k=41` and
  `k=128`, paired by seed combination. Class order never changes.
- Path grid fixed before results: `k=[41,64,96,128,160]`. Sparse-only uses
  the same ridge normal equations in fixed sparse coordinates; fixed-WB1
  and evolving-WB use 4096-dimensional overlap features.
- Write-budget control: only `k=128`, with one global plateau-probability
  multiplier `41/128=0.3203125`, fixed before the run. The intended match
  target is cumulative gross binary flips, not task-by-task test accuracy.
- Train-only choice grid `[32,41,64,96,128,160]`. Each of the first five
  training tasks has a seed-20260928 stratified 400/100 development/validation
  split per class. Validation samples never update eligibility, `W_B`, `C/D`
  or ridge. Select largest Task1–5 mean cumulative validation accuracy;
  exact ties go to smaller `k`. The development models are discarded.

## Train-only selection

| k | Task1–5 validation average | Task5 validation cumulative |
|---:|---:|---:|
| 32 | 76.0453% | 69.76% |
| 41 | 76.50% | 71.20% |
| 64 | 78.6760% | 73.28% |
| 96 | 80.5907% | 75.72% |
| 128 | 80.7807% | 75.92% |
| 160 | **81.5847%** | 75.84% |

The locked rule selected `k=160` **before the selected run accessed any test
split**. A fresh full-training-set Task1–20 run used `k=160`; test was evaluated
once at Task20. Its 100-class cumulative accuracy was **81.16%** (old 81.3053%,
current 78.40%, current 5-way 92.60%). No Task1–20 test average exists for
this selected retraining run by design. The validation average is not a test
average and must not be compared as if it were the paper's 87.82% metric.

## Paired seed robustness

All 18 trajectories completed Task1–20, giving 360 unique task rows and
nine complete pairs. Differences below are k=128 minus k=41 in percentage
points (pp), not relative percentages.

| RF seed | Write seed | k41 final | k128 final | Paired final delta | Paired average delta |
|---:|---:|---:|---:|---:|---:|
| 1993 | 1993 | 73.30% | 80.62% | +7.32pp | +5.80pp |
| 1993 | 4001 | 73.53% | 80.59% | +7.06pp | +5.82pp |
| 1993 | 8001 | 73.55% | 80.49% | +6.94pp | +5.86pp |
| 3001 | 1993 | 73.17% | 79.94% | +6.77pp | +5.11pp |
| 3001 | 4001 | 72.99% | 80.13% | +7.14pp | +5.41pp |
| 3001 | 8001 | 72.97% | 80.02% | +7.05pp | +5.16pp |
| 7001 | 1993 | 72.63% | 80.22% | +7.59pp | +5.59pp |
| 7001 | 4001 | 72.76% | 80.16% | +7.40pp | +5.86pp |
| 7001 | 8001 | 72.83% | 79.81% | +6.98pp | +5.56pp |

| Statistic across nine crossings | Average incremental | Task20 cumulative |
|---|---:|---:|
| k41 mean (sample SD) | 80.57% (0.28pp) | 73.08% (0.33pp) |
| k128 mean (sample SD) | 86.14% (0.28pp) | 80.22% (0.29pp) |
| Paired delta mean (sample SD) | **+5.57pp (0.29pp)** | **+7.14pp (0.25pp)** |
| Paired delta range | +5.11 to +5.86pp | +6.77 to +7.59pp |
| Positive / negative signs | **9 / 0** | **9 / 0** |
| Nominal t(8) 95% interval | +5.35 to +5.80pp | +6.94 to +7.33pp |
| Crossed bootstrap 95% interval | +5.24 to +5.83pp | +6.97 to +7.43pp |

The crossed interval independently resamples the three RF and three writer
seeds (20,000 replicates, seed 1993), respecting the shared row/column
factors. This CI computation was fixed while the seed runs were still in
progress. The nominal t interval treats the nine crossings as independent,
which is not strictly true; report it only for comparison. With just three
seeds on each axis, neither interval is a reliable population-wide guarantee.

Task20 old-class accuracy rises from 73.31% mean at k41 to 80.45% at k128;
current-task accuracy rises 68.76% to 75.84%; current 5-way rises 88.38%
to 91.33%. The corresponding paired deltas are positive for **all nine**
crossings in each of these three metrics. Thus the cumulative gain is not
only a short-lived new-task effect or only an old-class effect.

## Path decomposition and write budget

All five predeclared k values completed 20 tasks. Values are percentages;
"average" is the mean of Task1–20 cumulative test accuracy.

| k | Sparse-only average / final | Fixed-WB1 average / final | Evolving-aligned average / final |
|---:|---:|---:|---:|
| 41 | 66.22 / 64.62 | 80.46 / 72.62 | 80.68 / 73.30 |
| 64 | 68.52 / 69.97 | 83.55 / 77.08 | 83.60 / 77.10 |
| 96 | 71.06 / 73.29 | 85.86 / 79.27 | 85.54 / 79.15 |
| 128 | 72.45 / 75.12 | 86.60 / 80.39 | 86.49 / 80.62 |
| 160 | 74.61 / 76.90 | 87.08 / 81.43 | 86.90 / 81.16 |

From k=41 to k=128, final sparse-only accuracy rises **10.50pp**, fixed-WB1
**7.77pp**, and evolving-aligned **7.32pp**. At k=128, continual writes add
only **0.23pp** relative to its WB1-frozen readout; at k=160 they are **0.27pp
lower**. Thus the observed final k benefit is already present without
post-Task1 writes. This is a performance-path comparison, not an additive
information-theoretic decomposition: `W_B s/k` is a linear transform of `s`,
so a fixed `W_B` cannot create new information. Dimensions, feature geometry
and the fixed gamma's effective regularization differ across readouts.

At Task20, increasing k=41 to k=128 changes the sparse-code same-minus-different
cosine margin **0.06795 → 0.09050**, the top-1%-bit activation share
**19.38% → 13.29%**, and the entropy effective support **2157 → 3019 bits**.
Raw cross-task mean overlap J rises **1.011 → 6.531**, and `P(J>=5)` rises
**1.37% → 67.19%**. These raw overlap metrics are *not scale normalized*:
larger k mechanically raises them. They cannot by themselves be read as
increased harmful collision. The class margin and downstream readouts give
the relevant companion evidence.

The predeclared write-budget branch uses k=128 and a global plateau-rate
multiplier 41/128. At Task20 it has **6,803,692** cumulative gross flips versus
**6,315,038** for k=41 (+7.74%), and **21,311,700** for ordinary k=128.
Its final accuracy is **80.72%** (average 86.91%), versus 73.30% (80.68%)
for k=41 and 80.62% (86.49%) for ordinary k=128. Therefore tripling gross
write count is not necessary for the k=128 gain under this control. The
control does **not** perfectly match memory state: Task20 `W_B` density is
11.35%, compared with 9.37% for k=41 and 20.61% for ordinary k=128. Plateau
timing and write locations also change. It does not establish a clean causal
percentage for representation size versus occupancy.

## Interpretation versus the paper

`k=128` was a **test-screen candidate** from the prior unresolved-parameter
experiment. Its nine-pair confirmation does not identify the authors' setting.
`k=160` was selected **only from current Task1–5 train-side validation** in
this round and then evaluated once after a fresh full20 training run. It is
the upper boundary of the predeclared grid, not a demonstrated global optimum.

Numerically, the independently evaluated evolving k160 path has an 86.90%
Task1–20 test average and 81.16% Task20 final. Compared with the paper's
reported 87.82% / 80.15%, that is −0.92pp / +1.01pp. The selected run's
Task20 final independently reproduces the path run's 81.16%; it did not
evaluate test at intermediate tasks, so its own test average is **not
available**. The path average is a separate predeclared diagnostic, never
used for k selection. Numerical proximity cannot establish strict paper
reproduction: coordinate-aligned ridge and tau=0 are experimental repairs,
not the published default algorithm, and the paper does not disclose k.

## Data and reproducibility

- `experiments/configs/k_robustness_and_selection.yaml`: locked grids/seeds/rule.
- `experiments/run_k_robustness_and_selection.py`: independent runner.
- `diagnostics/k_seed_runs_taskwise.csv`: per-task paired trajectories.
- `diagnostics/k_seed_robustness.csv`: nine-pair summary.
- `diagnostics/k_path_decomposition.csv`: full20 layer and budget branches.
- `diagnostics/k_train_only_selection.csv`: five-task validation and one final
  selected-run test record.

Validation checks: 360 unique `(RF, writer, k, task)` seed rows; nine unique
paired summary rows; 320 unique `(run, task, branch)` path rows; 30 validation
rows plus exactly one selected-run final test row. Each of Task1–5 has 400
development and 100 validation training examples per class, with disjoint
indices. The default-seed k41/k128 seed trajectories match the independent
path experiment at all 20 tasks exactly. The selected k160 final result
matches the independent k160 evolving path (81.16%). Existing test suite:
39 passed and 3 subtests passed. The SHA-256 values of the formal default
YAML and core `model.py`, `eligibility.py`, and `binary_memory.py` are unchanged
from before this experiment.
