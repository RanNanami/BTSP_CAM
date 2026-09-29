# BTSP-CAM representation/readout diagnosis

## Scope

This diagnostic uses the same cached features and protocol as the formal run:
CIFAR-100, B0-Inc5, PILOT class order, seed 1993, PILOT preprocessing, frozen
ViT-B/16 ImageNet-21K, current-task-only streaming training, cumulative
seen-class evaluation, and no replay. Only Tasks 1-5 were run. No model or
protocol hyperparameter was changed.

## Main result

At Task 5, frozen ViT features remain strongly separable, the Gaussian+k-WTA
code retains useful but reduced class information, and the CA1 representation
collapses to chance-level current-task discrimination.

| Method | Current task, all 25 logits | Current task, restricted 5 logits | Cumulative 25-class | Old classes | Ridge condition |
|---|---:|---:|---:|---:|---:|
| ViT -> ridge | 95.40% | 99.00% | 94.64% | 94.45% | 7.78e7 |
| ViT -> Gaussian+k-WTA -> ridge | 71.00% | 88.60% | 65.16% | 63.70% | 6.69e6 |
| Full BTSP-CAM | 0.00% | 20.20% | 3.96% | 4.95% | 1.46e2 |

The Task-5 cumulative drop is 29.48 percentage points from ViT to sparse code
and 61.20 points from sparse code to full BTSP-CAM. The largest loss therefore
occurs after the sparse code, inside the BTSP-memory/CA1 representation path.

## Accuracy across the five tasks

| Task | ViT ridge | Sparse ridge | BTSP-CAM |
|---:|---:|---:|---:|
| 1 | 98.20% | 78.00% | 20.20% |
| 2 | 96.70% | 55.70% | 10.00% |
| 3 | 95.93% | 56.93% | 6.93% |
| 4 | 94.95% | 60.30% | 5.20% |
| 5 | 94.64% | 65.16% | 3.96% |

Baseline A rules out the frozen backbone, PILOT preprocessing, class order,
streaming sufficient statistics, and cumulative evaluation protocol as the
primary failure. Baseline B shows a real cost from random projection+k-WTA,
but its Task-5 restricted 5-way accuracy remains 88.60%, so the sparse code is
not the catastrophic bottleneck.

## Representation geometry

Task-5 current-task training geometry is:

| Representation | Same-class cosine | Different-class cosine | Margin | Nonzero density | Zero rows |
|---|---:|---:|---:|---:|---:|
| ViT `h` | 0.449636 | 0.285442 | 0.164194 | 1.000000 | 0.00% |
| Sparse `s` | 0.085608 | 0.036242 | 0.049367 | 0.005005 | 0.00% |
| CA1 `z` | 0.001459 | 0.001003 | 0.000456 | 0.001331 | 4.08% |

The class-cosine margin loses 99.08% of its sparse-code value at `s -> z`.
The same collapse is already present at Task 1: the margin falls from 0.051544
for `s` to 0.000108 for `z`. This precedes cross-task forgetting.

The stronger signal is the train/test activity mismatch. Current-task training
CA1 density ranges from 7.53e-4 to 1.33e-3, while cumulative test CA1 density is
only 4.88e-7 at Task 1 and 5.76e-6 at Task 5. At Task 1, nearly every test CA1
vector is therefore zero even though the training representation is nonzero.
This is consistent with a CA1 threshold that admits memory overlap for training
codes but almost never for nearby unseen codes.

Sparse-code diagnostics do not show collision collapse. Every task has 2,500
unique codes for 2,500 samples, zero exact collisions, exact density 41/8192,
and higher within-class than between-class cosine.

## Ridge numerics

At Task 5, the regularized ridge condition estimates are 7.78e7 for ViT,
6.69e6 for sparse code, and 1.46e2 for CA1. ViT and sparse ridge obtain high or
useful accuracy despite much larger condition estimates. CA1 is not the most
ill-conditioned representation.

The unregularized Task-5 `A` spectra show `eig_min=0` for sparse and CA1 because
rank deficiency is proven by the active-dimension/sample bounds. The exact ViT
rank is 768. For the large matrices, the diagnostic reports a deterministic
512-column principal-submatrix rank lower bound and an exact rank upper bound,
not an unjustified exact rank. At Task 5 these bounds are 440-6965 for sparse
and 512-4095 for CA1. Participation-ratio effective ranks are 300.05 and
2283.53 respectively. These algebraic rank measures do not recover the class
geometry lost by CA1.

`ridge_condition` is LAPACK's one-norm estimate for `A + gamma I`; `eig_min`
and `eig_max` describe unregularized `A`. They are intentionally not presented
as a quotient of one another.

## BTSP learning dynamics

Adaptive plateau activity is low but not stuck at zero:

| Task | Mean `f_eff` | Plateau rate | 0->1 flips | 1->0 flips | Net flips | Memory active | CA1 train density |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.001629 | 0.001637 | 325,398 | 18,130 | 307,268 | 0.009157 | 0.000837 |
| 2 | 0.001491 | 0.001484 | 279,988 | 30,781 | 249,207 | 0.016584 | 0.000753 |
| 3 | 0.001505 | 0.001501 | 272,846 | 41,732 | 231,114 | 0.023472 | 0.000803 |
| 4 | 0.001494 | 0.001517 | 267,950 | 50,872 | 217,078 | 0.029941 | 0.000953 |
| 5 | 0.001530 | 0.001542 | 262,229 | 61,511 | 200,718 | 0.035923 | 0.001331 |

For all 25 newly introduced classes, the first class sample has exactly
`F=1`, `N=0`, `f_eff=0`, and zero plateaus. Positive `f_eff` begins at sample 2
for every class, so update-before-gating creates a one-sample zero-plasticity
prefix rather than a prolonged plateau shutdown. Memory updates remain strongly
net-potentiating throughout the five tasks.

## Diagnosis and next check

The immediate failure is the CA1 mapping

`z = max(0, W_B s / k - tau)`.

With `k=41` and the local-choice `tau=0.5`, a slot must match at least 21 active
code bits before it emits a positive value. Meanwhile, within-class sparse-code
cosine 0.07-0.10 corresponds to only about 3-4 shared active bits on average.
The observed near-zero test density and chance-level 5-way accuracy are the
direct empirical consequence. This is a localization result, not a proposal to
tune `tau`.

The next experiment should instrument the pre-threshold overlap
`a = W_B s / k` for train/current-test/old-test samples and report its quantiles,
class margins, fraction above `tau`, and per-task drift under later `W_B` states.
Add two diagnostic readouts without selecting parameters: ridge on unthresholded
`a`, and ridge on the existing thresholded `z`. This will separate loss caused
by the learned memory geometry from loss caused specifically by thresholding,
and it will quantify representation drift because old ridge statistics were
formed under earlier memory states while inference uses the latest `W_B`.

## Artifacts

- `diagnostics/representation_ablation.csv`: requested accuracy/readout table.
- `diagnostics/accuracy_drops.csv`: A-to-B and B-to-C drops per task.
- `diagnostics/ridge_spectrum.csv`: spectrum, rank bounds, effective rank, and condition definitions.
- `diagnostics/representation_geometry.csv`: class geometry for `h`, `s`, and `z`.
- `diagnostics/sparse_diagnostics.csv`: collision and activation-frequency diagnostics.
- `diagnostics/btsp_learning_dynamics.csv`: task-level memory and gating dynamics.
- `diagnostics/btsp_early_class_dynamics.csv`: first 1/5/20 samples for each new class.

