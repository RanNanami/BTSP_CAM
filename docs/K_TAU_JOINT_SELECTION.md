# Local k × q selection for a working component-ablation baseline

**Decision: `ABLATION_BASELINE_READY`.** The one frozen configuration is
`k=96`, `q=4`, `tau=q/k=0.041666666666666664`. This is a **LOCAL WORKING
CHOICE**, not an author configuration, a paper-parity claim, or a change to the
formal reproduction. The readout is `CURRENT_WB_REPLAY_REFIT`: historical
sparse codes are replayed after each task under the current memory. In
particular, the nonlinear thresholded readout cannot use the tau=0 C/D linear
closure and is **not a strict exemplar-free BTSP-CAM algorithm**.

## Locked protocol and implementation boundary

- CIFAR100 B0-Inc5, PILOT cached frozen ViT features; class-order, random
  projection, and writer seeds all 1993. Exact top-k, m=8192, slots=4096.
- Fixed local choices: f_base=.005, beta=1, lambda=.01, p=.5, gamma=.1.
- Per-class Task1–5 split: 400 development-train and 100 validation samples,
  seed 20260928, using the previously validated `development_split` function.
  Validation data never entered writer, eligibility, or ridge fitting.
- Predeclared k `[41,64,80,88,96]`, q `[0,1,2,3,4,5,8]`; tau=q/k. For each
  k, writer and sparse codes were generated once. Every q used the same saved
  W_B checkpoints. The q=12,16 boundary extension was not triggered: no k had
  its best validation result at q=8 with the required >=0.20pp gain over q=5.
- Selection used the mean of five cumulative validation accuracies. Candidates
  within 0.10pp of the peak were tied; lower k then lower q won. Formal test
  data was accessed only after writing the frozen config, whose validation and
  overlap CSV SHA-256 values were checked by the confirmation run.

For binary W_B and s, `J=w_j^T s` is an integer and
`z_j=max(0,(J_j-q)/k)`. Thus exactly `J<=q` is truncated. Each q was fit by
replaying every seen development-training sparse code into the current W_B
coordinate system and using the existing `StreamingRidge` implementation.

## Baseline consistency and selection

The k=96, q=0 Task1–5 validation trajectory exactly reproduced the archived
working run: `90.40, 82.00, 78.7333, 76.10, 75.72%`. Newly encoded k=96
sparse codes were bitwise identical to the archived sparse cache. A fresh
independent Task1 development writer replay matched its saved W_B checkpoint
bitwise. The full20 k=96 writer checkpoints also matched all 20 archived
high-k-mechanism W_B checkpoints bitwise.

| k | q=0 mean validation | best q | best mean validation | best-q gain vs q=0 |
|---:|---:|---:|---:|---:|
| 41 | 76.500% | 0 | 76.500% | 0.000pp |
| 64 | 78.676% | 0 | 78.676% | 0.000pp |
| 80 | 79.263% | 4 | 79.469% | +0.206pp |
| 88 | 80.286% | 3 | 80.463% | +0.177pp |
| 96 | 80.591% | 5 | 80.730% | +0.139pp |

The global peak was k=96,q=5 (80.730%). k=96,q=4 achieved 80.6847%, only
0.0453pp below it, so the **predeclared tie rule** selected the lower q=4.
The same-k q=0 mean is 80.5907%; selected positive-threshold gain is only
**+0.0940pp** on validation. k=96,q=4 Task5 validation is 75.64%, compared
with q=0's 75.72%. No result was chosen because it resembled the paper.

There is a modest k×q interaction: q>0 hurts k=41 and k=64 on the five-task
mean, while an interior positive q helps k=80/88/96 by 0.14–0.21pp. It does
**not** recover the smaller-k q=0 performance deficit relative to k=96.
The peak q=5 and selected q=4 are interior, not at the search boundary.

## One frozen full20 confirmation

After freezing, a new full-data writer trajectory was run from Task1 to 20,
without changing k/q or any other setting. The formal CIFAR100 test set was
then used once for this confirmation. Accuracy fractions are recorded taskwise
in `diagnostics/k_tau_full20_confirmation.csv`; the column named
`cumulative_validation_accuracy` is the generic replay evaluator's accuracy
field, and the `evaluation_split` column identifies these rows as **test
confirmation**, not validation.

| Pipeline | Average incremental | Task20 / 100-class | Task20 old | Task20 current | Task20 current 5-way |
|---|---:|---:|---:|---:|---:|
| Paper BTSP-only reference | 87.82% | 80.15% | — | — | — |
| Previous repaired working k=96, tau=0 | 85.542% | 79.150% | — | — | — |
| Frozen repaired replay k=96, q=4 | **85.553%** | **79.180%** | **79.463%** | **73.800%** | **92.800%** |

The frozen result differs from the old working baseline by +0.011pp average
and +0.030pp final, far too small to characterize as a meaningful gain. The
paper numbers are reference values only; this experiment did not fit to them.

The selected sparse encoder remains in a sparse input regime:
`96/8192 = 1.171875%` active bits. However, the thresholded **CA1 output**
is not sparse late in the stream. For q=4, test CA1 element density is 76.42%
at Task1, 99.89% at Task5, and effectively 100% at Task20. Mean raw J rises
from 8.96 to 19.16 to 32.14 at those tasks, in step with W_B density rising
from 2.140% to 7.653% to 17.204%. Thus the small q=4 has little late-task
sparsification; it should not be presented as biologically validated CA1
competition. Full J moments and bins 0–30 plus tail are in the overlap and
confirmation CSVs.

This frozen setting is suitable as a **consistent local comparison point**
for tomorrow's component ablations, with the explicit replay and CA1-density
caveats. Even if a later ablation scores better or worse, this lock should not
be revised using its test result absent an implementation bug.

## Artifacts and integrity

- Code: `experiments/run_k_tau_joint_selection.py`
- Predeclared search: `experiments/configs/k_tau_joint_selection.yaml`
- Frozen lock: `experiments/configs/ablation_baseline_frozen.yaml`
- Diagnostics: `diagnostics/k_tau_validation_grid.csv` (175 rows),
  `diagnostics/k_tau_overlap_statistics.csv` (25 rows),
  `diagnostics/k_tau_full20_confirmation.csv` (20 rows).
- Isolated experimental packed codes and W_B checkpoints:
  `diagnostics/k_tau_joint_cache/`.
- Existing suite: `39 passed, 3 subtests passed`.
- `git diff --name-only` was empty. Formal source/config/checkpoints and prior
  diagnostics were never opened for writing; the pre/post captured SHA lines
  for the formal YAML, previous beta CSV, and sampled source files match.
  The initial shell hash listing was output-truncated, so it cannot establish
  a **complete** pre/post SHA comparison for every protected cache file.
  Separate bitwise checks covered the 20 archived k=96 W_B checkpoints.
- No commit or push was performed. Existing unrelated untracked experimental
  files were preserved.
