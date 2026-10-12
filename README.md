# BTSP-CAM reproduction and research diagnostics

Independent implementation and audit of the BTSP-CAM core model. The formal
path runs an exemplar-free CIFAR-100 class-incremental pipeline. Separate
experimental readouts and diagnostics investigate ambiguities in the public
description; they are **not** claimed as the authors' algorithm or as an exact
reproduction of the paper's reported results. Start with the
[current research status](docs/CURRENT_RESEARCH_STATUS.md) and
[documentation index](docs/README.md).

ImageNet-R server infrastructure is ready; a paper-comparable run remains
conditional on verifying the provenance of PILOT-compatible train/test folders
(`FINAL_CASE = B`: PILOT expects pre-split data, origin unresolved). The fixed
k=96 cross-dataset transfer uses an experimental current-W_B replay-refit
readout alongside a literal Eq.(8) streaming control; neither is a newly claimed
paper parameter configuration. See the [protocol audit](docs/IMAGENET_R_PROTOCOL_AUDIT.md) and
[server runbook](docs/IMAGENET_R_SERVER_RUN.md). No ImageNet-R accuracy has
been produced locally.

## Environment and install

The validated environment is Windows 11, Python 3.14, CPU PyTorch 2.14,
torchvision 0.29, timm 1.0.29, NumPy 2.3, and SciPy 1.18.

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src"
python -m pytest -q
```

The runner downloads the official torchvision CIFAR-100 train/test split under
`data/` when it is absent. The pretrained ViT checkpoint must load successfully;
there is no random-backbone fallback.

## Pipeline

```text
CIFAR-100 image
  -> frozen ViT-B/16 ImageNet-21K (768-D)
  -> frozen Gaussian projection and exact k-WTA
  -> class eligibility traces
  -> familiarity / novelty / interference
  -> adaptive slot-wise plateau sampling
  -> synapse-wise Bernoulli XOR binary memory
  -> normalized continuous CA1 representation
  -> streaming ridge sufficient statistics
  -> seen-class prediction
```

The protocol is fixed to PILOT CIFAR-100 B0-Inc5: 20 five-class tasks, seed
1993, the parity-tested class order, current-task-only training, cumulative
seen-class testing, no replay, and no task ID at inference. Preprocessing is the
actual PILOT `cifar224` transform and does not apply timm normalization.

## Run and resume

Run one formal-size task (`m=8192`, `S=4096`):

```powershell
python experiments/run_btsp_cam_cifar100.py --config configs/btsp_cam_cifar100.yaml --max-tasks 1
```

Continue through Task 2 from the saved state:

```powershell
python experiments/run_btsp_cam_cifar100.py --config configs/btsp_cam_cifar100.yaml --resume results/btsp_cam_cifar100/checkpoint_task1.npz --max-tasks 2
```

Continue through all 20 tasks:

```powershell
python experiments/run_btsp_cam_cifar100.py --config configs/btsp_cam_cifar100.yaml --resume results/btsp_cam_cifar100/checkpoint_task2.npz --max-tasks 20
```

Each task writes accuracy, average incremental accuracy, plateau statistics,
binary-memory activity, CA1 density, ridge solve/condition status, runtime,
memory use, and an `.npz` checkpoint. Every checkpoint is immediately reloaded
and checked for identical predictions.

## Current research status

### A. Formal paper-description baseline

The **STRICT PUBLIC-DESCRIPTION BASELINE / FORMAL REPRODUCTION PATH** implements
the published equations, including literal streaming ridge sufficient
statistics from Eq. (8), with explicitly documented local choices for
unpublished hyperparameters. On CIFAR-100 B0-Inc5 its Task20 final cumulative
accuracy is **4.08%** and its average incremental accuracy is **5.167%**.
All 20 tasks completed; checkpoint reloads, binary memory, and ridge solves
passed their checks. This is a reproduction reference, **not** the current
working baseline or a claim of paper parity. The five-task diagnosis found
94.64% for ViT ridge, 65.16% for sparse-code ridge, and 3.96% for the full
thresholded path at Task5; see [performance diagnosis](docs/PERFORMANCE_DIAGNOSIS.md)
and [CA1 diagnosis](docs/CA1_DIAGNOSIS_ROUND1.md).

### B. Coordinate-consistent experimental readout

**This is NOT claimed as the paper algorithm.** The binary memory `W_B`
evolves across tasks, while literal streaming ridge retains historical `A/B`
formed under earlier CA1 feature coordinates. Experimental coordinate-consistent
variants recompute or reconstruct readout statistics in the *current* `W_B`
coordinate system. At `tau=0`, aggregated sparse-code `C/D` statistics give
an exact exemplar-free algebraic alignment; for positive threshold, the local
working implementation instead replays historical **training sparse codes**
and refits ridge. That replay variant is **not strict exemplar-free**. See the
[paper readout semantics audit](docs/PAPER_READOUT_SEMANTICS_AUDIT.md),
[coordinate-aligned ridge derivation](docs/COORDINATE_ALIGNED_RIDGE.md), and
[full20 result](docs/COORDINATE_ALIGNED_FULL20.md).

### C. CIFAR-100 local working baseline

The currently frozen *local experimental* baseline uses `k=96`, discrete
overlap-count threshold `q=4` (`tau=4/96`), and current-`W_B` historical
training sparse-code replay/refit. It reaches **85.553%** average incremental
accuracy and **79.180%** Task20 final accuracy. It was selected by the
documented training-side validation rule and is neither an author-exact
configuration nor the formal default. Historical sparse-code replay means
this result cannot be described as strict exemplar-free. Details:
[k × q selection and full20 confirmation](docs/K_TAU_JOINT_SELECTION.md).

### D. CUB-200 frozen-parameter transfer

The CIFAR-selected local `k=96, q=4` settings were transferred to the official
CUB-200 split **without CUB retuning**. Under the same local current-`W_B`
replay/refit readout, CUB-200 B0-Inc10 reaches **87.8606%** average
incremental and **80.6006%** Task20 final accuracy: strong frozen-parameter
cross-dataset transfer under the *local coordinate-consistent readout*.
This is not a paper-exact or directly paper-Table-1-comparable result, and
should not be described as “beating the paper.” See the
[CUB baseline](docs/CUB200_FULL_BASELINE.md),
[leakage forensic audit](docs/CUB200_LEAKAGE_FORENSIC_AUDIT.md), and
[readout decomposition](docs/CUB200_READOUT_CAUSAL_DECOMPOSITION.md).

### E. Literal streaming versus coordinate-consistent readout

All values below are percent; `Abar` is the mean of taskwise cumulative
accuracy and `Final` is the final-task cumulative accuracy. These are local
diagnostic paths, not assertions about the authors' private implementation.

| Dataset / local readout | Abar | Final |
|---|---:|---:|
| CIFAR current-`W_B` replay/refit | 85.553 | 79.180 |
| CIFAR literal streaming Eq. (8) | 31.934 | 20.630 |
| CUB current-`W_B` replay/refit | 87.8606 | 80.6006 |
| CUB literal streaming Eq. (8) | 54.8096 | 33.3276 |
| CUB stable-`W_B1` streaming control | 84.9342 | 77.2006 |

These diagnostics isolate a temporal coordinate mismatch between evolving
`W_B` features and historical ridge statistics **in this local pipeline**.
They do not establish that Eq. (8) is wrong, that the authors omitted an
alignment step, or that the authors used hidden replay.

### F. k robustness and component diagnostics

The table lists already saved full20 experimental results; entries are not
one common paper configuration. The k96 row is the frozen `q=4` replay
baseline; the other rows use `tau=0` coordinate-aligned experimental readouts.
Class order, RF and writer seeds are fixed at 1993 for these displayed paths.

| k | Local readout | Abar | Final |
|---:|---|---:|---:|
| 41 | aligned, `tau=0` | 80.685 | 73.300 |
| 64 | aligned, `tau=0` | 83.601 | 77.100 |
| 96 | replay/refit, `q=4` | 85.553 | 79.180 |
| 128 | aligned, `tau=0` | 86.487 | 80.620 |
| 160 | aligned, `tau=0` | 86.901 | 81.160 |
| 768 | aligned, `tau=0` | 89.390 | 83.920 |
| 1024 | aligned, `tau=0` | 89.552 | 84.340 |
| 6144 | aligned, `tau=0` | 90.243 | 84.580 |
| 7168 | aligned, `tau=0` | 89.745 | 83.750 |

Increasing `k` substantially improves sparse/readout representation in the
tested region, but performance eventually peaks and declines. Under matched
controls, much of the high-`k` gain is representation/readout-driven; learned
BTSP `W_B` provides only a modest additional advantage at high `k`. This does
**not** establish that BTSP learning is useless. See
[k robustness](docs/K_ROBUSTNESS_AND_SELECTION.md),
[high-k stress](docs/HIGH_K_STRESS_TEST.md), and
[mechanism controls](docs/HIGH_K_MECHANISM_DECOMPOSITION.md).

Local Section-6-style component ablations of trace gating, collision term,
and LTD change Abar by **−0.698/−0.156/−0.078 pp** at k96 and
**−0.125/−0.043/−0.057 pp** at k1024, respectively. The k1024 Task20 LTD
effect is about **−0.90 pp**. Strong Section-6-style component effects were
not recovered under either k=96 or k=1024 with the local coordinate-consistent
readout. These are documented *local operationalizations*, not paper-exact
ablations or evidence that the paper's ablation is wrong. See
[operationalization](docs/PAPER_ABLATION_OPERATIONALIZATION.md),
[k96 ablation](docs/PAPER_STYLE_COMPONENT_ABLATION.md), and
[k1024 ablation](docs/CIFAR_K1024_COMPONENT_ABLATION.md).

## Important caveat: prototype stability vs readout consistency

The paper reports CA1 prototype similarity `>0.9` and describes a “slowly
drifting” CA1 space. In existing **local pre-threshold `a`** CIFAR Task1→20
diagnostics, class-prototype cosine is approximately 0.914 at k96, 0.924 at
k128, 0.930 at k160, and 0.966 at k1024. The public Figure-4 prototype
construction is not fully specified, so these are not paper-exact replicas.
Prototype stability diagnoses first-order class centers; it does not
mathematically imply that historical ridge sufficient statistics remain
aligned. Ridge depends on second-order `Z^T Z` and cross-statistics `Z^T Y`.
**This does not prove the paper implementation is incorrect.** It shows only
that prototype similarity alone is insufficient evidence for historical-
statistic coordinate consistency. See the
[Figure-4-style drift audit](docs/FIGURE4_STYLE_PROTOTYPE_DRIFT_AUDIT.md)
and [multi-k writer drift audit](docs/K_WRITER_DRIFT_READONLY_AUDIT.md).

## Reproducibility assets and boundaries

The repository publishes the formal implementation, experimental scripts,
parameter locks, reports and a curated set of small results. It intentionally
does **not** publish CIFAR/CUB images, ViT embeddings, sparse-code matrices,
writer/ridge checkpoints or large forensic logs. Rerunning CUB diagnostics
requires the official CUB-200-2011 archive, the pinned LAMDA-PILOT checkout
(`7a6e904c5bc5cb7a4e1823b3434020be27469b63`), and the documented
pretrained ViT family. Scripts use repository-relative paths and do not embed
the original machine's directories. The
[result index](docs/README.md#public-lightweight-result-index) identifies
which lightweight artifacts are available in a fresh clone; a report's
local-only detailed artifact is not silently represented as public data.

## Model API

```python
import yaml
from btsp_cam import BTSPCAM, BTSPCAMConfig
from btsp_cam.backbone import PilotFrozenViT

raw = yaml.safe_load(open("configs/btsp_cam_cifar100.yaml", encoding="utf-8"))
backbone = PilotFrozenViT(pretrained=True)
model = BTSPCAM(BTSPCAMConfig.from_mapping(raw), backbone=backbone)

model.learn_task(train_loader, task_id=1)
metrics = model.evaluate(test_loader)
model.save("checkpoint_task1.npz")

model = BTSPCAM.load("checkpoint_task1.npz", backbone=backbone)
model.learn_task(next_train_loader, task_id=2)
```

The checkpoint contains `W_B`, eligibility traces, ridge `A/B/W_out`, seen
classes, task index, `R/b`, all model RNG states, train-order state, config,
class order, and backbone metadata. It never contains old raw training samples.

## Parameter provenance

Every model parameter in `configs/btsp_cam_cifar100.yaml` has a `value`,
`source`, and `status`.

| parameter | value | status |
|---|---:|---|
| `m` / `code_dim` | 8192 | `PAPER_SPECIFIED` |
| `S` / `memory_slots` | 4096 | `PAPER_SPECIFIED` |
| `k` | 41 | `PLAUSIBLE_INHERITANCE` |
| `f_q_base` | 0.005 | `PLAUSIBLE_INHERITANCE` |
| synaptic `p` | 0.5 | `PLAUSIBLE_INHERITANCE` |
| `eta` | 1.0 | `DERIVED` |
| `beta` | 1.0 | `LOCAL_CHOICE` |
| trace decay | 0.01 | `LOCAL_CHOICE` |
| `tau` | 0.5 | `LOCAL_CHOICE` |
| `gamma` | 0.1 | `LOCAL_CHOICE` |

These values were not tuned toward the paper's accuracy. The local choices are
the non-degenerate values already exercised by the real-CIFAR protocol smoke.

## Known specification ambiguity and caching

`PAPER_SPECIFICATION_AMBIGUITY`: printed Algorithm 1 updates eligibility before
computing familiarity. A new class's first sample can therefore produce `F=1`,
`N=0`, and zero plateau probability. The implementation preserves that order.

Evaluation features are cached safely because the test transform is
deterministic. Training uses PILOT's stochastic crop/flip; the runner fixes one
seeded augmentation per image and task and caches the resulting feature. This
is explicitly recorded as `ENGINEERING_CHOICE`, not claimed to be equivalent to
re-augmenting images on repeated epochs.
