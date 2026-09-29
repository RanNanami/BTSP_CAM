# BTSP-CAM core

Independent implementation of the BTSP-CAM core model. It is designed to run
the complete exemplar-free CIFAR-100 class-incremental pipeline; it is **not**
an exact reproduction of the paper's reported numerical results because several
method hyperparameters were not published.

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

## Current validation

All 20 formal-size CIFAR-100 tasks have been executed. The model processed all
50,000 training images task-by-task and evaluated all 10,000 test images after
Task 20. Every ridge solve succeeded, every task checkpoint produced identical
predictions after reload, the binary memory invariant held, and CA1 remained
nonzero. Total task runtime was 5,038.9 seconds (84.0 minutes); peak working set
was 3.31 GB. Task 1 was also resumed in a separate process to train Task 2.

Final cumulative accuracy was 0.0408 and average incremental accuracy was
0.05167. These values are reported transparently and were not used to tune the
unpublished parameters. They must not be presented as competitive or
paper-parity results.

The five-task representation/readout diagnosis localizes the largest loss to
the BTSP-memory/CA1 path: Task-5 cumulative accuracy is 94.64% for ViT ridge,
65.16% for sparse-code ridge, and 3.96% for full BTSP-CAM. See
`docs/PERFORMANCE_DIAGNOSIS.md` and the machine-readable files under
`diagnostics/`. No hyperparameter was changed for this diagnosis.

The follow-up CA1 diagnosis shows that pre-threshold overlap remains strongly
readable for the current task: unthresholded overlap ridge reaches 92.40% at
Task 1 and 94.40% current-task accuracy at Task 5. The configured threshold
makes 99.4%-100% of current-test CA1 rows zero, while continuing memory updates
also drift old overlap features. See `docs/CA1_DIAGNOSIS_ROUND1.md`.

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
