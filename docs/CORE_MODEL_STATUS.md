# BTSP-CAM core model status

Status: **BTSP-CAM core model is runnable across all 20 CIFAR-100 tasks.** This
is an independent core implementation, not a paper-number reproduction.

## Complete pipeline

CIFAR-100 image -> frozen PILOT ViT-B/16 ImageNet-21K -> 768-D feature ->
Gaussian `R h + b` -> exact k-WTA -> class eligibility -> F/N/I -> adaptive
slot plateau -> synapse-wise XOR binary memory -> normalized continuous CA1 ->
streaming ridge -> unified seen-class prediction.

Task timing follows printed Algorithm 1: all current-task memory updates finish
before current-task `Z_t` is constructed and added to ridge `A/B`. No old train
sample or feature is read for learning a later task.

## Checkpoint contract

Each checkpoint contains binary `W_B`, eligibility, ridge `A/B/W_out`, seen
classes, task index, frozen `R/b`, projection/plateau/synapse RNG states,
train-order state, provenance-bearing config, exact class order, metrics history,
and backbone metadata. Task 1 was saved, the process exited, Task 1 was loaded in
a new process, and Task 2 learning continued successfully. The run then
continued through Task 20. All 20 checkpoints passed full cumulative-test
prediction equality immediately after reload.

## Formal-size validation

| item | Task 1 | Task 2 after resume | Task 20 |
|---|---:|---:|---:|
| train samples | 2,500 | 2,500 | 2,500 |
| cumulative test samples | 500 | 1,000 | 10,000 |
| cumulative accuracy | 0.202 | 0.100 | 0.0408 |
| current accuracy | 0.202 | 0.000 | 0.148 |
| plateau probability mean | 0.001629 | 0.001491 | 0.001439 |
| memory active fraction | 0.009157 | 0.016584 | 0.093686 |
| CA1 nonzero density | 0.000837 | 0.000753 | 0.050123 |
| ridge status | ok | ok | ok |
| ridge condition estimate | 15.02 | 12.96 | 94,202.92 |
| runtime | 234.9 s | 229.2 s | 290.8 s |
| reload prediction match | exact | exact | exact |

The formal arrays are `R=[8192,768]`, `b=[8192]`,
`W_B=[4096,8192]`, eligibility `[100,8192]`, and `A=[4096,4096]`.
After Task 20, `B` and `W_out` are `[4096,100]`. `W_B` remains binary.
The complete task runtime was 5,038.9 seconds (84.0 minutes), with a maximum
recorded working set of 3.31 GB. A fresh process independently loaded the final
checkpoint and reproduced accuracy 0.0408 on all 10,000 test images.

The low Task 2 accuracy is not hidden and was not used for tuning. The run
establishes execution, persistence, resume behavior, and numerical stability;
it does not establish competitive accuracy.

## Non-paper parameter policy

| parameter | value | status |
|---|---:|---|
| k | 41 | PLAUSIBLE_INHERITANCE |
| base plateau | 0.005 | PLAUSIBLE_INHERITANCE |
| synaptic flip p | 0.5 | PLAUSIBLE_INHERITANCE |
| eta | 1.0 | DERIVED |
| beta | 1.0 | LOCAL_CHOICE |
| trace decay | 0.01 | LOCAL_CHOICE |
| tau | 0.5 | LOCAL_CHOICE |
| gamma | 0.1 | LOCAL_CHOICE |

Training-feature caching fixes one seeded PILOT augmentation per image and task
and is labeled `ENGINEERING_CHOICE`. Evaluation caching is deterministic and
protocol-preserving.

## Re-run all tasks from Task 2

```powershell
$env:PYTHONPATH = "$PWD\src"
python experiments/run_btsp_cam_cifar100.py --config configs/btsp_cam_cifar100.yaml --resume results/btsp_cam_cifar100/checkpoint_task2.npz --max-tasks 20
```

Final verification: 32 tests plus 3 parameterized subtests passed.
