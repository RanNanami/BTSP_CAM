# Implementation plan and current status

> Round-2 update: the first-round uncertainties below are superseded where
> noted by `CIFAR100_PROTOCOL.md` and `PARAMETER_PROVENANCE_ROUND2.md`.

## Target benchmark audit

The first target is CIFAR-100 `B0-Inc5` from Table 1.

| item | evidence-backed result |
|---|---|
| total classes | 100 |
| initial task classes | 5 in the adopted PILOT protocol (`init_cls=5`) |
| later increment | 5 classes |
| task count | 20 equal tasks; verified against the official PILOT `DataManager` |
| class order | shuffled once with seed 1993 |
| exact order | resolved and parity-tested with PILOT's legacy NumPy RNG; see `CIFAR100_PROTOCOL.md` |
| training visibility | only current task `D_t` |
| testing | cumulative union of seen-task test sets |
| task ID at inference | not provided; one unified classifier |
| exemplar storage | prohibited |
| metrics | final Top-1 `A_B`; mean stage accuracy `Abar=(1/B) sum_b A_b` |

## Phased implementation

| phase | deliverable | status |
|---|---|---|
| 0 | immutable supplied upstream snapshot and evidence audit | COMPLETE |
| 1 | frozen-feature interface | COMPLETE for protocol validation: PILOT checkpoint alias, transforms, freeze/eval, and 768-D output verified |
| 2 | Gaussian random projection and exact k-WTA | COMPLETE |
| 3 | class-specific eligibility Eq. (4) | COMPLETE with documented constant-decay LOCAL_CHOICE |
| 4 | F/N/I gating and adaptive plateau Eq. (5) | COMPLETE |
| 5 | binary memory and per-synapse XOR Eq. (6) | COMPLETE |
| 6 | CA1 normalized threshold competition Eq. (7) | COMPLETE |
| 7 | streaming ridge statistics and solve Eq. (8) | COMPLETE |
| 8 | full CIL pipeline | COMPLETE for Layer A protocol validation; Layer B parameters remain unresolved |

## Reuse decision

- Directly reused semantically: binary initial state, active-input-only plasticity, sequential sample updates, plateau-gated binary state transitions.
- Preserved as a tested reference primitive: the exact official set/reset expression in `official_simplebtsp_step`.
- Not imported directly: upstream scripts execute CUDA sweeps at import time and expose no reusable functions. Copying them into the runtime would also retain the wrong sampling granularity for BTSP-CAM.
- Reimplemented from the BTSP-CAM paper: random sparse encoder, eligibility/gating, XOR memory, CA1, ridge, CIL orchestration.

## Formal CIFAR-100 readiness gates

1. Resolve or explicitly approve local choices for all missing hyperparameters in `PARAMETER_AUDIT.md`.
2. ~~Identify the exact ImageNet-21K ViT-B/16 checkpoint and preprocessing/feature point.~~ Resolved at PILOT-protocol evidence level.
3. ~~Confirm the authors' `B0` convention and concrete class order under seed 1993.~~ Resolved and parity-tested.
4. ~~Add CIFAR-100 data loading and frozen-feature extraction.~~ Implemented.
5. Complete and inspect the two-task, full-sample real-data protocol smoke before any `8192 x 4096` experiment.

The project is not yet qualified for a paper-number comparison because gate 1
remains unresolved. Protocol/backbone parity alone does not supply the missing
BTSP-CAM numeric parameters.
