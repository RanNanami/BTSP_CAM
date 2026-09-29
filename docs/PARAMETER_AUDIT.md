# Parameter audit

> Historical first-round audit. For current statuses and search evidence, use
> `PARAMETER_PROVENANCE_ROUND2.md`. In particular, B0-Inc5, class order,
> preprocessing, and the PILOT checkpoint path are no longer protocol blockers.

Statuses are restricted to the requested vocabulary. Values used only by the synthetic smoke test are not presented as paper hyperparameters.

| parameter | symbol | value | source | location | status | implementation |
|---|---|---:|---|---|---|---|
| official input dimension | `m` | 25000 | supplied upstream code | feedforward `:32` | BTSP_OFFICIAL_CODE | reference only |
| official output dimension | `n` | 39000 | supplied upstream code | feedforward `:31` | BTSP_OFFICIAL_CODE | reference only |
| official input activity probability | `fp` | 0.005 | supplied upstream code | feedforward `:34` | BTSP_OFFICIAL_CODE | reference only |
| official plateau probability | `fq` | 0.005 | supplied upstream code | feedforward `:25` | BTSP_OFFICIAL_CODE | reference only |
| official LTP event probability | `fq_ltp` | 0.0025 | derived from code | feedforward `:114-117` | DERIVED | reference test |
| official LTD event probability | `fq_ltd` | 0.0025 | derived from code | feedforward `:115-119` | DERIVED | reference test |
| official connectivity probability | `p_w` | 0.6 | supplied upstream code | feedforward `:35` | BTSP_OFFICIAL_CODE | not carried into CAM |
| official recall threshold | `vth` | `int(I*p_w*0.45)` | supplied upstream code | feedforward `:95` | BTSP_OFFICIAL_CODE | not carried into CAM |
| RF/code dimension | `m` | 8192 | BTSP-CAM paper | Sec. 5.1 | BTSP_CAM_PAPER | formal config target |
| memory slots | `S` | 4096 | BTSP-CAM paper | Sec. 5.1 | BTSP_CAM_PAPER | formal config target |
| backbone | `phi` | ViT-B/16, ImageNet-21K | BTSP-CAM paper | Sec. 5.1 | BTSP_CAM_PAPER | interface only; exact checkpoint unclear |
| class-order seed | - | 1993 | BTSP-CAM paper | Sec. 5.1 | BTSP_CAM_PAPER | `equal_increment_tasks` |
| sparse active count | `k` | not reported | BTSP-CAM paper | Eq. (3), Table 6 | UNCLEAR | required argument |
| projection distribution | `R,b` | standard normal | BTSP-CAM paper | Eq. (3), App. C.2 | BTSP_CAM_PAPER | frozen NumPy arrays |
| binary-memory initialization | `WB(0)` | not reported | BTSP-CAM paper | Algorithm 1; App. D.1 is only a lemma | UNCLEAR | zero (`LOCAL_CHOICE`) |
| base plateau probability | `f_q_base` | not reported | BTSP-CAM paper | Eq. (5), Algorithm 1 | UNCLEAR | required argument |
| interference scale | `beta` | not reported | BTSP-CAM paper | Sec. 4.2 | UNCLEAR | required argument |
| synaptic flip probability | `p` | not reported | BTSP-CAM paper | Eq. (6), App. D | UNCLEAR | required argument |
| trace learning rate | `eta` | not reported | BTSP-CAM paper | Eq. (4) | UNCLEAR | required argument |
| trace decay | `lambda_c,t` | state-dependent, formula/value absent | BTSP-CAM paper | Eq. (4) | UNCLEAR | constant scalar/vector (`LOCAL_CHOICE`) |
| CA1 threshold | `tau` | not reported | BTSP-CAM paper | Eq. (7) | UNCLEAR | required argument |
| ridge regularization | `gamma` | not reported | BTSP-CAM paper | Eq. (8) | UNCLEAR | required argument |
| fusion weight | `alpha` | not reported | BTSP-CAM paper | Eq. (9) | UNCLEAR | not implemented in standalone core |
| ridge orientation | `W_out` | `S x C` | dimensional derivation from Eq. (8) | Eq. (8)-(9) | DERIVED | logits are `Z @ W_out` |
| CIFAR class count | - | 100 | dataset name/definition | Sec. 5.1 | BTSP_CAM_PAPER | plan only |
| CIFAR split | - | `B0-Inc5` | BTSP-CAM paper | Table 1 | BTSP_CAM_PAPER | literal operational meaning unclear |
| CIFAR task count | - | 20 equal tasks | local interpretation of B0-Inc5 | implementation plan | LOCAL_CHOICE | protocol helper; not formal experiment yet |
| train access | - | current task only | BTSP-CAM paper | Sec. 3.1 | BTSP_CAM_PAPER | `learn_task` takes only current batch |
| test access | - | all seen classes, no task ID | CIL objective | Sec. 3.1 Eq. (1) | BTSP_CAM_PAPER | seen-class logit mask only |

## Consolidated LOCAL_CHOICE list

1. Zero-initialize `WB` because Algorithm 1 supplies a binary memory but does not state initialization; Appendix D.1 analyzes a zero slot without specifying global initialization.
2. Decay every class trace on every sample using a constant scalar/vector `lambda`; the paper says state-dependent decay but gives no state function.
3. Clip `f_q_eff` to `[0,1]` as a Bernoulli-validity safeguard. This is mathematically derived, not fitted.
4. Use `Z @ W_out` because the printed transpose in Eq. (9) is dimensionally inconsistent with Eq. (8).
5. Interpret `B0-Inc5` as 20 equal five-class tasks with no separate base session for the smoke protocol only.
6. Smoke-only values: `m=256`, `k=16`, `S=128`, `f_q_base=1.0`, `beta=0.5`, `p=0.5`, `eta=0.5`, `lambda=0.15`, `tau=0.5`, `gamma=0.1`. These validate execution and must not be compared with Table 1. `tau` was raised from `0.1` (100% active) through `0.3` (95.8% active) to `0.5` solely to exercise non-degenerate CA1 competition; it was not selected against accuracy.

## Missing information blocking a faithful formal run

- Exact ViT-B/16 ImageNet-21K checkpoint, feature extraction point, image preprocessing, and feature normalization.
- `k`, every adaptive-gating numeric hyperparameter, trace initialization/decay schedule, CA1 threshold, ridge strength, fusion weight.
- Whether memory is reset between seeds/runs, sample order within each task, batch/shuffle semantics, and number of evaluation runs.
- Exact `B0` operational convention and the actual shuffled class-order array produced under the authors' framework/version.
