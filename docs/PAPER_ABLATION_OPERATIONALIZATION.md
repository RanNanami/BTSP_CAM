# Section 6 component-ablation operationalization audit

Source: local BTSP-CAM paper PDF, printed pp. 4–8 and Appendix pp. 12–18; Tables 2–5 were also checked on rendered pp. 7–8. The paper's Section 6 Full Abar is **86.82%**, whereas its Table 1 CIFAR100 BTSP-only Abar/final is **87.82/80.15%**. Their experimental-context difference is not fully specified. Neither number is a selection target here. The sole local Full comparator is the frozen k=96, q=4 replay-refit baseline.

| Variant | Paper evidence | Is code-level removal unique? | Local operation | Status |
|---|---|---|---|---|
| Full | Eq. (4)–(8), Algorithm 1 | Formula/order stated; several hyperparameters and nonlinear replay policy not specified | Frozen working baseline: k96, q4, f_base .005, beta1, lambda .01, p .5, gamma .1; current-WB replay refit | PAPER_EXPLICIT equations; LOCAL working values/readout |
| A1 – trace gating | §6.1/Table 2 says it removes trace-gated plateau probability | No replacement plateau probability given | Keep traces for logging; sample each slot at constant f_base=.005, bypass F/N/I modulation | PAPER_UNSPECIFIED_LOCAL_OPERATIONALIZATION |
| A2 – collision term | Eq. (5) identifies `I=exp(-beta max rival cosine)`; Table 3 names its removal | Algebraically clear, though not separately given as code | Set only `I:=1`; retain `4N(1-N)F` | PAPER_IMPLIED |
| A3 – LTD (only LTP) | Eq. (6), Appendix D Eq. (18), Table 3 “only LTP” | Yes at state-transition level | Preserve plateau and per-synapse Bernoulli(p), but suppress all sampled 1→0 transitions | PAPER_IMPLIED |
| A4a – CA1 threshold | Eq. (7) specifies normalization + ReLU threshold; Table 4 says competition normalizes and sparsifies | No: Table 4 gives no replacement operator | `z=W_B s/k`, removing threshold only | PAPER_UNSPECIFIED_LOCAL_OPERATIONALIZATION |
| A4b – entire CA1 operator | Same | No | `z=W_B s`, removing normalization and threshold | PAPER_UNSPECIFIED_LOCAL_OPERATIONALIZATION |
| A5 – consolidation | Eq. (8), Algorithm 1 Phase 2 and Table 4 give removal label and outcome | **No replacement classifier or temporal readout rule** | Do not fabricate a “paper – consolidation” run | PAPER_ABLATION_NOT_REPRODUCIBLE_FROM_PUBLIC_DESCRIPTION |
| R1 partial cues | §6.4/Table 5 says randomly mask fraction p of active sparse bits and evaluate old-class accuracy | Mask count/RNG coupling unspecified | Drop exactly round(ratio × active-bit-count) bits per test sample, independently for five fixed seeds; no model retraining | PAPER_IMPLIED masking; LOCAL corruption implementation |

The appendix provides theoretical analyses, not missing code-level ablation recipes. Appendix F.1 reiterates Eq. (7) and partial-cue retrieval. Appendix F.2 analyzes idealized readout margin; it does not specify A4 or A5 removal procedures.

## Metric definitions and limits

The PDF defines average incremental accuracy as mean task-stage top-1 accuracy (§5.1). It **names** Fgt, mean prototype overlap Ov., and Old-to-New O→N in Tables 2–4 but provides no exact formula, normalization, or evaluation stage. No corresponding formula was found in the current project. For this experiment, separately labeled **LOCAL METRIC OPERATIONALIZATION** is:

- `Fgt`: average over source tasks 1–19 of each source's best accuracy at any evaluation stage from its introduction through Task20, minus its Task20 accuracy; report percentage points. This is the common per-task max-to-final forgetting statistic, **not claimed paper-exact**.
- `Ov.`: mean off-diagonal cosine similarity of class-mean CA1 test features at Task20, excluding zero-norm prototypes. This is a feature-prototype measure, not a claim about the paper's memory-trace normalization.
- `O→N`: among Task20 test examples from Tasks1–19, fraction predicted as one of the last five (Task20) classes. Other plausible “new” sets exist; this choice is local.
- `E[F_correct]` / `E[F_error]`: mean cosine of **Task20 current-five-class test** sparse code and final updated true-class eligibility trace, split by Task20 classifier correctness. Some old-class traces underflow to zero after repeated per-sample decay; including those would make this proxy ill-defined. This is familiarity F, not F-score, and not claimed paper-exact because the table does not specify trace snapshot or split.
- CA1 margin: difference between exact mean same-class and different-class **sample** cosine similarities at Task20. This is not automatically identical to Table 4's unspecified margin.

Partial-cue curves use **Task20 old-class test samples** only. At drop0, each curve must equal that variant's Task20 old-class accuracy. Corruption occurs after sparse encoding, only at evaluation; writer, trace, W_B, ridge weights, and frozen parameters are unchanged. The five predeclared corruption seeds are in the experiment YAML.

## Scale and randomness caveats

A4a and A4b have the same nonzero support and directional geometry because `W_B s = k·(W_B s/k)`. With fixed gamma=.1, however, a linear ridge readout on these differently scaled features is not algebraically identical: raw overlap changes effective ridge regularization by k². Feature norm and spectrum/Gram scale must therefore accompany A4b; an accuracy difference cannot be attributed solely to biological competition.

All writer variants start from the same data order, sparse encoder, zero memory and writer RNG seeds. Different plateau events and conditional synaptic draws make subsequent RNG consumption diverge. **Paired initialization is not event-wise paired randomness.** The local experimental code does not change the formal source implementation.
