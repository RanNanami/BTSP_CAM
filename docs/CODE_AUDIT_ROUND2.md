# Code audit round 2

## Scope and evidence

Audited against BTSP-CAM Eq. (2)-(9), Algorithm 1, Sections 5-6, and Supplement A-F. PILOT parity is pinned to official `LAMDA-CL/LAMDA-PILOT` commit `7a6e904c5bc5cb7a4e1823b3434020be27469b63`.

Status vocabulary: `PASS`, `BUG`, `AMBIGUOUS`, `LOCAL_CHOICE`, `NEEDS_CHANGE`.

## Findings by module

| area | status | implementation/evidence | finding and action |
|---|---|---|---|
| Eq. (3) projection shape | PASS | `sparse_encoder.py:RandomSparseEncoder:8-18` | `R` is `[m,d]`, `b` is `[m]`; both are sampled once from standard normal and never updated. Train/test share one encoder instance. |
| Eq. (3) Top-k | PASS / LOCAL_CHOICE | `sparse_encoder.py:encode:20-33` | Output has exactly `k` ones. Round 1 used `argpartition`, whose tie membership was not explicitly stable. Fixed to stable descending sort; lower index wins equal scores. No projection normalization is applied. |
| formal dimensions | PASS at config boundary | paper Sec. 5.1; formal target `m=8192`, `S=4096` | Core is dimension-generic so smoke can shrink it. Formal config must use paper dimensions. |
| Eq. (4) decay/update | PASS | `eligibility.py:ClassEligibilityTraces.update:44-51` | Every class trace decays on every sample, then the current class gets `eta*s`. |
| trace/gate order | PASS | `model.py:_learn_code:59-63` | Trace update precedes F/N/I and `f_eff`, exactly Algorithm 1 lines 5-7. |
| `cos(s,0)` | LOCAL_CHOICE | `eligibility.py:cosine_similarity:10-12` | Defined as zero. The paper does not define zero-vector cosine. |
| empty rival set | DERIVED / LOCAL_CHOICE | `eligibility.py:signals:53-72` | All zero unseen traces contribute cosine zero; an actually empty rival set has max zero, hence `I=exp(0)=1`. |
| F/N/I formulas | PASS | `eligibility.py:signals:53-72` | `F=cos(s,e_y)`, `N=1-F`, `I=exp(-beta*max rival cosine)`. `beta<0` is rejected. |
| F clipping | PASS numeric safety | `eligibility.py:57` | Codes and traces are nonnegative, so analytic cosine is already `[0,1]`; clipping only removes floating-point excursions. |
| Eq. (5) product | PASS | `plateau.py:plateau_probability:10-30` | Contains every factor `f_base*4*N*(1-N)*F*I`, with the interference sign unchanged. |
| invalid `f_base` | BUG, FIXED | `plateau.py:11-20`; `test_plateau.py` | Round 1 silently clipped even `f_base=10`. It now rejects bases outside `[0,1]`; final clipping is boundary-only numerical safety. Old test expectation was changed for this reason, not hidden. |
| slot-wise plateau | PASS | `plateau.py:sample_slot_plateaus:33-36`; `model.py:64-66` | Draws one independent uniform variate per slot for every sample; no global/class broadcast. |
| plateau/synapse RNG | BUG, FIXED | `model.py:31-56,64-70`; `rng.py:RNGProvenance` | Round 1 used one generator for both draws. Separate named streams now drive `q_j` and `u_ji`. |
| Eq. (6) synaptic update | PASS | `binary_memory.py:BinaryMemory.update:37-60` | Draw is i.i.d. over selected slot x active input index. Only `s_i=1`, `q_j=1` cells are XORed. Matrix remains `uint8` binary with paper orientation `[S,m]`. |
| official simpleBTSP contamination | PASS | `binary_memory.py:official_simplebtsp_step:8-26` | Official neuron-shared set/reset is isolated in a reference helper and is not used by BTSP-CAM runtime. |
| Eq. (7) CA1 | PASS | `ca1.py:ca1_features:8-22` | Computes `(s @ WB.T)/||s||_1`, then continuous `max(0,a-tau)`. It is neither binary thresholding nor cosine. |
| Algorithm 1 phase order | PASS | `model.py:learn_task:82-95` | All codes first update memory; only after the loop is final-memory `Z_t` constructed and added to ridge. `test_phase_two_uses_final_post_task_memory` verifies `A=Z_final^T Z_final`. |
| Eq. (8) sufficient statistics | PASS | `ridge.py:update:30-40` | `Z:[N,S]`, `Y:[N,C]`, `A:[S,S]`, `B:[S,C]`. Both accumulate without resetting and no raw old sample is retained. |
| ridge class expansion | NEEDS_CHANGE, FIXED | `ridge.py:expand_classes:17-28`; `model.py:92` | Round 1 preallocated all class columns. It now pads columns as labels arrive and preserves old `B` exactly. |
| ridge solve | PASS | `ridge.py:solve:42-50` | Uses float64 `np.linalg.solve`, not explicit inverse; records solve status and exposes regularized condition number. |
| Eq. (9) orientation | AMBIGUOUS | `ridge.py:logits:52-54` | Eq. (8) yields `W_out:[S,C]`; printed Eq. (9) transpose is dimensionally inconsistent. Runtime uses `Z @ W_out` (`DERIVED`). |
| BTSP-only | PASS | `model.py:logits:97-106` | Predictions use only CA1-to-ridge logits. No PTM classifier or alpha fusion is present. |
| class order RNG | BUG, FIXED | `protocol.py:pilot_class_order:8-14` | Round 1 used `default_rng`, which does not match PILOT. Now uses local `RandomState(seed)` to reproduce PILOT's legacy `np.random.seed` + `np.random.permutation` without mutating global state. |
| named RNG provenance | NEEDS_CHANGE, FIXED | `rng.py:RNGProvenance:11-39` | Separate seeds now exist for class order, train order, augmentation, projection, plateau, and synapse update. |
| reproducibility | PASS | `test_reproducibility.py`; `test_pilot_protocol.py` | Same configuration/seed reproduces memory and ridge state; class order matches an instantiated, monkeypatched official PILOT `DataManager` exactly. |

## Remaining ambiguities, not silently fixed

1. Algorithm 1 updates `e_y` before gating. With zero initialization, the first class sample gives `F=1`, `N=0`, and therefore no plateau. Runtime follows the printed order.
2. `lambda_{c,t}` is described as state-dependent but its state function is absent.
3. Paper does not state whether BTSP-only extracts current-task features with PILOT's stochastic train transform or deterministic test transform. Protocol smoke uses the actual train transform once per training image and labels this choice.
4. Global `WB` initialization is absent. Zero initialization remains `LOCAL_CHOICE`, supported but not specified by Appendix D.1's zero-slot analysis.

## Fixed bugs by severity

1. **High:** class order RNG differed from PILOT, changing every task composition.
2. **Medium:** plateau and synapse draws lacked separate RNG provenance.
3. **Medium:** ridge class columns did not actually exercise incremental expansion.
4. **Medium:** invalid base probabilities could be silently clipped instead of rejected.
5. **Low:** Top-k ties were deterministic only incidentally; tie policy is now explicit and tested.
6. **Low (diagnostics only):** the first real smoke attempt bound the indexed
   dataset tuple as `(image, _, _)` instead of `(_, image, _)` while collecting
   input statistics. It failed immediately on a `LongTensor` mean before any
   result was emitted; `run_cifar100_protocol_smoke.py:tensor_batch_summary`
   now receives the image tensor. The feature-extraction loader binding was
   already correct.
