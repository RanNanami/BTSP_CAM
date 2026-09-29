# Official simpleBTSP to BTSP-CAM audit

## Evidence boundary

- Upstream snapshot: `upstream/simpleBTSP`, extracted unchanged from the supplied `simpleBTSP-main.zip`.
- BTSP-CAM source: supplied 18-page PDF, including Algorithm 1 and Supplementary Sections A-F.
- The upstream repository is a collection of top-level experiment scripts, not an importable library. Its README also says a more complete code version would be released after publication (`README.md`, lines 13-26).
- `OFFICIAL_CODE_BEHAVIOR` and `PAPER_BEHAVIOR` are kept separate wherever the supplied artifacts differ. No prior `reproduction/core_btsp` or CIL experiment code was used.

## Component-by-component audit

| Component | official simpleBTSP | source location | BTSP-CAM | paper location | action |
|---|---|---|---|---|---|
| input representation | Bernoulli binary random patterns, shape `(m, M)` in feedforward code | `main_BTSP_feedforward_model.py:101-102` | Frozen PTM feature `h`, then fixed Gaussian projection and exact k-WTA binary code `s` | Sec. 3.1; Sec. 4.1, Eq. (3) | MODIFIED |
| CA3 / input dimension | `m=25000` | feedforward `:32`; feedback `:16` | RF/code dimension `m=8192`; the paper does not explicitly call this CA3 | Sec. 5.1 | MODIFIED |
| random projection | A file named `main_random_projection.py` exists, but it imports missing `hopfield_models` and its shown training block is still the BTSP update, not Eq. (3) Gaussian+k-WTA | `main_random_projection.py:8, 116-132` | `u=Rh+b`, `R,b~N(0,1)`, then Top-k | Sec. 4.1 Eq. (3); App. C.2 Eq. (10) | NEW |
| input sparsity | Bernoulli `fp=p=0.005`; number active is random | feedforward `:34, 101-102` | Exactly `k` active of `m`, but experimental `k` is not reported | Eq. (3); Table 6 | MODIFIED |
| connectivity mask | Independent Bernoulli mask with `p_w=0.6`, applied after training | feedforward `:109,128` | No connectivity mask is defined; `WB` is a dense logical matrix | Sec. 3.2; Table 6 | REMOVED |
| binary synaptic matrix | Code orientation `(m,n)` after first update; boolean then mask multiplication | feedforward `:123-128` | `WB in {0,1}^{S x m}` | Sec. 3.2 text; Table 6 | MODIFIED |
| weight initialization | Scalar `W=0` expands on first update; feedback fast path also begins at zero | feedforward `:113`; feedback `:68-81` | Initialization is not stated; implementation uses zero and marks it LOCAL_CHOICE | Algorithm 1 input; App. D.1 considers `wj(0)=0` only for the lemma | UNCLEAR |
| plateau event | Separate neuron-level LTP and LTD winner arrays | feedforward `:115-119` | Independent slot-wise `q_j` | Algorithm 1 lines 8-10 | MODIFIED |
| plateau probability | Fixed total `fq=0.005`, split to `fq/2` LTP and LTD; LTD conditional sampling makes events mutually exclusive | feedforward `:25,114-119` | Adaptive scalar `f_q_eff(s,y)` applied independently over slots | Sec. 4.2 Eq. (5) | MODIFIED |
| LTP | On LTP event, every active-input synapse is set to 1 | feedforward `:123-126` | If plateau occurs, every active synapse independently flips with probability `p`; 0 to 1 is LTP | Eq. (6); App. D Eq. (18) | MODIFIED |
| LTD | On LTD event, every active-input synapse is reset to 0 | feedforward `:123-126` | Same independent flip process; 1 to 0 is LTD | Eq. (6); App. D Eq. (18) | MODIFIED |
| binary set/reset/flip rule | Deterministic set/reset conditioned on mutually-exclusive neuron event | feedforward `:117-126` | XOR flip, i.i.d. Bernoulli `p` per `(slot,synapse)` after slot plateau | Algorithm 1 lines 8-11; Eq. (6) | MODIFIED |
| update timing | Sequential over stored patterns; connectivity mask only after all updates | feedforward `:123-128` | Per sample: code, update traces, compute gate, sample plateaus/flips; consolidate after task | Algorithm 1 | MODIFIED |
| eligibility trace | Absent | not present | Class-specific nonnegative `e_c`; all traces use state-dependent decay and current class receives `eta*s` | Sec. 4.2 Eq. (4) | NEW |
| class-specific eligibility | Absent | not present | One trace per class | Eq. (4), Table 6 | NEW |
| familiarity | Absent | not present | `F=cos(s,e_y)` | Sec. 4.2 | NEW |
| novelty | Absent | not present | `N=1-F` | Sec. 4.2 | NEW |
| interference / collision | Not used to gate updates | not present | `I=exp(-beta max_{k!=y} cos(s,e_k))` | Sec. 4.2 | NEW |
| adaptive plateau probability | Absent | not present | `f_base * 4N(1-N) * F * I` | Eq. (5) | NEW |
| recall | Feedforward threshold on `W.T @ data`; output shape `(M,n)` | feedforward `:165-169` | Normalized memory overlap followed by thresholded ReLU and ridge logits | Eq. (7)-(8) | MODIFIED |
| threshold | `int(I*p_w*0.45)`; comment says grid-search optimized | feedforward `:95` | CA1 threshold `tau`; value not reported | Eq. (7), Table 6 | MODIFIED |
| memory trace | Implicit binary output assemblies; no class trace | feedforward `:165-183` | `WB` plus explicit class eligibility traces | Fig. 1; Eq. (4)-(7) | MODIFIED |
| CA1 activity | Binary `1[W.T x > vth]` | feedforward `:165-169` | Real-valued `z=max(0, WB*s/||s||_1-tau)` | Eq. (7) | MODIFIED |
| CA1 competition | A fixed threshold only | feedforward `:95,165-169` | Normalization by active input count plus thresholded ReLU; no top-k/lateral competition is specified | Eq. (7); Sec. 6.3 | MODIFIED |
| feedback mechanism | Separate feedforward and back matrices; threshold CA1 then project back and threshold CA3 | feedback `:80-88,132-145` | No feedback/reconstruction path in Algorithm 1 | Fig. 1; Algorithm 1 | REMOVED |
| classifier/readout | Random linear probe only in the RP comparison script; not trained on labels | `main_random_projection.py:100-103,199-205` | Standalone ridge branch; optionally ensemble with host PTM logits | Eq. (8)-(9); Sec. 5.2 | NEW |
| ridge regression | Absent | not present | Closed-form ridge from cumulative sufficient statistics | Sec. 4.5 Eq. (8) | NEW |
| sufficient statistics | Absent | not present | `A_t=A_{t-1}+Z_t^T Z_t`, `B_t=B_{t-1}+Z_t^T Y_t` | Eq. (8) | NEW |
| consolidation | Absent | not present | At task end, update statistics and solve ridge | Algorithm 1 lines 13-15 | NEW |
| incremental-learning protocol | Offline sweeps over number of random memories; no labels/tasks | feedforward `:97-103` onward | Disjoint class tasks, current task training data only, cumulative seen-class test, exemplar-free | Sec. 3.1 Eq. (1); Sec. 5.1 | NEW |
| dataset | Synthetic Bernoulli patterns | feedforward `:101-102` | CIFAR-100 and five other image benchmarks | Sec. 5.1 | MODIFIED |
| pretrained backbone | None | not present | Frozen ViT-B/16 pretrained on ImageNet-21K | Sec. 5.1 | NEW |
| evaluation metrics | Hamming distance/accuracy, firing rate, weight ratio | feedforward `:54-60,165-221` | Top-1 final accuracy `A_B` and mean incremental accuracy `Abar`; mechanism diagnostics also reported | Sec. 5.1; Tables 2-5 | MODIFIED |

## Exact official update truth table

For an active input bit, the supplied official feedforward expression is
`((W + x*LTP) >= 1) & ((W + x*LTD) <= 1)`.

| old weight | LTP event | LTD event | new weight |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 1 |
| 0 | 1 | 0 | 1 |
| 1 | 1 | 0 | 1 |
| 0 | 0 | 1 | 0 |
| 1 | 0 | 1 | 0 |

LTP and LTD are sampled mutually exclusively. Inactive input bits do not change. Thus the official code performs set/reset, not per-synapse XOR. `src/btsp_cam/binary_memory.py::official_simplebtsp_step` preserves this primitive, while `BinaryMemory.update` implements the changed BTSP-CAM Eq. (6).

## Other official-code findings

- **W shape:** official feedforward `W` is `(m,n)` = input-by-output. BTSP-CAM paper uses `(S,m)` = slot-by-code. The reproduction follows the paper and isolates the official orientation in its reference helper.
- **Sampling granularity:** official LTP/LTD randomness is per pattern and output neuron; all active synapses for the selected neuron are changed together. BTSP-CAM adds an i.i.d. per-synapse Bernoulli flip.
- **Partial cue:** feedforward removes a random fraction of each pattern's active bits (`:151-161`). Feedback instead zeros a fixed leading coordinate block (`feedback:121-122`). These are different official-code behaviors.
- **Feedforward vs feedback:** feedforward retrieves an output assembly only. Feedback adds a learned `CA1 -> input` binary matrix and reconstructs the input (`feedback:132-145`). BTSP-CAM uses only the feedforward memory-to-CA1 path.
- **Random projection executability:** `main_random_projection.py` imports `hopfield_models`, which is absent from the supplied zip. Its visible BTSP training code is otherwise executable only with CUDA and the heavyweight top-level sweep. It is not the paper's Gaussian+k-WTA encoder.
- **Seed bug:** `fix_random_seeds` calls `torch.random.seed()` without the provided seed (`feedback:26-29`), so that script is not reproducible from `seed_value` alone.

## Paper-internal ambiguities preserved

1. Algorithm 1 updates the current class trace before computing familiarity. With a zero trace, the first update makes `e_y` parallel to `s`, hence `F=1`, `N=0`, and `f_q_eff=0`. The implementation follows the printed order and records this as an unresolved specification issue.
2. Eq. (8) yields `(A+gamma I)^-1 B` with shape `S x C` for row-major `Z`, while Eq. (9) prints `z W_out^T`, which is dimensionally inconsistent. The implementation uses the conventional `z @ W_out` and marks this `DERIVED`.
3. The text calls Eq. (7) “competition,” but only normalization and thresholded ReLU are defined; there is no CA1 top-k or lateral inhibition.
4. `B0-Inc5` conflicts with the paper's literal definition that the initial task has `m` classes. The local smoke protocol interprets it as no special base session and 20 equal five-class tasks; this is `LOCAL_CHOICE`, not a claimed paper fact.

## OFFICIAL_CODE_BEHAVIOR versus PAPER_BEHAVIOR

Here `PAPER_BEHAVIOR` refers only to the supplied BTSP-CAM paper's own description of simpleBTSP in Sec. 3.2/Eq. (2), not to an unrequested external implementation.

| issue | OFFICIAL_CODE_BEHAVIOR | PAPER_BEHAVIOR in supplied BTSP-CAM PDF | resolution |
|---|---|---|---|
| flip probability `p` | no independent per-synapse flip draw; LTP/LTD plateau selection controls deterministic set/reset of all active synapses | Eq. (2) states each active synapse changes with probability `p` | BTSP-CAM follows its own Eq. (6), i.i.d. per synapse |
| LTP/LTD event | separate mutually-exclusive neuron arrays at `fq/2` each | one plateau `q_j`, then weight-dependent 0-to-1 or 1-to-0 stochastic change | keep official helper separate; runtime follows CAM |
| connectivity | post-hoc Bernoulli mask `p_w=0.6` | no mask in `WB` definition | omit from CAM runtime |
| dimensions | `(m,n)=(25000,39000)` | generic `WB in {0,1}^{N x M}`; CAM later uses `(S,m)` with 4096 by 8192 | runtime uses CAM orientation and sizes |
| recall | binary threshold of `W.T @ x` | CAM uses normalized real CA1 threshold plus ridge | runtime follows Eq. (7)-(8) |
