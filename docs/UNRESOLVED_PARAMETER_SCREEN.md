# Unresolved-parameter screen (independent experiment)

This is an **experimental search**, not a reconstruction of the authors' hidden configuration. The formal model, `configs/btsp_cam_cifar100.yaml`, `src/`, original checkpoints, backbone, class order and frozen-ViT feature cache are unchanged. All branches use 4096 binary memory slots, the same seed 1993 and training order, `a=W_Bs/k` with `tau=0`, and coordinate-aligned `C=Σssᵀ`, `D=Σsyᵀ`, `A=W_BCW_Bᵀ/k²`, `B=W_BD/k`. No result below identifies an author's actual hyperparameter.

## Source status before screening

| Symbol | Current reproduction | Source status | What the public sources establish |
|---|---:|---|---|
| `k` | 41 | `PLAUSIBLE_INHERITANCE` / paper value `UNRESOLVED` | Eq.(3) specifies exact top-k but not numerical `k`; 41 comes from applying simpleBTSP's 0.005 sparse ratio to `m=8192`. |
| `f_q_base` | 0.005 | `PLAUSIBLE_INHERITANCE` / paper value `UNRESOLVED` | Eq.(5) contains a base plateau rate but gives no numerical setting; 0.005 is inherited from simpleBTSP's default, whose learning mechanism is not identical. |
| `p` | 0.5 | `PLAUSIBLE_INHERITANCE` / paper value `UNRESOLVED` | Eq.(6) names a synaptic flip probability, with no reported number; 0.5 is a weak inheritance from simpleBTSP. |
| `beta` | 1 | `LOCAL_CHOICE` / paper value `UNRESOLVED` | Eq.(5)'s interference factor includes beta but the public PDF does not give its numerical value. |
| `lambda_{c,t}` | fixed 0.01 | `LOCAL_CHOICE` / state rule `UNRESOLVED` | Eq.(4) explicitly calls decay state-dependent, but supplies neither the state function nor a numeric schedule. Fixed decay is not a faithful claim about that rule. |
| `gamma` | 0.1 | `LOCAL_CHOICE` / paper value `UNRESOLVED` | Eq.(8) supplies the ridge formula but no numeric regularization. |

Source anchors: [supplied BTSP-CAM PDF](../../2233_BTSP_CAM_A_Brain_Inspired.pdf), Eq.(3)–(8), Sec. 5.1 and Table 1; local provenance in `configs/btsp_cam_cifar100.yaml` and `docs/PARAMETER_PROVENANCE_ROUND2.md`. The paper specifies `m=8192`, `S=4096` and seed 1993; those remain fixed. The reported CIFAR B0-Inc5 BTSP-only 87.82% average / 80.15% final is a **comparison target only**, never an estimator of missing parameters.

## Experimental design and safeguards

- Task1–10 screening is one-factor at a time: `k={16,24,32,41,64,96,128}`, `p={0.1,0.25,0.5,0.75,1}`, `f_q_base={0.001,0.0025,0.005,0.01,0.02}`, `beta={0.5,1,2,5,10}`, and `gamma={1e-4,1e-3,1e-2,0.1,1,10}`. Current-value duplicates alias the baseline rather than spawning a second stochastic run. Gamma reuses the *same* baseline W_B and C/D trajectory, changing only the solve.
- Eligibility families: `L0=(current 0,inactive 0)`; `L1=L2=(.01,.01)` alias the baseline; `L3=(.05,.05)`; `L4=(.1,.1)`; `L5=(0,.01)`; `L6=(.01,.05)`. `L5/L6` are only mechanism-family probes, not author implementations. Eq.(4) update-before-gating is preserved.
- Evaluation includes each task's all-seen cumulative, old and current accuracy, current 5-way accuracy, class-pair cosine geometry, Task1 feature drift, hotspot concentration, cross-task sparse-code collision, W_B density and F/N/I/plateau statistics. Average incremental is the arithmetic mean of per-task cumulative accuracy. Collision `P(J>=5)` is evaluated on all pairs of current-task and prior-task **test** sparse codes; `E[J]` uses their exact bit frequencies.
- A provisional Task1–10 screen used an accelerated experimental trace (cached norms and active-bit dot products). It first diverged from the archived baseline at Task16 (46,387 WB bits differed), despite matching through Task15; a tiny floating-point difference can flip a Bernoulli event and cascade through RNG use. **Both delivered CSVs were then run/re-run with the original eligibility/plateau/write path.** Against the provisional screen, 22 candidate-task cumulative accuracies changed, all in L3/L4/L6, with maximum absolute change 0.525 pp. The provisional file is retained only under `tmp/parameter_screen_fast_provisional.csv` for audit and is not a result source. Experimental matrix multiplication only reorders the same float64 `W_B C W_Bᵀ` products; Task1 A/B was checked elementwise against the existing C/D implementation. No approximate float32 C/D is used.

Implementation: `experiments/configs/unresolved_parameter_screen.yaml`, `experiments/run_unresolved_parameter_screen.py`. Task-level outputs: `diagnostics/parameter_screen_task10.csv` and `diagnostics/parameter_candidates_full20.csv`.

## Task1–10 screen

All one-factor points completed with the original trace implementation: 29 result labels (baseline, 6 new k, 4 new p, 4 new plateau-base, 4 new beta, 5 new gamma readouts on the shared baseline trajectory, and 5 new lambda trajectories), plus one `k=128+L0` small-factorial consistency trajectory. The joint Task1–10 replay was added after the full20 shortlist was fixed; it did not select that shortlist. In total there are 25 distinct W_B trajectories and 5 additional gamma readouts. L1/L2 are explicitly marked exact baseline aliases in the CSV, with **no** fictitious retrain. This yields 320 rows: 30 actual result labels ×10 tasks plus 2 aliases ×10. The baseline reproduced every archived Task1–10 W_B checkpoint bit-for-bit and every archived aligned cumulative accuracy. Task10 baseline was 80.38%, with 85.31% Task1–10 average cumulative, 80.38% old-class and 93.6% current-task 5-way accuracy.

| One-factor family | Task10 cumulative range | Task1–10 average-cumulative range | Main observation |
|---|---:|---:|---|
| `k=16…128` | 70.80–86.42% | 77.69–90.06% | By far largest sensitivity: higher k improves readout and sparse class geometry but increases overlap/capacity pressure. |
| `p=0.1…1` | 79.86–80.44% | 85.21–85.91% | WB density changes, accuracy barely moves. |
| `f_q_base=.001….02` | 79.36–80.66% | 83.69–86.64% | Low base rate has better early average and much lower occupancy, but only small Task10 gain. |
| `beta=.5…10` | 79.96–80.68% | 85.22–86.37% | Higher beta reduces writes/occupancy; Task10 gain remains small. |
| `gamma=1e-4…10` | 79.62–80.84% | 75.90–86.84% | Very low regularization harms early accuracy; gamma 1–10 helps the average without changing WB or geometry. |
| `lambda` L0/L3/L4/L5/L6 | 79.86–80.36% | 84.79–85.68% | These simple fixed/conditional families do not recover a major gain. |

For k specifically, `k=16/41/64/96/128` gives Task10 `70.80/80.38/83.08/85.32/86.42%`; old-class accuracy `70.64/80.38/83.11/85.44/86.56%`; current 5-way `91.6/93.6/96.8/97.4/97.2%`; WB density `2.59/6.04/8.73/11.97/14.78%`. Task1-feature cosine to its initial WB rises `0.621/0.790/0.848/0.885/0.904`, while all-seen overlap cosine margin drops `0.0259/0.0102/0.0061/0.0037/0.0026`. These opposing metrics make a 20-task capacity check essential.

The hotspot statistic improves with higher k (`top1%` share 25.51% at k16, 19.54% at k41, 13.31% at k128; zero-win bits 1820, 737, 136), but absolute cross-task `E[J]` increases 0.205, 0.983, 6.536, respectively. `P(J>=5)` is **not comparable across different k as a fixed severity threshold**: the expected intersection grows with code size even for uniform codes. The sparse cosine class margin rises from 0.0520 at k16 to 0.0920 at k128, while the post-memory overlap cosine margin shrinks. Ridge accuracy is not determined by that single margin statistic.

For the lambda probes, L0 has the strongest multi-metric balance: 85.68% average, 80.36% Task10, 80.47% old, 93.6% 5-way, 4.55% WB density and 0.0143 overlap margin. The baseline is 85.31/80.38/80.38/93.6%, 6.04% density and 0.0102 margin. L5 (only inactive classes decay) is 80.14% Task10; L6 is 80.04%. At Task10 L0 has mean `F=0.344`, `N=0.656`, `I=0.767`, effective plateau probability 0.00114; baseline is `F=0.426`, `N=0.574`, `I=0.784`, effective probability 0.00157. L6 raises mean F to 0.612 and effective probability to 0.00230, with 8.00% WB density. This confirms the decay rule affects gating/write dynamics even though these particular rules give little accuracy gain.

### Frozen full20 shortlist

The five names were fixed in `experiments/configs/unresolved_parameter_screen.yaml` before running any full20 candidate: `baseline`, `k_128`, `L0`, `joint_k128_L0`, `f_q_base_0.001`. This includes the strongest k-family point on average/Task10/old/current-5way, despite its high density, expressly to test its late capacity risk. L0 is the lambda representative because it modestly improves average/old accuracy and reduces density; the joint branch tests interaction. `f_q_base=.001` is the separate write-rate candidate because it has the strongest early average among that family and a much larger overlap margin, despite only +0.28 pp Task10. Gamma 1–10 improved Task1–10 average, but its Task10 gain was small and it was not included within the five-run cap. The shortlist is a **diagnostic**, not a test-set-calibrated final hyperparameter choice.

The subsequent Task1–10 joint consistency run was 90.18% average / 86.48% Task10, versus 90.06% / 86.42% for k=128 alone: a small early difference, not evidence of strong interaction.

## Full20 candidate comparison

All five frozen-shortlist trajectories completed with the **original** eligibility/plateau/write implementation. The baseline W_B matched every formal checkpoint bit-for-bit through Task20; its cumulative accuracy matched the archived coordinate-aligned series task by task, including 73.30% at Task20. The `C/D`-aligned readout remained unthresholded in every branch.

| Experimental branch | Average incremental | Task20 all 100 | Task20 old 95 | Task20 current 5 | Current 5-way | WB density Task20 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline `k=41,lambda=.01,f_q=.005` | 80.68% | 73.30% | 73.49% | 69.60% | 89.0% | 9.37% |
| `k=128` only | 86.49% | **80.62%** | 80.86% | 76.00% | 92.2% | 20.61% |
| `L0` only (`lambda=0`) | 80.86% | 73.43% | 73.60% | 70.20% | 89.2% | 7.30% |
| `k=128 + L0` | 86.54% | 80.29% | 80.56% | 75.20% | 91.8% | 18.43% |
| `f_q_base=.001` only | 81.45% | 73.53% | 73.72% | 70.00% | 89.4% | 2.99% |
| Paper BTSP-only CIFAR B0-Inc5 | 87.82% | 80.15% | — | — | — | — |

`k=128` improves this experimental baseline by **+5.81 pp average / +7.32 pp Task20**. The Task20 number is 0.47 pp above the paper's reported final score, while the 20-task average remains 1.33 pp below it. Numerical proximity is **not** evidence that the authors used `k=128`; the paper does not publish its k, and the independent variant also differs from the paper by using coordinate-aligned ridge and `tau=0`. The joint candidate adds only +0.05 pp average and *loses* 0.33 pp final versus `k=128` alone. L0 and the lower plateau base are individually small (Task20 +0.13 and +0.23 pp), so the tested lambda/write-rate changes do not explain the primary gain.

At Task20, higher k improves the sparse-code class cosine margin from 0.06795 to 0.09050 and current 5-way accuracy from 89.0% to 92.2%, but **reduces** the pre-threshold memory-overlap cosine margin from 0.00561 to 0.00144. It increases WB occupancy from 9.37% to 20.61%. Thus the classifier remains accurate on 100 seen classes despite a compressed scalar cosine margin and higher capacity pressure. This does not prove absence of later saturation beyond Task20, or that the overlap geometry is universally better. Raw cross-task `P(J>=5)` rises from 1.37% to 67.19% mostly because k itself is larger; it must not be read as directly comparable hotspot severity. The top-1%-bit share falls from 19.38% to 13.29% and zero-win bits from 398 to 64. The k intervention changes code density, writing volume, `a=W_Bs/k` scaling, and the effective strength of a fixed gamma together; this experiment does **not** identify which of these mediators causes the accuracy gain.

## Final judgment

1. **Observed sensitivity:** among the screened numerical choices, k dominates Task10 and full20 performance. Gamma is also important for the early average, particularly avoiding near-zero regularization, but only the fixed-gamma k candidate was tested full20. `p`, `f_q_base`, `beta`, and the tested simple lambda families show much smaller Task10/final effects in this one-factor design.
2. **Collision and downstream classification:** k materially changes both. Lower k reduces absolute bit intersections but also harms accuracy; higher k reduces winner concentration and raises sparse class margin while increasing expected intersections and WB occupancy. A fixed `J>=5` collision tail is not scale-invariant across k. No isolated causal claim about collision is justified.
3. **Lambda uncertainty:** the paper's unspecified *state-dependent rule* remains the largest **specification uncertainty**, even though L0/L3/L4/L5/L6 are not the strongest **observed performance sensitivity**. These probes cannot exhaust possible state-dependent schedules.
4. **Gap:** an experimental, tau-free `k=128` candidate reaches 80.62% Task20, versus 73.30% local baseline and paper 80.15%; its average is 86.49% versus paper 87.82%. This demonstrates that an unpublished/local k choice can account for much of the numerical gap under this independent aligned-ridge setting. It does not reproduce the authors' method or prove a hidden value.
5. **Reproducibility boundary:** public BTSP-CAM PDF + simpleBTSP + PILOT do not uniquely determine `k`, `f_q_base`, `p`, `beta`, `lambda_{c,t}`, `gamma` (nor the earlier `tau`). Single-seed test-task screening is mechanism exploration, not held-out model selection or a performance guarantee.

## Interpretation limits

The screen uses the same fixed test tasks to characterize mechanisms and compare candidates; it does **not** provide a held-out validation protocol for choosing deployable hyperparameters. No replicate seeds or uncertainty intervals were run. Matching Table 1 numerically would not identify the authors' settings. The missing `lambda_{c,t}` rule in particular prevents unique reconstruction from the PDF, simpleBTSP, and PILOT alone. All delivered numerical results use the unaccelerated original trace/write implementation; the separate fast-trace provisional run is excluded from these conclusions.
