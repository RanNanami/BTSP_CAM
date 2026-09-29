# Experimental repair ladder: CIFAR-100 Task1–5

This is an **experimental repair**, not a strict paper reproduction and not a parameter search. The formal model, default YAML, and original checkpoints were not modified. The project directory is not a Git repository; isolation is by the separate `experiments/configs/repair_ladder_task1_5.yaml` and `experiments/run_repair_ladder_task1_5.py`, with output only to `diagnostics/repair_ladder_task1_5.csv` and this report. No Task6–20 training or evaluation was run.

## Protocol and comparability

- Same cached PILOT-processed frozen ViT-B/16 features, CIFAR-100 B0-Inc5 class order and train/test sample order, seed 1993, original R and b, exact k=41, 4096 memory slots, p, f_base, beta, lambda/trace decay, gamma, and the existing `StreamingRidge` sufficient-statistics update and solve. The experimental script asserts that R and b equal those in the original Task1 checkpoint.
- Every BTSP branch reads out **a = W_B s / ||s||_1** directly. Because a is nonnegative, this equals `max(0,a)` at tau=0. The default tau=0.5 remains untouched; this experiment does not select a new tau.
- R0 uses the original Task1–5 W_B checkpoints, preserving their exact Eq.(5)–(6) write trajectory; only its unthresholded ridge is fitted again. R1 uses the same raw sparse codes and Task1 W_B, then never calls plateau sampling, XOR, or any other memory update in Task2–5. Both update ridge once per new task without replay.
- R2/R3 subtract only the **Task1 training-set mean** of h, frozen thereafter, before the same R, b and top-k. Task1 is re-encoded too. R2 replays the formal `_learn_code` write routine on the centered stream using the original plateau/synapse RNG seeds and sample order. R3 shares that Task1 centered W_B, freezes it, and updates only ridge thereafter. R0/R1 codes are identical, as are R2/R3 codes.
- REF is the already validated sparse→ridge reference from `diagnostics/representation_ablation.csv` on these same feature caches and encoder; it was **reused**, not retrained in this run. Its Task5 cumulative accuracy is 65.16%. All four BTSP ridge readouts were fitted in this run. Fresh R0 cumulative accuracies match the earlier `ca1_readout_ablation.csv` unthresholded run at all five tasks to numerical precision.

`current_accuracy` evaluates current-task test images against **all seen-class logits**. `current_5way_accuracy` restricts those same logits to the current five classes; it does not train a separate classifier. `old_accuracy` uses all prior-task test images against all seen-class logits. `cumulative_accuracy` uses all seen test images. Task1 old-class and cross-task J values are undefined (`nan`). Geometry (`same`, `different`, `margin`) is exact pairwise cosine on the **current task's 500 test examples** at the post-task W_B. Top-1% bit share and zero-winner count use **all training sparse codes seen through that task**. Cross-task J uses all pairs of current-task versus prior-task training codes, exactly (no sampled pairs); at Task5 this is 25 million pairs. W_B occupancy is the binary fraction, and column occupancy divides each column's active-slot count by 4096. REF's a/W_B fields are not applicable.

## Accuracy results

All entries below are percentages; differences in the discussion are percentage points (pp).

| Task | REF sparse ridge | R0 raw/evolving | R1 raw/frozen | R2 centered/evolving | R3 centered/frozen |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 78.00 | 92.40 | 92.40 | 95.80 | 95.80 |
| 2 | 55.70 | 51.40 | 90.60 | 70.40 | 89.40 |
| 3 | 56.93 | 36.87 | 88.33 | 57.47 | 86.53 |
| 4 | 60.30 | 33.70 | 86.15 | 53.75 | 85.50 |
| 5 | 65.16 | 33.00 | 85.92 | 47.08 | 85.72 |

| Task5 metric | REF | R0 | R1 | R2 | R3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current task, all seen logits | 71.00 | 94.40 | 86.60 | 95.80 | 88.00 |
| Old classes, all seen logits | 63.70 | 17.65 | 85.75 | 34.90 | 85.15 |
| Current task, five-way logits | 88.60 | 94.60 | 95.60 | 96.00 | 96.20 |

## Mechanism checks at Task5

| Quantity | Raw codes / R0 | Centered codes / R2 | Interpretation |
| --- | ---: | ---: | --- |
| Top 1% bit activation share | 19.75% | 7.35% | Base-task-only centering reduces persistent winner concentration. |
| Zero-winner bits (of 8192) | 1227 | 2 | Centered stream uses almost every projection bit by Task5. |
| Cross-task mean J | 1.5608 | 0.4192 | Exact current-vs-old training-code overlap. |
| P(J≥5) | 4.5433% | 0.3715% | High-overlap cross-task pairs fall about 12.2×. |
| Evolving W_B density | 3.5923% | 4.0257% | Centering does **not** merely reduce overall memory occupancy. |
| Evolving maximum column occupancy | 49.76% | 38.35% | Hotspot pressure falls, although writing still accumulates. |
| Current-test sparse cosine margin | 0.0702 | 0.0806 | Sparse geometry improves modestly. |
| Current-test evolving-a cosine margin | 0.0215 | 0.0371 | Overlap geometry improves under centered evolving writes. |

Frozen W_B density stays at Task1: R1 0.9157%, R3 0.8846%. At Task5 their maximum column occupancies are 36.69% and 14.11%, respectively. Their current-test a margins are 0.0377 (R1) and 0.0660 (R3). These geometry figures are descriptive, not an accuracy decomposition.

## Answers to the repair questions

1. **R1 versus R0:** Task5 cumulative accuracy rises **52.92 pp** (33.00%→85.92%); old-class accuracy rises **68.10 pp** (17.65%→85.75%). This is strong intervention evidence that the evolving W_B trajectory, including its mismatch with previously accumulated ridge statistics, is the dominant practical failure in this ladder. Freezing also removes new-class writes, so this contrast alone does **not** assign all 52.92 pp specifically to the algebraic ridge-feature mismatch rather than other consequences of memory evolution.
2. **R2 versus R0:** Task5 cumulative rises **14.08 pp** (33.00%→47.08%), old-class rises 17.25 pp, and current-task rises 1.40 pp. The hotspot/collision measures move sharply in the intended direction. Thus the original hotspot is not merely a harmless descriptive oddity: an encoder intervention that suppresses it improves downstream performance. The intervention also changes all sparse codes and the resulting write trajectory, so the gain is not a separately identified causal effect of the collision statistic alone.
3. **R3 versus R1/R2:** R3 reaches **85.72%**, 0.20 pp below R1 and 38.64 pp above R2. Centering gives no additional Task5 cumulative gain once W_B is frozen, despite better collision/geometry statistics. The factorial interaction on cumulative accuracy is `R3−R2−R1+R0 = −14.28 pp`; the effects are not additive, and the high frozen-branch performance leaves limited headroom.
4. **Against sparse→ridge:** R0 and R2 remain below REF's 65.16% (by 32.16 and 18.08 pp). Both frozen-W_B branches exceed REF: R1 by 20.76 pp and R3 by 20.56 pp. This is an experimental five-task result, not a paper accuracy claim.
5. **Value of continual writes for new tasks:** At Task5, evolving writes improve current-task accuracy among all seen logits by **7.80 pp** in both code families (R0 94.40% versus R1 86.60%; R2 95.80% versus R3 88.00%). However, they do **not** improve current-task five-way accuracy: R0/R2 score 94.60%/96.00%, versus R1/R3 95.60%/96.20%. Their main visible advantage is favoring new-class logits in the all-seen competition, accompanied by severe old-class degradation.
6. **Fixed feature-map interpretation:** Frozen W_B still gives 95.60% (R1) and 96.20% (R3) Task5 five-way accuracy, plus 86.60%/88.00% all-seen current-task accuracy. In this five-task setting, W_B can therefore act largely as a fixed, useful feature map; later continual writes are **not necessary** for new-class five-way separability. This does not establish the same for Task6–20 or a different class order.
7. **Hotspot/collision evidence:** Task1-only centering lowers top-bit share 19.75%→7.35%, zero-winner bits 1227→2, and P(J≥5) 4.54%→0.37%, while R2 improves cumulative accuracy by 14.08 pp. That is intervention evidence that the original encoder statistics matter downstream; it is not evidence that centering is the authors' hidden preprocessing or an approved change to the formal model.

## Limits and provenance

The two contrasts quantify **bundled intervention effects**, not path-specific causal mediation. R1/R3 remove writes as well as feature-space drift; R2/R3 alter bit usage, class geometry, plateau gating and W_B allocation together. The five-way metric is a restricted-logit diagnostic and is not the official CIL score. REF was reused from the validated archived run, while R0–R3 were fitted here. No default hyperparameter, formal model implementation, original W_B checkpoint, or formal ridge checkpoint was overwritten.
