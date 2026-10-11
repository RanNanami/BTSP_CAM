# CIFAR-100 k=1024 paper-style component ablation

Status: **LOCAL EXPERIMENTAL DIAGNOSTIC; NOT PAPER-EXACT.** This run does not identify the authors' k or other hidden parameters. It compares component operations under a deliberately fixed, current-W_B historical-TRAIN replay/refit readout. It is **not** the paper's literal streaming Eq. (8), and the replay readout is **not exemplar-free**.

## Locked protocol and provenance

- CIFAR-100 B0-Inc5, 20 × 5 classes, fixed class order/stream and frozen ViT/projection caches, seed 1993; no task identity at cumulative seen-class evaluation.
- `m=8192`, `slots=4096`, `k=1024`, `q=4`, `tau=q/k=0.00390625`; `f_base=0.005`, `beta=1`, `lambda=0.01`, `eta=1`, `p=0.5`, `gamma=0.1`.
- All branches use exactly the same cached binary codes with 1024 active bits per sample. Full, A1, A2, and A3 use separate writer states and RNG instances with identical seed provenance. A4a/A4b reuse Full W_B checkpoint hashes task by task.
- Full W_B was reused only **after** an in-memory replay of all 20 tasks verified archived taskwise density, gross flips and every checkpoint **bit for bit**. No archived checkpoint was overwritten. A1/A2/A3 checkpoints are new, isolated, bit-packed diagnostic files.
- Each task's ridge is freshly fitted from all **seen TRAIN** sparse codes transformed by that branch's **current** W_B. No validation/test sample or future-task train sample enters the fit. Cumulative seen-class test is evaluation only.
- Existing k=96 q=4 results were read from the archived table, not rerun. Existing k=1024 tau=0 high-k results are a regression reference, not a required q=4 target.

Source: [`experiments/audit/cifar_k1024_component_ablation.py`](../experiments/audit/cifar_k1024_component_ablation.py). Existing Full trajectory: `diagnostics/high_k_mechanism_cache/k1024/WB_task01...20.npz`. Verification: [`diagnostics/cifar_k1024_full_writer_verification.json`](../diagnostics/cifar_k1024_full_writer_verification.json).

## Main result

Accuracies and deltas are in percentage points (pp); occupancy/density are percentages. Abar is the arithmetic mean of the 20 cumulative seen-class accuracies, not Task20 alone. `Old` and `Current` use the 95/5 class split at Task20; `5-way` restricts Task20 logits to its five classes.

| Variant | Abar | Δ Abar vs Full | Final | Δ Final vs Full | Old T20 | Current T20 | 5-way T20 | W_B density T20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| K1024_FULL | 89.552 | 0.000 | 84.340 | 0.000 | 84.516 | 81.000 | 94.000 | 46.087 |
| A1_TRACE_FIXED_FBASE | 89.426 | −0.125 | 84.150 | −0.190 | 84.305 | 81.200 | 94.600 | 48.854 |
| A2_NO_COLLISION_I1 | 89.509 | −0.043 | 84.150 | −0.190 | 84.274 | 81.800 | 94.000 | 47.734 |
| A3_ONLY_LTP | 89.495 | −0.057 | 83.440 | −0.900 | 83.621 | 80.000 | 94.200 | 84.514 |
| A4A_NO_THRESHOLD | 89.552 | 0.000 | 84.340 | 0.000 | 84.516 | 81.000 | 94.000 | 46.087 |
| A4B_RAW_WBS_SCALE_CONFOUNDED | 87.420 | −2.132 | 84.230 | −0.110 | 84.411 | 80.800 | 94.000 | 46.087 |

Full's Task20 84.340% and Abar 89.552% equal the archived k=1024 tau=0 accuracy aggregates. A4a's **all 20** cumulative accuracy values match that archived trajectory within `1e-12`, establishing the tau=0 readout regression. The q=4 Full and tau=0 A4a accuracies also agree at every task, although their feature magnitudes and Task20 cosine margins are not identical.

### Writer operations and scale

| Variant | Plateau events / task | Cumulative 0→1 | Cumulative 1→0 | Task20 CA1 element density | Task20 cosine margin |
|---|---:|---:|---:|---:|---:|
| Full | 17,304 | 96,334,516 | 80,870,280 | 100% | 0.000132 |
| A1 fixed f_base | 51,224 | 270,481,610 | 254,088,761 | 100% | 0.000125 |
| A2 I=1 | 29,193 | 157,488,450 | 141,471,740 | 100% | 0.000128 |
| A3 only LTP | 17,304 | 28,358,114 | **0** | 100% | 0.000011 |
| A4a no threshold | Full | Full | Full | 100% | 0.000130 |
| A4b raw W_Bs | Full | Full | Full | 100% | 0.000130 |

A1 is `PAPER_UNSPECIFIED_LOCAL_OPERATIONALIZATION`: eligibility is still updated for logging, but plateau sampling uses exactly `f_base`, not F/N/I. A2 (`PAPER_IMPLIED`) only sets interference I to 1; F/N/eligibility remain active. A3 (`PAPER_IMPLIED`) consumes the same plateau and synaptic Bernoulli draws as Full, then restores sampled 1→0 outcomes. Its per-task plateau counts and sampled flip attempts equal Full, and actual LTD is zero. A4a is a local threshold-only readout operation. A4b is **NON_PURE_ABLATION / SCALE_CONFOUNDED**. A5 consolidation was not run: its replacement classifier is `NOT_REPRODUCIBLE_FROM_PUBLIC_DESCRIPTION`.

At Task20, A4b multiplies the unthresholded normalized feature by 1024; its Gram scale is 1,048,576× while gamma remains 0.1. Its Task2 accuracy drops from Full 96.0% to 70.9% and its ridge condition estimate grows from ~`6.70e8` to ~`4.73e10`, then later tasks largely recover. This unstable readout-scale effect is not evidence for or against biological CA1 competition.

## k=96 vs k=1024, same discrete q=4

The archived k=96 Full was Abar 85.553%, Final 79.180%. Effects below are **ablated − Full** within each k, in pp; the last column is the k=1024 effect minus the k=96 effect.

| Operation | k=96 Δ Abar | k=1024 Δ Abar | Change in Abar effect | k=96 Δ Final | k=1024 Δ Final |
|---|---:|---:|---:|---:|---:|
| A1 −trace gating | −0.698 | −0.125 | +0.573 | −0.180 | −0.190 |
| A2 −collision I | −0.156 | −0.043 | +0.113 | +0.410 | −0.190 |
| A3 −LTD | −0.078 | −0.057 | +0.021 | +0.040 | −0.900 |
| A4a −threshold | −0.011 | 0.000 | +0.011 | −0.030 | 0.000 |
| A4b raw W_Bs | −4.044 | −2.132 | +1.913 | −0.130 | −0.110 |

Thus k=1024 **does not restore a strong Section-6-style classification ablation effect** under this repaired readout: A1/A2/A3 Abar effects are all smaller in magnitude than at k=96. There is a **partial terminal LTD signal**: A3's Task20 effect changes from +0.04 to −0.90 pp, but its Abar effect remains −0.057 pp. The A4b comparison is descriptive only because the raw-feature gamma confound differs enormously with k.

## Saturation and writer distance

| Task | Full W_B density | A1 | A2 | A3 |
|---:|---:|---:|---:|---:|
| 1 | 16.990% | 28.268% | 22.026% | 21.041% |
| 5 | 36.197% | 44.198% | 40.444% | 57.000% |
| 10 | 42.209% | 47.269% | 45.053% | 72.627% |
| 20 | 46.087% | 48.854% | 47.734% | 84.514% |

Full/A1/A2 move toward the ~50% occupancy regime as both XOR directions continue; their density gap compresses, especially late. This is a plausible occupancy-ceiling context, **not proof** that their connection structures or mechanisms are equivalent. A3 has no LTD and therefore exceeds 50% dramatically. At Task20, Full–A1, Full–A2 and Full–A3 Hamming distances are respectively **16,393,927 (48.86%)**, **16,009,258 (47.71%)**, and **12,893,878 (38.43%)** of all 33,554,432 synapses. Thus near-equal accuracy does **not** mean unchanged W_B. Row- and column-degree differences at Tasks 1/5/10/20 are retained in the writer-distance CSV. A3's Task20 cosine margin falls from Full `1.32e−4` to `1.13e−5`; the terminal classification loss is modest by comparison, so geometry and accuracy should not be equated.

The CA1 element density reaches **100% for every branch** at Task20. In this high-k regime, q=4 does not produce a sparse CA1 layer; Full Task1 was already 98.706% dense. The Full/A4a Task20 accuracy equality therefore does not validate q=4 as a functional competition threshold.

## Existing learned-vs-random context (not rerun here)

The separate, existing k=1024 tau=0 diagnostic reports learned W_B Abar/Final **89.552/84.340%**, matched Bernoulli **89.396/84.060%**, and row-shuffled **89.397/84.088%**. The learned-memory advantage there is only about **+0.155 pp Abar** and **+0.25–0.28 pp Final**. That contextualizes why large writer-level changes can have small classification effects here, but does **not** causally explain each ablation delta and is not a new control from this run.

## Verification and outputs

- `diagnostics/cifar_k1024_ablation_full20.csv`: all 120 branch/task readouts, including old/current/5-way, density, cosine geometry, ridge condition, flips, running Abar.
- `diagnostics/cifar_k1024_ablation_mechanisms.csv`: 80 Full/A1/A2/A3 taskwise writer records, RNG-attempt accounting and checkpoint digests.
- `diagnostics/cifar_k1024_ablation_writer_distance.csv`: 12 Full-vs-writer comparisons at Tasks 1/5/10/20.
- `diagnostics/cifar_k96_vs_k1024_ablation_compare.csv`: within-k effect comparison; k=96 archive unchanged.
- `diagnostics/cifar_k1024_ablation_summary.json`: independent aggregates and protocol labels.
- New writer checkpoint roots: `diagnostics/cifar_k1024_ablation_cache/A1_TRACE_FIXED_FBASE`, `A2_NO_COLLISION_I1`, `A3_ONLY_LTP`. Existing Full checkpoint root remains untouched.

`PYTHONPATH=src;experiments;upstream/LAMDA-PILOT python -m pytest -q tests` passed: **62 passed, 3 subtests passed**. Full replay verification took 105.35 s, new writer trajectories 340.87 s, six readout trajectories 4800.63 s; total measured computational phases ~**87.45 min**, excluding tests/reporting. No formal source/default config, existing cache/result/checkpoint, commit, or remote was changed by this audit. The workspace had unrelated pre-existing untracked files; `git status --short` should be interpreted accordingly.

**Answer:** `DOES_HIGH_K_RESTORE_PAPER_STYLE_ABLATION_EFFECT = NO` for the primary Abar criterion under this specific repaired readout. A3 has a limited Task20-only loss. `READOUT = CURRENT_WB_REPLAY_REFIT`; `PAPER_EXACT = NO`.
