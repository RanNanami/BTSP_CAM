# Documentation index

Start with [Current Research Status](CURRENT_RESEARCH_STATUS.md), then use the topic groups below. The **formal public-description path**, **tau=0 coordinate-aligned diagnostic**, and **positive-threshold current-`W_B` replay/refit working baseline** are different implementations/readouts. An archived result is evidence for its stated setting, not a replacement for a later result. Large datasets, feature/sparse caches and checkpoints are not committed; small curated result summaries are linked from the relevant reports.

## A. Paper, implementation and protocol audits

- [Final paper-alignment audit](FINAL_PAPER_ALIGNMENT_AUDIT.md); [core model status](CORE_MODEL_STATUS.md); [BTSP to BTSP-CAM inheritance](BTSP_TO_BTSP_CAM.md).
- [CIFAR protocol](CIFAR100_PROTOCOL.md); [preprocessing provenance](CIFAR100_PREPROCESS_PROVENANCE_AUDIT.md); [ViT feature output](VIT_FEATURE_OUTPUT_AUDIT.md); [ViT weight provenance](VIT_WEIGHT_PROVENANCE_AUDIT.md).
- [Parameter audit](PARAMETER_AUDIT.md); [second provenance round](PARAMETER_PROVENANCE_ROUND2.md); [code audit](CODE_AUDIT_ROUND2.md).

## B. CA1 and memory diagnostics

- [CA1 first-round diagnosis](CA1_DIAGNOSIS_ROUND1.md); [paper formula audit](CA1_PAPER_FORMULA_AUDIT.md); [reference audit](CA1_REFERENCE_AUDIT.md); [final formula interpretation](CA1_FINAL_INTERPRETATION.md).
- [Tau provenance and discrete threshold](TAU_PROVENANCE_AND_DISCRETE_THRESHOLD_AUDIT.md) (its then-unavailable k128/k160 Part E is **SUPERSEDED FOR CURRENT INTERPRETATION** by the later [k128/k160 q-oracle](TAU_Q_ORACLE_K128_K160.md)); [tau=0.005 check](TAU_005_TARGETED_CHECK.md); [aligned threshold oracle](ALIGNED_THRESHOLD_ORACLE_AUDIT.md).
- [Memory drift source](MEMORY_DRIFT_SOURCE_AUDIT.md); [W_B capacity](WB_CAPACITY_AUDIT.md); [column hotspots](COLUMN_HOTSPOT_AUDIT.md); [gating and functional drift](GATING_AND_FUNCTIONAL_DRIFT_AUDIT.md); [class drift](GATING_CLASS_DRIFT_AUDIT.md).

## C. Readout semantics and theory

- [Paper readout semantics](PAPER_READOUT_SEMANTICS_AUDIT.md); [ridge/W_B temporal consistency](RIDGE_WB_CONSISTENCY_AUDIT.md); [Appendix F.2 drift theory](DRIFT_THEORY_AUDIT.md).
- [Gamma provenance and spectral scale](GAMMA_PROVENANCE_AND_SPECTRAL_SCALE_AUDIT.md); [readout numerical/parameter screening](UNRESOLVED_PARAMETER_SCREEN.md).

## D. Coordinate-alignment experiments

- [Five-task repair ladder](REPAIR_LADDER_TASK1_5.md); [W_B drift decomposition](WB_DRIFT_DECOMPOSITION.md); [exact sparse C/D alignment](COORDINATE_ALIGNED_RIDGE.md); [full20 scaling](COORDINATE_ALIGNED_FULL20.md).
- [Tau/drift diagnostic](TAU_DRIFT_ANALYSIS.md); [counterfactual gating replay](COUNTERFACTUAL_GATING_REPLAY.md).

## E. k, sparse representation and high-k analysis

- [k robustness and train-only selection](K_ROBUSTNESS_AND_SELECTION.md); [extended search](K_EXTENDED_SEARCH_AND_FREEZE.md); [frozen k41 verification](K41_FROZEN_VERIFICATION.md); [local k×q selection](K_TAU_JOINT_SELECTION.md).
- [High-k stress](HIGH_K_STRESS_TEST.md); [learned versus random memory controls](HIGH_K_MECHANISM_DECOMPOSITION.md).
- [Random feature scale audit](RANDOM_FEATURE_SCALE_AUDIT.md); [scale counterfactuals](RANDOM_FEATURE_SCALE_DIAGNOSIS.md); [sparse encoder bias](SPARSE_ENCODER_BIAS_AUDIT.md); [Appendix C collision assumptions](APPENDIX_C_COLLISION_ASSUMPTION_TEST.md); [collision to interference](COLLISION_TO_INTERFERENCE_AUDIT.md).

## F. Component ablations and local parameter mechanisms

- [Paper Section-6 operation definitions](PAPER_ABLATION_OPERATIONALIZATION.md); [k96 component ablation](PAPER_STYLE_COMPONENT_ABLATION.md); [k1024 component ablation](CIFAR_K1024_COMPONENT_ABLATION.md); [ablation forensic checks](PAPER_STYLE_ABLATION_FORENSIC_AUDIT.md).
- [Fbase/beta sensitivity](FBASE_BETA_SENSITIVITY.md); [eligibility decay provenance](LAMBDA_PROVENANCE_AND_STATE_DEPENDENCE_AUDIT.md); [p write-structure audit](P_PROVENANCE_AND_WRITE_STRUCTURE_AUDIT.md).

## G. CUB-200 frozen-parameter transfer

- [CUB protocol](CUB200_PROTOCOL_AUDIT.md); [CUB frozen Full baseline](CUB200_FULL_BASELINE.md).

## H. Leakage and readout causality

- [CUB test-leakage forensic audit](CUB200_LEAKAGE_FORENSIC_AUDIT.md); [same-writer readout causal decomposition](CUB200_READOUT_CAUSAL_DECOMPOSITION.md).

## I. Prototype and writer drift

- [Figure-4-style prototype drift](FIGURE4_STYLE_PROTOTYPE_DRIFT_AUDIT.md); [multi-k writer drift, pre-threshold](K_WRITER_DRIFT_READONLY_AUDIT.md).

## J. Historical / superseded for current interpretation

These documents remain available as dated or phase-specific records; **SUPERSEDED FOR CURRENT INTERPRETATION** does not mean their original measurements have been altered.

- [Implementation plan](IMPLEMENTATION_PLAN.md), [reproduction log](REPRODUCTION_LOG.md), [round-two report](ROUND2_REPORT.md), [baseline-freeze recommendation](BASELINE_FREEZE_RECOMMENDATION.md): early project state; use [Current Research Status](CURRENT_RESEARCH_STATUS.md) for present conclusions.
- [Initial performance diagnosis](PERFORMANCE_DIAGNOSIS.md): valid for its original formal settings, not the local working readout.

## Result-artifact boundary

The public repository includes only a small allowlist of human-reviewable CSV/JSON summaries under `diagnostics/`. Unlisted diagnostic tables may be referenced by historical reports but remain local; binary `W_B`, ridge, sparse-code and feature caches, dataset images and prediction dumps are intentionally excluded. A missing local-only artifact in a fresh clone is not evidence that it was not used in the archived run.

### Public lightweight result index

| Topic | Curated result files |
|---|---|
| Formal first-round diagnosis | [representation summary](../diagnostics/representation_diagnostics_summary.json), [CA1 summary](../diagnostics/ca1_diagnostics_round1_summary.json) |
| Frozen CIFAR k96/q4 confirmation | [20-task test confirmation](../diagnostics/k_tau_full20_confirmation.csv) |
| k/seed/full20 paths | [paired-seed summary](../diagnostics/k_seed_robustness.csv), [k-path taskwise](../diagnostics/k_path_decomposition.csv), [extended full20](../diagnostics/k_extended_full20.csv), [high-k stress full20](../diagnostics/high_k_stress_full20.csv), [mechanism summary](../diagnostics/mechanism_decomposition.csv) |
| Component ablation | [k96 taskwise](../diagnostics/paper_style_ablation_full20.csv), [k1024 summary](../diagnostics/cifar_k1024_ablation_summary.json), [k1024 taskwise](../diagnostics/cifar_k1024_ablation_full20.csv) |
| CUB transfer | [Full summary](../diagnostics/cub200_full_summary.json), [Full taskwise](../diagnostics/cub200_full_taskwise.csv) |
| CUB readout and leakage | [readout summary](../diagnostics/cub200_readout_decomposition_summary.json), [readout taskwise](../diagnostics/cub200_readout_decomposition_taskwise.csv), [coordinate mismatch](../diagnostics/cub200_readout_coordinate_mismatch.csv), [split audit](../diagnostics/cub200_raw_split_leakage.json), [near-duplicate review](../diagnostics/cub200_near_duplicate_review.json) |
| Prototype/writer drift | [Figure-4-style summary](../diagnostics/figure4_style_prototype_drift_summary.json), [multi-k writer summary](../diagnostics/k_writer_drift_readonly_summary.csv), [classwise Task1→20](../diagnostics/k_task1_to_task20_prototype_drift.csv), [20-point trajectory](../diagnostics/k_task1_drift_trajectory.csv) |

The allowlisted files retain their original numerical values. Some detailed per-sample evidence remains local because of size or machine-specific paths; the reports distinguish those cases from files available in this index.
