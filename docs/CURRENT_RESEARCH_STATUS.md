# BTSP-CAM Reproduction: Current Research Status

This document separates the **formal public-description reconstruction** from later **local experimental diagnostics**. Values are percentages unless explicitly labelled otherwise. `Abar` means the arithmetic mean of cumulative seen-class accuracy across tasks; `Final` is the last task's cumulative accuracy. See the [documentation index](README.md) for the full evidence trail. Dataset files, feature caches and binary checkpoints are intentionally not distributed in the public repository.

## 1. Scope

The formal path tests a frozen ViT-B/16, Gaussian projection plus exact k-WTA, class eligibility, adaptive plateau gating, binary XOR memory, CA1, and streaming ridge in CIFAR-100 B0-Inc5. Separate studies diagnose representation, writer drift, readout semantics, k sensitivity, component ablations, and frozen-parameter transfer to CUB-200 B0-Inc10. The latter are research variants, not silent changes to the formal model.

## 2. What is paper-aligned

The core follows the public equations and printed Algorithm 1 ordering, including eligibility update before familiarity/gating, binary memory orientation, normalized CA1 overlap, and literal per-task Eq. (8) accumulation in the formal runner. The CIFAR protocol follows the audited PILOT class order, task splits and preprocessing. Implementation and protocol evidence: [formal alignment](FINAL_PAPER_ALIGNMENT_AUDIT.md), [model status](CORE_MODEL_STATUS.md), [PILOT protocol](CIFAR100_PROTOCOL.md), and [feature output](VIT_FEATURE_OUTPUT_AUDIT.md).

## 3. What remains unspecified

The public description does not uniquely supply all numerical hyperparameters or a state-dependent eligibility-decay rule. In particular, the formal `k=41`, `f_base=0.005` and `p=0.5` are plausible inheritance choices, while `beta=1`, fixed decay `0.01`, `tau=0.5` and `gamma=0.1` are local choices; none should be presented as author settings. The paper describes prototype stability but does not fully define the Figure-4-style prototype metric. Readout wording and Appendix F.2 do not specify an executable historical-feature coordinate-alignment mechanism. See [parameter audit](PARAMETER_AUDIT.md), [tau provenance](TAU_PROVENANCE_AND_DISCRETE_THRESHOLD_AUDIT.md), and [paper readout semantics](PAPER_READOUT_SEMANTICS_AUDIT.md).

## 4. Formal reproduction baseline

The 20-task strict public-description baseline uses literal Eq. (8), formal local parameter choices and thresholded CA1. It completed all tasks with Task20 **Final 4.08% / Abar 5.167%**. This is preserved as a reproduction reference, not promoted as the present working result. The first five tasks showed Task5 94.64% for frozen ViT ridge, 65.16% for sparse-code ridge and 3.96% for full formal BTSP-CAM, focusing diagnosis on the memory/CA1/readout path. See [performance diagnosis](PERFORMANCE_DIAGNOSIS.md) and [CA1 diagnosis](CA1_DIAGNOSIS_ROUND1.md).

## 5. Coordinate-consistency problem

In the literal streaming interpretation, historical `A/B` were formed from old CA1 features, while evolving `W_B` changes the features used for later evaluation. Experimental current-`W_B` replay refit re-encodes all **seen training sparse codes** into a common current feature space. For `tau=0`, aggregated sparse-code `C=Σssᵀ` and `D=Σsyᵀ` allow exact coordinate-aligned statistics without historical raw examples; this algebra does **not** close exactly after positive-threshold ReLU. The experimental replay readout is **not strict exemplar-free**. This local contrast isolates temporal coordinate mismatch; it does not establish what the authors implemented. See [Eq. (8) audit](RIDGE_WB_CONSISTENCY_AUDIT.md), [coordinate-aligned derivation](COORDINATE_ALIGNED_RIDGE.md), and [full20 confirmation](COORDINATE_ALIGNED_FULL20.md).

## 6. CIFAR working baseline

A training-side k×q selection rule froze `k=96`, `q=4`, `tau=4/96`; all other local working settings remained fixed. Current-`W_B` historical **training sparse-code** replay/refit reached **Abar 85.553% / Final 79.180%** on 20 CIFAR tasks. This is the current *local working baseline*: it is not the paper/default configuration, not author-exact, and not strict exemplar-free. The corresponding literal streaming readout on the same local setting gave **31.934% / 20.630%**. Details: [selection and lock](K_TAU_JOINT_SELECTION.md), [ablation baseline](PAPER_STYLE_COMPONENT_ABLATION.md), and [readout comparison](PAPER_READOUT_SEMANTICS_AUDIT.md).

## 7. k robustness

On saved `tau=0` coordinate-aligned CIFAR paths, `k=41/64/128/160` have Abar/Final **80.685/73.300**, **83.601/77.100**, **86.487/80.620**, and **86.901/81.160**. The high-k paths report `k=768/1024/6144/7168` at **89.390/83.920**, **89.552/84.340**, **90.243/84.580**, and **89.745/83.750**. The distinct frozen k96/q4 replay path is **85.553/79.180**. Increased k improves the local representation over much of this range, but the curve is not globally monotone. [Nine paired seed runs](K_ROBUSTNESS_AND_SELECTION.md), [extended search](K_EXTENDED_SEARCH_AND_FREEZE.md), [high-k stress](HIGH_K_STRESS_TEST.md), and [learned-versus-random controls](HIGH_K_MECHANISM_DECOMPOSITION.md) delimit the claim. Neither a high score nor an approximate match to a paper number identifies an author k.

## 8. Component ablations

Under the current-`W_B` replay/refit local operationalization, removing trace gating, collision interference and LTD changes Abar by **−0.698 / −0.156 / −0.078 pp** at k96 and **−0.125 / −0.043 / −0.057 pp** at k1024. The k1024 Task20 LTD effect is about **−0.90 pp**, despite its small Abar effect. Strong Section-6-style effects were not recovered under these two local settings. Some ablations cannot be made paper-exact where the operation is unspecified; raw unnormalized `W_Bs` additionally changes ridge scale. See [operation definitions](PAPER_ABLATION_OPERATIONALIZATION.md), [k96 results](PAPER_STYLE_COMPONENT_ABLATION.md), [k1024 results](CIFAR_K1024_COMPONENT_ABLATION.md), and [forensic checks](PAPER_STYLE_ABLATION_FORENSIC_AUDIT.md).

## 9. CUB-200 transfer

The CIFAR-selected k96/q4 parameters were transferred **unchanged**, without CUB tuning, to the official 5,994-train/5,794-test CUB-200 split and PILOT B0-Inc10 order. The local replay/refit readout reached **Abar 87.8606% / Final 80.6006%**. This supports strong frozen-parameter cross-dataset transfer **under the local coordinate-consistent readout**. It should not be described as “beating the paper” or as directly comparable to paper Table 1 because the readout and possibly image-list details differ. See [protocol](CUB200_PROTOCOL_AUDIT.md) and [full baseline](CUB200_FULL_BASELINE.md).

## 10. Leakage audit

The CUB forensic audit found **no evidence of test leakage** through source split, feature/sparse cache identity, writer, eligibility, ridge fitting, test labels, or evaluation-state mutation. It did find `HISTORICAL_TRAIN_REPLAY=YES`, so `STRICT_EXEMPLAR_FREE=NO` for the local working path. A small number of official cross-split near-duplicate photographs are a benchmark-content caveat, not observed pipeline leakage. This is a bounded audit, not proof that every conceivable pathway is impossible. See [leakage forensic audit](CUB200_LEAKAGE_FORENSIC_AUDIT.md).

## 11. Readout causal decomposition

On one verified CUB writer trajectory, current-`W_B` replay/refit (R0) yields **87.8606/80.6006**, literal old-`A/B` streaming (R1) **54.8096/33.3276**, and a stable-`W_B1` streaming control (R3) **84.9342/77.2006** in Abar/Final. R0 and R1 use identical taskwise writer features at evaluation; they differ in historical readout-statistic maintenance. The stable-map control largely removes the temporal mismatch. These interventions support a local readout-coordinate explanation but not an assertion about author code. See [readout causal decomposition](CUB200_READOUT_CAUSAL_DECOMPOSITION.md).

## 12. Prototype-drift diagnostics

The paper describes a slowly drifting CA1 space and reports prototype similarity `>0.9`; the exact Figure 4 construction is not fully public. On local CIFAR Task1→20 fixed TEST samples, pre-threshold `a` class-prototype cosine is **0.914/0.924/0.930/0.966** for k96/128/160/1024, respectively. For the frozen k96 thresholded `z`, a separate local Figure-4-style audit finds **0.8285** (CIFAR Task1 classes) and **0.4418** (CUB Task1 classes); choices of feature and exposure duration matter. The multi-k writer comparison is non-monotonic: k6144 pre-threshold `a` prototype cosine falls to **0.8736** despite higher gross writes. These are local metrics, not paper-exact Figure 4 reproduction. See [Figure-4-style audit](FIGURE4_STYLE_PROTOTYPE_DRIFT_AUDIT.md) and [multi-k writer drift](K_WRITER_DRIFT_READONLY_AUDIT.md).

## 13. Current interpretation

High class-prototype cosine is a first-order center summary. Ridge statistics depend on the entire feature distribution through `ZᵀZ` and `ZᵀY`; class-center stability alone cannot prove that mixed-time historical statistics remain in the current feature coordinate system. In the local implementation, holding `W_B` fixed or rebuilding historical training features under current `W_B` changes outcomes substantially. The gap is best stated as a **local theory–algorithm/interface and temporal-coordinate diagnostic**, with public-method ambiguity left explicit. It is not a causal percentage decomposition of every experimental difference.

## 14. What is NOT claimed

- No **author-exact reproduction** or inference of the authors' undisclosed k, tau, decay, readout maintenance or CUB parameters.
- No proof of an author bug, omitted alignment operation, or hidden replay.
- No claim that experimental coordinate-consistent repair is the published algorithm.
- No claim that the local CUB score is directly comparable to the paper's Table 1.
- No claim that local Section 6 ablations are paper-exact where their operations or measurement conventions are unspecified.
- No claim that prototype similarity alone establishes historical ridge-statistic coordinate consistency.

Small curated summary CSV/JSON files may be distributed for review; ignored datasets, embeddings, sparse matrices and checkpoints remain local. Their absence from the public repository does not imply they were absent from the experiments.
