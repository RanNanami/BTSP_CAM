# Baseline freeze recommendation

Status: recommendation only. No default configuration or formal reproduction file is changed by this document.

## Keep two named, non-interchangeable baselines

**FORMAL_BASELINE — untouched paper-formula reproduction.** Keep the current default configuration and original readout/CA1 path unchanged, including its explicitly documented local choices (`k=41`, `tau=0.5`, `gamma=0.1`, `beta=1`, fixed `lambda=0.01`). Its prior low accuracy remains a diagnostic result, not a reason to silently overwrite the formal track. Paper-specified architecture/protocol items include `m=8192`, 4096 memory slots, binary memory and BTSP update equations, CIFAR-100 B0-Inc5 and seed 1993 as recorded in the project provenance. The frozen ViT/PILOT extraction protocol is retained. The *numerical values* of several hyperparameters are not uniquely recoverable from the public paper; plausible simpleBTSP inheritance is not equivalent to a BTSP-CAM author disclosure.

**REPAIRED_BASELINE — separate experimental path.** Use the existing coordinate-aligned ridge variant with `z=a=W_Bs/k`, `tau=0`, `k=1024` selected solely by the locked Task1–5 training-side validation rule, `gamma=0.1`, and the same `m=8192`, 4096 slots, class order, frozen ViT cache, augmentation, RF/writer seeds, BTSP writer, `p=0.5`, `f_base=0.005`, `beta=1`, and `lambda=0.01`. The ridge coordinate-alignment (`C=Σssᵀ`, `D=Σsyᵀ`; reconstructing `A=W_BCW_Bᵀ/k²`, `B=W_BD/k`) and removal of the CA1 threshold are **experimental repairs**, not statements of the paper's implementation. The selected `k` is a local, train-only choice for this repaired pipeline, not an author parameter.

## Locked provenance and unresolved points

| Item | Freeze status |
|---|---|
| CIFAR-100 B0-Inc5, class-order seed 1993, frozen ViT/PILOT protocol | Fixed experimental/benchmark protocol; preserve cached extraction provenance. |
| `m=8192`, memory slots=4096, binary BTSP memory rules | Paper-specified model components in project provenance; leave unchanged. |
| `k=41` formal default | Plausible inheritance/local reconstruction; **not paper-disclosed**. Preserve formal default. |
| `k=1024` repaired baseline | Selected from predeclared 128–1024 grid by Task1–5 train-side validation; global optimum **not bracketed**. |
| `f_base=0.005`, `p=0.5` | Plausible simpleBTSP inheritance in the formal config; not proof of BTSP-CAM author values. |
| `beta=1`, fixed `lambda=0.01`, `gamma=0.1` | Local choices; keep numerically fixed for this frozen comparison. State-dependent author `lambda` rule remains unresolved. |
| `tau=0.5` formal versus `tau=0` repaired | Formal `tau` is a local choice with destructive collapse in the current implementation; repaired `tau=0` removes that confound, **not** a recovered author threshold. |
| Coordinate-aligned ridge | Experimental remedy for evolving-`W_B` / historical ridge-statistic mismatch; not the published Eq.(8) implementation claim. |

## Rules after freeze

1. Do not replace the formal default with the repaired variant or back-fill repaired metrics into formal reproduction claims. Label every result by baseline, `k`, `tau`, readout, seed, and selection provenance.
2. Do not use the full20 test curve, its neighbor comparison, or the paper's reported accuracy to change `k`, `gamma`, `tau`, writer settings or the selected baseline. New settings require a new *prospectively locked*, train-only protocol and a new variant name; they must not silently redefine this baseline.
3. No further test-accuracy-driven `k` search follows this round. The extended validation curve is `STILL_INCREASING_AT_BOUNDARY`; the global optimum is unresolved, and `1024` is a constrained grid selection rather than a peak estimate.
4. Treat the 2026-09-29 fresh full20 selected result (89.55% average incremental, 84.34% Task20) and its `k=768` neighbor (89.39%, 83.92%) as locked evaluations of this experimental repair. Numerical proximity to, or exceedance of, a paper number is not evidence of author hyperparameters or strict paper reproduction.

Evidence: `docs/K_EXTENDED_SEARCH_AND_FREEZE.md`, `diagnostics/k_extended_validation.csv`, `diagnostics/k_extended_selection_lock.json`, and `diagnostics/k_extended_full20.csv`.
