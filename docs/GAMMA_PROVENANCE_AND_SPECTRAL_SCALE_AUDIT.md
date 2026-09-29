# Ridge γ provenance, spectral scale, and readout-only diagnostics

Date: 2026-09-29. Scope: CIFAR-100 B0-Inc5, class-order seed 1993, frozen ViT cache, archived `W_B` trajectories, baseline `p=0.5` and trace decay `λ=0.01`, and the **experimental** coordinate-aligned `τ=0` readout. Other non-γ settings, writer sequence, sparse encoder, and sample order remain locked to each archived k trajectory. No `W_B` was retrained, and the formal reproduction/configuration was not changed. This is not a claim about the authors' executable implementation. Accuracy in the diagnostic CSVs is a fraction (multiply by 100 for percent); `N` is the number of seen training samples.

**中文结论。** 论文给出 sum-Gram ridge 公式，但没有公开 γ 数值或选择规则；本地 `γ=0.1` 是 `LOCAL_CHOICE`。固定 γ 相对 Gram trace 的强度从 Task1 到 Task20 下降约 116–186 倍。预先锁定的归一化规则使 k=128/160 平均增量准确率分别提高约 0.59/0.71 个百分点，却几乎不改变 Task20；因此 γ 有功能影响，但不能据此认定它解释论文数值差距，也不能反推出作者 γ。k 的主要准确率优势在归一化后仍保留。下文把论文事实、代码事实与诊断观察分开记录。

## 1. Provenance and formula

| Item | Evidence | Status |
| --- | --- | --- |
| Ridge form | BTSP-CAM [PDF](../../2233_BTSP_CAM_A_Brain_Inspired.pdf), PDF p. 6, §4.5 Eq. (8): `A_t=A_{t-1}+Z_tᵀZ_t`, `B_t=B_{t-1}+Z_tᵀY_t`, `W_out^(t)=(A_t+γI)^{-1}B_t`. Algorithm 1 (p. 5) includes γ as an input. | `PAPER_SPECIFIED` |
| γ numeric value | Full main text, Algorithm 1, Appendix B symbol glossary (p. 12), Appendix F.2 ridge analysis (p. 17), and hyperparameter/implementation passages were checked. The glossary calls it regularization strength but supplies no value. | `GAMMA_VALUE_NOT_SPECIFIED` |
| γ selection, validation, grid search, or dependence on task, feature dimension, sample count, `Z` scale or `k` | No explicit rule found in the supplied paper/appendices. This is **not** evidence that the authors never validated γ; it is a public-description limit. | `NOT_FOUND` |
| simpleBTSP inheritance | The provided [simpleBTSP paper](../../s41467-024-55563-6.pdf), [supplement](../../supplement.pdf), and `upstream/simpleBTSP` sources do not implement an Eq. (8)-style multiclass ridge readout or supply its γ. `gamma_binomal` in upstream theory code is a combinatorial probability routine, not ridge regularization. | `NOT_FOUND` |
| Local value | [`configs/btsp_cam_cifar100.yaml`](../configs/btsp_cam_cifar100.yaml) has `gamma: 0.1` marked `LOCAL_CHOICE`, sourced to a smoke configuration. | `LOCAL_CHOICE`, **not an author parameter** |

Eq. (8) is a **sum Gram**, not `ZᵀZ/N`: `A_t` and `B_t` add unnormalized per-task products. `γI` acts on all CA1 feature coordinates appearing in the formula. The equation contains no separate intercept/bias coefficient, bias exemption, feature-column standardization, or numerical linear-algebra algorithm. Those implementation details are **not specified**, rather than implicitly supplied. A schematic “Norm” label elsewhere in the paper does not define a replacement for Eq. (8)'s Gram scaling. The inverse is mathematical notation; it does **not** establish that the authors explicitly inverted a matrix rather than solving a system.

The local [`src/btsp_cam/ridge.py`](../src/btsp_cam/ridge.py) accumulates float64 `ZᵀZ` and `Zᵀ` times one-hot labels, pads the class columns as new classes arrive, and uses float64 Cholesky (`cho_factor`/`cho_solve`) plus a LAPACK 1-norm condition estimate. It neither divides by `N`, standardizes features, adds an intercept, nor explicitly forms an inverse. Its solve rejects `γ≤0`. The experimental [`coordinate_aligned_ridge.py`](../experiments/coordinate_aligned_ridge.py) maintains `C=Σssᵀ`, `D=Σsyᵀ` and forms `A=W_B C W_Bᵀ/k²`, `B=W_B D/k`, also without `/N`; this is **not** the paper's Eq. (8) temporal update. The current audit changes only γ while keeping these aligned `A/B`, `W_B`, data, and `τ=0` fixed.

## 2. Predeclared experiment and numerical definitions

The separate [audit configuration](../experiments/configs/gamma_spectral_audit.yaml) and [runner](../experiments/run_gamma_spectral_audit.py) lock the absolute grid to `{0, 0.001, 0.01, 0.1, 1, 10}` for k=128/Task1–10. `γ=0` uses an exact-eigenbasis pseudoinverse with numerical-rank threshold `d·ε·max(λmax,1)`, not a fragile inverse. For full20, `FIXED_GAMMA=0.1`; `TRACE_NORMALIZED` uses `α_trace·trace(A_t)/4096`; `MAX_NORMALIZED` uses `α_max·λmax(A_t)`. Both α values are determined **once** from k=128/Task1 train statistics so that its normalized γ equals 0.1, and are then frozen across all tasks and both k values. No test result enters these rules.

The spectrum CSV uses exact float64 symmetric eigendecomposition at tasks 1, 5, 10, 15, and 20 for k=41/128/160. Intermediate-task `λmax` for normalized γ uses Lanczos (`eigsh`, relative tolerance `1e-7`); key checkpoints use exact eigenvalues. `rank` counts eigenvalues above the stated tolerance; `effective_rank=exp(−Σ p_i log p_i)` with positive eigenvalues normalized to sum 1. Thus near-one effective rank can coexist with full numerical rank when one large common mode dominates. Eigenvalue quantiles include all 4096 eigenvalues; Task1's tiny negative entries are float64 roundoff around the exact zeros of its rank-deficient Gram. `condition_A_plus_gamma0` is the spectral 2-norm condition; the ridge CSV separately records LAPACK's 1-norm estimate when a Cholesky solve was used. `NaN` in a condition field means that particular estimator was not run; it is not zero. `Z` first/second moments and column-variance quantiles refer to all seen train samples; the separately labeled current-test density/active mean are descriptive diagnostic measurements. The finalizer replays cached historical **train features only to measure exact `Z` L2/active moments**; it does not feed those samples into `C/D`, the ridge fits, or any `W_B` update.

The algebraic sample-count control is exact:

`(A/N + γI)^{-1}(B/N) = (A + NγI)^{-1}B`.

Therefore a fixed `γ` in the paper's sum-Gram convention is **not** a fixed penalty per sample. Using a mean Gram with the same printed γ corresponds to a sum-Gram penalty multiplied by `N`; this control clarifies scale, it is not another author-algorithm claim.

## 3. Results

### 3.1 Gram spectrum and dimensionless strength

Full results (including tasks 5/10/15, eigenvalue p1–p99, train `Z` moments, and column-variance distribution) are in [`gamma_spectrum.csv`](../diagnostics/gamma_spectrum.csv). All three k values use exactly the same task/class order and a separate archived memory trajectory appropriate to that k; changing k is **not** merely rescaling an unchanged `Z`.

| k | `trace(A)/d`, T1 → T20 | growth | `λmax`, T1 → T20 | `γ0/(trace(A)/d)`, T1 → T20 | `γ0/median(λ>0)`, T1 → T20 | T20 effective rank |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 41 | 34.99 → 6516.25 | 186.3× | 1.166×10⁵ → 2.569×10⁷ | 2.858×10⁻³ → 1.535×10⁻⁵ | 0.5235 → 0.00360 | 1.517 |
| 128 | 73.33 → 9355.72 | 127.6× | 2.817×10⁵ → 3.797×10⁷ | 1.364×10⁻³ → 1.069×10⁻⁵ | 0.3126 → 0.00416 | 1.126 |
| 160 | 85.22 → 9891.29 | 116.1× | 3.316×10⁵ → 4.023×10⁷ | 1.173×10⁻³ → 1.011×10⁻⁵ | 0.2978 → 0.00466 | 1.096 |

`γ0/λmax` likewise falls, e.g. k=128: `3.550×10⁻⁷ → 2.634×10⁻⁹` (135× weaker). All Task1 Gram matrices have numerical rank **2500/4096**; all Task5–20 checkpoint Grams are full numerical rank under the declared tolerance. Task20 `cond₂(A+0.1I)` is approximately `9.46×10⁷` (k=41), `7.19×10⁷` (k=128), and `7.11×10⁷` (k=160). Their very low entropy effective ranks do **not** imply numerical rank one; they reflect a dominant noncentered common mode. Task20 all-train mean `Z` is `0.350/0.429/0.442` for k=41/128/160, respectively. The detailed quantiles show substantial positive tail/mid-spectrum beyond that mode.

The exact all-seen-train mean `||z||₂` rises from **7.31→22.90** (k=41), **10.74→27.63** (k=128), and **11.61→28.42** (k=160). The corresponding positive-element density approaches 100% by Task20 at `τ=0`; this is unthresholded overlap, not the paper's sparse CA1 claim. These feature-scale observations explain why the Gram grows much faster than the sample count alone.

From Task1 to Task20, `N` grows 20×, while `trace(A)/d` grows 116–186×. Thus sample accumulation explains only part of the fixed-γ weakening; per-sample squared feature norm also rises about 5.8–9.3× as the archived memory evolves. At Task20, the same printed `γ=0.1` occupies different spectral regimes across k, especially k=41 versus 128/160. `r_trace`, `r_max`, and `r_med` are **diagnostic definitions**, not author-prescribed normalizations.

### 3.2 Absolute-γ grid, k=128/Task1–10

This grid was locked before the run; all points use the **same** `W_B/C/D` for each task. [`gamma_fixed_task10.csv`](../diagnostics/gamma_fixed_task10.csv) has all 60 task–γ rows, including old/current/5-way accuracy, weight norm, logit norm, train residual, spectral condition and post-regularization minimum eigenvalue. Selected cumulative accuracies:

| γ | Task1 | Task5 | Task10 |
| ---: | ---: | ---: | ---: |
| 0 (spectral pseudoinverse) | 68.00% | 88.76% | 86.06% |
| 0.001 | 74.00% | 88.84% | 86.08% |
| 0.01 | 88.00% | 89.04% | 86.10% |
| 0.1 (`LOCAL_CHOICE`) | 96.60% | 90.40% | 86.42% |
| 1 | 98.00% | 91.12% | 86.76% |
| 10 | 98.00% | 91.84% | 86.94% |

At Task1, γ=0 is rank-deficient (2500/4096) and fits its training one-hot targets to numerical precision, yet generalizes much worse; this is not a failed inverse. Across this **predeclared** grid, γ materially affects classification, most dramatically early. The monotone values in this narrow grid are descriptive and do **not** select γ=10 or imply a better author configuration. There is only one benchmark class order/trajectory here, not a validation confidence interval.

### 3.3 Locked normalized-γ full20

From k=128/Task1 train Gram, `α_trace = 0.00136366168004` and `α_max = 3.54951094640×10⁻⁷`. These constants were not changed for k=160. [`gamma_normalized_full20.csv`](../diagnostics/gamma_normalized_full20.csv) records every task for k=128/160 plus a **Task20-only** k=41 mechanism control. The 20-task average is the mean of cumulative accuracy after each task, matching the archived incremental-accuracy convention.

| k | Readout | γ at Task20 | Avg T1–20 | T5 | T10 | T15 | T20 | T20 old / current / current-5way |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | fixed 0.1 | 0.10 | 86.49% | 90.40% | 86.42% | 83.20% | 80.62% | 80.86 / 76.0 / 92.2% |
| 128 | trace-normalized | 12.76 | 87.07% | 91.44% | 86.94% | 83.16% | 80.54% | 80.72 / 77.2 / 93.0% |
| 128 | max-normalized | 13.48 | 87.08% | 91.48% | 86.92% | 83.17% | 80.50% | 80.67 / 77.2 / 93.0% |
| 160 | fixed 0.1 | 0.10 | 86.90% | 90.48% | 86.34% | 83.65% | 81.16% | 81.31 / 78.4 / 92.6% |
| 160 | trace-normalized | 13.49 | 87.61% | 91.84% | 87.56% | 83.93% | 81.19% | 81.36 / 78.0 / 93.6% |
| 160 | max-normalized | 14.28 | 87.62% | 91.80% | 87.58% | 83.93% | 81.20% | 81.37 / 78.0 / 93.8% |

The full20 fixed-γ rows reproduce the archived aligned baseline **at every task**: k=128 final 80.62%, k=160 final 81.16%. Locked normalized γ raises average incremental accuracy by **+0.59 pp** (k=128) and **+0.71 pp** (k=160), but Task20 changes by only **−0.08 to −0.12 pp** and **+0.03 to +0.04 pp**, respectively. It helps early/midstream generalization more than final 100-class accuracy. No per-task γ was chosen from test results.

For a k=41 Task20-only same-rule control, fixed/trace/max give **73.30/73.83/73.84%**. The k=128–k=41 final gap is 7.32 pp at fixed γ and 6.71 pp under trace normalization; k=160–k=41 is 7.86 pp fixed and 7.36 pp trace-normalized. Consequently, most observed k benefit remains after this readout-scale control; it is **not** solely a fixed-γ scale artifact. These are comparisons of differently trained `W_B` trajectories, not a causal decomposition of sparse encoding versus writing.

### 3.4 Numerical and sample-count controls

[`ridge_numerics.csv`](../diagnostics/ridge_numerics.csv) records all 15 exact spectral checkpoints. `A` symmetry error is exactly zero at these checkpoints; every positive-γ float64 Cholesky solve succeeded. The largest relative `(A+γI)W−B` residual for baseline γ=0.1 is `1.07×10⁻¹³`. At `N=2500` the smallest eigenvalues are numerical zeros, while at later checkpoints `A` is full rank; high condition numbers do not by themselves constitute observed solve failure. A small backward residual also does **not** guarantee tiny forward coefficient error for an ill-conditioned system. Independent mean-Gram versus sum-Gram solves implementing the identity above agree in weights within `1.47×10⁻¹⁵` maximum absolute difference. The formal/local solve is float64; a separate float32 Cholesky experiment was **not** performed, so no float32 stability claim is made. Temporary sparse binary overlap multiplication uses float32 only to sum ≤160 exact integer bits before division in float64.

## 4. Classification and next priority

**Provenance: `C — GAMMA_VALUE_NOT_SPECIFIED`.** The paper specifies the ridge **formula**, not the numeric γ, validation rule, or scaling schedule; the local 0.1 is `LOCAL_CHOICE`. This is not `PAPER_GAMMA_FULLY_SPECIFIED`, and the implemented sum-Gram formula is not itself a demonstrated local mismatch. The experimental coordinate-aligned temporal repair is intentionally separate from the paper's stated incremental Eq. (8).

**Function: `FIXED_GAMMA_SCALE_DRIFT` and `GAMMA_MATERIALLY_AFFECTS_CLASSIFICATION` are supported.** `GAMMA_K_INTERACTION` is supported in the limited sense that the Gram spectrum and sensitivity of the *fixed* γ differ across k; the k advantage persists after the two predeclared normalizations. `GAMMA_NEGLIGIBLE` is rejected. The strongest effect is early; normalized γ does not produce a reliable Task20 improvement. The fact that k=160/max-normalized yields 87.62% average and 81.20% final—numerically near/above the paper's 87.82/80.15—**cannot identify the authors' γ**, because this is the tau-free coordinate-aligned repair with an experimental k and a test-evaluated diagnostic. It does not resolve the remaining paper-method gap.

Of the proposed next priorities, **readout hidden implementation/source clarification** has higher evidential value than more `f_base` or `beta` sweeps: Eq. (8)'s temporal semantics and unpublished γ remain unresolved, whereas the current fixed-grid/normalization results do not justify further test-driven parameter search. If no author implementation/configuration is available, stop trying to infer author settings from score proximity. Preserve the formal baseline unchanged.

### Direct answers

1. **γ published?** Formula/symbol yes; numeric value and selection rule no (`GAMMA_VALUE_NOT_SPECIFIED`).
2. **Is 0.1 an author parameter?** No evidence; it is local.
3. **Sum or normalized Gram?** Eq. (8) writes an unnormalized sum; no `/N`.
4. **Does effective fixed-γ strength drift?** Yes, strongly downward relative to trace, spectral maximum and median positive eigenvalue.
5. **Magnitude?** `trace(A)/d` grows 186×/128×/116× for k=41/128/160, so `r_trace` falls by those factors; `N` alone grows 20×.
6. **Different k regimes?** Yes: at Task20 `trace(A)/d` is 6516/9356/9891, with differing spectra and feature scales.
7. **Does normalized γ change results?** It raises the average by ~0.6–0.7 pp for k=128/160, but changes Task20 by at most 0.12 pp in either direction.
8. **Does k benefit survive normalization?** Yes; Task20 k=128/160 remain ~6.7/7.4 pp above k=41 under trace normalization.
9. **Does γ explain the paper-score gap?** Not established. The diagnostic shifts early averages but barely shifts final accuracy, and it is not the author's specified algorithm.
10. **Next priority?** Clarify hidden readout implementation/provenance, not another broad `f_base`/`beta` accuracy search; stop parameter inference if that source cannot be obtained.
