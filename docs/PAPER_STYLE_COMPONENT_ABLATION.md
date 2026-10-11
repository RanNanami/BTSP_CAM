# Section 6-style component ablations on the frozen local baseline

**Scope:** CIFAR100 B0-Inc5, class/RF/writer seeds 1993, k=96, q=4,
tau=4/96, m=8192, slots=4096, f_base=.005, beta=1, lambda=.01, p=.5,
gamma=.1. All results use current-W_B historical sparse-code replay refit,
which is an **experimental coordinate-consistent readout, not the strict
exemplar-free paper algorithm**. No parameter was retuned per ablation.
Operational definitions and evidence status are in
`docs/PAPER_ABLATION_OPERATIONALIZATION.md`. The local paper PDF's Section 6
Tables 2–5 were checked both in extracted text and page renders.

The Section 6 paper Full Abar is **86.82%**, distinct from Table 1's CIFAR100
BTSP-only **87.82/80.15%**. The difference is recorded as
`PAPER_INTERNAL_EXPERIMENTAL_CONTEXT_NOT_FULLY_SPECIFIED`; neither was used
as a target. The only Full reference below is the already frozen **85.553%
Abar / 79.180% final**. A new readout pass matched its 20-task cumulative
accuracy trajectory within 1e-12; no new Full writer trajectory was trained.

## Main full20 table

Percentages are absolute except ΔAbar/Fgt. Fgt, Ov. and O→N use explicitly
local definitions, **not paper-exact metrics**. LTP/LTD are total gross
transitions across all 20 tasks; #Plateau is the mean per task. The paper
does not disclose enough to compute its Table 4 margin on the same scale;
the last column here is our same-minus-different sample cosine margin.

| Variant | Abar | ΔAbar | Final | Old | Fgt local | W_B density | #Plateau/task | LTP total | LTD total | Ov. local | O→N local | CA1 density | Cosine margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 85.553% | — | 79.180% | 79.463% | 5.842pp | 17.204% | 16,340 | 10,729,173 | 4,956,425 | .99710 | .926% | ~100% | .00265 |
| A1 – trace gating (fixed f_base; local) | 84.855% | −.698pp | 79.000% | 79.253% | 5.758pp | 27.134% | 51,224 | 29,139,553 | 20,034,861 | .99834 | 1.000% | ~100% | .00149 |
| A2 – collision term (I:=1) | 85.397% | −.156pp | 79.590% | 79.842% | 5.389pp | 19.356% | 21,601 | 13,616,073 | 7,121,159 | .99755 | .989% | ~100% | .00223 |
| A3 – LTD (only LTP) | 85.475% | −.078pp | 79.220% | 79.474% | 5.842pp | 23.814% | 16,340 | 7,990,743 | **0** | .99879 | .895% | ~100% | .00118 |
| A4a – threshold (z=a; local) | 85.542% | −.011pp | 79.150% | 79.432% | 5.853pp | 17.204% | 16,340 | 10,729,173 | 4,956,425 | .99778 | .926% | ~100% | .00204 |
| A4b – full Eq.(7) operator (z=W_B s; local) | 81.509% | −4.044pp | 79.050% | 79.326% | 4.958pp | 17.204% | 16,340 | 10,729,173 | 4,956,425 | .99778 | .926% | ~100% | .00204 |
| A5 – consolidation | — | — | — | — | — | — | — | — | — | — | — | — | — |

All variants' Task20 current-class and restricted current-5-way accuracies,
per-task cumulative accuracies, writer statistics and feature metrics are in
`diagnostics/paper_style_ablation_full20.csv`. The separate 120-row writer
table is `diagnostics/paper_style_ablation_mechanisms.csv`.

### Correct/error familiarity F

The final true-class eligibility trace for some old classes underflows to
zero under per-sample lambda=.01 decay. An initial all-seen test calculation
therefore had a division-by-zero warning; its newly generated CSV was removed
and replaced. The valid **local proxy** restricts the comparison to Task20's
current five test classes (500 samples); this is not necessarily the paper's
Table 2 sampling convention.

| Variant | E[F_correct] | E[F_error] | Gap |
|---|---:|---:|---:|
| Full | .24693 | .16267 | .08426 |
| A1 fixed gate | .24740 | .16003 | .08737 |
| A2 no collision | .24654 | .16050 | .08604 |
| A3 only LTP | .24611 | .16309 | .08302 |

The frozen trace representation is independent of writer ablation; gaps shift
only because predictions partition the samples differently. A1's gap is
slightly larger, not smaller; it does **not** reproduce Table 2's direction
under this local metric. All six values are in
`diagnostics/paper_style_ablation_familiarity.csv`.

## Partial-cue old-class accuracy at Task20

After encoding, exactly `round(96 × drop_ratio)` active bits were dropped
from each test cue, using five fixed seeds. No writer, trace, W_B or ridge
updates used corrupted cues. Values are mean ± sample standard deviation
across seeds, in percent. Drop0 was checked against each variant's Task20
old-class accuracy. A4a/A4b are separate local implementations, neither
claimed paper-equivalent Table 5 “– CA1 competition.”

| Variant | drop0 | drop0.3 | drop0.6 | drop0.8 |
|---|---:|---:|---:|---:|
| Full | 79.463 ± 0.000 | 76.691 ± .149 | 69.844 ± .128 | 56.434 ± .352 |
| A4a no threshold | 79.432 ± 0.000 | 76.674 ± .140 | 69.842 ± .131 | 56.457 ± .378 |
| A4b no CA1 operator | 79.326 ± 0.000 | 76.560 ± .145 | 69.741 ± .097 | 56.234 ± .374 |
| A3 only LTP | 79.474 ± 0.000 | 76.577 ± .208 | 69.718 ± .264 | 56.017 ± .226 |

The curves are close. From drop0 to drop0.8, Full loses 23.029pp, A4a
22.975pp, A4b 23.092pp, and A3 23.457pp. Full is **not clearly more
robust** than A4a/A4b here. The paper's Table 5 has much wider gaps, but
absolute comparison is invalid because the parameter/readout and A4
operationalizations differ. Individual seed rows are in
`diagnostics/partial_cue_robustness.csv`.

## Interpretation and paper-direction labels

1. **Trace gating:** Full has 16,340 rather than A1's 51,224 plateaus/task,
   and A1 lowers Abar by .698pp. Plateau and Abar directions match Table 2
   (`DIRECTION_MATCH`); local Fgt and correct/error F-gap do not (`DIRECTION_MISMATCH`
   under local definitions). The paper did not specify A1's replacement
   plateau law (`PAPER_OPERATION_UNSPECIFIED`).
2. **Collision term:** Removing I raises plateau rate and local prototype
   overlap (.99710→.99755) and O→N (.926→.989%), directions matching Table 3
   but effects are tiny and these metrics are locally defined. Abar decreases
   .156pp (`DIRECTION_MATCH`); Task20 final/old *improve* .410/.379pp
   (`DIRECTION_MISMATCH` for final/old). No large conclusion from one seed.
3. **LTD:** A3 had zero 1→0 transitions and higher W_B occupancy, prototype
   overlap (.99879), and lower cosine margin (.00118), consistent with less
   repulsion (`DIRECTION_MATCH` on geometry). Yet Abar changes only −.078pp,
   final +.040pp, and local O→N slightly *falls*; the paper's pronounced
   performance/old-to-new penalty is absent (`DIRECTION_MISMATCH` or
   `INCONCLUSIVE` at this scale). “Largest repulsion effect” applies to the
   local overlap/margin trend, **not** to accuracy.
4. **CA1:** Removing only threshold is nearly neutral (−.011pp Abar,
   −.030pp final); q=4 Task20 CA1 density is already effectively 100%.
   Removing normalization as well yields −4.044pp Abar, mostly from large
   early-task losses (Task1 70.8%, Task2 56.0% vs Full 96.4/92.7%). A4b's
   mean feature norm at Task20 is ~2077.6 vs A4a's ~21.6, and Gram trace
   per sample is ~32.67 million vs ~3545, close to the exact k²=9216
   scaling. Holding gamma=.1 changes effective regularization enormously;
   therefore this does **not** isolate a biological competition benefit.
   The Table 4 operation remains `PAPER_OPERATION_UNSPECIFIED`.
5. **Consolidation:** Table 4 does not say what classifier replaces ridge or
   how historical/current features are handled. We did **not** invent a
   paper “– consolidation” variant:
   `PAPER_ABLATION_NOT_REPRODUCIBLE_FROM_PUBLIC_DESCRIPTION`.
6. **Partial cues:** Full is not distinctly slower to degrade under this
   frozen local pipeline (`DIRECTION_MISMATCH` for the strong Table 5 effect;
   the exact A4 paper operation is unspecified). The normalization by the
   *remaining* cue active bits follows Eq. (7); no parameter was retuned.

The experiment preserves paired initial data order, sparse encoder, zero W_B
and RNG seeds; branch-dependent events diverge later, so this is **not
event-wise paired randomness**. All new logic is isolated under
`experiments/`; formal source/config/checkpoints were not changed.
