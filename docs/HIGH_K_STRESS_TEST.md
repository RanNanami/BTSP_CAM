# High-k stress test (exploratory repaired pipeline)

## Scope and protocol

This is an exploratory stress test of the experimental coordinate-aligned ridge
repair, not a change to the formal BTSP-CAM reproduction and not an assertion
about the paper's hidden configuration.  All runs used CIFAR-100 B0-Inc5,
the fixed class order/RF seed/writer seed `1993`, frozen cached ViT features,
`m=8192`, `slots=4096`, `p=0.5`, `beta=1`, `lambda=0.01`, `gamma=0.1`, and
unthresholded CA1 overlap (`tau=0`).  The feature/readout update was the
experimental coordinate-aligned `C/D` sufficient-statistics construction.

The validation-only stream used a fixed stratified split of Task1--5 training
data (400 development-train and 100 validation examples per class; split seed
`20260928`).  Validation examples did not write memory, eligibility, `C/D`, or
ridge statistics.  The fixed grid was `1024, 1536, 2048, 3072, 4096`; the
predeclared expansion criterion was met at 4096 (the equal-write condition
improved by 0.305 percentage points over 3072 while the fixed-write sparse
margin decline was 19.4%, below the 20% guard), so `5120, 6144, 7168` were
also run.  No test accuracy was used to select a value of `k`.

Two write-rate conditions were compared:

- **FIXED_FBASE:** `f_base=0.005` for every `k`.
- **EQUAL_WRITE_APPROX:** `f_base=0.005 * 1024 / k`, specified before the run;
  it equalizes write opportunity only approximately, not realized flips.

The pre-test selection lock chose `k=6144` under FIXED_FBASE: it had the
largest Task1--5 validation mean (85.316%).  The full20 set was prelocked as
`{1024, 6144, 7168}`.  The 1024 full20 row is an archived exact control from
the already-verified identical k=1024 configuration; its independently rerun
Task1--5 validation trajectory was reproduced exactly in both conditions.

## Task1--5 train-side validation

| k | fixed mean cumulative | equal-write mean cumulative | fixed Task5 cumulative | equal-write Task5 cumulative |
|---:|---:|---:|---:|---:|
| 1024 | 83.860% | 83.860% | 79.200% | 79.200% |
| 1536 | 84.094% | 84.175% | 79.920% | 79.640% |
| 2048 | 84.188% | 84.273% | 79.440% | 79.880% |
| 3072 | 85.072% | 85.307% | 80.360% | 80.200% |
| 4096 | 84.886% | **85.612%** | 80.080% | **80.760%** |
| 5120 | 85.194% | 85.117% | 80.320% | 80.000% |
| 6144 | **85.316%** | 84.902% | 80.280% | 79.960% |
| 7168 | 84.051% | 82.985% | 79.120% | 78.160% |

The validation-only maximum is therefore not a monotone trend.  The
predeclared fixed-write selection says `PLATEAU_OR_NOISY`; the equal-write
series brackets a peak at 4096 (`PEAK_BRACKETED`).  Neither result is an
estimate of an author-selected or universally optimal `k`.

## Sparse-code geometry

At Task5, larger `k` made the exact top-k codes much less discriminative by
cosine margin: 0.07519 at k=1024, 0.04874 at 4096, 0.02338 at 6144, and
0.01069 at 7168.  The average different-class overlap similarly grew from
254.9 to 2441.4, 4872.9, and 6397.7 active-bit overlaps respectively.  No
exact duplicate codes were observed in any Task5 validation condition.

This is the expected capacity pressure of denser fixed-cardinality codes; it
does not by itself establish why downstream accuracy changes.  The detailed
Task1--5 sparse activation, support-entropy, RF-bit, overlap, Hamming, and
CA1-overlap measurements are in `diagnostics/high_k_stress_geometry.csv`.

## Writer and memory pressure

For fixed `f_base`, Task5 gross flips/sample rose from 3,646 (k=1024) to
9,204 (4096), then declined to 7,547 (6144) and 4,714 (7168), because plateau
events/sample themselves fell from 7.12 to 4.49, 2.46, and 1.32.  Memory
density rose from 33.84% at k=1024 to about 48.8% at k=6144 and was 46.95% at
k=7168.  Column occupancy was comparatively homogeneous: at fixed k=6144 the
mean was 48.82%, p99 51.34%, max 52.49%, and the most occupied 1% of columns
held 1.06% of active synapses.

Equal-write scaling changed the memory regime materially.  At k=6144 it used
`f_base=0.000833`, generated 1,215 Task5 gross flips/sample rather than
7,547, and left density at 25.95% rather than 48.82%.  Thus this is a useful
diagnostic comparison, not a controlled equality of all writing dynamics.

The `a`-feature cosine margins were already very small at Task5 and shrink
with k (fixed: 0.000196 at 1024, 0.000016 at 4096, 0.000007 at 6144, and
0.000004 at 7168).  These values are evidence that overlap features become
more common-mode under memory pressure; they are descriptive statistics, not
a causal accuracy decomposition.

## Full20 results

| condition | k | average incremental | Task5 | Task10 | Task15 | Task20 | Task20 old | Task20 current | Task20 5-way | Task20 WB density |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed | 1024 | 89.552% | 93.840% | 89.140% | 86.387% | 84.340% | 84.516% | 81.000% | 94.000% | 46.087% |
| fixed | 6144 | **90.243%** | 93.960% | **90.060%** | 87.067% | **84.580%** | **84.821%** | 80.000% | **95.600%** | 49.999% |
| fixed | 7168 | 89.745% | 93.800% | 89.160% | 86.747% | 83.750% | 83.958% | 79.800% | 94.600% | 50.004% |
| equal-write | 1024 | 89.552% | 93.840% | 89.140% | 86.387% | 84.340% | 84.516% | 81.000% | 94.000% | 46.087% |
| equal-write | 6144 | **90.220%** | 93.960% | 89.880% | **87.227%** | **84.600%** | **84.768%** | **81.400%** | 95.400% | 48.009% |
| equal-write | 7168 | 89.561% | 92.960% | 89.040% | 86.347% | 83.800% | 84.000% | 80.000% | 94.200% | 43.122% |

The full20 result has the same broad shape as validation: k=6144 improves
over the 1024 control by 0.691 pp (fixed) or 0.668 pp (equal-write) in average
incremental accuracy, and by 0.240 pp or 0.260 pp at Task20.  k=7168 reverses
that gain.  Because the full20 values are held-out outcomes after the
predeclared validation lock, they are reported rather than used for another
selection round.

The k=6144 full20 advantage exists under both write-rate conditions, while
the two conditions finish extremely close in cumulative accuracy (84.58% vs
84.60%) despite radically different Task20 gross flips/sample (7,405 vs
1,220).  This limited evidence says neither raw write amount nor the sparse
code alone is sufficient to explain the curve; it does not prove a causal
mechanism.

## Interpretation and limits

1. **Do not replace the frozen k=1024 baseline.**  It remains the protected
   repaired baseline.  High-k values are exploratory stress-test conditions.
2. **No evidence supports unlimited `k` scaling.**  Both validation and
   full20 show deterioration by k=7168, alongside collapsing sparse and
   overlap-feature cosine margins.
3. **k=6144 is a pre-test-selected local stress-test point, not a paper
   parameter.**  It gives the highest observed full20 mean among the
   predeclared full20 runs, but only a small final-accuracy improvement over
   k=1024.
4. **The repaired pipeline is still experimental.**  Coordinate-aligned ridge
   differs from the paper's online old-coordinate sufficient-statistics path.
   Results cannot establish the authors' implementation or a causal account of
   their reported performance.
5. **Computational limits matter.**  The high-k run used the exact dense
   float64 `C` statistic for correctness (about 512 MiB for 8192x8192), cached
   frozen features, and a single fixed seed pair.  It is not a broad
   seed-robustness study.

## Artifacts

- `diagnostics/high_k_stress_validation.csv`: Task1--5 train-side validation
  classification metrics for every grid point and condition.
- `diagnostics/high_k_stress_geometry.csv`: sparse-code and current-memory
  overlap geometry, including duplicate checks and RF-bit support measures.
- `diagnostics/high_k_stress_writer.csv`: writing/flip/occupancy summaries.
- `diagnostics/high_k_stress_full20.csv`: prelocked full20 outcomes.
- `diagnostics/high_k_stress_selection_lock.json`: pre-test validation choice
  and checksum of the validation result table.
