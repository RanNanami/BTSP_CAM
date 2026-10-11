# High-k mechanism decomposition

## Motivation and isolation

This exploratory diagnostic asks how much of the k=96 to k=6144 gain in the
experimental coordinate-aligned, unthresholded pipeline is already present
in the sparse code, how much appears after a generic 4096-dimensional binary
memory transform, and whether the learned `W_B` connections add measurable
value. Here **k=96 is the sparse working baseline**. It is not a published
BTSP-CAM parameter or an author configuration. The formal paper-description
reconstruction and the frozen k=1024 repaired baseline were not edited.

All contrasts use CIFAR-100 B0-Inc5, class order seed 1993, the same frozen
ViT cache, Gaussian `R,b` from RF seed 1993, exact top-k, the original sample
order, 8192 code dimensions, 4096 memory rows, `gamma=0.1`, cumulative
seen-class classification, and no task ID at inference. Learned memories use
the original writer with seed 1993 and `f_base=0.005`. They were rebuilt into
an isolated diagnostic cache; every task's density and gross flip count were
checked against the archived trajectory before use.

The readout conditions are distinct experiments. `s -> ridge` bypasses
memory. All memory controls use the same current-task sparse statistics
`C=Σssᵀ`, `D=Σsyᵀ` and compute the exact **tau=0** aligned readout for the
matrix under evaluation. The tau sweep separately uses a replay oracle and
refits on all seen historical training codes under the current `W_B`.

## Sparse code to ridge: full20

| k | activation density | average incremental | Task20 final | Task20 old | Task20 current | Task20 restricted 5-way | sparse cosine margin | Fisher-like ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ViT h reference | — | 90.685% | 85.000% | 85.253% | 80.200% | 95.400% | — | 0.2322 |
| 96 | 1.172% | 71.058% | 73.290% | 73.621% | 67.000% | 86.800% | 0.08465 | 0.06849 |
| 1024 | 12.500% | 82.892% | 83.820% | 83.979% | 80.800% | 93.200% | 0.12579 | 0.12234 |
| 4096 | 50.000% | 84.918% | 84.640% | 84.821% | 81.200% | 95.200% | 0.08303 | 0.14996 |
| 6144 | 75.000% | 84.254% | 84.260% | 84.505% | 79.600% | 94.800% | 0.03941 | 0.13969 |
| 7168 | 87.500% | 82.829% | 83.680% | 83.789% | 81.600% | 92.800% | 0.01782 | 0.12183 |

The k=96 sparse-only path reproduces the archived 20-task sparse path with
zero accuracy difference at every task. From k=96 to k=6144, sparse-only
Task20 increases by **10.97 percentage points**, while average incremental
accuracy increases by **13.20 points**. Thus much of the high-k representational
gain exists before any BTSP memory write. The sparse-only curve itself peaks
within this grid around k=4096 and deteriorates by 7168; increasing density
does not yield a monotone improvement. Sparse cosine margin and the exact
train-side Fisher-like ratio tell a similar broad story, but neither substitutes
for held-out linear readout accuracy.

The Fisher-like ratio is `between-class centroid scatter / within-class
covariance trace`, computed from training sufficient statistics in the current
feature space. It is descriptive and independent of a fitted classifier's
test predictions. The per-task Hamming distances, top-1%-bit shares and
effective RF supports are in `diagnostics/mechanism_sparse_ridge.csv`.

## Learned memory versus matched controls: design

At each task, each control is refit in its **own** current feature space using
the same historical sparse-code sufficient statistics. The controls do not
reuse learned-memory ridge weights. The learned matrix is compared with five
independent Bernoulli matrices drawn at its actual global density and five
independent row permutations of that learned matrix. The row permutations
preserve every row's degree and the total edge count exactly; they disrupt
the input-bit association and do not preserve column degrees. The Bernoulli
matrices match density as an expectation, so their realized densities are
recorded separately. Seeds were fixed before reading control accuracies:
`2026101001` through `2026101005`, with per-task RNG seed
`seed + 100000*k + task`.

For every control, all seen-class test examples determine accuracy. The
current task's first 20 fixed test examples per class form a common
100-example probe for CA1 cosine geometry, keeping the geometry computation
tractable at 75% code density. The full seen-class training stream determines
the feature mean, standard deviation, and Fisher-like ratio exactly from
`A(W)` and `B(W)`. Geometry and accuracy therefore have different evaluation
sample counts and should not be treated as two estimators of the same quantity.

The column-shuffled control is optional in the request and was not included
in the locked five-seed main contrast. The row-shuffled control already
tests whether per-slot learned input associations matter while holding each
slot's occupancy fixed; its failure to preserve column degrees is an explicit
limitation.

The two completed intermediate checkpoints show the changing scale of the
learned-versus-control difference (percentage points):

| k | Task5 learned − Bernoulli mean | Task5 learned − row-shuffled mean | Task10 learned − Bernoulli mean | Task10 learned − row-shuffled mean |
|---:|---:|---:|---:|---:|
| 96 | +2.168 | +2.224 | +1.284 | +1.120 |
| 1024 | +0.960 | +0.848 | −0.012 | +0.100 |
| 6144 | +0.056 | −0.208 | +0.272 | +0.176 |

These are checkpoint descriptions; the full20 mean and final task, not a
single intermediate checkpoint, determine the verdict below.

All 20 tasks are now complete. Accuracy is in percent; control values are
five-seed means, with the final-task seed standard deviation in parentheses.
The learned-minus-control columns compare the learned single trajectory with
the five-seed control mean, not with a retrained writer.

| k | condition | average incremental | Task20 final (seed SD) | learned minus control: avg | learned minus control: final |
|---:|---|---:|---:|---:|---:|
| 96 | learned `W_B` | 85.542 | 79.150 | — | — |
| 96 | Bernoulli matched | 84.127 | 78.740 (0.176) | +1.415 | +0.410 |
| 96 | row-shuffled | 84.117 | 78.650 (0.174) | +1.425 | +0.500 |
| 1024 | learned `W_B` | 89.552 | 84.340 | — | — |
| 1024 | Bernoulli matched | 89.396 | 84.060 (0.174) | +0.155 | +0.280 |
| 1024 | row-shuffled | 89.397 | 84.088 (0.210) | +0.155 | +0.252 |
| 6144 | learned `W_B` | 90.243 | 84.580 | — | — |
| 6144 | Bernoulli matched | 90.103 | 84.374 (0.077) | +0.140 | +0.206 |
| 6144 | row-shuffled | 90.191 | 84.372 (0.124) | +0.052 | +0.208 |

At k=96, learned connections provide a clear advantage across the full
trajectory over both controls. At k=6144, the advantage is small in absolute
terms: +0.206 pp against matched Bernoulli and +0.208 pp against row-shuffling
on Task20. These differences are not literally zero, and the five seeds do
not quantify uncertainty over the data order or writer RNG. Nevertheless,
most of the high-k learned-memory accuracy is reproduced without learned
synaptic associations under this readout. At Task20 the learned matrix's
occupancy is 17.204% at k=96, 46.087% at k=1024, and 49.999% at k=6144.

## Geometry and linear separability

At Task5, the learned-memory current-task probe has the following cosine
margin and training-side Fisher-like ratio:

| k | learned `a` mean | learned `a` std | current-task cosine margin | Fisher-like ratio | cumulative accuracy |
|---:|---:|---:|---:|---:|---:|
| 96 | 0.28066 | 0.06810 | 0.009379 | 0.16247 | 89.36% |
| 1024 | 0.46391 | 0.02324 | 0.000335 | 0.17088 | 93.84% |
| 6144 | 0.49687 | 0.01333 | 0.00000827 | 0.14901 | 93.96% |

The near-zero high-k cosine margin is dominated by a shared positive
overlap component. The nonzero Fisher-like ratio and held-out ridge accuracy
show that small, consistent differences across many feature dimensions can
still support a linear classifier. This is compatible with the proposed
low-amplitude multidimensional signal; it does not prove which synapses
produced it or that every alternative classifier would recover it.

The Task20 learned-memory cosine margin falls from 0.001812 (k=96) to
0.0001185 (k=1024) and 0.00000588 (k=6144), while its exact train-side
Fisher-like ratio is 0.1075, 0.1248, and 0.1393 respectively. The Task20
sparse-code cosine margin is 0.08465, 0.12579, and 0.03941, with Fisher-like
ratios 0.06849, 0.12234, and 0.13969. Thus the almost vanishing raw cosine
margin is not by itself evidence that linear class information has vanished.

## Interpreting the high-k difference

For any fixed evaluation metric, let `S_k` be sparse-only accuracy, `R_k`
the five-seed mean of the Bernoulli memory control, and `L_k` learned-memory
accuracy. The observed k=96→6144 difference can be written exactly as a
telescoping **contrast**:

`L_6144 - L_96 = (S_6144 - S_96) + [(R_6144-S_6144)-(R_96-S_96)] + [(L_6144-R_6144)-(L_96-R_96)]`.

The three terms describe sparse representation, generic memory transform,
and learned-versus-random structure under this specific fixed ridge readout.
They are not intervention-identifiable causal shares. The same regularized
ridge solve is used in all conditions, so there is no separate numerical
"readout contribution" estimated by this experiment; the scale of each
feature map can change how fixed `gamma=0.1` acts. The experimental
coordinate-alignment supplies a common current feature coordinate system
for the matrix controls, but it is not an author algorithm.

For Task20, this contrast evaluates to:

| component of k=96→6144 learned Task20 difference | percentage points |
|---|---:|
| Sparse-only representation: `S_6144-S_96` | +10.970 |
| Change in generic matched-memory uplift: `(R-S)_6144-(R-S)_96` | −5.336 |
| Change in learned-versus-random advantage: `(L-R)_6144-(L-R)_96` | −0.204 |
| **Observed learned-memory `Delta_high_k`** | **+5.430** |

For average incremental accuracy, the analogous terms are +13.196, −7.219,
and −1.275 pp, summing to **+4.701 pp**. In particular, the dense sparse code
is a much better direct ridge feature than the k=96 sparse code, while a
generic 4096-D binary memory transform helps the low-k code much more than
the already informative high-k code. The learned-specific advantage does
*not* grow with k in this diagnostic. These are arithmetic contrasts across
different feature maps, not causal percentages or an allocation of unique
information. A separate readout share is not identified: every condition
uses ridge with the same nominal gamma, but its effective regularization
depends on feature scale.

## Tau replay-oracle experiment

For every k, task, and tau, all seen historical sparse training codes were
replayed through the *current* learned memory, the nonlinear ReLU feature was
rebuilt, and ridge was refit in that single coordinate system. This is a
diagnostic replay oracle. It stores/reuses historical codes and is therefore
neither the exemplar-free formal algorithm nor the linear C/D repair.

| k | tau | average incremental | Task20 final | Task20 positive density | delta final vs tau=0 |
|---:|---:|---:|---:|---:|---:|
| 96 | 0.00 | 85.542 | 79.150 | 100.000% | — |
| 96 | 0.10 | 85.016 | 79.170 | 99.997% | +0.020 |
| 96 | 0.20 | 81.455 | 79.190 | 98.673% | +0.040 |
| 96 | 0.30 | 69.146 | 75.710 | 73.055% | −3.440 |
| 96 | 0.50 | 8.420 | 15.430 | 0.308% | −63.720 |
| 1024 | 0.00 | 89.552 | 84.340 | 100.000% | — |
| 1024 | 0.40 | 87.637 | 84.340 | 100.000% | 0.000 |
| 1024 | 0.45 | 82.019 | 84.220 | 99.106% | −0.120 |
| 1024 | 0.48 | 73.698 | 83.490 | 70.471% | −0.850 |
| 1024 | 0.50 | 62.160 | 80.950 | 23.663% | −3.390 |
| 6144 | 0.00 | 90.243 | 84.580 | 100.000% | — |
| 6144 | 0.40 | 90.208 | 84.580 | 100.000% | 0.000 |
| 6144 | 0.45 | 90.239 | 84.580 | 100.000% | 0.000 |
| 6144 | 0.48 | 90.114 | 84.560 | 99.927% | −0.020 |
| 6144 | 0.50 | 88.332 | 83.840 | 49.730% | −0.740 |

The entire locked grid is retained in the CSV. At k=6144, tau up to 0.45
does not create Task20 competition at all: every CA1 element remains active,
and subtracting a common constant is absorbed by the refit readout closely
enough to leave predictions unchanged. Even tau=0.5 preserves a 2.89 pp
Task20 advantage over k=1024 while halving CA1 density. Thus the tested
high-k advantage is **not** confined to tau≈0. Conversely, the same numerical
tau has a radically different support meaning across k: tau=0.5 collapses
k=96 to 0.308% density and 15.43% final accuracy. This sweep diagnoses the
learned overlap scales; it does not nominate a threshold or recover an author
setting.

## Evidence-based verdict

The evidence most strongly supports
`HIGH_K_GAIN_PRIMARILY_REPRESENTATION_READOUT` (Case A):

1. The k=96→6144 gain is already +10.97 pp in Task20 sparse-to-ridge.
2. At k=6144 a density-matched random memory reaches 84.374% versus 84.580%
   for learned memory, and row-shuffling reaches 84.372%.
3. The learned-specific Task20 advantage is only about 0.21 pp at k=6144,
   versus 0.41–0.50 pp at k=96; its trajectory-average advantage similarly
   shrinks from about 1.42 pp to 0.05–0.14 pp.
4. High-k separability survives despite nearly zero cosine margin, consistent
   with weak but coherent information spread over many dimensions.
5. The k=6144 advantage survives every positive tau in the locked grid, so
   the evidence does not support Case C for this trajectory.

This verdict is specific to the coordinate-aligned, current-feature-space
ridge diagnostic with gamma=0.1. It says that learned BTSP associations add
little *beyond matched random transforms in this experimental pipeline*; it
does not say that the writer has no structure or no biological role.

## What this does and does not imply about BTSP-CAM

- k=96 is a **sparse working baseline**, not a paper baseline or an author
  parameter. k=1024 and k=6144 are experimental regimes, not parameter
  recovery.
- Coordinate-aligned ridge is an experimental repair, not an implementation
  claimed by the paper or its authors.
- The nonlinear tau results use historical-code replay and are not an
  exemplar-free algorithm.
- Random-memory controls are mechanism diagnostics. Their proximity to the
  learned memory cannot establish what an author implementation would do.
- These experiments do show that, under the repaired readout, dense exact
  top-k codes plus a regularized linear readout account for most observed
  high-k performance.

## Limitations and integrity

The CIFAR class order, RF seed, writer seed, ViT cache, and learned writer
trajectory are single-seed (1993). Five control seeds characterize only the
random-memory construction, not variation across datasets, orders, RFs, or
writers. The Bernoulli control matches density in expectation; the row shuffle
preserves row degrees exactly but not column degrees. Cosine geometry uses a
fixed current-task probe, whereas Fisher statistics use full training
sufficient statistics. Fixed gamma is not scale-normalized, so feature-scale
effects and ridge regularization cannot be separated here. No nearest-centroid
classifier or column-degree-preserving shuffle was added.

Before and after the run, SHA-256 was compared for 46 protected artifacts:
the formal YAML, all 15 core `src/btsp_cam/*.py` files, all 20 formal
checkpoints, and the frozen repaired baseline files. All hashes are unchanged.
All new code, caches, CSVs, and this report live on the experimental path; no
formal checkpoint/config was overwritten.
