# f_base and beta sensitivity in the local k=96 pipeline

## Scope and provenance

This is a Task1–5 parameter sensitivity diagnostic. It does not recover paper
hyperparameters or replace the formal reconstruction. The local experimental
setting is CIFAR-100 B0-Inc5, fixed class order/RF/writer seed 1993, frozen
ViT feature cache, exact top-k with m=8192 and k=96, 4096 binary memory rows,
p=0.5, fixed lambda=0.01, gamma=0.1, tau=0, and the experimental
coordinate-aligned ridge. Values k=96, p=0.5, lambda=0.01, gamma=0.1 and
tau=0 are local choices. No claim is made that an author used them.

Each condition starts with the same ordered, cached train/test sparse codes
and the same independent plateau and synapse RNG initial states. The current
task alone updates eligibility, memory and C/D; all seen test classes are
evaluated together without task ID. Once conditions yield different plateau
events, the synapse RNG can consume a different number of draws. Paired
initialization / common starting RNG state does not guarantee event-wise
identical draws after branching. The experiment preserves the existing writer
sampling semantics.

The single-factor grids are f_base = {0.00125, 0.0025, 0.005, 0.01, 0.02}
at beta=1, and beta = {0, 0.25, 0.5, 1, 2, 4} at f_base=0.005. beta=0 is the
I=1 sensitivity endpoint, not a formal published −I ablation. If the locked
diagnostic thresholds trigger a 3×3 follow-up, its f_base values are
{0.00125, 0.005, 0.02} and beta values are {0, 1, 4}. These span the
predeclared low, baseline and high mechanism range. Shared one-factor
conditions are reused exactly instead of retrained.

## Measurement definitions

Per-sample F, N, I, rival cosine and f_eff come directly from the unchanged
`BTSPCAM._learn_code` log, after eligibility update and before the next
sample. `G_FN = 4N(1−N)F`; `f_eff = f_base G_FN I`. Plateau count is the sum
of sampled active slots, and plateau rate is that count divided by
`samples × 4096`. Gross flips count 0→1 and 1→0 changes. The CSV reports
task totals and per-sample/per-plateau rates, together with each task's
post-write W_B density.

For each task, samples are stably sorted by their logged maximum rival
cosine and divided into four equal-count rank quartiles (625 samples each).
Tied collision values retain stream order. The quartile plateau rate uses
the same per-slot denominator. Q4/Q1 is undefined if Q1 has no plateaus.
Task5 sparse cosine geometry uses all 2500 seen test examples; Task5 CA1
overlap cosine geometry uses the fixed first 20 test examples per current
class (100 samples). The CSV labels both populations. Neither geometry
measurement is a substitute for held-out classification accuracy.

## Archived control

The f_base=0.005, beta=1 condition is checked after every task against the
archived k=96 evolving aligned trajectory for cumulative/old/current/5-way
accuracy, plateau count, gross flips and W_B density. The rebuilt W_B is
also compared bit for bit with the isolated archived checkpoint at each task.
All five tasks pass. Task5 cumulative accuracy is 89.36%. This is the
five-task control, separate from the archived full20 average 85.542% and
Task20 final 79.150%.

## One-factor results

The f_base sweep is complete. Task5 outcomes are below; accuracy is percent,
density is the active fraction of all W_B synapses, and counts refer to Task5
alone.

| f_base | cumulative | old | current | plateau count | gross flips | W_B density | CA1 cosine margin |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00125 | 89.88 | 89.20 | 92.60 | 4,213 | 202,092 | 2.572% | 0.03372 |
| 0.0025 | 89.80 | 88.95 | 93.20 | 8,338 | 399,592 | 4.589% | 0.01872 |
| 0.005 | 89.36 | 88.70 | 92.00 | 16,868 | 810,951 | 7.653% | 0.00938 |
| 0.01 | 88.76 | 87.65 | 93.20 | 33,632 | 1,614,581 | 11.806% | 0.00462 |
| 0.02 | 88.60 | 87.65 | 92.40 | 67,065 | 3,219,440 | 16.925% | 0.00253 |

At fixed eligibility and beta, mean f_eff scales exactly with f_base.
Observed plateau counts closely follow the 1:2:4:8:16 f_base grid: the Task5
count is 4,213 at the low end and 67,065 at the high end (15.92×). Gross
flips also increase about 15.93×. W_B density rises by 14.35 percentage
points, with sublinear occupancy growth as already active synapses can flip
back or be rewritten. Task5 cumulative accuracy spans only 1.28 points and
CA1 cosine margin falls by more than an order of magnitude. Thus write volume
and raw cosine geometry are much more sensitive than held-out ridge accuracy
in this local five-task window.

The beta sweep is complete. Collision mean (0.255659), p90 (0.339214),
F/N and G_FN are identical across all six Task5 conditions. This follows
from the shared sparse stream and eligibility dynamics, which do not depend
on the memory writes or beta. Mean I falls monotonically as beta rises.

| beta | cumulative | old | mean I | Task5 plateaus | gross flips | W_B density | Q4/Q1 plateau ratio | CA1 margin |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 88.92 | 88.05 | 1.000 | 21,749 | 1,044,107 | 8.936% | 1.130 | 0.00751 |
| 0.25 | 89.32 | 88.35 | 0.938 | 20,376 | 978,598 | 8.601% | 1.088 | 0.00784 |
| 0.5 | 89.60 | 88.65 | 0.880 | 19,142 | 919,932 | 8.280% | 1.043 | 0.00841 |
| 1 | 89.36 | 88.70 | 0.776 | 16,868 | 810,951 | 7.653% | 0.970 | 0.00938 |
| 2 | 89.32 | 88.60 | 0.604 | 13,104 | 629,157 | 6.483% | 0.845 | 0.01201 |
| 4 | 89.80 | 89.05 | 0.370 | 7,890 | 378,202 | 4.569% | 0.630 | 0.01912 |

At beta=0, I is exactly 1 by construction. At beta=4, mean I is 0.370;
the Q4 mean is 0.262 versus Q1 mean 0.481. The Q4/Q1 plateau ratio falls
from 1.130 to 0.630, so the high-collision quartile receives fewer writes
relative to the low-collision quartile. This is more than a global reduction
in plateau count: the relative allocation changes. The Task5 cumulative
accuracy range is 0.88 pp, and old-class accuracy range is 1.00 pp, despite
2.76× gross-flip and 4.37 pp memory-density ranges. beta=4 has a 0.35 pp
old-class advantage over beta=1 in this single trajectory, which is too
small and insufficiently replicated to establish protection of old classes.

## 3×3 mechanism range

The grid was triggered by the locked mechanism criteria. Relative to the
baseline at Task5, the f_base sweep reached a 9.27 pp W_B density difference
and a 297% gross-flips-per-sample difference; both exceed their 2 pp and 20%
thresholds. The beta sweep changed Q4/Q1 plateau-rate ratio by up to 35.0%
relative to beta=1 and CA1 margin by up to 103.9%; both exceed their 15%
thresholds. The endpoints below were declared in the config before either
sweep was read. Accuracy is reported, not used to choose grid coordinates.

| f_base | beta | Task5 cumulative | Task5 old | W_B density | gross flips | Q4/Q1 plateau ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00125 | 0 | 90.08% | 89.40% | 3.167% | 259,357 | 1.168 |
| 0.00125 | 1 | 89.88% | 89.20% | 2.572% | 202,092 | 0.997 |
| 0.00125 | 4 | 90.24% | 89.65% | 1.361% | 93,534 | 0.594 |
| 0.005 | 0 | 88.92% | 88.05% | 8.936% | 1,044,107 | 1.130 |
| 0.005 | 1 | 89.36% | 88.70% | 7.653% | 810,951 | 0.970 |
| 0.005 | 4 | 89.80% | 89.05% | 4.569% | 378,202 | 0.630 |
| 0.02 | 0 | 88.48% | 87.40% | 18.721% | 4,145,295 | 1.125 |
| 0.02 | 1 | 88.60% | 87.65% | 16.925% | 3,219,440 | 0.969 |
| 0.02 | 4 | 88.76% | 87.95% | 11.920% | 1,528,027 | 0.620 |

Across the nine corners, Task5 memory density spans 17.36 percentage points
and gross flips span 4,051,761 per task (44.3× between the two extremes).
Cumulative accuracy spans 1.76 pp. beta=4 versus beta=0 changes Task5
accuracy by +0.16, +0.88, and +0.28 pp at low, baseline, and high f_base,
respectively. This does not show a stable monotone accuracy interaction,
although beta consistently changes Q4/Q1 write allocation. The largest
absolute suppression of writes by beta occurs at high f_base.

## Interpretation

**f_base.** Plateau events and gross flips are nearly proportional to the
nominal base probability in this range. Memory density rises strongly but
less than proportionally, while Task5 accuracy moves by 1.28 pp across the
five one-factor conditions. The sparse code geometry is identical; Task5
CA1 cosine margin varies more than 13×. This is a clear memory-dynamics
sensitivity with a much smaller classification response.

**beta.** The maximum rival-collision distribution itself is identical at
each task across all beta settings, as are F/N and G_FN. Increasing beta
monotonically reduces I and both overall plateaus and gross flips. It also
reallocates writes: the high-collision Q4/Q1 plateau ratio falls from 1.130
at beta=0 to 0.630 at beta=4. The old-class Task5 accuracy changes by only
1.00 pp across the grid and is 0.35 pp higher at beta=4 than beta=1. A
single-seed, five-task result does not establish that this protects old
representations. Old-class CA1 geometry was not independently measured;
the CA1 cosine probe in these CSVs is current-task only.

**Paper-gap assessment.** The status for f_base as a *major classification
gap explanation in this local Task1–5 configuration* is
`NOT_SUPPORTED_IN_THIS_LOCAL_CONFIG`; for beta it is also
`NOT_SUPPORTED_IN_THIS_LOCAL_CONFIG`. Both parameters strongly alter writes,
but neither produces a multi-point early classification shift comparable to
the changes in memory activity. Whether later continual dynamics change that
assessment is `INCONCLUSIVE` without an authorized full20 follow-up. No
condition is selected based on test accuracy or treated as an author value.

**Component ablation decision.** The gate's I factor has a measurable,
collision-dependent write effect. beta=0 already gives the exact I=1
parameter limit locally, so rerunning that single endpoint with an
"ablation" label adds no information. A next-stage formal component study
would be justified if it extends this question to paired seeds and longer
continual trajectories, especially for I and F/N. This sweep provides no
direct evidence about the necessity of eligibility or LTD and does not
justify drawing conclusions about either component.

## Verification and boundaries

The 25 f_base rows, 30 beta rows and 45 grid rows have unique
`(f_base,beta,task)` keys. Every reused grid row matches its one-factor
source, beta=0 has I=1 exactly, and Task1–5 collision summaries match
exactly across beta. All five baseline accuracy, write and density rows
match the archived source at numerical tolerance 1e−12, and the available
W_B checkpoints match bit for bit. SHA-256 of 40 protected formal/core,
checkpoint, repaired-script and archived files was unchanged. The old CSVs,
formal defaults and checkpoints were not edited. The CSV validation follows
the scientific data-table discipline of explicit units, populations, source
and missing-value semantics; Task1 old accuracy is intentionally undefined.
