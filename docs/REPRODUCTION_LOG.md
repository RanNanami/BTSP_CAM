# Reproduction log

## 2026-09-23 - CA1 collapse diagnosis round 1

- **Scope:** Re-ran Tasks 1-5 without parameter changes; recorded pre-threshold
  overlap distributions, an unthresholded diagnostic ridge, current/old class
  geometry, five binary-memory snapshots, representation drift, and memory
  write distributions.
- **Threshold localization:** Unthresholded overlap ridge reached 0.924 at Task
  1 and 0.944 Task-5 current accuracy, versus 0.202 and 0.000 after thresholding.
  Current-test P99 overlap stayed below 0.30 while `tau=0.5`; 99.4%-100% of
  current-test thresholded rows were zero.
- **Continual drift:** Unthresholded Task-5 cumulative accuracy was 0.330 but old
  accuracy was only 0.1765. Fixed Task-1 overlap cosine fell from 1.0 under
  `WB_task1` to 0.834 under `WB_task5`.
- **Memory writes:** Occupancy increased from 0.00916 to 0.03592. Slot plateau
  P99 was 9 and the maximum was 13 at Task 5, with no class-level write
  starvation. Column occupancy was strongly skewed and remains a secondary
  diagnostic target.
- **Conclusion:** The catastrophic first-task loss occurs at `a -> z`.
  Evolving-memory feature drift independently damages old-class readout. The
  BTSP memory update retains strong current-task signal and is not the first
  module to modify.
- **Verification:** Five requested CSVs passed row-count checks; all five saved
  `W_B` matrices are `[4096,8192]` and binary; thresholded accuracies exactly
  reproduce the prior diagnostic. 34 tests plus 3 parameterized subtests passed.

## 2026-09-23 - five-task representation/readout localization

- **Scope:** Ran frozen-ViT ridge, Gaussian+k-WTA ridge, and full BTSP-CAM on
  Tasks 1-5 using the exact formal feature caches and protocol. No parameter
  was changed.
- **Accuracy:** Task-5 cumulative accuracy was 0.9464, 0.6516, and 0.0396.
  The cumulative drops were 0.2948 for ViT-to-sparse and 0.6120 for
  sparse-to-BTSP/CA1.
- **Current-task ceiling:** Task-5 restricted 5-way accuracy was 0.990, 0.886,
  and 0.202. Full BTSP-CAM therefore fails to separate even the current five
  classes, independently of old-class logit interference.
- **Geometry:** Task-5 same-minus-different cosine margin was 0.164194 for
  `h`, 0.049367 for `s`, and 0.000456 for `z`. Cumulative-test CA1 density was
  5.76e-6.
- **Numerics:** Task-5 regularized one-norm condition estimates were 7.78e7,
  6.69e6, and 1.46e2. CA1 is not uniquely ill-conditioned; its class geometry
  and test activation collapse first.
- **Learning dynamics:** Mean `f_eff` stayed near 0.0015. The first sample of
  every one of 25 new classes had zero `f_eff`, but positive `f_eff` began at
  sample 2 for every class. The printed update-before-gating order does not
  cause a long zero-plasticity interval.
- **Artifacts:** Seven CSVs under `diagnostics/` plus
  `docs/PERFORMANCE_DIAGNOSIS.md`. The diagnostic runner also corrected its
  condition estimator to capture the original matrix norm before in-place
  Cholesky factorization; a regression test covers this diagnostics-only issue.

## 2026-09-23 - complete 20-task formal-size execution

- **Change:** Resumed from the real Task 2 checkpoint and completed Tasks 3-20
  without changing model or protocol parameters.
- **Coverage:** 50,000 task-local training images across 20 stages; final
  cumulative evaluation on all 10,000 test images and 100 classes; no task ID
  and zero old-train replay samples at every stage.
- **Numerics:** all 20 ridge solves returned `ok`; all memories remained binary;
  plateau rates and CA1 activity remained nonzero; all 20 checkpoint reloads
  reproduced their complete cumulative-test predictions exactly. Final ridge
  condition estimate was 94,202.92.
- **Runtime:** summed per-task runtime 5,038.9 s (84.0 min); maximum recorded
  working set 3.31 GB. Final checkpoint size 231,286,365 bytes.
- **Metrics:** final cumulative accuracy 0.0408; average incremental accuracy
  0.0516685; final current-task accuracy 0.148. No parameter was tuned in
  response to these results.
- **Independent final check:** a fresh process loaded `checkpoint_task20.npz`
  and reproduced accuracy 0.0408 on all 10,000 test images.
- **Tests:** 32 tests plus 3 parameterized subtests passed after the run.

## 2026-09-22 - runnable formal-size BTSP-CAM core

- **Change:** Added a validated `BTSPCAMConfig`, image-loader API, complete NPZ
  checkpoints, exact load/resume, deterministic evaluation cache, explicitly
  labeled fixed-augmentation training cache, formal runner, and numerical
  safety checks. CA1 now exploits exact k-WTA sparsity without changing Eq. (7).
- **Formal dimensions:** `R=[8192,768]`, `b=[8192]`, `W_B=[4096,8192]`,
  eligibility `[100,8192]`, ridge `A=[4096,4096]`; Task 2 `B/W_out=[4096,10]`.
- **Task 1:** 2,500 train / 500 test; ridge `ok`; condition 15.02; CA1 density
  0.000837; runtime 234.9 s; peak working set 2.37 GB; reload predictions exact.
- **Task 2 resume:** loaded Task 1 in a new process, then trained 2,500 current
  samples and tested 1,000 cumulative samples; ridge `ok`; condition 12.96;
  CA1 density 0.000753; runtime 229.2 s; peak 3.33 GB; reload predictions exact.
- **Accuracy disclosure:** cumulative accuracy was 0.202 then 0.100; Task 2
  current-class accuracy was 0.0. No parameter was changed in response.
- **Final verification:** 32 tests plus 3 parameterized subtests passed.

## 2026-09-22 - round-2 semantic and protocol audit

- **Change:** Audited Eq. (2)-(9) and Algorithm 1; fixed PILOT class-order RNG,
  separated plateau/synapse streams, added dynamic ridge class expansion,
  rejected invalid base probabilities, and made Top-k ties explicit.
- **Protocol basis:** Pinned official PILOT commit `7a6e904c...`; implemented
  `cifar224` train/test transforms, 20 x 5 B0-Inc5 tasks, exact remapping, and
  the current timm alias for PILOT's ViT-B/16 ImageNet-21K checkpoint.
- **Test result:** 25/25 passed before adding final dataset-split coverage; the
  final suite is rerun after the real-data smoke.
- **Remaining uncertainty:** `k`, `f_q_base`, `p`, `beta`, state-dependent
  `lambda`, `tau`, and `gamma` are not paper-resolved. Layer A remains
  `NOT PAPER-COMPARABLE`; Layer B is blocked.

## 2026-09-22 - real CIFAR-100 protocol smoke

- **Change:** Downloaded and MD5-verified the official CIFAR-100 Python archive,
  extracted 50,000 train and 10,000 test samples, and ran the first two full
  five-class stages through the pretrained frozen ViT, sparse encoder, BTSP
  memory, CA1, and streaming ridge.
- **Protocol basis:** Task 1 classes `68 56 78 8 23`; Task 2 classes
  `84 90 65 74 76`; train counts 2,500 each; cumulative test counts 500 and
  1,000; no old-train replay or task-ID inference.
- **Test result:** passed. Feature standard deviations were 1.861/1.841,
  Top-k was exactly 3, mean plateau probabilities were 0.001441/0.001278,
  final binary-memory activity was 0.01748, CA1 activity was nonzero, and both
  ridge solves returned `ok` (final condition number 2833.18). Diagnostic
  cumulative accuracies were 0.370 and 0.278 and are not paper-comparable.
- **Observed/fixed:** the first attempt exposed a diagnostics-only tuple-order
  bug when recording input statistics; it failed before training and was fixed
  explicitly. The successful run records float32 `[32,3,224,224]` input batches
  spanning `[0,1]` with no normalization.
- **Final verification:** 27/27 tests passed; the standalone artifact-invariant
  audit also passed every requested real-smoke check.

## 2026-09-22 - artifact intake and audit

- **Change:** Extracted the supplied zip to `upstream/simpleBTSP` without modifying it; inspected all source files and all 18 PDF pages, including Algorithm 1 and supplementary derivations.
- **Paper basis:** Sections 3-6; Algorithm 1; Eqs. (2)-(9); Supplementary A-F.
- **Source-code basis:** supplied `main_BTSP_feedforward_model.py`, `main_BTSP_feedback_model.py`, `main_random_projection.py`, README.
- **Test result:** source behavior recorded; official exact update isolated for unit testing.
- **Remaining uncertainty:** numeric CAM hyperparameters, backbone identity/preprocessing, B0 semantics.

## 2026-09-22 - minimum audited implementation

- **Change:** Added paper-ordered sparse encoding, eligibility, adaptive plateau, binary memory, CA1, streaming ridge, and a no-replay CIL coordinator.
- **Paper basis:** Eqs. (3)-(8), Algorithm 1.
- **Source-code basis:** active-input-only update and binary state invariants checked against official feedforward lines 123-128.
- **Test result:** pending at time of entry; see the next entry for executed results.
- **Remaining uncertainty:** Eq. (4) state-dependent decay is unspecified; Eq. (9) orientation is inconsistent; first-sample gating degeneracy follows printed Algorithm 1 order.

## 2026-09-22 - smoke protocol

- **Change:** Added a synthetic six-class, three-task smoke runner and full result artifact schema.
- **Paper basis:** exemplar-free current-task-only training and cumulative testing from Sec. 3.1; sufficient statistics from Eq. (8).
- **Source-code basis:** none beyond the binary update audit.
- **Test result:** first run completed but CA1 activity density was 1.0 at the smoke-only `tau=0.1`; `tau=0.3` still yielded 0.958 density. Changed only this LOCAL_CHOICE to `tau=0.5` to exercise a non-degenerate CA1 regime, without selecting against accuracy.
- **Remaining uncertainty:** smoke hyperparameters are explicitly LOCAL_CHOICE and accuracy is not comparable to the paper.

## 2026-09-22 - verification result

- **Change:** Executed all unit tests and the final synthetic smoke configuration.
- **Paper basis:** invariants from Eqs. (3)-(8) and Algorithm 1.
- **Source-code basis:** official truth table from feedforward lines 123-126.
- **Test result:** 15/15 unit tests passed. Three-task smoke completed with sparse-code density 0.0625, active-weight fraction 0.2627, plateau rate 0.3167, CA1 activity density 0.3577, ridge matrix `128x128`, final cumulative accuracy 0.5694, and average incremental accuracy 0.7905. Accuracy is diagnostic only.
- **Remaining uncertainty:** unchanged formal-run blockers listed in `PARAMETER_AUDIT.md` and `IMPLEMENTATION_PLAN.md`.
