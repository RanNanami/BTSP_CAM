# CUB-200 data / test leakage forensic audit

**Verdict: `TEST_LEAKAGE = NO_EVIDENCE`.** Across the audited official split, all 11,788 source-image identities, all 40 feature and 40 sparse cache files, and an instrumented 20-task sparse-cache replay, no test identity entered the writer, eligibility update, or ridge fit. No future-task training identity entered an earlier update. The Task20 test-label permutation changed accuracy but left every logit and prediction identical; evaluation changed none of the hashed learning state. This is evidence about the inspected pipeline and files, not a proof that leakage is mathematically impossible.

The important qualification is **`HISTORICAL_TRAIN_REPLAY = YES` and `STRICT_EXEMPLAR_FREE = NO`**. At every task, `CURRENT_WB_REPLAY_REFIT` reloads all *seen training sparse codes* for a current-memory ridge fit. That is neither test leakage nor the paper's strict exemplar-free readout. The local CUB score must not be presented as a paper-exact replication.

## Scope and evidence

No formal source, frozen or CUB configuration, baseline result, existing checkpoint, feature cache, or sparse cache was edited. The audit reused the existing cached features/codes and rebuilt only an isolated in-memory writer/readout trajectory for runtime canaries; it did not rerun ViT or tune parameters. New audit programs are `experiments/audit/cub200_data_cache_audit.py` and `experiments/audit/cub200_runtime_access_audit.py`. New evidence is in `diagnostics/cub200_raw_split_leakage.json`, `cub200_near_duplicate_review.json`, `cub200_feature_cache_provenance.json`, `cub200_cache_row_provenance.jsonl`, `cub200_runtime_access_audit.json`, and `cub200_task20_prediction_canary.npz` (all under `diagnostics/`).

The pinned PILOT checkout is commit `7a6e904c5bc5cb7a4e1823b3434020be27469b63`. Its CUB loader uses separate `./data/cub/train/` and `./data/cub/test/` ImageFolders (`upstream/LAMDA-PILOT/utils/data.py:260–279`), and `DataManager.get_dataset` selects `_train_data` or `_test_data` by `source` (`utils/data_manager.py:33–43`). The local runner fixes the class order, lock parameters, and replay mode at `experiments/run_btsp_cam_cub200.py:34–75`.

## 1. Official split, paths, bytes, and ImageFolder labels

An independent parse of `images.txt`, `train_test_split.txt`, `image_class_labels.txt`, and `classes.txt` found 11,788 unique image IDs, equal image-ID sets across metadata, 5,994 train and 5,794 test images, and 200 classes. Every one of the 11,788 ImageFolder rows resolves to exactly one metadata image; its split flag and original class index agree. Train/test `class_to_idx` mappings are identical and complete. The incremental remap was checked for every feature-cache row against PILOT's seed-1993 class order.

| Cross-split check | Result |
|---|---:|
| Canonical `<class>/<filename>` path overlap | 0 |
| Resolved absolute-path overlap | 0 |
| SHA-256 content overlap after hashing every train/test image | 0 |
| Windows file identity `(st_dev, st_ino)` overlap | 0 (supported) |

Hardlinks from each split into the original image tree are expected; the check above is **train against test**, not either split against the raw source. The independent dataset digest is `d4b321f10635f985fa79fd0d333941526c530e362ba7d3690e7fce76b9c85af7`, equal to the prior Full-run manifest.

### Visual near-duplicates are a separate finding

The cross-split dHash scan reports the 50 closest pairs in `cub200_raw_split_leakage.json`; this is a lightweight candidate screen, not an exhaustive perceptual-duplicate census. Manual image inspection of the five same-class pairs in that top 50 found two effectively same photographs encoded as different JPEGs (Bronzed Cowbird and Prothonotary Warbler), one very similar altered-framing scene (Green Kingfisher), and two closely related scenes or possible burst frames requiring judgment (Marsh Wren and Golden-winged Warbler). A dHash-distance-1 *different-class* pair was plainly different on visual inspection, illustrating why a small hash distance is not enough. Exact pairs and assessments are recorded in `cub200_near_duplicate_review.json`.

These files are assigned opposite splits by the **official source metadata**, and their bytes differ. They are a benchmark-content caveat, not evidence that the cache or code crossed a split boundary. All five flagged test images were classified correctly at Task20; even if all five correct decisions were wholly attributed to visual memorization, their direct contribution to the 5,794-image final metric would be at most `5/5794 = 0.0863` percentage points (the two same-photo variants alone: `0.0345` pp). This bound applies **only to reviewed candidates**. Other, less dHash-similar variants may exist, so no exhaustive near-duplicate bound is claimed.

## 2. Feature and sparse cache row provenance

Each of 40 feature files has `features[N,768]`, `labels[N]`, and DataLoader-returned `indices[N]`. The audit reconstructed each indexed sample's source path using the *actual* PILOT `DataManager.get_dataset` ordering, then independently checked source split, original image ID, label, incremental remapping, file count, and row order. This is stronger than trusting a `train`/`test` filename or metadata string. Sparse files have `packed[N,1024]` and labels; every row has exactly 96 active bits. Re-encoding **every** cached 768-D feature row with the frozen seed-1993 projection reproduced its corresponding packed sparse row exactly. The feature and sparse metadata agree on split/task/dataset digest/class order/source cache/projection seed.

| Cache | Train rows | Test rows | Train/test image-ID intersection |
|---|---:|---:|---:|
| Feature | 5,994 | 5,794 | 0 |
| Sparse | 5,994 | 5,794 | 0 |

The row ledger `cub200_cache_row_provenance.jsonl` contains `sample_id`, `source_path`, `split`, `task`, `label`, `feature_row_index`, and `sparse_row_index` for each sample. `sample_id` is `<split>:<official image_id>`; the ledger binds it one-to-one to the canonical path, so runtime IDs also have independently verified path provenance. Identical sparse bit patterns, if any, are not treated as sample identity.

## 3. Static data flow and state boundary

| Stage and source | Labels read? | Learning state changed? | Evidence |
|---|---|---|---|
| Current task **train** ImageFolder → ViT feature cache → exact-k encoder | Yes, for supervised writer target and cache checks | Cache creation only if absent; no writer yet | `experiments/run_btsp_cam_cub200.py:186–228`; `src/btsp_cam/feature_cache.py:73–84` |
| Current task **test** ImageFolder → ViT feature cache → exact-k encoder | Yes, cached for eventual metrics | No writer, trace, or ridge operation | `experiments/run_btsp_cam_cub200.py:304–309` |
| Current task **train** sparse rows → `_learn_code` → eligibility / plateau / binary memory | Yes, current training label | **Yes**: trace and `W_B` | `experiments/run_btsp_cam_cub200.py:310–319`; `src/btsp_cam/model.py:140–160` |
| All **seen train** sparse files → current-`W_B` CA1 → `fit_replay` / `ridge.update` / `solve` | Yes, historical training labels | **Yes**: new ridge `A/B/W_out` | `experiments/run_btsp_cam_cub200.py:168–180,320–323`; `src/btsp_cam/ridge.py:32–77` |
| All **seen test** sparse rows → CA1 → `ridge.logits` → argmax / metric | Yes, only for correctness count | **No**: prediction and metric only | `experiments/run_btsp_cam_cub200.py:230–260,324` |

Although the runner loads each current test cache before performing that task's train writes, the test object is only appended to `test_sources` and passed to `evaluate` after the writer and replay refit. The `fit_replay` loader path is hardcoded `train_task{source:02d}.npz` with `source = 1..task`; it never points to a test sparse file. Source/task label-range assertions additionally reject wrong-task training labels. This static reading is backed by the runtime canary below.

## 4. Runtime access canary, historical replay, labels, and state

The audit instrumented the same `_learn_code`, `fit_replay`, and `evaluate` functions used by the Full runner. On the saved sparse codes it reconstructed the 20-task trajectory from fresh `W_B=0`, recorded **ordered** per-task `writer_update_ids`, `eligibility_update_ids`, `ridge_fit_ids`, `prediction_ids`, and `metric_ids`, and checked every recorded identity against the official-path provenance ledger. Every task's cumulative accuracy exactly matched the archived `cub200_full_taskwise.csv` (`<1e-12` absolute difference), an independent trajectory check.

For **all 20 tasks**, writer and eligibility IDs were exactly the current task's train IDs; ridge IDs were exactly all seen train IDs; prediction and metric IDs were exactly all seen test IDs. Test IDs in writer, eligibility, and ridge: **0 each**. Future train IDs in writer, eligibility, and ridge: **0 each**. Task20 ridge used **5,994 unique train IDs**. Its source-task counts were `300,300,300,299,300,300,300,300,298,300,300,300,299,300,300,300,300,299,300,299`, summing to 5,994. Thus the correct classification is `HISTORICAL_TRAIN_REPLAY=YES`, `STRICT_EXEMPLAR_FREE=NO`, `TEST_LEAKAGE=NO_EVIDENCE`.

At Task20, the audit evaluated the same 5,794 test codes twice, once with real labels and once with a deterministic permutation of labels (`seed=20261011`) **without fitting anything between evaluations**. The two logit arrays are bit-identical, predictions exactly equal. Accuracy changed from `4670/5794 = 80.600621%` to `0.465999%`, near the 200-class chance rate of `0.5%`. Labels thus affected the metric but not the predictions in this check.

Before and after the full seen-test evaluation and label-canary evaluation, SHA-256 hashes were unchanged for `W_B`, eligibility, encoder projection `R` and bias `b`, ridge `A/B/W_out`, seen-class state, RNG states, task index, training-order state, memory flip counters, and other model counters/summary fields. No logging-only counter changed either. The full before/after hash map is in `cub200_runtime_access_audit.json`.

## 5. Parameter-selection and paper-number provenance

The source lock `experiments/configs/ablation_baseline_frozen.yaml` was written at **2026-10-10 17:15:14** local time, the CUB transfer config at **20:54:37**, and CUB Full summary at **21:47:45**. The lock's recorded SHA-256 values exactly match the pre-lock CIFAR validation CSVs, both written at 17:14:49. The CUB runner's `load_config` checks every transferred parameter against the frozen lock at startup: `k=96`, `q=4`, `tau=4/96`, `f_base=.005`, `beta=1`, `lambda=.01`, `p=.5`, `gamma=.1`, plus dimensions, seeds, and `eta=1`. The lock's stated selection metric is the mean Task1–5 **CIFAR train-side validation** cumulative accuracy; `docs/K_TAU_JOINT_SELECTION.md` records the 400/100 per-class development/validation split, seed 20260928, predeclared candidate grids, and tie rule. No CUB-specific selection code or CUB test metric is referenced by the lock or runner. **`CUB_TEST_USED_FOR_HPARAM_SELECTION=NO` for this frozen Full run**, as supported by the lock, CSV hashes, chronology, and config assertion.

The validation *evaluation* used only CIFAR train data held out from the development stream. However, the broader historical design of candidate grids was preceded by earlier CIFAR mechanism screens, so this audit does **not** prove that every candidate idea was conceived without awareness of any earlier CIFAR test result. This caveat does not turn CUB test data into a tuning input. The lock is a local working choice, not an author configuration.

A search of executable `src/`, `experiments/`, `configs/`, `scripts/`, and `tests/` for paper CUB reference values `85.15` and `72.91` found no computational-code match. Their appearance in reports is reference text, not a loss, stopping rule, threshold, or branch condition.

## 6. Independent metrics and interpretation

The separate saved Task20 file `cub200_task20_prediction_canary.npz` contains labels, logits, argmax predictions, and permuted labels. Without using the runner's metric helper, direct `pred == label` counting gives **4,670 / 5,794 = 80.600621%**. Directly averaging the 20 archived cumulative-accuracy fractions gives **87.860595%**. Every stored prediction equals the corresponding logit argmax. The runtime audit independently recomputed all 20 cumulative accuracies from captured logits/labels and matched the archive. `METRIC_RECOMPUTATION=PASS`.

**Does test leakage explain the local 87.86% / 80.60%? `NO` on the examined pathways.** No raw exact duplicate, cache split crossover, test/future-train update access, test-label-dependent prediction, evaluation mutation, CUB-test-driven config selection, or aggregation error was observed. The five visually similar same-class candidates are real source-benchmark caveats, but their observed count is far too small to account directly for the **+7.6906 pp** final gap to the paper. This is **not** a complete causal explanation of the gap: the local current-`W_B` replay-refit readout differs from a strict exemplar-free algorithm, author CUB hyperparameters are not disclosed, and identity to PILOT's separately distributed preprocessed image lists is unverified. The audit does not claim those differences caused a specific number of percentage points.

## Classification ledger

| Required field | Result |
|---|---|
| `RAW_SPLIT_EXACT_DUPLICATE` | 0 cross-split SHA-256 overlaps |
| `TRAIN_TEST_PATH_OVERLAP` | 0 canonical and 0 resolved absolute |
| `TRAIN_TEST_FEATURE_CACHE_ID_OVERLAP` | 0 |
| `TRAIN_TEST_SPARSE_CACHE_ID_OVERLAP` | 0 |
| `TEST_IDS_IN_WRITER_UPDATE` | 0 across 20 tasks |
| `TEST_IDS_IN_ELIGIBILITY_UPDATE` | 0 across 20 tasks |
| `TEST_IDS_IN_RIDGE_FIT` | 0 across 20 tasks |
| `FUTURE_TRAIN_IDS_ACCESSED` | 0 across writer, eligibility, and ridge in 20 tasks |
| `TEST_LABEL_AFFECTS_PREDICTIONS` | NO; logits bit-identical, predictions exactly equal |
| `EVALUATION_MUTATES_TRAIN_STATE` | NO; all hashed state components unchanged |
| `CUB_TEST_USED_FOR_HPARAM_SELECTION` | NO for the frozen Full run; broader CIFAR candidate-idea history qualified above |
| `HISTORICAL_TRAIN_REPLAY` | YES, 5,994 unique Task20 training sparse codes |
| `STRICT_EXEMPLAR_FREE` | NO |
| `METRIC_RECOMPUTATION` | PASS |
| `TEST_LEAKAGE` | NO_EVIDENCE |

## Reproduction and limitations

From the repository root, the two audit scripts can be rerun with `python experiments/audit/cub200_data_cache_audit.py` and `python experiments/audit/cub200_runtime_access_audit.py` **only after moving or explicitly preserving their existing outputs**: the runtime script deliberately refuses to overwrite the saved Task20 prediction canary. This report used the already existing frozen caches; it did not independently regenerate every ViT feature from raw images or cryptographically attest to the cached ViT weights. Source-path/order and sparse-from-feature identity are exhaustively checked, while near-duplicate discovery is a top-50 dHash diagnostic, not an exhaustive perceptual audit. The original official CUB split also need not be byte-identical to any distinct PILOT-provided preprocessed image list.

No commit or push was performed.

## Verification command

With `PYTHONPATH=src;experiments;upstream/LAMDA-PILOT`, `python -m pytest -q tests` passed **50 tests and 3 subtests**. A first invocation without that project import path stopped during test collection with `ModuleNotFoundError: btsp_cam`; it did not run tests or alter project files. The correctly configured invocation passed. `git diff --name-only` was empty; `git status --short` shows this audit report and audit-code directory as new/untracked alongside pre-existing untracked CUB/experiment materials. No existing tracked source/config/result was modified.
