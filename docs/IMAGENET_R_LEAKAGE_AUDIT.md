# ImageNet-R split and replay access audit

Status on this workstation: **DATASET BLOCKED**. `data/imagenet-r` is absent, so no real ImageNet-R split/hash/accuracy claim is made. Synthetic tests are `INFRASTRUCTURE_ONLY`.

Implemented checks in `src/btsp_cam/imagenet_r.py`, `experiments/run_btsp_cam_imagenet_r.py`, and `scripts/check_imagenet_r.py`:

- Supplied PILOT `train/` and `test/` ImageFolders must expose the same 200 sorted class names and nonempty classes. Task groups are disjoint remapped intervals of 20; train/test relative sample names must not overlap. Duplicate path IDs in either split are rejected.
- Feature-cache metadata binds dataset, split, task, sample IDs/relative paths and labels, transform, backbone, output point, timm version, class order, augmentation/sample-order seeds and cache dtype. Sparse-cache metadata binds that feature manifest, split, task and projection/k.
- The writer receives only current-task train codes; the replay-refit closure can address only `train_taskXX` sparse files for `source<=task`. No test cache path is used by writer, eligibility, literal A/B, or replay ridge. The literal branch accumulates current-task Z only; it neither reloads old train codes nor transports A/B.
- Evaluation uses test codes of tasks 1..t and all seen logits, with a separately reported current-task-restricted diagnostic. Future test tasks are not accessed. `--max-tasks` and resume retain this boundary.

The current manifest hashes relative path/label/class mapping plus file size and modification time; it does **not** hash image content. This cannot rule out identical pixels under different paths, undisclosed external preprocessing/resplitting, or maliciously altered cache contents. On a real server, after `scripts/check_imagenet_r.py` passes, the appropriate conclusion is **NO EVIDENCE OF TEST LEAKAGE**, not “leakage impossible.” The original provenance of the supplied PILOT train/test trees remains **UNRESOLVED** until verified against a provider's release manifest.
