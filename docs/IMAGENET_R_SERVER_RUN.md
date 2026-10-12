# ImageNet-R server runbook

1. Supply a PILOT-compatible dataset tree (do not commit images): `$IMAGENET_R_ROOT/train/<200 class folders>/*` and `$IMAGENET_R_ROOT/test/<same class folders>/*`. The upstream loader does not create an 80/20 split. Record provider, version, checksums and train/test generation separately.
2. Install project dependencies and verify frozen ViT AugReg ImageNet-21K weights are accessible. Preparation scripts do not download datasets.
3. From repository root, run:

```bash
python scripts/check_server_env.py --data-root "$IMAGENET_R_ROOT"
python scripts/check_imagenet_r.py --data-root "$IMAGENET_R_ROOT" --manifest results/imagenet_r_k96/dataset_manifest.json
python experiments/run_btsp_cam_imagenet_r.py --check-data --data-root "$IMAGENET_R_ROOT"
IMAGENET_R_ROOT=/path/to/imagenet-r CUDA_VISIBLE_DEVICES=0 MAX_TASKS=1 bash scripts/run_imagenet_r_k96_server.sh
```

The one-task command is a real-data smoke test, not a full run. Inspect `results/imagenet_r_k96/{task_metrics.csv,summary.json,protocol.json}`. Continue from the exact checkpoint:

```bash
IMAGENET_R_ROOT=/path/to/imagenet-r CUDA_VISIBLE_DEVICES=0 \
RESUME=results/imagenet_r_k96/checkpoints/task_01.npz \
bash scripts/run_imagenet_r_k96_server.sh
```

With no `MAX_TASKS`, the script stops at task **10**, not task 20: ImageNet-R B0-Inc20 has 200/20=10 tasks. `CUDA_VISIBLE_DEVICES` is inherited; no GPU index is embedded. The script accepts `RESUME` and `MAX_TASKS`. Never use a prior output directory without `RESUME`.

Output: `results/imagenet_r_k96/{config_snapshot.yaml,protocol.json,task_metrics.csv,writer_stats.csv,readout_comparison.csv,summary.json,runtime.json,checkpoints/task_XX.{npz,json}}`. Frozen feature/test sparse caches are under ignored `cache/imagenet_r/`; historical training sparse codes for the experimental oracle are under ignored `experimental_replay_codes/imagenet_r/`. A checkpoint contains writer W_B, eligibility traces, RNG state, projection R/b, class order, task index, seen classes and literal ridge A/B/W. The sidecar records config/dataset/checkpoint hashes, both readout metrics and the experimental replay-storage boundary. Resume rejects hash, class-order and task-history mismatches.

If the server dataset is not identical to a prior checkpoint/cache, do not resume it. The path/mtime manifest is a practical guard, not cryptographic image-integrity proof. `EXPERIMENTAL_CURRENT_WB_REPLAY_REFIT` is not a strict exemplar-free or paper-original result.
