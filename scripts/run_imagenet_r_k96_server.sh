#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
: "${IMAGENET_R_ROOT:?Set IMAGENET_R_ROOT to the supplied PILOT train/test folder root}"

args=(--config configs/btsp_cam_imagenet_r_k96.yaml
      --data-root "$IMAGENET_R_ROOT"
      --output-dir results/imagenet_r_k96)
if [[ -n "${RESUME:-}" ]]; then
  args+=(--resume "$RESUME")
fi
if [[ -n "${MAX_TASKS:-}" ]]; then
  args+=(--max-tasks "$MAX_TASKS")
fi
python experiments/run_btsp_cam_imagenet_r.py "${args[@]}"
