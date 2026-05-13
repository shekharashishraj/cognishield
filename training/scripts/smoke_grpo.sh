#!/usr/bin/env bash
# Phase-2 GRPO smoke. Mock judges + mock student; transformers backend.
# Must finish in < 15 minutes on one H200.
set -euo pipefail

CONFIG="${GRPO_CONFIG:-training/configs/rl/grpo_smoke.yaml}"
DATA="${DATA_PATH:-cognibench.jsonl}"

if [[ ! -f "$DATA" ]]; then
  echo "[smoke_grpo] $DATA not found. Generate it with:  python cognibench_pipeline.py" >&2
  exit 2
fi

python -m training.rl.train --config "$CONFIG" --override "data.path=$DATA"
