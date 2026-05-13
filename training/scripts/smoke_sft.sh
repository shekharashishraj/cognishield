#!/usr/bin/env bash
# Phase-1 SFT smoke. Must finish in < 5 minutes on one H200.
set -euo pipefail

CONFIG="${SFT_CONFIG:-training/configs/sft/qwen35_9b_smoke.yaml}"
DATA="${DATA_PATH:-cognibench.jsonl}"

if [[ ! -f "$DATA" ]]; then
  echo "[smoke_sft] $DATA not found. Generate it with:  python cognibench_pipeline.py" >&2
  exit 2
fi

python -m training.sft.train --config "$CONFIG" --override "data.path=$DATA"
