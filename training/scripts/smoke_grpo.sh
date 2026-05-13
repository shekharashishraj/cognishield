#!/usr/bin/env bash
# Phase-2 GRPO smoke. Mock judges + mock student; transformers backend.
# Must finish in < 15 minutes on one H200.
set -euo pipefail

CONFIG="${GRPO_CONFIG:-training/configs/rl/grpo_smoke.yaml}"
DATA="${DATA_PATH:-data/grpo_smoke_problems.jsonl}"

if [[ ! -f "$DATA" ]]; then
  echo "[smoke_grpo] $DATA not found." >&2
  echo "  GRPO needs records with 'problem' and 'solution' fields (the SFT JSONL doesn't carry GT)." >&2
  echo "  Seed a tiny file or set DATA_PATH=/path/to/problems.jsonl" >&2
  exit 2
fi

python -m training.rl.train --config "$CONFIG" --override "data.path=$DATA"
