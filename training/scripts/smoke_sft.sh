#!/usr/bin/env bash
# Phase-1 SFT smoke. Must finish in < 5 minutes on one H200.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

CONFIG="${SFT_CONFIG:-$REPO_ROOT/training/configs/sft/qwen35_9b_smoke.yaml}"
DATA="${DATA_PATH:-$REPO_ROOT/sft.generated.batch.jsonl}"

if [[ ! -f "$DATA" ]]; then
  echo "[smoke_sft] $DATA not found. Override with: DATA_PATH=/path/to/sft.jsonl $0" >&2
  exit 2
fi

cd "$REPO_ROOT"
python -m training.sft.train --config "$CONFIG" --override "data.path=$DATA"
