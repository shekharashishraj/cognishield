#!/usr/bin/env bash
# Full GRPO. ~36-48 hours on one H200 (paper §5.2).
# Requires:
#   1. checkpoints/tutor_sft/final/ (run_full_sft.sh first)
#   2. judge server on :8001 (training/scripts/start_judge_server.sh)
#   3. student server on :8002 (training/scripts/start_student_server.sh)
set -euo pipefail

CONFIG="${GRPO_CONFIG:-training/configs/rl/grpo_full.yaml}"

# Sanity probe — fail fast if the helper servers aren't up.
for url in http://localhost:8001/v1/models http://localhost:8002/v1/models; do
  if ! curl -fsS "$url" >/dev/null 2>&1; then
    echo "[run_full_grpo] $url is not responding. Start judge + student servers first." >&2
    exit 2
  fi
done

python -m training.rl.train --config "$CONFIG"
