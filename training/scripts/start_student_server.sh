#!/usr/bin/env bash
# Launch the frozen student simulator (Llama-3.1-8B-Instruct) on port 8002.
set -euo pipefail

PORT="${STUDENT_PORT:-8002}"
MODEL="${STUDENT_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

exec vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.25 \
  --served-model-name "$MODEL"
