#!/usr/bin/env bash
# Launch the judge vLLM server (Qwen2.5-14B-Instruct AWQ) on port 8001.
# Co-resident with the tutor and student on a single H200.
set -euo pipefail

PORT="${JUDGE_PORT:-8001}"
MODEL="${JUDGE_MODEL:-Qwen/Qwen2.5-14B-Instruct-AWQ}"

exec vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --quantization awq \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.15 \
  --served-model-name "$MODEL" \
  --enforce-eager
