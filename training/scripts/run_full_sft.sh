#!/usr/bin/env bash
# Full SFT warm-start. ~6-12 hours for 20-30k dialogs on one H200.
set -euo pipefail

CONFIG="${SFT_CONFIG:-training/configs/sft/qwen35_9b_full.yaml}"

accelerate launch \
  --config_file training/configs/accelerate/h200_single.yaml \
  -m training.sft.train --config "$CONFIG"
