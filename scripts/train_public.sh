#!/usr/bin/env bash
# Train all 5 public_set scenes (HCM0181, HCM0193, HCM0204, hcm0031, hcm0034 -- the only
# scenes that ship ground-truth test images, see CLAUDE.md section 2). Use this to validate
# the Tier 1 changes (antialiasing, densify tuning -- see arguments/__init__.py) against real
# metrics via scripts/eval_public.sh BEFORE spending compute on the 8 private_set1 scenes.
#
# Must run on a CUDA machine with the diff-gaussian-rasterization submodule built (see
# scripts/colab_setup.sh / CLAUDE.md section 3) -- this repo has no local CUDA.
#
# Usage:
#   bash scripts/train_public.sh
#   ITERATIONS=40000 bash scripts/train_public.sh          # override any var below
#   bash scripts/train_public.sh --no_antialiasing          # extra args forwarded straight to
#                                                            #   train.py (no leading "--" needed,
#                                                            #   this script already adds one --
#                                                            #   e.g. for an A/B comparison run)

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASET_ROOT="${DATASET_ROOT:-dataset/phase1/public_set}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/public_set}"
DATA_DEVICE="${DATA_DEVICE:-cuda}"
ITERATIONS="${ITERATIONS:-30000}"

python3 scripts/train_all.py \
  --dataset_root "$DATASET_ROOT" \
  --output_root "$OUTPUT_ROOT" \
  --data_device "$DATA_DEVICE" \
  --iterations "$ITERATIONS" \
  --skip_existing \
  -- \
  --save_iterations 7000 15000 30000 \
  --val_interval 1000 \
  "$@"
