#!/usr/bin/env bash
# Render + score public_set checkpoints against their real ground-truth test images, at EVERY
# saved iteration (7000/15000/30000 by default) -- not just the final one. This is PLAN.md
# Phase 6 point 6: pick the iteration with the best competition score on public_set instead of
# always submitting whatever the last checkpoint happens to be (the best-scoring iteration is
# not always the last one, e.g. if opacity resets or overfitting hurt late iterations on a
# small scene).
#
# Two steps per iteration: render_submission.py (produce PNGs for every test_poses.csv row) ->
# evaluate.py (LPIPS/SSIM/PSNR + the competition's score formula, see CLAUDE.md section 1).
# Requires that scripts/train_public.sh actually saved checkpoints at all requested iterations
# (its default --save_iterations is 7000 15000 30000, matching this script's default).
#
# Run this AFTER scripts/train_public.sh. Needs a CUDA machine (same constraint as training).
#
# Usage:
#   bash scripts/eval_public.sh
#   PSNR_MAX=45 bash scripts/eval_public.sh
#   ITERATIONS="15000 30000" bash scripts/eval_public.sh   # only compare a subset
#   bash scripts/eval_public.sh --no_antialiasing            # forwarded to render_submission.py
#                                                              #   (must match the training run)

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASET_ROOT="${DATASET_ROOT:-dataset/phase1/public_set}"
MODELS_ROOT="${MODELS_ROOT:-output/public_set}"
OUTPUT_ROOT="${OUTPUT_ROOT:-submission/public_set}"
DATA_DEVICE="${DATA_DEVICE:-cuda}"
# No official PSNR_max in the brief provided to this repo (see CLAUDE.md section 1/5) --
# 50 here is only evaluate.py's own documented placeholder, NOT a competition-confirmed value.
# Override with PSNR_MAX=<value> once/if the organizer publishes one, so scores stay comparable
# across runs done with different placeholders.
PSNR_MAX="${PSNR_MAX:-50}"
REPORT_PREFIX="${REPORT_PREFIX:-eval_report_public}"
ITERATIONS="${ITERATIONS:-7000 15000 30000}"

REPORT_ARGS=()
for it in $ITERATIONS; do
    render_dir="${OUTPUT_ROOT}/iter_${it}"
    report_path="${REPORT_PREFIX}_iter_${it}.json"

    echo "== [iter $it] Rendering $DATASET_ROOT test poses from $MODELS_ROOT =="
    python3 src/render_submission.py \
        --dataset_root "$DATASET_ROOT" \
        --models_root "$MODELS_ROOT" \
        --output_root "$render_dir" \
        --data_device "$DATA_DEVICE" \
        --iteration "$it" \
        "$@"

    echo "== [iter $it] Scoring (psnr_max=$PSNR_MAX) =="
    python3 src/evaluate.py \
        --renders_root "$render_dir" \
        --dataset_root "$DATASET_ROOT" \
        --psnr_max "$PSNR_MAX" \
        --report_path "$report_path"

    REPORT_ARGS+=("${it}:${report_path}")
    echo
done

echo "== Summary across iterations =="
python3 - "${REPORT_ARGS[@]}" <<'PYEOF'
import json
import sys

rows = []
for entry in sys.argv[1:]:
    it, path = entry.split(":", 1)
    with open(path) as f:
        report = json.load(f)
    means = [s["mean"] for s in report["scenes"].values()]
    n = len(means)
    row = {
        "iteration": int(it),
        "psnr": sum(m["psnr"] for m in means) / n,
        "ssim": sum(m["ssim"] for m in means) / n,
        "lpips": sum(m["lpips"] for m in means) / n,
        "score": report["final_score"],
    }
    rows.append(row)

print(f"{'iteration':>10} | {'PSNR':>7} | {'SSIM':>7} | {'LPIPS':>7} | {'score':>7}")
print("-" * 52)
for r in rows:
    print(f"{r['iteration']:>10} | {r['psnr']:7.2f} | {r['ssim']:7.4f} | "
          f"{r['lpips']:7.4f} | {r['score']:7.4f}")

best = max(rows, key=lambda r: r["score"])
print(f"\nBest iteration by competition score: {best['iteration']} (score={best['score']:.4f})")
PYEOF
