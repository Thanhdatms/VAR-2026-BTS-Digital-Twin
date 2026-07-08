#!/usr/bin/env bash
# Set up this project on a Google Colab runtime (GPU or CPU — see CLAUDE.md section 3).
#
# Usage, inside a Colab cell after cloning/uploading the repo:
#   !bash scripts/colab_setup.sh
#
# On a GPU runtime this builds the official diff-gaussian-rasterization CUDA extension
# (PLAN.md Phase 0), so training uses the real tiled rasterizer from
# graphdeco-inria/gaussian-splatting. Note: this project does NOT need the `simple-knn`
# submodule the upstream repo also builds — its only use upstream is the neighbor-distance
# point-cloud initialization, which we compute with scipy.spatial.cKDTree instead (CPU-only,
# no CUDA needed either way — see scene/gaussian_model.py's module docstring).
#
# On a CPU runtime (no nvcc / no CUDA-capable GPU), the CUDA build is skipped entirely and
# training/rendering automatically falls back to gaussian_renderer/cpu_rasterizer.py — a
# correct but unoptimized reference implementation, fine for smoke tests, not for a
# competitive full training run (see that file's docstring for why).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "== Installing project Python dependencies =="
pip install -q -e .

if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "== CUDA GPU detected: building the official diff-gaussian-rasterization extension =="
    mkdir -p submodules
    if [ ! -d submodules/diff-gaussian-rasterization ]; then
        git clone --recursive https://github.com/graphdeco-inria/diff-gaussian-rasterization \
            submodules/diff-gaussian-rasterization
    fi
    pip install -q submodules/diff-gaussian-rasterization
    python -c "from diff_gaussian_rasterization import GaussianRasterizer; print('diff_gaussian_rasterization: OK')"
else
    echo "== No CUDA GPU detected: skipping the CUDA rasterizer build =="
    echo "   Training/rendering will use gaussian_renderer/cpu_rasterizer.py (slow reference path)."
    echo "   Switch the Colab runtime to a GPU (Runtime > Change runtime type) for real training."
fi

echo "== Setup complete =="
