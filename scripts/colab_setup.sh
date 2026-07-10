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
pip install -v -e .

if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "== CUDA GPU detected: building the official diff-gaussian-rasterization extension =="

    # torch.utils.cpp_extension refuses to build if the system nvcc's CUDA version doesn't
    # match the CUDA version PyTorch itself was built against (RuntimeError:
    # "The detected CUDA version (X) mismatches the version that was used to compile
    # PyTorch (Y)"). This happens on Colab because the pip-installed torch wheel and the
    # VM image's preinstalled system CUDA toolkit (nvcc) can drift out of sync -- pip always
    # pulls the newest torch/CUDA build, while the VM's nvcc is whatever the image shipped
    # with. Fix: reinstall torch matching the system nvcc's version, *before* attempting the
    # extension build.
    if command -v nvcc >/dev/null 2>&1; then
        SYSTEM_CUDA="$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' || true)"
        TORCH_CUDA="$(python -c 'import torch; print(torch.version.cuda or "")')"
        if [ -n "$SYSTEM_CUDA" ] && [ -n "$TORCH_CUDA" ] && [ "$SYSTEM_CUDA" != "$TORCH_CUDA" ]; then
            echo "   System nvcc is CUDA $SYSTEM_CUDA but installed torch was built for CUDA $TORCH_CUDA"
            echo "   -> reinstalling torch/torchvision for cu$(echo "$SYSTEM_CUDA" | tr -d '.') to match"
            # --force-reinstall is required: pip's version comparator treats the local CUDA
            # tag (+cu130 vs +cu128) such that "already satisfied" can win over --upgrade
            # alone, silently leaving the mismatched build in place. Deliberately NOT using
            # --no-deps: torch's bundled nvidia-*-cuXX runtime packages must be swapped too,
            # or torch would compile against cu128 headers but load cu130 runtime libs at
            # import time.
            pip install --index-url "https://download.pytorch.org/whl/cu$(echo "$SYSTEM_CUDA" | tr -d '.')" \
                --force-reinstall torch torchvision
        fi
    fi

    # Pin the build to the actual GPU's compute capability. Without this, nvcc falls back to
    # the extension's setup.py default arch list, which can include compute capabilities
    # (e.g. compute_35/50) that newer CUDA toolkits (12.x/13.x, common on current Colab
    # images) have dropped support for -- a common cause of "unsupported gpu architecture"
    # build failures on this exact extension.
    export TORCH_CUDA_ARCH_LIST="$(python -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability(0))')"
    echo "   TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST (detected from current GPU)"

    mkdir -p submodules
    if [ ! -d submodules/diff-gaussian-rasterization ]; then
        git clone --recursive https://github.com/graphdeco-inria/diff-gaussian-rasterization \
            submodules/diff-gaussian-rasterization
    fi
    # Plain `pip install <path>` hides the actual nvcc/gcc compiler output on failure (it only
    # prints a generic "did not run successfully, see above for output" wrapper, with nothing
    # useful above it) -- confirmed empirically while debugging this exact extension. -v forces
    # pip to stream the real build subprocess output so failures are actually diagnosable.
    pip install -v submodules/diff-gaussian-rasterization
    python -c "from diff_gaussian_rasterization import GaussianRasterizer; print('diff_gaussian_rasterization: OK')"
else
    echo "== No CUDA GPU detected: skipping the CUDA rasterizer build =="
    echo "   Training/rendering will use gaussian_renderer/cpu_rasterizer.py (slow reference path)."
    echo "   Switch the Colab runtime to a GPU (Runtime > Change runtime type) for real training."
fi

echo "== Setup complete =="
