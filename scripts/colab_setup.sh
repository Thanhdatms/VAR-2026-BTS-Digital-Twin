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
pip install --no-cache-dir -v -e .

echo "== Checking for a physical NVIDIA GPU (independent of torch) =="
# torch.cuda.is_available() alone can't distinguish "no GPU" from "GPU present but the
# installed torch build can't talk to it" (e.g. pip's default `torch` wheel bundles a newer
# CUDA runtime -- cu13 as of this writing -- than the Colab host's NVIDIA driver supports).
# nvidia-smi reports the physical device and the driver's max-supported CUDA version even
# when torch can't use the GPU at all, so use it to decide whether a reinstall is worth
# attempting *before* gating on torch.cuda.is_available().
HAS_NVIDIA_GPU=0
DRIVER_CUDA=""
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    HAS_NVIDIA_GPU=1
    DRIVER_CUDA="$(nvidia-smi | grep -oP 'CUDA Version:\s*\K[0-9]+\.[0-9]+' || true)"
fi

if [ "$HAS_NVIDIA_GPU" = "1" ] && ! python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    if [ -n "$DRIVER_CUDA" ]; then
        echo "   GPU present but current torch build can't use it (driver supports up to CUDA $DRIVER_CUDA)"
        echo "   -> reinstalling torch/torchvision for cu$(echo "$DRIVER_CUDA" | tr -d '.') to match the driver"
        pip install --index-url "https://download.pytorch.org/whl/cu$(echo "$DRIVER_CUDA" | tr -d '.')" \
            --force-reinstall torch torchvision || \
            echo "   WARNING: reinstall for cu$(echo "$DRIVER_CUDA" | tr -d '.') failed (that exact wheel tag may not" \
                 "exist) -- pick a matching build manually from https://pytorch.org/get-started/locally/."
    else
        echo "   WARNING: nvidia-smi found a GPU but its driver's CUDA version string could not be parsed --" \
             "skipping auto-reinstall, torch will likely stay CPU-only."
    fi
fi

if python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "== CUDA GPU detected: building the official diff-gaussian-rasterization extension =="

    export TORCH_CUDA_ARCH_LIST="$(python3 -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability(0))')"
    echo "   TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST (detected from current GPU)"

    mkdir -p submodules
    if [ ! -d submodules/diff-gaussian-rasterization ]; then
        git clone --recursive https://github.com/graphdeco-inria/diff-gaussian-rasterization \
            submodules/diff-gaussian-rasterization
    fi

    # Upstream rasterizer_impl.h uses std::uintptr_t/uint32_t/uint64_t without including
    # <cstdint>. Older nvcc/libstdc++ combos pulled those in transitively via other headers,
    # but that doesn't happen with newer CUDA toolkits (12.x/13.x) in C++20 mode, producing
    # "identifier is undefined" build errors. Patch it in until upstream fixes it.
    RASTERIZER_IMPL_H="submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h"
    if ! grep -q '#include <cstdint>' "$RASTERIZER_IMPL_H"; then
        sed -i '1i #include <cstdint>' "$RASTERIZER_IMPL_H"
    fi

    pip install -v submodules/diff-gaussian-rasterization
    python3 -c "from diff_gaussian_rasterization import GaussianRasterizer; print('diff_gaussian_rasterization: OK')"

    echo "== Building the AbsGS rasterizer (arXiv:2404.10484, optional) =="
    echo "   Fixes 'gradient collision' blur on thin structures (BTS wires) and high-frequency"
    echo "   patterns (striped/corrugated roofs) by also densifying on an abs-value screen-space"
    echo "   gradient that can't cancel out the way the signed one does. Preferred automatically"
    echo "   by gaussian_renderer/__init__.py when present; best-effort here -- if this fails,"
    echo "   training still proceeds on the official rasterizer built above."
    #
    # AbsGS isn't a standalone repo: its patched rasterizer lives as a subdirectory inside the
    # TY424/AbsGS fork of the full gaussian-splatting repo (verified by inspecting that repo's
    # tree -- submodules/diff-gaussian-rasterization-abs is a real checked-in directory there,
    # not a git submodule pointer), so shallow-clone the whole fork and lift just that
    # subdirectory instead of `git clone`-ing a rasterizer repo directly.
    if [ ! -d submodules/diff-gaussian-rasterization-abs ]; then
        rm -rf /tmp/absgs_src
        if git clone --recursive --depth 1 https://github.com/TY424/AbsGS.git /tmp/absgs_src; then
            cp -r /tmp/absgs_src/submodules/diff-gaussian-rasterization-abs submodules/diff-gaussian-rasterization-abs
        else
            echo "   WARNING: could not fetch TY424/AbsGS -- continuing with the official rasterizer only."
        fi
        rm -rf /tmp/absgs_src
    fi

    if [ -d submodules/diff-gaussian-rasterization-abs ]; then
        # Same missing-<cstdint>-include issue as the official rasterizer (see the sed above),
        # same fix.
        ABS_RASTERIZER_IMPL_H="submodules/diff-gaussian-rasterization-abs/cuda_rasterizer/rasterizer_impl.h"
        if [ -f "$ABS_RASTERIZER_IMPL_H" ] && ! grep -q '#include <cstdint>' "$ABS_RASTERIZER_IMPL_H"; then
            sed -i '1i #include <cstdint>' "$ABS_RASTERIZER_IMPL_H"
        fi

        if pip install -v submodules/diff-gaussian-rasterization-abs && \
           python3 -c "from diff_gaussian_rasterization_abs import GaussianRasterizer; print('diff_gaussian_rasterization_abs: OK')"; then
            echo "   AbsGS rasterizer built OK -- will be used automatically for training/rendering."
        else
            echo "   WARNING: AbsGS rasterizer build failed -- continuing with the official rasterizer only"
            echo "   (training still works, just without the abs-gradient densification fix)."
        fi
    fi
else
    echo "== No usable CUDA GPU: skipping the CUDA rasterizer build =="
    if [ "$HAS_NVIDIA_GPU" = "1" ]; then
        echo "   nvidia-smi detected a physical GPU but torch still can't use it after the reinstall"
        echo "   attempt above -- check 'python3 -c \"import torch; print(torch.cuda.is_available())\"'"
        echo "   manually and pick a matching wheel from https://pytorch.org/get-started/locally/."
    else
        echo "   Switch the Colab runtime to a GPU (Runtime > Change runtime type) for real training."
    fi
    echo "   Training/rendering will use gaussian_renderer/cpu_rasterizer.py (slow reference path)."
fi

echo "== Setup complete =="