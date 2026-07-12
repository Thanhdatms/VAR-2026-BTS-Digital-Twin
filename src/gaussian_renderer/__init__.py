"""Single entry point for turning a GaussianModel + Camera into a rendered image.

This is the one place in the codebase that is allowed to know about the rasterizer backend
(see CLAUDE.md section 3 / PLAN.md Phase 2 — the compute target, Colab GPU vs Colab CPU, was
left open, so the backend is chosen at call time instead of being hard-coded):

- CUDA device + `diff_gaussian_rasterization_abs` importable -> AbsGS-patched kernels
  (gaussian_renderer/_cuda_backend_abs.py) — preferred whenever built: same render math as
  upstream, plus a homodirectional/abs screen-space gradient that fixes "gradient collision"
  blur on thin structures and high-frequency patterns (see that module's docstring).
- CUDA device + only `diff_gaussian_rasterization` importable -> official upstream kernels
  (gaussian_renderer/_cuda_backend.py), matching graphdeco-inria/gaussian-splatting exactly.
- Anything else (CPU, or CUDA without either extension built) -> gaussian_renderer/cpu_rasterizer.py,
  a pure-PyTorch reference implementation. It is correct but O(N_gaussians * H * W) per image
  (no tiling), so treat it as a debug/smoke-test path — not for full-resolution or
  full-iteration-count training. See its module docstring for scale guidance.

train.py / render_submission.py should only ever call `render()` from this module.
"""

import torch


def render(viewpoint_camera, pc, bg_color: torch.Tensor, scaling_modifier=1.0, sh_degree_override=None):
    """Returns a dict with keys: render (3,H,W), viewspace_points, visibility_filter, radii."""
    device = pc.get_xyz.device

    if device.type == "cuda":
        try:
            from gaussian_renderer._cuda_backend_abs import render_cuda_abs
        except ImportError:
            pass
        else:
            return render_cuda_abs(viewpoint_camera, pc, bg_color, scaling_modifier, sh_degree_override)

        try:
            from gaussian_renderer._cuda_backend import render_cuda
        except ImportError as e:
            raise ImportError(
                "Gaussians are on a CUDA device but neither `diff_gaussian_rasterization_abs` "
                "nor `diff_gaussian_rasterization` is installed. On a Colab GPU runtime, build "
                "the submodules first (see scripts/colab_setup.sh / PLAN.md Phase 0)."
            ) from e
        return render_cuda(viewpoint_camera, pc, bg_color, scaling_modifier, sh_degree_override)

    from gaussian_renderer.cpu_rasterizer import render_cpu
    return render_cpu(viewpoint_camera, pc, bg_color, scaling_modifier, sh_degree_override)
