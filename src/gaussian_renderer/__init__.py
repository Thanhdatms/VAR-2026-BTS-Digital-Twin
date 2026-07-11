"""Single entry point for turning a GaussianModel + Camera into a rendered image.

This is the one place in the codebase that is allowed to know about the rasterizer backend
(see CLAUDE.md section 3 / PLAN.md Phase 2 — the compute target, Colab GPU vs Colab CPU, was
left open, so the backend is chosen at call time instead of being hard-coded):

- CUDA device + `diff_gaussian_rasterization` importable -> official upstream kernels
  (gaussian_renderer/_cuda_backend.py), matching graphdeco-inria/gaussian-splatting exactly.
- Anything else (CPU, or CUDA without the extension built) -> gaussian_renderer/cpu_rasterizer.py,
  a pure-PyTorch reference implementation. It is correct but O(N_gaussians * H * W) per image
  (no tiling), so treat it as a debug/smoke-test path — not for full-resolution or
  full-iteration-count training. See its module docstring for scale guidance.

train.py / render_submission.py should only ever call `render()` from this module.
"""

import torch


def render(viewpoint_camera, pc, bg_color: torch.Tensor, scaling_modifier=1.0,
           sh_degree_override=None, antialiasing=0.0):
    """Returns a dict with keys: render (3,H,W), viewspace_points, visibility_filter, radii.

    antialiasing: Mip-Splatting-style opacity compensation for the fixed screen-space
        dilation (see PipelineParams.antialiasing in arguments/__init__.py for why this
        matters). 0.0/False disables it, 1.0/True is full compensation, a float in between
        linearly ramps toward it (see train.py's antialiasing_progress -- switching this on
        abruptly mid-training causes a loss spike, since opacity was fit assuming no
        compensation). Must match between the render() calls used for training and the ones
        used for submission rendering (both at the final, fully-ramped value), or output
        brightness/opacity will be systematically off.
    """
    device = pc.get_xyz.device

    if device.type == "cuda":
        try:
            from gaussian_renderer._cuda_backend import render_cuda
        except ImportError as e:
            raise ImportError(
                "Gaussians are on a CUDA device but `diff_gaussian_rasterization` is not "
                "installed. On a Colab GPU runtime, build the official submodules first "
                "(see scripts/colab_setup.sh / PLAN.md Phase 0)."
            ) from e
        return render_cuda(viewpoint_camera, pc, bg_color, scaling_modifier,
                            sh_degree_override, antialiasing)

    from gaussian_renderer.cpu_rasterizer import render_cpu
    return render_cpu(viewpoint_camera, pc, bg_color, scaling_modifier,
                       sh_degree_override, antialiasing)
