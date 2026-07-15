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

train.py / render_submission.py should only ever call `render()` from this module. Call
`describe_backend(device)` once at the start of a run (train.py / render_submission.py both
do) to print which backend was actually resolved -- the selection below is a silent
try/except ImportError chain, so without that print a failed AbsGS build on Colab
(colab_setup.sh only warns, never fails, if the AbsGS extension doesn't compile) would
silently downgrade training to vanilla CUDA with no indication in the logs.
"""

import torch

# Cached per device-type string ("cuda" / "cpu") so repeated render() calls in the training
# loop don't redo the import-probing every iteration.
_resolved_backend = {}


def _resolve_backend(device_type):
    if device_type in _resolved_backend:
        return _resolved_backend[device_type]

    if device_type == "cuda":
        try:
            from gaussian_renderer._cuda_backend_abs import render_cuda_abs
        except ImportError:
            pass
        else:
            _resolved_backend[device_type] = ("cuda_abs", render_cuda_abs)
            return _resolved_backend[device_type]

        try:
            from gaussian_renderer._cuda_backend import render_cuda
        except ImportError as e:
            raise ImportError(
                "Gaussians are on a CUDA device but neither `diff_gaussian_rasterization_abs` "
                "nor `diff_gaussian_rasterization` is installed. On a Colab GPU runtime, build "
                "the submodules first (see scripts/colab_setup.sh / PLAN.md Phase 0)."
            ) from e
        _resolved_backend[device_type] = ("cuda", render_cuda)
        return _resolved_backend[device_type]

    from gaussian_renderer.cpu_rasterizer import render_cpu
    _resolved_backend[device_type] = ("cpu", render_cpu)
    return _resolved_backend[device_type]


_BACKEND_DESCRIPTIONS = {
    "cuda_abs": "AbsGS-patched CUDA rasterizer (diff_gaussian_rasterization_abs) -- "
                "abs-gradient split criterion is ACTIVE.",
    "cuda": "official upstream CUDA rasterizer (diff_gaussian_rasterization) -- "
            "AbsGS is NOT active (extension not found/built); densify_and_prune falls back "
            "to vanilla single-threshold behavior. See scripts/colab_setup.sh to build it.",
    "cpu": "pure-PyTorch CPU reference rasterizer (gaussian_renderer/cpu_rasterizer.py) -- "
           "debug/smoke-test only, not for real training. AbsGS is NOT active.",
}


def describe_backend(device):
    """Resolve and print which rasterizer backend will actually be used for `device`.
    Safe/cheap to call once at the start of a training or render run."""
    device_type = torch.device(device).type
    name, _ = _resolve_backend(device_type)
    print(f"[gaussian_renderer] backend = {name}: {_BACKEND_DESCRIPTIONS[name]}")
    return name


def render(viewpoint_camera, pc, bg_color: torch.Tensor, scaling_modifier=1.0, sh_degree_override=None):
    """Returns a dict with keys: render (3,H,W), viewspace_points, visibility_filter, radii."""
    device = pc.get_xyz.device
    _, render_fn = _resolve_backend(device.type)
    return render_fn(viewpoint_camera, pc, bg_color, scaling_modifier, sh_degree_override)
