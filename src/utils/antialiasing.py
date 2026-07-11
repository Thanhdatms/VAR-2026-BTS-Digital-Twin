"""Mip-Splatting-style antialiasing, implemented as a Python-side opacity pre-compensation
that works with the *vanilla* graphdeco-inria/diff-gaussian-rasterization build (confirmed by
reading its source directly: GaussianRasterizationSettings has no 'antialiasing'/'kernel_size'
field -- that API belongs to the separate autonomousvision/mip-splatting fork, which needs its
own rasterizer submodule and per-Gaussian 3D filter state. We deliberately do NOT depend on
that fork here, since colab_setup.sh builds the vanilla one).

The vanilla CUDA kernel always inflates each Gaussian's 2D screen-space covariance by a fixed
+0.3px on the diagonal for numerical stability, regardless of the Gaussian's true footprint.
For a Gaussian much smaller than a pixel (thin wires, high-frequency stripe textures), that
fixed inflation makes it render far more spread-out/opaque than it should be -- this is the
main source of the blur artifacts described in the 2026-07-12 review (power-line wires,
corrugated-roof stripes).

Fix: compute each Gaussian's true (pre-dilation) 2D covariance in Python, and scale its opacity
down by sqrt(det(true_cov)/det(dilated_cov)) before handing opacities to the rasterizer. This
is the "2D Mip filter" half of Mip-Splatting (not the "3D smoothing filter" half, which needs
persistent per-Gaussian state computed across all training views and a custom kernel -- see
module docstring). It's a stateless per-render-call computation, cheap (O(N) 3x3 matrix ops,
no H*W blow-up), and safe to enable/disable per call. Shared by both rasterizer backends
(gaussian_renderer/_cuda_backend.py, gaussian_renderer/cpu_rasterizer.py) so they use identical
math instead of two hand-copies that could drift.
"""

import torch

from utils.general_utils import build_scaling_rotation
from utils.graphics_utils import fov2focal

LOW_PASS_FILTER = 0.3  # matches the rasterizer's fixed screen-space dilation


def compute_screenspace_cov2d(means3D, scales, rotations, viewpoint_camera, scaling_modifier=1.0):
    """Pre-dilation 2D (screen-space, pixel^2 units) covariance of each Gaussian's EWA-splatting
    projection: Cov = J @ (W^T Sigma3D W) @ J^T, before the rasterizer's own +0.3px inflation.

    Returns (a, b, c, tz): symmetric 2x2 covariance entries [[a,b],[b,c]] and view-space depth,
    each shape (N,).
    """
    device = means3D.device
    N = means3D.shape[0]
    world_view = viewpoint_camera.world_view_transform  # row-vector convention

    points_hom = torch.cat([means3D, torch.ones(N, 1, device=device, dtype=means3D.dtype)], dim=1)
    points_view = points_hom @ world_view
    tz = points_view[:, 2]

    L = build_scaling_rotation(scaling_modifier * scales, rotations)
    Sigma3D = L @ L.transpose(1, 2)

    Wr = world_view[:3, :3]
    Sigma_view = torch.einsum('ab,nbc,cd->nad', Wr.t(), Sigma3D, Wr)

    W, H = int(viewpoint_camera.image_width), int(viewpoint_camera.image_height)
    fx = fov2focal(viewpoint_camera.FoVx, W)
    fy = fov2focal(viewpoint_camera.FoVy, H)
    tz_safe = tz.clamp(min=1e-6)
    J = torch.zeros(N, 3, 3, device=device, dtype=means3D.dtype)
    J[:, 0, 0] = fx / tz_safe
    J[:, 0, 2] = -fx * points_view[:, 0] / (tz_safe ** 2)
    J[:, 1, 1] = fy / tz_safe
    J[:, 1, 2] = -fy * points_view[:, 1] / (tz_safe ** 2)

    Cov = torch.einsum('nab,nbc,ndc->nad', J, Sigma_view, J)
    return Cov[:, 0, 0], Cov[:, 0, 1], Cov[:, 1, 1], tz


def mip_antialiasing_opacity_coef(a0, b0, c0, low_pass_filter=LOW_PASS_FILTER):
    """sqrt(det(true_cov)/det(dilated_cov)) per Gaussian, in [0, 1]."""
    a = a0 + low_pass_filter
    c = c0 + low_pass_filter
    det0 = (a0 * c0 - b0 * b0).clamp(min=0.0)
    det = (a * c - b0 * b0).clamp(min=1e-8)
    return torch.sqrt((det0 / det).clamp(min=0.0, max=1.0))
