"""Pure-PyTorch differentiable Gaussian rasterizer — fallback backend used when no CUDA
device / no `diff_gaussian_rasterization` build is available (see gaussian_renderer/__init__.py).

This exists only because the compute target (Colab GPU vs Colab CPU) was left open (CLAUDE.md
section 3): there is no CPU/portable path in the official graphdeco-inria/gaussian-splatting
repo at all — the whole point of that project is a custom tiled CUDA rasterizer. What's here
is a from-scratch reimplementation of the same math (EWA splatting projection + sequential
alpha compositing, see the 3DGS paper), done in plain tensor ops so autograd "just works",
with NO tiling / spatial acceleration structure.

Complexity: O(N_gaussians * H * W) per rendered image (every gaussian is composited against
the full image, batched with a sequential-in-batch, vectorized-within-batch alpha-blend via
`cumprod`, since alpha compositing is inherently order-dependent and can't be reduced to a
single matmul). This is many orders of magnitude slower than the tiled CUDA kernel and is only
intended for:
  - correctness smoke tests of the data/training pipeline without a GPU
  - tiny scenes / heavily downscaled renders during local development

Do NOT expect this path to produce a competitive submission — see CLAUDE.md section 3 and
PLAN.md Phase 0 for the real (CUDA) training path.
"""

import torch

from utils.antialiasing import (LOW_PASS_FILTER as _LOW_PASS_FILTER,
                                 compute_screenspace_cov2d, mip_antialiasing_opacity_coef)
from utils.sh_utils import eval_sh

_NEAR_PLANE = 0.2
_TARGET_ELEMENTS_PER_BATCH = 3_000_000  # bounds peak memory of the (B,H,W) intermediate tensors


def render_cpu(viewpoint_camera, pc, bg_color: torch.Tensor, scaling_modifier=1.0,
                sh_degree_override=None, antialiasing=False):
    device = pc.get_xyz.device
    H, W = int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)
    N = pc.get_xyz.shape[0]

    means3D = pc.get_xyz
    # zero-valued screen-space offset the training loop reads .grad from for densification
    # (kept shape-compatible with the CUDA backend's screenspace_points, see render_cuda).
    screenspace_points = torch.zeros_like(means3D, requires_grad=True)

    world_view = viewpoint_camera.world_view_transform  # row-vector convention (see scene/cameras.py)
    full_proj = viewpoint_camera.full_proj_transform

    points_hom = torch.cat([means3D, torch.ones(N, 1, device=device, dtype=means3D.dtype)], dim=1)
    points_view = points_hom @ world_view
    tz = points_view[:, 2]

    points_clip = points_hom @ full_proj
    w_clip = points_clip[:, 3].clamp(min=1e-6)
    ndc = points_clip[:, :3] / w_clip.unsqueeze(1)

    px = ((ndc[:, 0] + 1.0) * W - 1.0) * 0.5 + screenspace_points[:, 0]
    py = ((ndc[:, 1] + 1.0) * H - 1.0) * 0.5 + screenspace_points[:, 1]

    # --- 3D covariance -> 2D screen-space covariance (EWA splatting Jacobian) ---
    # Shared with gaussian_renderer/_cuda_backend.py's antialiasing pre-compensation so both
    # backends use identical math (see utils/antialiasing.py).
    a0, b0, c0, _ = compute_screenspace_cov2d(
        means3D, pc.get_scaling, pc.get_rotation, viewpoint_camera, scaling_modifier)
    a = a0 + _LOW_PASS_FILTER
    b = b0
    c = c0 + _LOW_PASS_FILTER
    det = (a * c - b * b).clamp(min=1e-8)
    inv_a, inv_b, inv_c = c / det, -b / det, a / det

    # Mip-Splatting-style antialiasing (see utils/antialiasing.py): the +_LOW_PASS_FILTER
    # dilation above is a fixed per-Gaussian screen-space inflation applied for numerical
    # stability, but it uniformly blurs any Gaussian whose true footprint is smaller than ~1px
    # (thin wires, high-frequency stripe textures). Compensate by scaling opacity down by
    # sqrt(det(true_cov)/det(dilated_cov)) so sub-pixel Gaussians contribute proportionally
    # less instead of being rendered as if they were pixel-sized.
    aa_coef = mip_antialiasing_opacity_coef(a0, b0, c0) if antialiasing else None

    mid = 0.5 * (a + c)
    disc = (mid ** 2 - det).clamp(min=0.1)
    lambda_max = mid + torch.sqrt(disc)
    radius = torch.ceil(3.0 * torch.sqrt(lambda_max.clamp(min=0.0)))

    valid = (
        (tz > _NEAR_PLANE) & (radius > 0) &
        (px + radius >= 0) & (px - radius < W) &
        (py + radius >= 0) & (py - radius < H)
    )

    radii_full = torch.zeros(N, device=device)
    rendered = torch.zeros(3, H, W, device=device, dtype=means3D.dtype)

    if valid.any():
        idx = valid.nonzero(as_tuple=True)[0]
        radii_full[idx] = radius[idx]

        opacity = pc.get_opacity.squeeze(-1)[idx]
        if aa_coef is not None:
            opacity = opacity * aa_coef[idx]
        opacity = opacity.clamp(max=0.9999)
        campos = viewpoint_camera.camera_center
        dirs = means3D[idx] - campos
        dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp(min=1e-8)
        deg = pc.active_sh_degree if sh_degree_override is None else sh_degree_override
        shs_view = pc.get_features[idx].transpose(1, 2)  # (n,3,K)
        colors = torch.clamp_min(eval_sh(deg, shs_view, dirs) + 0.5, 0.0)  # (n,3)

        # depth-sort near -> far so sequential "over" compositing is front-to-back
        order = torch.argsort(tz[idx])
        idx, px_s, py_s = idx[order], px[idx][order], py[idx][order]
        inv_a_s, inv_b_s, inv_c_s = inv_a[idx], inv_b[idx], inv_c[idx]
        opacity_s, colors_s = opacity[order], colors[order]

        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=means3D.dtype),
            torch.arange(W, device=device, dtype=means3D.dtype), indexing="ij")

        n_valid = idx.shape[0]
        batch_size = max(1, min(n_valid, _TARGET_ELEMENTS_PER_BATCH // max(1, H * W)))
        T = torch.ones(H, W, device=device, dtype=means3D.dtype)

        for start in range(0, n_valid, batch_size):
            end = min(start + batch_size, n_valid)
            dx = xx.unsqueeze(0) - px_s[start:end].view(-1, 1, 1)
            dy = yy.unsqueeze(0) - py_s[start:end].view(-1, 1, 1)
            power = -0.5 * (dx * dx * inv_a_s[start:end].view(-1, 1, 1)
                             + dy * dy * inv_c_s[start:end].view(-1, 1, 1)
                             + 2 * dx * dy * inv_b_s[start:end].view(-1, 1, 1))
            alpha = (opacity_s[start:end].view(-1, 1, 1) * torch.exp(power)).clamp(max=0.9999)

            ones = torch.ones(1, H, W, device=device, dtype=means3D.dtype)
            trans_before_in_batch = torch.cumprod(
                torch.cat([ones, 1.0 - alpha[:-1]], dim=0), dim=0)
            weight = alpha * trans_before_in_batch * T.unsqueeze(0)  # (b,H,W)

            color_contribution = torch.einsum('bhw,bc->chw', weight, colors_s[start:end])
            rendered = rendered + color_contribution
            T = T * torch.prod(1.0 - alpha, dim=0)

        rendered = rendered + T.unsqueeze(0) * bg_color.view(3, 1, 1)
    else:
        rendered = rendered + bg_color.view(3, 1, 1)

    return {
        "render": rendered,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii_full > 0,
        "radii": radii_full,
    }
