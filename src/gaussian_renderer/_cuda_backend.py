#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# Ported from graphdeco-inria/gaussian-splatting (gaussian_renderer/__init__.py's render()),
# mostly unmodified — only renamed/isolated so gaussian_renderer/__init__.py can dispatch to
# it (see that file's docstring). Requires the official `diff_gaussian_rasterization` CUDA
# extension to be installed (Colab GPU runtime only — see scripts/colab_setup.sh).
#
# One deliberate addition: GaussianRasterizationSettings in the vanilla official build has no
# 'antialiasing' field (verified by reading diff_gaussian_rasterization/__init__.py directly —
# that API belongs to the separate autonomousvision/mip-splatting fork, which needs its own
# rasterizer submodule and persistent per-Gaussian 3D filter state, not just a flag). Instead,
# the `antialiasing` opacity pre-compensation is computed here in Python before calling the
# unmodified rasterizer — see utils/antialiasing.py for why/what.

import math
import torch

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from utils.antialiasing import compute_screenspace_cov2d, mip_antialiasing_opacity_coef


def render_cuda(viewpoint_camera, pc, bg_color: torch.Tensor, scaling_modifier=1.0,
                 sh_degree_override=None, antialiasing=0.0):
    """antialiasing: 0.0/False disables it (no extra compute); 1.0/True is full compensation;
    any value in between linearly blends opacity toward the fully-compensated result (see
    utils/antialiasing.py and OptimizationParams.antialiasing_ramp_iters for why train.py
    ramps this instead of switching it on all at once)."""
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype,
                                           requires_grad=True, device=pc.get_xyz.device)
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree if sh_degree_override is None else sh_degree_override,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    opacities = pc.get_opacity
    strength = float(antialiasing)
    if strength > 0.0:
        # detach(): mip_antialiasing_opacity_coef involves sqrt(det0/det), whose gradient blows
        # up as det0 (the *undilated* 2D covariance determinant) approaches 0 -- which is
        # exactly what happens for the thin, disk-like Gaussians a well-converged 3DGS scene is
        # full of (surfaces are represented as near-degenerate flat splats). Left attached, that
        # gradient reaches back through pc.get_scaling/get_rotation/get_xyz and can explode,
        # corrupting the optimizer state irrecoverably (observed: public_set/hcm0031 loss
        # jumped 0.04 -> 0.23 at iteration 26000 -- mid-ramp -- and never recovered by 30000,
        # unlike the opacity-reset dips elsewhere in the same log, which self-heal in <2000
        # iters). Detaching keeps this a pure per-render opacity multiplier: gradient still
        # flows into the learnable `opacity` parameter itself (which is what should adapt to
        # compensate), just not through this secondary, numerically fragile path into geometry.
        with torch.no_grad():
            a0, b0, c0, _ = compute_screenspace_cov2d(
                pc.get_xyz, pc.get_scaling, pc.get_rotation, viewpoint_camera, scaling_modifier)
            aa_coef = mip_antialiasing_opacity_coef(a0, b0, c0).unsqueeze(-1)
            if strength < 1.0:
                aa_coef = (1.0 - strength) + strength * aa_coef
        opacities = opacities * aa_coef

    rendered_image, radii = rasterizer(
        means3D=pc.get_xyz,
        means2D=screenspace_points,
        shs=pc.get_features,
        colors_precomp=None,
        opacities=opacities,
        scales=pc.get_scaling,
        rotations=pc.get_rotation,
        cov3D_precomp=None,
    )

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
