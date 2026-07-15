#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# Uses the AbsGS-patched `diff_gaussian_rasterization_abs` CUDA extension (TY424/AbsGS, a fork
# of graphdeco-inria/diff-gaussian-rasterization implementing arXiv:2404.10484 "AbsGS: Recovering
# Fine Details for 3D Gaussian Splatting" -- see scripts/colab_setup.sh for the build step).
#
# Root cause this fixes: vanilla 3DGS densification decides whether to split a Gaussian by
# summing its per-pixel screen-space gradient with sign. A large Gaussian smearing across a thin
# wire or a high-frequency stripe pattern (BTS towers/roofs in this project) gets pixel gradients
# pointing in different directions that cancel out in that signed sum ("gradient collision"), so
# it never crosses densify_grad_threshold and stays one blurry Gaussian standing in for detail
# that needs many small ones.
#
# The AbsGS kernel additionally accumulates the ABSOLUTE VALUE of each pixel's contribution,
# which can't cancel. It reports both via a single (P,4) gradient tensor on the screen-space
# tensor: channels 0:2 are the normal (signed) gradient, channels 2:4 are the "homodirectional"
# (abs) gradient (see cuda_rasterizer/backward.cu in the fork: `dL_dmean2D` is a float4 per
# Gaussian, .x/.y accumulated with atomicAdd as before, .z/.w accumulated with
# atomicAdd(..., fabs(...))). scene/gaussian_model.py's add_densification_stats()/
# densify_and_prune() read both: normal gradient still gates cloning, abs gradient gates
# splitting.
#
# Preferred over _cuda_backend.py whenever built -- see gaussian_renderer/__init__.py's
# try/except chain. Falls back automatically (ImportError) if this extension isn't built, so it
# is safe to attempt unconditionally.

import math
import torch

from diff_gaussian_rasterization_abs import GaussianRasterizationSettings, GaussianRasterizer


def render_cuda_abs(viewpoint_camera, pc, bg_color: torch.Tensor, scaling_modifier=1.0, sh_degree_override=None):
    """Returns a dict with keys: render (3,H,W), viewspace_points, visibility_filter, radii.
    Same contract as gaussian_renderer/_cuda_backend.py's render_cuda(), except
    viewspace_points is (P,4) instead of (P,3) -- see module docstring."""
    # Must be (P,4): the rasterizer's backward returns a (P,4) gradient (normal + abs channels),
    # and PyTorch autograd requires a custom Function's returned gradient to match the shape of
    # the corresponding forward input exactly.
    screenspace_points = torch.zeros((pc.get_xyz.shape[0], 4), dtype=pc.get_xyz.dtype,
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

    # Third return value (per-Gaussian accumulated blend weight, "gs_w" upstream) feeds AbsGS's
    # optional weight-based pruning, which this project doesn't use -- discarded here.
    rendered_image, radii, _gs_w = rasterizer(
        means3D=pc.get_xyz,
        means2D=screenspace_points,
        shs=pc.get_features,
        colors_precomp=None,
        opacities=pc.get_opacity,
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
