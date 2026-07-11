#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# Ported from graphdeco-inria/gaussian-splatting (gaussian_renderer/__init__.py's render()),
# unmodified logic — only renamed/isolated so gaussian_renderer/__init__.py can dispatch to
# it (see that file's docstring). Requires the official `diff_gaussian_rasterization` CUDA
# extension to be installed (Colab GPU runtime only — see scripts/colab_setup.sh).

import math
import torch

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def render_cuda(viewpoint_camera, pc, bg_color: torch.Tensor, scaling_modifier=1.0,
                 sh_degree_override=None, antialiasing=False):
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype,
                                           requires_grad=True, device=pc.get_xyz.device)
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    settings_kwargs = dict(
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
    try:
        raster_settings = GaussianRasterizationSettings(antialiasing=antialiasing, **settings_kwargs)
    except TypeError:
        # Installed diff_gaussian_rasterization predates the antialiasing kwarg (merged into
        # the official repo mid-2024) -- fall back to the non-antialiased build rather than
        # hard-crashing. Rebuild the submodule (scripts/colab_setup.sh) to pick it up.
        if antialiasing:
            print("[render_cuda] WARNING: installed diff_gaussian_rasterization has no "
                  "'antialiasing' kwarg -- rebuild the submodule (scripts/colab_setup.sh) to "
                  "use it. Rendering without antialiasing for now.")
        raster_settings = GaussianRasterizationSettings(**settings_kwargs)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    rendered_image, radii = rasterizer(
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
