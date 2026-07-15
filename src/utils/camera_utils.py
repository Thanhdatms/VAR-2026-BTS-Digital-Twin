#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# Adapted from graphdeco-inria/gaussian-splatting (utils/camera_utils.py): simplified to
# this project's needs (our images are already delivered pre-scaled — see CLAUDE.md 2 —
# so there's no dataset-driven reason for a `--resolution` knob like the original repo's).
#
# `resolution_scale` here exists for a different reason: quick local CPU smoke runs (see
# PLAN.md / COLAB.md "CPU runtime" path). gaussian_renderer/cpu_rasterizer.py is
# O(N_gaussians * H * W); shrinking H*W is the cheapest way to make a local sanity-check
# run finish in minutes instead of hours. It must stay 1.0 for any real (Colab GPU) run —
# test_poses.csv's fx/fy/cx/cy/width/height define the actual images being graded.

from PIL import Image
import numpy as np
from scene.cameras import Camera
from utils.edge_utils import compute_edge_weight
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal


def loadCam(cam_info, data_device="cpu", resolution_scale=1.0):
    image = None
    edge_weight = None
    width, height = cam_info.width, cam_info.height
    if cam_info.image_path:
        pil_image = Image.open(cam_info.image_path).convert("RGB")
        if resolution_scale != 1.0:
            width = round(pil_image.width / resolution_scale)
            height = round(pil_image.height / resolution_scale)
            pil_image = pil_image.resize((width, height))
        image = PILtoTorch(pil_image, pil_image.size)
        # Computed from the same (post-resolution_scale-resize) pil_image used for `image`,
        # so it stays pixel-aligned with gt_image at whatever resolution training actually
        # runs at (see CLAUDE.md / utils/edge_utils.py).
        edge_weight = compute_edge_weight(pil_image)
    elif resolution_scale != 1.0:
        width = round(width / resolution_scale)
        height = round(height / resolution_scale)

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                   FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                   image=image, image_name=cam_info.image_name, uid=cam_info.uid,
                   data_device=data_device, width=width, height=height,
                   edge_weight=edge_weight)


def cameraList_from_camInfos(cam_infos, data_device="cpu", resolution_scale=1.0):
    return [loadCam(c, data_device=data_device, resolution_scale=resolution_scale) for c in cam_infos]


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]

    return {
        'id': id,
        'img_name': camera.image_name,
        'width': camera.image_width,
        'height': camera.image_height,
        'position': pos.tolist(),
        'rotation': [x.tolist() for x in rot],
        'fx': fov2focal(camera.FoVx, camera.image_width),
        'fy': fov2focal(camera.FoVy, camera.image_height),
    }
