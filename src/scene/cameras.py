#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# Adapted from graphdeco-inria/gaussian-splatting (scene/cameras.py). Deviation from the
# original: `image` may be None (width/height must be given instead). The original repo
# always has a ground-truth image because it renders train/test splits that both live in
# the COLMAP folder; here, private_set1 test poses (test_poses.csv) have no shipped
# ground-truth image at all (see CLAUDE.md section 2), so render_submission.py needs to
# build a Camera purely from pose + intrinsics.

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cpu",
                 width=None, height=None, znear=0.01, zfar=100.0):
        super().__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(f"[Camera] Could not use device '{data_device}' ({e}), falling back to cpu")
            self.data_device = torch.device("cpu")

        if image is not None:
            self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]
        else:
            if width is None or height is None:
                raise ValueError("Camera needs either `image` or explicit `width`/`height`.")
            self.original_image = None
            self.image_width = width
            self.image_height = height

        self.zfar = zfar
        self.znear = znear
        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(
            getWorld2View2(R, T, trans, scale)).transpose(0, 1).to(self.data_device)
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
        ).transpose(0, 1).to(self.data_device)
        self.full_proj_transform = self.world_view_transform.unsqueeze(0).bmm(
            self.projection_matrix.unsqueeze(0)).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
