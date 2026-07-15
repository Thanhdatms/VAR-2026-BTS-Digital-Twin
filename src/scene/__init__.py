#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# Adapted from graphdeco-inria/gaussian-splatting (scene/__init__.py) — simplified since
# this project only ever loads COLMAP scenes (no Blender path), never needs the
# cameras.json web-viewer export, and needs a way to build test cameras straight from
# test_poses.csv (see CLAUDE.md section 5 / PLAN.md Phase 1) rather than from a COLMAP
# train/test split.

import os
import random

import numpy as np

from scene.dataset_readers import readColmapSceneInfo, readTestPosesCSV
from scene.gaussian_model import GaussianModel
from utils.camera_utils import cameraList_from_camInfos
from utils.graphics_utils import BasicPointCloud
from utils.system_utils import mkdir_p, searchForMaxIteration


def _subsample_pcd(pcd: BasicPointCloud, limit: int, seed: int = 0) -> BasicPointCloud:
    n = pcd.points.shape[0]
    if n <= limit:
        return pcd
    idx = np.random.default_rng(seed).choice(n, size=limit, replace=False)
    return BasicPointCloud(points=pcd.points[idx], colors=pcd.colors[idx], normals=pcd.normals[idx])


class Scene:
    def __init__(self, source_path, gaussians: GaussianModel, model_path=None,
                 load_iteration=None, shuffle=True, device="cpu", undistort=True,
                 resolution_scale=1.0, init_point_limit=None):
        """
        source_path: scene dir, e.g. dataset/phase1/private_set1/HCM0249
        model_path: where checkpoints are written/read, e.g. output/HCM0249 (None = don't
            persist anything, useful for quick smoke tests)
        load_iteration: None = initialize from the COLMAP point cloud (fresh training);
            -1 = load the latest checkpoint found under model_path; N = load that iteration.
        resolution_scale: only meant for local CPU smoke runs (see utils/camera_utils.py
            docstring) — must be 1.0 for any real training/submission run.
        init_point_limit: randomly subsample the initial COLMAP point cloud to at most this
            many points. Only meant for local CPU smoke runs, where gaussian_renderer's
            O(N*H*W) reference rasterizer makes the full ~100k-200k point cloud impractically
            slow (PLAN.md Phase 0). Leave None for any real training run — subsampling
            throws away real reconstructed geometry.
        """
        self.source_path = source_path
        self.model_path = model_path
        self.gaussians = gaussians
        self.device = device
        self.loaded_iter = None

        self.scene_info = readColmapSceneInfo(source_path, undistort=undistort)
        self.cameras_extent = self.scene_info.nerf_normalization["radius"]

        self.train_cameras = cameraList_from_camInfos(
            self.scene_info.train_cameras, data_device=device, resolution_scale=resolution_scale)
        if shuffle:
            random.shuffle(self.train_cameras)

        if load_iteration is not None and model_path is not None:
            ckpt_dir = os.path.join(model_path, "point_cloud")
            self.loaded_iter = (searchForMaxIteration(ckpt_dir) if load_iteration == -1
                                 else load_iteration)
            ply_path = os.path.join(ckpt_dir, f"iteration_{self.loaded_iter}", "point_cloud.ply")
            print(f"Loading checkpoint: {ply_path}")
            self.gaussians.load_ply(ply_path, device=device)
        else:
            pcd = self.scene_info.point_cloud
            if init_point_limit is not None:
                pcd = _subsample_pcd(pcd, init_point_limit)
                print(f"[Scene] subsampled init point cloud to {pcd.points.shape[0]} points "
                      f"(init_point_limit={init_point_limit}) -- smoke-test mode only.")
            self.gaussians.create_from_pcd(pcd, self.cameras_extent, device=device)

    def getTrainCameras(self):
        return self.train_cameras

    def getTestCamerasFromCSV(self, csv_path, images_folder=None, resolution_scale=1.0):
        """Build Camera objects straight from a test_poses.csv (see CLAUDE.md section 5)."""
        cam_infos = readTestPosesCSV(csv_path, images_folder=images_folder)
        return cameraList_from_camInfos(cam_infos, data_device=self.device, resolution_scale=resolution_scale)

    def save(self, iteration):
        if self.model_path is None:
            raise ValueError("Scene was created without model_path; nothing to save to.")
        point_cloud_path = os.path.join(self.model_path, "point_cloud", f"iteration_{iteration}")
        mkdir_p(point_cloud_path)
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
