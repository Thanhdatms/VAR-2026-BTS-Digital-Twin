#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# Adapted from graphdeco-inria/gaussian-splatting (scene/dataset_readers.py) for this
# competition's data layout. See CLAUDE.md sections 2 and 5 for what changed and why:
#   - only the COLMAP loader is kept (no Blender/transforms.json path — never present here)
#   - train images are undistorted (SIMPLE_RADIAL -> PINHOLE) before CameraInfo is built,
#     via preprocess.undistort, so training uses the same camera model test_poses.csv assumes
#   - there is no train-folder llffhold/eval split: test cameras come from test_poses.csv
#     (readTestPosesCSV), not from holding out COLMAP-registered images
#   - depth supervision / NeRF-synthetic fields from the original CameraInfo were dropped:
#     this dataset ships neither depth maps nor Blender transforms

import os
import csv
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal, BasicPointCloud
from preprocess.undistort import undistort_scene
import numpy as np
from plyfile import PlyData, PlyElement


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: float
    FovX: float
    image_path: str
    image_name: str
    width: int
    height: int
    is_test: bool


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def _check_centered_principal_point(cx, cy, width, height, source, tol=2.0):
    # getProjectionMatrix (utils/graphics_utils.py) assumes a centered principal point —
    # true for every scene checked in this dataset (see CLAUDE.md 2.1), but silently wrong
    # if violated, so fail loudly instead of producing a subtly-misaligned render.
    if abs(cx - width / 2) > tol or abs(cy - height / 2) > tol:
        raise AssertionError(
            f"{source}: principal point ({cx}, {cy}) is not centered for a {width}x{height} "
            f"image. This pipeline's projection matrix assumes a centered principal point.")


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, valid_image_names):
    cam_infos = []
    for key in cam_extrinsics:
        extr = cam_extrinsics[key]
        if extr.name not in valid_image_names:
            # images.bin may contain more registered images than exist in train/images/
            # (the held-out test views) — see CLAUDE.md 2.2.
            continue

        intr = cam_intrinsics[extr.camera_id]
        height, width = intr.height, intr.width
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "PINHOLE":
            fx, fy, cx, cy = intr.params
        elif intr.model == "SIMPLE_PINHOLE":
            fx, cx, cy = intr.params
            fy = fx
        else:
            raise AssertionError(
                f"Unsupported camera model '{intr.model}' for {extr.name}. Only PINHOLE "
                f"cameras are expected here — run preprocess.undistort first (CLAUDE.md 2.1).")
        _check_centered_principal_point(cx, cy, width, height, extr.name)

        FovY = focal2fov(fy, height)
        FovX = focal2fov(fx, width)

        cam_infos.append(CameraInfo(
            uid=intr.id, R=R, T=T, FovY=FovY, FovX=FovX,
            image_path=os.path.join(images_folder, extr.name), image_name=extr.name,
            width=width, height=height, is_test=False))

    return cam_infos


def readTestPosesCSV(csv_path, images_folder=None):
    """Parse a competition test_poses.csv into CameraInfo objects.

    image_path/is_test carries a ground-truth image path only if images_folder is given and
    the file actually exists there (true for public_set/*/test/images, never true for
    private_set1). Callers must handle image_path == "" (no GT available).
    """
    cam_infos = []
    with open(csv_path, newline="") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            image_name = row["image_name"]
            qvec = np.array([float(row["qw"]), float(row["qx"]), float(row["qy"]), float(row["qz"])])
            T = np.array([float(row["tx"]), float(row["ty"]), float(row["tz"])])
            R = np.transpose(qvec2rotmat(qvec))
            fx, fy = float(row["fx"]), float(row["fy"])
            cx, cy = float(row["cx"]), float(row["cy"])
            width, height = int(row["width"]), int(row["height"])
            _check_centered_principal_point(cx, cy, width, height, f"{csv_path}:{image_name}")

            image_path = os.path.join(images_folder, image_name) if images_folder else ""
            if not image_path or not os.path.exists(image_path):
                image_path = ""

            cam_infos.append(CameraInfo(
                uid=idx, R=R, T=T, FovY=focal2fov(fy, height), FovX=focal2fov(fx, width),
                image_path=image_path, image_name=image_name,
                width=width, height=height, is_test=True))
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    vertex_element = PlyElement.describe(elements, 'vertex')
    PlyData([vertex_element]).write(path)


def readColmapSceneInfo(path, undistort=True, force_undistort=False):
    """path is a scene directory, e.g. dataset/phase1/private_set1/HCM0249."""
    sparse_path = os.path.join(path, "train", "sparse", "0")
    try:
        cam_extrinsics = read_extrinsics_binary(os.path.join(sparse_path, "images.bin"))
        cam_intrinsics = read_intrinsics_binary(os.path.join(sparse_path, "cameras.bin"))
    except FileNotFoundError:
        cam_extrinsics = read_extrinsics_text(os.path.join(sparse_path, "images.txt"))
        cam_intrinsics = read_intrinsics_text(os.path.join(sparse_path, "cameras.txt"))

    images_dir = os.path.join(path, "train", "images")
    if undistort:
        images_dir, new_camera = undistort_scene(path, force=force_undistort)
        cam_intrinsics = {new_camera.id: new_camera}

    valid_image_names = set(os.listdir(images_dir))
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
        images_folder=images_dir, valid_image_names=valid_image_names)
    train_cam_infos = sorted(cam_infos_unsorted, key=lambda x: x.image_name)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(sparse_path, "points3D.ply")
    bin_path = os.path.join(sparse_path, "points3D.bin")
    txt_path = os.path.join(sparse_path, "points3D.txt")
    if not os.path.exists(ply_path):
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except FileNotFoundError:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    pcd = fetchPly(ply_path)

    return SceneInfo(point_cloud=pcd,
                      train_cameras=train_cam_infos,
                      test_cameras=[],
                      nerf_normalization=nerf_normalization,
                      ply_path=ply_path)
