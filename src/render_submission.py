"""Render competition test poses from trained checkpoints into the submission folder layout:

    <output_root>/<scene_name>/<image_name>   # image_name exactly as given in test_poses.csv

Per the README shipped with the dataset, test_poses.csv's `width`/`height` columns are the
"desired render resolution" and `image_name` is the "image filename" -- so both the output
resolution and the output filename must match those columns exactly (not an invented naming
scheme). Since scoring averages per-scene over ALL test poses (CLAUDE.md sec 1), a missing
render for even one pose drags the whole scene's score down, so this script fails loudly
(non-zero exit) if any expected image wasn't written, instead of silently submitting a
partial scene.

Example:
    python src/render_submission.py \
        --dataset_root dataset/phase1/private_set1 \
        --models_root output \
        --output_root submission/private_set1
"""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from PIL import Image

from scene.gaussian_model import GaussianModel
from scene.dataset_readers import readTestPosesCSV
from utils.camera_utils import cameraList_from_camInfos
from utils.system_utils import searchForMaxIteration
from gaussian_renderer import render, describe_backend


def pick_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        print("[render_submission] --data_device cuda requested but unavailable -> using cpu "
              "(will be slow for full-resolution renders, see gaussian_renderer/cpu_rasterizer.py).")
        return torch.device("cpu")
    return torch.device(requested)


def find_checkpoint(model_path, iteration=None):
    pc_dir = os.path.join(model_path, "point_cloud")
    it = searchForMaxIteration(pc_dir) if iteration is None else iteration
    return os.path.join(pc_dir, f"iteration_{it}", "point_cloud.ply"), it


def discover_scenes(dataset_root):
    scenes = []
    for name in sorted(os.listdir(dataset_root)):
        p = os.path.join(dataset_root, name)
        if os.path.isfile(os.path.join(p, "test", "test_poses.csv")):
            scenes.append(p)
    return scenes


def save_image(arr, path):
    img = Image.fromarray(arr)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        # quality=95 keeps JPEG's own encoding noise floor (~51dB self-PSNR, measured against
        # the dataset's ground-truth JPEGs) far below actual render error (~20dB PSNR from
        # training logs), so it's visually/metrically indistinguishable from quality=100 while
        # roughly halving file size. PIL's default quality=75 would visibly hurt LPIPS/SSIM/PSNR.
        img.save(path, quality=95)
    else:
        img.save(path)


def _supersampled_cam_infos(cam_infos, factor):
    """Scale width/height by `factor` and drop image_path so Camera builds at the
    supersampled resolution instead of latching onto a GT image's native size (Camera.__init__
    ignores the width/height args whenever an image is attached -- see scene/cameras.py)."""
    if factor <= 1:
        return cam_infos
    return [ci._replace(width=ci.width * factor, height=ci.height * factor, image_path="")
            for ci in cam_infos]


def _box_downsample(arr, factor):
    """Exact box-filter downsample by an integer factor (area-average over each factor x
    factor block). This is the correction Mip-Splatting motivates at the training/rasterizer
    level (2D box filter instead of a dilation-only filter) -- applied here as a cheap
    render-time post-process instead, so it needs no rasterizer changes: render at N x the
    target resolution, then average back down. Mitigates aliasing/moire on thin wires and
    high-frequency stripe patterns without touching training."""
    h, w, c = arr.shape
    h2, w2 = h // factor, w // factor
    arr = arr[: h2 * factor, : w2 * factor]
    return arr.reshape(h2, factor, w2, factor, c).mean(axis=(1, 3))


def render_scene(scene_source_path, model_path, output_root, sh_degree=3, iteration=None,
                  device=torch.device("cpu"), white_background=False, supersample=1):
    scene_name = os.path.basename(os.path.normpath(scene_source_path))
    ckpt_path, loaded_it = find_checkpoint(model_path, iteration)
    print(f"[{scene_name}] loading checkpoint iter {loaded_it}: {ckpt_path}")

    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(ckpt_path, device=device)

    csv_path = os.path.join(scene_source_path, "test", "test_poses.csv")
    gt_dir = os.path.join(scene_source_path, "test", "images")
    gt_dir = gt_dir if os.path.isdir(gt_dir) else None  # only public_set scenes have this

    cam_infos = readTestPosesCSV(csv_path, images_folder=gt_dir)
    render_cam_infos = _supersampled_cam_infos(cam_infos, supersample)
    cameras = cameraList_from_camInfos(render_cam_infos, data_device=device)

    bg_color = [1.0, 1.0, 1.0] if white_background else [0.0, 0.0, 0.0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    out_dir = os.path.join(output_root, scene_name)
    os.makedirs(out_dir, exist_ok=True)

    expected_names = [cam_info.image_name for cam_info in cam_infos]
    failures = []

    with torch.no_grad():
        for cam_info, cam in zip(cam_infos, cameras):
            try:
                image = torch.clamp(render(cam, gaussians, background)["render"], 0.0, 1.0)
                arr = (image.permute(1, 2, 0).cpu().numpy() * 255.0)
                if supersample > 1:
                    arr = _box_downsample(arr, supersample)
                arr = arr.round().clip(0, 255).astype(np.uint8)
                if arr.shape[0] != cam_info.height or arr.shape[1] != cam_info.width:
                    raise ValueError(
                        f"render size {arr.shape[1]}x{arr.shape[0]} != test_poses.csv "
                        f"{cam_info.width}x{cam_info.height} for {cam_info.image_name}")
                save_image(arr, os.path.join(out_dir, cam_info.image_name))
            except Exception as e:
                print(f"[{scene_name}] FAILED to render {cam_info.image_name}: {e}")
                failures.append(cam_info.image_name)

    written = set(os.listdir(out_dir))
    missing = [name for name in expected_names if name not in written]
    if missing:
        raise RuntimeError(
            f"[{scene_name}] incomplete submission: {len(missing)}/{len(expected_names)} test "
            f"poses have no output image (score is a per-scene average over ALL test poses -- "
            f"a missing image drags the whole scene down, see CLAUDE.md sec 1): {missing}")

    print(f"[{scene_name}] wrote {len(expected_names)} renders -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=str, required=True,
                         help="e.g. dataset/phase1/private_set1")
    parser.add_argument("--models_root", type=str, required=True,
                         help="Root of trained checkpoints; expects "
                              "<models_root>/<scene_name>/point_cloud/iteration_*/point_cloud.ply")
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--scene", type=str, default=None,
                         help="Render only this scene name; default renders every scene under --dataset_root")
    parser.add_argument("--iteration", type=int, default=None,
                         help="Checkpoint iteration to load; default = latest available per scene")
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--data_device", type=str, default="cuda")
    parser.add_argument("--supersample", type=int, default=2,
                         help="Render at this integer factor x the target width/height, then "
                              "box-downsample back down before saving. Mitigates aliasing/blur "
                              "on thin wires and high-frequency stripe patterns (see CLAUDE.md "
                              "conversation notes). Costs supersample^2 render compute -- pass 1 "
                              "to disable. Only affects render_submission.py, not training.")
    args = parser.parse_args()

    device = pick_device(args.data_device)
    describe_backend(device)
    scene_paths = ([os.path.join(args.dataset_root, args.scene)] if args.scene
                    else discover_scenes(args.dataset_root))

    incomplete_scenes = []
    for scene_path in scene_paths:
        scene_name = os.path.basename(os.path.normpath(scene_path))
        try:
            render_scene(scene_path, os.path.join(args.models_root, scene_name), args.output_root,
                         sh_degree=args.sh_degree, iteration=args.iteration, device=device,
                         white_background=args.white_background, supersample=args.supersample)
        except RuntimeError as e:
            print(e)
            incomplete_scenes.append(scene_name)

    if incomplete_scenes:
        sys.exit(f"[render_submission] {len(incomplete_scenes)} scene(s) incomplete, "
                  f"fix before submitting: {incomplete_scenes}")
