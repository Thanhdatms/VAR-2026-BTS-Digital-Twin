#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# Adapted from graphdeco-inria/gaussian-splatting (train.py) for this competition's data
# layout. See CLAUDE.md / PLAN.md Phase 3. Deviations from the original:
#   - no COLMAP-folder eval/llffhold test split (doesn't exist here — see dataset_readers.py);
#     if the scene has a public_set-style `test/images/` + `test/test_poses.csv`, those are
#     used as an optional validation signal during training, otherwise training just logs
#     the train loss (true for every private_set1 scene).
#   - `--data_device` degrades from "cuda" to "cpu" automatically with a warning instead of
#     erroring, since the compute target was left open (CLAUDE.md section 3).

import os
import sys
from argparse import ArgumentParser
from random import randint

import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render, describe_backend
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim


def pick_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        print("[train] --data_device cuda requested but torch.cuda.is_available() is False "
              "-> falling back to cpu. The CPU rasterizer is a slow reference implementation "
              "(see gaussian_renderer/cpu_rasterizer.py) — not meant for full training runs. "
              "See CLAUDE.md section 3 / PLAN.md Phase 0.")
        return torch.device("cpu")
    return torch.device(requested)


def find_public_gt_images_dir(source_path):
    d = os.path.join(source_path, "test", "images")
    return d if os.path.isdir(d) else None


def validate(val_cameras, gaussians, background, iteration):
    psnrs = []
    with torch.no_grad():
        for cam in val_cameras:
            image = torch.clamp(render(cam, gaussians, background)["render"], 0.0, 1.0)
            psnrs.append(psnr(image.unsqueeze(0), cam.original_image.unsqueeze(0)).mean().item())
    mean_psnr = sum(psnrs) / len(psnrs)
    print(f"\n[iter {iteration}] validation PSNR over {len(psnrs)} held-out GT images: {mean_psnr:.2f}")
    return mean_psnr


def training(dataset, opt, pipe, save_iterations, val_interval,
             resolution_scale=1.0, init_point_limit=None):
    
    print("Training ")
    
    device = pick_device(dataset.data_device)
    describe_backend(device)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset.source_path, gaussians, model_path=dataset.model_path,
                  shuffle=True, device=device, resolution_scale=resolution_scale,
                  init_point_limit=init_point_limit)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    test_poses_csv = os.path.join(dataset.source_path, "test", "test_poses.csv")
    gt_images_dir = find_public_gt_images_dir(dataset.source_path)
    val_cameras = None
    if os.path.exists(test_poses_csv) and gt_images_dir is not None:
        val_cameras = [c for c in scene.getTestCamerasFromCSV(
            test_poses_csv, gt_images_dir, resolution_scale=resolution_scale)
            if c.original_image is not None]
        print(f"[train] found {len(val_cameras)} test poses with ground-truth images "
              f"(public_set-style scene) — will log validation PSNR every {val_interval} iters.")
    else:
        print("[train] no ground-truth test images available for this scene "
              "(expected for private_set1) — logging train loss only.")

    viewpoint_stack = []
    ema_loss_for_log = 0.0
    max_gaussians_warned = False
    progress_bar = tqdm(range(1, opt.iterations + 1), desc="Training")

    for iteration in progress_bar:
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render(viewpoint_cam, gaussians, background)
        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
            1.0 - ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))
        if opt.lambda_scale_reg > 0:
            loss = loss + opt.lambda_scale_reg * gaussians.get_scaling.amax(dim=1).mean()
        loss.backward()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "loss": f"{ema_loss_for_log:.5f}",
                    "#gaussians": gaussians.get_xyz.shape[0],
                })

            if iteration % 1000 == 0:
                # tqdm.write (not print) so this doesn't get overwritten by the progress bar's
                # own carriage-return redraw -- unlike set_postfix above, this leaves a
                # permanent line in the log, so #gaussians over time (a convergence proxy: it
                # should climb during densify_from_iter..densify_until_iter then plateau once
                # opacity/screen-size pruning balances further splitting) survives even when
                # stdout is captured to a file or a notebook cell that doesn't replay
                # in-place bar updates.
                tqdm.write(f"[iter {iteration}] loss={ema_loss_for_log:.5f} "
                           f"#gaussians={gaussians.get_xyz.shape[0]}")

            if iteration < opt.densify_until_iter:
                vis_filter = render_pkg["visibility_filter"]
                gaussians.max_radii2D[vis_filter] = torch.max(
                    gaussians.max_radii2D[vis_filter], render_pkg["radii"][vis_filter])
                gaussians.add_densification_stats(render_pkg["viewspace_points"], vis_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    n_gaussians = gaussians.get_xyz.shape[0]
                    skip_densify = bool(opt.max_gaussians) and n_gaussians >= opt.max_gaussians
                    if skip_densify and not max_gaussians_warned:
                        tqdm.write(f"[iter {iteration}] hit --max_gaussians={opt.max_gaussians} "
                                   f"({n_gaussians} Gaussians) -- skipping further clone/split to "
                                   f"avoid OOM, pruning continues.")
                        max_gaussians_warned = True
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_grad_abs_threshold,
                                                 0.005, scene.cameras_extent, size_threshold,
                                                 skip_densify=skip_densify)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

            if val_cameras and (iteration % val_interval == 0 or iteration == opt.iterations):
                validate(val_cameras, gaussians, background, iteration)

            if iteration in save_iterations or iteration == opt.iterations:
                tqdm.write(f"[iter {iteration}] saving checkpoint ({gaussians.get_xyz.shape[0]} gaussians)")
                scene.save(iteration)

    return scene


if __name__ == "__main__":
    parser = ArgumentParser(description="Per-scene 3D Gaussian Splatting training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 15000, 30000])
    parser.add_argument("--val_interval", type=int, default=1000)
    parser.add_argument("--resolution_scale", type=float, default=1.0,
                         help="Divide image width/height by this factor. Only for local CPU "
                              "smoke runs (see utils/camera_utils.py) -- leave at 1.0 for any "
                              "real training/submission run.")
    parser.add_argument("--init_point_limit", type=int, default=None,
                         help="Randomly subsample the initial point cloud to at most this many "
                              "points. Only for local CPU smoke runs (see scene/__init__.py) -- "
                              "leave unset for any real training run.")
    args = parser.parse_args()

    dataset = lp.extract(args)
    if not dataset.model_path:
        parser.error("--model_path is required (e.g. output/HCM0249)")

    training(dataset, op.extract(args), pp.extract(args), args.save_iterations, args.val_interval,
              resolution_scale=args.resolution_scale, init_point_limit=args.init_point_limit)
    print("Training complete.")
