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
import random
import sys
from argparse import ArgumentParser

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
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


def set_seed(seed):
    # Seeds every RNG this pipeline touches at the Python/PyTorch level: train-camera shuffle
    # order (scene/__init__.py's random.shuffle), per-iteration camera sampling order (below),
    # and the random offsets densify_and_split draws when splitting a Gaussian in two
    # (scene/gaussian_model.py's torch.normal). This does NOT make CUDA training bit-exact
    # reproducible, though: the diff_gaussian_rasterization backward pass accumulates
    # per-Gaussian gradients via atomicAdd across parallel CUDA threads, and that summation
    # order (hence float rounding) isn't controlled by any seed -- a documented limitation of
    # the official kernel (no determinism flag exists). Over 10000s of iterations that's
    # enough to occasionally flip which Gaussians land on the wrong side of the
    # opacity/screen-size prune thresholds, so a ~1 PSNR unit spread between "identical" seeded
    # runs is expected. Seeding is still worth it -- it removes the larger, fully-avoidable
    # noise from unseeded camera order / split randomness when A/B-testing hyperparameters.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate(val_cameras, gaussians, background, iteration, antialiasing=False, train_loss=None):
    psnrs = []
    with torch.no_grad():
        for cam in val_cameras:
            image = torch.clamp(
                render(cam, gaussians, background, antialiasing=antialiasing)["render"], 0.0, 1.0)
            psnrs.append(psnr(image.unsqueeze(0), cam.original_image.unsqueeze(0)).mean().item())
    mean_psnr = sum(psnrs) / len(psnrs)
    # train_loss is the EMA train loss at this iteration (see ema_loss_for_log in the training
    # loop) -- printed alongside validation PSNR so you can see both on one line instead of
    # having to cross-reference the tqdm postfix, e.g. to tell a PSNR dip (like the
    # opacity-reset-induced ones, see earlier discussion) apart from a genuine regression.
    loss_str = f", train loss(ema): {train_loss:.5f}" if train_loss is not None else ""
    # tqdm.write (not print) so this doesn't break/duplicate the progress bar line -- same
    # convention as the checkpoint-save message below, which is why that one prints cleanly.
    tqdm.write(f"[iter {iteration}] Gaussian: {gaussians.get_xyz.shape[0]}{loss_str}, "
               f"validation PSNR over {len(psnrs)} held-out GT images: {mean_psnr:.2f}")
    return mean_psnr


def training(dataset, opt, pipe, save_iterations, val_interval,
             resolution_scale=1.0, init_point_limit=None, seed=0):

    print("Training ")

    if seed is not None:
        set_seed(seed)

    device = pick_device(dataset.data_device)

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

    # Antialiasing (utils/antialiasing.py) is only applied in a trailing window of training,
    # ramped in rather than switched on all at once (see OptimizationParams.antialiasing_window
    # / antialiasing_ramp_iters docstring). Two reasons, both observed on public_set/HCM0181:
    #   1. Every Gaussian starts sub-pixel-sized at init (nearest-neighbor-distance-based
    #      scale) and opacity is fit for 20k+ iterations assuming no compensation -- flipping
    #      it on abruptly multiplies opacity by a factor the model was never trained against,
    #      causing a sudden loss spike that then has to be optimized away.
    #   2. It's real per-iteration extra compute (a fresh, view-dependent 2D-covariance pass
    #      over the whole point cloud every iteration -- can't be cached across iterations
    #      since a different random camera is sampled each time), so it measurably slows
    #      training for as long as it's active. Restricting it to the last antialiasing_window
    #      iterations (instead of the whole post-densification tail) bounds that cost.
    antialiasing_from_iter = max(opt.densify_until_iter, opt.iterations - opt.antialiasing_window)

    viewpoint_stack = []
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(1, opt.iterations + 1), desc="Training")

    for iteration in progress_bar:
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack) - 1))

        if pipe.antialiasing and iteration > antialiasing_from_iter:
            antialiasing_progress = min(
                1.0, (iteration - antialiasing_from_iter) / max(1, opt.antialiasing_ramp_iters))
        else:
            antialiasing_progress = 0.0
        render_pkg = render(viewpoint_cam, gaussians, background, antialiasing=antialiasing_progress)
        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
            1.0 - ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))
        loss.backward()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "loss": f"{ema_loss_for_log:.5f}",
                    "#gaussians": gaussians.get_xyz.shape[0],
                })

            if iteration < opt.densify_until_iter:
                vis_filter = render_pkg["visibility_filter"]
                gaussians.max_radii2D[vis_filter] = torch.max(
                    gaussians.max_radii2D[vis_filter], render_pkg["radii"][vis_filter])
                gaussians.add_densification_stats(render_pkg["viewspace_points"], vis_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = (opt.densify_max_screen_size
                                       if iteration > opt.opacity_reset_interval else None)
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity_prune,
                                                 scene.cameras_extent, size_threshold,
                                                 max_gaussians=opt.max_gaussians)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

            if val_cameras and (iteration % val_interval == 0 or iteration == opt.iterations):
                validate(val_cameras, gaussians, background, iteration,
                         antialiasing=antialiasing_progress, train_loss=ema_loss_for_log)

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
    parser.add_argument("--val_interval", type=int, default=5000)
    parser.add_argument("--resolution_scale", type=float, default=1.0,
                         help="Divide image width/height by this factor. Only for local CPU "
                              "smoke runs (see utils/camera_utils.py) -- leave at 1.0 for any "
                              "real training/submission run.")
    parser.add_argument("--init_point_limit", type=int, default=None,
                         help="Randomly subsample the initial point cloud to at most this many "
                              "points. Only for local CPU smoke runs (see scene/__init__.py) -- "
                              "leave unset for any real training run.")
    parser.add_argument("--seed", type=int, default=0,
                         help="Seed for Python/numpy/torch RNGs (camera order, split noise) -- "
                              "cuts most but not all run-to-run variance, see set_seed()'s "
                              "docstring for why it isn't fully deterministic on CUDA. Pass "
                              "different values to deliberately probe run-to-run spread.")
    args = parser.parse_args()

    dataset = lp.extract(args)
    if not dataset.model_path:
        parser.error("--model_path is required (e.g. output/HCM0249)")

    training(dataset, op.extract(args), pp.extract(args), args.save_iterations, args.val_interval,
              resolution_scale=args.resolution_scale, init_point_limit=args.init_point_limit,
              seed=args.seed)
    print("Training complete.")
