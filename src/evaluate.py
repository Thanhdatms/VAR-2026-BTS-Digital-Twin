"""Score render_submission.py output against ground-truth test images.

Only usable where ground truth actually exists — i.e. public_set (see CLAUDE.md section 2:
private_set1 ships no test images at all, by design, since it's what the organizer grades
on). Use this against public_set to sanity-check the pipeline and tune hyperparameters
before submitting private_set1 renders blind.

Score per image, per the competition brief:
    psnr_norm = clamp(PSNR / PSNR_max, 0, 1)
    score = 0.4*(1 - LPIPS) + 0.3*SSIM + 0.3*psnr_norm
Final score = mean over all scenes' mean-per-image score.

`--psnr_max` has no official value in the brief provided to this repo (see CLAUDE.md
section 5) -- it is a required CLI argument here on purpose, not a silently-assumed default,
so a wrong/placeholder number can't leak into a reported score unnoticed.

Example:
    python src/evaluate.py \
        --renders_root submission/public_set \
        --dataset_root dataset/phase1/public_set \
        --psnr_max 50 \
        --report_path eval_report.json
"""

import os
import sys
import json
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from PIL import Image

from utils.image_utils import psnr as psnr_fn
from utils.loss_utils import ssim as ssim_fn


def load_image_tensor(path, device):
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).to(device)


def get_lpips_model(device):
    import lpips
    model = lpips.LPIPS(net="alex").to(device)
    model.eval()
    return model


def evaluate_scene(render_dir, gt_dir, lpips_model, device, psnr_max):
    # render_submission.py now names each output file after test_poses.csv's `image_name`
    # directly, so the render and ground-truth filenames are identical -- no index_map needed.
    gt_names = sorted(f for f in os.listdir(gt_dir) if not f.startswith("."))
    missing = [name for name in gt_names if not os.path.exists(os.path.join(render_dir, name))]
    if missing:
        raise FileNotFoundError(
            f"{render_dir}: missing render for {len(missing)}/{len(gt_names)} test poses: {missing}")

    per_image = []
    for name in gt_names:
        render_path = os.path.join(render_dir, name)
        gt_path = os.path.join(gt_dir, name)

        pred = load_image_tensor(render_path, device)
        gt = load_image_tensor(gt_path, device)
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch on {name}: render {tuple(pred.shape)} "
                              f"vs ground truth {tuple(gt.shape)}")

        with torch.no_grad():
            psnr_val = psnr_fn(pred.unsqueeze(0), gt.unsqueeze(0)).item()
            ssim_val = ssim_fn(pred.unsqueeze(0), gt.unsqueeze(0)).item()
            lpips_val = lpips_model(pred.unsqueeze(0) * 2 - 1, gt.unsqueeze(0) * 2 - 1).item()

        psnr_norm = min(max(psnr_val / psnr_max, 0.0), 1.0)
        score = 0.4 * (1 - lpips_val) + 0.3 * ssim_val + 0.3 * psnr_norm

        per_image.append({"render": name, "gt": name, "psnr": psnr_val,
                           "ssim": ssim_val, "lpips": lpips_val, "psnr_norm": psnr_norm,
                           "score": score})

    n = len(per_image)
    means = {k: sum(r[k] for r in per_image) / n
             for k in ("psnr", "ssim", "lpips", "psnr_norm", "score")}
    return means, per_image


def discover_scene_pairs(renders_root, dataset_root):
    pairs = []
    for name in sorted(os.listdir(renders_root)):
        render_dir = os.path.join(renders_root, name)
        if not os.path.isdir(render_dir):
            continue
        gt_dir = os.path.join(dataset_root, name, "test", "images")
        if os.path.isdir(gt_dir):
            pairs.append((name, render_dir, gt_dir))
    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renders_root", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--psnr_max", type=float, required=True,
                         help="Normalization ceiling for PSNR (no official value in the "
                              "brief provided to this repo -- see CLAUDE.md section 5).")
    parser.add_argument("--report_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    lpips_model = get_lpips_model(device)

    pairs = discover_scene_pairs(args.renders_root, args.dataset_root)
    if not pairs:
        raise SystemExit(f"No scenes with ground truth found: checked every subdirectory of "
                          f"{args.renders_root} against {{--dataset_root}}/<scene>/test/images")

    report = {"psnr_max": args.psnr_max, "scenes": {}}
    for name, render_dir, gt_dir in pairs:
        print(f"Evaluating {name}...")
        means, per_image = evaluate_scene(render_dir, gt_dir, lpips_model, device, args.psnr_max)
        report["scenes"][name] = {"mean": means, "per_image": per_image}
        print(f"  PSNR={means['psnr']:.2f}  SSIM={means['ssim']:.4f}  LPIPS={means['lpips']:.4f}  "
              f"score={means['score']:.4f}")

    scene_scores = [s["mean"]["score"] for s in report["scenes"].values()]
    final_score = sum(scene_scores) / len(scene_scores)
    report["final_score"] = final_score
    print(f"\nFinal score (mean over {len(scene_scores)} scenes): {final_score:.4f}")

    if args.report_path:
        os.makedirs(os.path.dirname(args.report_path) or ".", exist_ok=True)
        with open(args.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to {args.report_path}")
