#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# Ported from graphdeco-inria/gaussian-splatting (utils/image_utils.py), unmodified.

import torch


def mse(img1, img2):
    # .reshape (not .view): callers may pass permuted/sliced tensors (e.g. evaluate.py loads
    # images via .permute(2,0,1)) that aren't contiguous, which .view() rejects.
    return ((img1 - img2) ** 2).reshape(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse_val = mse(img1, img2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse_val))
