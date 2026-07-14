#
# New for this project (not part of graphdeco-inria/gaussian-splatting): per-pixel edge
# weight for the photometric loss, following EGGS (arXiv:2404.09105) -- give edge pixels
# (thin wires, panel rims) a higher L1 weight so their backprop gradient is less likely to
# get diluted/cancelled out before crossing densify_grad_threshold. See train.py /
# arguments/__init__.py OptimizationParams.lambda_edge.
#
# Deliberately Scharr gradient magnitude, not Canny: Canny needs two hysteresis thresholds
# that would have to be retuned per scene (HCM vs HNI lighting/contrast differ, and the two
# heavily-distorted scenes -- HNI0131/HNI0265 -- get resampled by undistort before this ever
# sees them, changing local contrast at edges). Gradient magnitude is threshold-free and
# already continuous, so blurring it is just smoothing an existing soft signal rather than
# reconstructing one out of a binarized/thinned Canny mask.

import cv2
import numpy as np
import torch


def compute_edge_weight(pil_image, blur_sigma=1.5, normalize_percentile=99.0):
    """Continuous [0,1] edge-strength map from an RGB PIL image, shape (1, H, W).

    Higher values = stronger local gradient (real edges: wire silhouettes, panel rims).
    Near-flat regions are ~0, leaving the loss there unchanged when used as
    `(1 + lambda_edge * edge_weight)`.
    """
    gray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(gx, gy)
    magnitude = cv2.GaussianBlur(magnitude, ksize=(0, 0), sigmaX=blur_sigma)

    denom = np.percentile(magnitude, normalize_percentile)
    if denom > 1e-6:
        magnitude = magnitude / denom
    weight = np.clip(magnitude, 0.0, 1.0)

    return torch.from_numpy(weight).float().unsqueeze(0)
