#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# Adapted from graphdeco-inria/gaussian-splatting (arguments/__init__.py): trimmed to the
# options this baseline actually uses (dropped depth-regularization / exposure-compensation
# flags added in later upstream revisions, since this dataset ships neither depth maps nor
# per-image exposure metadata — see CLAUDE.md section 5).

import os
from argparse import ArgumentParser


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, "-" + key[0:1], default=value, action="store_true")
                else:
                    group.add_argument("--" + key, "-" + key[0:1], default=value, type=t)
            else:
                if t == bool:
                    if value:
                        # Ported from upstream, which only ever had default=False bool flags
                        # (store_true). PipelineParams.antialiasing defaults to True, where
                        # store_true would be a no-op (already True, and passing --antialiasing
                        # can't turn it off) -- generate a --no_<key> switch instead so it's
                        # actually possible to disable via CLI for A/B comparisons.
                        group.add_argument("--no_" + key, dest=key, default=True, action="store_false")
                    else:
                        group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg_key, arg_value in vars(args).items():
            if arg_key in vars(self) or ("_" + arg_key) in vars(self):
                setattr(group, arg_key, arg_value)
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._white_background = False
        self.data_device = "cuda"
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        # Mip-Splatting-style opacity pre-compensation for the rasterizer's fixed screen-space
        # dilation -- computed in Python (utils/antialiasing.py) before calling the vanilla
        # diff-gaussian-rasterization kernel, NOT a kwarg on the kernel itself (verified by
        # reading its source: GaussianRasterizationSettings has no 'antialiasing' field in the
        # official graphdeco-inria build that colab_setup.sh clones -- that field belongs to
        # the separate autonomousvision/mip-splatting fork, which needs a different submodule
        # and persistent per-Gaussian state; we intentionally don't depend on it). Without this,
        # every Gaussian's screen-space covariance gets a fixed +0.3px dilation for numerical
        # stability regardless of how small its true footprint is; that fixed dilation is what
        # blurs out sub-pixel/high-frequency content (power line wires, corrugated-roof stripe
        # patterns — see the 2026-07-12 artifact review). With antialiasing=True, opacity is
        # scaled by sqrt(det(true_cov)/det(dilated_cov)) per Gaussian before rasterizing, so
        # thin/high-freq primitives don't get uniformly smeared. MUST be set identically at
        # train time and render time (a mismatch shifts brightness/opacity systematically) --
        # train.py and render_submission.py both read this same flag for that reason.
        self.antialiasing = True
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        # Extended from the upstream default (15_000) and densify_grad_threshold halved: BTS
        # towers/antenna panels/guy wires are thin, low-SfM-coverage structures that need more
        # densification cycles and a lower gradient bar to spawn enough small Gaussians to
        # represent them sharply (vanilla settings tend to leave them as few large, blurry
        # Gaussians -- see the 2026-07-12 artifact review for the specific failure images).
        # Raising --iterations further (e.g. 40_000) for tower-heavy scenes is recommended;
        # if you do, also raise --position_lr_max_steps and --densify_until_iter to match, or
        # the position LR schedule / densification window will end early relative to training.
        self.densify_until_iter = 20_000
        self.densify_grad_threshold = 0.00008
        self.random_background = False
        # Previously hard-coded as literals in train.py's densify_and_prune call; pulled out
        # here so they can be tuned per scene without editing code (e.g. tightening
        # densify_max_screen_size helps prune the large, under-constrained Gaussians that cause
        # blur in far/sparsely-covered image corners).
        self.min_opacity_prune = 0.005
        self.densify_max_screen_size = 20
        super().__init__(parser, "Optimization Parameters")
