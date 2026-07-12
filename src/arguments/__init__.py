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
                        # (store_true). Any ParamGroup bool that defaults to True (none currently
                        # -- PipelineParams.antialiasing was the motivating case but now defaults
                        # to False, see its docstring) would make plain store_true a no-op, so
                        # generate a --no_<key> switch instead so it's actually possible to
                        # disable via CLI for A/B comparisons.
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
        # NOTE: train.py only actually applies this once iteration > opt.densify_until_iter --
        # every Gaussian starts sub-pixel-sized at init, so compensating opacity from iteration
        # 0 crushes the whole scene's visibility before densification can grow points to a
        # sane size (measured: tanked validation PSNR early in training). See train.py's
        # `use_antialiasing` for the exact gating and the numbers that motivated it.
        #
        # DEFAULTED OFF (2026-07-12 re-review): confirmed on a private_set1 run (224_928 init
        # points, densify_until_iter=20_000, antialiasing_window=5_000) that ramping antialiasing
        # in AFTER densify_until_iter is not just a transient dip -- validation PSNR collapsed
        # 20.24 -> 12.72 -> 13.51 -> 13.72 (iters 24000/26000/28000/30000) and train loss kept
        # climbing (0.05 -> 0.19 -> 0.22 -> 0.224), i.e. it never recovered by the end of
        # training. Root cause: once densify_until_iter passes, densify_and_prune stops running,
        # so there is no structural mechanism (prune/clone/split) left to react when the
        # opacity-compensation coefficient collapses toward 0 for the thin/near-degenerate
        # Gaussians that make up most of a converged scene -- gradient descent on existing
        # Gaussians alone can't absorb that scale of distribution shift in the few thousand
        # iterations left. This is distinct from (and much worse than) the periodic
        # opacity_reset_interval dips, which self-heal within ~1-2k iters because densification
        # is still active when they happen. Re-enable only after fixing the scheduling (e.g.
        # overlapping the ramp with the tail of the active densify_until_iter window instead of
        # starting after it) and re-validating -- until then, leave this False so
        # train_all.py / train.py runs are safe by default. Pass --antialiasing explicitly to
        # opt back in for testing.
        self.antialiasing = False
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
        # Extended from the upstream default (15_000): BTS towers/antenna panels/guy wires are
        # thin, low-SfM-coverage structures that need more densification cycles to spawn
        # enough small Gaussians to represent them sharply (vanilla settings tend to leave them
        # as few large, blurry Gaussians -- see the 2026-07-12 artifact review for the specific
        # failure images). Raising --iterations further (e.g. 40_000) for tower-heavy scenes is
        # recommended; if you do, also raise --position_lr_max_steps and --densify_until_iter
        # to match, or the position LR schedule / densification window will end early.
        self.densify_until_iter = 20_000
        # NOTE: an earlier version of this file set this to 0.00008 (2.5x more aggressive than
        # upstream's 0.0002) to densify thin structures harder. That grows the *whole* scene
        # faster, not just thin structures (the threshold has no notion of "this Gaussian is
        # part of a wire") -- compounding over more densification cycles, it OOM'd a 32GB GPU
        # by iteration ~5400/20000 on public_set/HCM0181 (224_928 init points), well before
        # densify_until_iter. 0.00015 is a smaller, safer step down from the upstream default;
        # max_gaussians below is a hard backstop regardless of how this is tuned.
        self.densify_grad_threshold = 0.00015
        # Hard cap on Gaussian count: densify_and_prune stops growing (but keeps pruning) past
        # this, so a hard/dense scene can't OOM and take down the rest of train_all.py's batch.
        # 6M is comfortably under the ~7.2M the original (pre-tuning) settings reached on
        # private_set1/HCM0249 without issue on this project's GPU -- lower it if your GPU has
        # less headroom, e.g. via `--max_gaussians 4000000`.
        self.max_gaussians = 9_000_000
        # PipelineParams.antialiasing (if enabled) is only actually applied in this trailing
        # window of training, ramped in rather than switched on all at once -- see train.py's
        # antialiasing_from_iter/antialiasing_progress. Two reasons, both observed in practice
        # on public_set/HCM0181:
        #   1. Flipping antialiasing on abruptly at densify_until_iter multiplies every
        #      Gaussian's opacity by a compensation factor it was never trained against,
        #      causing a sudden loss spike that then has to be re-optimized away.
        #   2. The 2D-covariance opacity compensation (utils/antialiasing.py) is recomputed
        #      from scratch every iteration for the *entire* point cloud in plain
        #      PyTorch/Python (it's view-dependent -- a different random camera every
        #      iteration -- so it can't be cached across iterations the way a per-iteration
        #      constant could be). On a multi-million-Gaussian scene that's real, unavoidable
        #      per-iteration overhead, so it's only paid for the last antialiasing_window
        #      iterations rather than the entire post-densification tail.
        self.antialiasing_window = 5_000
        self.antialiasing_ramp_iters = 2_500
        self.random_background = False
        # Previously hard-coded as literals in train.py's densify_and_prune call; pulled out
        # here so they can be tuned per scene without editing code (e.g. tightening
        # densify_max_screen_size helps prune the large, under-constrained Gaussians that cause
        # blur in far/sparsely-covered image corners).
        self.min_opacity_prune = 0.005
        self.densify_max_screen_size = 20
        super().__init__(parser, "Optimization Parameters")
