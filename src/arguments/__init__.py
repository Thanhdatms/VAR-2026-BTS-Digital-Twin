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
        # percent_dense/densify_grad_threshold/densify_until_iter lowered from upstream 3DGS
        # defaults (0.01 / 0.0002 / 15_000) to fight blur on thin structures (BTS wires) and
        # high-frequency stripe patterns (checkered/corrugated roofs) -- see CLAUDE.md /
        # conversation notes. Root cause (AbsGS, arXiv:2404.10484 / FreGS, CVPR 2024): the
        # vanilla positional-gradient threshold under-splits Gaussians in these regions because
        # per-pixel gradients under one large Gaussian cancel out ("gradient collision") before
        # ever tripping densify_grad_threshold, so one blurry Gaussian keeps standing in for
        # detail that needs many small ones. Lowering the threshold + percent_dense makes
        # splitting trigger earlier/for smaller primitives; extending densify_until_iter gives
        # more densification passes before the position LR decays. Start conservative
        # (0.00015) and lower further (e.g. 0.0001) if thin/high-freq regions are still blurry
        # after re-evaluating on public_set.
        self.percent_dense = 0.005
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 20_000
        self.densify_grad_threshold = 0.00015
        self.random_background = False
        # Optional heuristic regularization (off by default, not from a specific paper): mean
        # per-Gaussian max-scale penalty, discouraging large "sheet" Gaussians from smearing
        # across multiple stripes/wire segments instead of splitting to represent them. Sweep
        # small values (e.g. 0.001-0.01) on public_set before trusting it -- an untested weight
        # can just as easily blur small-scale detail by over-shrinking everything.
        self.lambda_scale_reg = 0.0
        super().__init__(parser, "Optimization Parameters")
