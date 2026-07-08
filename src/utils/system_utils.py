#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# Ported from graphdeco-inria/gaussian-splatting (utils/system_utils.py), trimmed to the
# two helpers this project actually uses.

import os
import errno


def mkdir_p(folder_path):
    try:
        os.makedirs(folder_path, exist_ok=True)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(folder_path):
            pass
        else:
            raise


def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)
