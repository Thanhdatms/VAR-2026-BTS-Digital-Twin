"""Train every scene under a dataset root one at a time (3DGS is per-scene, no weight
sharing across scenes — see CLAUDE.md section 5). Thin subprocess wrapper around
src/train.py so each scene gets a fresh process (avoids any cross-scene CUDA memory
fragmentation from a long-lived Python process).

Example (Colab GPU runtime, all 8 private_set1 scenes):
    python scripts/train_all.py \
        --dataset_root dataset/phase1/private_set1 \
        --output_root output/private_set1 \
        --data_device cuda
"""

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PY = os.path.join(REPO_ROOT, "src", "train.py")


def discover_scenes(dataset_root):
    return sorted(
        name for name in os.listdir(dataset_root)
        if os.path.isfile(os.path.join(dataset_root, name, "test", "test_poses.csv")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--scenes", nargs="*", default=None,
                         help="Train only these scene names; default trains every scene "
                              "found under --dataset_root")
    parser.add_argument("--data_device", default="cuda")
    parser.add_argument("--iterations", type=int, default=30_000)
    parser.add_argument("--skip_existing", action="store_true",
                         help="Skip scenes that already have a point_cloud checkpoint dir")
    parser.add_argument("train_py_args", nargs=argparse.REMAINDER,
                         help="Everything after `--` is forwarded verbatim to src/train.py")
    args = parser.parse_args()

    extra = args.train_py_args[1:] if args.train_py_args[:1] == ["--"] else args.train_py_args
    scenes = args.scenes or discover_scenes(args.dataset_root)
    print(f"Training {len(scenes)} scene(s): {scenes}")

    for scene in scenes:
        source_path = os.path.join(args.dataset_root, scene)
        model_path = os.path.join(args.output_root, scene)

        if args.skip_existing and os.path.isdir(os.path.join(model_path, "point_cloud")):
            print(f"[{scene}] checkpoint already exists, skipping")
            continue

        cmd = [sys.executable, TRAIN_PY,
               "--source_path", source_path,
               "--model_path", model_path,
               "--data_device", args.data_device,
               "--iterations", str(args.iterations)] + extra
        print(f"\n=== [{scene}] {' '.join(cmd)} ===")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
