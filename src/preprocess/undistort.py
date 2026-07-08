"""Undistort COLMAP SIMPLE_RADIAL train images to PINHOLE, so training uses the same
camera model that test_poses.csv assumes (fx, fy, cx, cy only, no distortion term).

See CLAUDE.md section 2.1 for why this step is required: all 8 private_set1 scenes are
calibrated as SIMPLE_RADIAL (params = [f, cx, cy, k]); two scenes (HNI0131, HNI0265) have
strong distortion (k ~ -0.115) that is clearly visible at image borders if left untouched.

COLMAP's SIMPLE_RADIAL model (x_d = x_u * (1 + k*r^2), single coefficient) is numerically
identical to OpenCV's radial-only distortion model when given distCoeffs=[k, 0, 0, 0] with
the same camera matrix, so cv2.undistort can be used directly without needing the COLMAP CLI.
"""

import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UNDISTORTED_DIRNAME = "images_undistorted"

# Camera models that already match test_poses.csv's implicit PINHOLE assumption and need
# no pixel resampling — only a params reshape to (fx, fy, cx, cy).
_PASSTHROUGH_MODELS = {"PINHOLE", "SIMPLE_PINHOLE"}
_SUPPORTED_DISTORTED_MODELS = {"SIMPLE_RADIAL"}


class UndistortedCamera:
    """Minimal stand-in for colmap_loader.Camera after undistortion: always PINHOLE."""

    def __init__(self, id, width, height, fx, fy, cx, cy):
        self.id = id
        self.model = "PINHOLE"
        self.width = width
        self.height = height
        self.params = np.array([fx, fy, cx, cy], dtype=np.float64)


def _load_cameras(sparse_path):
    # Import kept local (not at module top-level): scene/dataset_readers.py imports this
    # module, and scene/colmap_loader.py's package (`scene`) in turn imports
    # scene/dataset_readers.py at package-init time -- a top-level import here would create
    # scene -> dataset_readers -> undistort -> scene (colmap_loader) circular import.
    from scene.colmap_loader import read_intrinsics_binary, read_intrinsics_text

    bin_path = os.path.join(sparse_path, "cameras.bin")
    txt_path = os.path.join(sparse_path, "cameras.txt")
    if os.path.exists(bin_path):
        return read_intrinsics_binary(bin_path)
    return read_intrinsics_text(txt_path)


def build_undistort_map(camera):
    """Return (K, dist_coeffs, new_camera) for a raw COLMAP camera.

    new_camera is an UndistortedCamera describing the PINHOLE model that the *output* pixels
    obey (same focal length / principal point — cv2.undistort with default newCameraMatrix=K
    resamples the image content but keeps K fixed, it does not crop/change intrinsics).
    """
    model = camera.model
    w, h = camera.width, camera.height

    if model in _PASSTHROUGH_MODELS:
        if model == "SIMPLE_PINHOLE":
            f, cx, cy = camera.params
            fx = fy = f
        else:  # PINHOLE
            fx, fy, cx, cy = camera.params
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(4, dtype=np.float64)
        new_camera = UndistortedCamera(camera.id, w, h, fx, fy, cx, cy)
        return K, dist, new_camera, False

    if model not in _SUPPORTED_DISTORTED_MODELS:
        raise NotImplementedError(
            f"Camera model '{model}' not handled by preprocess/undistort.py. "
            f"Only PINHOLE/SIMPLE_PINHOLE (passthrough) and SIMPLE_RADIAL (undistorted) "
            f"are supported for this dataset.")

    f, cx, cy, k = camera.params
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k, 0.0, 0.0, 0.0], dtype=np.float64)
    new_camera = UndistortedCamera(camera.id, w, h, f, f, cx, cy)
    return K, dist, new_camera, True


def undistort_scene(source_path, force=False, image_names=None):
    """Undistort every image in <source_path>/train/images into
    <source_path>/train/images_undistorted, caching the result on disk.

    Returns (images_dir_to_use, new_camera) where images_dir_to_use is the undistorted
    directory when the scene's camera needs undistortion, or the original images dir
    unchanged when the camera is already PINHOLE/SIMPLE_PINHOLE (no resampling needed).
    """
    sparse_path = os.path.join(source_path, "train", "sparse", "0")
    images_dir = os.path.join(source_path, "train", "images")
    cameras = _load_cameras(sparse_path)
    if len(cameras) != 1:
        raise NotImplementedError(
            f"{source_path}: expected exactly 1 camera per scene, found {len(cameras)}. "
            f"Multi-camera rigs are not handled by this baseline (see CLAUDE.md section 2.2).")
    camera = next(iter(cameras.values()))
    K, dist, new_camera, needs_undistort = build_undistort_map(camera)

    if not needs_undistort:
        return images_dir, new_camera

    out_dir = os.path.join(source_path, "train", UNDISTORTED_DIRNAME)
    os.makedirs(out_dir, exist_ok=True)

    names = image_names if image_names is not None else sorted(os.listdir(images_dir))
    names = [n for n in names if not n.startswith(".")]

    todo = []
    for name in names:
        dst = os.path.join(out_dir, name)
        if force or not os.path.exists(dst):
            todo.append(name)

    for name in todo:
        src = os.path.join(images_dir, name)
        img = cv2.imread(src, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {src}")
        undistorted = cv2.undistort(img, K, dist, None, K)
        dst = os.path.join(out_dir, name)
        ok = cv2.imwrite(dst, undistorted)
        if not ok:
            raise IOError(f"Failed to write undistorted image: {dst}")

    return out_dir, new_camera


def _main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_path", type=str,
                         help="Path to a single scene dir (contains train/, test/).")
    parser.add_argument("--root", type=str,
                         help="Process every scene subdirectory under this root instead.")
    parser.add_argument("--force", action="store_true",
                         help="Re-run undistortion even if cached output already exists.")
    args = parser.parse_args()

    if not args.source_path and not args.root:
        parser.error("Provide either --source_path or --root")

    scene_paths = []
    if args.source_path:
        scene_paths = [args.source_path]
    else:
        for name in sorted(os.listdir(args.root)):
            p = os.path.join(args.root, name)
            if os.path.isdir(os.path.join(p, "train", "sparse", "0")):
                scene_paths.append(p)

    for scene_path in scene_paths:
        scene_name = os.path.basename(os.path.normpath(scene_path))
        out_dir, new_camera = undistort_scene(scene_path, force=args.force)
        print(f"[{scene_name}] images dir -> {out_dir} | camera -> model=PINHOLE "
              f"fx={new_camera.params[0]:.3f} fy={new_camera.params[1]:.3f} "
              f"cx={new_camera.params[2]:.1f} cy={new_camera.params[3]:.1f}")


if __name__ == "__main__":
    _main()
