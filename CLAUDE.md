# CLAUDE.md — BTS Digital Twin: Novel View Synthesis

Kiến trúc & ngữ cảnh dự án cho cuộc thi AI "BTS Digital Twin (Novel View Synthesis)". Đọc file này trước khi
đụng vào `src/`, `dataset/`, hoặc bàn về pipeline train/test/eval.

## 1. Bài toán & scoring

- Input mỗi scene: ảnh multi-view (drone/hand-held) + COLMAP sparse reconstruction (đã dựng sẵn) + danh sách
  camera pose cần render (`test_poses.csv`).
- Output: ảnh RGB tại từng test pose, càng giống ground-truth càng tốt.
- Điểm mỗi scene: `score = 0.4*(1 - LPIPS) + 0.3*SSIM + 0.3*psnr_norm`, với
  `psnr_norm = clamp(PSNR / PSNR_max, 0, 1)`. Điểm cuối = trung bình các scene. **Thiếu/thừa scene so với
  groundtruth → không được tính điểm** (bắt buộc submit đúng tập scene).
- `PSNR_max` dùng để chuẩn hóa chưa được ban tổ chức nêu giá trị cụ thể trong brief đã cung cấp → để làm
  hằng số cấu hình được (`--psnr_max`) trong `evaluate.py`, không hard-code.
- Phần "7. Định dạng nộp bài" của brief không có trong nội dung được cung cấp (nhảy từ mục 6 sang mục 8) →
  cấu trúc thư mục nộp bài (`scene_001/0001.png…`) trong yêu cầu là suy đoán hợp lý, cần đối chiếu lại với đề
  bài gốc khi có. Xem giả định cụ thể ở §5.

## 2. Dữ liệu thực tế (đã kiểm tra bằng cách đọc trực tiếp binary/CSV, không đoán)

Vị trí: `dataset/phase1/{private_set1,public_set}/<SCENE_ID>/`.

```
<SCENE_ID>/
├── train/
│   ├── images/{*.JPG}          # ảnh gốc từ drone (DJI_*), độ phân giải cố định 1320x989
│   └── sparse/0/
│       ├── cameras.bin          # 1 camera duy nhất/scene, model SIMPLE_RADIAL
│       ├── images.bin           # extrinsics (qvec,tvec) theo format COLMAP cổ điển
│       ├── points3D.bin/.ply    # sparse point cloud (dùng để init Gaussians)
│       ├── frames.bin, rigs.bin # COLMAP "generalized rig" format mới — KHÔNG cần dùng (xem bên dưới)
└── test/
    └── test_poses.csv           # image_name,qw,qx,qy,qz,tx,ty,tz,fx,fy,cx,cy,width,height
```

- **private_set1** (8 scene: HCM0249, HCM0254, HCM0276, HCM1439, HNI0131, HNI0265, HNI0366, HNI0437):
  chỉ có `test_poses.csv`, **KHÔNG có ảnh ground-truth test** (ban tổ chức giữ để chấm). Train images:
  103–240 ảnh/scene, test poses: 26–60/scene, tỷ lệ ~80/20 đúng như brief.
- **public_set** (5 scene: HCM0181, HCM0193, HCM0204, hcm0031, hcm0034): có thêm `test/images/` với ảnh
  ground-truth thật → **đây là bộ validation nội bộ tốt nhất** để đo LPIPS/SSIM/PSNR thật trước khi nộp.
- Cả hai bộ dùng chung README.txt: scale ảnh đã là 1/4 kích thước gốc, tỷ lệ train/test 80/20.

### 2.1 Camera model — điểm quan trọng nhất cần xử lý đúng

Đọc trực tiếp `cameras.bin` của cả 8 scene private:

| Scene | model_id | width×height | fx=fy | cx,cy | k (distortion) |
|---|---|---|---|---|---|
| HCM0249 | 2 (SIMPLE_RADIAL) | 1320×989 | 928.20 | 660.0, 494.5 | +0.00891 |
| HCM0254 | 2 | 1320×989 | 926.36 | 660.0, 494.5 | +0.00982 |
| HCM0276 | 2 | 1320×989 | 925.68 | 660.0, 494.5 | +0.00810 |
| HCM1439 | 2 | 1320×989 | 926.26 | 660.0, 494.5 | +0.00759 |
| HNI0131 | 2 | 1320×989 | 925.48 | 660.0, 494.5 | **-0.11479** |
| HNI0265 | 2 | 1320×989 | 926.47 | 660.0, 494.5 | **-0.11470** |
| HNI0366 | 2 | 1320×989 | 926.49 | 660.0, 494.5 | +0.01196 |
| HNI0437 | 2 | 1320×989 | 926.75 | 660.0, 494.5 | +0.01383 |

- Toàn bộ ảnh train dùng model **SIMPLE_RADIAL** (COLMAP model_id=2, params `[f, cx, cy, k]`). `cx,cy` luôn
  đúng bằng tâm ảnh (`width/2, height/2`).
- `HNI0131` và `HNI0265` có méo ảnh **đáng kể** (k ≈ -0.115) — không thể bỏ qua, sẽ tạo lỗi hình học rõ rệt
  ở rìa ảnh nếu train trực tiếp trên ảnh gốc.
- `test_poses.csv` chỉ cấp `fx, fy, cx, cy` — **không có hệ số méo** → ground-truth test được coi là chụp/
  render với model **PINHOLE** (không méo). Điều này bắt buộc một bước tiền xử lý: **undistort ảnh train
  từ SIMPLE_RADIAL → PINHOLE** trước khi đưa vào training, nếu không scene 3D Gaussian sẽ học lệch so với
  hệ hình học mà test poses giả định.
  - COLMAP `SIMPLE_RADIAL` (`x_d = x_u * (1 + k*r²)`, chỉ 1 hệ số) tương đương chính xác với OpenCV
    `distCoeffs = [k, 0, 0, 0]` khi dùng cùng ma trận nội tại → `cv2.undistort`/`cv2.initUndistortRectifyMap`
    dùng được trực tiếp, không cần build COLMAP CLI.
  - Sau undistort, camera dùng cho training chuyển thành PINHOLE `[f, f, cx, cy]` — khớp format mà
    `test_poses.csv` mô tả.

### 2.2 Format COLMAP binary — đã xác minh, không phải định dạng "rig" mới

- `cameras.bin`/`images.bin`/`points3D.bin` là **format COLMAP cổ điển** (giống format mà
  `graphdeco-inria/gaussian-splatting` gốc đọc bằng `read_intrinsics_binary`/`read_extrinsics_binary`).
  Đã verify bằng cách parse tay `images.bin` của HCM0249: `qvec`/`tvec` của ảnh `DJI_..._0006_V.JPG` khớp
  **chính xác từng chữ số** với hàng tương ứng trong `test_poses.csv` → xác nhận test poses được trích trực
  tiếp từ cùng một COLMAP reconstruction, không phải pose tính lại độc lập.
- `frames.bin`/`rigs.bin` là 2 file phụ của **COLMAP rig/frame format mới** (hỗ trợ multi-camera rig). Vì
  mỗi scene chỉ có 1 camera/1 rig nên 2 file này không mang thêm thông tin so với `images.bin` — **baseline
  không cần đọc chúng**. Ghi chú lại phòng khi COLMAP version khác đổi format `images.bin` chính (không phải
  trường hợp ở đây, đã verify).
- `images.bin` chứa **nhiều ảnh hơn** số file trong `train/images/` (vd. HCM0249: 306 ảnh registered nhưng
  chỉ 240 ảnh trong `train/images/`). Phần chênh lệch (~60-66) chính là các test poses — nghĩa là sparse
  reconstruction/point cloud gốc được dựng trên **toàn bộ** ảnh (train+test), rồi tổ chức mới tách ảnh test
  ra. Point cloud init (`points3D.ply`) vì vậy có hình học "biết" cả các vị trí test (không phải leak màu
  sắc/ảnh, chỉ là cấu trúc SfM) — đây là dữ liệu tổ chức cấp sẵn nên dùng thẳng, không cần lo compliance.
- Không có đảm bảo về scale mét thực — tọa độ là scale COLMAP tùy ý (translation ~O(1-10)). Dùng chuẩn hóa
  kiểu NeRF++ (`getNerfppNorm`) như repo gốc là đủ.

## 3. Môi trường tính toán — ràng buộc kiến trúc quan trọng

- Máy dev hiện tại (Windows, Intel Arc 130T GPU) **không có CUDA** → không build/chạy được
  `diff-gaussian-rasterization` / `simple-knn` (CUDA C++ extension) — 2 submodule bắt buộc của repo gốc
  `graphdeco-inria/gaussian-splatting`.
- `torch==2.12.1` đã lock trong `uv.lock`; trên Windows PyPI resolve về wheel **CPU-only** (các gói
  `nvidia-*`/`cuda-bindings` xuất hiện trong lockfile chỉ vì lock đa nền tảng, không áp dụng cho wheel
  Windows thực tế được cài).
- **Chưa chốt nơi train thật** (cloud NVIDIA GPU vs. local) — quyết định này để sau. Hệ quả thiết kế:
  - Toàn bộ lời gọi rasterizer phải cô lập trong `src/gaussian_renderer/__init__.py::render()` — một điểm
    vào duy nhất, để có thể thay backend (CUDA submodule chính thức ↔ fallback khác) mà không đụng
    training loop / eval code.
  - Baseline mặc định nhắm tới **NVIDIA CUDA** (đúng repo gốc, chất lượng/tốc độ chuẩn) vì đó là cách duy
    nhất thực tế để train 3DGS ở quy mô 8+ scene trong thời gian hợp lý. Local Windows/Intel Arc chỉ dùng để
    code, đọc dataset, viết/debug logic không cần rasterizer (data loader, undistort, evaluate trên ảnh có
    sẵn) trên CPU.
  - Trước khi chạy Phase "Core Gaussian Splatting model" trong `PLAN.md` bắt buộc phải chốt máy CUDA.

## 4. Kiến trúc module (đích, dựa trên repo gốc + thích nghi data của dự án)

```
src/
├── scene/
│   ├── colmap_loader.py      # port nguyên bản: đọc cameras/images/points3D .bin & .txt
│   ├── dataset_readers.py    # ĐÃ CÓ SẴN (copy từ repo gốc) — cần sửa: xử lý SIMPLE_RADIAL (gọi
│   │                          #   preprocess/undistort trước khi build CameraInfo), bỏ nhánh eval/llffhold
│   │                          #   cũ (không áp dụng vì test set là file CSV riêng, không phải ảnh cùng thư mục)
│   ├── cameras.py            # Camera class (world_view_transform, projection_matrix, ...) — chưa có, cần tạo
│   ├── gaussian_model.py     # GaussianModel — port nguyên bản (phần lớn thuần PyTorch, không phụ thuộc CUDA
│   │                          #   ngoại trừ nơi gọi rasterizer)
│   └── __init__.py           # Scene class: load SceneInfo -> train/(val) cameras + gaussians — chưa có
├── gaussian_renderer/
│   └── __init__.py           # render() — điểm cô lập duy nhất gọi diff_gaussian_rasterization — chưa có
├── utils/
│   ├── graphics_utils.py     # BasicPointCloud, getWorld2View2, getProjectionMatrix, focal2fov/fov2focal
│   ├── sh_utils.py           # spherical harmonics (SH2RGB, RGB2SH, eval_sh)
│   ├── general_utils.py      # build_rotation, strip_symmetric, inverse_sigmoid, PILtoTorch, lr schedule — chưa có
│   ├── loss_utils.py         # l1_loss, ssim/d-ssim — chưa có
│   ├── image_utils.py        # psnr — chưa có
│   └── camera_utils.py       # cameraList_from_camInfos, loadCam — chưa có
├── preprocess/
│   └── undistort.py          # MỚI (không có trong repo gốc): SIMPLE_RADIAL -> PINHOLE cho toàn bộ ảnh train
│                              #   của 1 scene, cache ra đĩa, rewrite camera params dùng cho training
├── arguments/
│   └── __init__.py           # ModelParams, PipelineParams, OptimizationParams — chưa có
├── train.py                  # SKELETON HIỆN TẠI RỖNG — training loop per-scene, port từ repo gốc
├── render_submission.py      # MỚI: đọc test_poses.csv trực tiếp (không phải COLMAP images.bin), render,
│                              #   ghi ra <output>/<scene>/<NNNN>.png
└── evaluate.py                # MỚI: so khớp renders với public_set ground-truth, tính LPIPS/SSIM/PSNR + score
```

- Các file hiện đang rỗng/1-dòng (`colmap_loader.py`, `gaussian_model.py`, `train.py`,
  `utils/graphics_utils.py`, `utils/sh_utils.py`) là do người dùng đã bắt đầu paste code gốc nhưng chưa xong —
  cần điền đầy đủ theo `PLAN.md`.
- `dataset_readers.py` hiện tại **đã là bản copy gần như nguyên vẹn từ repo gốc** — vẫn còn nhánh
  `readColmapSceneInfo`/`readColmapCameras` giả định model `PINHOLE`/`SIMPLE_PINHOLE` only (sẽ `assert False`
  trên data của chúng ta vì model thật là `SIMPLE_RADIAL`) và logic `llffhold`/`test.txt` (không áp dụng vì
  test set của cuộc thi là `test_poses.csv` riêng biệt, không phải ảnh test nằm cùng thư mục train). Cả hai
  điểm này cần sửa.

## 5. Điểm khác biệt so với repo gốc & giả định cần xác nhận lại

| Điểm | Repo gốc | Cuộc thi này | Xử lý |
|---|---|---|---|
| Camera model | PINHOLE/SIMPLE_PINHOLE only | SIMPLE_RADIAL (có méo) | Thêm bước undistort (§2.1) |
| Test split | `--eval` + llffhold trên cùng thư mục ảnh | `test_poses.csv` riêng, **không có ảnh** (trừ public_set) | Đọc CSV trực tiếp, không dùng `is_test` từ COLMAP folder |
| Convert/undistort | Chạy `convert.py` gọi COLMAP CLI | Không có COLMAP CLI cài sẵn, tự làm bằng OpenCV | `preprocess/undistort.py` |
| Render script | `render.py` render lại train+test split có sẵn trong COLMAP | Phải render đúng theo `test_poses.csv`, đặt tên file theo yêu cầu nộp bài | `render_submission.py` (mới) |
| Metrics | `metrics.py` (LPIPS/SSIM/PSNR không chuẩn hóa, không có công thức tổng hợp) | Cần đúng công thức `0.4*(1-LPIPS)+0.3*SSIM+0.3*psnr_norm`, `psnr_norm` có `PSNR_max` cấu hình | `evaluate.py` (mới) |
| Nhiều scene | Thường train 1 scene/lần | 8 scene private + 5 scene public, mỗi scene train **độc lập** (3DGS không share weight giữa scene) | script orchestrate train tất cả scene tuần tự |

**Giả định cần xác nhận lại với đề bài gốc (mục 7 bị thiếu trong brief đã cung cấp):**
- Tên thư mục output: dùng tên scene thật (`HCM0249/`, …) hay `scene_001/`, `scene_002/` theo thứ tự nào?
  Baseline sẽ mặc định dùng **tên scene thật** làm tên thư mục (dễ truy vết hơn), có thể đổi bằng config.
- Thứ tự/tên file ảnh trong mỗi scene (`0001.png`, `0002.png`, …): mặc định đánh số theo **thứ tự dòng
  trong `test_poses.csv`** (1-indexed, zero-pad 4 chữ số) và luôn kèm file `index_map.json` map
  `NNNN.png -> image_name gốc` để dễ đối chiếu/debug — không tự tin thứ tự này khớp 100% với thứ tự
  ground-truth phía ban tổ chức chấm, cần xác nhận.
- `PSNR_max` dùng để chuẩn hóa: chưa có giá trị cụ thể từ đề bài → để làm tham số cấu hình, không hard-code.

## 6. Quy ước code/tooling

- Quản lý gói: `uv` (`pyproject.toml` + `uv.lock`). `.python-version` ghim `3.10` — **phải tạo venv bằng
  `uv venv --python 3.10`** vì môi trường mặc định trên máy này đang resolve về Python khác (3.12/3.14).
- Khi có máy CUDA thật: cài `torch` theo đúng index CUDA của máy đó (vd.
  `uv pip install torch --index-url https://download.pytorch.org/whl/cu124`), không dùng thẳng bản đã lock
  cho Windows CPU.
- License: code gốc `graphdeco-inria/gaussian-splatting` dùng giấy phép **non-commercial research only** —
  ghi chú lại, không phải việc của agent quyết định compliance nhưng cần biết khi trích dẫn/port code.
- `dataset/` bị `.gitignore` (dòng `phase1`) — không commit dữ liệu.

## 7. Tham chiếu

- Repo gốc: https://github.com/graphdeco-inria/gaussian-splatting
- Xem `PLAN.md` cho lộ trình implement theo từng bước.
