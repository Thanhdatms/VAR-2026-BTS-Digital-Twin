# PLAN.md — Lộ trình implement baseline 3D Gaussian Splatting

Đọc `CLAUDE.md` trước. Plan này chia theo phase, mỗi phase có mục tiêu rõ + tiêu chí "xong". Thứ tự phase
là thứ tự implement khuyến nghị; Phase 0 là blocking (không train được nếu chưa chốt).

## Phase 0 — Chốt môi trường compute (BLOCKING)

- [ ] Chốt nơi train thật: cloud NVIDIA GPU (Colab/Kaggle/RunPod/AWS/GCP...) hay máy khác có CUDA.
- [ ] Trên máy train: cài CUDA toolkit khớp driver, cài `torch` bản CUDA tương ứng, clone 2 submodule CUDA
      của repo gốc (`diff-gaussian-rasterization`, `simple-knn`), build bằng `pip install submodules/...`
      (cần nvcc + compiler C++ — Linux thường dễ hơn Windows cho bước này).
- [ ] `uv venv --python 3.10` để khớp `.python-version`; thêm dependency còn thiếu vào `pyproject.toml`:
      `opencv-python` (undistort), `lpips`, `scikit-image` hoặc `torchmetrics` (SSIM/PSNR tham chiếu để
      double-check), `tqdm`, `pandas` (đọc `test_poses.csv`).
- **Xong khi**: `python -c "import torch; print(torch.cuda.is_available())"` → `True` trên máy train, và
  `from diff_gaussian_rasterization import GaussianRasterizer` import được không lỗi.

## Phase 1 — Data pipeline (làm được ngay trên máy local, không cần CUDA)

1. **`src/scene/colmap_loader.py`**: port nguyên bản từ repo gốc (`read_extrinsics_binary`,
   `read_intrinsics_binary`, `read_points3D_binary`, `qvec2rotmat`, + bản `_text` tương ứng). Đây là code
   thuần Python/numpy, không cần sửa gì so với gốc — đã verify format binary khớp (xem `CLAUDE.md` §2.2).
2. **`src/utils/graphics_utils.py`**: port `BasicPointCloud` (NamedTuple), `getWorld2View2`,
   `getProjectionMatrix`, `focal2fov`, `fov2focal`, `geom_transform_points`.
3. **`src/utils/sh_utils.py`**: port `RGB2SH`, `SH2RGB`, `eval_sh`, hằng số `C0..C4`.
4. **`src/preprocess/undistort.py`** (MỚI — không có trong repo gốc):
   - Input: `cameras.bin` (SIMPLE_RADIAL `[f,cx,cy,k]`) + thư mục `train/images/`.
   - Với mỗi ảnh: `cv2.undistort(img, K, distCoeffs=[k,0,0,0])` với `K=[[f,0,cx],[0,f,cy],[0,0,1]]`.
   - Cache ảnh đã undistort vào `<scene>/train/images_undistorted/` (không ghi đè ảnh gốc).
   - Trả về camera params PINHOLE mới `[f, f, cx, cy]` (K không đổi vì undistort giữ nguyên K đã dùng, chỉ
     resample nội dung ảnh) để dataset_readers dùng.
   - Viết 1 script CLI nhỏ chạy thử trên `HCM0249`, so sánh trực quan 1 ảnh trước/sau (đặc biệt HNI0131,
     HNI0265 — k lớn, dễ thấy khác biệt ở rìa ảnh) để verify bằng mắt trước khi tự động hoá toàn bộ.
5. **Sửa `src/scene/dataset_readers.py`**:
   - `readColmapCameras`: thêm nhánh `SIMPLE_RADIAL` gọi `preprocess/undistort` thay vì `assert False`.
   - Bỏ nhánh `llffhold`/`test.txt` trong `readColmapSceneInfo` — với dataset này **toàn bộ ảnh trong
     `train/images/` đều là train** (không tự suy test split từ COLMAP folder). Test cameras (nếu cần cho
     validation) sẽ đến từ một hàm riêng đọc `test_poses.csv` (Phase 4), không phải từ
     `readColmapSceneInfo`.
   - Thêm hàm mới `readTestPosesCSV(csv_path) -> List[CameraInfo]` để dùng chung cho cả validation nội bộ
     (public_set, có GT) và render nộp bài (private_set1, không có GT) — parse `qw,qx,qy,qz,tx,ty,tz` giống
     `qvec2rotmat`/`R,T` trong `readColmapCameras`, `fx,fy,cx,cy` → `FovX,FovY` qua `focal2fov`.
6. **Kiểm tra**: viết script nhỏ load `SceneInfo` cho `HCM0249` và `public_set/HCM0181`, in số lượng camera,
   số điểm point cloud, kích thước ảnh sau undistort — chạy trên CPU, không cần GPU.

- **Xong khi**: load được `SceneInfo` cho cả 8 scene private + 5 scene public không lỗi, ảnh undistort trông
  đúng (không méo ở rìa), số lượng camera train khớp số ảnh trong `train/images/`.

## Phase 2 — Core Gaussian Splatting model (cần máy CUDA để chạy thật, code có thể viết trên local)

1. **`src/scene/gaussian_model.py`**: port `GaussianModel` nguyên bản — `create_from_pcd`,
   `training_setup` (optimizer per-param-group với LR riêng cho xyz/f_dc/f_rest/opacity/scaling/rotation),
   `densify_and_clone/split/prune`, `reset_opacity`, `save_ply`/`load_ply`, các hàm activation
   (`scaling_activation=exp`, `opacity_activation=sigmoid`, `rotation_activation=normalize`).
2. **`src/utils/general_utils.py`**: `inverse_sigmoid`, `PILtoTorch`, `get_expon_lr_func`, `strip_symmetric`,
   `build_rotation`, `build_scaling_rotation`.
3. **`src/scene/cameras.py`**: `Camera` class — nhận `CameraInfo`, dựng `world_view_transform`,
   `projection_matrix`, `full_proj_transform`, `camera_center`; load ảnh GT (nếu có) resize theo
   `resolution_scale`.
4. **`src/utils/camera_utils.py`**: `loadCam`, `cameraList_from_camInfos`, `camera_to_JSON` (để lưu
   `cameras.json` trong output model — hữu ích để audit sau).
5. **`src/scene/__init__.py`**: `Scene` class nối `SceneInfo` (Phase 1) với `GaussianModel` — load hoặc khởi
   tạo từ point cloud, quản lý `getTrainCameras()`.
6. **`src/gaussian_renderer/__init__.py`**: hàm `render(viewpoint_camera, pc, pipe, bg_color)` — điểm DUY
   NHẤT gọi `diff_gaussian_rasterization.GaussianRasterizer`. Giữ signature giống repo gốc để nếu sau này
   đổi backend rasterizer thì chỉ sửa file này.
7. **`src/arguments/__init__.py`**: `ModelParams`, `PipelineParams`, `OptimizationParams` (giữ default gốc:
   30_000 iters, `densify_until_iter=15_000`, `sh_degree=3`, ... — đây là baseline, tinh chỉnh ở Phase 6).
8. **`src/utils/loss_utils.py`**: `l1_loss`, `ssim` (window-based, dùng cho loss D-SSIM trong training, khác
   với SSIM dùng để evaluate ở Phase 5 — có thể tái dùng cùng implementation).
9. **`src/utils/image_utils.py`**: `psnr`.

- **Xong khi**: trên máy CUDA, chạy được 1 vòng forward+backward render cho 1 camera của `HCM0249` không lỗi
  shape/device, loss giảm sau vài chục iteration trên 1 scene nhỏ thử nghiệm.

## Phase 3 — Training script (`src/train.py`)

- Port training loop gốc: random shuffle camera mỗi epoch, render, tính `Ll1 + lambda_dssim * (1-ssim)`,
  backward, `densify_and_prune` theo lịch, `reset_opacity` định kỳ, tăng dần `sh_degree`
  (`oneupSHdegree`), lưu checkpoint (`point_cloud/iteration_XXXX/point_cloud.ply`) theo `--test_iterations`/
  `--save_iterations`.
- Bỏ phần compute test-loss trong loop gốc dựa trên COLMAP test split (không có với private_set1); thay
  bằng: nếu scene thuộc `public_set` (có `test/images/`), log thêm PSNR trên `test_poses.csv` mỗi N iter để
  theo dõi; nếu không có GT thì bỏ qua, chỉ log train loss.
- CLI: `--source_path <scene dir> --model_path output/<scene_id>` (không phụ thuộc cwd, không hard-code tên
  scene).
- Thêm `src/train_all.py` (hoặc shell script) loop qua toàn bộ scene (8 private + 5 public — public dùng để
  validate pipeline end-to-end) gọi `train.py` tuần tự — mỗi scene là 1 lần train độc lập (3DGS không share
  gì giữa các scene).

- **Xong khi**: train xong ít nhất 1 scene public (`HCM0181`) đến hội tụ hợp lý, `point_cloud.ply` cuối cùng
  render ra ảnh nhìn giống ảnh train (kiểm tra bằng mắt).

## Phase 4 — Inference script nộp bài (`src/render_submission.py`)

- Input: model đã train (`output/<scene_id>/point_cloud/iteration_XXXX/point_cloud.ply`) +
  `test/test_poses.csv` của đúng scene đó.
- Dùng `readTestPosesCSV` (Phase 1) để build `Camera` object trực tiếp từ CSV (không qua COLMAP
  images.bin), render bằng `gaussian_renderer.render()` với background mặc định (đen hoặc trắng — kiểm tra
  scene có nền trời/không để chọn, baseline dùng đen).
- Output đúng cấu trúc yêu cầu:
  ```
  <output_root>/<scene_id>/0001.png
  <output_root>/<scene_id>/0002.png
  ...
  <output_root>/<scene_id>/index_map.json   # {"0001.png": "<image_name gốc trong CSV>", ...}
  ```
  (xem giả định về tên thư mục/thứ tự file ở `CLAUDE.md` §5 — cần xác nhận lại với đề bài gốc).
- CLI hỗ trợ chạy 1 scene hoặc toàn bộ `dataset/phase1/private_set1/*` một lượt (`--all`).
- Resize/crop output đúng `width,height` từng dòng CSV (mỗi scene test_poses width/height đồng nhất trong
  data đã kiểm tra, nhưng code không nên giả định — đọc từng dòng).

- **Xong khi**: chạy `render_submission.py --all` ra đủ 8 thư mục scene, mỗi thư mục đúng số ảnh bằng số
  dòng trong `test_poses.csv` tương ứng (26–60 tùy scene).

## Phase 5 — Evaluation script (`src/evaluate.py`)

- Chỉ chạy được đầy đủ trên **public_set** (có GT thật ở `test/images/`). Với private_set1 chỉ dùng để tự
  kiểm tra format output (không tính được điểm thật vì thiếu GT).
- Với mỗi scene có GT: match `index_map.json` (hoặc trực tiếp `image_name`) → cặp (render, GT), resize nếu
  lệch kích thước (không nên xảy ra nhưng defensive), tính:
  - **PSNR**: pixel MSE trên ảnh `[0,1]`, rồi `psnr_norm = clamp(psnr/psnr_max, 0, 1)` (`psnr_max` là
    tham số CLI, default có thể để 50 nhưng **flag rõ đây là giá trị giả định**, không phải số chính thức từ
    đề bài).
  - **SSIM**: dùng đúng công thức chuẩn (skimage `structural_similarity` hoặc torchmetrics
    `StructuralSimilarityIndexMeasure`), multichannel RGB.
  - **LPIPS**: thư viện `lpips` (paper gốc, AlexNet hoặc VGG backbone — ghi rõ backbone dùng để tránh nhầm
    khi so kết quả với leaderboard).
- Tính `score = 0.4*(1-LPIPS) + 0.3*SSIM + 0.3*psnr_norm` mỗi ảnh, trung bình theo ảnh trong scene, rồi
  trung bình các scene = điểm cuối cùng.
- Output: in ra bảng per-scene (mean LPIPS/SSIM/PSNR/score) + tổng, và ghi 1 file `eval_report.json`/`.csv`.

- **Xong khi**: chạy được trên 5 scene `public_set`, ra bảng điểm hợp lý (SSIM/PSNR không NaN, LPIPS trong
  khoảng [0,1]).

## Phase 6 — Cải thiện chất lượng (sau khi baseline chạy end-to-end)

Chỉ làm sau khi Phase 1-5 đã cho ra pipeline train→render→eval hoạt động trọn vẹn trên ít nhất vài scene.
Ưu tiên theo lợi ích/công sức:

1. **Kiểm chứng lại bước undistort** bằng cách so ảnh train sau undistort với 1 vài ảnh GT của public_set
   (nếu overlap góc nhìn) — đảm bảo không lệch tâm/scale.
2. **Tinh chỉnh lịch densify/prune** theo mật độ scene thực tế (BTS: trụ ăng-ten, kết cấu mảnh — dễ bị
   under-reconstruct). Cân nhắc tăng `densify_grad_threshold` resolution hoặc kéo dài `densify_until_iter`.
   nếu artefact ở kết cấu mảnh nhiều.
3. **SH degree & số iteration**: baseline dùng default gốc (`sh_degree=3`, 30k iters) — theo dõi PSNR trên
   public_set để quyết định có cần train lâu hơn không (scene ít ảnh như `HCM1439` 103 ảnh có thể hội tụ
   nhanh hơn/cần regularize nhiều hơn để tránh overfit).
4. **Exposure/white-balance**: ảnh drone cùng 1 chuyến bay thường exposure khá đồng nhất — kiểm tra bằng mắt
   trước khi thêm exposure compensation (tăng độ phức tạp không cần thiết nếu không thấy vấn đề).
5. **Background color**: kiểm tra scene có bầu trời/nền đồng nhất hay không để chọn `--white_background`
   phù hợp thay vì mặc định đen của repo gốc.
6. **So sánh nhiều checkpoint** (`7000`, `15000`, `30000` iters) trên public_set để chọn iteration tốt nhất
   theo đúng metric cuộc thi (không chỉ theo loss training).

## Phase 7 — Đóng gói & tái lập

- 1 script/lệnh tổng: `preprocess (Phase1) → train_all (Phase3) → render_submission --all (Phase4) →
  evaluate trên public_set (Phase5)`.
- README ngắn ghi lại: cách set up venv/CUDA (Phase 0), cách chạy từng bước, thời gian train ước tính/scene.

## Việc cần chốt với người dùng / ban tổ chức trước khi chấm điểm thật

- [ ] Nơi train (Phase 0).
- [ ] Cấu trúc thư mục nộp bài chính xác (tên scene thật hay `scene_001…`, thứ tự file) — xem `CLAUDE.md` §5.
- [ ] Giá trị `PSNR_max` chính thức dùng để chuẩn hóa.
