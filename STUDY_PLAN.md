# STUDY_PLAN.md — Lộ trình đọc hiểu toàn bộ codebase

Tài liệu này khác `CLAUDE.md` (kiến trúc/quyết định thiết kế) và `PLAN.md` (roadmap implement).
Đây là **lộ trình học** — đọc code theo đúng thứ tự dữ liệu chảy qua hệ thống, từ file `.JPG`
thô trên đĩa cho tới điểm số cuối cùng. Mỗi đơn vị có: mục tiêu, input/output, file cần đọc,
hàm/class quan trọng, và câu hỏi tự kiểm tra — trả lời được hết câu hỏi nghĩa là đã hiểu đơn vị đó.

**Cách dùng:** đọc lần lượt Đơn vị 0 → 10. Mỗi đơn vị nên vừa đọc code vừa chạy thử lệnh gợi ý
(nếu có) để thấy dữ liệu thực tế đi qua, không chỉ đọc suông. Đừng nhảy cóc — đơn vị sau giả định
đã hiểu đơn vị trước.

---

## Đơn vị 0 — Bối cảnh tối thiểu cần biết trước khi đọc code

**Mục tiêu:** hiểu 3D Gaussian Splatting (3DGS) là gì ở mức khái niệm, không cần hiểu hết toán.

3DGS biểu diễn 1 scene 3D bằng **hàng trăm nghìn đến hàng triệu "quả cầu mờ" (Gaussian)** lơ lửng
trong không gian, mỗi Gaussian có: vị trí, hình dạng/kích thước (ellipsoid), độ trong suốt
(opacity), và màu sắc (biểu diễn bằng spherical harmonics — màu thay đổi theo góc nhìn). Để tạo ra
1 ảnh từ 1 góc camera: chiếu tất cả Gaussian xuống mặt phẳng ảnh 2D, rồi chồng màu từng Gaussian
lên nhau theo thứ tự gần→xa (giống vẽ nhiều lớp kính màu mờ chồng lên nhau). Training = tối ưu vị
trí/hình dạng/màu của từng Gaussian sao cho ảnh render ra giống ảnh thật chụp được.

**Đọc trước khi tiếp tục:** `CLAUDE.md` mục 1-2 (bài toán + cấu trúc data thật của bạn).

**Câu hỏi tự kiểm tra:**
- 1 "Gaussian" trong 3DGS gồm những thuộc tính gì? (không cần công thức, chỉ cần liệt kê đúng ý nghĩa)
- Vì sao 3DGS render nhanh hơn NeRF? (gợi ý: không cần "đoán" màu bằng neural network cho mỗi tia sáng)

---

## Đơn vị 1 — Đọc dữ liệu COLMAP thô (binary → Python object)

**Mục tiêu:** hiểu cách đọc file nhị phân COLMAP (`cameras.bin`, `images.bin`, `points3D.bin`) ra
thành object Python dùng được.

**Input:** file nhị phân trong `dataset/.../train/sparse/0/*.bin`
**Output:** dict `{camera_id: Camera}`, dict `{image_id: Image}`, mảng numpy `(xyz, rgb, error)`

**File:** [`src/scene/colmap_loader.py`](src/scene/colmap_loader.py) — toàn bộ file này chỉ làm
đúng 1 việc: đọc byte thô theo đúng format COLMAP quy định, không có logic nghiệp vụ nào khác.

| Hàm | Input | Output | Ghi chú |
|---|---|---|---|
| `read_intrinsics_binary(path)` | `cameras.bin` | `{camera_id: Camera(id,model,width,height,params)}` | `params` là mảng số thực, ý nghĩa từng số phụ thuộc `model` (xem Đơn vị 2) |
| `read_extrinsics_binary(path)` | `images.bin` | `{image_id: Image(qvec,tvec,camera_id,name,...)}` | `qvec`=quaternion xoay, `tvec`=vị trí camera |
| `read_points3D_binary(path)` | `points3D.bin` | `(xyz, rgb, error)` 3 mảng numpy | Đây là sparse point cloud — nguồn khởi tạo Gaussian ban đầu |
| `qvec2rotmat(qvec)` | quaternion (4 số) | ma trận xoay 3×3 | Dùng ở khắp nơi để đổi quaternion COLMAP → ma trận xoay |

**Chạy thử để thấy dữ liệu thật:**
```python
import sys; sys.path.append("src")
from scene.colmap_loader import read_intrinsics_binary, read_extrinsics_binary
cams = read_intrinsics_binary("dataset/phase1/private_set1/HCM0249/train/sparse/0/cameras.bin")
print(cams)  # chỉ có 1 camera, model SIMPLE_RADIAL — xem Đơn vị 2 vì sao quan trọng
imgs = read_extrinsics_binary("dataset/phase1/private_set1/HCM0249/train/sparse/0/images.bin")
print(len(imgs), "ảnh đã đăng ký trong COLMAP")  # nhiều hơn số ảnh trong train/images/ — vì sao?
```

**Câu hỏi tự kiểm tra:**
- `qvec` và `tvec` của 1 `Image` biểu diễn điều gì? (pose camera đó trong không gian world)
- Tại sao `len(imgs)` (từ `images.bin`) lại nhiều hơn số file ảnh thật trong `train/images/`?
  (Trả lời nằm ở `CLAUDE.md` mục 2.2 nếu chưa đoán ra)

---

## Đơn vị 2 — Camera model & vấn đề méo ảnh (undistortion)

**Mục tiêu:** hiểu vì sao ảnh train cần được "sửa méo" trước khi train, và cách sửa.

**Input:** ảnh gốc (`train/images/*.JPG`) + camera SIMPLE_RADIAL (có hệ số méo `k`)
**Output:** ảnh đã sửa méo (`train/images_undistorted/*.JPG`, tự sinh, cache lại) + camera PINHOLE mới

**File:** [`src/preprocess/undistort.py`](src/preprocess/undistort.py)

| Hàm | Việc làm |
|---|---|
| `build_undistort_map(camera)` | Từ 1 `Camera` COLMAP → tính ma trận nội tại `K` + hệ số méo `dist`, trả về camera PINHOLE tương ứng |
| `undistort_scene(source_path, force=False)` | Chạy `cv2.undistort` cho toàn bộ ảnh 1 scene, cache kết quả ra đĩa (chỉ chạy lại nếu `force=True` hoặc file chưa tồn tại) |

**Vì sao bước này tồn tại (không có trong repo gốc):** dữ liệu của bạn dùng camera model
`SIMPLE_RADIAL` (có méo ảnh nhẹ-đến-mạnh tuỳ scene), nhưng `test_poses.csv` chỉ cấp
`fx,fy,cx,cy` — ngầm định ảnh test là PINHOLE (không méo). Nếu train thẳng trên ảnh méo, model
3D sẽ học sai hình học. Đọc `CLAUDE.md` mục 2.1 để xem bảng hệ số méo thật của 8 scene.

**Câu hỏi tự kiểm tra:**
- Nếu bỏ qua bước undistort này, hậu quả cụ thể là gì khi render ảnh test?
- Vì sao `cv2.undistort` với `distCoeffs=[k,0,0,0]` lại tương đương đúng với model `SIMPLE_RADIAL`
  của COLMAP? (đọc docstring đầu file `undistort.py`)

---

## Đơn vị 3 — Từ COLMAP camera → `CameraInfo` (dữ liệu trung gian, chưa phải object dùng để render)

**Mục tiêu:** hiểu bước "dịch" dữ liệu COLMAP thô sang cấu trúc dùng chung cho cả train và test.

**Input:** output của Đơn vị 1 + 2 (extrinsics, intrinsics đã undistort)
**Output:** list các `CameraInfo` (namedtuple: R, T, FovX, FovY, đường dẫn ảnh, kích thước...)

**File:** [`src/scene/dataset_readers.py`](src/scene/dataset_readers.py)

| Hàm | Input | Output | Khi nào dùng |
|---|---|---|---|
| `readColmapCameras(...)` | extrinsics+intrinsics đã undistort | `List[CameraInfo]` | Xây camera **train** |
| `readTestPosesCSV(csv_path, images_folder=None)` | `test_poses.csv` | `List[CameraInfo]` | Xây camera **test** (không qua COLMAP, đọc CSV thẳng) |
| `readColmapSceneInfo(path, undistort=True)` | đường dẫn thư mục scene | `SceneInfo` (point_cloud + train_cameras + ...) | Hàm "tổng" gọi khi bắt đầu train 1 scene |
| `getNerfppNorm(cam_info)` | list camera | `{translate, radius}` | Chuẩn hoá không gian scene — dùng làm learning-rate scale và bán kính để prune Gaussian quá to |

**Điểm quan trọng:** `readColmapCameras` và `readTestPosesCSV` tạo ra **cùng 1 kiểu dữ liệu**
(`CameraInfo`) dù nguồn khác nhau hẳn (COLMAP binary vs CSV) — đây là lý do phần code phía sau
(dựng `Camera` object, render...) dùng chung được cho cả train lẫn test/inference, không cần biết
camera đó từ đâu ra.

**Chạy thử:**
```python
from scene.dataset_readers import readColmapSceneInfo
info = readColmapSceneInfo("dataset/phase1/private_set1/HCM0249")
print(len(info.train_cameras), info.point_cloud.points.shape, info.nerf_normalization)
```

**Câu hỏi tự kiểm tra:**
- `CameraInfo` có field `image_path` — với camera train và camera test (private_set1) thì field
  này khác nhau thế nào? Vì sao?
- `SceneInfo.test_cameras` luôn là `[]` — tại sao? (gợi ý: test camera không lấy từ đây)

---

## Đơn vị 4 — Từ `CameraInfo` → `Camera` object dùng để render (ma trận biến đổi)

**Mục tiêu:** hiểu các ma trận toán học biến 1 điểm 3D trong world thành pixel trên ảnh.

**Input:** 1 `CameraInfo` (R, T, FovX, FovY, width, height)
**Output:** `Camera` object có sẵn `world_view_transform`, `projection_matrix`, `full_proj_transform`, `camera_center`

**File:**
- [`src/scene/cameras.py`](src/scene/cameras.py) — class `Camera`, tính toán ma trận
- [`src/utils/graphics_utils.py`](src/utils/graphics_utils.py) — các hàm toán thuần tuý dùng bên trong
- [`src/utils/camera_utils.py`](src/utils/camera_utils.py) — `loadCam()`/`cameraList_from_camInfos()`, cầu nối `CameraInfo → Camera` (load ảnh GT nếu có, resize nếu cần)

| Hàm/thuộc tính | Ý nghĩa |
|---|---|
| `getWorld2View2(R, T, translate, scale)` | Ma trận world→camera (4×4) |
| `getProjectionMatrix(znear, zfar, fovX, fovY)` | Ma trận camera→clip-space (phép chiếu phối cảnh chuẩn, giống OpenGL) |
| `Camera.world_view_transform` | = `getWorld2View2(...).T` — lưu dạng chuyển vị vì quy ước nhân ma trận theo row-vector xuyên suốt codebase |
| `Camera.full_proj_transform` | = `world_view_transform @ projection_matrix` — đi thẳng từ world → clip-space trong 1 phép nhân |
| `Camera.camera_center` | vị trí camera trong world (suy ra từ nghịch đảo `world_view_transform`) |

**Điểm dễ nhầm nhất trong toàn bộ codebase:** quy ước **row-vector** (`điểm @ ma_trận`, không phải
`ma_trận @ điểm`). Nếu sau này bạn tự viết thêm code liên quan tới camera, phải giữ đúng quy ước
này — trộn lẫn 2 quy ước là nguồn lỗi khó debug nhất trong đồ hoạ máy tính.

**Câu hỏi tự kiểm tra:**
- Vì sao `Camera` cho phép `image=None`? Trường hợp nào dùng đến?
- `full_proj_transform` được tính từ 2 ma trận nào nhân với nhau?

---

## Đơn vị 5 — `GaussianModel`: 1 Gaussian được lưu trữ thế nào trong bộ nhớ

**Mục tiêu:** hiểu cấu trúc dữ liệu lõi — mọi thứ model "học" được nằm hết trong class này.

**Input:** sparse point cloud (từ Đơn vị 3) khi khởi tạo, hoặc file `.ply` khi load checkpoint
**Output:** N Gaussian, mỗi cái có đủ vị trí/scale/rotation/opacity/màu, đều là `nn.Parameter`
(tức là PyTorch tự tính gradient được)

**File:** [`src/scene/gaussian_model.py`](src/scene/gaussian_model.py) — file dài nhất, quan
trọng nhất, nên đọc chậm.

**6 thuộc tính lõi** (mỗi cái là 1 tensor, N = số Gaussian):

| Tên nội bộ | Shape | Ý nghĩa | Activation khi dùng thật |
|---|---|---|---|
| `_xyz` | (N,3) | vị trí | không có (dùng trực tiếp) |
| `_scaling` | (N,3) | log(kích thước ellipsoid theo 3 trục) | `exp()` — `get_scaling` |
| `_rotation` | (N,4) | quaternion xoay ellipsoid | `normalize()` — `get_rotation` |
| `_opacity` | (N,1) | độ trong suốt (chưa qua sigmoid) | `sigmoid()` — `get_opacity` |
| `_features_dc` | (N,1,3) | màu trung bình (SH bậc 0) | dùng trực tiếp trong `get_features` |
| `_features_rest` | (N,K-1,3) | các hệ số SH bậc cao (màu đổi theo góc nhìn) | dùng trực tiếp |

Lưu ý: các giá trị lưu trong `_scaling`/`_opacity` KHÔNG phải giá trị dùng thật — phải qua hàm
activation (`get_scaling`, `get_opacity`...) mới ra giá trị vật lý thật. Lý do: để Adam optimizer
tối ưu trong không gian không giới hạn (log-scale, logit) thay vì phải ràng buộc trực tiếp (scale
luôn dương, opacity luôn trong [0,1]).

**Các hàm quan trọng theo nhóm:**

| Nhóm | Hàm | Việc làm |
|---|---|---|
| Khởi tạo | `create_from_pcd(pcd, spatial_lr_scale, device)` | Biến N điểm sparse COLMAP → N Gaussian ban đầu (scale suy từ khoảng cách tới 3 điểm gần nhất — xem `_mean_sq_dist_to_nearest_neighbors`) |
| Lưu/đọc | `save_ply(path)` / `load_ply(path, device)` | Ghi/đọc toàn bộ N Gaussian ra file `.ply` chuẩn (tương thích viewer 3DGS khác) |
| Densify | `densify_and_clone`, `densify_and_split`, `densify_and_prune` | Thêm/bớt Gaussian trong lúc train — xem Đơn vị 7 để hiểu khi nào các hàm này được gọi |
| Optimizer | `training_setup(opt)` | Tạo Adam optimizer, mỗi thuộc tính (`xyz`,`opacity`,...) có learning rate riêng |

**Chạy thử:**
```python
from scene.gaussian_model import GaussianModel
from scene.dataset_readers import readColmapSceneInfo
info = readColmapSceneInfo("dataset/phase1/private_set1/HCM0249")
g = GaussianModel(sh_degree=3)
g.create_from_pcd(info.point_cloud, info.nerf_normalization["radius"], device="cpu")
print(g.get_xyz.shape, g.get_scaling.shape, g.get_opacity.shape, g.get_features.shape)
```

**Câu hỏi tự kiểm tra:**
- Vì sao `_scaling` lưu dạng log thay vì lưu trực tiếp kích thước?
- `get_features` nối 2 tensor nào lại với nhau? Kết quả shape là gì (theo N, K = (sh_degree+1)²)?
- Vì sao `create_from_pcd` không dùng `simple_knn` (CUDA) như repo gốc mà dùng `cKDTree`? (đọc
  docstring đầu file — liên quan tới `CLAUDE.md` mục 3)

---

## Đơn vị 6 — Renderer: biến N Gaussian + 1 Camera → 1 ảnh RGB

**Mục tiêu:** hiểu (ở mức toán/thuật toán, không cần code CUDA) cách 1 Gaussian 3D trở thành vệt
màu trên ảnh 2D, và vì sao có 2 backend khác nhau trong project này.

**Input:** `GaussianModel` + `Camera` + màu nền
**Output:** ảnh render `(3, H, W)` + `viewspace_points`/`visibility_filter`/`radii` (dùng cho
densification, xem Đơn vị 7)

**File:**
- [`src/gaussian_renderer/__init__.py`](src/gaussian_renderer/__init__.py) — điểm vào duy nhất,
  chỉ làm 1 việc: chọn backend nào (CUDA hay CPU) dựa trên `device.type`
- [`src/gaussian_renderer/_cuda_backend.py`](src/gaussian_renderer/_cuda_backend.py) — gọi thẳng
  package `diff_gaussian_rasterization` (code CUDA gốc từ paper, KHÔNG tự viết) — đây là backend
  dùng khi train thật trên GPU
- [`src/gaussian_renderer/cpu_rasterizer.py`](src/gaussian_renderer/cpu_rasterizer.py) — renderer
  **tự viết lại từ đầu** bằng thuần PyTorch, dùng khi không có GPU CUDA. Đọc kỹ file này nếu muốn
  hiểu **toán học thật sự** đằng sau 3DGS, vì `_cuda_backend.py` chỉ là lớp gọi hàm, không có công
  thức nào lộ ra ở phía Python.

**Các bước toán học chính** (theo đúng thứ tự trong `cpu_rasterizer.py`, đối chiếu code khi đọc):

1. **Chiếu vị trí 3D → pixel 2D**: world → view-space (`points_view`) → clip-space
   (`points_clip`) → chia phối cảnh (NDC) → toạ độ pixel `(px, py)`.
2. **Chiếu hình dạng 3D → hình dạng 2D (EWA splatting)**: từ scale+rotation dựng ma trận hiệp
   phương sai 3D (`Sigma3D`), chiếu qua ma trận Jacobian của phép chiếu phối cảnh → ra ellipse 2D
   (`cov2d`) mô tả Gaussian đó "loang" ra bao nhiêu trên ảnh, từ đó suy ra bán kính ảnh hưởng
   (`radius`).
3. **Tính màu theo hướng nhìn**: dùng spherical harmonics (`eval_sh`, xem Đơn vị 5 phần
   `_features_rest`) — cùng 1 Gaussian nhìn từ 2 góc khác nhau có thể ra màu hơi khác (mô phỏng
   phản chiếu ánh sáng).
4. **Chồng màu theo thứ tự gần→xa (alpha compositing)**: sort Gaussian theo độ sâu, mỗi Gaussian
   "phủ" 1 lớp màu mờ lên ảnh theo công thức `alpha-over` chuẩn (giống chồng layer trong
   Photoshop với opacity), Gaussian gần che Gaussian xa dần dần.

**Vì sao có 2 backend:** backend CUDA (`_cuda_backend.py`) dùng thuật toán **tile-hoá** (chia ảnh
thành ô 16×16, mỗi ô chỉ tính Gaussian thật sự phủ lên nó) → cực nhanh, dùng để train thật.
`cpu_rasterizer.py` làm đúng cùng công thức toán nhưng **không tile hoá** (tính toàn bộ Gaussian
so với toàn bộ ảnh) → chậm hơn hàng nghìn lần, chỉ dùng để test không cần GPU. Đọc thêm
`CLAUDE.md` mục 3.

**Câu hỏi tự kiểm tra:**
- `screenspace_points` trong cả 2 backend đều là 1 tensor toàn số 0 nhưng có `requires_grad=True`
  — nó dùng để làm gì? (gợi ý: liên quan tới `visibility_filter`/densification ở Đơn vị 7)
- Vì sao phải sort Gaussian theo độ sâu trước khi chồng màu? Nếu không sort thì sai ở đâu?

---

## Đơn vị 7 — Vòng lặp Training: loss, densify, prune

**Mục tiêu:** hiểu 1 iteration training làm những gì, theo đúng thứ tự.

**Input:** `Scene` (đã có `GaussianModel` khởi tạo) + `OptimizationParams`
**Output:** `GaussianModel` đã tối ưu, checkpoint `.ply` lưu định kỳ

**File:** [`src/train.py`](src/train.py), tham số điều khiển ở
[`src/arguments/__init__.py`](src/arguments/__init__.py)

**1 iteration gồm các bước (đọc song song với hàm `training()` trong `train.py`):**

1. `gaussians.update_learning_rate(iteration)` — learning rate của `xyz` giảm dần theo hàm mũ
   (các thuộc tính khác learning rate cố định).
2. Mỗi 1000 iteration: `oneupSHdegree()` — tăng dần độ phức tạp màu (bắt đầu từ màu đơn sắc, tăng
   dần lên màu phụ thuộc góc nhìn) — giúp training ổn định hơn là học hết ngay từ đầu.
3. Chọn 1 camera train ngẫu nhiên (không lặp lại cho tới khi hết vòng — xem `viewpoint_stack`).
4. `render()` (Đơn vị 6) → so với ảnh thật (`gt_image`) → tính loss = trộn **L1** (sai khác pixel)
   và **D-SSIM** (sai khác cấu trúc, dùng `utils/loss_utils.py::ssim`).
5. `loss.backward()` — PyTorch tự tính gradient cho toàn bộ `_xyz`, `_scaling`, ... (đây là lý do
   mọi thứ ở Đơn vị 5 phải là `nn.Parameter`).
6. **Densification** (chỉ trong giai đoạn `densify_from_iter` → `densify_until_iter`):
   - `add_densification_stats(...)`: tích luỹ gradient vị trí trên ảnh (`viewspace_points.grad`)
     — Gaussian nào bị "kéo" nhiều qua nhiều ảnh khác nhau nghĩa là chưa đủ chi tiết ở đó.
   - Mỗi `densification_interval` iteration: `densify_and_prune(...)` — Gaussian gradient cao +
     nhỏ → **clone** (nhân đôi tại chỗ); Gaussian gradient cao + đã to → **split** (tách làm 2, mỗi
     cái nhỏ hơn); Gaussian opacity quá thấp hoặc quá to → **xoá**.
   - Mỗi `opacity_reset_interval` iteration: `reset_opacity()` — ép toàn bộ opacity về gần 0, dọn
     "floater" tích luỹ (giải thích hiện tượng PSNR tụt tạm thời mà bạn đã thấy khi train thật).
7. `optimizer.step()` + `zero_grad()` — cập nhật tham số theo Adam.

**Câu hỏi tự kiểm tra:**
- Vì sao PSNR tụt mạnh rồi phục hồi mỗi 3000 iteration khi train thật? (đã giải thích ở hội thoại
  trước — giờ hãy tự chỉ ra đúng dòng code gây ra hiện tượng đó)
- `densify_and_clone` và `densify_and_split` khác nhau ở điều kiện nào (xem `self.get_scaling`
  so với `percent_dense * scene_extent`)?
- `--load_iteration`/resume hoạt động thế nào? Vì sao phải tính lại `active_sh_degree` khi resume
  thay vì dùng giá trị trong checkpoint?

---

## Đơn vị 8 — `Scene`: lớp kết nối tất cả lại với nhau

**Mục tiêu:** hiểu class `Scene` đóng vai trò "nhạc trưởng" — không tự làm toán gì, chỉ điều phối.

**File:** [`src/scene/__init__.py`](src/scene/__init__.py)

`Scene.__init__` làm đúng theo thứ tự: gọi Đơn vị 3 (`readColmapSceneInfo`) → gọi Đơn vị 4
(`cameraList_from_camInfos`) → gọi Đơn vị 5 (`create_from_pcd` hoặc `load_ply` nếu resume).
`Scene.save()` gọi `gaussians.save_ply()`. `Scene.getTestCamerasFromCSV()` gọi
`readTestPosesCSV` (Đơn vị 3) rồi `cameraList_from_camInfos` (Đơn vị 4) — dùng chung code building
camera với train, chỉ khác nguồn dữ liệu đầu vào.

**Câu hỏi tự kiểm tra:**
- Vẽ (trên giấy hoặc mô tả bằng lời) sơ đồ: `Scene.__init__` gọi những hàm nào, theo thứ tự nào,
  từ những file nào?

---

## Đơn vị 9 — Sinh ảnh nộp bài (inference) & chấm điểm

**Mục tiêu:** hiểu sự khác biệt giữa "render lúc train" (Đơn vị 6-7, luôn có ảnh thật để so sánh)
và "render lúc inference" (không có ảnh thật, chỉ có pose).

**File:**
- [`src/render_submission.py`](src/render_submission.py) — load checkpoint (Đơn vị 5's
  `load_ply`) → với mỗi dòng trong `test_poses.csv` (Đơn vị 3's `readTestPosesCSV`) → dựng
  `Camera` (Đơn vị 4, `image=None` vì private_set1 không có ảnh thật) → `render()` (Đơn vị 6) →
  lưu PNG, đặt tên `0001.png, 0002.png...` + `index_map.json` (map lại tên gốc).
- [`src/evaluate.py`](src/evaluate.py) — chỉ chạy được nếu có ảnh ground-truth thật
  (`public_set`): so từng cặp (render, ground-truth) bằng `psnr` (`utils/image_utils.py`), `ssim`
  (`utils/loss_utils.py`, dùng lại đúng hàm từ lúc training), `lpips` (thư viện ngoài) → tính
  điểm theo đúng công thức đề bài.

**Câu hỏi tự kiểm tra:**
- Vì sao `render_submission.py` gọi được `render()` giống hệt lúc train dù không có ảnh ground
  truth? (gợi ý: `render()` không cần biết ảnh thật để tạo ra ảnh render)
- `evaluate.py` dùng lại hàm `ssim` từ `loss_utils.py` (vốn viết cho training loss) — việc dùng
  lại này có hợp lý không, vì sao?

---

## Đơn vị 10 — Tổng hợp: vẽ lại toàn bộ luồng 1 lần

**Mục tiêu:** không đọc file mới, chỉ ghép lại toàn bộ 9 đơn vị trên thành 1 bức tranh.

Tự làm bài tập sau (không nhìn code, chỉ dùng trí nhớ, rồi đối chiếu lại nếu sai):

> Vẽ sơ đồ khối (boxes + mũi tên) từ `file .JPG + .bin thô trên đĩa` đi qua bao nhiêu bước, qua
> những file/hàm nào, để cuối cùng ra được **1 số điểm duy nhất** trong `eval_report.json`.

Nếu vẽ được đầy đủ và giải thích được từng mũi tên bằng lời — đã hiểu hết toàn bộ pipeline.

**Bảng tra cứu nhanh (file → vai trò 1 dòng):**

| File | Vai trò |
|---|---|
| `scene/colmap_loader.py` | Đọc byte thô COLMAP |
| `preprocess/undistort.py` | Sửa méo ảnh |
| `scene/dataset_readers.py` | COLMAP/CSV → `CameraInfo` chung |
| `utils/graphics_utils.py` | Công thức ma trận camera thuần toán |
| `scene/cameras.py` | `CameraInfo` → `Camera` (có ma trận sẵn) |
| `utils/camera_utils.py` | Cầu nối load ảnh + build `Camera` hàng loạt |
| `scene/gaussian_model.py` | Lưu trữ + densify/prune N Gaussian |
| `utils/general_utils.py` | Hàm toán phụ trợ cho `GaussianModel` (build_rotation...) |
| `utils/sh_utils.py` | Công thức spherical harmonics (màu theo góc nhìn) |
| `gaussian_renderer/__init__.py` | Chọn backend render |
| `gaussian_renderer/_cuda_backend.py` | Gọi CUDA rasterizer gốc |
| `gaussian_renderer/cpu_rasterizer.py` | Renderer tự viết (CPU, chậm, để học/test) |
| `utils/loss_utils.py` | L1/SSIM cho training loss |
| `utils/image_utils.py` | PSNR |
| `arguments/__init__.py` | Toàn bộ hyperparameter (CLI) |
| `scene/__init__.py` | Điều phối: data → model → (train/save) |
| `train.py` | Vòng lặp training |
| `render_submission.py` | Sinh ảnh nộp bài |
| `evaluate.py` | Tính điểm cuối cùng |

---

## Gợi ý cách học hiệu quả

- Với mỗi đơn vị: đọc code trước, đoán output, rồi mới chạy lệnh gợi ý để xem có đúng không.
- Dùng debugger (breakpoint) hoặc `print()` thêm tạm thời để in ra `shape`/giá trị thật của tensor
  ở từng bước — nhìn số thật giúp nhớ lâu hơn đọc code suông.
- Sau Đơn vị 5-6, thử tự sửa 1 dòng nhỏ (vd. đổi màu nền, đổi hệ số 0.3 trong low-pass filter ở
  `cpu_rasterizer.py`) rồi chạy lại xem ảnh render đổi thế nào — cách học nhanh nhất là phá rồi sửa.