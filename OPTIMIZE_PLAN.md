# OPTIMIZE_PLAN.md — Lộ trình nâng điểm 58 → 80

Đọc `CLAUDE.md` (kiến trúc/data) và `PLAN.md` (đã hoàn thành — baseline chạy end-to-end) trước
khi đọc file này. File này chỉ nói về **tối ưu chất lượng sau khi baseline đã chạy được**,
tương ứng PLAN.md Phase 6 nhưng chi tiết hơn nhiều, viết lại sau khi có dữ liệu thật từ các lần
train trên GPU (không còn là suy đoán thuần).

## 0. Trạng thái hiện tại (đã làm, tính đến 2026-07-12)

Đã ship và verify (compile + synthetic test, **chưa có số liệu `eval_public.sh` thật** vì các
lần train trên GPU vừa qua đều bị gián đoạn bởi bug — xem §1):

- `src/utils/antialiasing.py` — bù opacity kiểu Mip-Splatting (2D-only, không phải bản đầy đủ),
  tính bằng Python trước khi gọi rasterizer CUDA gốc.
- `arguments/PipelineParams.antialiasing` (mặc định `True`) + `OptimizationParams`:
  `antialiasing_window=5000`, `antialiasing_ramp_iters=2500` — chỉ áp dụng ở cửa sổ cuối
  training, ramp tuyến tính, tránh sốc loss.
- `densify_grad_threshold=0.00015` (giảm nhẹ từ gốc 0.0002), `densify_until_iter=20_000`
  (tăng từ gốc 15_000), `max_gaussians=9_000_000` (hard cap chống OOM), `min_opacity_prune`/
  `densify_max_screen_size` được đưa ra CLI.
- `train.py --seed` (mặc định 0) — giảm phần lớn (không phải toàn bộ, xem docstring
  `set_seed()`) nhiễu giữa các lần train.
- Log validation in kèm số gaussian + train loss EMA (`validate()` trong `train.py`).
- `scripts/train_public.sh`, `scripts/eval_public.sh` (so sánh 3 checkpoint 7000/15000/30000).

**Việc cần làm ngay, trước khi đọc tiếp §2 trở đi**: chưa có một lần train nào chạy trọn vẹn
30000 iteration + `eval_public.sh` ra điểm số thật kể từ khi thêm Tier 1. Toàn bộ ưu tiên bên
dưới có thể đổi thứ tự sau khi có số liệu thật đầu tiên — xem Phase 1.

## 1. Nguyên tắc ưu tiên

Không đoán mò tiếp — mọi thay đổi từ giờ phải đi kèm số đo trên `public_set` qua
`scripts/eval_public.sh` trước khi áp dụng cho `private_set1`. Thứ tự ưu tiên dựa trên
**impact/effort**, không phải "kỹ thuật nào hay nhất":

| Lỗi quan sát được | Nguyên nhân gốc (đã xác nhận hoặc nghi vấn mạnh) | Kỹ thuật sửa | Effort |
|---|---|---|---|
| Dây điện mờ/ghost | SfM (COLMAP) gần như không match được điểm trên dây (mảnh, ít tương phản, đu đưa giữa các frame) → không có Gaussian khởi tạo ở đó | Densification nhạy hơn theo pixel-coverage (AbsGS/Pixel-GS style) + loss ưu tiên vùng biên | Trung bình |
| Sọc tôn nhòe | Texture tần số cao bị alias; **chưa loại trừ khả năng ảnh gốc drone đã motion-blur sẵn** | Antialiasing (đã có, 2D-only) + kiểm tra ảnh train gốc + edge-aware loss | Thấp→Trung bình |
| Viền tấm anten nhòe | Floater/boundary do thiếu ràng buộc hình học ở biên vật thể mỏng nhìn nghiêng | Scale/opacity regularization + (tùy chọn, effort cao) StopThePop | Thấp→Cao |
| Góc xa mờ | Vùng ít camera quan sát → Gaussian to, under-constrained | Scale regularization + prune mạnh hơn theo world-size | Thấp |

## Phase 1 — Đo baseline thật (BLOCKING, làm trước mọi thứ khác)

Không đổi code. Mục tiêu duy nhất: có **một** con số real từ `eval_public.sh` để so sánh mọi
thay đổi sau này.

1. Train trọn vẹn 5 scene `public_set` với code hiện tại (`bash scripts/train_public.sh`),
   theo dõi log không còn OOM / loss spike bất thường ở vùng 25000-27500 (đã fix ở lượt trước,
   cần xác nhận bằng mắt trên log thật).
2. Chạy `bash scripts/eval_public.sh` → có bảng PSNR/SSIM/LPIPS/score cho 3 checkpoint.
3. Chạy lại **đúng cấu hình đó nhưng `--no_antialiasing`** (train lại, hoặc ít nhất render lại
   + eval lại checkpoint 30000 với `--no_antialiasing` ở `render_submission.py` — lưu ý: nếu
   không train riêng một lần không-AA, checkpoint 30000 hiện có **đã được fit dưới ảnh hưởng
   của AA ramp** nên render không-AA từ checkpoint đó sẽ hơi lệch, chỉ mang tính tham khảo
   nhanh, không phải A/B chuẩn — muốn A/B chuẩn phải train lại từ đầu với `--no_antialiasing`).
4. Ghi lại 2 con số (có AA / không AA) vào một chỗ cố định (vd. `eval_report_public_iter_30000.json`
   đổi tên lưu lại, hoặc note tay) để tham chiếu cho các Phase sau.

**Xong khi**: có ít nhất 1 bảng điểm `public_set` đầy đủ 5 scene, biết được AA hiện tại đang
+/- bao nhiêu điểm so với không AA — quyết định giữ/bỏ/đầu tư thêm AA dựa trên số này, không
dựa trên suy đoán.

## Phase 2 — Densification nhạy theo cấu trúc mảnh (ROI cao nhất, effort trung bình)

Đây là nút thắt chính theo cả phân tích ảnh lỗi ban đầu lẫn nhận định vừa rồi. Vanilla 3DGS
densify theo **gradient trung bình** (`xyz_gradient_accum / denom`), không phân biệt "Gaussian
này đại diện cho vật thể mảnh chiếm ít pixel" — đây chính là vấn đề mà paper AbsGS/Pixel-GS
giải quyết.

1. **`scene/gaussian_model.py::add_densification_stats`**: hiện dùng
   `torch.norm(viewspace_point_tensor.grad[...,:2], dim=-1)` (gradient trung bình theo số lần
   nhìn thấy). Đổi sang tích lũy **giá trị tuyệt đối lớn nhất** quan sát được (kiểu AbsGS) thay
   vì trung bình cộng dồn rồi chia — điểm mảnh, ít được nhìn thấy trực diện, vẫn giữ được tín
   hiệu gradient mạnh thay vì bị pha loãng.
2. Cân nhắc trọng số densify theo **screen-space coverage** (Pixel-GS): Gaussian nào chiếm ít
   pixel nhưng gradient cao → ưu tiên split trước Gaussian chiếm nhiều pixel gradient thấp.
   Effort cao hơn AbsGS thuần, có thể để sau nếu AbsGS đã đủ cải thiện.
3. Test trên `HNI0366` (có tấm phát sóng bị mờ viền rõ nhất) + `HCM0249` (dây điện) trước, vì
   đây là 2 case rõ nhất để đánh giá bằng mắt trước khi tin vào số liệu.

**Xong khi**: `eval_public.sh` cho LPIPS thấp hơn baseline Phase 1 rõ rệt trên ít nhất 3/5 scene
public_set, và nhìn bằng mắt thấy dây/viền sắc nét hơn ở ảnh render so sánh trực tiếp.

## Phase 3 — Loss ưu tiên vùng biên/chi tiết (ROI cao, effort thấp)

1. **`utils/loss_utils.py`**: thêm hàm tính edge map (Sobel/Laplacian) trên ảnh ground-truth,
   dùng làm trọng số nhân vào L1 loss (`l1_loss` hiện tại là `.mean()` phẳng, không phân biệt
   vùng). Vùng có gradient ảnh lớn (biên vật thể, sọc tôn, dây điện) được nhân trọng số cao
   hơn, ép model học đúng các chi tiết khó thay vì tối ưu trung bình toàn ảnh (vốn bị vùng
   phẳng/mái nhà lớn chi phối vì chiếm nhiều pixel hơn).
2. Không đổi `ssim`/D-SSIM — giữ nguyên vì đã hoạt động hợp lý, chỉ thêm 1 loss term mới, có
   trọng số riêng (`--lambda_edge`, mặc định nhỏ ~0.05-0.1, cần tune).
3. Đây là thay đổi rẻ (không đụng CUDA, không đụng densify), nên làm **song song** Phase 2, dễ
   tách A/B riêng.

**Xong khi**: có `--lambda_edge` cấu hình được, so sánh có/không qua `eval_public.sh`, giữ nếu
LPIPS cải thiện mà không phá vùng phẳng (theo dõi PSNR tổng thể không tụt).

## Phase 4 — Regularization chống floater/mờ vùng ít quan sát (ROI trung bình-cao, effort thấp)

Nhắm trực tiếp vào "góc xa mờ" và một phần "viền tấm anten mờ":

1. Thêm loss phạt Gaussian có `scaling` lớn bất thường so với mật độ điểm 3D lân cận (hoặc so
   với `scene.cameras_extent`) — Gaussian to bất thường thường là floater/under-constrained.
2. Xem lại `densify_max_screen_size` (hiện 20px) và `min_opacity_prune` (0.005) — thử prune
   chặt hơn cho Gaussian ở vùng world-scale lớn (`get_scaling.max(dim=1).values > k * extent`
   với `k` nhỏ hơn 0.1 hiện tại) để loại nhanh hơn Gaussian "cloud" mờ ở rìa scene.
3. **Không** động vào camera sampling (đề xuất "camera sampling cho vùng xa" trong bản phản
   biện) ở giai đoạn này — dataset không cho phép thêm ảnh, và random uniform sampling qua hết
   1 epoch đã đảm bảo mọi camera được huấn luyện đều; trọng số hóa sampling theo "khó" cần
   thêm cơ chế đo per-camera loss (phức tạp, để Phase 6 nếu cần).

**Xong khi**: `eval_public.sh` cho thấy giảm floater bằng mắt ở vùng rìa ảnh test, số lượng
Gaussian cuối cùng không tăng bất thường so với Phase 2.

## Phase 5 — Xác minh giả thuyết dữ liệu (effort thấp, làm song song bất kỳ lúc nào)

1. Mở trực tiếp vài ảnh `train/images/*.JPG` gốc (không phải ảnh render) của `HCM0249`
   vùng có sọc tôn — kiểm tra bằng mắt xem **ảnh gốc drone có bị motion-blur sẵn hay không**.
   Nếu có, không kỹ thuật render nào sửa được — cần loại ảnh blur nặng khỏi tập train (đo bằng
   variance-of-Laplacian, ngưỡng đơn giản) hoặc chấp nhận giới hạn này.
2. Xác nhận lại `PSNR_max` với ban tổ chức nếu có kênh liên hệ — hiện đang dùng giá trị giả
   định (mặc định script `50`), ảnh hưởng trực tiếp điểm cuối nhưng nằm ngoài phạm vi code.

**Xong khi**: biết chắc sọc tôn mờ là do render hay do dữ liệu gốc — quyết định có đầu tư thêm
vào Phase 2/3 cho vấn đề này hay chấp nhận giới hạn.

## Phase 6 — Chỉ làm nếu Phase 2-4 chưa đủ để chạm 80 điểm (effort cao, rủi ro cao hơn)

Theo đúng thứ tự ROI giảm dần, **không làm tắt trước khi có số liệu Phase 1-4**:

1. **Mip-Splatting đầy đủ** (3D smoothing filter + rasterizer riêng
   `autonomousvision/mip-splatting`) — sửa đúng phần antialiasing hiện đang là approximation,
   nhưng cần đổi submodule CUDA khác, thêm `get_opacity_with_3D_filter`/
   `get_scaling_with_3D_filter` vào `GaussianModel`, tính filter 1 lần dựa trên toàn bộ camera
   train sau khi `Scene` load xong. Effort/rủi ro cao nhất trong list — chỉ làm nếu A/B Phase 1
   cho thấy bản 2D-only hiện tại đã có ích nhưng chưa đủ.
2. **StopThePop** (sort/rasterizer viết lại) — nhắm đúng "viền tấm anten mờ" (popping artifact
   ở object mỏng chồng lấn) nhưng phải đổi hẳn CUDA kernel, effort cao.
3. **Depth regularization** dùng sparse COLMAP points hoặc mono-depth prior — thêm dependency
   mới, ổn định hình học vùng ít texture (dây điện trên nền trời).
4. Re-run COLMAP feature extraction với tham số nhạy hơn để bắt thêm điểm trên dây điện — chạy
   được trên CPU (không cần GPU) nhưng chậm, kết quả không đảm bảo (dây điện vốn khó match kể
   cả với COLMAP nhạy).
5. Optional/rẻ: reset Adam `exp_avg`/`exp_avg_sq` của riêng `opacity` tại đúng
   `antialiasing_from_iter` (theo góp ý "objective đổi nhưng không reset optimizer state") — thử
   nhanh, effort thấp, lợi ích chưa rõ, xếp cuối vì không phải nút thắt chính.

## 2. Cách đo tiến độ (bắt buộc cho mọi thay đổi từ Phase 2 trở đi)

- Luôn chạy trên `public_set` qua `scripts/eval_public.sh` trước khi đụng `private_set1`.
- So sánh **cùng seed** (`--seed 0` mặc định) giữa các lần thay đổi để tách được cải thiện thật
  khỏi nhiễu run-to-run (~1 PSNR unit, xem `train.py::set_seed` docstring).
- Ghi lại mỗi lần thử nghiệm: cấu hình thay đổi, `eval_report_public_iter_*.json`, nhận xét
  bằng mắt trên ít nhất 1 ảnh có dây điện (`HCM0249`), 1 ảnh có sọc tôn, 1 ảnh có tấm anten
  (`HNI0366`) — không chỉ tin vào con số trung bình, vì đúng 4 loại lỗi này là thứ đang kéo
  điểm xuống, có thể bị pha loãng trong điểm trung bình toàn ảnh.

## 3. Rủi ro/giả định còn mở

- `PSNR_max` chưa có giá trị chính thức (CLAUDE.md §5) — điểm tuyệt đối có thể đổi khi có giá
  trị thật, nhưng thứ tự ưu tiên tương đối giữa các kỹ thuật trong plan này không phụ thuộc vào
  đó.
- Ngân sách compute/thời gian train cho 8 scene private_set1 × nhiều lần thử nghiệm A/B — nên
  thử nghiệm trên 1-2 scene public_set đại diện trước khi chạy full 5 scene, càng chưa nói đến
  8 scene private.
- Chưa có số liệu Phase 1 → toàn bộ độ ưu tiên Phase 2-6 có thể cần điều chỉnh lại sau khi có
  con số thật đầu tiên.
