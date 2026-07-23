# Task 18 — OBJECT REMOVAL specialist (LaMa inpainting, self-hosted CPU — AutoHDR's "removals")

**Assigned to:** Worker R (Sonnet on Max) · **Review:** Claude architect · **Read `CLAUDE.md` first.**

## Context
AutoHDR's removals run mask-based inpainting (they even outsource to fal.ai). We self-host LaMa — it is NOT diffusion, keeps resolution, works tiled on CPU. `requirements.txt` already lists `simple-lama-inpainting`. Principle #1 still holds: generative touches ONLY the masked crop, composited back onto the untouched master.

## OPERATOR CONTRACT — with a mask twist
`apply(img float32 [0,1] HxWx3 BGR, params) -> same shape`. params: `mask_path` (str, path to a grayscale mask image; white=remove) — the planner/UI supplies masks; the operator must NOT invent masks. If `mask_path` missing/unreadable/empty-mask → return unchanged with a warning.

## Files (`ai_engine/specialists/object_removal/`)
1. `remover.py`:
   - Lazy-load simple-lama once (module-level cache). `pip install simple-lama-inpainting` if missing (do it in your session, and note the exact version in the report). Torch is CPU — fine.
   - Pipeline per call: read mask → binarize + dilate slightly (LaMa likes a margin) → **CROP around the mask bbox with padding** (do NOT feed the whole 2048px frame if the object is small — crop, inpaint, paste back) → run LaMa on the crop (its own size handling is fine within the crop) → paste result into a copy of the master ONLY where the (feathered) mask is on, via `ai_engine/core/quality.composite_mask` → assert bit-identical outside feathered mask; assert output size == input size.
2. `make_test_masks.py` — build 3 REAL test cases from `data/pairs/before/`: pick images with a discrete object (e.g., a car on the street shot, a bin, a small wall fixture). Draw the mask programmatically (filled polygon/rect over the object you chose by looking at the image with the Read tool first — hardcode the coords you chose, that's fine for tests). Save masks to `outputs/removal_samples/mask_<name>.png`.
3. `run_samples.py` — run the 3 cases, save `outputs/removal_samples/<name>.jpg` = [original | mask overlay | removed] 1800px + a 100% crop of the removed area.

## Acceptance (run + LOOK)
- [ ] Object gone; fill looks plausible (continues floor/wall/asphalt texture); NO smeared rainbow blur.
- [ ] 100% crop: the seam is invisible or near-invisible; pixels outside mask bit-identical (print assert).
- [ ] Runtime per image printed (CPU seconds). Report honest limits (LaMa struggles with big structured occlusions etc).
- [ ] Register nothing in the orchestrator registry yet (architect will wire it — mask params need UI thought). Just build + prove the specialist.

## DO NOT
- No files outside `ai_engine/specialists/object_removal/` + `outputs/removal_samples/` + report here. `data/` read-only. `cv2.setNumThreads(2)`; torch threads 2 (`torch.set_num_threads(2)`). One python process at a time. Model weights download is allowed (state the size).

---

## BÁO CÁO (Worker R, 2026-07-14)

### Đã làm
- `remover.py`: `apply(img, {"mask_path": ...})` — đọc mask xám → binarize (thresh 127) → dilate ellip 8px (`MASK_DILATE_PX`) → bbox mask + pad 60px (`CROP_PAD_PX`) → **crop quanh bbox** (KHÔNG đưa cả frame 2048px vào LaMa) → chạy LaMa trên crop → feather mask (Gaussian sigma=4px, `FEATHER_PX`) → `composite_mask` dán lại vào bản copy của ảnh gốc full-res → `assert` bit-identical ngoài feathered mask (so từng pixel `np.array_equal`) + `assert` size khớp. Nếu `mask_path` thiếu/không đọc được/mask rỗng → trả ảnh **không đổi** + in cảnh báo (đã test path này, không crash).
- `make_test_masks.py`: 3 case thật từ `data/pairs/before/` — mask vẽ bằng rect/ellipse hardcode toạ độ, chọn bằng mắt qua Read tool + crop-zoom kiểm tra toạ độ trước khi hardcode (không đoán mò).
- `run_samples.py`: chạy 3 case qua `remover.apply`, ghi panel `[Before | Mask overlay | Removed]` + crop 100% vùng xoá vào `outputs/removal_samples/<name>.jpg`.
- **Cài đặt:** `simple-lama-inpainting==0.1.0` (đã có sẵn trong venv, không cần cài lại), `torch==2.13.0+cpu`. Model weight `big-lama.pt` tải tự động lần đầu chạy, **~197MB** (205,803,670 bytes), lưu `~/.cache/torch/hub/checkpoints/big-lama.pt`.
- **Vấn đề gặp phải + cách sửa:** thư viện `simple_lama_inpainting.SimpleLama.__init__` gọi `torch.jit.load(model_path)` **không có `map_location`** → file `.pt` được trace sẵn với tensor CUDA bên trong, load thẳng trên máy CPU-only crash (`NotImplementedError: Could not run 'aten::empty_strided' with arguments from the 'CUDA' backend`). Đây là máy production **không có GPU** (theo rule #3 của session). Đã viết class `_CpuLama` riêng trong `remover.py` (không đụng site-packages) — tự `torch.jit.load(model_path, map_location=torch.device("cpu"))`, tái dùng `prepare_img_and_mask` gốc của thư viện cho phần pad/normalize. Sau khi vá: load model 1.4s, chạy ổn định CPU.

### Kết quả thật (đã LOOK bằng Read tool, không chỉ tin số liệu)
| case | ảnh nguồn | mask px (người vẽ) | crop size | runtime CPU | assert bit-identical |
|---|---|---|---|---|---|
| switch_plate | 20260703-DSC1105.jpg | 1,295 | 171×173 | 6.13s | PASS |
| marquee_letter | 20260703-DSC1226.jpg | 40,920 | 291×400 | 5.59s | PASS |
| decor_bowl | 20260703-DSC1161.jpg | 44,465 | 433×327 | 4.91s | PASS |

(Lần chạy thứ 2 độc lập để zoom-crop kiểm tra bằng mắt cho runtime tương tự 4-7.5s/case — dao động do máy đang chạy song song việc khác lúc benchmark.)

- **switch_plate:** SẠCH. Ổ cắm/công tắc tối màu biến mất hoàn toàn, tường vàng nhạt liền mạch, không thấy seam. Công tắc trắng bên cạnh (không nằm trong mask) giữ nguyên đúng như kỳ vọng. Đây là ca dễ nhất (nền phẳng đồng màu).
- **marquee_letter:** TỐT, có 1 điểm nhỏ đáng lưu ý. Chữ "C" đèn trang trí biến mất sạch trên tường trắng. Tuy nhiên khi zoom 100%: phần trên khu vực fill có 1 vệt mờ/blotchy nhẹ (không phải rainbow-blur, chỉ là texture hơi loang so với tường xung quanh) — do đúng chỗ đó trong ảnh gốc có bóng đổ nhẹ của chữ C lên tường, LaMa không tái tạo lại độ mịn 100%. Dây kéo đèn (sợi dây trắng mảnh) và 1 hộp điện nhỏ nằm NGOÀI mask (không xoá, đúng vì không nằm trong mask do user không vẽ) vẫn còn — đúng hành vi theo hợp đồng operator (không tự bịa mask).
- **decor_bowl:** ĐÂY LÀ CA KHÓ, giới hạn thật của LaMa. Mask phủ cả bát sứ ĐEN lẫn một phần cửa sổ + toà nhà phía sau (occlusion lớn, có cấu trúc — đúng loại CLAUDE.md cảnh báo LaMa yếu). Bát biến mất, LaMa CÓ cố gắng tiếp tục các đường thẳng đứng của khung cửa sổ/mặt toà nhà phía sau — không bị nhoè cầu vồng, nhưng kết quả **mờ/soft hơn** vùng xung quanh thật, không khớp hoàn toàn hoạ tiết mullion cửa sổ hay vân gỗ của kệ; đặc biệt vùng đáy bát (gần bóng đổ trên mặt kệ gỗ) tái tạo thành một mảng bóng mờ chung chung thay vì vân gỗ sắc nét. **Không nên dùng con này cho vật thể lớn che cấu trúc phức tạp (cửa sổ, gạch, vân gỗ chi tiết) mà không có bước refine riêng.**

### Giới hạn trung thực (đúng tinh thần acceptance)
1. LaMa (big-lama, CPU, chạy trên crop chứ không phải full-res) hoạt động rất tốt với occlusion nhỏ trên nền phẳng/đồng chất (switch_plate, marquee_letter).
2. Với occlusion lớn che cấu trúc lặp lại phức tạp (cửa sổ nhiều mullion, vân gỗ) — decor_bowl — kết quả plausible nhưng có thể nhìn ra là "mờ hơn" khi zoom 100%, không đạt chất lượng seamless tuyệt đối. Đây là giới hạn model, không phải bug của operator.
3. Vệt mờ nhẹ ở marquee_letter cho thấy composite feather 4px đôi khi để lộ 1 viền transition mềm nếu LaMa không khớp y hệt tông màu nền ngay sát biên mask — có thể cải thiện bằng cách tăng margin dilate hoặc feather ở bản production, nhưng KHÔNG sửa ở task này vì ngoài scope.
4. Runtime ~5-7s/ảnh trên crop nhỏ (CPU, torch threads=2) — chưa test crop lớn (vật thể to chiếm nửa khung hình), có thể chậm hơn nhiều.

### Quyết định khi mơ hồ (theo rule 1 của prompt)
- Spec không nói rõ dilate bao nhiêu px / feather bao nhiêu px / pad crop bao nhiêu — chọn số nhỏ, an toàn: dilate 8px, pad 60px, feather 4px (đơn giản nhất thoả acceptance "seam vô hình/gần vô hình" + "margin cho LaMa").
- Mask test cho `decor_bowl` dùng ellipse (không phải rect) vì object là bát tròn — rect sẽ đưa quá nhiều background (cửa sổ) vào vùng inpaint không cần thiết, chọn ellipse bám sát silhouette hơn (vẫn hardcode toạ độ theo đúng tinh thần spec).
- Chưa đăng ký vào `ai_engine/orchestrator/registry.py` theo đúng yêu cầu "Register nothing... architect will wire it".

TASK18=DONE + 3/3 case PASS assert bit-identical, runtime 4.91–6.13s CPU/ảnh, model big-lama.pt ~197MB, decor_bowl cho thấy giới hạn thật với occlusion cấu trúc lớn (mờ hơn, không rainbow-blur).
