# Task 21 — FULL-RES DELIVERY PIPELINE (fix the customer's #1 complaint: compressed/soft output)

**Assigned to:** Worker (Fable 5) · **Review:** Claude architect · **Read `CLAUDE.md` first.**

## Context (real customer feedback we must fix)
A real customer said our output is "compressed, pixels break when zoomed, very blurry" while AutoHDR delivers full quality. ROOT CAUSE (already diagnosed): our demo ran on 2048px downscaled training images. The engine + specialists are resolution-independent (operator-not-pixel), so applying the chain to the ORIGINAL full-res photo yields full-res sharp output. We must PRODUCTIONIZE this as a delivery pipeline.

## Goal
`ai_engine/delivery/deliver.py` — take an ORIGINAL full-res photo (or a whole folder = one listing) and produce full-quality edited images. NO downscaling anywhere in the path. Output JPEG quality 100 (or PNG on request).

## Requirements
1. `deliver_image(in_path, out_path, scene=None, opts=None)`:
   - Read the ORIGINAL at full resolution (cv2.IMREAD_COLOR, keep native size). Never resize the master.
   - Build the deterministic edit plan (all specialists already exist in `ai_engine/orchestrator/registry.py`; use their `apply` via the engine or directly). A sensible default chain: `auto_white_balance` → `denoise` (light) → `straighten` (skip for aerial) → adaptive tone (`highlights_recover`+`shadows_lift` scaled to the image's dynamic range) → `saturation` (gentle) → `sharpen` (gentle). Do NOT include `auto_enhance` (learned model still washes) unless `opts["use_model"]` is set.
   - Apply every op at FULL resolution on the master. Output size MUST equal input size (assert).
   - Save JPEG q100 by default; PNG if out_path ends .png.
2. `deliver_folder(in_dir, out_dir, opts=None)`:
   - Process every image in in_dir. For CONSISTENCY across a listing, estimate white-balance/exposure params ONCE from a representative subset (or per-image but clamp variation) so all photos of a home look cohesive — document your choice.
   - Print per-file: in size, out size, out KB.
3. CLI: `python -m ai_engine.delivery.deliver --in <file|dir> --out <file|dir> [--png] [--use-model]`.

## Acceptance (run for real + LOOK at outputs with Read)
- [x] On a real full-res source (use `data/newbatch/mixed_probe/01-RAW-Photos/*.jpg`, 3000x2250): output is SAME resolution (3000x2250), file >2MB, visibly sharp at 100% crop (render a 100% crop to prove). Assert out size == in size.
- [x] Folder mode processes all and looks cohesive.
- [x] No `auto_enhance` in default path; output has correct rich color (not washed).
- [x] Honest report at end of this file: sizes, timing, any op that misbehaves on full-res, limits.

## DO NOT
- No files outside `ai_engine/delivery/` + `outputs/delivery/` + report here. `data/` read-only. `cv2.setNumThreads(3)`. Never downscale the master. One python process at a time for your tests.

---

## BÁO CÁO WORKER (Fable 5, 2026-07-14) — ĐÃ CHẠY THẬT, ĐÃ NHÌN ẢNH

### Đã build
`ai_engine/delivery/deliver.py` (+ `__init__.py`). Chain mặc định:
`auto_white_balance` (WB gains + auto-exposure) → `denoise` (0.25, chưa sharpen) → `straighten` (bỏ qua aerial) → `highlights_recover`/`shadows_lift` (adaptive) → `saturation` (0.12) → sharpen (0.3, qua `ds.sharpen` chống halo — KHÔNG dùng `ops_basic.sharpen` vì không có chống halo ở full-res). KHÔNG có `auto_enhance` trừ khi `--use-model`.

### Quyết định thiết kế (chỗ spec mơ hồ → chọn phương án đơn giản nhất, ghi lại đây)
1. **Ước lượng tham số trên proxy 1024px, áp lên master full-res** (đúng nguyên tắc operator-không-pixel). Master không bao giờ bị resize; assert `out_shape == in_shape` trong `deliver_image`.
2. **Cohesion folder = per-image-clamped, KHÔNG phải 1 gain chung:** gain WB từng ảnh bị kẹp về median của subset đại diện (≤7 ảnh cách đều) ±0.10/kênh. Lý do: listing trộn nội thất (đèn vàng) + aerial (trời) — 1 gain cứng sẽ sai ít nhất 1 nhóm. Exposure chuẩn hoá per-image về cùng `target_median=0.42`.
3. **auto_exposure lặp tối đa 3 lần:** 1 lần bị kẹt gamma clamp (0.6) với phòng RẤT tối → median dừng ở 0.27, listing lệch sáng. Sau khi lặp: median nội thất 0.420–0.471 (input 0.098–0.446) — đo thật, xem bảng dưới.
4. **Aerial = heuristic tên file `DJI_*`** (dự án chưa có scene classifier nhóm A). Ảnh drone hãng khác sẽ KHÔNG bị nhận là aerial → vẫn qua straighten (có safety riêng tự trả nguyên ảnh, nhưng không đảm bảo tuyệt đối).
5. **JPEG q100 + chroma 4:4:4** (`IMWRITE_JPEG_SAMPLING_FACTOR_444`) — cv2 mặc định 4:2:0 kể cả q100 → nhòe cạnh màu, đúng lỗi khách chê. File to hơn (~5.9MB → 9.1MB ảnh aerial) nhưng đây là pipeline GIAO HÀNG.

### Số liệu THẬT (máy này, CPU, cv2.setNumThreads(3))
| Ảnh | In | Out | KB | Giây |
|---|---|---|---|---|
| DJI_...0921_D.JPG (aerial) | 3000x2250 | 3000x2250 | 9132 | 8.2 |
| DSC01341.JPG (nội thất) | 3000x2000 | 3000x2000 | 5503 | 7.8 |
| DSC01344 (--png) | 3000x2000 | 3000x2000 | 8335 (PNG) | 10.4 |
| Folder 7 ảnh (5 nội thất + 2 aerial) | — | 100% giữ size | 4620–9132 | 64.5s tổng (~9.2s/ảnh) |

- Median luma output listing: nội thất 0.420/0.427/0.429/0.420/0.471, aerial 0.431/0.425 → **cohesive, đã nhìn contact sheet xác nhận** (`outputs/delivery/crops/listing_contact_sheet.jpg`).
- Crop 100% (không resize) input vs output: `outputs/delivery/crops/*_crop100_{in,out}.png` — ngói mái từng viên rõ, texture tường + ốc bản lề sắc nét, KHÔNG vỡ pixel, KHÔNG halo. Đã mở nhìn từng ảnh bằng mắt.
- Màu: nội thất tối nâu bùn → trắng sáng airy đúng gu BĐS, KHÔNG bạc màu. Aerial giữ màu tự nhiên, hơi ấm hơn.
- `--use-model` chạy được (auto_enhance load checkpoint, +12s/ảnh) nhưng output **BẠC MÀU NẶNG** (đã nhìn: `outputs/delivery/crops/DSC01344_model_preview.jpg` — cây xám, mờ sương). Đúng như cảnh báo CLAUDE.md → giữ mặc định TẮT.
- `python -m ai_engine.conformance_check`: 6 PASS / 0 FAIL (không đụng specialist nào).

### Hạn chế & thất bại (thành thật)
1. **"Folder mode processes all"** chạy trên listing test 7 ảnh trộn (copy vào `outputs/delivery/listing_in/`), KHÔNG phải cả 290 ảnh probe — 290 × ~9s ≈ 43 phút CPU, không có gì trong code phụ thuộc số lượng ảnh. Chưa đo trên listing 290 ảnh thật.
2. **Tốc độ ~8–10s/ảnh 6.7MP trên CPU** — chậm cho volume lớn; cần batch/parallel sau (ngoài scope, luật 1-process).
3. **Straighten trên aerial chỉ né bằng tên file DJI** (điểm 4 trên). Fix thật = con scene-classifier (roster nhóm A).
4. Không op nào misbehave ở full-res trong test; adaptive tone hoạt động (highlights 0.02–0.30, shadows 0.01–0.19 tuỳ ảnh). Chưa test ảnh >12MP (không có trong data probe).
5. Ảnh RẤT tối vẫn có thể dừng dưới target sau 3 vòng lặp exposure (chưa gặp trong test — ảnh tối nhất median 0.098 lên được 0.420).

**TASK21=DONE** — 3000x2250 in → 3000x2250 out, JPEG q100 4:4:4 9132 KB (>2MB), 8.2s/ảnh; folder 7 ảnh 64.5s, median luma 0.420–0.471 cohesive; crop 100% sắc nét đã kiểm bằng mắt.
