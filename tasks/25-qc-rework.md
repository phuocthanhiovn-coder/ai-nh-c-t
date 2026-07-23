# Task 25 — Rework QC scorer (chấm lại 3 tiêu chí hỏng, calib trên 940 cặp)

**Ngày giao:** 23/07/2026 · **Trạng thái:** ⬜ CHƯA LÀM (spec sẵn, chờ làm khi quay lại)

## Bối cảnh (số liệu thật, chạy 22/07 trên 940 cặp — `outputs/qc_calib_940.csv` + log)

Chạy `tools/qc_calib_940.py`: score cả before (xấu biết trước) lẫn after (chuẩn AutoHDR)
để xem từng tiêu chí có TÁCH được 2 nhóm không. Kết quả:

| metric | before p10/p50/p90 | after p10/p50/p90 | phán quyết |
|---|---|---|---|
| exposure_score | 47.7 / 82.7 / 99.6 | 94.5 / 99.3 / 99.9 | ✅ tách tốt nhất — GIỮ |
| color_cast_score | 45.3 / 88.3 / 97.5 | 76.8 / 94.9 / 100 | 🔄 tách vừa — giữ, tinh chỉnh |
| tilt_score | 88.7 / 97.6 / 99.9 | 92.0 / 98.3 / 99.9 | 🔄 tách yếu nhưng vô hại |
| overall | 77.3 / 86.3 / 92.7 | 85.1 / 90.6 / 95.6 | ⚠️ chồng lấn nặng |
| blur_score | 97.9 / 100 / 100 | 100 / 100 / 100 | ❌ MÙ hoàn toàn |
| washout_score | 64.6 / 91.6 / 98.3 | 47.6 / 94.2 / 99.3 | ❌ p10 NGƯỢC (phạt oan after) |
| noise_score | 41.8 / 74.0 / 84.0 | 27.9 / 71.7 / 83.8 | ❌ NGƯỢC nhẹ (phạt texture) |

## Yêu cầu sửa

1. **blur_score:** `BLUR_REF_VAR=260` calib từ thời 49 cặp — vô dụng: mọi ảnh đều ~100.
   Đo lại trên chuẩn mới (22/07): ảnh giao đạt yêu cầu có Laplacian var toàn ảnh
   ~130–550 (xem CLAUDE.md mục bài học "cấm demo bằng data/pairs"); ảnh mờ khách chê
   ~11–42. Yêu cầu: score phải TÁCH được 2 vùng đó (đo trên đúng cỡ ảnh thật,
   KHÔNG đo trên proxy thu nhỏ — thu nhỏ là hết mờ). Acceptance: chấm 940 cặp,
   before p50 < 50, after p50 > 80 trên blur_score mới.
2. **washout_score:** đang phạt gu sáng-airy của AutoHDR (p10 after = 47.6 tệ hơn
   before 64.6). Rework: washout = sáng + ĐỘ BÃO HÒA THẤP + tương phản cục bộ thấp
   ĐỒNG THỜI, không phải chỉ sáng. Tham khảo `finish_detail` (local_std) làm tín hiệu.
   Acceptance: p10 after > p10 before, và bản render pilot bạc màu cũ
   (`checkpoints/auto_enhance.pt`) phải bị chấm < 50.
3. **noise_score:** texture/sharpening hợp lệ đang bị đếm là nhiễu (after thấp hơn
   before). Gate theo vùng phẳng (đo nhiễu CHỈ trên vùng ít gradient), bỏ qua vùng texture.
   Acceptance: after p50 >= before p50 trên 940 cặp.
4. **overall + needs_human:** tính lại trọng số sau khi 3 metric trên sửa xong;
   needs_human hiện gần như không bao giờ bật (p90=0). Acceptance: overall tách
   before/after với khoảng cách p50 >= 15 điểm; needs_human bật trên >=80% ảnh
   thuộc nhóm "hỏng nhân tạo" (tự chế 20 ảnh hỏng: bạc màu, tối om, nghiêng 5°,
   nhiễu nặng, mờ gaussian).

## Ràng buộc
- Giữ nguyên hợp đồng `score(img_f32_01_bgr) -> dict` các key cũ (service đang đọc).
- KHÔNG chấm trên proxy < 1024px cho blur (lý do ở mục 1).
- Test trước khi báo xong: chạy lại `tools/qc_calib_940.py` + bảng mới vào cuối file này,
  kèm 5 ảnh minh họa nhóm bị chấm sai trước/sau khi sửa. Architect sẽ TỰ MỞ ẢNH đối chiếu.
