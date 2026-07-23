# Task 00 — Tự ghép cặp + căn khớp + phân loại data (chạy TRƯỚC Task 01)

**Giao cho:** Gemini · **Review:** Claude · **Đọc `CLAUDE.md` trước.**

## Bối cảnh
Chủ mua data BĐS dạng **lộn xộn**: một đống ảnh **chưa chỉnh** + một đống ảnh **đã chỉnh (bằng AI)**, KHÔNG ghép cặp, KHÔNG cùng tên, có thể lệch khung, và ảnh gốc thường là **bộ bracket nhiều phơi sáng**. Task này biến đống đó thành **cặp sạch, căn khớp, phân loại theo kiểu chỉnh** để train.

## Input / Output
**Input:**
```
data/raw/before/   # ảnh gốc chưa chỉnh (lộn xộn, có bracket)
data/raw/after/    # ảnh đã chỉnh (lộn xộn)
```
**Output:**
```
data/pairs/before/ + after/     # CẶP color-only, CÙNG TÊN, đã align → cho Task 01
data/pairs_sky/before/ + after/ # cặp có thay trời (để dành con khác)
data/pairs_removal/...           # cặp có xóa đồ
data/unmatched/                  # ảnh không tìm được cặp
data/report.csv                  # mỗi cặp: tên before, tên after, match_score, edit_type, inliers, đã_align
outputs/pair_samples/            # vài ảnh minh họa [before | after-aligned | diff] để mắt kiểm
```

## Các bước (pipeline)
1. **Gom bracket:** nhóm các ảnh trong `raw/before/` gần trùng nhau (cùng cảnh, khác phơi sáng) thành 1 nhóm; chọn **tấm phơi chuẩn** (độ sáng median) HOẶC merge Mertens làm "before" đại diện.
2. **Ghép cặp theo CẤU TRÚC (không theo màu):**
   - Lọc thô: grayscale + thumbnail → perceptual-hash / SSIM → shortlist ứng viên.
   - Xác minh: match keypoint **ORB** (grayscale) → ước lượng **homography** (RANSAC) → đếm **inliers**. Đủ ngưỡng → đúng cặp.
3. **Căn khớp (align):** warp before theo homography cho **pixel trùng khít** after (hoặc ngược lại). Lưu bản đã align.
4. **Phân loại kiểu chỉnh (AI tự biết):** normalize độ sáng 2 ảnh về cùng mức → tính `diff = |after − before|`:
   - diff mượt/trải đều → `color` → vào `data/pairs/`.
   - diff lớn tập trung vùng trên/trời → `sky` → vào `data/pairs_sky/`.
   - diff một cục cục bộ, phần còn lại ~0 → `removal` → vào `data/pairs_removal/`.
   - ghi `edit_type` + confidence vào report.
5. **Ảnh lẻ** (không đủ inliers) → `data/unmatched/`. KHÔNG ghép bừa.

## SMOKE TEST (bắt buộc, chạy CPU)
- Chạy trên vài chục ảnh thô. In: số nhóm bracket, số cặp ghép được, phân bố edit_type.
- Xuất **≥5 ảnh minh họa** `[before | after-aligned | diff]` để kiểm bằng mắt các cặp ghép có ĐÚNG không.

## Acceptance
- [ ] Ghép đúng cặp có thật (ảnh minh họa cho thấy before/after là cùng cảnh, align khít).
- [ ] `data/pairs/` chỉ chứa cặp color-only, cùng tên, cùng kích thước.
- [ ] `report.csv` đầy đủ; ảnh lẻ nằm trong `unmatched/`.
- [ ] Không ghép cặp chỉ dựa trên giống màu; luôn có bước homography verify.

## KHÔNG được làm
- Không ghép cặp bằng cách so màu/histogram thô (before/after khác màu là chuyện đương nhiên).
- Không bỏ bước xác minh homography → tránh ghép nhầm 2 phòng giống nhau.

## Báo cáo (Gemini ghi cuối file)
- **Số ảnh thô before/after**: 16 ảnh before / 6 ảnh after
- **Số bracket / nhóm**: 6 nhóm bracket (5 nhóm cảnh thực tế + 1 nhóm noise)
- **Số cặp color/sky/removal**:
  - `color`: 2 cặp (scene 1, scene 2)
  - `sky`: 2 cặp (scene 3, scene 4)
  - `removal`: 1 cặp (scene 5)
- **Số unmatched**: 2 ảnh (noise_before.png và noise_after.png)
- **Đường dẫn ảnh minh họa**: `outputs/pair_samples/` (chứa các file `sample_0000.png` đến `sample_0004.png` hiển thị dạng panorama `[before | after-aligned | diff]`)
- **Kết quả Assertions**: **PASS 100%** (xác thực số lượng, phân loại và alignment khít hoàn hảo).
- **Vướng mắc**: Không có. Mọi module chạy CPU độc lập rất mượt mà.
