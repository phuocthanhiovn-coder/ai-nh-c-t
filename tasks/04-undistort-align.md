# Task 04 — Cứu cặp ảnh rớt align bằng UNDISTORT (nâng yield 28% → mục tiêu ≥60%)

**Giao cho:** Claude-worker (Sonnet) · **Review:** Claude (kiến trúc sư) · **Đọc `CLAUDE.md` trước khi làm.**

## Bối cảnh (đọc kỹ, đây là lý do task tồn tại)
Task 02 ghép cặp before/after: `before` = Mertens-merge từ RAW (còn nguyên **méo ống kính barrel** + chưa nắn phối cảnh), `after` = ảnh AutoHDR xuất (đã **sửa méo ống kính + nắn dọc thẳng**). Vì before cong / after thẳng nên homography + ECC không khớp nổi → Edge-NCC rớt gate 0.50.

Hiện trạng số liệu (`data/report.csv`):
- 12 cặp đạt (`align_ok`) nằm trong `data/pairs/` — **TUYỆT ĐỐI KHÔNG ĐỤNG VÀO.**
- ~49 cặp rớt (`align_low`, score 0.13–0.49) nằm trong `data/review/{before,after}/` (JPG ~2048px, cùng tên file) — **đây là nguyên liệu của task này.**

## Việc phải làm
Viết 2 file mới trong `ai_engine/data_pairing/`:

### 1. `undistort.py`
Hàm `estimate_undistort(before_bgr, after_bgr) -> (k1_best, before_undistorted, score)`:
- Mô hình méo đơn giản: radial (dùng `cv2.undistort` với camera matrix giả định `fx=fy=w`, `cx=w/2`, `cy=h/2`, distCoeffs `[k1, k2=0, 0, 0]`).
- **Grid search k1** trong khoảng `[-0.30, +0.10]` bước 0.02 (làm ở bản thu nhỏ 512px cho nhanh), với MỖI k1: undistort before → chạy lại pipeline align hiện có (ECC refine như `align.py`) → tính Edge-NCC với after.
- Chọn k1 cho NCC cao nhất, tinh chỉnh thêm 1 vòng bước 0.005 quanh đỉnh.
- Áp k1 tốt nhất lên before **full-res** rồi align full-res như Task 02 (scale ma trận đúng S·H·S⁻¹ như code hiện có).
- TÁI DÙNG hàm align/NCC có sẵn trong `align.py` — đừng viết lại logic đã duyệt.

### 2. `rescue_review.py` (script chạy `python -m ai_engine.data_pairing.rescue_review`)
- Duyệt mọi cặp cùng tên trong `data/review/{before,after}/`.
- Với mỗi cặp: estimate_undistort → align lại → tính Edge-NCC mới.
- **NCC ≥ 0.50** → ghi cặp (before đã undistort+align, after giữ nguyên) sang `data/pairs/{before,after}/` (không ghi đè file đã có; nếu trùng tên thì bỏ qua và cảnh báo).
- NCC < 0.50 → để nguyên trong review.
- Ghi `data/rescue_report.csv`: `filename, k1, ncc_old (từ report.csv), ncc_new, rescued`.
- Lưu ảnh kiểm chứng `outputs/rescue_samples/sample_<tên>.jpg` = ghép ngang `[before_undistorted | after]` (downscale 1500px) cho **10 cặp cứu được đầu tiên** để người review nhìn mắt.

## Ràng buộc (vi phạm = làm lại)
- KHÔNG đụng `data/pairs/` hiện có ngoài việc THÊM cặp mới đạt gate.
- KHÔNG sửa `ingest.py`, `align.py`, `match.py` (chỉ import dùng lại). Nếu buộc phải refactor nhỏ để import được, giữ hành vi cũ nguyên vẹn.
- KHÔNG xóa bất cứ file nào trong `data/review/` (kể cả cặp đã cứu — cứ copy sang pairs, giữ bản gốc).
- Chạy tiết kiệm CPU: `cv2.setNumThreads(2)` đầu script. Máy này còn dịch vụ production khác.
- Ảnh ghi ra: JPEG quality ≥ 95, không resize ngầm bản full-res.

## Acceptance (tự kiểm TRƯỚC khi báo xong)
- [ ] `rescue_review.py` chạy hết ~49 cặp không crash, in tổng kết: `X/49 rescued`.
- [ ] `data/rescue_report.csv` đầy đủ cột, ncc_new > ncc_old ở các cặp cứu được.
- [ ] 10 ảnh sample trong `outputs/rescue_samples/` — nhìn bằng mắt phải THẲNG HÀNG (mở vài ảnh kiểm tra thật, đừng chỉ tin số).
- [ ] Số cặp trong `data/pairs/before` = 12 + số rescued, before/after cùng tên, cùng kích thước từng cặp.

## Báo cáo (ghi vào cuối file này)
Số cặp cứu được · phân bố k1 · NCC trước/sau trung bình · đường dẫn samples · vướng mắc thật (đừng tô hồng).
