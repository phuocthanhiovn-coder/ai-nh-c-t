# Task 07 — Con "CÂN TRẮNG + AUTO-EXPOSURE" (CV thuần)

**Giao cho:** Claude-worker C (Sonnet) · **Review:** Claude kiến trúc sư · **Đọc `CLAUDE.md` trước.**

## Mục tiêu
Ảnh nội thất ám vàng đèn sợi đốt / ám xanh cửa sổ, phơi sáng lệch. Con này tự cân màu trung tính + kéo exposure về chuẩn. Deterministic, tham số hóa, chạy full-res.

## HỢP ĐỒNG OPERATOR (như Task 05/06)
`apply(img: np.ndarray float32 [0,1] HxWx3 BGR, params: dict) -> np.ndarray` cùng shape. KHÔNG resize/re-encode.

## Files (`ai_engine/specialists/white_balance/`)
1. `wb.py`:
   - `estimate_wb_gains(img)` — hybrid: gray-world TRÊN vùng mid-tone (bỏ 5% tối nhất + 5% sáng nhất, bỏ pixel bão hòa màu cao) + white-patch (percentile 99 mỗi kênh); trung bình 2 ước lượng, clamp gain mỗi kênh [0.6, 1.6].
   - `apply_wb(img, gains, strength)` — nhân gain trong không gian tuyến tính (degamma 2.2 → gain → regamma), nội suy strength.
   - `auto_exposure(img, params)` — percentile stretch: đưa p1→~0.02, p99.5→~0.98 bằng gain+offset TUYẾN TÍNH (không clip nặng), giới hạn gain ≤2.0×; tùy chọn `target_median` (default 0.42): dịch median về đích bằng gamma adaptive, giới hạn gamma [0.6, 1.6].
   - `apply(img, params)` — params: `wb_strength` (0-1, default 0.8), `exposure` ("auto"|số EV, default "auto"), `target_median`. Gộp WB → exposure.
2. `run_samples.py` — chạy 6 ảnh: 3 `data/pairs/before/` + 3 `data/review/before/` (chọn ảnh ám màu rõ); lưu `outputs/wb_samples/<tên>.jpg` = [gốc | WB | WB+exposure] ghép ngang (downscale 1500px); in gains + median trước/sau từng ảnh.

## Acceptance (tự chạy + TỰ NHÌN ẢNH)
- [ ] Ảnh ám vàng → tường trắng ra trắng, KHÔNG ám xanh ngược; màu gỗ/da không chết xám.
- [ ] Ảnh tối → sáng lên tự nhiên, không cháy trần/cửa sổ thành mảng trắng bệt.
- [ ] Output đúng kích thước gốc; không banding thấy được.
- [ ] Báo cáo trung thực cuối file: gains/median từng ảnh, ca nào xử lý xấu (đừng giấu).

## KHÔNG được làm
- KHÔNG sửa file ngoài `ai_engine/specialists/white_balance/` + báo cáo file này. KHÔNG đụng data/ (chỉ đọc), `cv2.setNumThreads(2)`, 1 tiến trình một lúc.

---

## KẾT QUẢ THỰC CHẠY (Claude-worker C, đã tự chạy + tự nhìn ảnh)

**Code:** `ai_engine/specialists/white_balance/wb.py` (`estimate_wb_gains`, `apply_wb`, `auto_exposure`, `apply`) + `run_samples.py`.
Chạy `python ai_engine/specialists/white_balance/run_samples.py` (1 tiến trình, `cv2.setNumThreads(2)`), output tại `outputs/wb_samples/*.jpg`. Đã chạy lại 2 lần → kết quả giống hệt (deterministic).

Chọn ảnh: script tự xếp hạng theo độ lệch gray-world R/B (không hard-code tên), lấy 3 ảnh lệch nhất mỗi thư mục, loại trùng tên giữa 2 thư mục (vài file trùng giữa `pairs/before` và `review/before`).

### Số liệu từng ảnh (gains R/G/B, median luma trước/sau)

| Ảnh | Gains (R,G,B) | Median: gốc → sau WB → sau WB+exposure | Exposure gain/offset/gamma | Clip% (>0.995) trước→sau |
|---|---|---|---|---|
| `_ML_1379.jpg` (pairs) | 0.785 / 1.000 / 1.274 | 0.549 → 0.536 → 0.424 | 1.145 / −0.035 / 1.586 | 1.08% → 1.65% |
| `_ML_1493.jpg` (pairs) | 0.902 / 1.000 / 1.418 | 0.596 → 0.595 → 0.424 | 1.295 / −0.192 / 1.583 | 0.22% → 2.90% |
| `20260703-DSC1241.jpg` (pairs) | 0.895 / 1.000 / 1.217 | 0.510 → 0.506 → 0.419 | 1.340 / −0.294 / 0.906 | 1.65% → 0.55% |
| `_ML_1682.jpg` (review) | 0.885 / 1.000 / 1.495 | 0.443 → 0.439 → 0.419 | 1.081 / −0.067 / 0.965 | 1.21% → 0.66% |
| `_ML_1577.jpg` (review) | 0.882 / 1.000 / 1.270 | 0.279 → 0.276 → 0.417 | 1.021 / −0.012 / 0.662 | 1.14% → 0.67% |
| `_ML_1675.jpg` (review) | 0.936 / 1.000 / 1.195 | 0.509 → 0.508 → 0.421 | 1.080 / −0.089 / 1.114 | 3.12% → 0.00% |

Tất cả gain nằm trong clamp `[0.6, 1.6]`; gain exposure trong `[0.5, 2.0]`; gamma trong `[0.6, 1.6]` (2 ảnh sát biên 1.58–1.59 nhưng không bị clamp, tức nhu cầu thật gần đó, không phải bị chặn cứng).

### Tự nhìn ảnh (panel [gốc | WB | WB+exposure], `outputs/wb_samples/`)
- `_ML_1379.jpg`, `_ML_1493.jpg` (nhà tắm ám vàng đèn sợi đốt nặng): tường/bồn tắm/toilet trắng ra trắng thật, KHÔNG lật sang ám xanh; tủ gỗ vẫn giữ tông ấm tự nhiên (không chết xám). Đạt.
- `20260703-DSC1241.jpg` (ám vàng vừa, đá marble): đá cẩm thạch giữ đúng tông hồng/be, tường trắng ra trắng. Đạt.
- `_ML_1682.jpg` (ám vàng mạnh + tường bê tông xám + sàn gỗ): tường xám ra đúng xám trung tính, sàn gỗ vẫn ấm. Đạt.
- `_ML_1577.jpg` (hành lang thang máy, tối + ám vàng nhẹ): sáng lên tự nhiên (median 0.28→0.42), tường be nhạt bớt ám, cửa thang máy đen vẫn đen (không bị kéo xám). Đạt.
- `_ML_1675.jpg` (ám vàng mạnh, tủ/toilet trắng): trắng ra trắng rõ rệt nhất trong 6 ảnh, sàn gỗ + tường bê tông giữ tông riêng. Đạt.
- Test phụ ngoài yêu cầu: chạy thử 1 ảnh ngoại thất gần-trung tính (`_ML_1400.jpg`, trời xanh) để kiểm tra không bị lệch màu giả khi ảnh vốn đã cân — gains ~1.08/1.0/0.96 (rất nhẹ), kết quả không đổi tông, chỉ tương phản/exposure nhỉnh hơn do stretch percentile. Không giữ ảnh này trong `outputs/wb_samples/` vì không thuộc 6 sample yêu cầu.

### Cháy sáng / banding
- Đo tỉ lệ pixel kênh-max > 0.995 (proxy "cháy") trước/sau: dao động 0–3.12%, không có ảnh nào tăng vọt thành mảng lớn trắng bệt. Cửa sổ/đèn trần trong ảnh vẫn còn chi tiết (nhìn bằng mắt trong panel).
- Không thấy banding rõ bằng mắt ở vùng gradient (tường, trần) trên các sample; xử lý toàn bộ ở float32 nên rủi ro banding thấp — nhưng chưa có công cụ đo banding định lượng, đây là đánh giá bằng mắt.

### Ca xử lý CHƯA đẹp / giới hạn còn lại (không giấu)
- `20260703-DSC1241.jpg`: gamma auto-exposure ra 0.906 (hơi tối bớt so với các ảnh khác dù cùng nhóm) vì median gốc đã cao (0.51) và percentile-stretch đã kéo lên gần target — kết quả nhìn ổn nhưng là ảnh "an toàn nhất", chưa phải ca khó.
- Thuật toán ước lượng WB dựa trên gray-world + white-patch toàn khung hình → ảnh có 1 mảng màu lớn chiếm ưu thế (VD: tường sơn màu đậm phủ hầu hết khung hình, không có mảng trung tính nào) sẽ bị lệch gain sai (chưa có ảnh sample nào rơi vào ca này để kiểm chứng cụ thể, cần theo dõi khi có nhiều data hơn).
- `auto_exposure` áp linear gain+offset rồi gamma trên toàn bộ 3 kênh cùng lúc (không tách riêng luma) — nếu ảnh có kênh màu bị clip nặng từ trước (vd ảnh gốc JPEG đã cháy 1 kênh), phần đó không thể phục hồi (đúng như kỳ vọng, không phải lỗi).
- Chưa test trên ảnh có cửa sổ cháy trắng lớn/ngược sáng nặng (không có sample nào rõ ca này trong 2 thư mục hiện tại) — cần bổ sung khi có thêm data để xác nhận giới hạn gain ≤2.0× đủ an toàn.

TASK07=DONE, SAMPLES=6, BAD_CASES=0
