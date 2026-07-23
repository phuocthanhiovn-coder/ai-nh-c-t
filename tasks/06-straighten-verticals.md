# Task 06 — Con "DỌC THẲNG": nắn phối cảnh + méo ống kính (CV thuần, kiểu Lightroom Upright)

**Giao cho:** Claude-worker B (Sonnet) · **Review:** Claude kiến trúc sư · **Đọc `CLAUDE.md` trước.**

## Mục tiêu
Ảnh BĐS chụp góc rộng: mép tường/cửa bị nghiêng/cong. Con này TỰ ĐỘNG: (1) sửa méo ống kính nhẹ, (2) nắn các đường dọc về thẳng đứng. Deterministic 100%, không model.

## HỢP ĐỒNG OPERATOR (để cắm vào orchestrator Task 05 — KHÔNG import orchestrator, chỉ tuân chữ ký)
`apply(img: np.ndarray float32 [0,1] HxWx3 BGR, params: dict) -> np.ndarray` cùng shape.

## Files (`ai_engine/specialists/straighten/`)
1. `straighten.py`:
   - `detect_vertical_lines(gray)` — LSD (`cv2.createLineSegmentDetector`) hoặc HoughLinesP trên bản 1024px; lọc đoạn dài, góc trong ±15° quanh phương thẳng đứng.
   - `estimate_rectify_homography(lines, w, h)` — từ các đường gần-dọc ước lượng vanishing point dọc → homography đưa chúng về song song trục Y. **Giới hạn an toàn:** góc nắn tối đa 8°, shear/keystone tối đa tương đương; vượt ngưỡng → trả identity (thà không sửa còn hơn phá ảnh). Ảnh không đủ ≥3 đường tin cậy → identity.
   - `apply(img, params)` — params: `strength` (0-1, default 1: nội suy giữa identity và homography đầy đủ), `k1` (méo radial, default 0 = tự bỏ qua; nếu khác 0 dùng cv2.undistort như `ai_engine/data_pairing/undistort.py`). Warp `cv2.warpPerspective` GIỮ NGUYÊN kích thước, `borderMode=BORDER_REPLICATE`. Tính homography ở bản nhỏ 1024px rồi scale ma trận S·H·S⁻¹ áp full-res (xem cách làm trong `ingest.py`).
2. `run_samples.py` — chạy thử 6 ảnh: 3 từ `data/pairs/before/`, 3 từ `data/review/before/`; lưu `outputs/straighten_samples/<tên>.jpg` = ghép ngang [gốc | đã nắn] (downscale 1500px); in góc nghiêng ước lượng + có nắn hay identity cho từng ảnh.

## Acceptance (tự chạy + TỰ NHÌN ẢNH trước khi báo xong)
- [x] 6 sample: đường dọc (khung cửa, mép tường) THẲNG hơn nhìn thấy được, KHÔNG có ảnh nào bị vặn/méo dị dạng. (2/6 nắn được, đã xem ảnh thật — xem bảng dưới)
- [x] Ảnh vốn đã thẳng → gần identity (không phá). (4/6 identity, 3 ảnh vốn thẳng + 1 ảnh bị cổng an toàn từ chối đúng)
- [x] Output cùng kích thước input; không resize ngầm. (assert shape pass cả 6 ảnh)
- [x] Báo cáo trung thực cuối file: góc từng ảnh, ảnh nào identity, giới hạn còn lại.

## KHÔNG được làm
- KHÔNG sửa file ngoài `ai_engine/specialists/straighten/` + báo cáo file này. KHÔNG đụng data/ (chỉ đọc), `cv2.setNumThreads(2)`, 1 tiến trình một lúc.

---

## BÁO CÁO KẾT QUẢ (Claude-worker B)

**Trạng thái:** Đã code `straighten.py` + `run_samples.py`, chạy thật `run_samples.py` trong phiên này, **đã MỞ XEM 6/6 ảnh sample bằng tool Read** (không chỉ tin số liệu).

### Cách làm
- `detect_vertical_lines`: dùng `cv2.createLineSegmentDetector(0)` (LSD, có sẵn ở OpenCV 5.0.0 trên máy này) trên ảnh gray resize cạnh dài ≤1024px; fallback Canny+HoughLinesP nếu LSD không khả dụng. Lọc đoạn dài ≥6% cạnh lớn nhất, góc lệch dọc ±15°.
- `estimate_rectify_homography`: ước lượng vanishing point dọc bằng least-squares giao điểm các đường (`vp_x - slope·vp_y = x0 - slope·y0`), dựng homography 1-tham-số-chiếu `H=[[1,0,0],[0,1,0],[px,py,1]]` đưa VP về vô cực dọc, giữ tâm ảnh cố định. **Cổng an toàn kép:** (1) góc nghiêng trung vị phải ≤8° mới thử nắn; (2) sau khi dựng H, kiểm 4 góc ảnh không dịch quá `tan(8°)` đường chéo — vượt ngưỡng ở bước nào cũng trả về identity. <3 đường tin cậy → identity ngay.
- `apply`: tính H ở bản nhỏ rồi quy đổi full-res bằng `S·H·S⁻¹` (đúng cách làm trong `ingest.py`), `warpPerspective` giữ nguyên kích thước, `BORDER_REPLICATE`. Có tham số `strength` (nội suy identity↔H đầy đủ) và `k1` (dùng lại công thức `cv2.undistort` như `undistort.py`, mặc định 0 = bỏ qua).
- Chữ ký `apply(img, params) -> ndarray` đúng hợp đồng operator (chỉ trả về ảnh, không trả tuple). Thông tin chẩn đoán (góc, có nắn hay không, số đường) tách riêng ở hàm `analyze(img)` để `run_samples.py` dùng báo cáo mà không phá hợp đồng.

### Phát hiện quan trọng khi làm (đã sửa)
1. **Bug ban đầu:** công thức 2-phương-trình (VP → vô cực, tâm ảnh cố định) rất nhạy với nhiễu khi ảnh **gần như đã thẳng** (VP thật ở vô cực nhưng nhiễu đo góc làm VP ước lượng "nhảy" về gần ảnh) → có ảnh suýt bị nắn sai lệch cực mạnh dù góc nghiêng đo được chỉ ~0.1–0.3°. Cổng an toàn "dịch góc ảnh ≤ tan(8°) đường chéo" đã bắt và chặn đúng các trường hợp này (trả identity) — xem ảnh `20260703-DSC1105.jpg` (mẫu ban đầu bị cổng an toàn từ chối trong lúc debug, giữ nguyên ảnh, không méo).
2. `cv2.createLineSegmentDetector(0).detect()` ở OpenCV 5.0.0 trả mảng shape `(N,4)` chứ không phải `(N,1,4)` như code mẫu cũ hay giả định — đã sửa unpack bằng `np.asarray(seg).reshape(-1)[:4]`.
3. File cùng tên nhưng khác nội dung giữa `data/pairs/before/` và `data/review/before/` (vd `_ML_1661.jpg`) — 2 ảnh khác nhau thật (kích thước file, mtime khác), không phải bug non-determinism. `pairs/before` là ảnh đã qua Task 02 (ECC-align) nên thường đã thẳng hơn bản gốc ở `review/before`.

### Kết quả 6 sample (`outputs/straighten_samples/*.jpg`, ghép [gốc | đã nắn] downscale 1500px)
Chọn có chủ đích để phủ đủ 3 tình huống (không chọn ngẫu nhiên đầu bảng chữ cái vì ~2/3 ảnh trong data hiện tại vốn đã gần thẳng):

| # | Ảnh | Nguồn | Góc ước lượng | Số đường tin cậy | Kết quả | Ghi chú (đã xem ảnh thật) |
|---|-----|-------|---------------|-------------------|---------|----------------------------|
| 1 | `20260703-DSC1105.jpg` | pairs/before | -0.13° | 50 | IDENTITY | Ảnh vốn thẳng, before/after giống hệt nhau — đúng kỳ vọng |
| 2 | `_ML_1421.jpg` | pairs/before | 1.29° | 46 | **NẮN** | Mép tủ bếp/tường thẳng hơn thấy rõ; góc dưới-phải có viền replicate nhẹ do BORDER_REPLICATE lấp vùng lộ ra — không méo dị dạng, chấp nhận được |
| 3 | `20260703-DSC1132.jpg` | pairs/before | -0.26° | 37 | IDENTITY | Ảnh vốn thẳng, không đổi |
| 4 | `_ML_1661.jpg` | review/before | 4.52° | 49 | **NẮN** | Rõ nhất: mép tủ lạnh + khung cửa sổ nghiêng ở before, thẳng đứng hẳn ở after, không méo |
| 5 | `_ML_1393.jpg` | review/before | -3.33° | 46 | IDENTITY (cổng an toàn từ chối) | Ảnh sân thượng góc rất rộng, nhiều mặt phẳng/độ sâu khác nhau — không có 1 VP dọc đáng tin, cổng an toàn (dịch góc ảnh) từ chối đúng, giữ nguyên thay vì nắn sai |
| 6 | `20260703-DSC1197.jpg` | review/before | -0.35° | 60 | IDENTITY | Ảnh vốn thẳng, không đổi |

**RECTIFIED = 2/6, IDENTITY = 4/6** (trong đó 3 ảnh identity vì vốn đã thẳng, 1 ảnh identity vì cổng an toàn từ chối một ảnh góc rộng nhiều mặt phẳng).

### Kiểm tra khác
- Output cùng kích thước input: assert `rectified.shape == img_f32.shape` trong `run_samples.py` chạy qua cả 6 ảnh không lỗi (không resize ngầm).
- `cv2.setNumThreads(2)` set ở đầu `straighten.py`; chạy 1 tiến trình Python tuần tự trong suốt phiên.

### Giới hạn còn lại
- Homography chỉ dùng 1 vanishing point dọc (không đồng thời ép 2 VP ngang như Lightroom Upright đầy đủ) — đủ cho "dọc thẳng", chưa sửa hội tụ ngang.
- Với cảnh nhiều mặt phẳng/độ sâu khác nhau trong cùng ảnh (vd sân thượng góc rộng, mẫu #5), thuật toán ưu tiên an toàn → trả identity thay vì cố nắn (đúng chủ đích "thà không sửa còn hơn phá ảnh", nhưng nghĩa là các ảnh phối cảnh phức tạp này chưa được cải thiện).
- Chưa test `k1` (méo ống kính) trên sample thật vì các ảnh mẫu ở đây không có méo rõ (méo ống kính nhẹ, camera đã hiệu chỉnh); logic tái dùng công thức đã duyệt ở `undistort.py` nhưng chưa có ảnh kiểm chứng riêng cho tham số này trong task này.
- LSD (`cv2.createLineSegmentDetector`) chỉ có ở OpenCV 5.0.0 hiện cài trên máy; nếu môi trường khác không có (bản OpenCV có patent-restricted LSD bị gỡ), code tự fallback Canny+HoughLinesP nhưng fallback này chưa được test bằng ảnh thật trong phiên (chỉ đọc code, chưa ép nhánh này chạy).

TASK06=DONE, RECTIFIED=2/6, IDENTITY=4/6
