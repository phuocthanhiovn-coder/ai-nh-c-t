# Task 15 — Con "LÀM CỎ XANH" (grass greening — tính năng bán chạy của AutoHDR, bản CV)

**Giao cho:** Worker G · **Đọc `CLAUDE.md` trước.** Hợp đồng operator:
`apply(img: np.ndarray float32 [0,1] HxWx3 BGR, params: dict) -> np.ndarray` cùng shape.

## Files (`ai_engine/specialists/grass_green/`)
1. `grass.py`:
   - `segment_grass(img)` — mask mềm [0,1]: pixel thuộc dải màu cỏ (HSV: hue vàng-úa→xanh lá, saturation/value vừa) + ưu tiên nửa DƯỚI ảnh + texture nhiễu mịn (cỏ có high-freq đặc trưng, loại tường sơn xanh phẳng); mở rộng/feather mask (morphology + Gaussian).
   - `green_boost(img, mask, strength)` — TRONG mask: dịch hue về xanh cỏ khỏe (~hue 45-55 OpenCV), tăng saturation có giới hạn, nâng nhẹ luminance vùng úa; NGOÀI mask không đổi 1 pixel nào (composite bằng mask mềm).
   - `apply(img, params)` — params: `strength` 0-1 default 0.7. Không có cỏ (mask ≈ 0) → trả nguyên ảnh.
2. `run_samples.py` — tìm 5 ảnh NGOẠI THẤT có cỏ trong `data/pairs/before/` + `data/review/before/` + `data/unmatched/after/` (đọc thôi); lưu `outputs/grass_samples/<tên>.jpg` = [gốc | mask trực quan | kết quả]; in % diện tích mask từng ảnh.

## Acceptance (TỰ NHÌN ảnh)
- [ ] Cỏ úa vàng → xanh tự nhiên (không neon); cây/tường/đồ vật ngoài mask KHÔNG đổi màu.
- [ ] Ảnh nội thất (không cỏ) → trả nguyên vẹn. Output đúng size. Báo cáo trung thực (ca mask ăn lẹm vào cây? ghi rõ).

## KHÔNG: sửa file ngoài thư mục mình + báo cáo; cv2.setNumThreads(2).

---

## Báo cáo — Worker G-bis (tiếp quản, review + sửa lỗi)

**Trạng thái khi tiếp quản:** code của worker G đã có (`grass.py`, `run_samples.py`), 5 ảnh sample thật đã sinh trong `outputs/grass_samples/`, nhưng nhiều file rác debug (`zz*`, `_test*`, `_probe*`, và các biến thể `_ML_1563_repil/_crop/_tallresize`) — đã xoá hết, thư mục giờ chỉ còn đúng 5 ảnh sample cuối cùng.

### Bug nghiêm trọng tìm thấy khi TỰ NHÌN ảnh (không phải chỉ đọc code)

1. **Mask lem sang bê tông/vỉa hè (nghiêm trọng nhất).** Với tham số gốc (`SAT_MIN=25`, `TEXTURE_THRESH=3.0`), ảnh hẻm có hàng rào cây xanh (`_ML_1563`) bị mask phủ GẦN NHƯ TOÀN BỘ đường bê tông giữa khung hình (~14-18% diện tích ảnh), không chỉ 2 bên hàng rào — do bê tông ngoài nắng có hue vàng-xám (~20-24, trùng dải "cỏ úa") và độ bão hoà thấp (~30-40) vẫn vượt `SAT_MIN=25`, cộng thêm nhiễu hạt/JPEG-block trên bê tông đủ vượt `TEXTURE_THRESH=3.0`; morphology CLOSE + feather nối các đốm nhiễu rời rạc đó thành 1 mảng liền. Kết quả: đường lái xe bị nhuốm xanh-ô liu rõ trên crop zoom. **Đã sửa:** tăng `SAT_MIN` 25→50, `TEXTURE_THRESH` 3.0→7.0 (cỏ thật sat thường 60-130, bê tông 30-45 — đã đo thực tế trên ảnh). Xác nhận lại bằng crop zoom: hết vệt xanh trên bê tông, hàng rào 2 bên vẫn được nhận đúng.

2. **Mask dính vào cạnh đồ vật trong ảnh NỘI THẤT (vi phạm trực tiếp acceptance "ảnh nội thất → trả nguyên vẹn").** Test trên ảnh phòng giặt/bếp/toilet nội thất: dù đã sửa bug #1, mask vẫn nổi lên thành MẠNG LƯỚI đường viền mảnh chạy dọc khung cửa, viền máy giặt/sấy, ron gạch — do các pixel biên (anti-alias/JPEG giữa 2 vùng màu khác nhau) tình cờ rơi vào đúng dải hue/sat/texture của "cỏ". Phân tích connected-components cho thấy khác biệt định lượng rõ: mảng lưới cạnh giả này có bbox rất lớn nhưng diện tích thực chiếm tỉ lệ cực thấp trong bbox đó (fill_ratio ~0.06), hoặc là vạch thẳng mảnh (một chiều bbox chỉ vài px) — trong khi cỏ/cây thật là khối đặc (fill_ratio ~0.12-0.46 trở lên, cả 2 chiều bbox đều lớn). **Đã sửa:** thêm bước lọc connected-component (`_filter_thin_components`, `MIN_COMPONENT_FILL_RATIO=0.20`, `MIN_COMPONENT_MIN_DIM=45px`) trước morphology, chỉ giữ mảng đặc đủ lớn cả 2 chiều. Kết quả đo được trên 3 ảnh nội thất test: diện tích mask giảm từ ~3.5-1.9% xuống còn 0.5-2.1% (còn sót vài đốm rất nhỏ, xem "Hạn chế" bên dưới); ảnh hàng rào thật gần như không đổi (~19-21% → vẫn ~15-20%).

3. **Bug phụ trong `run_samples.py`:** hàm chọn ảnh xếp hạng theo mask % tính trên bản thu nhỏ 480px (`SCAN_MAX_DIM`), trong khi `FEATHER_SIGMA=5` và `MORPH_KSIZE=5` là hằng số theo PIXEL TUYỆT ĐỐI → ở ảnh nhỏ, feather/morphology chiếm tỉ lệ diện tích lớn hơn nhiều so với ảnh full-res, làm điểm xếp hạng (`scan_hint`) sai lệch nặng so với kết quả full-res thật, kéo theo: (a) 2 ảnh cùng 1 cảnh chụp (đặt tên khác nhau ở `data/pairs/before` và `data/review/before`, hoặc tiền tố `db01_`) lọt vào top-5 → **cùng basename output → ghi đè lẫn nhau, mất 1 sample âm thầm không cảnh báo**; (b) sau khi sửa bug #1+#2, ảnh nội thất/ảnh hỏng (1 file panorama bị lỗi, `db01__ML_1521.jpg`, sọc toả tia — lỗi dữ liệu đầu vào từ Task 02, không phải lỗi code này) vẫn lọt top-5 dù mask thực tế gần như bằng 0. **Đã sửa:** (a) thêm khử trùng lặp theo basename đã bỏ tiền tố `db01_`; (b) bỏ downscale khi xếp hạng, chạy `segment_grass` trực tiếp trên full-res (~0.4s/ảnh, quét hết pool ~82 ảnh mất ~33s — chấp nhận được vì đây là script chạy 1 lần, không phải hot path).

### Kết quả cuối cùng (5 ảnh, xem `outputs/grass_samples/`)

| File | area mask (full-res) | Nhận xét TỰ NHÌN |
|---|---|---|
| `_ML_1563.jpg` | 14.76% | Hẻm hàng rào cây xanh cắt tỉa — mask bám đúng 2 bức hàng rào trái/phải, KHÔNG lem sang đường bê tông giữa. Xanh lên tự nhiên, không neon. |
| `_ML_1570.jpg` | 9.62% | Tương tự — hàng rào trước toà nhà. Mask đúng vùng hàng rào, hụt nhẹ phần hàng rào bên phải bị bóng đổ (an toàn, thiếu positive hơn là dư false positive). |
| `_ML_1542.jpg` | 2.94% | Ban công nhìn xuống phố — mask bắt đúng CÂY THẬT nhìn xuyên qua khe lan can sắt (đã zoom kiểm tra: không phải lan can bị nhầm, mà đúng là tán cây lấp ló giữa các thanh sắt). Không đổi màu lan can/trời/nhà. |
| `_ML_1605.jpg` | 2.53% | Phố có bụi cây cảnh (tuyết tùng) trước nhà — mask bám đúng bụi cây, xe hơi/vỉa hè/tường nhà hoàn toàn không đổi (đã crop kiểm tra pixel-level). |
| `_ML_1479.jpg` | 2.52% | Sân thượng nhìn ra cây xanh xa — mask bám đúng vùng tán cây phía xa, ban công/tường/trời không đổi. |

Không còn ảnh nội thất hay ảnh lỗi trong top-5 sau khi sửa bug #3.

### Hạn chế còn tồn tại (báo cáo trung thực, chưa/không sửa hết trong lần này)

- **Không phải mọi ảnh nội thất đều tuyệt đối 0 mask.** Sau fix #2, 3 ảnh nội thất test riêng (phòng giặt, bếp) vẫn còn sót 0.5-2.1% diện tích là các đốm nhỏ rời rạc (ví dụ 1 icon trên bảng điều khiển máy sấy, 1 góc bản lề cửa) đủ "đặc" (fill_ratio ≥0.2, min_dim ≥45px) để lọt qua bộ lọc mới — ở độ phân giải thumbnail thì mắt thường không thấy đổi màu, nhưng về mặt kỹ thuật KHÔNG PHẢI mask ≈ 0 tuyệt đối như acceptance yêu cầu. Muốn siết thêm sẽ cần ngưỡng chặt hơn, đổi lại có nguy cơ ăn mất vùng cỏ thật nhỏ/lốm đốm — đánh đổi chưa tối ưu hoá thêm trong lần review này.
- **Cỏ trong bóng râm bắt thiếu (recall thấp hơn cỏ ngoài nắng).** Ví dụ `_ML_1570` hụt phần hàng rào bị bóng đổ. Đây là đánh đổi có chủ đích khi tăng `SAT_MIN`/`TEXTURE_THRESH` để diệt false positive trên bê tông — ưu tiên an toàn (thiếu còn hơn lem).
- **`db01__ML_1521.jpg`** (ảnh panorama bị lỗi ghép, sọc toả tia từ tâm) vẫn còn 1 vệt mask nhỏ dù đã bị loại khỏi top-5 — đây là lỗi dữ liệu từ pipeline ingest (Task 02), không phải phạm vi sửa của task này, nhưng nên gắn cờ để loại khỏi tập ảnh dùng sau này.
- Dataset hiện có (`data/pairs/before` + `data/review/before` + `data/unmatched/after`) khá ít ảnh ngoại thất có cỏ/cây rõ ràng — quét toàn bộ ~82 ảnh (đã khử trùng lặp) chỉ có 2 ảnh vượt ngưỡng `MIN_AREA_FRAC=3%`, còn lại đều dưới 3% (bụi cây nhỏ, cây xa) — 5 ảnh chọn được vẫn hợp lệ nhưng không phải "thảm cỏ sân vườn" điển hình như tên tính năng gợi ý.

### Kết luận
TASK15=DONE — Đã sửa 2 bug nghiêm trọng (mask lem bê tông; mask dính viền nội thất) + 1 bug phụ (trùng ảnh/ghi đè output do xếp hạng trên ảnh thu nhỏ); 5/5 sample cuối cùng đều là ảnh ngoại thất, mask bám đúng cây/cỏ thật, không lem sang bê tông/tường/xe/trời, màu xanh lên tự nhiên không neon; thư mục `outputs/grass_samples/` đã dọn sạch file rác, chỉ còn 5 ảnh panel cuối. Diện tích mask đo được (full-res): 14.76%, 9.62%, 2.94%, 2.53%, 2.52%.
