# NHIỆM VỤ: Tự chỉnh ảnh BĐS đứng thẳng (verticals straightening)

Ảnh BĐS chuyên nghiệp BẮT BUỘC các đường thẳng đứng (khung cửa, tường, cạnh tủ) phải
THẲNG ĐỨNG, không nghiêng/đổ. Xây module tự phát hiện độ nghiêng và xoay lại cho thẳng.

## FILE MỚI (chỉ tạo mới, không sửa file khác)
`ai_engine/specialists/straighten/straighten.py`  (tạo cả `ai_engine/specialists/straighten/__init__.py` rỗng)

## API
```python
def estimate_tilt(bgr) -> float:
    """Ước lượng góc nghiêng (độ) từ các đường GẦN THẲNG ĐỨNG.
    Canny -> HoughLinesP -> lọc các đoạn có góc gần 90° (trong ±20°) ->
    lấy TRUNG VỊ độ lệch so với phương thẳng đứng. Trả góc (độ), + = nghiêng phải."""

def straighten(bgr, max_deg=8.0) -> np.ndarray:
    """Xoay ảnh để verticals thẳng. Nếu |tilt|>max_deg thì clamp về max_deg (tránh xoay quá).
    Xoay quanh tâm, giữ nguyên kích thước (cắt viền đen bằng cách phóng nhẹ ~ scale để lấp góc).
    Nếu không tìm đủ đường vertical (ví dụ ảnh trời/drone) -> trả ảnh gốc, không xoay."""
```

## CLI + TỰ TEST (bắt buộc)
`python -m ai_engine.specialists.straighten.straighten --test [--sample <path.jpg>]`
- Lấy 1 ảnh nội thất (mặc định ảnh đầu `data/pairs/before/*.jpg` — chọn ảnh có cửa/tường).
- Xoay GIẢ 4.0° để tạo ảnh nghiêng.
- Chạy estimate_tilt (kỳ vọng ~ -4°) rồi straighten.
- In: góc ước lượng, góc còn dư sau khi sửa (kỳ vọng gần 0).
- Lưu dải ghép [NGHIÊNG 4° | ĐÃ SỬA | GỐC] ra `outputs/straighten_test.jpg` (nhãn chữ, q92).

`python -m ai_engine.specialists.straighten.straighten --in <folder> --out <outdir>` (xử lý cả folder).

## RÀNG BUỘC
- Chỉ cv2 + numpy. `cv2.setNumThreads(2)`. Chỉ tạo file mới. Không đụng ngoài dự án.
- Xử lý ảnh None/hỏng. Docstring tiếng Việt ngắn. Chạy `--test` tạo được ảnh, in "TASK DONE".
- Trung thực: nếu estimate_tilt sai nhiều thì IN ra thật, đừng giả vờ pass.
