# NHIỆM VỤ: Tự cân bằng trắng / khử ám màu (fix "nhiễm màu")

Khách chê ảnh bị "nhiễm màu" (ám màu). Xây module tự trung hoà ám màu (auto white balance).

## FILE MỚI (chỉ tạo mới)
`ai_engine/specialists/white_balance/white_balance.py` (tạo cả `__init__.py` rỗng)

## API
```python
def auto_wb(bgr, method="grayworld", strength=1.0) -> np.ndarray:
    """Khử ám màu.
    - method="grayworld": giả định trung bình cảnh là xám -> scale kênh B,G,R để mean bằng nhau.
    - method="whitepatch": lấy vùng sáng nhất (top ~1% độ sáng) làm trắng chuẩn -> scale.
    - method="combined": trung bình 2 cách trên.
    strength (0..1): pha trộn giữa ảnh gốc (0) và ảnh đã sửa hoàn toàn (1), để không quá tay.
    KHÔNG được đổi độ sáng tổng thể nhiều (chuẩn hoá lại luma sau khi scale màu)."""

def measure_cast(bgr) -> dict:
    """Trả {'a': mean_a, 'b': mean_b} trong Lab (đã trừ 128). Gần 0 = trung tính, không ám."""
```

## CLI + TỰ TEST (bắt buộc, để soi mắt)
`python -m ai_engine.specialists.white_balance.white_balance --test [--sample <path.jpg>]`
- Lấy 1 ảnh (mặc định ảnh đầu `data/pairs/before/*.jpg`).
- Tạo ảnh ÁM XANH giả: nhân kênh B (blue, index 0 của BGR) x1.18, clip.
- Chạy auto_wb(method="combined") lên ảnh ám.
- In measure_cast của: gốc, ảnh ám, ảnh đã sửa. (Kỳ vọng: ảnh sửa có |a|,|b| nhỏ hơn ảnh ám, gần gốc.)
- Lưu dải [GỐC | ÁM XANH | ĐÃ SỬA] ra `outputs/wb_test.jpg` (nhãn chữ, q92).

`python -m ai_engine.specialists.white_balance.white_balance --in <folder> --out <outdir> [--method combined] [--strength 0.8]`

## RÀNG BUỘC
- Chỉ cv2 + numpy. `cv2.setNumThreads(2)`. Chỉ tạo file mới. Xử lý ảnh None.
- Docstring tiếng Việt ngắn. Chạy `--test` tạo được ảnh, in "TASK DONE".
- Trung thực về số đo, đừng thổi phồng.
