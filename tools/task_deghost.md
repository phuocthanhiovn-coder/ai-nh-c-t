# NHIỆM VỤ: Khử bóng ma (deghost) khi trộn bracket

Khi trộn bracket, vật DI CHUYỂN giữa các tấm (người đi qua, rèm bay, cây lay) tạo "bóng ma"
(ghost) — vật xuất hiện mờ/nhân đôi. Xây bản trộn có khử ghost.

## FILE MỚI (chỉ tạo mới, KHÔNG sửa bracket_merge.py)
`ai_engine/specialists/auto_enhance/deghost.py`

Có thể import `merge_brackets` từ `ai_engine.specialists.auto_enhance.bracket_merge` để so sánh,
nhưng tự viết hàm khử ghost riêng.

## API
```python
def merge_deghost(paths: list, ref_index: int = None) -> np.ndarray:
    """Trộn bracket nhưng ở vùng CHUYỂN ĐỘNG chỉ lấy 1 tấm tham chiếu (tránh ghost).
    Bước:
    1. Đọc + đồng nhất size + AlignMTB (như bracket_merge).
    2. Chọn tấm tham chiếu ref (mặc định tấm có phơi sáng 'giữa' = độ sáng trung vị).
    3. Phát hiện vùng chuyển động: với mỗi tấm, |gray(tấm) - gray(ref)| sau khi bù độ sáng
       (khớp histogram/median) > ngưỡng -> mask motion.
    4. Fusion Mertens toàn ảnh, NHƯNG ở pixel thuộc motion mask -> thay bằng pixel từ ref
       (đã đưa về cùng phơi sáng của kết quả fusion, hoặc đơn giản: dùng ref gốc).
    5. Trả BGR uint8. Làm mịn mép mask (GaussianBlur) để không lộ đường ghép."""
```

## CLI + TỰ TEST (bắt buộc, để soi mắt)
`python -m ai_engine.specialists.auto_enhance.deghost --test [--sample <path.jpg>]`
- Lấy 1 ảnh (mặc định ảnh đầu `data/pairs/before/*.jpg`).
- Tạo 3 bản phơi sáng tổng hợp (x0.4 / x1.0 / x2.4). ĐỒNG THỜI vẽ 1 HÌNH VUÔNG sáng
  (giả "vật di chuyển") ở VỊ TRÍ KHÁC NHAU trên mỗi tấm (ví dụ x=100/300/500).
- Chạy: (a) merge thường (import merge_brackets), (b) merge_deghost.
- Lưu dải [MERGE THƯỜNG (có ghost) | DEGHOST (sạch)] ra `outputs/deghost_test.jpg` (nhãn, q92).
- In: % pixel khác biệt giữa 2 kết quả (cho thấy vùng ghost đã được xử lý).

## RÀNG BUỘC
- Chỉ cv2 + numpy. `cv2.setNumThreads(2)`. Chỉ tạo file mới. Xử lý ảnh None, <2 ảnh raise ValueError.
- Docstring tiếng Việt ngắn. Chạy `--test`, in "TASK DONE". Trung thực nếu deghost chưa sạch hẳn.
