# NHIỆM VỤ: Tạo bảng so sánh TRƯỚC/SAU + trang HTML giao khách

Khi giao ảnh, cần 1 bản tổng hợp đẹp để gửi khách xem nhanh. Xây công cụ tạo:
(a) 1 ảnh contact-sheet ghép nhiều cặp trước/sau, và (b) 1 trang HTML self-contained.

## FILE MỚI (chỉ tạo mới)
`ai_engine/report/report.py` (+ `ai_engine/report/__init__.py` rỗng)

## API
```python
def make_contact_sheet(pairs: list, out_path: str, cols: int = 2, cell_w: int = 700) -> str:
    """pairs = list các (before_path, after_path). Ghép lưới: mỗi hàng 1 cặp
    [BEFORE | AFTER] có nhãn, nhiều cặp xếp dọc (cols cặp mỗi hàng nếu cols>1).
    Lưu JPEG q92 ra out_path. Trả out_path. Xử lý ảnh None (bỏ qua + cảnh báo)."""

def make_html_report(pairs: list, out_path: str, title: str = "Ảnh chỉnh AI") -> str:
    """Tạo 1 file HTML DUY NHẤT (self-contained) nhúng ảnh dạng base64 data URI,
    mỗi cặp là 1 slider/2 ảnh cạnh nhau trước/sau, có tiêu đề. Mở bằng trình duyệt là xem
    được ngay, không cần file ngoài. Trả out_path."""
```

## CLI + TỰ TEST (bắt buộc)
`python -m ai_engine.report.report --test`
- Lấy 2-3 ảnh từ `data/pairs/before/` làm 'before', và ảnh cùng tên trong `data/pairs/after/`
  làm 'after' (nếu không có after thì tự chạy nhanh: copy before làm after tạm cũng được, miễn test chạy).
- Gọi make_contact_sheet -> `outputs/report_test.jpg`, và make_html_report -> `outputs/report_test.html`.
- Kiểm tra 2 file tồn tại, ảnh mở lại được bằng cv2, html chứa "data:image". In kích thước.
- In "TASK DONE".

`python -m ai_engine.report.report --before <dir> --after <dir> --out-jpg <f.jpg> --out-html <f.html>`

## RÀNG BUỘC
- Chỉ cv2 + numpy + base64 (chuẩn thư viện). `cv2.setNumThreads(2)`. Chỉ tạo file mới.
- Docstring tiếng Việt ngắn. Chạy `--test` tạo được 2 file, in "TASK DONE". Trung thực nếu lỗi.
