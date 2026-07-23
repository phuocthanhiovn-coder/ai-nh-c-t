# NHIỆM VỤ: Xây engine trộn HDR bracket cho autohdr

Bạn là worker code Python cho dự án chỉnh ảnh BĐS (giống AutoHDR). Nguyên lý dự án:
AI xuất OPERATOR không xuất pixel; đã có model màu (CH_C). VẤN ĐỀ CÒN THIẾU: khách
chụp mỗi khung bằng NHIỀU tấm phơi sáng khác nhau (bracket: thiếu/đủ/dư sáng) để giữ
được cả trong phòng lẫn ngoài cửa sổ. Hiện pipeline chỉ nhận 1 tấm nên mất dải sáng.
=> Bạn xây module TRỘN bracket thành 1 ảnh đủ sáng đều (exposure fusion), để bước sau
áp operator màu + grade lên.

## FILE CẦN TẠO (chỉ tạo mới, KHÔNG sửa file khác)
`ai_engine/specialists/auto_enhance/bracket_merge.py`

## API BẮT BUỘC
```python
def merge_brackets(paths: list[str], align: bool = True) -> np.ndarray:
    """Nhận list đường dẫn ảnh cùng 1 khung khác phơi sáng -> trả 1 ảnh BGR uint8
    đã fusion (đủ sáng đều). Bước:
    1. Đọc bằng cv2 (BGR). Nếu khác kích thước -> resize tất cả về kích thước ảnh đầu.
    2. Nếu align=True: canh thẳng chống rung tay bằng cv2.createAlignMTB() (chuẩn cho
       bracket). Bọc try/except, lỗi thì bỏ qua align (dùng ảnh gốc).
    3. Fusion bằng cv2.createMergeMertens() -> float [0,1] -> nhân 255 clip uint8.
    4. Trả ảnh BGR uint8.
    Bọc lỗi rõ ràng; list rỗng hoặc <2 ảnh thì raise ValueError."""

def group_brackets(folder: str, group_size: int = 0) -> list[list[str]]:
    """Gom ảnh trong folder thành các bracket (mỗi bracket = list path).
    - Nếu group_size>0: cứ group_size ảnh liên tiếp (đã sort tên) = 1 bracket.
    - Nếu group_size==0: THỬ đọc EXIF thời gian phơi (ExposureTime) bằng Pillow;
      gom các ảnh liên tiếp có phơi sáng TĂNG DẦN rồi reset khi giảm (1 chu kỳ = 1 bracket).
      Nếu không đọc được EXIF -> fallback group_size=3 và IN cảnh báo.
    Trả list các bracket."""
```

## CLI + TỰ TEST (bắt buộc có, để tôi soi mắt)
`python -m ai_engine.specialists.auto_enhance.bracket_merge --test [--sample <path.jpg>]`
- Nếu không cho --sample: tự lấy 1 ảnh bất kỳ trong `data/pairs/before/*.jpg`.
- Tạo 3 bản phơi sáng TỔNG HỢP từ ảnh đó: tối (nhân ~0.35), vừa (x1.0), sáng (nhân ~2.6),
  đều clip [0,255]. Đây giả lập 1 bracket.
- Chạy merge_brackets lên 3 bản đó.
- Lưu 1 dải ảnh ghép ngang [TỐI | VỪA | SÁNG | ĐÃ MERGE] ra
  `outputs/bracket_test.jpg` (có nhãn chữ trên mỗi ô, JPEG q92).
- In ra: shape ảnh merge, và độ lệch chuẩn độ sáng (contrast) của merge so với ảnh VỪA
  (kỳ vọng merge giữ chi tiết cả vùng tối lẫn sáng -> ít pixel cháy trắng/chết đen hơn).

`python -m ai_engine.specialists.auto_enhance.bracket_merge --folder <dir> --group-size N --out <outdir>`
- Gom bracket -> merge từng bracket -> lưu `<outdir>/merged_000.jpg` q95 ...

## RÀNG BUỘC
- Chỉ dùng cv2 (opencv-python), numpy, và Pillow CHO EXIF (PIL.Image, PIL.ExifTags).
  KHÔNG dùng GPU (Mertens chạy CPU). `cv2.setNumThreads(2)` ở đầu.
- KHÔNG sửa/xóa file nào khác. KHÔNG đụng gì ngoài thư mục dự án autohdr.
- Code sạch, có docstring tiếng Việt ngắn gọn, xử lý lỗi rõ ràng (ảnh None, folder rỗng).
- Chạy được với Python 3, torch KHÔNG cần.

## KHI XONG
Chạy chính lệnh `--test` của bạn, đảm bảo nó tạo được `outputs/bracket_test.jpg` không lỗi,
rồi in "TASK DONE" ở cuối. Nếu gặp lỗi thư viện thiếu, tự `pip install` (opencv-python-headless,
pillow) rồi thử lại.
