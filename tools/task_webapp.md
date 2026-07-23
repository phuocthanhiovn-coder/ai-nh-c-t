# NHIỆM VỤ: Web UI local kéo-thả — xem trước/sau, để demo cho khách

Dựng app web CHẠY LOCAL: kéo-thả ảnh -> chạy pipeline chỉnh -> hiện slider so sánh trước/sau.
Dùng để chủ dự án demo chất lượng cho khách hàng ngay trên máy.

## FILE MỚI (chỉ tạo mới)
`ai_engine/webapp/app.py`  (+ `ai_engine/webapp/__init__.py` rỗng)

## YÊU CẦU
- Dùng Flask (nếu chưa có: `pip install flask`). CHẠY LOCAL, host 127.0.0.1, port 8760.
- Route `GET /`: trả 1 trang HTML (inline, không cần file tĩnh ngoài) có:
  - Ô kéo-thả / chọn ảnh (input file).
  - Khi chọn ảnh -> POST lên `/process` -> nhận ảnh đã chỉnh -> hiện 2 ảnh trước/sau cạnh nhau
    (hoặc slider kéo). CSS inline gọn gàng, nền tối, tiếng Việt.
- Route `POST /process`: nhận file ảnh upload, chạy pipeline chỉnh ảnh ĐƠN:
  ```python
  from ai_engine.specialists.auto_enhance.gpu.render_delivery import apply_fullres
  from ai_engine.specialists.auto_enhance.gpu.finish_grade import grade_auto
  from ai_engine.specialists.auto_enhance.bracket_deliver import load_model
  # load_model 1 lần lúc khởi động app (global), device CPU
  ```
  Đọc ảnh upload bằng cv2 (từ bytes), apply_fullres -> grade_auto -> encode JPEG q95 ->
  trả về (base64 data URL hoặc raw bytes) để trang hiển thị. Cap chiều rộng xử lý ~1600px
  cho nhanh (nhưng vẫn nét khi xem web).
- Model nạp 1 LẦN lúc start (không nạp lại mỗi request).

## TỰ TEST (bắt buộc — KHÔNG chạy server treo)
`python -m ai_engine.webapp.app --selftest`
- Dùng Flask test client (`app.test_client()`), KHÔNG mở server thật:
  - GET `/` -> status 200, HTML chứa chữ "kéo" hoặc "chọn ảnh".
  - POST `/process` với 1 ảnh thật từ `data/pairs/before/*.jpg` (multipart) -> status 200,
    trả về dữ liệu ảnh (len > 1000 bytes hoặc data URL hợp lệ).
- In "SELFTEST PASS" nếu cả 2 đạt, kèm kích thước ảnh trả về. Cuối in "TASK DONE".
- Lệnh chạy server thật (cho người dùng): `python -m ai_engine.webapp.app` (host 127.0.0.1:8760),
  nhưng --selftest thì TUYỆT ĐỐI không được block.

## RÀNG BUỘC
- device="cpu", cv2.setNumThreads(2), torch.set_num_threads(2). Chỉ tạo file mới.
- Không đụng ngoài dự án. Xử lý lỗi upload (không phải ảnh -> trả 400 + thông báo).
- Docstring/UI tiếng Việt. Chạy `--selftest` PASS, in "TASK DONE". Trung thực nếu fail.
