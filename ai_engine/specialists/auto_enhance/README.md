# Specialist #1: Auto-Enhance (HDRnet-style)

Mô hình học máy xử lý màu sắc, ánh sáng và tông màu (color, brightness, contrast) dựa trên kiến trúc HDRnet (Gharbi et 2017).

## Nguyên lý hoạt động
Mô hình đi theo triết lý **"AI xuất OPERATOR, không xuất pixel"**:
1. Ảnh đầu vào được thu nhỏ về kích thước Proxy $256 \times 256$.
2. Mạng Convolutional (Coefficient Predictor) xử lý Proxy để dự đoán một Bilateral Grid chứa các hệ số affine $3 \times 4$ với kích thước $[B, 12, 8, 16, 16]$.
3. Ảnh gốc full-res được biến đổi thành một Guidance Map 1 kênh (luminance-like).
4. Phép toán Bilateral Slicing khả vi (sử dụng `torch.nn.functional.grid_sample`) thực hiện tra cứu lưới hệ số 3D theo tọa độ $(x, y, Guidance(x, y))$ của từng pixel.
5. Áp dụng ma trận biến đổi Affine $3 \times 4$ thu được trực tiếp lên màu RGB của ảnh gốc full-res để tạo ra ảnh kết quả có cùng kích thước 100%.

## Cấu trúc thư mục
- `model.py` — Kiến trúc HDRNet.
- `dataset.py` — Dataset loader overfit 8 cặp ảnh sạch.
- `train.py` — Huấn luyện overfit pilot.
- `infer.py` — Inference kiểm chứng nguyên lý.
- `config.yaml` — File cấu hình.

## Cách chạy
### Huấn luyện overfit 8 cặp:
```bash
python -m ai_engine.specialists.auto_enhance.train --epochs 150 --lr 0.001
```

### Chạy smoke test tự động kiểm chứng:
```bash
python -m ai_engine.specialists.auto_enhance.infer --verify
```
