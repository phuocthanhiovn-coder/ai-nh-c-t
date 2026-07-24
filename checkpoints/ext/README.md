# Weights ngoài (external, KHÔNG commit)

Các weights của bên thứ ba tải riêng, không nằm trong git (xem `.gitignore`).

## realesr-general-x4v3.pth (Real-ESRGAN, BSD-3-Clause — dùng thương mại được)
Con `detail_restore` cần file này. Tải:

```
https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth
```

Đặt vào đúng thư mục này (`checkpoints/ext/`). ~4.7 MB, SRVGGNetCompact
(num_feat=64, num_conv=32, upscale=4). Không có file → `detail_restore` tự bỏ qua
(trả ảnh gốc, không crash).
