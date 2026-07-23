# Task 01 — Con AI #1: Auto-Enhance (màu · sáng · tone), HDRnet-style

**Giao cho:** Gemini · **Review:** Claude · **Đọc `CLAUDE.md` trước.**
**Giai đoạn hiện tại = PILOT KIỂM MÁY MÓC** trên 8 cặp sạch mà Task 02 sinh ra (`data/pairs/`). Mục tiêu KHÔNG phải model dùng được (8 cặp quá ít, sẽ học vẹt) — mà **chứng minh code train chạy đúng nguyên lý**. Train thật làm sau khi có nhiều data.

## Mục tiêu
Model học mapping **before → after** (chỉ màu/sáng/tone) và **giữ nguyên full-res**. Đây là con lõi chứng minh nguyên lý **"AI xuất OPERATOR, không xuất pixel"**.

## ⭐ KIẾN TRÚC BẮT BUỘC (làm SAI cái này = làm lại). ĐỌC KỸ.
**KHÔNG được** làm UNet/autoencoder nhả thẳng ảnh. Phải đúng HDRnet (Gharbi 2017):

```
ẢNH GỐC (full-res, HxW)
   ├─────────────────────────────────────────────┐
   │ (a) thu nhỏ về PROXY 256x256                 │ (c) GUIDANCE MAP full-res
   ▼                                              │     (1 kênh, từ luminance
 MẠNG CONV (chạy trên proxy)                      │      hoặc pointwise-net nhỏ)
   ▼                                              │
 BILATERAL GRID hệ số affine                      │
   shape [B, 12, D, gh, gw]                       │
   (D=8 bins độ sáng, gh=gw=16 lưới không gian,   │
    12 = ma trận màu affine 3x4 mỗi bin)          │
   └──────────────► (d) SLICING ◄─────────────────┘
        Với MỖI pixel full-res (x,y): tra lưới bằng
        (x, y, guidance[x,y]) → nội suy 3 chiều (trilinear)
        → ra 1 ma trận 3x4 → nhân với [R,G,B,1] của CHÍNH pixel gốc
        → pixel output
   ▼
 ẢNH RA (full-res HxW) = áp affine lên ẢNH GỐC, KHÔNG phải mạng vẽ ra
```

**Mấu chốt:** mạng chỉ xuất **grid hệ số** (nhỏ). Pixel giao hàng = **áp affine từ grid lên ảnh gốc full-res**. Mạng KHÔNG bao giờ nhả pixel. Slicing phải **khả vi** (dùng `torch.nn.functional.grid_sample` để nội suy grid) để loss lan ngược về mạng.

## Files (`ai_engine/specialists/auto_enhance/`)
- `model.py` — `class HDRNet`: (1) `coefficient_predictor` (conv trên proxy 256 → tensor grid `[B,12,8,16,16]`); (2) `guidance` (luminance hoặc pointwise net → map full-res 1 kênh, giá trị [0,1]); (3) `slice_and_apply(grid, guidance, full_res_img)` dùng `grid_sample` → ảnh ra. Forward nhận `(proxy, full_res_img)` → ảnh full-res.
- `dataset.py` — load cặp cùng tên trong `data/pairs/{before,after}/`; trả `(full_before, proxy_before_256, after)`. Augment: flip ngang. Ảnh về float [0,1].
- `train.py` — forward → **loss = L1(output, after)** (+ tùy chọn perceptual sau) → Adam → in loss mỗi epoch → lưu `checkpoints/auto_enhance.pt`. Cờ `--smoke`.
- `infer.py` — nạp checkpoint, 1 ảnh vào → ảnh full-res ra.
- `config.yaml`, `README.md`.

## Data
Dùng 8 cặp trong `data/pairs/{before,after}/` (Task 02, ~2048px, đã lọc `color` + align đạt). CHỈ nhóm `data/pairs/` (không đụng pairs_sky/pairs_removal).

## SMOKE TEST (CPU, làm TRƯỚC khi báo xong)
1. **Overfit 8 cặp** nhiều epoch. Loss **phải giảm rõ** (in đầu→cuối). Output tiến gần ảnh after.
2. Lưu `outputs/smoke_compare.png` = `[before | output | after]` cho 2–3 cặp.
3. **CHỨNG MINH operator-không-pixel (bắt buộc, đây là điểm review chính):**
   - a. `infer` một ảnh → in `input_size` và `output_size` → **phải BẰNG nhau** với ảnh kích thước BẤT KỲ (thử cả 2048px lẫn một ảnh to hơn).
   - b. In `grid.shape` để thấy mạng xuất **grid hệ số** (vd `[1,12,8,16,16]`), KHÔNG phải ảnh HxW.
   - c. Chạy inference CÙNG một ảnh ở **2 độ phân giải khác nhau** → cả hai ra ảnh đúng size tương ứng, nhìn cùng "gu" → chứng minh mạng học transform, không học pixel.

## Acceptance
- [ ] Loss overfit giảm rõ; `smoke_compare.png` cho thấy output bám after.
- [ ] Mạng xuất GRID hệ số (in shape chứng minh), pixel ra từ áp affine lên ảnh gốc.
- [ ] `output_size == input_size` với mọi kích thước; chạy được ở 2 độ phân giải.
- [ ] Không recompress ngầm; lưu PNG/JPEG q≥95.

## KHÔNG được làm
- KHÔNG UNet/diffusion nhả thẳng ảnh.
- KHÔNG cho mạng chạy trực tiếp trên full-res (mạng chỉ ăn proxy 256; full-res chỉ để ÁP grid).
- KHÔNG train "thật" — đây là pilot overfit 8 cặp kiểm máy móc.

## Báo cáo (Gemini ghi cuối file)
- Loss đầu→cuối · grid.shape · input_size vs output_size (nhiều kích thước) · đường dẫn smoke_compare · vướng mắc.
