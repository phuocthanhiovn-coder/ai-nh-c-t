# NHIỆM VỤ: Công cụ product 1 lệnh — nhận ảnh khách BẤT KỲ -> giao ảnh chỉnh

Bối cảnh: dự án chỉnh ảnh BĐS (giống AutoHDR). Đã có các mảnh, giờ cần 1 ENTRY POINT
để người dùng đưa ảnh khách vào là ra ảnh chỉnh, KHÔNG phụ thuộc data/pairs (đó chỉ là data demo).

ĐÃ CÓ (import, đừng chế lại):
- `ai_engine/specialists/auto_enhance/gpu/render_delivery.py`: `load_cfg(meta,device)`,
  `apply_fullres(model, bgr, device)`, `save_hq(path, img)` (JPEG q100 4:4:4).
- `ai_engine/specialists/auto_enhance/gpu/model_v2.py`: `HDRNetV2`.
- `ai_engine/specialists/auto_enhance/gpu/finish_grade.py`: `grade_auto(bgr, name=None)`.
- `ai_engine/specialists/auto_enhance/bracket_deliver.py`: `load_model(ckpt, device)`,
  `deliver_bracket(paths, model, device, grade=True)`.
- `ai_engine/specialists/auto_enhance/bracket_merge.py`: `group_brackets(folder, group_size)`.
- Model: `checkpoints/gpu/CH_C.pt`. Local CPU torch 2.13, cv2, PIL. device="cpu".

## FILE CẦN TẠO (chỉ tạo mới)
`ai_engine/process.py`

## API + LUỒNG
```python
def process_folder(in_dir, out_dir, brackets=1, grade=True, ckpt="checkpoints/gpu/CH_C.pt"):
    """Nhận folder ảnh khách -> lưu ảnh đã chỉnh vào out_dir (q100 4:4:4, giữ res gốc).
    - brackets == 1  (mặc định): MỖI ảnh là 1 ảnh đơn -> apply_fullres(model) -> grade_auto -> save_hq.
      Tên ra: <ten_goc>_edited.jpg
    - brackets >  1: coi input là các BỘ bracket. Gom bằng group_brackets(in_dir, group_size=brackets),
      mỗi bộ -> deliver_bracket(paths, model, device, grade) -> save_hq. Tên ra: <ten_anh_dau>_hdr.jpg
    Load model 1 LẦN dùng cho cả folder. In mỗi ảnh: tên, WxH, KB. Cuối in tổng số ảnh + tổng MB."""
```
Load model dùng `bracket_deliver.load_model(ckpt, torch.device("cpu"))`.
Ảnh đơn: `ai = apply_fullres(model, bgr, device); out = grade_auto(ai, name) if grade else ai`.

## CLI
`python -m ai_engine.process --in <folder> --out <folder> [--brackets N] [--no-grade]`
- Quét ảnh trong --in (jpg/jpeg/png/tif/webp, KHÔNG đệ quy trừ khi brackets>1 thì cho phép subfolder).
- Mặc định --brackets 1, có grade. --no-grade tắt grade.

## TỰ TEST (bắt buộc)
`python -m ai_engine.process --selftest`
- Tạo thư mục tạm `outputs/_proc_selftest_in/`, copy 2 ảnh từ `data/pairs/before/` vào.
- Chạy process_folder(brackets=1) ra `outputs/_proc_selftest_out/`.
- Kiểm tra: có đúng 2 file *_edited.jpg, mỗi file mở lại được bằng cv2 (không hỏng), KB > 200.
- In "SELFTEST PASS" nếu đạt, ngược lại in rõ cái gì fail. Cuối in "TASK DONE".

## RÀNG BUỘC
- device="cpu", `cv2.setNumThreads(2)`, `torch.set_num_threads(2)`. Chỉ tạo file mới.
- Bọc lỗi từng ảnh (1 ảnh hỏng KHÔNG làm sập cả folder — in cảnh báo, bỏ qua, chạy tiếp).
- KHÔNG đụng gì ngoài dự án. Docstring tiếng Việt ngắn.
- Chạy `--selftest`, đảm bảo PASS, in "TASK DONE". Nếu fail thật thì in lỗi thật, đừng giả vờ.
