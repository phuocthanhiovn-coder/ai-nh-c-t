# NHIỆM VỤ: Nối bracket_merge -> model màu CH_C -> grade thành pipeline giao ảnh HDR

Bối cảnh: dự án chỉnh ảnh BĐS (giống AutoHDR). ĐÃ CÓ:
- `ai_engine/specialists/auto_enhance/bracket_merge.py`: `merge_brackets(paths)` (Mertens fusion)
  và `group_brackets(folder, group_size)`.
- `ai_engine/specialists/auto_enhance/gpu/render_delivery.py`: có sẵn hàm module-level
  `load_cfg(meta_path, device)`, `apply_fullres(model, before_bgr, device)`, `save_hq(path, img)`.
- `ai_engine/specialists/auto_enhance/gpu/finish_grade.py`: `grade_auto(bgr, name=None)` (tăng bão hòa/ấm/tương phản scene-aware).
- `ai_engine/specialists/auto_enhance/gpu/model_v2.py`: class `HDRNetV2`.
- Model đã train: `checkpoints/gpu/CH_C.pt` (+ `CH_C.pt.meta`).

Local có torch 2.13 CPU, cv2, PIL. KHÔNG có GPU — chạy CPU hết (device="cpu").

## FILE CẦN TẠO (chỉ tạo mới, KHÔNG sửa file khác)
`ai_engine/specialists/auto_enhance/bracket_deliver.py`

## LUỒNG: folder bracket -> gom -> merge -> operator màu CH_C -> grade -> ảnh full-res q100

Dùng ĐÚNG đoạn load model này (đừng tự chế khác):
```python
import torch
from ai_engine.specialists.auto_enhance.gpu.model_v2 import HDRNetV2
from ai_engine.specialists.auto_enhance.gpu.render_delivery import load_cfg, apply_fullres, save_hq
from ai_engine.specialists.auto_enhance.gpu.finish_grade import grade_auto
from ai_engine.specialists.auto_enhance.bracket_merge import merge_brackets, group_brackets

def load_model(ckpt="checkpoints/gpu/CH_C.pt", device=None):
    device = device or torch.device("cpu")
    cfg = load_cfg(ckpt + ".meta", device)
    model = HDRNetV2(**cfg).to(device)
    st = torch.load(ckpt, map_location=device)
    if isinstance(st, dict) and "state_dict" in st:
        st = st["state_dict"]
    model.load_state_dict(st); model.eval()
    return model, device
```

## API BẮT BUỘC
```python
def deliver_bracket(paths: list, model, device, grade: bool = True) -> np.ndarray:
    """1 bracket (list path) -> merge -> operator CH_C -> (grade) -> ảnh BGR uint8 full-res.
    merged = merge_brackets(paths); ai = apply_fullres(model, merged, device);
    return grade_auto(ai, paths[0]) if grade else ai"""

def deliver_folder(folder, out_dir, group_size=0, grade=True, ckpt="checkpoints/gpu/CH_C.pt"):
    """Gom bracket trong folder -> deliver_bracket từng cái -> lưu q100 4:4:4 bằng save_hq.
    Đặt tên file ra: hdr_000.jpg, hdr_001.jpg ... In kích thước + KB mỗi ảnh."""
```

## CLI + TỰ TEST (bắt buộc, để tôi soi mắt)
`python -m ai_engine.specialists.auto_enhance.bracket_deliver --test [--sample <path.jpg>]`
- Lấy 1 ảnh (mặc định ảnh đầu trong `data/pairs/before/*.jpg`).
- Tạo bracket TỔNG HỢP tối/vừa/sáng (nhân ~0.35 / 1.0 / 2.6, clip) như bracket_merge làm.
- Chạy: merge -> operator CH_C -> grade.
- Lưu 1 dải ghép ngang [VỪA(gốc) | ĐÃ MERGE | MERGE+AI+GRADE] ra `outputs/bracket_deliver_test.jpg`
  (nhãn chữ mỗi ô, JPEG q92). In shape + KB ảnh cuối.

`python -m ai_engine.specialists.auto_enhance.bracket_deliver --folder <dir> --group-size N --out <outdir>`

## RÀNG BUỘC
- device="cpu" mọi nơi. `cv2.setNumThreads(2)`, `torch.set_num_threads(2)` đầu file.
- Chỉ tạo file mới, KHÔNG sửa file khác, KHÔNG đụng ngoài dự án autohdr.
- Bọc lỗi rõ ràng. Nếu import numpy báo lỗi tương thích (numpy 2.x vs cv2), thử `pip install "numpy<2.3"` rồi chạy lại; nếu vẫn lỗi thì IN rõ lỗi ra, ĐỪNG giả vờ pass.
- Docstring tiếng Việt ngắn. Chạy chính lệnh `--test`, đảm bảo tạo được ảnh, in "TASK DONE" cuối.

## QUAN TRỌNG (trung thực)
Bracket test là TỔNG HỢP từ 1 ảnh nên không chứng minh được lợi ích HDR thật — chỉ chứng minh
ĐƯỜNG ỐNG chạy thông (merge->AI->grade). Ghi rõ điều này khi báo cáo, đừng thổi phồng.
