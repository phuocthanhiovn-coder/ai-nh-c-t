"""segment_room — CON MẮT ĐẦU TIÊN của roster nhóm A (24/07/2026).

VÌ SAO: chủ dự án chốt đúng bệnh sau 7 vòng chấm: "AI không phân biệt được đồ
vật nên đa số ảnh mờ và nhạt". Model màu (HDRNetV2) pha theo vùng mượt, không
biết tường/sàn/đồ vật — nên phòng trống đẹp, phòng đầy đồ đuối. Con này cho
pipeline ĐÔI MẮT: tách ảnh thành 5 nhóm ngữ nghĩa để các op sau chỉnh đúng
kiểu từng vùng.

MODEL: SegFormer-B0 fine-tune ADE20K (nvidia/segformer-b0-finetuned-ade-512-512)
— model MỞ tự host (đúng nguyên tắc 4), 3.7M tham số, chạy CPU ~1–2 s/ảnh proxy
512. Lazy-load 1 lần / process.

API (không phải operator ảnh — trả MASK, giống window_mask):
    segment(img_bgr_f32_01) -> dict các mask float32 [0,1] HxW full-res:
        "wall", "floor", "ceiling", "window", "object"
    (mask mềm — lấy softmax prob, KHÔNG argmax cứng — để op sau feather tự nhiên)

Nhóm lớp ADE20K (0-based): wall=0 (+ 32 fence? không — giữ tối thiểu chắc chắn),
floor=3 (+ rug=28), ceiling=5, window=8, door=14 gộp vào wall-group (mặt phẳng
kiến trúc); còn lại = object.
"""
import numpy as np
import cv2

cv2.setNumThreads(3)

_PROXY = 512

# ADE20K 150 lớp — nhóm dùng cho BĐS nội thất
_WALL_IDS = [0, 14]          # wall, door (mặt phẳng kiến trúc đứng)
_FLOOR_IDS = [3, 28]         # floor, rug
_CEIL_IDS = [5]              # ceiling
_WINDOW_IDS = [8]            # windowpane

_model = None
_processor = None


def _load():
    global _model, _processor
    if _model is None:
        import torch
        from transformers import (SegformerForSemanticSegmentation,
                                  SegformerImageProcessor)
        torch.set_num_threads(3)
        name = "nvidia/segformer-b0-finetuned-ade-512-512"
        _model = SegformerForSemanticSegmentation.from_pretrained(name).eval()
        _processor = SegformerImageProcessor.from_pretrained(name)
    return _model, _processor


_MATERIAL_NAME_GROUPS = {
    # nhom chat lieu -> tu khoa ten lop ADE20K (khop theo TEN tu id2label, khong hardcode index)
    "dark_appliance": ["stove", "oven", "refrigerator", "television", "computer",
                       "microwave", "monitor", "screen", "fireplace"],
    "wood": ["table", "desk", "countertop", "chest of drawers", "wardrobe",
             "shelf", "chair", "bench", "stairs", "bannister", "coffee table"],
    "fabric": ["sofa", "cushion", "pillow", "curtain", "rug", "blanket",
               "armchair", "bed ", "towel", "apparel"],
    "plant": ["plant", "flower", "tree", "palm", "grass"],
    "fixture_white": ["sink", "toilet", "bathtub", "counter"],
    "window_glass": ["windowpane"],
    "art": ["painting", "poster", "mirror"],
}


def segment_fine(img, proxy=_PROXY):
    """Mask MEM theo NHOM CHAT LIEU (150 lop ADE20K gop theo ten). Tra dict
    {nhom: mask HxW [0,1]} — dung cho material_grade (chinh theo tung mon do)."""
    import torch

    model, processor = _load()
    h, w = img.shape[:2]
    rgb_u8 = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8),
                          cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb_u8, size={"height": proxy, "width": proxy},
                       return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1)[0]

    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    out = {}
    for group, keywords in _MATERIAL_NAME_GROUPS.items():
        ids = [i for i, name in id2label.items()
               if any(kw in name for kw in keywords)]
        if not ids:
            continue
        m = probs[ids].sum(dim=0).numpy()
        out[group] = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return out


def segment(img, proxy=_PROXY):
    """img: float32 [0,1] HxWx3 BGR. Trả dict mask float32 [0,1] HxW full-res."""
    import torch

    model, processor = _load()
    h, w = img.shape[:2]
    rgb_u8 = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8),
                          cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb_u8, size={"height": proxy, "width": proxy},
                       return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits          # [1,150,h/4,w/4]
        probs = torch.softmax(logits, dim=1)[0]  # [150,h',w']

    def group(ids):
        m = probs[ids].sum(dim=0).numpy()
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)

    wall = group(_WALL_IDS)
    floor = group(_FLOOR_IDS)
    ceiling = group(_CEIL_IDS)
    window = group(_WINDOW_IDS)
    other = 1.0 - np.clip(wall + floor + ceiling + window, 0.0, 1.0)
    return {
        "wall": wall.astype(np.float32),
        "floor": floor.astype(np.float32),
        "ceiling": ceiling.astype(np.float32),
        "window": window.astype(np.float32),
        "object": other.astype(np.float32),
    }
