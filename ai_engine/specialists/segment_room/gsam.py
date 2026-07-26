"""MAT MOI (26/07) — open-vocabulary segmentation: Grounding DINO (detect theo
PROMPT chu) + SAM 2 (mask chinh xac). Thay SegFormer-B0 (yeu + license NVIDIA NC).

VI SAO (chu du an chi ro 25-26/07): SegFormer-B0 gan nhu MU voi cua so (mask 2 dot
= 0.5%) -> pipeline khong keo cua so xuong nhu AutoHDR -> model thoi trang. Grounding
DINO + SAM2 bat dung 'window glass / television / wooden floor...' theo prompt, recall
cao hon HAN + license Apache-2.0 (ban duoc). Nghiem thu fp104610: window 0.65 dung o.

License: Grounding DINO (IDEA-Research) Apache-2.0; SAM2 (facebook) Apache-2.0.
Chay CPU local duoc (cham ~vai giay/anh); GPU khi giao lo.
"""
import numpy as np
import cv2

_GDINO = None
_GDINO_PROC = None
_SAM = None
_SAM_PROC = None

GDINO_ID = "IDEA-Research/grounding-dino-tiny"
SAM_ID = "facebook/sam2.1-hiera-tiny"


def _load_gdino():
    global _GDINO, _GDINO_PROC
    if _GDINO is None:
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        _GDINO_PROC = AutoProcessor.from_pretrained(GDINO_ID)
        _GDINO = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_ID).eval()
    return _GDINO, _GDINO_PROC


def _load_sam():
    global _SAM, _SAM_PROC
    if _SAM is None:
        from transformers import Sam2Processor, Sam2Model
        _SAM_PROC = Sam2Processor.from_pretrained(SAM_ID)
        _SAM = Sam2Model.from_pretrained(SAM_ID).eval()
    return _SAM, _SAM_PROC


def detect_boxes(bgr, prompt, threshold=0.25):
    """prompt = 'window. television. wooden floor.' -> [(box[x0,y0,x1,y1], score, label)]."""
    import torch
    from PIL import Image
    model, proc = _load_gdino()
    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    inp = proc(images=img, text=prompt, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp)
    res = proc.post_process_grounded_object_detection(
        out, threshold=threshold, target_sizes=[img.size[::-1]])[0]
    labs = res.get("text_labels", res.get("labels"))
    return [(b.tolist(), float(s), l) for b, s, l in zip(res["boxes"], res["scores"], labs)]


def box_to_mask(bgr, box):
    """SAM2: box -> mask nhi phan float32 (HxW) chinh xac trong box."""
    import torch
    from PIL import Image
    model, proc = _load_sam()
    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    inp = proc(images=img, input_boxes=[[box]], return_tensors="pt")
    with torch.no_grad():
        out = model(**inp)
    masks = proc.post_process_masks(out.pred_masks, inp["original_sizes"])[0]
    m = masks[0].cpu().numpy()
    if m.ndim == 3:
        m = m[0]
    return (m > 0.5).astype(np.float32)


# Prompt mac dinh cho BDS: nhom theo cong dung xu ly (window pull, TV fill, material)
DEFAULT_PROMPTS = ("window. television. wooden floor. potted plant. "
                   "sky. painting. kitchen countertop. sofa.")


def segment_open(bgr, prompt=DEFAULT_PROMPTS, min_score=0.35, want=None):
    """Tra dict {label: mask_float32 HxW} gop moi box cung label bang SAM2.
    want = tap label muon giu (None = tat ca)."""
    h, w = bgr.shape[:2]
    dets = detect_boxes(bgr, prompt)
    masks = {}
    for box, score, label in dets:
        key = label.strip().lower()
        if score < min_score:
            continue
        if want is not None and not any(k in key for k in want):
            continue
        m = box_to_mask(bgr, box)
        masks[key] = np.maximum(masks.get(key, np.zeros((h, w), np.float32)), m)
    return masks


def window_mask(bgr, min_score=0.35):
    """Chi mask cua so (gop). Rong = zeros."""
    h, w = bgr.shape[:2]
    out = np.zeros((h, w), np.float32)
    for box, score, label in detect_boxes(bgr, "window."):
        if score >= min_score and "window" in label.lower():
            out = np.maximum(out, box_to_mask(bgr, box))
    return out
