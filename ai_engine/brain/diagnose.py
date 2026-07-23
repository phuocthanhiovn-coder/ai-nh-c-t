"""brain.diagnose — GOM GIÁC QUAN (24/07/2026, thiết kế chốt với chủ dự án).

Nhìn 1 ảnh bằng mọi giác quan hiện có -> BỆNH ÁN (dict số liệu + mô tả), làm
đầu vào cho brain.prescribe kê toa. Chỉ ĐO, không chỉnh — mọi giác quan chạy
trên proxy để nhanh.

Giác quan: scene_classify (loại cảnh) · segment_room (tường/sàn/trần/cửa sổ/
đồ vật — SegFormer) · đo sáng (p5/p50/p95, vùng tối, bùn màu vùng tối) ·
đo cast màu (gain WB ước lượng) · đo dải sáng (cháy/kẹt bóng) · đo nét (lap var).
"""
import cv2
import numpy as np

cv2.setNumThreads(3)

from ai_engine.specialists.scene_classify.classify import classify as _classify
from ai_engine.specialists.white_balance import wb as _wb


def _proxy(img, max_dim=1024):
    h, w = img.shape[:2]
    s = max_dim / max(h, w)
    if s >= 1.0:
        return img
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def diagnose(img, with_masks=True):
    """img: float32 [0,1] HxWx3 BGR full-res. Trả bệnh án dict."""
    p = _proxy(img)
    g = cv2.cvtColor((p * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor((p * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)

    d = {}
    # loại cảnh
    try:
        c = _classify(p)
        d["scene"] = c.get("scene", "general")
        d["scene_conf"] = round(float(c.get("confidence", 0.0)), 3)
    except Exception:
        d["scene"], d["scene_conf"] = "general", 0.0

    # ánh sáng
    d["p5"], d["p50"], d["p95"] = [round(float(np.percentile(g, q)), 1) for q in (5, 50, 95)]
    dark = g < 80
    d["dark_frac"] = round(float(dark.mean()), 3)
    d["dark_sat"] = round(float(hsv[..., 1][dark].mean()), 1) if dark.sum() else 0.0
    bright = g > 245
    d["blown_frac"] = round(float(bright.mean()), 3)

    # cast màu (gain WB cần thiết — lệch xa 1.0 = ám màu nặng)
    try:
        gains = _wb.estimate_wb_gains(p)
        d["cast_b"], d["cast_r"] = round(gains["b"], 3), round(gains["r"], 3)
    except Exception:
        d["cast_b"] = d["cast_r"] = 1.0

    # nét
    d["lap_var"] = round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 1)

    # mắt ngữ nghĩa (chậm hơn — tùy chọn)
    if with_masks:
        try:
            from ai_engine.specialists.segment_room.seg import segment
            masks = segment(img)
            d["frac_window"] = round(float(masks["window"].mean()), 3)
            d["frac_object"] = round(float(masks["object"].mean()), 3)
            d["frac_wall"] = round(float(masks["wall"].mean()), 3)
            d["_masks"] = masks  # cho op mask-based dùng lại, không serialize
        except Exception as e:
            d["frac_window"] = d["frac_object"] = d["frac_wall"] = -1.0
            d["_masks"] = None
            d["mask_error"] = str(e)[:80]
    return d
