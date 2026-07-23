"""
Con "SKY REPLACE" (deterministic v0, CV thuan — khong neural net ngoai API dung 1 lan
de dung sky_assets.py, ma sky_assets.py lai KHONG goi API nao, chi numpy thuan).

HOP DONG OPERATOR:
    apply(img: np.ndarray float32 [0,1] HxWx3 BGR, params: dict) -> np.ndarray (cung shape)

params:
    sky        ("blue"|"golden"|"dusk"|"hazy", default "blue")
    strength   (0-1, default 1.0) — noi suy opacity cua mask thay troi
    harmonize  (bool, default True) — dich Lab mean/std cua plate ve thong ke
               vung troi GOC de khop do sang cua canh.

Gate: sky_fraction < SKY_FRACTION_GATE -> tra nguyen anh (anh noi that / khong co troi ro).
Ngoai mask (final alpha == 0): pixel PHAI giu bit-identical voi input.
"""

import os
import sys

import cv2
import numpy as np

cv2.setNumThreads(2)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from ai_engine.core.quality import composite_mask  # noqa: E402
from ai_engine.specialists.sky_replace.sky_assets import ensure_skies  # noqa: E402
from ai_engine.specialists.sky_replace.sky_mask import detect_sky  # noqa: E402


# Task 17: sky_mask.py rewrite dung tolerance HEP (precision-first) nen sky_fraction
# gio THAP HON NHIEU so voi ban cu (ban cu bi bleed nen fraction bi thoi phong ao).
# Anh ngoai troi that voi 1 manh troi nho (VD _ML_1605: bi nha/cay che gan het,
# fraction moi chi ~0.01) truoc day van qua nguong 0.04 (nho bleed), gio se bi
# gate oan neu giu nguyen 0.04. Ha nguong xuong duoi muc thap nhat cua exterior
# that do (~0.004-0.01) nhung van cao han han 0.0 (interior/false-seed da bi loai
# tan goc trong detect_sky qua MIN_SEED_TOP_FRAC) -> gate van an toan.
SKY_FRACTION_GATE = 0.006
# AN TOAN: troi that phai MAT/XANH (kenh B > R). Tuong/tran trang noi that trung tinh
# hoac am (B<=R) -> loai, tranh son xanh len tuong phong tam (bug that da thay _ML_1336).
# Danh doi: troi am u trang/xam bi bo qua (an toan hon la pha noi that).
SKY_COOLNESS_MIN = 0.025
FEATHER_PX = 4.0
HARMONIZE_L_BLEND = 0.85  # bao nhieu % dich mean/std luminance ve thong ke troi goc


def _to_u8(img):
    return np.clip(np.asarray(img, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)


def _fit_plate_cover(plate_f32, H, W):
    """Resize plate (cover-fill, giu ti le) de phu het khung HxW, ANCHOR LEN TREN
    (zenith cua plate = dinh khung), crop giua theo chieu ngang. Plate luon phu
    het bbox cua mask troi vi bbox <= toan khung."""
    ph, pw = plate_f32.shape[:2]
    scale = max(W / pw, H / ph)
    new_w = max(W, int(np.ceil(pw * scale)))
    new_h = max(H, int(np.ceil(ph * scale)))
    resized = cv2.resize(plate_f32, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    x0 = (new_w - W) // 2
    y0 = 0  # anchor top: dinh plate (zenith) = dinh khung
    return resized[y0 : y0 + H, x0 : x0 + W].astype(np.float32)


def _harmonize_to_scene(plate_bgr, scene_bgr, sky_mask_bool, blend=HARMONIZE_L_BLEND):
    """Dich Lab-L (luminance) cua plate ve gan thong ke (mean/std) cua vung troi GOC,
    giu nguyen mau sac (a/b) cua plate de khong mat 'ban sac' cua tung loai troi."""
    plate_u8 = _to_u8(plate_bgr)
    scene_u8 = _to_u8(scene_bgr)

    plate_lab = cv2.cvtColor(plate_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    scene_lab = cv2.cvtColor(scene_u8, cv2.COLOR_BGR2LAB).astype(np.float32)

    if sky_mask_bool.sum() < 50:
        return plate_bgr  # khong du diem troi goc de uoc luong thong ke, giu plate nguyen

    orig_L = scene_lab[:, :, 0][sky_mask_bool]
    orig_mean, orig_std = float(orig_L.mean()), float(orig_L.std() + 1e-6)

    plate_L = plate_lab[:, :, 0]
    plate_mean, plate_std = float(plate_L.mean()), float(plate_L.std() + 1e-6)

    target_mean = plate_mean + blend * (orig_mean - plate_mean)
    target_std = plate_std + blend * (orig_std - plate_std)
    ratio = target_std / plate_std

    new_L = (plate_L - plate_mean) * ratio + target_mean
    new_L = np.clip(new_L, 0, 255)

    out_lab = plate_lab.copy()
    out_lab[:, :, 0] = new_L
    out_lab_u8 = np.clip(out_lab, 0, 255).astype(np.uint8)
    out_bgr_u8 = cv2.cvtColor(out_lab_u8, cv2.COLOR_LAB2BGR)
    return out_bgr_u8.astype(np.float32) / 255.0


def _final_alpha(mask, feather_px):
    """Tinh lai CHINH XAC alpha ma composite_mask() se dung (mask da full-res nen
    guided_upsample ben trong composite_mask se bi bo qua, chi con buoc feather +
    clip) — dung de assert ngoai mask khong doi 1 pixel."""
    m = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    if feather_px and feather_px > 0:
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=feather_px)
    return np.clip(m, 0.0, 1.0)


def apply(img, params=None):
    """
    HOP DONG OPERATOR: apply(img float32 [0,1] HxWx3 BGR, params) -> cung shape.
    """
    params = params or {}
    sky_name = params.get("sky", "blue")
    strength = float(np.clip(params.get("strength", 1.0), 0.0, 1.0))
    harmonize = bool(params.get("harmonize", True))

    assert img.dtype == np.float32 or img.dtype == np.float64
    img = np.asarray(img, dtype=np.float32)
    h, w = img.shape[:2]

    mask, sky_fraction = detect_sky(img)

    if sky_fraction < SKY_FRACTION_GATE:
        out = img.copy()
        assert out.shape[0] == h and out.shape[1] == w
        return out

    # Cong AN TOAN mau sac: vung "troi" phai mat/xanh that (mean B - mean R).
    # Chan false-positive tuong/tran trang noi that (trung tinh/am).
    sky_bool_gate = mask > 0.5
    if sky_bool_gate.sum() >= 50:
        region = img[sky_bool_gate]
        coolness = float(region[:, 0].mean() - region[:, 2].mean())  # B - R
        if coolness < SKY_COOLNESS_MIN:
            out = img.copy()
            assert out.shape[0] == h and out.shape[1] == w
            return out

    plates = ensure_skies()
    if sky_name not in plates:
        sky_name = "blue"
    plate_u8 = cv2.imread(plates[sky_name])
    if plate_u8 is None:
        raise IOError(f"Khong doc duoc sky plate: {plates[sky_name]}")
    plate_f32 = plate_u8.astype(np.float32) / 255.0

    plate_fitted = _fit_plate_cover(plate_f32, h, w)

    if harmonize:
        sky_bool = mask > 0.5
        plate_fitted = _harmonize_to_scene(plate_fitted, img, sky_bool)

    mask_eff = np.clip(mask * strength, 0.0, 1.0)

    out = composite_mask(img, plate_fitted, mask_eff, feather_px=FEATHER_PX)
    out = np.clip(out, 0.0, 1.0).astype(np.float32)

    final_alpha = _final_alpha(mask_eff, FEATHER_PX)
    outside = final_alpha <= 0.0
    if outside.any():
        diff = np.abs(out - img)
        max_diff_outside = float(diff[outside].max())
        assert max_diff_outside == 0.0, (
            f"Pixel ngoai mask bi doi: max_diff_outside={max_diff_outside}"
        )

    assert out.shape[0] == h and out.shape[1] == w, "Kich thuoc output phai khop input"
    return out


if __name__ == "__main__":
    print("Sky replace module loaded.")
