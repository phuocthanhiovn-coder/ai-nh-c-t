"""
Con "LAM CO XANH" (grass greening, CV thuan, deterministic).

HOP DONG OPERATOR:
    apply(img: np.ndarray float32 [0,1] HxWx3 BGR, params: dict) -> np.ndarray (cung shape)

Khong resize / re-encode / doi so kenh mau. Ngoai mask -> khong doi 1 pixel.
"""

import cv2
import numpy as np

cv2.setNumThreads(2)

# Dai hue OpenCV (0-179) tu co ua vang toi xanh la
HUE_LO = 18
HUE_HI = 95
# SAT_MIN=25 tung qua thap: be tong/via he ngoai nang (hue ~20-24, tan/xam am)
# de dang lot qua nguong nay va bi nham la co ua vang -> mask lem ca duong be tong.
# 50 la nguong toi thieu de tach co (sat thuong >=60-130) khoi be tong (sat thuong 30-45).
SAT_MIN = 50
VAL_MIN = 25
VAL_MAX = 245

# Uu tien nua duoi anh: duoi day anh trong so 1.0, tren dinh trong so sàn (khong loai han
# vi vi anh chup xa co the co doi co o giua/tren khung hinh)
POS_RAMP_START = 0.30   # tu day anh (0=dinh, 1=day)
POS_RAMP_END = 0.55
POS_FLOOR = 0.15

TEXTURE_BLUR_SIGMA = 3.0
TEXTURE_BOX_KSIZE = 9
# nguong nang luong high-freq (thang 0-255) de phan biet co (nham) vs be mat phang
# (tuong son, be tong). 3.0 qua thap: nhieu li ti tren be tong ngoai nang (soi da, JPEG
# blocking) da du vuot nguong roi bi morphology CLOSE + feather noi lien thanh mang lon.
TEXTURE_THRESH = 7.0

MORPH_KSIZE = 5
FEATHER_SIGMA = 5.0

# Loc thanh phan lien thong "mong" (canh khung cua, vien may giat, ron gach) truoc khi
# morphology: cac canh do tao mang luoi mau/texture gia trong anh NOI THAT du qua duoc
# color+texture mask nhung la duong net MONG trai dai (bbox lon, dien tich thuc rat nho
# -> fill_ratio thap) hoac VACH THANG mong (mot chieu bbox nho). Co that la mang lien
# thong DAC (fill_ratio cao VA ca hai chieu bbox du lon). Xem bao cao cuoi file spec.
MIN_COMPONENT_FILL_RATIO = 0.20
MIN_COMPONENT_MIN_DIM = 45  # px, o ca chieu rong lan cao cua bbox thanh phan

TARGET_HUE = 50.0       # hue OpenCV cho co xanh khoe (~45-55)
HUE_SHIFT_FRAC = 0.6    # chi dich toi da 60% khoang cach ve target, tranh qua tay / mau khac le
SAT_GAIN = 0.5
SAT_ABS_CLAMP = 70.0
VAL_GAIN = 0.08
VAL_ABS_CLAMP = 15.0


def _to_u8(img):
    return np.clip(img.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)


def _filter_thin_components(binary_u8, min_fill_ratio=MIN_COMPONENT_FILL_RATIO,
                             min_dim=MIN_COMPONENT_MIN_DIM):
    """Giu lai chi cac thanh phan lien thong DAC (fill_ratio = dien_tich/bbox_area cao
    VA ca hai chieu bbox du lon). Loai mang luoi canh mong (khung cua, ron gach, vien
    thiet bi) hay bi color+texture mask nham la co trong anh noi that."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary_u8, connectivity=8)
    if n <= 1:
        return binary_u8
    keep = np.zeros(n, dtype=bool)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area <= 0:
            continue
        fill_ratio = area / float(w * h)
        if min(w, h) >= min_dim and fill_ratio >= min_fill_ratio:
            keep[i] = True
    keep_mask = keep[labels]
    return np.where(keep_mask, binary_u8, 0).astype(np.uint8)


def segment_grass(img):
    """
    Tra ve mask mem [0,1] (float32, HxW) cho vung co.
    Ket hop: (1) dai mau HSV co ua-vang -> xanh la, (2) uu tien nua duoi anh,
    (3) texture high-freq dac trung cua co (loai tuong son phang).
    Feather bang morphology (open+close) + Gaussian blur.
    """
    h, w = img.shape[:2]
    img_u8 = _to_u8(img)

    hsv = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    color_mask = (
        (hue >= HUE_LO) & (hue <= HUE_HI) &
        (sat >= SAT_MIN) &
        (val >= VAL_MIN) & (val <= VAL_MAX)
    ).astype(np.float32)

    yy = np.arange(h, dtype=np.float32).reshape(-1, 1) / max(h - 1, 1)
    pos_ramp = np.clip((yy - POS_RAMP_START) / (POS_RAMP_END - POS_RAMP_START), 0.0, 1.0)
    pos_weight = POS_FLOOR + (1.0 - POS_FLOOR) * pos_ramp
    pos_weight = np.repeat(pos_weight, w, axis=1)

    gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=TEXTURE_BLUR_SIGMA)
    highfreq = np.abs(gray - blur)
    texture_energy = cv2.boxFilter(highfreq, -1, (TEXTURE_BOX_KSIZE, TEXTURE_BOX_KSIZE))
    texture_mask = (texture_energy >= TEXTURE_THRESH).astype(np.float32)

    raw = color_mask * texture_mask * pos_weight

    mask_u8 = np.clip(raw * 255.0, 0, 255).astype(np.uint8)
    mask_u8 = _filter_thin_components(mask_u8)
    kernel = np.ones((MORPH_KSIZE, MORPH_KSIZE), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    mask_soft = cv2.GaussianBlur(mask_u8.astype(np.float32), (0, 0), sigmaX=FEATHER_SIGMA)
    mask_soft = np.clip(mask_soft / 255.0, 0.0, 1.0)
    return mask_soft.astype(np.float32)


def green_boost(img, mask, strength):
    """
    Trong mask: dich hue ve xanh co khoe (~TARGET_HUE), tang saturation co gioi han,
    nang nhe luminance. Tra ve anh full-size da boost TOAN CUC (chua composite mask) -
    apply() se composite bang mask mem de ngoai mask khong doi 1 pixel.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    img_u8 = _to_u8(img)
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    shift_amount = mask * strength
    hue_diff = TARGET_HUE - hue
    new_hue = hue + hue_diff * shift_amount * HUE_SHIFT_FRAC
    new_hue = np.clip(new_hue, 0, 179)

    sat_gain = 1.0 + SAT_GAIN * shift_amount
    new_sat = sat * sat_gain
    new_sat = np.minimum(new_sat, sat + SAT_ABS_CLAMP)
    new_sat = np.clip(new_sat, 0, 255)

    val_gain = 1.0 + VAL_GAIN * shift_amount
    new_val = val * val_gain
    new_val = np.minimum(new_val, val + VAL_ABS_CLAMP)
    new_val = np.clip(new_val, 0, 255)

    hsv_out = np.stack([new_hue, new_sat, new_val], axis=-1)
    hsv_out = np.clip(hsv_out, 0, 255).astype(np.uint8)
    bgr_out = cv2.cvtColor(hsv_out, cv2.COLOR_HSV2BGR)
    return bgr_out.astype(np.float32) / 255.0


def apply(img, params=None):
    """
    HOP DONG OPERATOR: apply(img float32 [0,1] HxWx3 BGR, params) -> cung shape.
    params: strength 0-1 (default 0.7). Khong co co (mask ~ 0) -> tra nguyen anh.
    """
    params = params or {}
    strength = float(np.clip(params.get("strength", 0.7), 0.0, 1.0))

    assert img.dtype == np.float32 or img.dtype == np.float64
    h, w = img.shape[:2]

    mask = segment_grass(img)

    if mask.max() < 1e-6:
        out = img.astype(np.float32).copy()
        assert out.shape[0] == h and out.shape[1] == w
        return out

    boosted = green_boost(img, mask, strength)
    mask_3 = mask[:, :, None]
    out = img.astype(np.float32) * (1.0 - mask_3) + boosted * mask_3
    out = np.clip(out, 0.0, 1.0).astype(np.float32)

    assert out.shape[0] == h and out.shape[1] == w, "Kich thuoc output phai khop input"
    return out


if __name__ == "__main__":
    print("Grass green module loaded.")
