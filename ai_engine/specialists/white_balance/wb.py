"""
Con "CAN TRANG + AUTO-EXPOSURE" (CV thuan, deterministic).

HOP DONG OPERATOR:
    apply(img: np.ndarray float32 [0,1] HxWx3 BGR, params: dict) -> np.ndarray (cung shape)

Khong resize / re-encode / doi so kenh mau.
"""

import cv2
import numpy as np

EPS = 1e-6
GAMMA = 2.2

GAIN_CLAMP = (0.6, 1.6)
EXPOSURE_GAIN_MAX = 2.0
EXPOSURE_GAMMA_CLAMP = (0.6, 1.6)


def _degamma(x):
    return np.power(np.clip(x, 0.0, None), GAMMA)


def _regamma(x):
    return np.power(np.clip(x, 0.0, None), 1.0 / GAMMA)


def _luma(img):
    # img la BGR -> B=0, G=1, R=2
    return 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]


def estimate_wb_gains(img):
    """
    Hybrid gray-world (vung mid-tone, bo pixel bao hoa mau cao) + white-patch (percentile 99).
    Tra ve dict {'b':, 'g':, 'r':} - gain nhan truc tiep trong khong gian tuyen tinh.
    G luon = 1.0 (dung lam kenh tham chieu), R/B duoc can theo G.
    """
    img = img.astype(np.float32)
    lum = _luma(img)

    # 1) Gray-world tren vung mid-tone: bo 5% toi nhat + 5% sang nhat + pixel bao hoa mau cao
    p_lo, p_hi = np.percentile(lum, [5, 95])
    mask_tone = (lum >= p_lo) & (lum <= p_hi)

    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    mask_sat = sat < 0.4  # bo mau bao hoa cao (kho la be mat trung tinh)

    mask = mask_tone & mask_sat
    if mask.sum() < 200:
        mask = mask_tone
    if mask.sum() < 200:
        mask = np.ones_like(lum, dtype=bool)

    mean_b = float(img[:, :, 0][mask].mean())
    mean_g = float(img[:, :, 1][mask].mean())
    mean_r = float(img[:, :, 2][mask].mean())
    gw_gain_b = mean_g / (mean_b + EPS)
    gw_gain_r = mean_g / (mean_r + EPS)

    # 2) White-patch: percentile 99 moi kenh tren toan anh (gia dinh vung sang nhat ~ trang)
    wp_b = float(np.percentile(img[:, :, 0], 99))
    wp_g = float(np.percentile(img[:, :, 1], 99))
    wp_r = float(np.percentile(img[:, :, 2], 99))
    wp_gain_b = wp_g / (wp_b + EPS)
    wp_gain_r = wp_g / (wp_r + EPS)

    # 3) Trung binh 2 uoc luong, clamp gain moi kenh
    gain_b = (gw_gain_b + wp_gain_b) / 2.0
    gain_r = (gw_gain_r + wp_gain_r) / 2.0

    gain_b = float(np.clip(gain_b, *GAIN_CLAMP))
    gain_r = float(np.clip(gain_r, *GAIN_CLAMP))

    return {"b": gain_b, "g": 1.0, "r": gain_r}


def apply_wb(img, gains, strength=1.0):
    """Nhan gain trong khong gian tuyen tinh (degamma 2.2 -> gain -> regamma), noi suy theo strength."""
    img = img.astype(np.float32)
    strength = float(np.clip(strength, 0.0, 1.0))

    gain_vec = np.array([gains["b"], gains["g"], gains["r"]], dtype=np.float32)
    eff_gain = 1.0 + strength * (gain_vec - 1.0)

    linear = _degamma(img)
    linear = linear * eff_gain.reshape(1, 1, 3)
    out = _regamma(linear)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def auto_exposure(img, params=None):
    """
    Percentile stretch (p1 -> ~0.02, p99.5 -> ~0.98) bang gain+offset tuyen tinh, gain <= 2.0x.
    Tuy chon dich median (`target_median`) bang gamma adaptive, gamma trong [0.6, 1.6].

    Tra ve (img_out, info) - info de run_samples.py in so lieu; KHONG thuoc hop dong operator apply().
    """
    params = params or {}
    exposure = params.get("exposure", "auto")
    target_median = params.get("target_median", 0.42)

    img = img.astype(np.float32)
    lum = _luma(img)

    if exposure == "auto":
        p1, p995 = np.percentile(lum, [1, 99.5])
        p1 = float(p1)
        p995 = float(p995)
        spread = p995 - p1
        if spread < 1e-4:
            gain = 1.0
        else:
            gain = (0.98 - 0.02) / spread
        gain = float(np.clip(gain, 1.0 / EXPOSURE_GAIN_MAX, EXPOSURE_GAIN_MAX))
        offset = 0.02 - gain * p1
    else:
        ev = float(exposure)
        gain = float(np.clip(2.0 ** ev, 1.0 / EXPOSURE_GAIN_MAX, EXPOSURE_GAIN_MAX))
        offset = 0.0

    out = img * gain + offset
    out = np.clip(out, 0.0, 1.0)

    gamma = 1.0
    if target_median is not None:
        lum_out = _luma(out)
        med = float(np.clip(np.median(lum_out), 1e-3, 1.0 - 1e-3))
        tgt = float(np.clip(target_median, 1e-3, 1.0 - 1e-3))
        if abs(med - tgt) > 1e-3:
            gamma = float(np.log(tgt) / np.log(med))
            # Anh BDS chuan la sang-thoang: chi NANG anh toi (gamma<1),
            # anh von sang hon target thi giu nguyen do sang (toi da lam toi cuc nhe)
            # -> tranh ca darkening keo mau am dam len (bug phat hien khi review _ML_1493)
            if med > tgt:
                gamma = min(gamma, 1.08)
            gamma = float(np.clip(gamma, *EXPOSURE_GAMMA_CLAMP))
            out = np.power(np.clip(out, 0.0, 1.0), gamma)

    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    info = {"gain": gain, "offset": offset, "gamma": gamma}
    return out, info


def apply(img, params=None):
    """
    HOP DONG OPERATOR: apply(img float32 [0,1] HxWx3 BGR, params) -> cung shape.
    params: wb_strength (0-1, default 0.8), exposure ("auto"|so EV, default "auto"), target_median.
    """
    params = params or {}
    wb_strength = params.get("wb_strength", 0.8)
    exposure = params.get("exposure", "auto")
    target_median = params.get("target_median", 0.42)

    assert img.dtype == np.float32 or img.dtype == np.float64
    h, w = img.shape[:2]

    gains = estimate_wb_gains(img)
    out = apply_wb(img, gains, wb_strength)
    out, _info = auto_exposure(out, {"exposure": exposure, "target_median": target_median})

    assert out.shape[0] == h and out.shape[1] == w, "Kich thuoc output phai khop input"
    return out.astype(np.float32)
