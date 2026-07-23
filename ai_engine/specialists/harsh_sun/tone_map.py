"""
Con "HARSH-SUN EXTERIOR" (Task 22) - local tone-mapping deterministic, khong model.

Ca kho cua khach: ngoai that chup nang gat -> troi/tuong chay trang + bong den kit.
Nen dung 1 exposure duy nhat: tach base/detail tren log-luminance bang guided filter
(edge-aware, chong halo), NEN dai dong tren BASE (keo highlight xuong, nang shadow len
bang soft-knee bat doi xung), roi CONG LAI detail nguyen ven -> texture/canh giu net.
Mau: scale RGB tuyen tinh theo ti le luminance + phuc hoi saturation ti le voi muc nen
(kieu Mantiuk s-exponent) de KHONG bac mau nhu model hoc (loi washout cua auto_enhance).

HOP DONG OPERATOR:
    apply(img: np.ndarray float32 [0,1] HxWx3 BGR, params: dict) -> np.ndarray cung shape
Khong resize / re-encode. Toan bo tinh toan o linear-light, full-res.

params (deu clamp [0,1]):
    strength          0-1  default 0.7  - gain tong
    highlight_recover 0-1  default 0.8  - keo vung chay xuong
    shadow_lift       0-1  default 0.5  - mo vung toi
    local_contrast    0-1  default 0.3  - boost detail (clarity nhe)
    sat_restore       0-1  default 0.4  - phuc hoi mau theo muc nen

GATE tu dong: do % pixel chay (luma>=0.95) va % pixel kit (luma<=0.05) de scale
highlight/shadow rieng re -> anh phoi sang chuan gan nhu khong doi (acceptance 2).
"""

import cv2
import numpy as np

cv2.setNumThreads(2)

from ai_engine.core.quality import to_linear, to_srgb  # noqa: E402

LOG_EPS = 1e-6          # tranh log2(0); L=1e-6 ~ -20EV, duoi muc nhieu moi cam bien
LOG_NORM_LO = -14.0     # chuan hoa log2-luma ve [0,1] cho guided filter (eps on dinh)
LOG_NORM_HI = 0.0

# Guided filter: radius lon (~6% canh ngan) de base that su LOCAL-nhung-muot;
# eps nho tren thang [0,1] de canh manh (roofline/troi) KHONG bi nhoe vao base -> chong halo.
BASE_RADIUS_FRAC = 0.06
BASE_RADIUS_MIN = 16
GUIDED_EPS = 0.01               # (0.1)^2 tren guide da chuan hoa
BILATERAL_MAX_DIM = 1024        # fallback khong co ximgproc: bilateral tren ban thu nho

# Anchor CO DINH theo cam nhan (KHONG theo percentile anh - bai hoc lan chay dau:
# anh co bong kit lam percentile-mid tut xuong ~-7EV -> moi pixel that deu bi coi la
# "highlight" va keo toi ca anh). EV = log2(luma linear), 0 EV = trang.
HI_REF_EV = -1.25    # vai highlight: tren muc nay (srgb ~>0.78) moi bat dau keo xuong
HI_KNEE_SCALE = 1.25  # thang do knee highlight (EV)
LO_REF_EV = -3.3     # chan shadow: duoi muc nay (srgb ~<0.35) moi bat dau nang
LO_KNEE_SCALE = 4.0   # thang do knee shadow (EV)
ALPHA_HI_MAX = 2.0   # do doc knee toi da (params=1): dinh trang keo ~0.8EV
ALPHA_LO_MAX = 2.0
MAX_LIFT_EV = 3.0    # tran nang shadow (8x) - qua nua la khuech dai nhieu
# Den THAT (letterbox, goc vignette) khong duoc nang thanh xam: weight ve 0 duoi day
BLACK_GUARD_LO_EV = -11.5
BLACK_GUARD_HI_EV = -9.0

# Nguong gate (phan tram dien tich anh)
HI_CLIP_T = 0.95
LO_CRUSH_T = 0.05
HI_GATE_LO, HI_GATE_HI = 0.002, 0.03   # <0.2% chay -> khong dung; >3% -> gate=1
LO_GATE_LO, LO_GATE_HI = 0.005, 0.06

DETAIL_GAIN_MAX = 0.5   # local_contrast=1, strength=1 -> detail x1.5
SAT_EXP_MAX = 0.5       # sat_restore=1, nen >=2EV -> s-exponent 1.5
SAT_COMP_FULL_EV = 2.0  # muc nen (EV) coi la "het co" cho phuc hoi mau
CHROMA_RATIO_CAP = 8.0  # chan ti le kenh/luma o pixel gan den (chong noise mau)


def _smoothstep(x, lo, hi):
    t = np.clip((x - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _luminance_linear(lin_bgr):
    """Rec.709 luma tren BGR linear."""
    return (0.0722 * lin_bgr[:, :, 0]
            + 0.7152 * lin_bgr[:, :, 1]
            + 0.2126 * lin_bgr[:, :, 2]).astype(np.float32)


def _edge_aware_base(log_norm, h, w):
    """Base muot nhung bam canh cua log-luma da chuan hoa [0,1].
    Uu tien cv2.ximgproc.guidedFilter (O(N), radius lon mien phi);
    fallback bilateral tren ban thu nho + resize (chat luong thap hon, ghi chu bao cao)."""
    radius = max(BASE_RADIUS_MIN, int(round(min(h, w) * BASE_RADIUS_FRAC)))
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
        base = cv2.ximgproc.guidedFilter(
            guide=log_norm, src=log_norm, radius=radius, eps=GUIDED_EPS)
        return np.asarray(base, dtype=np.float32)

    # Fallback: bilateral full-res voi sigma lon qua cham -> lam tren ban nho.
    scale = min(1.0, BILATERAL_MAX_DIM / max(h, w))
    small = cv2.resize(log_norm, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    sig_space = max(8.0, min(small.shape[:2]) * BASE_RADIUS_FRAC)
    base_small = cv2.bilateralFilter(small, d=0, sigmaColor=0.1, sigmaSpace=sig_space)
    base = cv2.resize(base_small, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.asarray(base, dtype=np.float32)


def _soft_knee(d, alpha, scale):
    """Nen khoang cach EV d>=0 ve d/(1+alpha*d/scale): muot (C1), don dieu,
    cang xa midpoint nen cang manh, alpha=0 -> identity."""
    return d / (1.0 + alpha * d / max(scale, 1e-6))


def compute_gates(img_srgb):
    """(hi_gate, lo_gate, hi_frac, lo_frac) tu % pixel chay/kit tren luma sRGB."""
    y = (0.114 * img_srgb[:, :, 0] + 0.587 * img_srgb[:, :, 1]
         + 0.299 * img_srgb[:, :, 2])
    hi_frac = float((y >= HI_CLIP_T).mean())
    lo_frac = float((y <= LO_CRUSH_T).mean())
    hi_gate = float(_smoothstep(hi_frac, HI_GATE_LO, HI_GATE_HI))
    lo_gate = float(_smoothstep(lo_frac, LO_GATE_LO, LO_GATE_HI))
    return hi_gate, lo_gate, hi_frac, lo_frac


def apply(img, params=None):
    """HOP DONG OPERATOR: xem docstring dau file."""
    params = params or {}
    strength = float(np.clip(params.get("strength", 0.7), 0.0, 1.0))
    highlight_recover = float(np.clip(params.get("highlight_recover", 0.8), 0.0, 1.0))
    shadow_lift = float(np.clip(params.get("shadow_lift", 0.5), 0.0, 1.0))
    local_contrast = float(np.clip(params.get("local_contrast", 0.3), 0.0, 1.0))
    sat_restore = float(np.clip(params.get("sat_restore", 0.4), 0.0, 1.0))

    img = np.asarray(img, dtype=np.float32)
    assert img.ndim == 3 and img.shape[2] == 3, "Can anh HxWx3 BGR"
    h, w = img.shape[:2]

    if strength <= 0.0:
        return img.copy()

    hi_gate, lo_gate, _, _ = compute_gates(img)

    lin = to_linear(np.clip(img, 0.0, 1.0))
    lum = np.maximum(_luminance_linear(lin), 0.0)
    logl = np.log2(lum + LOG_EPS)

    log_norm = np.clip((logl - LOG_NORM_LO) / (LOG_NORM_HI - LOG_NORM_LO), 0.0, 1.0)
    base_norm = _edge_aware_base(log_norm.astype(np.float32), h, w)
    base = base_norm * (LOG_NORM_HI - LOG_NORM_LO) + LOG_NORM_LO   # ve lai don vi EV
    detail = logl - base

    # Soft-knee quanh anchor CO DINH: vai highlight (keo xuong) + chan shadow (nang len)
    alpha_hi = ALPHA_HI_MAX * strength * highlight_recover * hi_gate
    alpha_lo = ALPHA_LO_MAX * strength * shadow_lift * lo_gate

    d_hi = np.maximum(base - HI_REF_EV, 0.0)
    pull_ev = d_hi - _soft_knee(d_hi, alpha_hi, HI_KNEE_SCALE)

    d_lo = np.maximum(LO_REF_EV - base, 0.0)
    lift_ev = np.minimum(d_lo - _soft_knee(d_lo, alpha_lo, LO_KNEE_SCALE), MAX_LIFT_EV)
    lift_ev = lift_ev * _smoothstep(base, BLACK_GUARD_LO_EV, BLACK_GUARD_HI_EV)

    base_new = base - pull_ev + lift_ev

    detail_gain = 1.0 + DETAIL_GAIN_MAX * local_contrast * strength
    log_new = base_new + detail * detail_gain
    lum_new = np.maximum(np.exp2(log_new) - LOG_EPS, 0.0)

    # Mau: giu chromaticity bang ti le kenh/luma, phuc hoi saturation ti le voi
    # muc nen EV (|base - base_new|) -> vung bi nen manh khong bi bac mau.
    comp_ev = np.abs(base - base_new)
    s_exp = 1.0 + SAT_EXP_MAX * sat_restore * np.clip(comp_ev / SAT_COMP_FULL_EV, 0.0, 1.0)
    ratio = np.clip(lin / np.maximum(lum, LOG_EPS)[:, :, None], 0.0, CHROMA_RATIO_CAP)
    out_lin = np.power(ratio, s_exp[:, :, None]) * lum_new[:, :, None]

    out = np.clip(to_srgb(out_lin), 0.0, 1.0).astype(np.float32)
    assert out.shape == img.shape, "Output phai cung shape input"
    return out


if __name__ == "__main__":
    print("harsh_sun tone_map loaded.")
