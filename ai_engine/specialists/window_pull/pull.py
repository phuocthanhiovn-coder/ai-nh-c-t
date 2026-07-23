"""
Con "WINDOW PULL" v0 (deterministic, single-image — Task 16).

Phuc hoi ngoai canh qua cua so bi chay sang: nen highlight + tang bao hoa
CHI BEN TRONG vung cua so (mask tu window_mask.detect_windows), ngoai mask
feather pixel PHAI bit-identical voi input.

HOP DONG OPERATOR:
    apply(img: np.ndarray float32 [0,1] HxWx3 BGR, params: dict) -> np.ndarray (cung shape)

params:
    strength          (0-1, default 0.7)  — do manh duong cong nen highlight
    saturation_boost  (0-0.6, default 0.25) — tang bao hoa, scale theo luong
                       luminance da phuc hoi tung pixel

Gate: win_fraction < 0.005 (khong co cua so) hoac > 0.5 (mask nghi ngo, qua nua
khung la "cua so" = detect sai) -> tra nguyen anh.

Cach lam (theo spec Task 16):
  1. split_frequency: chi bien doi LOW-freq, bom lai HIGH-freq GOC de phan chieu
     kinh / mullion giu nguyen do net.
  2. Tren low-freq, doi sang linear-light, nen highlight kieu Reinhard tren luma
     PHIA TREN pivot; pivot = median luma NOI THAT (ngoai mask) — thich nghi
     tung anh. Duoi pivot: giu nguyen (noi that khong doi).
  3. Tang bao hoa (HSV S) ty le voi recovery (luma da keo xuong bao nhieu).
  4. composite_mask + feather; assert ngoai alpha==0 khong doi 1 bit.
"""

import os
import sys

import cv2
import numpy as np

cv2.setNumThreads(2)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from ai_engine.core.quality import (  # noqa: E402
    composite_mask,
    merge_frequency,
    split_frequency,
    to_linear,
    to_srgb,
)
from ai_engine.specialists.window_pull.window_mask import detect_windows  # noqa: E402

WIN_FRACTION_GATE_LO = 0.005
WIN_FRACTION_GATE_HI = 0.5
FEATHER_PX = 5.0
LOWFREQ_SIGMA = 8.0
# He so cong duong cong Reinhard t' = t/(1+a*t): a = CURVE_A_MAX * strength.
# strength=0.7 -> a=2.8 -> highlight trang tinh khoi (t=1) keo ve t'=0.263.
CURVE_A_MAX = 4.0
PIVOT_MIN, PIVOT_MAX = 0.02, 0.5  # chan pivot (linear luma) trong khoang lanh manh
INTERIOR_MASK_MAX = 0.25          # pixel mask < muc nay coi la "noi that" de tinh pivot
# Trong so "co chi tiet de phuc hoi": pixel trang CLIP HOAN TOAN (chroma ~ 0)
# khong con thong tin — keo xuong chi ra XAM BAN. Pull nhe pixel do, pull du
# manh pixel con mau (troi xanh, mat tien nha...). Ramp theo chroma tuong doi.
# Trong so PHAI MUOT theo khong gian (blur manh) — neu de tung pixel, vat trang
# giua vung mau bi pull lech nhau -> dom sang loang lo (da thay o DSC1197).
# DETAIL_W_MIN qua thap (0.12 da thu) -> vung trang clip GIU sang trong khi
# hang xom co mau bi keo toi -> nhin nhu VET SUA loang (da thay o DSC1251).
# 0.45 = keo trang clip vua phai, chenh lech voi vung mau it hon.
CHROMA_FULL = 0.10     # chroma tuong doi >= muc nay -> pull 100%
DETAIL_W_MIN = 0.45    # pixel trang tinh khoi van duoc pull 45% luc
DETAIL_W_SIGMA = 24.0  # blur Gaussian lam muot ban do trong so
# LOCAL TONE MAPPING: tinh gain tu ANH NEN (illumination base, bilateral tren
# ban thu nho) chu KHONG tu luma truc tiep — Reinhard truc tiep nghien contrast
# cuc bo vung sang xuong ~7% -> chi tiet toa nha trang bien thanh SUA loang
# (da thay o DSC1251/DSC1197). Gain muot cuc bo => keo sang xuong ma GIU contrast.
BASE_DOWN = 4              # tinh base tren ban thu nho 1/4
BASE_BILAT_SIGMA_COLOR = 0.15
BASE_BILAT_SIGMA_SPACE = 8  # ~32px o full-res sau khi upscale
# Knee: duong cong chi bat dau nen TU pivot + KNEE*span tro len — bao ve tuong
# sang lot vao mask (spill cua hull) khoi bi toi di; noi that giu nguyen.
KNEE = 0.12


def _luma_linear(lin_bgr):
    """Rec.709 luma tren anh linear BGR."""
    return (
        0.0722 * lin_bgr[:, :, 0]
        + 0.7152 * lin_bgr[:, :, 1]
        + 0.2126 * lin_bgr[:, :, 2]
    ).astype(np.float32)


def _final_alpha(mask, feather_px):
    """Tinh lai CHINH XAC alpha ma composite_mask() dung (mask da full-res nen
    chi con buoc feather + clip) — de assert ngoai mask khong doi 1 pixel."""
    m = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    if feather_px and feather_px > 0:
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=feather_px)
    return np.clip(m, 0.0, 1.0)


def _pull_lowfreq(low, mask, strength, saturation_boost):
    """Bien doi low-freq: nen highlight linear + sat boost. Tra ve low da sua."""
    lin = to_linear(np.clip(low, 0.0, 1.0))
    luma = _luma_linear(lin)

    # pivot thich nghi: median luma linear cua NOI THAT (ngoai mask)
    interior = mask < INTERIOR_MASK_MAX
    if int(interior.sum()) >= 100:
        pivot = float(np.median(luma[interior]))
    else:
        pivot = float(np.median(luma))
    pivot = float(np.clip(pivot, PIVOT_MIN, PIVOT_MAX))
    # knee: nguong bat dau nen thuc te (bao ve tuong sang lot vao mask)
    pivot_eff = pivot + KNEE * (1.0 - pivot)

    # anh nen (illumination): bilateral tren ban thu nho -> muot cuc bo, bam canh
    h, w = luma.shape[:2]
    small = cv2.resize(
        luma, (max(1, w // BASE_DOWN), max(1, h // BASE_DOWN)),
        interpolation=cv2.INTER_AREA,
    )
    small = cv2.bilateralFilter(
        small, d=-1,
        sigmaColor=BASE_BILAT_SIGMA_COLOR, sigmaSpace=BASE_BILAT_SIGMA_SPACE,
    )
    base = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    base = np.maximum(base, 1e-6)

    # Reinhard tren ANH NEN vuot pivot_eff: t=(B-p)/(1-p), t'=t/(1+a*t)
    a = CURVE_A_MAX * float(strength)
    span = max(1.0 - pivot_eff, 1e-6)
    t = np.clip((base - pivot_eff) / span, 0.0, None)
    t_c = t / (1.0 + a * t)
    base_new = np.where(base > pivot_eff, pivot_eff + t_c * span, base)

    gain = np.where(base > pivot_eff, base_new / base, 1.0).astype(np.float32)

    # trong so chi tiet: chroma tuong doi (max-min)/max cua linear RGB.
    # Trang clip hoan toan -> w thap -> pull nhe (tranh xam ban).
    c_max = lin.max(axis=2)
    c_min = lin.min(axis=2)
    chroma_rel = (c_max - c_min) / np.maximum(c_max, 1e-6)
    w_detail = np.clip(chroma_rel / CHROMA_FULL, DETAIL_W_MIN, 1.0).astype(np.float32)
    w_detail = cv2.GaussianBlur(w_detail, (0, 0), sigmaX=DETAIL_W_SIGMA)

    gain = 1.0 - (1.0 - gain) * w_detail
    luma_new = luma * gain
    lin_pulled = lin * gain[:, :, None]

    # recovery [0,1]: pixel bi keo xuong bao nhieu (0 = khong doi, 1 = keo het span)
    recovery = np.clip((luma - luma_new) / span, 0.0, 1.0).astype(np.float32)

    low_pulled = to_srgb(np.clip(lin_pulled, 0.0, 1.0))

    if saturation_boost > 0:
        # sat_w: KHONG boost bao hoa pixel gan trung tinh — boost cast am nhe cua
        # trang clip se nhuom mau kem/vang len toa nha trang (thay o DSC1251).
        sat_w = np.clip(chroma_rel / CHROMA_FULL, 0.0, 1.0).astype(np.float32)
        sat_w = cv2.GaussianBlur(sat_w, (0, 0), sigmaX=DETAIL_W_SIGMA)
        hsv = cv2.cvtColor(np.clip(low_pulled, 0.0, 1.0), cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(
            hsv[:, :, 1] * (1.0 + float(saturation_boost) * recovery * sat_w),
            0.0, 1.0,
        )
        low_pulled = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return np.clip(low_pulled, 0.0, 1.0).astype(np.float32)


def apply(img, params=None):
    """HOP DONG OPERATOR: apply(img float32 [0,1] HxWx3 BGR, params) -> cung shape."""
    params = params or {}
    strength = float(np.clip(params.get("strength", 0.7), 0.0, 1.0))
    saturation_boost = float(np.clip(params.get("saturation_boost", 0.25), 0.0, 0.6))

    assert img.dtype in (np.float32, np.float64)
    img = np.asarray(img, dtype=np.float32)
    h, w = img.shape[:2]

    mask, win_fraction = detect_windows(img)

    if (
        win_fraction < WIN_FRACTION_GATE_LO
        or win_fraction > WIN_FRACTION_GATE_HI
        or strength <= 0.0
    ):
        out = img.copy()
        assert out.shape[0] == h and out.shape[1] == w
        return out

    # 1. tach tan so — chi sua LOW, HIGH goc bom lai nguyen ven
    low, high = split_frequency(img, LOWFREQ_SIGMA)
    low_pulled = _pull_lowfreq(low, mask, strength, saturation_boost)
    edited = merge_frequency(low_pulled, high)

    # 2. composite theo mask feather
    out = composite_mask(img, edited, mask, feather_px=FEATHER_PX)
    out = np.clip(out, 0.0, 1.0).astype(np.float32)

    # 3. assert ngoai mask bit-identical (nhu sky_replace)
    final_alpha = _final_alpha(mask, FEATHER_PX)
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
    print("Window pull module loaded.")
