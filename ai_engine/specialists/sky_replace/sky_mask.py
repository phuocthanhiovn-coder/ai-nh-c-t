"""
Con "PHAT HIEN TROI" (sky detection, CV thuan, deterministic) — proxy -> soft mask.

Task 17 REWRITE (bleed fix). Thuat toan moi (khong dung model hoc):
  1. Seeded flood-growing: hat giong = diem tren/gan bien TREN khung, SANG va MAT
     (B>=R). Loang vung tren ban proxy da lam min, dung TOLERANCE HEP theo khoang
     cach Lab toi thong ke MAT/RUNNING cua vung (mean Lab cap nhat lien tuc khi
     them diem moi) -> vung dung lai ngay khi gap bien mai nha/toa nha (mau/sang
     khac han).
  2. San sang: diem troi phai SANG hon (median luma cua vung da loang - delta nho).
     Tuong/mai toi khong the lot vao.
  3. Nhat quan theo COT: moi cot, troi phai la 1 doan LIEN TUC tinh tu bien tren.
     Duoi doan khong-phai-troi DAU TIEN trong cot -> cat bo. Quy tac nay diet
     hien tuong loang xuong duoi mai nha.
  4. Bien mem: sau khi vung nhi phan da "dac", lam mem bang blur nho + upsample
     len full-res bang guided_upsample (dung lai ham chung o core/quality.py).
  5. Cay/anten cat ngang troi: pixel cua chung KHONG dat tolerance mau -> tao lo
     hong trong mask, DUNG (troi lo ra giua canh cay). Chi morphology nho, khong
     dong lo hong manh.

Tra ve (mask float [0,1] HxW full-res, sky_fraction float).
"""

import os
import sys

import cv2
import numpy as np

cv2.setNumThreads(2)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from ai_engine.core.quality import guided_upsample  # noqa: E402

PROXY_DIM = 768  # canh dai cua ban proxy de chay heuristics

# --- (1) hat giong tren bien tren ---
TOP_BORDER_ROWS_FRAC = 0.03  # ty le hang tren cung de tim hat giong
SEED_BRIGHT_MIN = 110.0      # (B+G+R)/3 toi thieu (thang 0-255) — "sang"
SEED_COOL_MIN = 0.0          # B - R >= 0 — "mat" (troi that khong bao gio am)
# An toan noi that: 1 vai pixel sang+mat le loi (VD anh sang den tran) khong du de
# tin day la troi that. Troi that ngoai troi luon chiem MOT MANG RONG cua hang bien
# tren (khong phai vai diem rai rac). Duoi nguong nay -> coi nhu KHONG co troi.
MIN_SEED_TOP_FRAC = 0.12

# --- lam min truoc khi loang (giam nhieu/texture nho, KHONG xoa canh lon) ---
SMOOTH_SIGMA = 1.4

# --- (1) dung sai mau khi loang (khoang cach Lab toi mean CHAY cua vung) ---
GROW_LAB_TOL = 18.0     # HEP: mai nha/tuong toi se vuot xa nguong nay
GROW_MAX_ITERS = 500    # moi vong chi no rong 1px (dilate 3x3) -> gioi han an toan

# --- (2) san sang toi thieu ---
LUMA_FLOOR_DELTA = 18.0  # (thang Lab-L 0-255 cua OpenCV) duoi (median - delta) -> loai

# --- (4)/(5) morphology + bien mem ---
MORPH_KSIZE = 3        # NHO — khong dong lo hong cua canh cay/anten manh
FEATHER_SIGMA = 1.5

MIN_SEED_PIXELS = 30   # qua it hat giong -> khong co troi, tra mask rong


def _to_u8(img):
    return np.clip(np.asarray(img, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)


def _proxy(img_u8):
    h, w = img_u8.shape[:2]
    scale = PROXY_DIM / max(h, w)
    if scale >= 1.0:
        return img_u8.copy()
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(img_u8, (nw, nh), interpolation=cv2.INTER_AREA)


def _seed_mask(smooth_u8):
    """Hat giong = hang tren cung, SANG va MAT (B>=R).

    Tra ve (seed_mask, seed_top_frac) — seed_top_frac = ty le hat giong tren
    TOAN BO dai bien tren (dung de loai anh chi co vai diem sang+mat rai rac,
    VD den tran noi that, khong phai troi that)."""
    h, w = smooth_u8.shape[:2]
    top_rows = max(1, int(round(h * TOP_BORDER_ROWS_FRAC)))
    b = smooth_u8[:, :, 0].astype(np.float32)
    g = smooth_u8[:, :, 1].astype(np.float32)
    r = smooth_u8[:, :, 2].astype(np.float32)
    bright = (b + g + r) / 3.0 >= SEED_BRIGHT_MIN
    cool = (b - r) >= SEED_COOL_MIN
    seed = np.zeros((h, w), dtype=bool)
    seed[:top_rows, :] = bright[:top_rows, :] & cool[:top_rows, :]
    top_total = top_rows * w
    seed_top_frac = float(seed.sum()) / float(max(top_total, 1))
    return seed, seed_top_frac


def _grow_region(lab, seed, tol=GROW_LAB_TOL, max_iters=GROW_MAX_ITERS):
    """Loang tu hat giong bang dilate 1px/vong, chi giu diem moi trong nguong
    Lab-distance toi mean CHAY (running) cua vung da giu. Dung khi khong con
    diem moi hop le (bien mau/sang) hoac het max_iters."""
    h, w = seed.shape[:2]
    region = seed.copy()
    if not region.any():
        return region

    kernel3 = np.ones((3, 3), np.uint8)
    sum_lab = lab[region].sum(axis=0).astype(np.float64)
    count = float(region.sum())
    mean_lab = sum_lab / count

    region_u8 = region.astype(np.uint8)
    for _ in range(max_iters):
        dilated = cv2.dilate(region_u8, kernel3, iterations=1).astype(bool)
        frontier = dilated & (~region)
        if not frontier.any():
            break
        fy, fx = np.nonzero(frontier)
        px = lab[fy, fx].astype(np.float64)
        dist = np.sqrt(np.sum((px - mean_lab) ** 2, axis=1))
        accept = dist <= tol
        if not accept.any():
            break
        ay, ax = fy[accept], fx[accept]
        region[ay, ax] = True
        sum_lab += px[accept].sum(axis=0)
        count += float(accept.sum())
        mean_lab = sum_lab / count
        region_u8 = region.astype(np.uint8)

    return region


def _apply_luma_floor(region, luma, delta=LUMA_FLOOR_DELTA):
    """San sang: diem troi phai sang hon (median luma cua vung - delta)."""
    if not region.any():
        return region
    med = float(np.median(luma[region]))
    floor = med - delta
    return region & (luma >= floor)


def _enforce_column_contiguity(region):
    """Moi cot: troi phai la doan lien tuc tu bien tren. Duoi diem khong-phai-troi
    DAU TIEN trong cot -> cat bo het (diet bleed duoi mai nha)."""
    h, w = region.shape[:2]
    not_sky = ~region
    has_non_sky = not_sky.any(axis=0)
    first_false = np.where(has_non_sky, not_sky.argmax(axis=0), h)
    rows = np.arange(h, dtype=np.int32).reshape(-1, 1)
    keep = rows < first_false.reshape(1, -1)
    return region & keep


def detect_sky(img):
    """img float32 [0,1] BGR HxWx3 -> (mask float32 [0,1] HxW full-res, sky_fraction float).

    sky_fraction = ty le dien tich mask (mean cua soft mask) tren proxy — dung de gate.
    """
    img = np.asarray(img, dtype=np.float32)
    H, W = img.shape[:2]
    img_u8 = _to_u8(img)
    proxy = _proxy(img_u8)
    ph, pw = proxy.shape[:2]

    smooth = cv2.GaussianBlur(proxy, (0, 0), sigmaX=SMOOTH_SIGMA)

    seed, seed_top_frac = _seed_mask(smooth)
    if int(seed.sum()) < MIN_SEED_PIXELS or seed_top_frac < MIN_SEED_TOP_FRAC:
        # Khong co hat giong troi chac chan (VD: noi that, den tran sang+mat le
        # loi) -> khong co troi, tra mask rong thang, KHONG loang bay ba.
        mask_full = np.zeros((H, W), dtype=np.float32)
        return mask_full, 0.0

    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB).astype(np.float32)
    luma = lab[:, :, 0].astype(np.float32)

    region = _grow_region(lab, seed)
    region = _apply_luma_floor(region, luma)
    region = _enforce_column_contiguity(region)

    # morphology NHO — don nhieu 1px, KHONG dong lo hong cay/anten
    r_u8 = (region.astype(np.uint8) * 255)
    kernel = np.ones((MORPH_KSIZE, MORPH_KSIZE), np.uint8)
    r_u8 = cv2.morphologyEx(r_u8, cv2.MORPH_OPEN, kernel)
    r_u8 = cv2.morphologyEx(r_u8, cv2.MORPH_CLOSE, kernel)

    mask_proxy = cv2.GaussianBlur(r_u8.astype(np.float32), (0, 0), sigmaX=FEATHER_SIGMA)
    mask_proxy = np.clip(mask_proxy / 255.0, 0.0, 1.0).astype(np.float32)

    sky_fraction = float(mask_proxy.mean())

    if (ph, pw) != (H, W):
        guide = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        mask_full = guided_upsample(mask_proxy, guide)
        mask_full = np.clip(np.asarray(mask_full, dtype=np.float32), 0.0, 1.0)
    else:
        mask_full = mask_proxy

    return mask_full.astype(np.float32), sky_fraction


if __name__ == "__main__":
    print("Sky mask module loaded. ximgproc:", hasattr(cv2, "ximgproc"))
