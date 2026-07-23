"""
Con "PHAT HIEN CUA SO" (window detection, CV thuan, deterministic) — proxy -> soft mask.

Chay tren ban proxy ~768px:
  (a) vung SANG hon phong ro ret: luma >= max(p90 toan anh, median noi that + margin),
  (b) blob compact, dien tich 0.5%-35% khung, KHONG cham ca 4 bien
      (ca mang tuong chay trang khong phai cua so),
  (c) mat do canh (edge density) tren VIEN blob = phieu bau ho tro (khung/mullion
      cua so co canh thang manh) — chi dieu chinh do dam alpha, KHONG phai luat cung.
Gop cac o kinh gan nhau (window grid) bang morphology close TRUOC khi tach blob.
Upsample soft mask len full-res bang guided_upsample (bien bam khung/mullion).

Tra ve (mask float32 [0,1] HxW full-res, win_fraction float).
win_fraction = mean cua soft mask tren proxy — dung de gate trong pull.apply().
"""

import os
import sys

import cv2
import numpy as np

cv2.setNumThreads(2)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from ai_engine.core.quality import guided_upsample  # noqa: E402
from ai_engine.specialists.sky_replace.sky_mask import detect_sky  # noqa: E402

PROXY_DIM = 768  # canh dai cua ban proxy

# --- (a) nguong sang (luma [0,1] tren proxy) ---
LUMA_PCTL = 90.0          # cua so phai sang hon p90 toan anh
MEDIAN_MARGIN = 0.18      # ... VA sang hon median noi that it nhat margin nay
LUMA_ABS_MIN = 0.55       # san tuyet doi: khong bat vung xam trung tinh o anh toi

# --- (b) blob ---
AREA_MIN_FRAC = 0.005     # 0.5% khung
AREA_MAX_FRAC = 0.35      # 35% khung
EXTENT_MIN = 0.20         # area / bbox_area — cua so gan chu nhat, blob loang thi loai
# Cua so that CHAY SANG: phai co phan pixel GAN CLIP (anh sang ngoai troi).
# Vat trang trong nha (may giat, tuong son, cua trang) sang nhung KHONG clip
# -> loai (false positive that da thay: may giat _ML_1500 bi nham la cua so).
CLIP_LUMA = 0.93
CLIP_FRAC_MIN = 0.08      # >= 8% pixel cua blob phai gan clip
# Nhiet do mau: ngoai canh (daylight) LANH hon anh sang noi that -> blob cua so
# phai LANH hon (hoac gan bang) noi that theo median(B-R). Vat trang noi that
# (may giat: -0.027) va san gach ngoai troi nang (-0.039/-0.071) AM hon ro ret;
# cua so that do duoc: +0.004 (DSC1197), -0.008 (_ML_1372, gach do am ap).
# ==> nguong -0.015 nam giua 2 nhom. MONG MANH (cach nhom that 0.007) — ghi ro
# trong report; ve dai han day la viec cua model segment cua so (nhom A roster).
COOL_DIFF_MIN = -0.015

# --- (c) edge density tren vien blob (phieu bau mem) ---
CANNY_LO, CANNY_HI = 60, 160
EDGE_RING_PX = 3          # be day vien de do edge density
EDGE_DENSITY_REF = 0.25   # density >= muc nay -> phieu bau max
ALPHA_BASE = 0.7          # blob dat luat cung (a)+(b) duoc it nhat alpha nay
ALPHA_EDGE_BONUS = 0.3    # + toi da chung nay theo edge vote
# San phieu bau: blob vien MEM (may troi, loang sang) khong co canh khung ->
# loai, TRU KHI blob rat chu-nhat (extent cao — cua so khong khung van giu).
# (Ghi chu deviation: spec noi edge vote "khong phai luat cung"; san thap nay
#  la muc toi thieu de chan MAY TROI o anh ngoai that bi pull thanh xam ban —
#  da kiem bang mat, khong co san nay thi fail acceptance "NOT gray/dirty".)
EDGE_VOTE_MIN = 0.15
EXTENT_RECT_ESCAPE = 0.80

# --- GATE ANH NGOAI THAT (troi lo thien): window pull la op NOI THAT. Anh ban
# cong/san vuon co MAY sang bi nham thanh "cua so" -> pull lam ban troi (da thay
# bang mat o _ML_1542; veto tung blob khong du vi detect_sky bo sot may & troi
# nhat). Dau hieu re va chac: vung (sang | troi detect_sky) NOI VOI BIEN TREN
# khung phu >= 35% hang dau = troi lo thien -> coi ca anh la ngoai that, tra
# mask RONG (se bi gate trong pull). Troi nhin QUA cua so bi tran/khung chan
# truoc bien tren nen khong dinh. Han che: noi that co cua so chiem >35% be
# rong SAT mep tren khung se bi bo qua oan (khong gap trong 51 anh hien co).
SKY_UNION_THRESH = 0.25   # nguong mem cua detect_sky khi union
TOP_OPEN_COVER = 0.35     # vung noi bien tren phu >= 35% hang dau -> ngoai that
OPEN_CLOSE_KSIZE = 5      # bac cau khe may<->troi xanh truoc connected components

# --- morphology ---
CLOSE_KSIZE = 9           # gop cac o kinh cua window grid (mullion mong tren proxy)
OPEN_KSIZE = 3            # nhat rac
FEATHER_SIGMA = 1.5       # lam mem mask tren proxy truoc khi upsample


def _to_u8(img):
    return np.clip(np.asarray(img, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)


def _proxy(img_u8):
    h, w = img_u8.shape[:2]
    scale = PROXY_DIM / max(h, w)
    if scale >= 1.0:
        return img_u8.copy()
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(img_u8, (nw, nh), interpolation=cv2.INTER_AREA)


def _edge_vote(comp_bool, canny_u8):
    """Edge density tren vien blob -> phieu bau [0,1]. Vien = dilate - erode."""
    comp_u8 = comp_bool.astype(np.uint8)
    kernel = np.ones((2 * EDGE_RING_PX + 1, 2 * EDGE_RING_PX + 1), np.uint8)
    ring = cv2.dilate(comp_u8, kernel) - cv2.erode(comp_u8, kernel)
    ring_bool = ring > 0
    n = int(ring_bool.sum())
    if n == 0:
        return 0.0
    density = float((canny_u8[ring_bool] > 0).mean())
    return float(np.clip(density / EDGE_DENSITY_REF, 0.0, 1.0))


def detect_windows(img):
    """img float32 [0,1] BGR HxWx3 -> (mask float32 [0,1] HxW full-res, win_fraction).

    win_fraction = mean cua soft mask tren proxy — ty le dien tich cua so uoc luong.
    """
    img = np.asarray(img, dtype=np.float32)
    H, W = img.shape[:2]
    img_u8 = _to_u8(img)
    proxy = _proxy(img_u8)
    ph, pw = proxy.shape[:2]
    proxy_area = float(ph * pw)

    gray_u8 = cv2.cvtColor(proxy, cv2.COLOR_BGR2GRAY)
    luma = gray_u8.astype(np.float32) / 255.0

    # --- (a) nguong sang thich nghi ---
    p90 = float(np.percentile(luma, LUMA_PCTL))
    median = float(np.median(luma))
    thresh = max(p90, median + MEDIAN_MARGIN, LUMA_ABS_MIN)
    bright = (luma >= thresh).astype(np.uint8)

    # --- gop o kinh (window grid) + nhat rac ---
    close_k = np.ones((CLOSE_KSIZE, CLOSE_KSIZE), np.uint8)
    open_k = np.ones((OPEN_KSIZE, OPEN_KSIZE), np.uint8)
    merged = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, close_k)
    merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, open_k)

    canny = cv2.Canny(gray_u8, CANNY_LO, CANNY_HI)
    proxy_f = proxy.astype(np.float32) / 255.0
    bmr = proxy_f[:, :, 0] - proxy_f[:, :, 2]  # B - R (coolness)
    interior_bool = merged == 0
    interior_cool = float(np.median(bmr[interior_bool])) if interior_bool.any() else 0.0

    # gate ngoai that: (sang | troi xanh) noi bien TREN tren dai rong -> mask rong
    sky_m, _sky_frac = detect_sky(proxy.astype(np.float32) / 255.0)
    open_u8 = ((merged > 0) | (sky_m > SKY_UNION_THRESH)).astype(np.uint8)
    open_u8 = cv2.morphologyEx(
        open_u8, cv2.MORPH_CLOSE, np.ones((OPEN_CLOSE_KSIZE,) * 2, np.uint8)
    )
    _n2, lab2 = cv2.connectedComponents(open_u8, connectivity=8)
    for lb2 in np.unique(lab2[0, :]):
        if lb2 == 0:
            continue
        if float((lab2 == lb2)[0, :].mean()) >= TOP_OPEN_COVER:
            return np.zeros((H, W), dtype=np.float32), 0.0  # anh ngoai that

    # --- (b)+(c) loc tung blob ---
    num, labels, stats, _cent = cv2.connectedComponentsWithStats(merged, connectivity=8)
    mask_proxy = np.zeros((ph, pw), dtype=np.float32)
    for lb in range(1, num):
        x, y, bw, bh, area = stats[lb]
        frac = area / proxy_area
        if frac < AREA_MIN_FRAC or frac > AREA_MAX_FRAC:
            continue
        extent = area / float(max(bw * bh, 1))
        if extent < EXTENT_MIN:
            continue
        touches = (x == 0) + (y == 0) + (x + bw >= pw) + (y + bh >= ph)
        if touches >= 4:  # cham ca 4 bien = ca khung chay trang, khong phai cua so
            continue
        comp = labels == lb
        clip_frac = float((luma[comp] >= CLIP_LUMA).mean())
        if clip_frac < CLIP_FRAC_MIN:
            continue  # sang nhung khong clip = vat trang noi that, khong phai cua so chay
        cool_diff = float(np.median(bmr[comp])) - interior_cool
        if cool_diff < COOL_DIFF_MIN:
            continue  # AM hon noi that = vat trang / san nang, khong phai ngoai canh daylight
        vote = _edge_vote(comp, canny)
        if vote < EDGE_VOTE_MIN and extent < EXTENT_RECT_ESCAPE:
            continue  # vien mem + khong chu nhat = may troi / loang sang, khong phai cua so
        alpha = ALPHA_BASE + ALPHA_EDGE_BONUS * vote
        # Fill CONVEX HULL cua blob: phu TOAN BO o cua so (ke ca phan toi hon
        # nguong ben trong) de pull dong deu, khong loang lo. Tone curve trong
        # pull.py chi dung den pixel sang hon pivot nen phan noi that lot vao
        # hull van an toan.
        comp_u8 = comp.astype(np.uint8)
        contours, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull_canvas = np.zeros((ph, pw), dtype=np.uint8)
        for cnt in contours:
            cv2.fillConvexPoly(hull_canvas, cv2.convexHull(cnt), 1)
        mask_proxy = np.maximum(mask_proxy, hull_canvas.astype(np.float32) * alpha)

    # lam mem tren proxy roi upsample bam canh
    if FEATHER_SIGMA > 0:
        mask_proxy = cv2.GaussianBlur(mask_proxy, (0, 0), sigmaX=FEATHER_SIGMA)
    mask_proxy = np.clip(mask_proxy, 0.0, 1.0).astype(np.float32)

    win_fraction = float(mask_proxy.mean())

    if (ph, pw) != (H, W):
        guide = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        mask_full = guided_upsample(mask_proxy, guide)
        mask_full = np.clip(np.asarray(mask_full, dtype=np.float32), 0.0, 1.0)
    else:
        mask_full = mask_proxy

    return mask_full.astype(np.float32), win_fraction


if __name__ == "__main__":
    print("Window mask module loaded. ximgproc:", hasattr(cv2, "ximgproc"))
