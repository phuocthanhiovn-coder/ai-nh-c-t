"""
Con "QC SCORER" v0 (deterministic, KHONG model).

HOP DONG:
    score(img_bgr_float01: np.ndarray HxWx3 float32 [0,1]) -> dict
        {
          "blur_score", "exposure_score", "tilt_score",
          "color_cast_score", "noise_score", "overall": float 0-100,
          "flags": list[str], "needs_human": bool,
          "_debug": {...}   # so lieu tho, khong dung de tinh diem lai
        }

Tat ca do dac chi doc anh, KHONG sua/tra ve anh moi.
"""

import cv2
import numpy as np

cv2.setNumThreads(2)

SMALL_DIM = 1024  # chuan hoa do phan giai truoc khi do (blur/tilt/noise phu thuoc scale)

WEIGHTS = {
    "blur_score": 0.25,
    "exposure_score": 0.20,
    "tilt_score": 0.10,
    "color_cast_score": 0.15,
    "noise_score": 0.10,
    "washout_score": 0.20,
}

# --- washout (bet/chay/mat dai tuong phan/mat bao hoa) — chieu do BAT LOI ma 5 chieu cu bo sot ---
WASHOUT_BLOWN_THRESH = 0.92   # pixel luma > nguong nay coi la "gan trang/chay"
WASHOUT_DYNRANGE_MIN = 0.55   # p95-p5 duoi day = dai tuong phan bi ep bet
WASHOUT_SAT_LO = 0.12         # mean saturation duoi day + sang = co ve mo suong/veil
FLAG_THRESH = 50.0
NEEDS_HUMAN_OVERALL = 55.0
NEEDS_HUMAN_MIN_FLAGS = 2

# --- blur ---
BLUR_REF_VAR = 260.0  # hang so calib: var Laplacian ung voi score ~63 (1 - e^-1)

# --- exposure ---
# Anh BDS da chinh (AutoHDR) co xu huong sang/airy: median luma thuc te ~0.6-0.72,
# khong phai 0.42 "trung tinh". Calib tu du lieu pairs thuc (xem bao cao cuoi file task).
EXPOSURE_BURNT_THRESH = 0.98
EXPOSURE_SUNK_THRESH = 0.02
EXPOSURE_MEDIAN_LO = 0.45
EXPOSURE_MEDIAN_HI = 0.72

# --- tilt ---
TILT_MAX_ANGLE_DEG = 8.0
TILT_ANGLE_FILTER_DEG = 15.0
TILT_MIN_LINES = 3
TILT_MIN_LEN_FRAC = 0.06

# --- color cast ---
CAST_REF_DEV = 25.0  # do lech a/b (thang Lab 0-255, tam 128) ung voi score ~0

# --- noise ---
NOISE_BLOCK = 24       # kich thuoc block de tim vung "phang nhat" (tranh lan voi texture thuc)
NOISE_PCTL = 10        # percentile thap trong phan bo std cac block -> vung phang nhat
NOISE_REF_STD = 4.5    # std (thang 0-255) cua vung phang-nhat ung voi score ~e^-1 (~37)


def _to_small_u8(img):
    """Resize (giu ti le, canh dai = SMALL_DIM) + tra ve BGR uint8, dung cho cac chi so can chuan hoa do phan giai."""
    h, w = img.shape[:2]
    scale = SMALL_DIM / max(h, w)
    if scale < 1.0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        small = img.copy()
    small_u8 = np.clip(small * 255.0, 0, 255).astype(np.uint8)
    return small_u8


def _luma01(img):
    return 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]


def blur_score(small_u8):
    """Variance of Laplacian tren anh da chuan hoa res, so vung net nhat / vung giua (grid 3x3)."""
    gray = cv2.cvtColor(small_u8, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)

    h, w = gray.shape[:2]
    gh, gw = max(1, h // 3), max(1, w // 3)
    patch_vars = []
    for i in range(3):
        for j in range(3):
            y0, y1 = i * gh, (h if i == 2 else (i + 1) * gh)
            x0, x1 = j * gw, (w if j == 2 else (j + 1) * gw)
            patch = lap[y0:y1, x0:x1]
            patch_vars.append(float(patch.var()) if patch.size > 0 else 0.0)

    center_var = patch_vars[4]
    max_var = max(patch_vars)
    global_var = float(lap.var())

    # vung giua la chu the chinh: tin no la chinh, nhung neu co vung net ro rang
    # o noi khac ma vung giua yeu hon nhieu -> nghi ngo out-of-focus, kep diem xuong.
    ref_var = 0.7 * center_var + 0.3 * max_var

    score = 100.0 * (1.0 - np.exp(-ref_var / BLUR_REF_VAR))
    score = float(np.clip(score, 0.0, 100.0))
    return score, {
        "global_var": global_var,
        "center_var": center_var,
        "max_var": max_var,
    }


def exposure_score(img):
    """Histogram: % chay (>0.98), % chim (<0.02), median lech khoi [0.35, 0.55]."""
    lum = _luma01(img)
    burnt_pct = float((lum > EXPOSURE_BURNT_THRESH).mean())
    sunk_pct = float((lum < EXPOSURE_SUNK_THRESH).mean())
    med = float(np.median(lum))

    if med < EXPOSURE_MEDIAN_LO:
        med_dev = EXPOSURE_MEDIAN_LO - med
    elif med > EXPOSURE_MEDIAN_HI:
        med_dev = med - EXPOSURE_MEDIAN_HI
    else:
        med_dev = 0.0

    score = 100.0
    score -= min(burnt_pct * 400.0, 40.0)
    score -= min(sunk_pct * 400.0, 40.0)
    score -= min(med_dev * 200.0, 40.0)
    score = float(np.clip(score, 0.0, 100.0))
    return score, {"burnt_pct": burnt_pct, "sunk_pct": sunk_pct, "median": med}


def _detect_vertical_lines(gray_u8):
    """Doan thang gan-doc (LSD, fallback Hough). Tra ve list angle_deg (lech khoi phuong doc)."""
    h, w = gray_u8.shape[:2]
    min_len = max(20.0, TILT_MIN_LEN_FRAC * max(h, w))

    segments = []
    try:
        lsd = cv2.createLineSegmentDetector(0)
        detected = lsd.detect(gray_u8)[0]
    except Exception:
        detected = None

    if detected is not None and len(detected) > 0:
        for seg in detected:
            x1, y1, x2, y2 = np.asarray(seg).reshape(-1)[:4]
            segments.append((float(x1), float(y1), float(x2), float(y2)))
    else:
        edges = cv2.Canny(gray_u8, 50, 150)
        hough = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                                 minLineLength=min_len, maxLineGap=8)
        if hough is not None:
            for seg in hough:
                x1, y1, x2, y2 = seg[0]
                segments.append((float(x1), float(y1), float(x2), float(y2)))

    angles = []
    for (x1, y1, x2, y2) in segments:
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if length < min_len or dy < 1e-3:
            continue
        angle = float(np.degrees(np.arctan2(dx, dy)))
        if abs(angle) > TILT_ANGLE_FILTER_DEG:
            continue
        angles.append(angle)

    return angles


def tilt_score(small_u8):
    """Goc nghieng trung vi cua cac duong gan-doc; khong du duong tin cay -> trung tinh (khong phat hien loi)."""
    gray = cv2.cvtColor(small_u8, cv2.COLOR_BGR2GRAY)
    angles = _detect_vertical_lines(gray)

    if len(angles) < TILT_MIN_LINES:
        return 100.0, {"angle_deg": 0.0, "num_lines": len(angles)}

    angle = float(np.median(angles))
    score = 100.0 * (1.0 - min(abs(angle) / TILT_MAX_ANGLE_DEG, 1.0))
    score = float(np.clip(score, 0.0, 100.0))
    return score, {"angle_deg": angle, "num_lines": len(angles)}


def color_cast_score(small_u8):
    """Do lech trung binh kenh a/b (Lab) o vung highlight trung tinh (sang, it bao hoa mau)."""
    lab = cv2.cvtColor(small_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    chroma = np.hypot(A - 128.0, B - 128.0)
    l_hi = np.percentile(L, 70)
    c_lo = np.percentile(chroma, 50)
    mask = (L >= l_hi) & (chroma <= c_lo)
    if mask.sum() < 200:
        mask = L >= l_hi
    if mask.sum() < 200:
        mask = np.ones_like(L, dtype=bool)

    mean_a = float(A[mask].mean()) - 128.0
    mean_b = float(B[mask].mean()) - 128.0
    dev = float(np.hypot(mean_a, mean_b))

    score = 100.0 * (1.0 - min(dev / CAST_REF_DEV, 1.0))
    score = float(np.clip(score, 0.0, 100.0))
    return score, {"dev_ab": dev, "mean_a": mean_a, "mean_b": mean_b}


def noise_score(small_u8):
    """
    Std cua vung 'phang nhat' anh, do theo block NOISE_BLOCK x NOISE_BLOCK.
    Dung percentile thap (NOISE_PCTL) tren std cac block thay vi mask theo gradient toan anh:
    anh noi that thuc co texture (go, gach, tham) lan khap noi nen threshold gradient toan cuc
    khong tach duoc vung phang; chia block roi lay block-std thap nhat moi ra dung nhieu nen.
    Dung scipy-free: chia luoi deu, khong overlap.
    """
    gray = cv2.cvtColor(small_u8, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape[:2]
    block = NOISE_BLOCK

    stds = []
    for y in range(0, max(1, h - block + 1), block):
        for x in range(0, max(1, w - block + 1), block):
            patch = gray[y:y + block, x:x + block]
            if patch.size >= block * block // 2:
                stds.append(float(patch.std()))

    if len(stds) < 4:
        std = float(gray.std())
    else:
        std = float(np.percentile(np.array(stds), NOISE_PCTL))

    score = 100.0 * np.exp(-std / NOISE_REF_STD)
    score = float(np.clip(score, 0.0, 100.0))
    return score, {"flat_block_std": std, "num_blocks": len(stds)}


def washout_score(img):
    """
    Bat anh 'bet/chay/mat dai tuong phan/mat bao hoa' — kieu output hong khi model generative
    hoac auto-enhance chua chin lam trang bech ca anh. 5 chieu cu KHONG bat duoc ca nay
    (median bi day len dung dai 'sang-airy tot', khong bi phat).
    """
    lum = _luma01(img)
    blown_frac = float((lum > WASHOUT_BLOWN_THRESH).mean())
    p5, p95 = np.percentile(lum, [5, 95])
    dyn_range = float(p95 - p5)
    med = float(np.median(lum))

    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    sat = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32) / 255.0
    mean_sat = float(sat.mean())

    score = 100.0
    score -= min(blown_frac * 250.0, 60.0)                    # nhieu vung gan-trang
    if dyn_range < WASHOUT_DYNRANGE_MIN:
        score -= min((WASHOUT_DYNRANGE_MIN - dyn_range) * 150.0, 40.0)   # dai tuong phan ep bet
    if mean_sat < WASHOUT_SAT_LO and med > 0.55:
        score -= 30.0                                         # mo suong: nhat mau + sang
    score = float(np.clip(score, 0.0, 100.0))
    return score, {"blown_frac": blown_frac, "dyn_range": dyn_range, "mean_sat": mean_sat, "median": med}


def score(img):
    """
    score(img_bgr_float01) -> dict. img: np.ndarray float32/float64 [0,1] HxWx3 BGR.
    """
    assert img.ndim == 3 and img.shape[2] == 3, "Anh phai la HxWx3 BGR"

    small_u8 = _to_small_u8(img)

    b_score, b_dbg = blur_score(small_u8)
    e_score, e_dbg = exposure_score(img)
    t_score, t_dbg = tilt_score(small_u8)
    c_score, c_dbg = color_cast_score(small_u8)
    n_score, n_dbg = noise_score(small_u8)
    w_score, w_dbg = washout_score(img)

    scores = {
        "blur_score": b_score,
        "exposure_score": e_score,
        "tilt_score": t_score,
        "color_cast_score": c_score,
        "noise_score": n_score,
        "washout_score": w_score,
    }

    overall = sum(scores[k] * w for k, w in WEIGHTS.items())
    overall = float(np.clip(overall, 0.0, 100.0))

    flag_names = {
        "blur_score": "blurry",
        "exposure_score": "overexposed",
        "tilt_score": "tilted",
        "color_cast_score": "color_cast",
        "noise_score": "noisy",
        "washout_score": "washed_out",
    }
    flags = [flag_names[k] for k, v in scores.items() if v < FLAG_THRESH]

    needs_human = (overall < NEEDS_HUMAN_OVERALL) or (len(flags) >= NEEDS_HUMAN_MIN_FLAGS)

    result = dict(scores)
    result["overall"] = overall
    result["flags"] = flags
    result["needs_human"] = bool(needs_human)
    result["_debug"] = {
        "blur": b_dbg,
        "exposure": e_dbg,
        "tilt": t_dbg,
        "color_cast": c_dbg,
        "noise": n_dbg,
        "washout": w_dbg,
    }
    return result


if __name__ == "__main__":
    print("QC scorer module loaded.")
