"""
Con "SCENE CLASSIFY" (Task 23, deterministic, KHONG model).

HOP DONG:
    classify(img_bgr_float01_or_u8) -> dict {
        "scene": "interior" | "exterior_ground" | "aerial" | "unknown",
        "confidence": float 0-1,
        "signals": {...}   # so lieu tho, dung de debug/tuning, KHONG dam bao on dinh format
    }

Muc dich: dinh tuyen anh cho Task 21 (delivery pipeline) truoc khi chon chuoi
operator (VD: khong straighten anh aerial, khong ap grade noi that len anh
aerial nhu su co da xay ra). Chay tren proxy ~768px canh dai, KHONG bao gio
crash — moi loi -> "unknown" + confidence thap thay vi doan bua.

Chien luoc tin hieu (deterministic, xem tasks/23-scene-classify.md):
  1. TROI: dung lai con sky_replace/sky_mask.detect_sky() (da kiem dinh, tranh
     lam lai logic loang mau) -> sky_fraction + ty le troi cham bien tren.
  2. DUONG CHAN TROI (horizon): hang co gradient-ngang trung binh manh bat
     thuong trong nua tren cua khung -> ranh gioi troi/dat 1 duong ro.
  3. AERIAL/NADIR: KHONG co troi + dai tren cung day CANH (mai nha, via he...)
     thay vi min (tran nha) hay muot (troi) + do "ban" (edge density) gan deu
     giua dai tren va phan con lai (ca khung deu la "mat dat", khong co tran/troi).
  4. NOI THAT: KHONG co troi + dai tren cung PHANG/MIN (tran nha, canh thap)
     + co nhieu duong gan-doc trai deu theo chieu ngang (tuong, khung cua,
     canh cua so) — dac trung kien truc noi that.
"""

import os
import sys

import cv2
import numpy as np

cv2.setNumThreads(2)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from ai_engine.specialists.sky_replace.sky_mask import detect_sky  # noqa: E402

PROXY_DIM = 768  # canh dai cua ban proxy chay heuristics

# --- (1) troi ---
SKY_FRAC_SOME = 0.03    # tren nguong nay: co troi dang ke trong khung
SKY_FRAC_STRONG = 0.20  # tren nguong nay: troi chiem phan lon -> rat chac chan ngoai troi
SKY_TOUCH_TOP_STRONG = 0.25  # ty le hang bien tren la troi -> rat chac troi cham bien

# --- dai "tran nha / troi" xet o tren cung khung hinh ---
TOP_STRIP_FRAC = 0.14

# --- (3)/(4) mat do canh (Canny), do min ---
CANNY_LO, CANNY_HI = 60, 160
CEIL_EDGE_LOW = 0.045     # duoi nguong nay: dai tren PHANG (tran nha ung vien)
AERIAL_EDGE_HIGH = 0.09   # tren nguong nay: dai tren "ban" (khong phai tran/troi)
UNIFORM_RATIO_BAND = 0.35  # |top_edge/bottom_edge - 1| duoi muc nay -> ca khung dong deu (goi y aerial)

# --- (4) duong gan-doc (noi that: tuong/khung cua/canh cua so) ---
VLINE_ANGLE_MAX_DEG = 12.0
VLINE_MIN_LEN_FRAC = 0.12
VLINE_MIN_COUNT = 4
VLINE_MIN_XSPREAD_FRAC = 0.15  # do trai rong toi thieu (std vi tri x / chieu rong)

# --- (2) duong chan troi ---
HORIZON_ROW_LO_FRAC = 0.06
HORIZON_ROW_HI_FRAC = 0.62
HORIZON_STRENGTH_MIN = 3.2  # ty le gradient-hang dinh / median cac hang khac

MIN_CONFIDENT_SCORE = 0.28  # diem scene thang < nguong nay -> "unknown"


def _to_proxy_u8(img):
    """img float [0,1] hoac uint8 [0,255], BGR HxWx3 -> proxy BGR uint8, canh dai <= PROXY_DIM."""
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Anh phai la HxWx3 BGR, nhan shape={arr.shape}")

    if np.issubdtype(arr.dtype, np.floating):
        u8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    else:
        u8 = np.clip(arr, 0, 255).astype(np.uint8)

    h, w = u8.shape[:2]
    scale = PROXY_DIM / max(h, w)
    if scale < 1.0:
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        u8 = cv2.resize(u8, (nw, nh), interpolation=cv2.INTER_AREA)
    return u8


def _edge_map(gray_u8):
    edges = cv2.Canny(gray_u8, CANNY_LO, CANNY_HI)
    return (edges > 0)


def _detect_vertical_lines(gray_u8):
    """Doan gan-doc (LSD, fallback Hough). Tra ve list (x_mid, length)."""
    h, w = gray_u8.shape[:2]
    min_len = max(20.0, VLINE_MIN_LEN_FRAC * h)

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

    lines = []
    for (x1, y1, x2, y2) in segments:
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if length < min_len or dy < 1e-3:
            continue
        angle = float(np.degrees(np.arctan2(abs(dx), dy)))
        if angle > VLINE_ANGLE_MAX_DEG:
            continue
        lines.append(((x1 + x2) / 2.0, length))

    return lines


def _horizon_signal(gray_u8):
    """Hang co gradient-ngang trung binh manh bat thuong trong nua tren khung."""
    h, w = gray_u8.shape[:2]
    gray_f = gray_u8.astype(np.float32)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    row_grad = np.mean(np.abs(gy), axis=1)  # (h,)

    lo = max(1, int(round(h * HORIZON_ROW_LO_FRAC)))
    hi = min(h - 1, int(round(h * HORIZON_ROW_HI_FRAC)))
    if hi <= lo:
        return False, 0.0, -1

    band = row_grad[lo:hi]
    peak_idx_local = int(np.argmax(band))
    peak_val = float(band[peak_idx_local])
    peak_row = lo + peak_idx_local

    others = np.concatenate([row_grad[:lo], row_grad[hi:]]) if (lo > 0 or hi < h) else row_grad
    med_other = float(np.median(others)) if others.size > 0 else 1e-6
    med_other = max(med_other, 1e-6)

    strength_ratio = peak_val / med_other
    present = strength_ratio >= HORIZON_STRENGTH_MIN
    return bool(present), float(strength_ratio), int(peak_row)


def _score_scene(signals):
    sky_frac = signals["sky_fraction"]
    sky_touch = signals["sky_touch_top"]
    top_edge = signals["top_edge_density"]
    bot_edge = signals["bottom_edge_density"]
    top_std = signals["top_luma_std"]
    horizon_present = signals["horizon_present"]
    horizon_ratio = signals["horizon_strength_ratio"]
    vline_count = signals["vline_count"]
    vline_xspread = signals["vline_xspread_frac"]

    has_no_sky = sky_frac < SKY_FRAC_SOME and sky_touch < 0.08

    edge_ratio = top_edge / max(bot_edge, 1e-4)
    uniform = abs(edge_ratio - 1.0) <= UNIFORM_RATIO_BAND

    good_vlines = vline_count >= VLINE_MIN_COUNT and vline_xspread >= VLINE_MIN_XSPREAD_FRAC

    scores = {"interior": 0.0, "exterior_ground": 0.0, "aerial": 0.0}

    # --- exterior_ground: co troi/duong chan troi ro ---
    if sky_frac >= SKY_FRAC_SOME or sky_touch >= 0.08:
        scores["exterior_ground"] += min(sky_frac / 0.22, 1.0) * 0.55
        scores["exterior_ground"] += min(sky_touch / SKY_TOUCH_TOP_STRONG, 1.0) * 0.25
        if horizon_present:
            scores["exterior_ground"] += 0.20
        if sky_frac >= SKY_FRAC_STRONG:
            scores["exterior_ground"] += 0.10
    elif horizon_present and horizon_ratio >= HORIZON_STRENGTH_MIN:
        # khong bat duoc troi qua flood-grow (VD troi xam/hazy) nhung co
        # duong-chan-troi manh trong nua tren -> van nghieng ve ngoai troi.
        scores["exterior_ground"] += 0.30

    # --- aerial: KHONG troi + dai tren "ban" + toan khung dong deu ---
    if has_no_sky:
        aerial_base = 0.0
        if top_edge >= AERIAL_EDGE_HIGH:
            aerial_base += min(top_edge / 0.20, 1.0) * 0.55
        if uniform:
            aerial_base += 0.30
        if not good_vlines:
            aerial_base += 0.15
        scores["aerial"] = aerial_base

    # --- interior: KHONG troi + tran phang + duong gan-doc trai deu ---
    if has_no_sky:
        interior_base = 0.0
        if top_edge <= CEIL_EDGE_LOW:
            interior_base += (1.0 - top_edge / max(CEIL_EDGE_LOW, 1e-6)) * 0.35
        if top_std <= 22.0:
            interior_base += max(0.0, (22.0 - top_std) / 22.0) * 0.15
        if good_vlines:
            interior_base += min(vline_count / 8.0, 1.0) * 0.45
        scores["interior"] = interior_base

    return scores


def classify(img):
    """img: np.ndarray HxWx3 BGR, float32 [0,1] hoac uint8 [0,255]. KHONG bao gio raise."""
    try:
        proxy_u8 = _to_proxy_u8(img)
        h, w = proxy_u8.shape[:2]
        gray = cv2.cvtColor(proxy_u8, cv2.COLOR_BGR2GRAY)

        proxy_f01 = proxy_u8.astype(np.float32) / 255.0
        sky_mask, sky_fraction = detect_sky(proxy_f01)
        top_rows = max(1, int(round(h * 0.03)))
        sky_touch_top = float(sky_mask[:top_rows, :].mean()) if sky_mask.size else 0.0

        top_strip_h = max(1, int(round(h * TOP_STRIP_FRAC)))
        edges = _edge_map(gray)
        top_edge_density = float(edges[:top_strip_h, :].mean())
        bottom_edge_density = float(edges[top_strip_h:, :].mean()) if h > top_strip_h else top_edge_density
        top_luma_std = float(gray[:top_strip_h, :].astype(np.float32).std())

        horizon_present, horizon_ratio, horizon_row = _horizon_signal(gray)

        vlines = _detect_vertical_lines(gray)
        vline_count = len(vlines)
        if vline_count > 0:
            xs = np.array([v[0] for v in vlines], dtype=np.float32)
            vline_xspread_frac = float(xs.std() / max(w, 1))
        else:
            vline_xspread_frac = 0.0

        signals = {
            "sky_fraction": sky_fraction,
            "sky_touch_top": sky_touch_top,
            "top_edge_density": top_edge_density,
            "bottom_edge_density": bottom_edge_density,
            "top_luma_std": top_luma_std,
            "horizon_present": horizon_present,
            "horizon_strength_ratio": horizon_ratio,
            "horizon_row_frac": (horizon_row / h) if horizon_row >= 0 else -1.0,
            "vline_count": vline_count,
            "vline_xspread_frac": vline_xspread_frac,
            "proxy_size": [w, h],
        }

        scores = _score_scene(signals)
        signals["scores"] = scores

        best_scene = max(scores, key=scores.get)
        best_score = scores[best_scene]
        sorted_scores = sorted(scores.values(), reverse=True)
        margin = sorted_scores[0] - (sorted_scores[1] if len(sorted_scores) > 1 else 0.0)

        if best_score < MIN_CONFIDENT_SCORE:
            return {"scene": "unknown", "confidence": round(float(best_score) * 0.5, 4), "signals": signals}

        confidence = float(np.clip(0.5 * best_score + 0.5 * margin, 0.0, 1.0))
        return {"scene": best_scene, "confidence": round(confidence, 4), "signals": signals}

    except Exception as e:
        return {"scene": "unknown", "confidence": 0.0, "signals": {"error": f"{type(e).__name__}: {e}"}}


if __name__ == "__main__":
    print("Scene classify module loaded.")
