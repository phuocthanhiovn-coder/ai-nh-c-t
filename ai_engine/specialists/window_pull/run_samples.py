"""
Chay thu con window_pull tren anh noi that co cua so.

Quet data/pairs/before/ (doc thoi), xep hang theo win_fraction tren ban thu nho,
lay top 6. Voi moi anh: luu outputs/window_samples/<ten>.jpg =
[goc | mask truc quan | da pull] ghep ngang 1800px + 1 crop 100% vung cua so
(goc|pull canh nhau) ghep duoi. In win_fraction + gated per anh.

Test gate: anh co win_fraction thap nhat phai qua gate KHONG DOI (bit-identical).
"""

import os
import sys

import cv2
import numpy as np

cv2.setNumThreads(2)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from ai_engine.specialists.window_pull.pull import (  # noqa: E402
    FEATHER_PX,
    WIN_FRACTION_GATE_HI,
    WIN_FRACTION_GATE_LO,
    apply,
)
from ai_engine.specialists.window_pull.window_mask import detect_windows  # noqa: E402

SRC_DIR = os.path.join("data", "pairs", "before")
OUT_DIR = os.path.join("outputs", "window_samples")
N_SAMPLES = 6
SCAN_MAX_DIM = 480
PANEL_W = 1800          # tong be rong panel 3 anh
CROP_W, CROP_H = 900, 500  # crop 100% (2 crop canh nhau = 1800 = PANEL_W)
IMG_EXTS = (".jpg", ".jpeg", ".png")


def _list_images(d):
    if not os.path.isdir(d):
        return []
    return [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.lower().endswith(IMG_EXTS)]


def _resize_max_dim(img, max_dim):
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1.0:
        return img.copy()
    return cv2.resize(
        img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def scan_win_fraction(path):
    img_u8 = cv2.imread(path)
    if img_u8 is None:
        return None
    small = _resize_max_dim(img_u8, SCAN_MAX_DIM).astype(np.float32) / 255.0
    _mask, frac = detect_windows(small)
    return frac


def _dedupe_key(path):
    """db02_20260703-DSC1161.jpg va 20260703-DSC1161.jpg la CUNG anh — khu trung."""
    import re

    base = os.path.splitext(os.path.basename(path))[0].lower()
    return re.sub(r"^db\d+_?", "", base).lstrip("_")


def rank_images():
    ranked = []
    seen = set()
    for path in _list_images(SRC_DIR):
        frac = scan_win_fraction(path)
        if frac is None:
            print(f"[!] Khong doc duoc: {path}")
            continue
        key = _dedupe_key(path)
        if key in seen:
            continue
        seen.add(key)
        ranked.append((frac, path))
    ranked.sort(key=lambda t: -t[0])
    return ranked


def mask_visual(mask):
    """Mask [0,1] -> BGR uint8: den=0, cam=1 (noi bat tren noi that)."""
    m_u8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    vis = np.zeros((*m_u8.shape, 3), dtype=np.uint8)
    vis[:, :, 1] = (m_u8 * 0.6).astype(np.uint8)  # G
    vis[:, :, 2] = m_u8                            # R -> cam
    return vis


def make_panel(imgs_u8, total_w=PANEL_W):
    n = len(imgs_u8)
    h, w = imgs_u8[0].shape[:2]
    scale = (total_w / float(n)) / w
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = [cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA) for im in imgs_u8]
    return np.hstack(resized)


def crop_100(img_u8, cy, cx, ch=CROP_H, cw=CROP_W):
    """Crop 100% (khong resize) quanh (cy,cx), tu dong ep vao trong bien anh."""
    H, W = img_u8.shape[:2]
    ch, cw = min(ch, H), min(cw, W)
    y0 = int(np.clip(cy - ch // 2, 0, H - ch))
    x0 = int(np.clip(cx - cw // 2, 0, W - cw))
    return img_u8[y0 : y0 + ch, x0 : x0 + cw]


def window_center(mask):
    """Tam blob cua so LON NHAT tren mask full-res (de crop 100%)."""
    m_u8 = (mask > 0.5).astype(np.uint8)
    num, labels, stats, cents = cv2.connectedComponentsWithStats(m_u8, connectivity=8)
    if num <= 1:
        H, W = mask.shape[:2]
        return H // 2, W // 2
    lb = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cx, cy = cents[lb]
    return int(round(cy)), int(round(cx))


def pad_to_width(img_u8, width):
    h, w = img_u8.shape[:2]
    if w >= width:
        return img_u8[:, :width]
    pad = np.zeros((h, width - w, 3), dtype=np.uint8)
    return np.hstack([img_u8, pad])


def process_one(path, frac_hint):
    name = os.path.basename(path)
    img_u8 = cv2.imread(path)
    if img_u8 is None:
        print(f"[!] Khong doc duoc: {path}")
        return None
    img = img_u8.astype(np.float32) / 255.0
    h, w = img.shape[:2]

    mask, win_fraction = detect_windows(img)
    gated = (
        win_fraction < WIN_FRACTION_GATE_LO or win_fraction > WIN_FRACTION_GATE_HI
    )

    result = apply(img, {"strength": 0.7, "saturation_boost": 0.25})
    assert result.shape == img.shape, f"Sai shape output cho {name}"
    result_u8 = np.clip(result * 255.0 + 0.5, 0, 255).astype(np.uint8)

    # do lech noi that: pixel ngoai mask feather phai == 0
    m = np.clip(mask, 0.0, 1.0)
    m_blur = cv2.GaussianBlur(m, (0, 0), sigmaX=FEATHER_PX)
    outside = m_blur <= 0.0
    max_diff_outside = (
        float(np.abs(result - img)[outside].max()) if outside.any() else -1.0
    )

    print(
        f"[{name}] size={w}x{h}  win_fraction={win_fraction:.4f} "
        f"(scan_hint={frac_hint:.4f})  gated={gated}  "
        f"max_diff_outside={max_diff_outside:.3e}"
    )

    mask_vis = mask_visual(mask)
    panel = make_panel([img_u8, mask_vis, result_u8])

    cy, cx = window_center(mask)
    crop_before = crop_100(img_u8, cy, cx)
    crop_after = crop_100(result_u8, cy, cx)
    crops = pad_to_width(np.hstack([crop_before, crop_after]), panel.shape[1])
    sep = np.full((4, panel.shape[1], 3), 255, dtype=np.uint8)
    sheet = np.vstack([panel, sep, crops])

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, os.path.splitext(name)[0] + ".jpg")
    cv2.imwrite(out_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  saved: {out_path}")
    return out_path, win_fraction, gated, max_diff_outside


def test_gate_unchanged(ranked):
    """(1) Anh tong hop KHONG cua so phai qua gate khong doi (bit-identical).
    (2) Anh that co win_fraction THAP NHAT: bao cao trung thuc co gate hay khong."""
    # (1) tong hop: gradient noi that toi, khong vung sang -> win_fraction ~ 0
    yy = np.linspace(0.15, 0.45, 600, dtype=np.float32).reshape(-1, 1, 1)
    synth = np.repeat(np.repeat(yy, 900, axis=1), 3, axis=2).astype(np.float32)
    _m, synth_frac = detect_windows(synth)
    out_s = apply(synth, {})
    synth_identical = bool(np.array_equal(out_s, synth))
    print(
        f"[test_gate/synthetic] win_fraction={synth_frac:.4f} "
        f"gated={synth_frac < WIN_FRACTION_GATE_LO} unchanged={synth_identical}"
    )
    assert synth_identical, "Anh khong cua so PHAI di qua gate khong doi!"

    # (2) anh that it cua so nhat
    if not ranked:
        print("[test_gate] Khong co anh that nao.")
        return
    frac_hint, path = ranked[-1]
    img_u8 = cv2.imread(path)
    img = img_u8.astype(np.float32) / 255.0
    _mask, win_fraction = detect_windows(img)
    out = apply(img, {})
    identical = bool(np.array_equal(out, img))
    gated = win_fraction < WIN_FRACTION_GATE_LO
    print(
        f"[test_gate/real-min] {os.path.basename(path)} win_fraction={win_fraction:.4f} "
        f"gate_lo={WIN_FRACTION_GATE_LO} gated={gated} unchanged={identical}"
    )
    return path, win_fraction, gated, identical


def main():
    ranked = rank_images()
    print(f"Da quet {len(ranked)} anh trong {SRC_DIR}.")
    print()

    print("=== Test gate: anh it cua so nhat phai khong doi ===")
    test_gate_unchanged(ranked)
    print()

    picks = ranked[:N_SAMPLES]
    print(f"Top {len(picks)} anh theo win_fraction (ban thu nho):")
    for frac, path in picks:
        print(f"  {path}  scan_win_fraction={frac:.4f}")
    print()

    for frac, path in picks:
        process_one(path, frac)


if __name__ == "__main__":
    main()
