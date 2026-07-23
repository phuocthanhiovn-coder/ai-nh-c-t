"""
Chay thu con white_balance tren anh sample that.
Chon 3 anh tu data/pairs/before + 3 anh tu data/review/before co lech mau ro nhat
(xep hang theo do lech gray-world R/B), luu panel [goc | WB | WB+exposure] va in so lieu.
"""

import os
import sys

import cv2
import numpy as np

cv2.setNumThreads(2)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from ai_engine.specialists.white_balance.wb import (  # noqa: E402
    apply,
    apply_wb,
    auto_exposure,
    estimate_wb_gains,
)

PAIRS_DIR = "data/pairs/before"
REVIEW_DIR = "data/review/before"
OUT_DIR = "outputs/wb_samples"
MAX_PANEL_W = 1500
IMG_EXTS = (".jpg", ".jpeg", ".png")


def _list_images(d):
    if not os.path.isdir(d):
        return []
    return [f for f in sorted(os.listdir(d)) if f.lower().endswith(IMG_EXTS)]


def _luma(img):
    return 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]


def pick_color_cast_images(d, n=3, exclude=()):
    """Xep hang anh trong thu muc theo do lech mau (R/B tren ban thu nho), chon n anh lech nhat."""
    scored = []
    for f in _list_images(d):
        if f in exclude:
            continue
        path = os.path.join(d, f)
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = 320.0 / w
        small = cv2.resize(img, (320, max(1, int(h * scale))))
        b = float(small[:, :, 0].mean())
        r = float(small[:, :, 2].mean())
        deviation = abs(r / (b + 1e-6) - 1.0)
        scored.append((deviation, f))
    scored.sort(key=lambda t: -t[0])
    return [f for _, f in scored[:n]]


def make_panel(orig_u8, wb_u8, full_u8, max_w=MAX_PANEL_W):
    h, w = orig_u8.shape[:2]
    scale = min(1.0, (max_w / 3.0) / w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    a = cv2.resize(orig_u8, (nw, nh))
    b = cv2.resize(wb_u8, (nw, nh))
    c = cv2.resize(full_u8, (nw, nh))
    return np.hstack([a, b, c])


def process_one(path):
    name = os.path.basename(path)
    img_u8 = cv2.imread(path)
    if img_u8 is None:
        print(f"[!] Khong doc duoc: {path}")
        return None
    img = img_u8.astype(np.float32) / 255.0

    gains = estimate_wb_gains(img)
    wb_only = apply_wb(img, gains, strength=0.8)
    full = apply(img, {"wb_strength": 0.8, "exposure": "auto", "target_median": 0.42})
    _, exp_info = auto_exposure(wb_only, {"exposure": "auto", "target_median": 0.42})

    med_before = float(np.median(_luma(img)))
    med_after_wb = float(np.median(_luma(wb_only)))
    med_after_full = float(np.median(_luma(full)))

    print(f"[{name}]  size={img_u8.shape[1]}x{img_u8.shape[0]}")
    print(f"  gains: R={gains['r']:.3f} G={gains['g']:.3f} B={gains['b']:.3f}")
    print(
        f"  median luma: before={med_before:.3f} -> after WB={med_after_wb:.3f}"
        f" -> after WB+exposure={med_after_full:.3f}"
    )
    print(
        f"  exposure: gain={exp_info['gain']:.3f} offset={exp_info['offset']:+.3f}"
        f" gamma={exp_info['gamma']:.3f}"
    )

    assert full.shape == img.shape, "Output phai dung kich thuoc goc"

    wb_u8 = np.clip(wb_only * 255.0, 0, 255).astype(np.uint8)
    full_u8 = np.clip(full * 255.0, 0, 255).astype(np.uint8)
    panel = make_panel(img_u8, wb_u8, full_u8)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, name.rsplit(".", 1)[0] + ".jpg")
    cv2.imwrite(out_path, panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  saved: {out_path}\n")
    return out_path


def main():
    picks_pairs = pick_color_cast_images(PAIRS_DIR, 3)
    picks_review = pick_color_cast_images(REVIEW_DIR, 3, exclude=set(picks_pairs))
    print(f"Chon tu {PAIRS_DIR}: {picks_pairs}")
    print(f"Chon tu {REVIEW_DIR}: {picks_review}\n")

    for f in picks_pairs:
        process_one(os.path.join(PAIRS_DIR, f))
    for f in picks_review:
        process_one(os.path.join(REVIEW_DIR, f))


if __name__ == "__main__":
    main()
