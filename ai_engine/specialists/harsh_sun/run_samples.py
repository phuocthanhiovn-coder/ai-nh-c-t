"""
Chay thu con harsh_sun (Task 22).

Chua co data harsh-sun that -> tu dung bo test:
  1. Quet data/pairs/before + data/review/before (READ-ONLY), xep hang "do gat"
     = %pixel chay + %pixel kit (uu tien anh co CA HAI) -> lay top anh tuong phan
     gat tu nhien (thuong la ngoai that troi chay / noi that cua so chay).
  2. Lay them 1 anh phoi sang CHUAN nhat (do gat thap nhat) -> test gate/subtle.
  3. SYNTH: boost contrast + clip 2 anh de gia lap troi chay + bong kit.

Moi test luu outputs/harshsun_samples/<ten>.jpg =
  hang tren  [input | tone-mapped] (thu nho de xem toan canh)
  hang duoi  [crop 100% input | crop 100% output] tai canh tuong-phan-cao
             (tu tim: canh Sobel manh nhat ke vung chay -> soi halo roofline).
In dynamic range truoc/sau: p99-p1 cua luma sRGB + khoang EV (log2 p99/p1 luma linear).
"""

import os
import sys

import cv2
import numpy as np

cv2.setNumThreads(2)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from ai_engine.specialists.harsh_sun.tone_map import apply, compute_gates  # noqa: E402

SRC_DIRS = [
    os.path.join("data", "pairs", "before"),
    os.path.join("data", "review", "before"),
]
OUT_DIR = os.path.join("outputs", "harshsun_samples")
IMG_EXTS = (".jpg", ".jpeg", ".png")
N_HARSH = 5          # anh tuong phan gat tu nhien
N_SYNTH = 2          # so anh dem synth hoa
PANEL_W = 1600       # be rong hang tren
CROP = 320           # kich thuoc crop 100%
JPEG_Q = 92


def _list_images(d):
    if not os.path.isdir(d):
        return []
    return [os.path.join(d, f) for f in sorted(os.listdir(d))
            if f.lower().endswith(IMG_EXTS)]


def _luma(img):
    return 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]


def harshness(path):
    """Cham do gat tren ban thu nho (IMREAD_REDUCED x8) cho re."""
    small = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_8)
    if small is None:
        return None
    y = _luma(small.astype(np.float32) / 255.0)
    hi = float((y >= 0.95).mean())
    lo = float((y <= 0.05).mean())
    return hi + lo + 2.0 * min(hi, lo)   # co CA chay va kit -> dung ca harsh-sun nhat


def rank_images():
    scored = []
    for d in SRC_DIRS:
        for p in _list_images(d):
            s = harshness(p)
            if s is not None:
                scored.append((s, p))
    scored.sort(key=lambda t: -t[0])
    return scored


def synth_harsh(img):
    """Gia lap nang gat: boost contrast quanh MEDIAN luma cua chinh anh do
    (pivot co dinh 0.45 tren anh toi se nghien ca anh ve 0 - khong giong nang gat
    that von co troi chay + bong kit nhung midtone van con)."""
    pivot = float(np.median(_luma(img)))
    out = (img - pivot) * 2.2 + pivot
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def dr_stats(img):
    """(p99-p1 luma sRGB, khoang EV log2(p99/p1) luma 'linear' xap xi gamma2.2)."""
    y = _luma(img)
    p1, p99 = np.percentile(y, [1, 99])
    ylin = np.power(np.maximum(y, 1e-6), 2.2)
    l1, l99 = np.percentile(ylin, [1, 99])
    ev = float(np.log2(max(l99, 1e-6) / max(l1, 1e-6)))
    return float(p99 - p1), ev


def find_crop_center(img, crop):
    """Tam crop = canh manh nhat KE vung chay (roofline vs troi) de soi halo.
    Khong co vung chay -> canh manh nhat toan anh."""
    h, w = img.shape[:2]
    y = _luma(img).astype(np.float32)
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.abs(gx) + np.abs(gy)

    blown = (y >= 0.92).astype(np.uint8)
    if blown.any():
        near = cv2.dilate(blown, np.ones((25, 25), np.uint8))
        edge = edge * near
    score = cv2.boxFilter(edge, -1, (crop // 2, crop // 2))

    m = crop // 2
    inner = score[m:h - m, m:w - m] if h > crop and w > crop else score
    iy, ix = np.unravel_index(np.argmax(inner), inner.shape)
    cy, cx = iy + (m if h > crop else 0), ix + (m if w > crop else 0)
    cy = int(np.clip(cy, m, max(h - m, m)))
    cx = int(np.clip(cx, m, max(w - m, m)))
    return cy, cx


def _label(img_u8, text):
    cv2.putText(img_u8, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img_u8, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 1, cv2.LINE_AA)
    return img_u8


def make_panel(img_in, img_out, crop=CROP, panel_w=PANEL_W):
    h, w = img_in.shape[:2]
    in_u8 = np.clip(img_in * 255.0, 0, 255).astype(np.uint8)
    out_u8 = np.clip(img_out * 255.0, 0, 255).astype(np.uint8)

    half = panel_w // 2
    scale = half / w
    nh = max(1, int(h * scale))
    top = np.hstack([
        _label(cv2.resize(in_u8, (half, nh)), "INPUT"),
        _label(cv2.resize(out_u8, (half, nh)), "TONE-MAPPED"),
    ])

    cy, cx = find_crop_center(img_in, crop)
    m = crop // 2
    y0, y1 = max(cy - m, 0), min(cy + m, h)
    x0, x1 = max(cx - m, 0), min(cx + m, w)
    ci = _label(in_u8[y0:y1, x0:x1].copy(), "crop in 100%")
    co = _label(out_u8[y0:y1, x0:x1].copy(), "crop out 100%")
    bottom = np.hstack([ci, co])
    if bottom.shape[1] < panel_w:
        pad = np.zeros((bottom.shape[0], panel_w - bottom.shape[1], 3), np.uint8)
        bottom = np.hstack([bottom, pad])
    return np.vstack([top, bottom[:, :panel_w]])


def run_case(name, img, note=""):
    hi_gate, lo_gate, hi_frac, lo_frac = compute_gates(img)
    out = apply(img, None)   # params mac dinh
    assert out.shape == img.shape and out.dtype == np.float32

    dr_in, ev_in = dr_stats(img)
    dr_out, ev_out = dr_stats(out)
    print(f"[{name}] {img.shape[1]}x{img.shape[0]}  {note}")
    print(f"   clip: hi={hi_frac*100:.2f}% lo={lo_frac*100:.2f}%"
          f"  gate hi={hi_gate:.2f} lo={lo_gate:.2f}")
    print(f"   DR p99-p1: {dr_in:.3f} -> {dr_out:.3f}"
          f"   |  EV(p99/p1): {ev_in:.2f} -> {ev_out:.2f}")
    mean_delta = float(np.abs(out - img).mean())
    print(f"   mean |delta| = {mean_delta:.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, name + ".jpg")
    panel = make_panel(img, out)
    ok = cv2.imwrite(out_path, panel, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    assert ok, f"Ghi that bai: {out_path}"
    print(f"   saved: {out_path}\n")
    return out_path


def main():
    scored = rank_images()
    if not scored:
        print("Khong tim thay anh nguon.")
        return
    print(f"Da quet {len(scored)} anh. Top do gat:")
    for s, p in scored[:6]:
        print(f"  {s:.4f}  {p}")
    normal = scored[-1]
    print(f"Anh phoi sang chuan nhat: {normal[1]} (score={normal[0]:.4f})\n")

    picks = scored[:N_HARSH]
    for i, (s, p) in enumerate(picks):
        img_u8 = cv2.imread(p)
        if img_u8 is None:
            continue
        img = img_u8.astype(np.float32) / 255.0
        base = os.path.splitext(os.path.basename(p))[0]
        run_case(f"harsh{i+1}_{base}", img, note=f"(tu nhien, score={s:.4f})")

    img_u8 = cv2.imread(normal[1])
    if img_u8 is not None:
        img = img_u8.astype(np.float32) / 255.0
        base = os.path.splitext(os.path.basename(normal[1]))[0]
        run_case(f"normal_{base}", img, note="(phoi sang chuan - test gate)")

    # SYNTH tu anh phoi sang chuan + 1 anh ngoai that ke tiep bang xep hang:
    # co midtone that de kiem tra phuc hoi, khong nhu anh von da toi.
    synth_srcs = [normal[1]]
    if len(scored) > N_HARSH:
        synth_srcs.append(scored[N_HARSH][1])
    for i, p in enumerate(synth_srcs[:N_SYNTH]):
        img_u8 = cv2.imread(p)
        if img_u8 is None:
            continue
        img = img_u8.astype(np.float32) / 255.0
        base = os.path.splitext(os.path.basename(p))[0]
        run_case(f"synth{i+1}_{base}", synth_harsh(img), note="(SYNTH clip)")


if __name__ == "__main__":
    main()
