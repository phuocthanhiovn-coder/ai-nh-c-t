"""
Chay 3 case test cua con object_removal (Task 18). Voi moi case:
  outputs/removal_samples/<name>.jpg = [goc | mask overlay | da xoa] (1800px)
                                        + crop 100% vung xoa (goc | da xoa)
In runtime (giay CPU) + assert bit-identical ngoai feathered mask (remover.py
da tu assert; o day chi in lai so lieu tong hop).

Chi DOC data/, ghi vao outputs/removal_samples/.
"""

import os
import sys
import time

import cv2
import numpy as np

cv2.setNumThreads(2)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

from ai_engine.core.quality import read_image_16, write_image  # noqa: E402
from ai_engine.specialists.object_removal import remover  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PAIRS_BEFORE = os.path.join(ROOT, "data", "pairs", "before")
OUT_DIR = os.path.join(ROOT, "outputs", "removal_samples")

CASES = [
    ("switch_plate", "20260703-DSC1105.jpg"),
    ("marquee_letter", "20260703-DSC1226.jpg"),
    ("decor_bowl", "20260703-DSC1161.jpg"),
]

PANEL_W = 1800
CROP_MARGIN = 60


def resize_to_width(img, target_w):
    h, w = img.shape[:2]
    scale = target_w / w
    return cv2.resize(img, (target_w, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


def label(img, text, color=(0, 255, 0)):
    out = img.copy()
    cv2.putText(out, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return out


def pad_to_width(img, target_w):
    h, w = img.shape[:2]
    if w >= target_w:
        return img
    canvas = np.zeros((h, target_w, 3), dtype=img.dtype)
    canvas[:, :w] = img
    return canvas


def to_u8(img_f32):
    return np.clip(img_f32 * 255.0 + 0.5, 0, 255).astype(np.uint8)


def run_case(name, fname):
    src_path = os.path.join(PAIRS_BEFORE, fname)
    mask_path = os.path.join(OUT_DIR, f"mask_{name}.png")

    img_f32 = read_image_16(src_path)
    h, w = img_f32.shape[:2]

    mask_u8 = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    assert mask_u8 is not None, f"Khong doc duoc mask: {mask_path}"

    t0 = time.time()
    removed_f32 = remover.apply(img_f32, {"mask_path": mask_path})
    elapsed = time.time() - t0

    assert removed_f32.shape == img_f32.shape, f"Sai shape output cho {name}"

    orig_u8 = to_u8(img_f32)
    removed_u8 = to_u8(removed_f32)

    overlay_u8 = orig_u8.copy()
    red = np.zeros_like(orig_u8)
    red[:, :, 2] = 255
    m3 = (mask_u8 > 0)[:, :, None]
    overlay_u8 = np.where(m3, cv2.addWeighted(orig_u8, 0.5, red, 0.5, 0), orig_u8)

    # --- panel toan canh: goc | mask overlay | da xoa ---
    col_w = PANEL_W // 3
    a = label(resize_to_width(orig_u8, col_w), "Before", (0, 255, 0))
    b = label(resize_to_width(overlay_u8, col_w), "Mask", (0, 0, 255))
    c = label(resize_to_width(removed_u8, col_w), "Removed", (0, 255, 255))
    hh = min(a.shape[0], b.shape[0], c.shape[0])
    main_panel = np.hstack([a[:hh], b[:hh], c[:hh]])

    # --- crop 100% vung xoa (bbox mask + le) ---
    ys, xs = np.where(mask_u8 > 0)
    y0 = max(0, int(ys.min()) - CROP_MARGIN)
    y1 = min(h, int(ys.max()) + CROP_MARGIN)
    x0 = max(0, int(xs.min()) - CROP_MARGIN)
    x1 = min(w, int(xs.max()) + CROP_MARGIN)

    crop_before = label(orig_u8[y0:y1, x0:x1].copy(), "crop before", (0, 0, 255))
    crop_after = label(removed_u8[y0:y1, x0:x1].copy(), "crop 100% removed", (0, 255, 255))
    crop_block = np.hstack([crop_before, crop_after])

    final_w = max(main_panel.shape[1], crop_block.shape[1])
    main_panel_p = pad_to_width(main_panel, final_w)
    crop_block_p = pad_to_width(crop_block, final_w)
    canvas = np.vstack([main_panel_p, crop_block_p])

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{name}.jpg")
    write_image(out_path, canvas.astype(np.float32) / 255.0, quality=92)

    # remover.apply() da tu assert bit-identical NGOAI feathered mask (crash neu sai) va
    # in so lieu chi tiet; day chi la thong ke tham khao ve vung anh huong (dilate+feather)
    # so voi mask goc nguoi ve, KHONG phai bien kiem tra hop dong.
    affected_raw = int(cv2.countNonZero(mask_u8))
    diff_all = np.abs(orig_u8.astype(np.int16) - removed_u8.astype(np.int16))
    affected_total = int((diff_all.max(axis=2) > 0).sum())

    print(
        f"[{name}] src={fname} size={w}x{h} mask_px_nguoi_ve={affected_raw} "
        f"crop_bbox=({x0},{y0})-({x1},{y1}) runtime={elapsed:.2f}s CPU "
        f"tong_px_thuc_te_bi_doi(dilate+feather)={affected_total} -> {out_path}"
    )
    return elapsed, affected_total, out_path


def main():
    results = []
    for name, fname in CASES:
        results.append((name,) + run_case(name, fname))

    print("\n=== TONG KET ===")
    for name, elapsed, affected_total, out_path in results:
        print(f"  {name}: {elapsed:.2f}s CPU, px_bi_doi={affected_total}, out={out_path}")


if __name__ == "__main__":
    main()
