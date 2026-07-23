"""
Chay thu con "KHU NHIEU + PHUC NET" tren 5 anh toi nhat (nhieu ro nhat) tu
data/pairs/before/. Luu outputs/ds_samples/<ten>.jpg =
    [goc | denoise | denoise+sharpen]  (downscale de xem toan canh)
    + crop 100% 400x400 vung chi tiet (goc/denoise/denoise+sharpen), ghep ben phai.
In so lieu nhieu (std vung phang) truoc/sau.

Chi DOC data/, KHONG ghi vao do.
"""

import os
import sys

import cv2
import numpy as np

cv2.setNumThreads(2)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ds  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PAIRS_BEFORE = os.path.join(ROOT, "data", "pairs", "before")
OUT_DIR = os.path.join(ROOT, "outputs", "ds_samples")

N_SAMPLES = 5
MAIN_PANEL_W = 1500
CROP_SIZE = 400


def _luma_u8(img_u8):
    return (0.114 * img_u8[:, :, 0] + 0.587 * img_u8[:, :, 1] + 0.299 * img_u8[:, :, 2])


def list_images(folder):
    return [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]


def pick_dark_noisy_images(paths, n):
    """Xep hang theo median luma tren ban thu nho (toi hon = uu tien), chon n anh toi nhat."""
    scored = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = 320.0 / max(h, w)
        small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
        med = float(np.median(_luma_u8(small.astype(np.float32))))
        scored.append((med, p))
    scored.sort(key=lambda t: t[0])
    return [p for _, p in scored[:n]]


def find_detail_crop(gray_u8, crop_size):
    """Tim vi tri crop_size x crop_size co gradient (chi tiet) cao nhat, tranh vien anh."""
    h, w = gray_u8.shape[:2]
    if h < crop_size or w < crop_size:
        crop_size = min(h, w)

    grad = cv2.Laplacian(gray_u8, cv2.CV_32F, ksize=3)
    grad = np.abs(grad)

    step = max(16, crop_size // 4)
    best_score = -1.0
    best_xy = (max(0, (w - crop_size) // 2), max(0, (h - crop_size) // 2))
    for y in range(0, h - crop_size + 1, step):
        for x in range(0, w - crop_size + 1, step):
            score = float(grad[y:y + crop_size, x:x + crop_size].sum())
            if score > best_score:
                best_score = score
                best_xy = (x, y)
    return best_xy[0], best_xy[1], crop_size


def resize_to_width(img, target_w):
    h, w = img.shape[:2]
    scale = target_w / w
    return cv2.resize(img, (target_w, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


def label(img, text, color=(0, 255, 0)):
    out = img.copy()
    cv2.putText(out, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return out


def pad_to_width(img, target_w):
    h, w = img.shape[:2]
    if w >= target_w:
        return img
    canvas = np.zeros((h, target_w, 3), dtype=img.dtype)
    canvas[:, :w] = img
    return canvas


def process_one(path):
    name = os.path.basename(path)
    img_u8 = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_u8 is None:
        print(f"[!] Khong doc duoc: {path}")
        return None

    img = img_u8.astype(np.float32) / 255.0
    h, w = img.shape[:2]

    denoised = ds.denoise(img, 0.35)
    full = ds.apply(img, {"denoise_strength": 0.35, "sharpen_amount": 0.3})

    assert denoised.shape == img.shape, f"Sai shape denoise() cho {name}"
    assert full.shape == img.shape, f"Sai shape apply() cho {name}"

    gray_before = ds._luma(img).astype(np.float32)
    gray_denoised = ds._luma(denoised).astype(np.float32)
    gray_full = ds._luma(full).astype(np.float32)

    noise_before, _ = ds._flat_region_noise_std(gray_before)
    noise_denoised, _ = ds._flat_region_noise_std(gray_denoised)
    noise_full, _ = ds._flat_region_noise_std(gray_full)

    print(f"[{name}]  size={w}x{h}")
    print(
        f"  noise std (vung phang, luma 0-1): before={noise_before:.5f}"
        f" -> after denoise={noise_denoised:.5f}"
        f" -> after denoise+sharpen={noise_full:.5f}"
    )

    denoised_u8 = np.clip(denoised * 255.0, 0, 255).astype(np.uint8)
    full_u8 = np.clip(full * 255.0, 0, 255).astype(np.uint8)

    # --- panel toan canh ---
    col_w = MAIN_PANEL_W // 3
    a = label(resize_to_width(img_u8, col_w), "Before")
    b = label(resize_to_width(denoised_u8, col_w), "Denoise")
    c = label(resize_to_width(full_u8, col_w), "Denoise+Sharpen")
    hh = min(a.shape[0], b.shape[0], c.shape[0])
    main_panel = np.hstack([a[:hh], b[:hh], c[:hh]])

    # --- crop 100% vung chi tiet ---
    gray_u8_before = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)
    cx, cy, csize = find_detail_crop(gray_u8_before, CROP_SIZE)
    crop_before = img_u8[cy:cy + csize, cx:cx + csize]
    crop_denoised = denoised_u8[cy:cy + csize, cx:cx + csize]
    crop_full = full_u8[cy:cy + csize, cx:cx + csize]

    crop_before_l = label(crop_before, "crop before", (0, 0, 255))
    crop_denoised_l = label(crop_denoised, "crop denoise", (0, 255, 0))
    crop_full_l = label(crop_full, "crop +sharpen", (0, 255, 255))
    crop_block = np.hstack([crop_before_l, crop_denoised_l, crop_full_l])

    final_w = max(main_panel.shape[1], crop_block.shape[1])
    main_panel_p = pad_to_width(main_panel, final_w)
    crop_block_p = pad_to_width(crop_block, final_w)
    canvas = np.vstack([main_panel_p, crop_block_p])

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, os.path.splitext(name)[0] + ".jpg")
    cv2.imwrite(out_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  detail crop at (x={cx}, y={cy}, size={csize})")
    print(f"  saved: {out_path}\n")
    return out_path


def main():
    all_paths = list_images(PAIRS_BEFORE)
    picks = pick_dark_noisy_images(all_paths, N_SAMPLES)
    print(f"Chon {len(picks)} anh toi nhat tu {PAIRS_BEFORE}:")
    for p in picks:
        print(f"  - {os.path.basename(p)}")
    print()

    saved = []
    for p in picks:
        out = process_one(p)
        if out:
            saved.append(out)

    print(f"Da luu {len(saved)}/{len(picks)} anh so sanh vao {OUT_DIR}")


if __name__ == "__main__":
    main()
