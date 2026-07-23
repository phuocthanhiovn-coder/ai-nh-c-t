"""
Task 17 REWRITE: chay thu con sky_replace (mask moi, chong bleed) tren anh EXTERIOR
that co troi.

Xep hang ung vien theo diem = sky_fraction * coolness(vung mask), tinh tren ban
thu nho (nhanh). Ly do doi metric: sky_fraction don thuan (ban cu) co the bi lua boi
tran/tuong noi that sang+hoi lanh (B>=R nhe) — nhan them coolness (B-R trung binh
CUA CHINH vung duoc mask) day cac ca noi that (coolness thap/am) xuong day bang xep
hang, chi con exterior that noi len top.

Voi moi anh duoc chon: luu outputs/sky_samples2/<ten>.jpg =
  hang tren: [goc | mask truc quan | ket qua thay troi (blue)]
  hang duoi: 100% CROP (KHONG resize) [goc | ket qua] quanh "duong mai nha"
             (roofline) — de nhin do net/khong-bleed o do phan giai that.

Them 1 anh NOI THAT ro rang vao cuoi danh sach (diem thap nhat / sky_fraction~0)
de chung minh mask KHONG con lot vao tran/tuong trang (khong chi dua vao gate).
"""

import os
import re
import sys

import cv2
import numpy as np

cv2.setNumThreads(2)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from ai_engine.specialists.sky_replace.replace import SKY_FRACTION_GATE, apply  # noqa: E402
from ai_engine.specialists.sky_replace.sky_assets import ensure_skies  # noqa: E402
from ai_engine.specialists.sky_replace.sky_mask import detect_sky  # noqa: E402

SRC_DIRS = [
    os.path.join("data", "pairs", "before"),
    os.path.join("data", "review", "before"),
    os.path.join("data", "unmatched", "after"),
]
OUT_DIR = os.path.join("outputs", "sky_samples2")
N_SAMPLES = 5
SCAN_MAX_DIM = 480
VIEW_MAX_W = 2100
CROP_W = 900
CROP_H = 320
MIN_MASK_PX_FOR_COOLNESS = 30
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
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


_DB_PREFIX_RE = re.compile(r"^db\d+_?")


def _dedupe_key(path):
    """Nhieu ban sao cung anh nam o cac thu muc db01..db06 (cung ten sau khi bo
    tien to dbNN_). Bo tien to nay de xep hang khong bi 1 anh chiem het top-5."""
    base = os.path.splitext(os.path.basename(path))[0].lower()
    base = _DB_PREFIX_RE.sub("", base)
    return base.lstrip("_")


def score_image(path):
    """(score, sky_fraction, coolness) tren ban thu nho — score = frac * coolness."""
    img_u8 = cv2.imread(path)
    if img_u8 is None:
        return None
    small_u8 = _resize_max_dim(img_u8, SCAN_MAX_DIM)
    small_f32 = small_u8.astype(np.float32) / 255.0
    mask, frac = detect_sky(small_f32)
    region = mask > 0.5
    if int(region.sum()) >= MIN_MASK_PX_FOR_COOLNESS:
        px = small_f32[region]
        coolness = float(px[:, 0].mean() - px[:, 2].mean())
    else:
        coolness = 0.0
    score = frac * coolness
    return score, frac, coolness


def pick_images(n=N_SAMPLES):
    """Tra ve (top_exterior_list, interior_pick, all_ranked) — moi phan tu la
    (score, frac, coolness, path)."""
    candidates = []
    for d in SRC_DIRS:
        for path in _list_images(d):
            r = score_image(path)
            if r is None:
                continue
            score, frac, coolness = r
            candidates.append((score, frac, coolness, path))
    candidates.sort(key=lambda t: -t[0])

    deduped = []
    seen_keys = set()
    for score, frac, coolness, path in candidates:
        key = _dedupe_key(path)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append((score, frac, coolness, path))

    top = deduped[:n]

    interior_pool = sorted(deduped, key=lambda t: (t[1], t[0]))
    interior_pick = interior_pool[0] if interior_pool else None

    return top, interior_pick, deduped


def mask_visual(mask):
    """Mask [0,1] -> anh mau gia (BGR uint8): den=0, xanh(cyan)=1."""
    m_u8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    vis = np.zeros((*m_u8.shape, 3), dtype=np.uint8)
    vis[:, :, 0] = m_u8  # B
    vis[:, :, 1] = m_u8  # G  -> cyan-ish
    return vis


def make_row(imgs_u8, max_w=VIEW_MAX_W):
    n = len(imgs_u8)
    h, w = imgs_u8[0].shape[:2]
    scale = min(1.0, (max_w / float(n)) / w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    resized = [cv2.resize(im, (nw, nh)) for im in imgs_u8]
    return np.hstack(resized)


def _roofline_row(mask_full):
    """Hang (row) dai dien cua duong mai nha: median cua 'diem khong-phai-troi
    dau tien' tren cac cot ma hang tren cung LA troi. None neu khong co troi."""
    H, W = mask_full.shape[:2]
    binm = mask_full > 0.5
    top_is_sky = binm[0, :]
    if not top_is_sky.any():
        return None
    not_sky = ~binm
    has_ns = not_sky.any(axis=0)
    first_false = np.where(has_ns, not_sky.argmax(axis=0), H)
    valid = first_false[top_is_sky & has_ns]
    if valid.size == 0:
        return None
    return int(np.median(valid))


def _crop_100pct(img_u8, cy, cx, ch=CROP_H, cw=CROP_W):
    H, W = img_u8.shape[:2]
    ch = min(ch, H)
    cw = min(cw, W)
    y0 = int(np.clip(cy - ch // 2, 0, max(H - ch, 0)))
    x0 = int(np.clip(cx - cw // 2, 0, max(W - cw, 0)))
    return img_u8[y0 : y0 + ch, x0 : x0 + cw].copy()


def _pad_to_width(img_u8, width):
    h, w = img_u8.shape[:2]
    if w >= width:
        return img_u8[:, :width]
    pad = np.zeros((h, width - w, 3), dtype=img_u8.dtype)
    return np.hstack([img_u8, pad])


def process_one(path, score, frac_hint, coolness_hint, tag=""):
    name = os.path.basename(path)
    img_u8 = cv2.imread(path)
    if img_u8 is None:
        print(f"[!] Khong doc duoc: {path}")
        return None
    img = img_u8.astype(np.float32) / 255.0
    h, w = img.shape[:2]

    mask, sky_fraction = detect_sky(img)
    gated = sky_fraction < SKY_FRACTION_GATE

    result_blue = apply(img, {"sky": "blue", "strength": 1.0, "harmonize": True})
    assert result_blue.shape == img.shape, f"Sai shape output cho {name}"
    result_u8 = np.clip(result_blue * 255.0 + 0.5, 0, 255).astype(np.uint8)

    print(
        f"[{tag}{name}] size={w}x{h}  sky_fraction={sky_fraction:.4f} "
        f"(scan_score={score:.4f} scan_frac={frac_hint:.4f} scan_cool={coolness_hint:+.4f})  "
        f"gate<{SKY_FRACTION_GATE}={gated}"
    )

    mask_vis = mask_visual(mask)
    top_row = make_row([img_u8, mask_vis, result_u8])

    roof_row = _roofline_row(mask)
    if roof_row is not None:
        cy = roof_row
    else:
        cy = max(1, h // 8)
    cx = w // 2
    crop_orig = _crop_100pct(img_u8, cy, cx)
    crop_result = _crop_100pct(result_u8, cy, cx)
    crop_row = np.hstack([crop_orig, crop_result])
    crop_row = _pad_to_width(crop_row, top_row.shape[1]) if crop_row.shape[1] < top_row.shape[1] \
        else crop_row[:, : top_row.shape[1]]

    panel = np.vstack([top_row, crop_row])

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, os.path.splitext(name)[0] + ".jpg")
    cv2.imwrite(out_path, panel, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  saved: {out_path}  (roofline_row_full={roof_row})")

    outside = mask <= 0.0
    if outside.any():
        diff = np.abs(result_blue - img)
        max_diff_outside = float(diff[outside].max())
    else:
        max_diff_outside = 0.0
    print(f"  assert outside-mask bit-identical: max_diff_outside={max_diff_outside:.3e}")

    return out_path, sky_fraction, gated, max_diff_outside


def main():
    plates = ensure_skies()
    print(f"Sky plates ready: {list(plates.keys())}")
    print()

    top, interior_pick, all_ranked = pick_images(N_SAMPLES)
    print(f"Da xep hang {len(all_ranked)} anh (dedup). Top {len(top)} exterior duoc chon:")
    for score, frac, cool, path in top:
        print(f"  score={score:+.4f} frac={frac:.4f} cool={cool:+.4f}  {path}")
    if interior_pick:
        print("Interior duoc chon rieng (frac thap nhat trong xep hang):")
        print(f"  score={interior_pick[0]:+.4f} frac={interior_pick[1]:.4f} "
              f"cool={interior_pick[2]:+.4f}  {interior_pick[3]}")
    print()

    for score, frac, cool, path in top:
        process_one(path, score, frac, cool)

    if interior_pick:
        score, frac, cool, path = interior_pick
        process_one(path, score, frac, cool, tag="INTERIOR_")


if __name__ == "__main__":
    main()
