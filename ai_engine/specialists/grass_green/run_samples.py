"""
Chay thu con grass_green tren anh ngoai that co co.
Tim 5 anh trong data/pairs/before/, data/review/before/, data/unmatched/after/ (doc thoi,
khong ghi vao data/) co dien tich mask co lon nhat (xep hang bang segment_grass tren anh
FULL-RES luon - ~0.4s/anh nen chay het pool la du re, va tranh sai lech giua diem xep
hang tren ban thu nho voi ket qua full-res thuc te, loai anh mask qua nho vi thuong la
anh noi that).
Luu outputs/grass_samples/<ten>.jpg = ghep ngang [goc | mask truc quan | ket qua].
In % dien tich mask cho tung anh.
"""

import os
import sys

import cv2
import numpy as np

cv2.setNumThreads(2)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from ai_engine.specialists.grass_green.grass import apply, segment_grass  # noqa: E402

SRC_DIRS = [
    os.path.join("data", "pairs", "before"),
    os.path.join("data", "review", "before"),
    os.path.join("data", "unmatched", "after"),
]
OUT_DIR = os.path.join("outputs", "grass_samples")
N_SAMPLES = 5
VIEW_MAX_W = 1500
IMG_EXTS = (".jpg", ".jpeg", ".png")
MIN_AREA_FRAC = 0.03  # duoi nguong nay coi nhu khong co co ro rang, bo qua khi xep hang


def _list_images(d):
    if not os.path.isdir(d):
        return []
    return [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.lower().endswith(IMG_EXTS)]


def score_grass_area(path):
    img_u8 = cv2.imread(path)
    if img_u8 is None:
        return None
    img_f32 = img_u8.astype(np.float32) / 255.0
    mask = segment_grass(img_f32)
    area_frac = float(mask.mean())
    return area_frac


def _dedupe_key(path):
    """Nhieu ban sao cung canh chup nam o cac thu muc/ten khac nhau (vd tien to
    'db01_', hoac cung ten o ca data/pairs va data/review). Gom theo ten file da
    bo tien to 'db01' de tranh chon trung 1 canh nhieu lan (va tranh dam ten file
    output len nhau)."""
    base = os.path.splitext(os.path.basename(path))[0].lower()
    if base.startswith("db01_"):
        base = base[len("db01_"):]
    elif base.startswith("db01"):
        base = base[len("db01"):]
    return base.lstrip("_")


def pick_grass_images(n=N_SAMPLES):
    candidates = []
    for d in SRC_DIRS:
        for path in _list_images(d):
            area = score_grass_area(path)
            if area is None:
                continue
            candidates.append((area, path))
    candidates.sort(key=lambda t: -t[0])

    deduped = []
    seen_keys = set()
    for area, path in candidates:
        key = _dedupe_key(path)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append((area, path))

    picked = [c for c in deduped if c[0] >= MIN_AREA_FRAC][:n]
    if len(picked) < n:
        picked = deduped[:n]
    return picked


def mask_visual(mask):
    """Mask [0,1] -> anh mau gia (BGR uint8) de xem truc quan: den=0, xanh=1."""
    m_u8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    vis = np.zeros((*m_u8.shape, 3), dtype=np.uint8)
    vis[:, :, 1] = m_u8  # kenh G
    return vis


def make_panel(orig_u8, mask_vis_u8, result_u8, max_w=VIEW_MAX_W):
    h, w = orig_u8.shape[:2]
    scale = min(1.0, (max_w / 3.0) / w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    a = cv2.resize(orig_u8, (nw, nh))
    b = cv2.resize(mask_vis_u8, (nw, nh))
    c = cv2.resize(result_u8, (nw, nh))
    return np.hstack([a, b, c])


def process_one(path, area_hint):
    name = os.path.basename(path)
    img_u8 = cv2.imread(path)
    if img_u8 is None:
        print(f"[!] Khong doc duoc: {path}")
        return None
    img = img_u8.astype(np.float32) / 255.0

    mask = segment_grass(img)
    result = apply(img, {"strength": 0.7})

    assert result.shape == img.shape, f"Sai shape output cho {name}"

    area_full = float(mask.mean()) * 100.0
    print(f"[{name}] size={img_u8.shape[1]}x{img_u8.shape[0]}  mask_area={area_full:.2f}%")

    mask_vis = mask_visual(mask)
    result_u8 = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    panel = make_panel(img_u8, mask_vis, result_u8)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, os.path.splitext(name)[0] + ".jpg")
    cv2.imwrite(out_path, panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  saved: {out_path}")
    return out_path


def main():
    picks = pick_grass_images(N_SAMPLES)
    print(f"Da chon {len(picks)} anh (xep hang theo % dien tich mask co, full-res):")
    for area, path in picks:
        print(f"  {path}  area={area*100:.2f}%")
    print()

    for area, path in picks:
        process_one(path, area)


if __name__ == "__main__":
    main()
