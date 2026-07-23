"""Chan doan "anh mo" (22/07): so ENGINE output vs TARGET AutoHDR.
- Crop 100% (khong resize) cung vi tri -> outputs/diagnose_blur/
- Metric: Laplacian variance (do net canh), local contrast (std cua luma trong o 32px),
  saturation trung binh. Do tren CUNG vung anh de cong bang.
"""
import os

import cv2
import numpy as np

cv2.setNumThreads(3)

from ai_engine.orchestrator.registry import REGISTRY

NAMES = [
    "_ML_1421.jpg",
    "after_pool2_gd09_783A9534.jpg",
    "j021_FP101671.jpg",
    "_ML_1444.jpg",
]
OUT = "outputs/diagnose_blur"
CROP = 640  # canh crop 100%


def metrics(bgr_f32):
    g = cv2.cvtColor((bgr_f32 * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(g, cv2.CV_64F).var()
    # local contrast: std luma trong o 32x32, lay trung binh
    gf = g.astype(np.float32)
    k = 32
    h, w = gf.shape
    hh, ww = h // k * k, w // k * k
    tiles = gf[:hh, :ww].reshape(hh // k, k, ww // k, k)
    local_std = tiles.std(axis=(1, 3)).mean()
    hsv = cv2.cvtColor((bgr_f32 * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].mean()
    return lap, local_std, sat


def best_crop(gray):
    """Chon vung nhieu chi tiet nhat (gradient cao) de crop 100%."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = cv2.boxFilter(np.abs(gx) + np.abs(gy), -1, (CROP, CROP))
    h, w = gray.shape
    m = mag[CROP // 2:h - CROP // 2, CROP // 2:w - CROP // 2]
    y, x = np.unravel_index(np.argmax(m), m.shape)
    return y, x  # goc trai tren cua crop


def label(img_u8, text):
    cv2.rectangle(img_u8, (0, 0), (img_u8.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(img_u8, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img_u8


def main():
    os.makedirs(OUT, exist_ok=True)
    fn = REGISTRY["auto_enhance"]["fn"]
    print(f"{'anh':<34} {'lap_var':>18} {'local_std':>18} {'sat':>14}")
    print(f"{'':<34} {'eng / tgt':>18} {'eng / tgt':>18} {'eng / tgt':>14}")
    for n in NAMES:
        b = cv2.imread(f"data/pairs/before/{n}").astype(np.float32) / 255.0
        a = cv2.imread(f"data/pairs/after/{n}").astype(np.float32) / 255.0
        if a.shape[:2] != b.shape[:2]:
            a = cv2.resize(a, (b.shape[1], b.shape[0]))
        e = fn(b, {})

        le, se, ce = metrics(e)
        lt, st, ct = metrics(a)
        print(f"{n:<34} {le:8.1f} /{lt:8.1f} {se:8.2f} /{st:8.2f} {ce:6.1f} /{ct:6.1f}")

        gray = cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        y, x = best_crop(gray)
        ce_ = np.ascontiguousarray((e[y:y + CROP, x:x + CROP] * 255).astype(np.uint8))
        ca_ = np.ascontiguousarray((a[y:y + CROP, x:x + CROP] * 255).astype(np.uint8))
        cb_ = np.ascontiguousarray((b[y:y + CROP, x:x + CROP] * 255).astype(np.uint8))
        row = np.hstack([label(cb_, "BEFORE 100%"), label(ce_, "ENGINE 100%"), label(ca_, "TARGET 100%")])
        cv2.imwrite(os.path.join(OUT, f"crop_{n}"), row, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print("DONE")


if __name__ == "__main__":
    main()
