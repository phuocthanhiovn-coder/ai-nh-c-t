"""Tune finish_detail tren cap thuc te: engine(CH_E) + finish vs TARGET AutoHDR.
Quet luoi tham so nho, cham diem = khoang cach metric (lap_var ratio, local_std ratio)
toi target. In bang + xuat crop 100% cho 2 bo tot nhat de NHIN MAT quyet dinh.
"""
import os

import cv2
import numpy as np

cv2.setNumThreads(3)

from ai_engine.orchestrator.registry import REGISTRY
from ai_engine.specialists.finish_detail import finish as fd

NAMES = [
    "_ML_1421.jpg",
    "after_pool2_gd09_783A9534.jpg",
    "j021_FP101671.jpg",
    "_ML_1444.jpg",
]
OUT = "outputs/finish_tune"
CROP = 640

GRID = [
    dict(clarity=0.3, detail=0.4, black=0.3),
    dict(clarity=0.5, detail=0.6, black=0.35),
    dict(clarity=0.7, detail=0.8, black=0.4),
    dict(clarity=0.5, detail=1.0, black=0.35),
    dict(clarity=0.8, detail=1.0, black=0.5),
]


def metrics(bgr):
    g = cv2.cvtColor((bgr * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(g, cv2.CV_64F).var()
    gf = g.astype(np.float32)
    k = 32
    h, w = gf.shape
    hh, ww = h // k * k, w // k * k
    local_std = gf[:hh, :ww].reshape(hh // k, k, ww // k, k).std(axis=(1, 3)).mean()
    return lap, local_std


def best_crop(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = cv2.boxFilter(np.abs(gx) + np.abs(gy), -1, (CROP, CROP))
    h, w = gray.shape
    m = mag[CROP // 2:h - CROP // 2, CROP // 2:w - CROP // 2]
    y, x = np.unravel_index(np.argmax(m), m.shape)
    return y, x


def label(img_u8, text):
    img_u8 = np.ascontiguousarray(img_u8)
    cv2.rectangle(img_u8, (0, 0), (img_u8.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(img_u8, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return img_u8


def main():
    os.makedirs(OUT, exist_ok=True)
    fn = REGISTRY["auto_enhance"]["fn"]

    data = []
    for n in NAMES:
        b = cv2.imread(f"data/pairs/before/{n}").astype(np.float32) / 255.0
        a = cv2.imread(f"data/pairs/after/{n}").astype(np.float32) / 255.0
        if a.shape[:2] != b.shape[:2]:
            a = cv2.resize(a, (b.shape[1], b.shape[0]))
        e = fn(b, {})
        data.append((n, b, a, e, metrics(a)))

    scores = []
    for gi, g in enumerate(GRID):
        tot = 0.0
        rows = []
        for (n, b, a, e, (lt, st)) in data:
            f = fd.apply(e, g)
            lf, sf = metrics(f)
            # diem: |log ty le| — muon lap/local cua ta ~ target
            tot += abs(np.log(max(lf, 1) / lt)) + abs(np.log(sf / st))
            rows.append((n, lf, lt, sf, st))
        scores.append((tot, gi, g, rows))
        print(f"[G{gi}] {g} -> score {tot:.3f}")
        for (n, lf, lt, sf, st) in rows:
            print(f"    {n:<32} lap {lf:7.1f}/{lt:7.1f}  lstd {sf:6.2f}/{st:6.2f}")

    scores.sort(key=lambda s: s[0])
    for rank, (tot, gi, g, _r) in enumerate(scores[:2]):
        for (n, b, a, e, _m) in data:
            f = fd.apply(e, g)
            gray = cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
            y, x = best_crop(gray)
            cr = lambda im, t: label((im[y:y + CROP, x:x + CROP] * 255).astype(np.uint8), t)
            row = np.hstack([cr(e, "ENGINE"), cr(f, f"+FINISH G{gi}"), cr(a, "TARGET")])
            cv2.imwrite(os.path.join(OUT, f"rank{rank}_G{gi}_{n}"), row,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print("BEST:", scores[0][2], "2nd:", scores[1][2])
    print("DONE")


if __name__ == "__main__":
    main()
