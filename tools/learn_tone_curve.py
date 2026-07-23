"""Hoc DUONG CONG SANG AutoHDR tu data that (23/07).

Chu du an: "AutoHDR do sang rat cao, moi goc deu sang; anh ta cho sang cho toi".
Cach lam: lay mau ~48 cap, chay model CH_F tren before, do luma percentile
p1..p99 cua (model_out) vs (target after) -> trung vi tung bac tren toan bo mau
= duong cong tone can ap them. Luu JSON cho op ap dung.
"""
import json
import os

import cv2
import numpy as np

cv2.setNumThreads(3)

from ai_engine.orchestrator.registry import REGISTRY

_LUMA_W = np.array([0.0722, 0.7152, 0.2126], dtype=np.float32)
PCTS = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 99]
OUT = "checkpoints/airy_tone_curve.json"
N_SAMPLE = 48


def main():
    fn = REGISTRY["auto_enhance"]["fn"]
    names = sorted(os.listdir("data/pairs/before"))
    step = max(1, len(names) // N_SAMPLE)
    sample = names[::step][:N_SAMPLE]

    xs, ys = [], []
    for i, n in enumerate(sample):
        b = cv2.imread(f"data/pairs/before/{n}")
        a = cv2.imread(f"data/pairs/after/{n}")
        if b is None or a is None:
            continue
        # proxy 1024 cho nhanh — phan bo luma gan nhu khong doi khi resize
        def rs(im):
            h, w = im.shape[:2]
            s = 1024.0 / max(h, w)
            return cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else im
        b = rs(b).astype(np.float32) / 255.0
        a = rs(a).astype(np.float32) / 255.0
        m = fn(b, {})
        ym = m @ _LUMA_W
        ya = a @ _LUMA_W
        xs.append(np.percentile(ym, PCTS))
        ys.append(np.percentile(ya, PCTS))
        if (i + 1) % 12 == 0:
            print(f"{i+1}/{len(sample)}", flush=True)

    X = np.median(np.stack(xs), axis=0)
    Y = np.median(np.stack(ys), axis=0)
    # ep don dieu tang de curve hop le
    for i in range(1, len(Y)):
        Y[i] = max(Y[i], Y[i - 1] + 1e-4)
        X[i] = max(X[i], X[i - 1] + 1e-4)
    curve = {"x": [0.0] + X.tolist() + [1.0], "y": [0.0] + Y.tolist() + [1.0],
             "pcts": PCTS, "n_sample": len(xs)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(curve, f, indent=1)
    print("\nDUONG CONG (model_out -> target):")
    for p, x, y in zip(PCTS, X, Y):
        print(f"  p{p:>2}: {x:.3f} -> {y:.3f}  (delta {255*(y-x):+5.1f}/255)")
    print("saved", OUT)


if __name__ == "__main__":
    main()
