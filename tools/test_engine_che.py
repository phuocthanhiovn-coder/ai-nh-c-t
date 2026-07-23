"""Nghiem thu noi CH_E_antiwash vao ops_basic.auto_enhance (22/07).
Chay op THAT qua registry tren anh full-res, xuat panel BEFORE | ENGINE CH_E | TARGET.
"""
import os
import time

import cv2
import numpy as np

cv2.setNumThreads(3)

from ai_engine.orchestrator.registry import REGISTRY

NAMES = [
    "_ML_1421.jpg",
    "after_pool2_gd09_783A9534.jpg",
    "drone01_DSC01518.jpg",
    "j021_FP101671.jpg",
    "_ML_1444.jpg",
    "j054_DSC4574.jpg",
]
OUT = "outputs/engine_che"


def panel(img, label, w=760):
    h = int(round(img.shape[0] * w / img.shape[1]))
    r = cv2.resize((img * 255).clip(0, 255).astype("uint8"), (w, h))
    cv2.rectangle(r, (0, 0), (w, 44), (0, 0, 0), -1)
    cv2.putText(r, label, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return r


def main():
    os.makedirs(OUT, exist_ok=True)
    fn = REGISTRY["auto_enhance"]["fn"]
    for n in NAMES:
        bp, ap = f"data/pairs/before/{n}", f"data/pairs/after/{n}"
        if not os.path.exists(bp):
            print(f"skip {n} (khong co)")
            continue
        b = cv2.imread(bp).astype(np.float32) / 255.0
        t0 = time.time()
        e = fn(b, {})
        dt = time.time() - t0
        assert e.shape == b.shape, f"{n}: shape {e.shape} != {b.shape}"
        assert e.dtype == np.float32
        changed = float(np.abs(e - b).mean())
        row = [panel(b, "BEFORE"), panel(e, "ENGINE auto_enhance (CH_E)")]
        if os.path.exists(ap):
            a = cv2.imread(ap).astype(np.float32) / 255.0
            if a.shape[:2] != b.shape[:2]:
                a = cv2.resize(a, (b.shape[1], b.shape[0]))
            row.append(panel(a, "TARGET (AutoHDR)"))
        cv2.imwrite(os.path.join(OUT, f"eng_{n}"), np.hstack(row),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        print(f"{n}: {b.shape[1]}x{b.shape[0]} | {dt:.1f}s | mean|delta|={changed:.4f}")
    print("DONE")


if __name__ == "__main__":
    main()
