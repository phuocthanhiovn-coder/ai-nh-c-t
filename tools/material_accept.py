"""Nghiem thu tang chat lieu: 10 anh NHIEU DO VAT qua full stack
(nao v1.5 + material_grade) vs target. Xuat side-by-side de nhin."""
import os

import cv2
import numpy as np

cv2.setNumThreads(3)

from ai_engine.brain.run import process
from ai_engine.brain.material_grade import apply_material_grade

NAMES = [
    "k001_DSC6441.jpg", "j021_FP101671.jpg", "j054_DSC4574.jpg",
    "drone01_DSC01518.jpg", "20260703-DSC1132.jpg", "after_pool2_gd19_783A9887.jpg",
    "k016_dsc02831.jpg", "j023_DSC9601.jpg", "j066_FP102874.jpg", "k007_227A2199.jpg",
]
OUT = "outputs/material_accept"


def panel(im_u8, label, w=640):
    h = int(round(im_u8.shape[0] * w / im_u8.shape[1]))
    r = np.ascontiguousarray(cv2.resize(im_u8, (w, h)))
    cv2.rectangle(r, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.putText(r, label, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return r


def main():
    os.makedirs(OUT, exist_ok=True)
    for n in NAMES:
        bp = f"data/pairs/before/{n}"
        if not os.path.exists(bp):
            print("skip", n); continue
        img = cv2.imread(bp).astype(np.float32) / 255.0
        base, _rec = process(img)
        log = []
        final = apply_material_grade(base, record=log)
        t = cv2.imread(f"data/pairs/after/{n}")
        if t.shape[:2] != final.shape[:2]:
            t = cv2.resize(t, (final.shape[1], final.shape[0]))
        f8 = (final * 255).clip(0, 255).astype(np.uint8)
        hh = min(panel(f8, "").shape[0], panel(t, "").shape[0])
        cv2.imwrite(os.path.join(OUT, "acc_" + n),
                    np.hstack([panel(f8, "FULL STACK")[:hh], panel(t, "TARGET")[:hh]]),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        mats = " | ".join(f"{s['op']}({s['frac']})" for s in log)
        print(f"{n}: {mats if mats else 'khong nhom chat lieu nao'}", flush=True)
    print("ACCEPT_DONE")


if __name__ == "__main__":
    main()
