"""Test stack v2: cua so khu-mu + art tuong phan + go tu nhien (25/07)."""
import os

import cv2
import numpy as np

cv2.setNumThreads(3)

from ai_engine.brain.run import process
from ai_engine.brain.material_grade import apply_material_grade

NAMES = ["after_pool2_gd09_783A9534.jpg", "20260703-DSC1132.jpg", "k001_DSC6441.jpg"]


def panel(im_u8, label, w=700):
    h = int(round(im_u8.shape[0] * w / im_u8.shape[1]))
    r = np.ascontiguousarray(cv2.resize(im_u8, (w, h)))
    cv2.rectangle(r, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(r, label, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return r


def main():
    os.makedirs("outputs/material_v2", exist_ok=True)
    for n in NAMES:
        img = cv2.imread(f"data/pairs/before/{n}").astype(np.float32) / 255.0
        base, _ = process(img)
        log = []
        final = apply_material_grade(base, record=log)
        tags = " | ".join("{}({})".format(s["op"], s["frac"]) for s in log)
        print(n, "|", tags, flush=True)
        t = cv2.imread(f"data/pairs/after/{n}")
        if t.shape[:2] != final.shape[:2]:
            t = cv2.resize(t, (final.shape[1], final.shape[0]))
        f8 = (final * 255).clip(0, 255).astype(np.uint8)
        hh = min(panel(f8, "").shape[0], panel(t, "").shape[0])
        cv2.imwrite(f"outputs/material_v2/v2_{n}",
                    np.hstack([panel(f8, "STACK v2")[:hh], panel(t, "TARGET")[:hh]]),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print("done")


if __name__ == "__main__":
    main()
