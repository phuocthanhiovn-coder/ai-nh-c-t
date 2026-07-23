"""
Tao 3 mask test THAT tu data/pairs/before/ cho con object_removal (Task 18).

3 anh + toa do vat the duoc CHON BANG MAT (da Read anh + crop-zoom de xac dinh
toa do), hardcode o day - dung theo tinh than spec "Draw the mask programmatically
... hardcode the coords you chose, that's fine for tests".

Chi DOC data/, ghi mask vao outputs/removal_samples/.
"""

import os

import cv2
import numpy as np

cv2.setNumThreads(2)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PAIRS_BEFORE = os.path.join(ROOT, "data", "pairs", "before")
OUT_DIR = os.path.join(ROOT, "outputs", "removal_samples")

# (ten_case, file_anh, kind, shape) - toa do duoc chon bang mat qua Read tool +
# crop-zoom kiem tra. kind="rect" -> [(x0,y0,x1,y1),...]; kind="ellipse" ->
# (cx,cy,ax,ay) de bam sat dang tron/oval cua vat the (rect se an qua nhieu
# background xung quanh mot vat the tron).
CASES = [
    (
        "switch_plate",
        "20260703-DSC1105.jpg",
        "rect",
        [(806, 818, 840, 854)],
    ),
    (
        "marquee_letter",
        "20260703-DSC1226.jpg",
        "rect",
        [(1378, 195, 1532, 458)],
    ),
    (
        # bowl hinh chom cau: 1 ellipse lon bam theo toan bo silhouette (mieng
        # toi day), tam ha thap hon mieng bat de trum het phan day toi ban.
        "decor_bowl",
        "20260703-DSC1161.jpg",
        "ellipse2",
        [(743, 1185, 148, 95)],
    ),
]


def build_mask(shape_hw, kind, shape):
    mask = np.zeros(shape_hw, dtype=np.uint8)
    if kind == "rect":
        for (x0, y0, x1, y1) in shape:
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)
    elif kind == "ellipse":
        cx, cy, ax, ay = shape
        cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, thickness=-1)
    elif kind == "ellipse2":
        for (cx, cy, ax, ay) in shape:
            cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, thickness=-1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    else:
        raise ValueError(f"kind khong ho tro: {kind}")
    return mask


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, fname, kind, shape in CASES:
        src_path = os.path.join(PAIRS_BEFORE, fname)
        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[!] Khong doc duoc: {src_path}")
            continue
        h, w = img.shape[:2]
        mask = build_mask((h, w), kind, shape)
        n_px = int(cv2.countNonZero(mask))

        mask_path = os.path.join(OUT_DIR, f"mask_{name}.png")
        cv2.imwrite(mask_path, mask)
        print(f"[{name}] src={fname} size={w}x{h} {kind}={shape} mask_px={n_px} -> {mask_path}")


if __name__ == "__main__":
    main()
