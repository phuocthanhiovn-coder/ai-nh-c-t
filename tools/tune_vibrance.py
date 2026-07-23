"""Tune vibrance tren 4 vung chu du an khoanh (k001) + kiem tra 2 anh khac khong vo."""
import cv2
import numpy as np

cv2.setNumThreads(3)

from ai_engine.orchestrator.registry import REGISTRY
from ai_engine.specialists.vibrance import vib

REGIONS = [
    ("gach op", 380, 560, 560, 300, "sat", 36.3),
    ("tuong", 1280, 120, 500, 320, "luma", 184.7),
    ("khung cua", 1380, 380, 560, 500, "luma", 154.8),
    ("san+lo", 820, 1000, 620, 360, "sat", 56.1),
]
GRID = [
    dict(whites=0.4, vibrance=0.4),
    dict(whites=0.6, vibrance=0.6),
    dict(whites=0.8, vibrance=0.8),
    dict(whites=0.6, vibrance=0.9),
]


def region_stat(img_u8, x, y, w, h, kind):
    c = img_u8[y:y+h, x:x+w]
    if kind == "luma":
        return cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).mean()
    return cv2.cvtColor(c, cv2.COLOR_BGR2HSV)[..., 1].mean()


def main():
    fn = REGISTRY["auto_enhance"]["fn"]
    dn = REGISTRY["denoise"]["fn"]
    b = cv2.imread("data/pairs/before/k001_DSC6441.jpg").astype(np.float32) / 255.0
    base = fn(dn(b, {"denoise_strength": 0.35, "sharpen_amount": 0.0}), {})

    print("== k001, muc tieu tung vung ==")
    for g in GRID:
        out = vib.apply(base, g)
        u8 = (out * 255).astype(np.uint8)
        vals = [region_stat(u8, x, y, w, h, kind) for (_, x, y, w, h, kind, _) in REGIONS]
        tag = " ".join(f"{n}={v:6.1f}/{t}" for (n, *_r, t), v in zip(
            [(r[0], r[6]) for r in REGIONS], vals))
        # gon: in thu cong
        print(f"w={g['whites']} v={g['vibrance']}: " + " | ".join(
            f"{REGIONS[i][0]} {vals[i]:6.1f} (dich {REGIONS[i][6]})" for i in range(4)))

    print("\n== kiem tra 2 anh khac (global luma/sat truoc->sau, muc g chon) ==")
    pick = dict(whites=0.6, vibrance=0.6)
    for n in ["_ML_1421.jpg", "after_pool2_gd09_783A9534.jpg"]:
        bb = cv2.imread(f"data/pairs/before/{n}").astype(np.float32) / 255.0
        bs = fn(dn(bb, {"denoise_strength": 0.35, "sharpen_amount": 0.0}), {})
        vv = vib.apply(bs, pick)
        for tag, im in [("model", bs), ("model+vib", vv)]:
            u8 = (im * 255).astype(np.uint8)
            l = cv2.cvtColor(u8, cv2.COLOR_BGR2GRAY).mean()
            s = cv2.cvtColor(u8, cv2.COLOR_BGR2HSV)[..., 1].mean()
            print(f"  {n} {tag:<10} luma={l:6.1f} sat={s:5.1f}")
        t = cv2.imread(f"data/pairs/after/{n}")
        print(f"  {n} TARGET     luma={cv2.cvtColor(t,cv2.COLOR_BGR2GRAY).mean():6.1f} "
              f"sat={cv2.cvtColor(t,cv2.COLOR_BGR2HSV)[...,1].mean():5.1f}")


if __name__ == "__main__":
    main()
