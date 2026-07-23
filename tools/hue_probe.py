"""Do saturation theo NHOM HUE (am vs khong-am) — ta vs target, vung gach + san k001."""
import cv2
import numpy as np

cv2.setNumThreads(3)

from ai_engine.orchestrator.registry import REGISTRY

WARM_LO, WARM_HI = 5, 25  # hue OpenCV 0..180: cam-do go/terracotta


def probe(u8, x, y, w, h, tag):
    c = u8[y:y+h, x:x+w]
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hch, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(float), hsv[..., 2]
    colored = s > 20  # bo pixel trang/trung tinh
    warm = colored & (hch >= WARM_LO) & (hch <= WARM_HI)
    cool = colored & ~warm
    def st(m):
        return (s[m].mean() if m.sum() else 0.0, 100.0 * m.mean())
    sw, pw = st(warm)
    sc, pc = st(cool)
    print(f"  {tag:<10} am: sat={sw:5.1f} ({pw:4.1f}% px) | khong-am: sat={sc:5.1f} ({pc:4.1f}% px)")


def main():
    fn = REGISTRY["auto_enhance"]["fn"]
    dn = REGISTRY["denoise"]["fn"]
    b = cv2.imread("data/pairs/before/k001_DSC6441.jpg").astype(np.float32) / 255.0
    ours = (fn(dn(b, {"denoise_strength": 0.35, "sharpen_amount": 0.0}), {}) * 255).astype(np.uint8)
    tgt = cv2.imread("data/pairs/after/k001_DSC6441.jpg")
    if tgt.shape[:2] != ours.shape[:2]:
        tgt = cv2.resize(tgt, (ours.shape[1], ours.shape[0]))
    print("VUNG GACH OP (380,560 560x300):")
    probe(ours, 380, 560, 560, 300, "TA")
    probe(tgt, 380, 560, 560, 300, "TARGET")
    print("VUNG SAN (820,1000 620x360):")
    probe(ours, 820, 1000, 620, 360, "TA")
    probe(tgt, 820, 1000, 620, 360, "TARGET")


if __name__ == "__main__":
    main()
