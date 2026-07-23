"""Soi 4 vung chu du an khoanh (23/07) tren k001: GIAO v3 vs TARGET.
Crop phong to + do luma/saturation tung vung."""
import cv2
import numpy as np

cv2.setNumThreads(3)

OURS = "outputs/compare_chf/v3_k001_DSC6441.jpg"
TGT = "data/pairs/after/k001_DSC6441.jpg"
# (ten, x, y, w, h) theo anh goc 2048x1365 — theo vi tri khoanh cam
REGIONS = [
    ("1 gach op + mat ban",    380, 560, 560, 300),
    ("2 tuong tren cua",      1280, 120, 500, 320),
    ("3 khung cua + phong sau",1380, 380, 560, 500),
    ("4 chan lo + san",        820, 1000, 620, 360),
]


def stats(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return g.mean(), hsv[:, :, 1].mean()


def label(im, text):
    im = np.ascontiguousarray(im)
    cv2.rectangle(im, (0, 0), (im.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(im, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return im


def main():
    ours = cv2.imread(OURS)
    tgt = cv2.imread(TGT)
    if tgt.shape[:2] != ours.shape[:2]:
        tgt = cv2.resize(tgt, (ours.shape[1], ours.shape[0]))
    rows = []
    print(f"{'vung':<28} {'luma ta/tgt':>16} {'sat ta/tgt':>16}")
    for (name, x, y, w, h) in REGIONS:
        co = ours[y:y+h, x:x+w]
        ct = tgt[y:y+h, x:x+w]
        lo, so = stats(co)
        lt, st = stats(ct)
        print(f"{name:<28} {lo:6.1f}/{lt:6.1f} {so:8.1f}/{st:6.1f}")
        scale = 700 / w
        co = cv2.resize(co, (700, int(h*scale)))
        ct = cv2.resize(ct, (700, int(h*scale)))
        rows.append(np.hstack([label(co, f"TA — {name}"), label(ct, "TARGET")]))
    maxw = max(r.shape[1] for r in rows)
    sheet = np.vstack([cv2.copyMakeBorder(r, 0, 6, 0, maxw-r.shape[1], cv2.BORDER_CONSTANT) for r in rows])
    cv2.imwrite("outputs/compare_chf/region_check.jpg", sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print("saved outputs/compare_chf/region_check.jpg")


if __name__ == "__main__":
    main()
