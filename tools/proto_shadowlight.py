"""Prototype 23/07: THAP SANG vung toi/goc (flambient-style) tren output v6.
Gia thuyet: cai 'nhat' chu du an thay = thieu anh sang goc, khong phai thieu mau
(chroma Lab ta >= target). Mo bong theo luma-band, giu mau bang ratio, khong halo
(per-pixel + guided base cho local contrast giu nguyen).
"""
import cv2
import numpy as np

cv2.setNumThreads(3)

_LUMA_W = np.array([0.0722, 0.7152, 0.2126], dtype=np.float32)


def _smoothstep(x, lo, hi):
    t = np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def shadow_light(img, amount=0.6):
    img = np.clip(img.astype(np.float32), 0, 1)
    y = img @ _LUMA_W
    # band bong toi 0.04..0.45: duoi 0.04 la den that (giu den sau), tren 0.45 khong dung
    w = _smoothstep(y, 0.04, 0.18) * (1.0 - _smoothstep(y, 0.30, 0.50))
    # gain toi da ~ +65% o giua band tai amount=1
    gain = 1.0 + amount * 0.65 * w
    out = img * gain[..., None]
    return np.clip(out, 0, 1)


def main():
    for n in ["k001_DSC6441.jpg", "after_pool2_gd09_783A9534.jpg"]:
        d = cv2.imread(f"outputs/compare_chf/v6_{n}").astype(np.float32) / 255.0
        t = cv2.imread(f"data/pairs/after/{n}")
        s = shadow_light(d, 0.7)
        if t.shape[:2] != d.shape[:2]:
            t = cv2.resize(t, (d.shape[1], d.shape[0]))
        def p(img, label, w=640):
            if img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            h = int(round(img.shape[0] * w / img.shape[1]))
            r = np.ascontiguousarray(cv2.resize(img, (w, h)))
            cv2.rectangle(r, (0, 0), (w, 38), (0, 0, 0), -1)
            cv2.putText(r, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return r
        hh = min(p(d, "").shape[0], p(s, "").shape[0], p(t, "").shape[0])
        row = np.hstack([p(d, "v6 (hien tai)")[:hh], p(s, "v6 + THAP SANG GOC")[:hh], p(t, "TARGET")[:hh]])
        cv2.imwrite(f"outputs/compare_chf/proto_light_{n}", row, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        print("saved", n)


if __name__ == "__main__":
    main()
