"""finish_detail — con hoan thien NET + VI TUONG PHAN + DIEM DEN (22/07/2026).

VI SAO CO CON NAY: khach che 3 lan "anh mo bot" (14/07, 16/07, 17/07 + chu du an 22/07).
Chan doan (outputs/diagnose_blur): output engine co nang luong canh (Laplacian var)
chi bang ~1/8..1/12 anh AutoHDR target, local contrast thap hon 10-35%. AutoHDR co
buoc phuc net + micro-contrast + ha den rat manh; pipeline ta chua co (op sharpen
cu qua nhe). Con nay lam DUNG 1 viec do.

CACH LAM (deterministic, khong model, khong halo):
  1. Tach lop chi tiet bang GUIDED FILTER (edge-preserving base -> khong halo
     quanh canh manh nhu unsharp Gaussian; bai hoc task 22).
  2. Hai tang: clarity (radius ~2%% canh ngan, vi tuong phan mang lon) +
     detail (radius nho, van go/texture). Detail co soft-gate d^2/(d^2+k)
     de khong khuech dai nhieu vung phang.
  3. Boost tren LUMA roi nhan ty le vao BGR -> khong lech mau.
  4. Diem den: keo p0.5 luma ve gan 0 (anchor CO DINH, gioi han 0.05) -> den sau,
     het "bot". strength=0 -> tra ve anh goc bit-identical.

Hop dong: apply(img float32 [0,1] HxWx3 BGR, params) -> cung shape.
Params: clarity 0..1 (default 0.5) · detail 0..1 (default 0.6) · black 0..1 (default 0.35)
"""
import cv2
import numpy as np

cv2.setNumThreads(3)

# Trong so luma BT.709 theo thu tu BGR.
_LUMA_W = np.array([0.0722, 0.7152, 0.2126], dtype=np.float32)

# Radius theo ty le canh ngan (clarity) va co dinh (detail). eps cua guided filter
# tren luma [0,1]: nho -> bam canh chat (it halo), qua nho -> boost ca nhieu.
_CLARITY_RADIUS_FRAC = 0.02
_CLARITY_EPS = 0.03 ** 2
_DETAIL_RADIUS = 3
_DETAIL_EPS = 0.015 ** 2

# He so gain toi da (tai clarity=1 / detail=1) — chon bang tune tren cap thuc te
# (tools/tune_finish.py) roi NHIN crop 100%.
_CLARITY_GAIN = 1.6
_DETAIL_GAIN = 2.6
_DETAIL_NOISE_K = 0.0004        # soft-gate: d*d/(d*d+k) ~0 voi |d|<0.01 (nhieu), ~1 voi |d|>0.06
_BLACK_CAP = 0.05               # anchor CO DINH: khong bao gio keo qua muc nay (task 22)


def _guided(y, radius, eps):
    r = max(2, int(radius))
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
        return cv2.ximgproc.guidedFilter(guide=y, src=y, radius=r, eps=eps)
    # Fallback O(N) guided filter tu viet (khong co ximgproc).
    ksize = (2 * r + 1, 2 * r + 1)
    mean_y = cv2.boxFilter(y, -1, ksize)
    mean_yy = cv2.boxFilter(y * y, -1, ksize)
    var_y = np.maximum(mean_yy - mean_y * mean_y, 0.0)
    a = var_y / (var_y + eps)
    b = mean_y - a * mean_y
    mean_a = cv2.boxFilter(a, -1, ksize)
    mean_b = cv2.boxFilter(b, -1, ksize)
    return mean_a * y + mean_b


def apply(img, params=None):
    params = params or {}
    clarity = float(np.clip(params.get("clarity", 0.5), 0.0, 1.0))
    detail = float(np.clip(params.get("detail", 0.6), 0.0, 1.0))
    black = float(np.clip(params.get("black", 0.35), 0.0, 1.0))

    img = np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)
    if clarity == 0.0 and detail == 0.0 and black == 0.0:
        return img

    h, w = img.shape[:2]
    y = img @ _LUMA_W  # HxW float32

    y_new = y
    if clarity > 0.0:
        base = _guided(y_new, _CLARITY_RADIUS_FRAC * min(h, w), _CLARITY_EPS)
        y_new = y_new + (clarity * _CLARITY_GAIN) * (y_new - base)

    if detail > 0.0:
        base_f = _guided(y_new, _DETAIL_RADIUS, _DETAIL_EPS)
        d = y_new - base_f
        gate = (d * d) / (d * d + _DETAIL_NOISE_K)
        y_new = y_new + (detail * _DETAIL_GAIN) * d * gate

    y_new = np.clip(y_new, 1e-4, 1.5)

    # Nhan ty le luma vao ca 3 kenh -> giu mau. Gioi han ty le de khong no vung toi.
    ratio = np.clip(y_new / np.maximum(y, 1e-4), 0.25, 4.0)
    out = img * ratio[..., None]

    if black > 0.0:
        y_out = out @ _LUMA_W
        p_low = float(np.percentile(y_out, 0.5))
        b0 = black * min(p_low, _BLACK_CAP)
        if b0 > 1e-5:
            out = (out - b0) / (1.0 - b0)

    return np.clip(out, 0.0, 1.0).astype(np.float32)
