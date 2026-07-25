"""window_dehaze — KHỬ MÙ/LÓA cảnh ngoài cửa sổ (25/07, Dark Channel Prior).

Vấn đề (chủ chê trên job thật): cảnh ngoài cửa sổ bị PHỦ TRẮNG SỮA (lóa/phản
chiếu trên kính) -> trời/lá/nhà nhìn không ra. Đây KHÔNG phải cháy sáng (kéo
sáng không cứu) mà là HAZE — cần khử lớp trắng.

Cách làm (He et al. Dark Channel Prior — chuẩn ngành, deterministic):
  dark channel J^dark = min_c min_local I_c ; vùng haze thì dark channel CAO.
  airlight A = màu của lớp mù (percentile cao của vùng sáng+xám).
  transmission t = 1 - w * darkchannel/A ; ảnh khử mù = (I - A)/t + A.
Sau đó phục hồi độ bão hòa (cảnh ngoài sau khử mù thường nhạt màu).

apply(img f32 [0,1] BGR, params) -> cùng shape. strength 0..1 (default 0.7).
Op chạy TOÀN ẢNH; nên gọi qua region_apply với mask cửa sổ (chỉ khử vùng kính).
strength=0 -> identity.
"""
import cv2
import numpy as np

cv2.setNumThreads(3)

_OMEGA = 0.85       # giữ lại chút mù cho tự nhiên (không khử 100%)
_PATCH = 15
_T0 = 0.1          # sàn transmission (tránh chia 0 -> nhiễu)


def _dark_channel(img, patch):
    mn = np.min(img, axis=2)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    return cv2.erode(mn, k)


def _airlight(img, dark):
    h, w = dark.shape
    n = max(1, int(h * w * 0.001))
    idx = np.argpartition(dark.ravel(), -n)[-n:]        # 0.1% pixel dark-channel cao nhất
    flat = img.reshape(-1, 3)
    return flat[idx].max(axis=0)                          # màu sáng nhất trong số đó = airlight


def apply(img, params=None):
    params = params or {}
    strength = float(np.clip(params.get("strength", 0.7), 0.0, 1.0))
    img = np.clip(np.asarray(img, dtype=np.float32), 1e-4, 1.0)
    if strength == 0.0:
        return img

    dark = _dark_channel(img, _PATCH)
    A = np.clip(_airlight(img, dark), 0.3, 1.0)          # airlight không quá tối

    # transmission
    norm = img / A.reshape(1, 1, 3)
    t = 1.0 - _OMEGA * _dark_channel(norm, _PATCH)
    t = cv2.GaussianBlur(t, (0, 0), 8)                   # làm mượt (thay guided filter)
    t = np.clip(t, _T0, 1.0)[..., None]

    recovered = (img - A) / t + A
    recovered = np.clip(recovered, 0.0, 1.0)

    # phục bão hòa (khử mù xong hay nhạt màu -> cảnh ngoài có màu lại)
    hsv = cv2.cvtColor(recovered, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.25, 0, 1)
    recovered = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    out = img * (1.0 - strength) + recovered * strength
    return np.clip(out, 0.0, 1.0).astype(np.float32)
