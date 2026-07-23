"""
Con "KHU NHIEU + PHUC NET" (deterministic, edge-aware). Xem tasks/08-denoise-sharpen.md.

HOP DONG OPERATOR:
    apply(img: np.ndarray float32 [0,1] HxWx3 BGR, params: dict) -> np.ndarray cung shape.
Khong resize / re-encode.
"""

import cv2
import numpy as np

cv2.setNumThreads(2)

_HAS_GUIDED = hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter")

MAX_OVERSHOOT = 0.04
HIGHLIGHT_THRESH = 0.92  # vung bao hoa/chay: khong sharpen


def _luma(img):
    return 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]


def _flat_region_noise_std(gray):
    """
    Uoc luong do lech chuan nhieu tren vung PHANG (gradient thap) bang high-pass
    (gray - blur nhe), loai canh de khong lan voi chi tiet thuc.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)
    flat_mask = grad_mag < np.percentile(grad_mag, 40)
    if flat_mask.sum() < 200:
        flat_mask = np.ones_like(gray, dtype=bool)
    high_freq = gray - blur
    return float(np.std(high_freq[flat_mask])), flat_mask


def _low_freq(img, radius=8, eps=0.02 ** 2):
    """Loc giu-canh (guided filter tren chinh anh la guide; fallback bilateral)."""
    if _HAS_GUIDED:
        return cv2.ximgproc.guidedFilter(img, img, radius, eps).astype(np.float32)
    d = radius * 2 + 1
    sigma_color = float(np.sqrt(eps))
    return cv2.bilateralFilter(img, d, sigma_color, radius).astype(np.float32)


def denoise(img, strength=0.35):
    """
    Tach tan so: low-freq = guided/bilateral filter (giu-canh); high-freq giu lai
    theo ty le (1 - strength * mask_nhieu). mask_nhieu = 1 o vung phang (nhieu ro,
    giam manh), thap o vung canh (giu chi tiet, khong lam mem canh).

    img: float32 [0,1] HxWx3 BGR. strength: 0..1 (default 0.35).
    Tra ve anh cung shape.
    """
    img = img.astype(np.float32)
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return img.copy()

    h, w = img.shape[:2]
    gray = _luma(img).astype(np.float32)

    noise_std, _flat_mask = _flat_region_noise_std(gray)

    low = _low_freq(img, radius=8, eps=0.02 ** 2)
    high = img - low

    # Mask nhieu: giu chi tiet manh o canh (gradient cao), giam manh o vung phang.
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)
    grad_norm = grad_mag / (np.percentile(grad_mag, 95) + 1e-6)
    grad_norm = np.clip(grad_norm, 0.0, 1.0)
    # edge_keep: 1 tai canh manh (giu nguyen high-freq), 0 tai vung phang (giam nhieu manh)
    edge_keep = grad_norm ** 0.5
    edge_keep = cv2.GaussianBlur(edge_keep.astype(np.float32), (5, 5), 0)

    keep_ratio = edge_keep + (1.0 - edge_keep) * (1.0 - strength)
    keep_ratio = np.clip(keep_ratio, 0.0, 1.0)[:, :, None]

    out = low + high * keep_ratio
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def sharpen(img, amount=0.3, radius=1.5):
    """
    Unsharp mask co chong halo: gioi han overshoot +-MAX_OVERSHOOT, khong sharpen
    vung da bao hoa/chay (luma > HIGHLIGHT_THRESH), va khong khuech dai nhieu o
    vung phang (texture-gate theo gradient) de khong "hoan tac" ket qua denoise.

    img: float32 [0,1] HxWx3 BGR. amount: 0..1 (default 0.3). radius: sigma blur (px).
    Tra ve anh cung shape.
    """
    img = img.astype(np.float32)
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0.0:
        return img.copy()

    radius = max(0.3, float(radius))
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    detail = img - blurred

    boosted = img + detail * (amount * 2.0)

    # Gioi han overshoot: khong cho pixel lech qua +-MAX_OVERSHOOT so voi ban goc.
    delta = np.clip(boosted - img, -MAX_OVERSHOOT, MAX_OVERSHOOT)

    # Khong sharpen vung da bao hoa/chay (highlight) de tranh khuech dai noise/halo tren vung chay.
    luma = _luma(img)
    highlight_mask = (luma > HIGHLIGHT_THRESH).astype(np.float32)
    highlight_mask = cv2.GaussianBlur(highlight_mask, (5, 5), 0)[:, :, None]
    delta = delta * (1.0 - highlight_mask)

    # Texture gate: unsharp mask ban chat khuech dai MOI tan so cao, ke ca nhieu con
    # sot lai o vung phang sau denoise. Chi cho boost di qua o vung co cau truc that
    # (gradient ro), giam dan ve 0 o vung phang -> khong tai-khuech-dai nhieu.
    gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)
    texture_gate = np.clip(grad_mag / (np.percentile(grad_mag, 90) + 1e-6), 0.0, 1.0)
    texture_gate = cv2.GaussianBlur(texture_gate.astype(np.float32), (3, 3), 0)[:, :, None]
    delta = delta * texture_gate

    out = img + delta
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def apply(img, params=None):
    """
    HOP DONG OPERATOR: apply(img float32 [0,1] HxWx3 BGR, params) -> cung shape.
    params: denoise_strength (default 0.35), sharpen_amount (default 0.3),
            sharpen_radius (default 1.5).
    Denoise truoc, sharpen sau.
    """
    params = params or {}
    denoise_strength = float(params.get("denoise_strength", 0.35))
    sharpen_amount = float(params.get("sharpen_amount", 0.3))
    sharpen_radius = float(params.get("sharpen_radius", 1.5))

    assert img.dtype == np.float32 or img.dtype == np.float64
    h, w = img.shape[:2]

    out = denoise(img, denoise_strength)
    out = sharpen(out, sharpen_amount, sharpen_radius)

    assert out.shape == img.shape, "Kich thuoc output phai khop input"
    return out.astype(np.float32)


if __name__ == "__main__":
    print("Denoise/Sharpen module loaded. guided_filter available:", _HAS_GUIDED)
