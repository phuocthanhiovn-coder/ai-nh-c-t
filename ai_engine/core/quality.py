"""
ai_engine.core.quality
=======================
Thu vien LOI giu chat luong dung chung cho moi specialist (Task 13).

Nguyen tac (xem CLAUDE.md muc 2): AI xuat OPERATOR, khong xuat pixel cuoi.
Cac ham o day la CONG CU DETERMINISTIC de ap operator len anh full-res
ma khong lam mat net / doi size / nen be.

Quy uoc: anh la numpy float32, BGR (OpenCV convention), gia tri [0,1]
tru khi noi khac trong docstring. Khong ham nao trong file nay tu y
downscale hoac doi kich thuoc anh dau vao (ngoai guided_upsample/
composite_mask, von co nhiem vu upsample ve dung size cua guide).
"""

import os

import cv2
import numpy as np

cv2.setNumThreads(2)


# ---------------------------------------------------------------------------
# 1. sRGB <-> linear
# ---------------------------------------------------------------------------

def to_linear(img_srgb):
    """sRGB [0,1] -> linear-light [0,1], piecewise dung cong thuc chuan (khong xap xi gamma 2.2)."""
    x = np.asarray(img_srgb, dtype=np.float32)
    a = 0.055
    low = x <= 0.04045
    linear = np.where(low, x / 12.92, ((x + a) / (1.0 + a)) ** 2.4)
    return linear.astype(np.float32)


def to_srgb(img_linear):
    """linear-light [0,1] -> sRGB [0,1], piecewise dung cong thuc chuan."""
    x = np.asarray(img_linear, dtype=np.float32)
    x_clamped = np.clip(x, 0.0, None)  # tranh pow() tren so am do sai so tich luy
    a = 0.055
    low = x <= 0.0031308
    srgb = np.where(low, x * 12.92, (1.0 + a) * np.power(x_clamped, 1.0 / 2.4) - a)
    return srgb.astype(np.float32)


# ---------------------------------------------------------------------------
# 2/3. Tach / gop tan so
# ---------------------------------------------------------------------------

def split_frequency(img, sigma):
    """Tach anh thanh (low, high) sao cho low + high == img chinh xac. low = Gaussian blur."""
    img = np.asarray(img, dtype=np.float32)
    low = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    high = img - low
    return low.astype(np.float32), high.astype(np.float32)


def merge_frequency(low, high):
    """Gop lai low+high, CHI clip [0,1] o buoc nay."""
    out = np.asarray(low, dtype=np.float32) + np.asarray(high, dtype=np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# 4. Guided upsample (Joint Bilateral Upsampling)
# ---------------------------------------------------------------------------

def _has_joint_bilateral():
    return hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "jointBilateralFilter")


def guided_upsample(small_map, guide_full):
    """Upsample small_map (1 hoac 3 kenh, res thap) len dung size cua guide_full
    bang Joint Bilateral Upsampling. Bien phai bam canh cua guide, khong nhoe qua canh.

    Neu cv2.ximgproc khong co, fallback bang bilateralFilter tren small_map da
    resize, dung guide (xam) nhu mot kenh phu de dieu huong range-weight.
    """
    small_map = np.asarray(small_map, dtype=np.float32)
    guide_full = np.asarray(guide_full, dtype=np.float32)
    gh, gw = guide_full.shape[:2]

    resized = cv2.resize(small_map, (gw, gh), interpolation=cv2.INTER_LINEAR)
    if resized.ndim == 2:
        resized = resized[:, :, None]
        squeeze_out = True
    else:
        squeeze_out = False

    if guide_full.ndim == 2:
        guide_j = guide_full[:, :, None]
    else:
        guide_j = guide_full

    if _has_joint_bilateral():
        try:
            out = cv2.ximgproc.jointBilateralFilter(
                guide_j.astype(np.float32), resized.astype(np.float32),
                -1, 0.1, 15,
            )
            out = np.asarray(out, dtype=np.float32)
            if out.ndim == 2:
                out = out[:, :, None]
            return out[:, :, 0] if squeeze_out else out
        except cv2.error:
            pass  # fallback ben duoi

    # Fallback: khong co ximgproc -> bilateral tren [map, guide_gray] gop kenh,
    # de range-weight bam theo do tuong dong cua guide (xam quanh canh).
    guide_gray = guide_j.mean(axis=2, keepdims=True).astype(np.float32)
    stacked = np.concatenate([resized, guide_gray], axis=2).astype(np.float32)
    filtered = cv2.bilateralFilter(stacked, d=9, sigmaColor=0.1, sigmaSpace=25)
    if filtered.ndim == 2:
        filtered = filtered[:, :, None]
    out = filtered[:, :, : resized.shape[2]]
    return out[:, :, 0] if squeeze_out else out


# ---------------------------------------------------------------------------
# 5. Composite theo mask
# ---------------------------------------------------------------------------

def composite_mask(base, edited, mask_float01, feather_px):
    """Tron edited vao base theo mask_float01 (feather bang Gaussian blur sigma=feather_px).
    mask duoc guided-upsample truoc neu nho hon base.
    """
    base = np.asarray(base, dtype=np.float32)
    edited = np.asarray(edited, dtype=np.float32)
    mask = np.asarray(mask_float01, dtype=np.float32)

    bh, bw = base.shape[:2]
    if mask.shape[:2] != (bh, bw):
        mask = guided_upsample(mask, base)

    if feather_px and feather_px > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather_px)

    mask = np.clip(mask, 0.0, 1.0)
    if mask.ndim == 2 and base.ndim == 3:
        mask = mask[:, :, None]

    out = base * (1.0 - mask) + edited * mask
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# 6. Ap bien doi mau chi len low-freq
# ---------------------------------------------------------------------------

def apply_color_on_lowfreq(img, color_transform_fn, sigma=8):
    """Tach tan so, ap color_transform_fn CHI len low, bom lai high tu GOC.
    Cac con mau/sang phai goi ham nay de chinh mau ma khong mat net.
    """
    img = np.asarray(img, dtype=np.float32)
    low, high = split_frequency(img, sigma)
    low_transformed = np.asarray(color_transform_fn(low), dtype=np.float32)
    return merge_frequency(low_transformed, high)


# ---------------------------------------------------------------------------
# 7. Doc / ghi anh
# ---------------------------------------------------------------------------

def read_image_16(path):
    """Doc moi dinh dang ve float32 [0,1] (giu 16-bit neu nguon co)."""
    path = str(path)
    img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {path}")
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
    return img


def write_image(path, img, quality=95):
    """Ghi JPEG (q=quality, 8-bit) hoac PNG (16-bit) theo duoi file. Assert khong doi size."""
    path = str(path)
    img = np.asarray(img, dtype=np.float32)
    h, w = img.shape[:2]
    ext = os.path.splitext(path)[1].lower()

    if ext in (".jpg", ".jpeg"):
        out = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
        ok = cv2.imwrite(path, out, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    elif ext == ".png":
        out = np.clip(img * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
        ok = cv2.imwrite(path, out)
    else:
        raise ValueError(f"Duoi file khong ho tro: {ext}")

    if not ok:
        raise IOError(f"Ghi anh that bai: {path}")

    check = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    assert check is not None and check.shape[:2] == (h, w), (
        f"write_image: size doi khac goc ({check.shape[:2] if check is not None else None} != {(h, w)})"
    )
    return path
