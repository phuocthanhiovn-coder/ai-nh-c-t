"""
Sky asset library — procedural plates (NO image-gen API, numpy only per CLAUDE.md rule 4).

ensure_skies() -> dict{name: path}
    If assets/skies/*.jpg already has >=3 plates, just return the existing ones.
    Otherwise generate 4 plates (blue/golden/dusk/hazy) at 2400x1600, deterministic
    (seeded per-plate), save as JPEG q95.
"""

import os

import cv2
import numpy as np

cv2.setNumThreads(2)

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "skies")
ASSETS_DIR = os.path.normpath(ASSETS_DIR)

PLATE_W, PLATE_H = 2400, 1600
MIN_EXISTING = 3

PLATE_NAMES = ("blue", "golden", "dusk", "hazy")


def _seed_for(name):
    return hash(name) & 0xFFFF


def _lowfreq_noise(rng, h, w, octaves=(8, 16, 32)):
    """Sum of a few small random fields resized up + blurred -> smooth cloud-like field [0,1]."""
    acc = np.zeros((h, w), dtype=np.float32)
    for oct_size in octaves:
        small = rng.random((oct_size, oct_size)).astype(np.float32)
        big = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        acc += big
    acc /= float(len(octaves))
    acc = cv2.GaussianBlur(acc, (0, 0), sigmaX=w / 40.0)
    lo, hi = float(acc.min()), float(acc.max())
    if hi - lo < 1e-6:
        return np.zeros((h, w), dtype=np.float32)
    return (acc - lo) / (hi - lo)


def _vertical_gradient(h, w, top_bgr, bottom_bgr):
    top = np.array(top_bgr, dtype=np.float32)
    bottom = np.array(bottom_bgr, dtype=np.float32)
    t = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(-1, 1, 1)
    grad = top.reshape(1, 1, 3) * (1.0 - t) + bottom.reshape(1, 1, 3) * t
    return np.repeat(grad, w, axis=1).astype(np.float32)


def _add_clouds(img, rng, alpha_max, n_blobs, horizon_bias=0.0, thin=False):
    """Alpha-blend soft white/warm cloud blobs onto img (float32 [0,1] BGR) using
    low-frequency noise fields, thresholded soft."""
    h, w = img.shape[:2]
    out = img.copy()
    for _ in range(n_blobs):
        field = _lowfreq_noise(rng, h, w, octaves=(6, 12, 24))
        thresh = 0.62 if thin else 0.55
        soft = np.clip((field - thresh) / (1.0 - thresh), 0.0, 1.0)
        soft = soft ** 1.5
        if horizon_bias > 0:
            yy = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(-1, 1)
            bias = 1.0 - horizon_bias + horizon_bias * yy
            soft = soft * bias
        soft = cv2.GaussianBlur(soft, (0, 0), sigmaX=w / 150.0)
        alpha = (soft * alpha_max)[:, :, None]
        white = np.ones_like(out)
        out = out * (1.0 - alpha) + white * alpha
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _make_blue(rng):
    img = _vertical_gradient(PLATE_H, PLATE_W, (0.75, 0.45, 0.15), (0.95, 0.85, 0.70))
    img = _add_clouds(img, rng, alpha_max=0.35, n_blobs=3, horizon_bias=0.3)
    return img


def _make_golden(rng):
    img = _vertical_gradient(PLATE_H, PLATE_W, (0.55, 0.30, 0.20), (0.55, 0.70, 0.98))
    img = _add_clouds(img, rng, alpha_max=0.22, n_blobs=2, horizon_bias=0.6, thin=True)
    return img


def _make_dusk(rng):
    img = _vertical_gradient(PLATE_H, PLATE_W, (0.30, 0.10, 0.10), (0.35, 0.25, 0.55))
    img = _add_clouds(img, rng, alpha_max=0.15, n_blobs=2, horizon_bias=0.8, thin=True)
    return img


def _make_hazy(rng):
    img = _vertical_gradient(PLATE_H, PLATE_W, (0.80, 0.72, 0.62), (0.93, 0.90, 0.86))
    img = _add_clouds(img, rng, alpha_max=0.18, n_blobs=3, horizon_bias=0.4, thin=True)
    return img


_BUILDERS = {
    "blue": _make_blue,
    "golden": _make_golden,
    "dusk": _make_dusk,
    "hazy": _make_hazy,
}


def _existing_plates():
    found = {}
    if not os.path.isdir(ASSETS_DIR):
        return found
    for name in PLATE_NAMES:
        path = os.path.join(ASSETS_DIR, f"{name}.jpg")
        if os.path.isfile(path):
            found[name] = path
    return found


def ensure_skies():
    """Return dict{name: path} for the 4 sky plates. Generates them procedurally
    (deterministic, seeded) if fewer than MIN_EXISTING already exist on disk."""
    existing = _existing_plates()
    if len(existing) >= MIN_EXISTING:
        return existing

    os.makedirs(ASSETS_DIR, exist_ok=True)
    result = {}
    for name in PLATE_NAMES:
        path = os.path.join(ASSETS_DIR, f"{name}.jpg")
        if os.path.isfile(path):
            result[name] = path
            continue
        rng = np.random.default_rng(_seed_for(name))
        plate = _BUILDERS[name](rng)
        out_u8 = np.clip(plate * 255.0 + 0.5, 0, 255).astype(np.uint8)
        ok = cv2.imwrite(path, out_u8, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise IOError(f"Failed to write sky plate: {path}")
        result[name] = path
    return result


if __name__ == "__main__":
    paths = ensure_skies()
    for name, path in paths.items():
        print(name, "->", path)
