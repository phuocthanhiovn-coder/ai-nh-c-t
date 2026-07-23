"""brain.doses — NAO v2: doan LIEU op tu features anh (hoc tu 900+ cap, 25/07 dem).

predict(feats9) -> dict {sl, wh, vb, dc, bk} da clamp ve dai an toan.
Thieu file predictor -> tra None (brain dung cong thuc tay nhu cu).
"""
import json
import os

import numpy as np

_PATH = "checkpoints/dose_predictor.json"
_CLAMP = {"sl": (0.0, 0.85), "wh": (0.0, 0.9), "vb": (0.0, 1.0),
          "dc": (0.0, 0.7), "bk": (0.3, 0.9)}
_cache = None


def _load():
    global _cache
    if _cache is None and os.path.exists(_PATH):
        _cache = json.load(open(_PATH, encoding="utf-8"))
    return _cache


def predict(feats9):
    p = _load()
    if p is None:
        return None
    x = (np.asarray(feats9, dtype=np.float64) - p["mu"]) / p["sd"]
    y = np.append(x, 1.0) @ np.array(p["W"])
    out = {}
    for c, v in zip(p["dose_cols"], y):
        lo, hi = _CLAMP[c]
        out[c] = round(float(np.clip(v, lo, hi)), 2)
    return out


def image_feats(u8):
    """9 features — PHAI khop thu tu voi tools/mine_doses.py."""
    import cv2
    g = cv2.cvtColor(u8, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(u8, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    C = np.sqrt((lab[..., 1] - 128) ** 2 + (lab[..., 2] - 128) ** 2)
    dark = g < 80
    return [float(np.percentile(g, 5)), float(np.percentile(g, 25)),
            float(np.percentile(g, 50)), float(np.percentile(g, 75)),
            float(np.percentile(g, 95)), float(dark.mean() * 100),
            float(hsv[..., 1][dark].mean()) if dark.sum() else 0.0,
            float(hsv[..., 1].mean()), float(C.mean())]
