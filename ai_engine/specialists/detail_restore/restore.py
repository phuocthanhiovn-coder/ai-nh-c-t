"""detail_restore — PHUC NET/CHI TIET THAT bang Real-ESRGAN general (24/07).

Op: chay SR x4 (SRVGGNetCompact) roi thu nho ve ĐUNG size goc = mot lop phuc
hoi chi tiet/khu nen, THEM chi tiet that ma op tuyen tinh khong lam duoc. Blend
theo strength de tranh "nhua" (plasticky). Tile de khong bung RAM voi anh lon.

Hop dong: apply(img f32 [0,1] HxWx3 BGR, params) -> cung shape.
Params: strength 0..1 (default 0.5) · denoise 0..1 (default 0.5, cho model general).
strength=0 -> bit-identical. Thieu weights -> tra anh goc (khong crash).
"""
import os

import cv2
import numpy as np

cv2.setNumThreads(3)

_CKPT = "checkpoints/ext/realesr-general-x4v3.pth"
_TILE = 512          # xu ly tung o 512 (input) de gioi han RAM
_PAD = 16
_model = None
_failed = False


def _load():
    global _model, _failed
    if _failed:
        return None
    if _model is None:
        try:
            import torch
            from ai_engine.specialists.detail_restore.srvgg import SRVGGNetCompact
            torch.set_num_threads(3)
            if not os.path.exists(_CKPT):
                raise FileNotFoundError(_CKPT)
            sd = torch.load(_CKPT, map_location="cpu")
            sd = sd.get("params", sd.get("params_ema", sd))
            m = SRVGGNetCompact(num_feat=64, num_conv=32, upscale=4)
            m.load_state_dict(sd, strict=True)
            m.eval()
            _model = m
        except Exception as exc:
            print(f"[WARN] detail_restore: khong nap duoc weights ({exc}). Bo qua.")
            _failed = True
            return None
    return _model


def _sr_tile(model, bgr01):
    """Chay SR x4 tren 1 o (bgr float32 [0,1]) -> x4."""
    import torch
    t = torch.from_numpy(bgr01[:, :, ::-1].transpose(2, 0, 1).copy()).unsqueeze(0).float()
    with torch.no_grad():
        out = model(t).clamp(0, 1)
    out = out.squeeze(0).numpy().transpose(1, 2, 0)[:, :, ::-1]  # RGB->BGR
    return out


def apply(img, params=None):
    params = params or {}
    strength = float(np.clip(params.get("strength", 0.5), 0.0, 1.0))
    img = np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)
    if strength == 0.0:
        return img
    model = _load()
    if model is None:
        return img

    h, w = img.shape[:2]
    up = np.zeros((h * 4, w * 4, 3), dtype=np.float32)
    for y0 in range(0, h, _TILE):
        for x0 in range(0, w, _TILE):
            y1, x1 = min(y0 + _TILE, h), min(x0 + _TILE, w)
            py0, px0 = max(0, y0 - _PAD), max(0, x0 - _PAD)
            py1, px1 = min(h, y1 + _PAD), min(w, x1 + _PAD)
            tile = img[py0:py1, px0:px1]
            sr = _sr_tile(model, tile)
            # cat bo phan pad (theo he so x4)
            cy0, cx0 = (y0 - py0) * 4, (x0 - px0) * 4
            up[y0 * 4:y1 * 4, x0 * 4:x1 * 4] = sr[cy0:cy0 + (y1 - y0) * 4,
                                                  cx0:cx0 + (x1 - x0) * 4]
    # thu nho ve dung size goc (INTER_AREA giu chi tiet)
    restored = cv2.resize(up, (w, h), interpolation=cv2.INTER_AREA)
    out = img * (1.0 - strength) + restored * strength
    return np.clip(out, 0.0, 1.0).astype(np.float32)
