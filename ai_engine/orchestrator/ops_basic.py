"""Operator xac dinh (numpy/cv2), float32 [0,1] HxWx3 BGR, full-res, khong resize/re-encode.

Hop dong: apply(img: np.ndarray float32 [0,1] HxWx3 BGR, params: dict) -> np.ndarray cung shape/dtype.
"""
import numpy as np
import cv2

_GAMMA = 2.2


def _clip01(img):
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def brightness(img, params):
    """Exposure +-stops, gamma-aware (linear-light multiply)."""
    amount = float(params.get("amount", 0.0))
    lin = np.power(np.clip(img, 0.0, 1.0), _GAMMA)
    lin = lin * (2.0 ** amount)
    out = np.power(np.clip(lin, 0.0, 1.0), 1.0 / _GAMMA)
    return _clip01(out)


def contrast(img, params):
    """Xoay quanh gia tri trung vi (median luminance)."""
    amount = float(params.get("amount", 0.0))
    factor = max(0.0, 1.0 + amount)
    median = float(np.median(img))
    out = median + (img - median) * factor
    return _clip01(out)


def saturation(img, params):
    """Scale kenh S trong HSV."""
    amount = float(params.get("amount", 0.0))
    factor = max(0.0, 1.0 + amount)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0.0, 1.0)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return _clip01(out)


def temperature(img, params):
    """Dich kenh R/B de am hon (+) hoac lanh hon (-)."""
    amount = float(params.get("amount", 0.0))
    out = img.copy()
    shift = amount * 0.15
    out[..., 2] = out[..., 2] + shift       # R (BGR order index 2) len khi am hon
    out[..., 0] = out[..., 0] - shift       # B giam khi am hon
    return _clip01(out)


def shadows_lift(img, params):
    """Nang vung toi, tone-curve. amount 0..1."""
    amount = float(params.get("amount", 0.0))
    weight = np.clip(1.0 - img * 2.0, 0.0, 1.0)
    out = img + amount * weight * 0.5
    return _clip01(out)


def highlights_recover(img, params):
    """Ha vung chay. amount 0..1."""
    amount = float(params.get("amount", 0.0))
    weight = np.clip((img - 0.5) * 2.0, 0.0, 1.0)
    out = img - amount * weight * 0.5
    return _clip01(out)


def white_balance(img, params):
    """Gray-world white balance, strength 0..1 blend voi anh goc."""
    strength = float(params.get("strength", 1.0))
    means = img.reshape(-1, 3).mean(axis=0)
    means = np.clip(means, 1e-6, None)
    target = float(means.mean())
    scale = target / means
    scale_final = 1.0 + strength * (scale - 1.0)
    out = img * scale_final.reshape(1, 1, 3)
    return _clip01(out)


def sharpen(img, params):
    """Unsharp mask nhe, amount <= 0.5."""
    amount = float(params.get("amount", 0.2))
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
    out = img + amount * (img - blurred)
    return _clip01(out)


_auto_enhance_model = None
_auto_enhance_device = None
_auto_enhance_failed = False
_auto_enhance_arch = None

_AUTO_ENHANCE_CONFIG = "checkpoints/auto_enhance_config.json"


def _load_auto_enhance():
    """Doc checkpoints/auto_enhance_config.json -> (model, device, arch).

    arch "v1": HDRNet pilot (infer.process_image, duong RGB, proxy 256).
    arch "v2": HDRNetV2 (BGR THANG nhu luc train/eval tren box GPU, make_proxy).
    Khong co config -> fallback v1 + checkpoints/auto_enhance.pt (hanh vi cu).
    """
    import json
    import os
    import torch

    torch.set_num_threads(3)

    arch = "v1"
    ckpt = "checkpoints/auto_enhance.pt"
    kwargs = {}
    if os.path.exists(_AUTO_ENHANCE_CONFIG):
        with open(_AUTO_ENHANCE_CONFIG, encoding="utf-8") as f:
            cfg = json.load(f)
        arch = cfg.get("arch", "v1")
        ckpt = cfg.get("checkpoint", ckpt)
        kwargs = cfg.get("model_kwargs", {}) or {}

    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)

    device = torch.device("cpu")
    if arch == "v2":
        from ai_engine.specialists.auto_enhance.gpu.model_v2 import HDRNetV2

        model = HDRNetV2(**kwargs).to(device)
    else:
        from ai_engine.specialists.auto_enhance.model import HDRNet

        model = HDRNet().to(device)

    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model, device, arch


def auto_enhance(img, params):
    """Chinh dep toan dien theo model hoc tu data (config: checkpoints/auto_enhance_config.json).

    Loi/thieu checkpoint -> bo qua (tra ve anh goc), khong crash pipeline.
    """
    global _auto_enhance_model, _auto_enhance_device, _auto_enhance_failed, _auto_enhance_arch

    if _auto_enhance_failed:
        return img

    if _auto_enhance_model is None:
        try:
            _auto_enhance_model, _auto_enhance_device, _auto_enhance_arch = _load_auto_enhance()
        except Exception as exc:
            print(f"[WARN] auto_enhance: khong load duoc checkpoint ({exc}). Bo qua op nay.")
            _auto_enhance_failed = True
            return img

    try:
        if _auto_enhance_arch == "v2":
            import torch
            from ai_engine.specialists.auto_enhance.gpu.model_v2 import HDRNetV2

            proxy_res = getattr(_auto_enhance_model, "proxy_res", 384)
            t = torch.from_numpy(
                np.clip(img, 0.0, 1.0).transpose(2, 0, 1).copy()
            ).unsqueeze(0).float().to(_auto_enhance_device)
            proxy = HDRNetV2.make_proxy(t, proxy_res)
            with torch.no_grad():
                out_t, _grid = _auto_enhance_model(proxy, t)
            out = out_t.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
            return _clip01(out)

        from ai_engine.specialists.auto_enhance.infer import process_image

        img_u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        out_u8, _grid_shape = process_image(_auto_enhance_model, img_u8, _auto_enhance_device)
        out = out_u8.astype(np.float32) / 255.0
        return _clip01(out)
    except Exception as exc:
        print(f"[WARN] auto_enhance: loi khi chay inference ({exc}). Bo qua op nay.")
        _auto_enhance_failed = True
        return img
