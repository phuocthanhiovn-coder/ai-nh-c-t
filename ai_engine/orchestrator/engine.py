"""Ap plan operator len anh full-res, giu nguyen kich thuoc."""
import os
import cv2
import numpy as np

from .registry import REGISTRY, clamp_params


def run_plan(img_path, plan, out_path):
    """Doc anh (giu nguyen res), float32 [0,1], ap op tuan tu, ghi ra out_path.

    Tra ve dict {"in_shape": (h, w), "out_shape": (h, w), "applied": [ten_op, ...]}.
    """
    img_u8 = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_u8 is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {img_path}")

    in_shape = img_u8.shape[:2]
    img = img_u8.astype(np.float32) / 255.0

    applied = []
    for step in plan:
        op_name = step["op"]
        params = step.get("params", {})
        entry = REGISTRY.get(op_name)
        if entry is None:
            print(f"[WARN] engine: bo qua op khong ton tai '{op_name}'.")
            continue
        # LUON chuan hoa params qua schema (float clamp / enum whitelist / bool) truoc khi ap,
        # ke ca khi plan den tu --plan JSON truc tiep. Specialist khong bao gio nhan rac.
        safe_params = clamp_params(op_name, params)
        img = entry["fn"](img, safe_params)
        applied.append(op_name)

    out_shape = img.shape[:2]
    assert out_shape == in_shape, f"Kich thuoc thay doi: vao {in_shape} != ra {out_shape}"

    out_u8 = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)

    ext = os.path.splitext(out_path)[1].lower()
    if ext == ".png":
        cv2.imwrite(out_path, out_u8)
    else:
        cv2.imwrite(out_path, out_u8, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return {"in_shape": in_shape, "out_shape": out_shape, "applied": applied}
