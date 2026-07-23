"""
Con "XOA VAT THE" (LaMa inpainting, tu host tren CPU). Xem tasks/18-object-removal-lama.md.

HOP DONG OPERATOR (co them mask):
    apply(img: np.ndarray float32 [0,1] HxWx3 BGR, params: dict) -> cung shape.
params:
    mask_path (str) - duong dan anh mask xam (trang = xoa). Planner/UI cung cap
    mask, operator KHONG tu bia mask. Neu mask_path thieu/khong doc duoc/mask
    rong -> tra ve anh KHONG DOI + in canh bao.

Nguyen tac #1 (CLAUDE.md): generative model chi cham vao vung crop quanh mask,
ket qua duoc composite lai vao BAN GOC full-res qua ai_engine.core.quality.composite_mask.
"""

import os
import sys
import time

import cv2
import numpy as np
import torch

cv2.setNumThreads(2)
torch.set_num_threads(2)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

from ai_engine.core.quality import composite_mask  # noqa: E402

MASK_DILATE_PX = 8      # LaMa thich mot bien do quanh vat the de khong con vien
CROP_PAD_PX = 60        # le xung quanh bbox mask truoc khi crop, cho LaMa co ngu canh
FEATHER_PX = 4           # feather bien mask khi composite lai, tranh seam cung

_LAMA = None  # cache module-level, chi load model 1 lan


class _CpuLama:
    """
    Thay the simple_lama_inpainting.SimpleLama: thu vien goc goi
    torch.jit.load(model_path) KHONG co map_location, nen file .pt (trace tren
    may co CUDA) load fail tren may CPU-only ("aten::empty_strided ... CUDA
    backend"). O day tu tai model voi map_location=cpu, tai dung utility
    prepare_img_and_mask cua chinh thu vien do de giu logic pad/normalize.
    """

    def __init__(self, model_path, device):
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        self.model.to(device)
        self.device = device

    def __call__(self, image, mask):
        from simple_lama_inpainting.utils import prepare_img_and_mask

        image_t, mask_t = prepare_img_and_mask(image, mask, self.device)
        with torch.inference_mode():
            inpainted = self.model(image_t, mask_t)
            cur_res = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
            cur_res = np.clip(cur_res * 255, 0, 255).astype(np.uint8)
        return cur_res


def _get_lama():
    global _LAMA
    if _LAMA is None:
        from simple_lama_inpainting.models.model import LAMA_MODEL_URL
        from simple_lama_inpainting.utils import download_model

        model_path = download_model(LAMA_MODEL_URL)
        _LAMA = _CpuLama(model_path, torch.device("cpu"))
    return _LAMA


def _load_mask(mask_path, h, w):
    if not mask_path or not os.path.isfile(mask_path):
        return None
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return m


def apply(img, params=None):
    """
    HOP DONG OPERATOR: apply(img float32 [0,1] HxWx3 BGR, params) -> cung shape.
    params: mask_path (bat buoc de co tac dung).
    """
    params = params or {}
    img = np.asarray(img, dtype=np.float32)
    assert img.ndim == 3 and img.shape[2] == 3, "img phai la HxWx3"
    h, w = img.shape[:2]

    mask_path = params.get("mask_path")
    mask_u8 = _load_mask(mask_path, h, w)
    if mask_u8 is None:
        print(f"[object_removal] CANH BAO: mask_path thieu/khong doc duoc ({mask_path!r}) -> tra ve anh KHONG DOI")
        return img.copy()

    _, mask_bin = cv2.threshold(mask_u8, 127, 255, cv2.THRESH_BINARY)
    if cv2.countNonZero(mask_bin) == 0:
        print(f"[object_removal] CANH BAO: mask rong ({mask_path!r}) -> tra ve anh KHONG DOI")
        return img.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MASK_DILATE_PX * 2 + 1,) * 2)
    mask_dilated = cv2.dilate(mask_bin, kernel)

    ys, xs = np.where(mask_dilated > 0)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0 = max(0, y0 - CROP_PAD_PX)
    x0 = max(0, x0 - CROP_PAD_PX)
    y1 = min(h, y1 + CROP_PAD_PX)
    x1 = min(w, x1 + CROP_PAD_PX)

    crop_bgr = img[y0:y1, x0:x1]
    crop_mask = mask_dilated[y0:y1, x0:x1]
    crop_u8 = np.clip(crop_bgr * 255.0 + 0.5, 0, 255).astype(np.uint8)

    lama = _get_lama()
    t0 = time.time()
    result_pil = lama(crop_u8, crop_mask)
    elapsed = time.time() - t0

    result_bgr = np.asarray(result_pil).astype(np.float32) / 255.0
    # simple-lama pads noi bo len boi so cua 8 va KHONG crop lai ve size goc
    # -> tu cat lai dung kich thuoc crop truoc khi paste.
    result_bgr = result_bgr[: crop_bgr.shape[0], : crop_bgr.shape[1]]
    assert result_bgr.shape == crop_bgr.shape, "LaMa tra ve sai shape crop"

    # Feather mask crop MOT LAN (khong feather 2 lan trong composite_mask).
    mask_feathered = cv2.GaussianBlur(
        (crop_mask.astype(np.float32) / 255.0), (0, 0), sigmaX=FEATHER_PX
    )
    mask_feathered = np.clip(mask_feathered, 0.0, 1.0)

    blended_crop = composite_mask(crop_bgr, result_bgr, mask_feathered, feather_px=0)

    out = img.copy()
    out[y0:y1, x0:x1] = blended_crop

    # Assert: ngoai vung mask (feathered) da-cat, pixel PHAI bit-identical voi goc.
    mask_full = np.zeros((h, w), dtype=np.float32)
    mask_full[y0:y1, x0:x1] = mask_feathered
    unaffected = mask_full <= 0.0
    assert np.array_equal(out[unaffected], img[unaffected]), (
        "object_removal: pixel NGOAI feathered mask bi thay doi -> vi pham hop dong operator"
    )
    assert out.shape == img.shape, "object_removal: kich thuoc output phai khop input"

    n_affected = int((~unaffected).sum())
    print(
        f"[object_removal] crop=({x0},{y0})-({x1},{y1}) "
        f"{x1 - x0}x{y1 - y0}px | mask px={int(cv2.countNonZero(mask_bin))} "
        f"dilated={int(cv2.countNonZero(mask_dilated))} | LaMa {elapsed:.2f}s | "
        f"{n_affected}/{h * w} px anh huong, ngoai mask bit-identical (assert PASS)"
    )

    return out.astype(np.float32)


if __name__ == "__main__":
    print("object_removal (LaMa) module loaded.")
