"""Test grade moi (co sharpen): render 1 anh val -> crop 100% [goc | AI+grade] de soi net."""
import os
import cv2
import numpy as np
import torch

from ai_engine.specialists.auto_enhance.bracket_deliver import load_model
from ai_engine.specialists.auto_enhance.gpu.render_delivery import apply_fullres
from ai_engine.specialists.auto_enhance.gpu.finish_grade import grade_auto

torch.set_num_threads(4)
cv2.setNumThreads(4)

NAME = os.environ.get("IMG", "drone01_DSC01518.jpg")
bd = "data/pairs/before"
b = cv2.imread(os.path.join(bd, NAME))
if b is None:
    import glob
    b = cv2.imread(sorted(glob.glob(bd + "/*.jpg"))[0])
    NAME = "auto"

model, device = load_model("checkpoints/gpu/CH_C.pt", torch.device("cpu"))
ai = apply_fullres(model, b, device)
graded = grade_auto(ai, NAME)

# luu full de xem tong the
cv2.imwrite("outputs/sharp_full.jpg", graded, [cv2.IMWRITE_JPEG_QUALITY, 95])

# crop 100% vung giua (600x400) de soi net that
h, w = graded.shape[:2]
cy, cx = h // 2, w // 2
ch, cw = min(360, h//2), min(560, w//2)
y0, x0 = cy - ch//2, cx - cw//2
def crop(im): return im[y0:y0+ch, x0:x0+cw]
strip = cv2.hconcat([crop(b), np.full((ch, 6, 3), 255, np.uint8), crop(graded)])
cv2.imwrite("outputs/sharp_crop100.jpg", strip, [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f"[test] {NAME}  full {w}x{h}  -> outputs/sharp_full.jpg + sharp_crop100.jpg (100% crop: GOC | AI+GRADE)")
