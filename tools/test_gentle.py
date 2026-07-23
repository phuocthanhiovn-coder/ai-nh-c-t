"""Thu che do NHE TAY: pha AI+grade voi anh GOC de giu do am/giau cua goc.
Render [GOC | AI+GRADE (manh) | NHE 55% | NHE 40%]."""
import os
import cv2, numpy as np, torch
from ai_engine.specialists.auto_enhance.bracket_deliver import load_model
from ai_engine.specialists.auto_enhance.gpu.render_delivery import apply_fullres
from ai_engine.specialists.auto_enhance.gpu.finish_grade import grade_auto, protect_highlights

torch.set_num_threads(4); cv2.setNumThreads(4)
model, device = load_model("checkpoints/gpu/CH_C.pt", torch.device("cpu"))
bd = "data/pairs/before"

def blend(a, b, w):  # w phan cua a (AI)
    return np.clip(a.astype(np.float32)*w + b.astype(np.float32)*(1-w), 0, 255).astype(np.uint8)

def lab(img, t):
    s = cv2.resize(img, (620, int(img.shape[0]*620/img.shape[1])), interpolation=cv2.INTER_AREA)
    strip = np.full((24, 620, 3), 20, np.uint8)
    cv2.putText(strip, t, (6,17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235,235,235), 1, cv2.LINE_AA)
    return np.vstack([strip, s])

rows = []
for name in ["_ML_1422.jpg", "after_pool2_gd09_783A9534.jpg", "after_pool2_gd19_783A9988.jpg"]:
    b = cv2.imread(os.path.join(bd, name))
    if b is None: continue
    ai = apply_fullres(model, b, device)
    graded = protect_highlights(b, grade_auto(ai, name))
    g55 = protect_highlights(b, grade_auto(blend(ai, b, 0.55), name))
    g40 = protect_highlights(b, grade_auto(blend(ai, b, 0.40), name))
    rows.append(np.hstack([lab(b,"GOC"), lab(graded,"AI MANH (hien tai)"),
                           lab(g55,"NHE 55%"), lab(g40,"NHE 40%")]))
sheet = np.vstack(rows)
cv2.imwrite("outputs/gentle_test.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
print("-> outputs/gentle_test.jpg")
