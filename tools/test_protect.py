"""Test protect_highlights: [goc | AI+grade | AI+grade+protect] cho anh cua so."""
import os, glob
import cv2, numpy as np, torch
from ai_engine.specialists.auto_enhance.bracket_deliver import load_model
from ai_engine.specialists.auto_enhance.gpu.render_delivery import apply_fullres
from ai_engine.specialists.auto_enhance.gpu.finish_grade import grade_auto, protect_highlights

torch.set_num_threads(4); cv2.setNumThreads(4)
model, device = load_model("checkpoints/gpu/CH_C.pt", torch.device("cpu"))
bd = "data/pairs/before"

def lab(img, t):
    s = cv2.resize(img, (760, int(img.shape[0]*760/img.shape[1])), interpolation=cv2.INTER_AREA)
    strip = np.full((26, 760, 3), 20, np.uint8)
    cv2.putText(strip, t, (6,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235,235,235), 1, cv2.LINE_AA)
    return np.vstack([strip, s])

rows = []
for name in ["_ML_1422.jpg", "after_pool2_gd09_783A9534.jpg", "_ML_1444.jpg"]:
    b = cv2.imread(os.path.join(bd, name))
    if b is None: continue
    ai = apply_fullres(model, b, device)
    graded = grade_auto(ai, name)
    prot = protect_highlights(b, graded)
    # do % pixel chay
    def blown(im): return 100.0*(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)>=250).mean()
    print(f"{name}: chay% goc={blown(b):.2f} grade={blown(graded):.2f} protect={blown(prot):.2f}")
    rows.append(np.hstack([lab(b,"GOC"), lab(graded,"AI+GRADE"), lab(prot,"AI+GRADE+BAO VE")]))
sheet = np.vstack(rows)
cv2.imwrite("outputs/protect_test.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
print("-> outputs/protect_test.jpg")
