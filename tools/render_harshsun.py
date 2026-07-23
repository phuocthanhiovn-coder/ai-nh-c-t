"""Demo xu ly DAI SANG RONG (nang gat): keo lai vung chay + mo bong + can trang + mau."""
import sys, os
sys.path.insert(0, "C:/Users/Administrator/Desktop/autohdr")
import cv2, numpy as np
cv2.setNumThreads(4)
from ai_engine.orchestrator.engine import run_plan

# pipeline "nang gat": phuc hoi vung chay MANH + mo bong MANH -> nen dai sang lai (giong HDR)
PLAN = [
    {"op": "highlights_recover", "params": {"amount": 0.7}},
    {"op": "shadows_lift",       "params": {"amount": 0.5}},
    {"op": "auto_white_balance", "params": {"wb_strength": 0.5}},
    {"op": "contrast",           "params": {"amount": 0.15}},
    {"op": "saturation",         "params": {"amount": 0.25}},
    {"op": "sharpen",            "params": {"amount": 0.15}},
]

os.makedirs("outputs/harshsun", exist_ok=True)
# anh tuong phan cao nhat co san (noi that cua so chay + _ML_1358 co troi)
picks = ["20260703-DSC1161.jpg", "_ML_1358.jpg"]
for name in picks:
    p = f"data/pairs/before/{name}"
    if not os.path.exists(p):
        print("skip (khong co):", name); continue
    out_path = f"outputs/harshsun/hdr_{name}"
    info = run_plan(p, PLAN, out_path)
    o = cv2.imread(out_path); cv2.imwrite(out_path, o, [cv2.IMWRITE_JPEG_QUALITY, 100])
    b = cv2.imread(p)
    h, w = b.shape[:2]
    panel = np.hstack([b, o]); th = 760; tw = int(panel.shape[1]*th/panel.shape[0])
    panel = cv2.resize(panel, (tw, th))
    cv2.putText(panel, "GOC", (12, 36), 0, 1.0, (255,255,255), 3); cv2.putText(panel,"GOC",(12,36),0,1.0,(20,20,20),1)
    cv2.putText(panel, "SAU (nen dai sang)", (tw//2+12, 36), 0, 1.0, (255,255,255), 3); cv2.putText(panel,"SAU (nen dai sang)",(tw//2+12,36),0,1.0,(20,20,20),1)
    cv2.imwrite(f"outputs/harshsun/cmp_{name}", panel, [cv2.IMWRITE_JPEG_QUALITY, 93])
    print(f"{name}: {info['out_shape'][1]}x{info['out_shape'][0]} applied={info['applied']}")
print("DONE")
