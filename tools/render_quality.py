"""Xuat anh CHAT LUONG CAO: day chuyen deterministic (chin, khong bac mau) tren file GOC full-res, q100."""
import sys, os, glob
sys.path.insert(0, "C:/Users/Administrator/Desktop/autohdr")
import cv2, numpy as np
cv2.setNumThreads(4)
from ai_engine.orchestrator.engine import run_plan

# day chuyen sach: can trang -> nang shadow/exposure nhe -> tuong phan -> bao hoa -> net
PLAN = [
    {"op": "auto_white_balance", "params": {"wb_strength": 0.55}},
    {"op": "shadows_lift",       "params": {"amount": 0.25}},
    {"op": "contrast",           "params": {"amount": 0.22}},
    {"op": "saturation",         "params": {"amount": 0.28}},
    {"op": "sharpen",            "params": {"amount": 0.18}},
]

os.makedirs("outputs/quality", exist_ok=True)
srcs = sorted(glob.glob("data/newbatch/mixed_probe/01-RAW-Photos/*.jpg"))[:3]
for p in srcs:
    name = os.path.basename(p)
    out_path = f"outputs/quality/enhanced_{name}"
    info = run_plan(p, PLAN, out_path)
    # re-save q100 de tranh nen
    img = cv2.imread(out_path)
    cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    b = cv2.imread(p)
    # before/after full-res canh nhau (thu nho de xem tong the, ban giao van la file rieng q100)
    h, w = b.shape[:2]
    panel = np.hstack([b, img]); th = 900; tw = int(panel.shape[1] * th / panel.shape[0])
    panel = cv2.resize(panel, (tw, th))
    cv2.putText(panel, "GOC", (14, 40), 0, 1.1, (255,255,255), 3); cv2.putText(panel, "GOC", (14,40),0,1.1,(20,20,20),1)
    cv2.putText(panel, "SAU (AI)", (tw//2+14, 40), 0, 1.1, (255,255,255), 3); cv2.putText(panel, "SAU (AI)", (tw//2+14,40),0,1.1,(20,20,20),1)
    cv2.imwrite(f"outputs/quality/cmp_{name}", panel, [cv2.IMWRITE_JPEG_QUALITY, 95])
    kb = os.path.getsize(out_path)//1024
    print(f"{name}: {info['out_shape'][1]}x{info['out_shape'][0]}  {kb}KB q100  applied={info['applied']}")
print("DONE")
