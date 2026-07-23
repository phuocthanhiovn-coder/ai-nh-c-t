"""Chung minh: ap operator mau len anh GOC FULL-RES -> output full-res net cang."""
import sys, os, glob
sys.path.insert(0, "C:/Users/Administrator/Desktop/autohdr")
import cv2, numpy as np, torch
cv2.setNumThreads(4)
from ai_engine.specialists.auto_enhance.gpu.model_v2 import HDRNetV2

m = HDRNetV2(grid_bins=16, grid_size=16, proxy_res=384, width=24).eval()
m.load_state_dict(torch.load("checkpoints/gpu/adv_best.pt", map_location="cpu"))

os.makedirs("outputs/fullres_demo", exist_ok=True)
srcs = sorted(glob.glob("data/newbatch/mixed_probe/01-RAW-Photos/*.jpg"))[:3]

for p in srcs:
    name = os.path.basename(p)
    b = cv2.imread(p)  # 3000x2250 GOC
    h, w = b.shape[:2]
    proxy = cv2.resize(b, (384, 384), interpolation=cv2.INTER_AREA)
    bt = torch.from_numpy(cv2.cvtColor(b, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy()).float().unsqueeze(0) / 255.
    pt = torch.from_numpy(cv2.cvtColor(proxy, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy()).float().unsqueeze(0) / 255.
    with torch.no_grad():
        out, _ = m(pt, bt)   # output o DUNG kich thuoc full-res 3000x2250
    o = cv2.cvtColor((out.squeeze(0).numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype("uint8"), cv2.COLOR_RGB2BGR)
    assert o.shape == b.shape, f"size mismatch {o.shape} vs {b.shape}"
    # luu AI full-res q100 (giao khach)
    cv2.imwrite(f"outputs/fullres_demo/AI_fullres_{name}", o, [cv2.IMWRITE_JPEG_QUALITY, 100])
    kb = os.path.getsize(f"outputs/fullres_demo/AI_fullres_{name}") // 1024
    # crop 100% de chung minh net khi phong to (vung giua)
    cy, cx = h // 2, w // 2
    crop_b = b[cy-250:cy+250, cx-400:cx+400]
    crop_o = o[cy-250:cy+250, cx-400:cx+400]
    cropcmp = np.hstack([crop_b, crop_o])
    cv2.putText(cropcmp, "GOC (crop 100%)", (10, 30), 0, 0.8, (255, 255, 255), 2)
    cv2.putText(cropcmp, "AI (crop 100%)", (810, 30), 0, 0.8, (60, 255, 60), 2)
    cv2.imwrite(f"outputs/fullres_demo/crop100_{name}", cropcmp, [cv2.IMWRITE_JPEG_QUALITY, 100])
    print(f"{name}: output {o.shape[1]}x{o.shape[0]}  AI file {kb}KB (full-res q100)")
print("DONE")
