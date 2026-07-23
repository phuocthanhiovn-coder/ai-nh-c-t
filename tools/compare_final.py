"""So adv_best (chi 136 noi that) vs final178 (co hoc 42 drone) tren anh drone."""
import sys, os, glob
sys.path.insert(0, "C:/Users/Administrator/Desktop/autohdr")
import cv2, numpy as np, torch
cv2.setNumThreads(4)
from ai_engine.specialists.auto_enhance.gpu.model_v2 import HDRNetV2

def load(ck):
    m = HDRNetV2(grid_bins=16, grid_size=16, proxy_res=384, width=24).eval()
    m.load_state_dict(torch.load(ck, map_location="cpu")); return m

def infer(m, b):
    proxy = cv2.resize(b,(384,384),interpolation=cv2.INTER_AREA)
    bt=torch.from_numpy(cv2.cvtColor(b,cv2.COLOR_BGR2RGB).transpose(2,0,1).copy()).float().unsqueeze(0)/255.
    pt=torch.from_numpy(cv2.cvtColor(proxy,cv2.COLOR_BGR2RGB).transpose(2,0,1).copy()).float().unsqueeze(0)/255.
    with torch.no_grad(): out,_=m(pt,bt)
    return cv2.cvtColor((out.squeeze(0).numpy().transpose(1,2,0)*255).clip(0,255).astype("uint8"),cv2.COLOR_RGB2BGR)

def sat(x): return float(cv2.cvtColor(x,cv2.COLOR_BGR2HSV)[:,:,1].mean())

m_old=load("checkpoints/gpu/adv_best.pt"); m_new=load("checkpoints/gpu/final178.pt")
os.makedirs("outputs/final_compare",exist_ok=True)
for p in sorted(glob.glob("data/newbatch/mixed_probe/01-RAW-Photos/*.jpg"))[:2]:
    name=os.path.basename(p); b=cv2.imread(p)
    o_old=infer(m_old,b); o_new=infer(m_new,b)
    panel=np.hstack([b,o_old,o_new]); th=720; tw=int(panel.shape[1]*th/panel.shape[0]); panel=cv2.resize(panel,(tw,th))
    cv2.putText(panel,"GOC",(10,32),0,0.9,(255,255,255),2)
    cv2.putText(panel,f"CU 136-noithat sat{sat(o_old):.0f}",(tw//3+10,32),0,0.7,(80,80,255),2)
    cv2.putText(panel,f"MOI 178-co-drone sat{sat(o_new):.0f}",(2*tw//3+10,32),0,0.7,(60,255,60),2)
    cv2.imwrite(f"outputs/final_compare/{name}",panel,[cv2.IMWRITE_JPEG_QUALITY,93])
    print(f"{name}: GOC sat={sat(b):.0f} | CU={sat(o_old):.0f} | MOI={sat(o_new):.0f}")
print("DONE")
