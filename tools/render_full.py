from tools.gpu_ssh import run, get, put
import os

# render per-image full-res [before | AI output | AutoHDR] for a few diverse val images
DRIVER = r'''
import sys, os, glob, hashlib, cv2, numpy as np, torch
sys.path.insert(0, "/workspace/autohdr"); os.chdir("/workspace/autohdr")
from ai_engine.specialists.auto_enhance.gpu.model_v2 import HDRNetV2
m = HDRNetV2(grid_bins=16, grid_size=16, proxy_res=384, width=24).cuda().eval()
m.load_state_dict(torch.load("checkpoints/adv_best.pt", map_location="cuda"))
befs = sorted(os.path.basename(p) for p in glob.glob("data/pairs/before/*.jpg"))
val = [f for f in befs if (int(hashlib.md5(f.encode()).hexdigest(),16)%1000)/1000.0 < 0.12]
os.makedirs("outputs/ai_outputs", exist_ok=True)
pick = val[:6]
for f in pick:
    b = cv2.imread(f"data/pairs/before/{f}"); a = cv2.imread(f"data/pairs/after/{f}")
    h,w = b.shape[:2]
    proxy = cv2.resize(b, (384,384), interpolation=cv2.INTER_AREA)
    bt = torch.from_numpy(cv2.cvtColor(b,cv2.COLOR_BGR2RGB).transpose(2,0,1).copy()).float().unsqueeze(0).cuda()/255.
    pt = torch.from_numpy(cv2.cvtColor(proxy,cv2.COLOR_BGR2RGB).transpose(2,0,1).copy()).float().unsqueeze(0).cuda()/255.
    with torch.no_grad(): out,_ = m(pt, bt)
    o = (out.squeeze(0).cpu().numpy().transpose(1,2,0)*255).clip(0,255).astype("uint8")
    o = cv2.cvtColor(o, cv2.COLOR_RGB2BGR)
    # save AI-only full res
    cv2.imwrite(f"outputs/ai_outputs/AI_{f}", o, [cv2.IMWRITE_JPEG_QUALITY,95])
    # save side-by-side at ~800 tall
    a2 = cv2.resize(a,(w,h))
    panel = np.hstack([b,o,a2]); th=800; tw=int(panel.shape[1]*th/panel.shape[0])
    panel = cv2.resize(panel,(tw,th))
    cv2.putText(panel,"GOC",(10,34),0,1.0,(60,60,255),2)
    cv2.putText(panel,"AI",(tw//3+10,34),0,1.0,(60,255,60),2)
    cv2.putText(panel,"AUTOHDR",(2*tw//3+10,34),0,1.0,(255,160,60),2)
    cv2.imwrite(f"outputs/ai_outputs/cmp_{f}", panel, [cv2.IMWRITE_JPEG_QUALITY,92])
    print("rendered", f, flush=True)
print("DONE")
'''
with open("outputs/render_full_driver.py","w",newline="\n",encoding="utf-8") as fp:
    fp.write(DRIVER)
put("outputs/render_full_driver.py","/workspace/autohdr/render_full_driver.py")
rc,out,err = run("cd /workspace/autohdr && PYTHONPATH=. /opt/conda/bin/python render_full_driver.py 2>&1 | tail -12", timeout=180)
print(out)
if err.strip(): print("STDERR:", err[:500])

# download comparison sheets + AI-only for the picks
os.makedirs("outputs/ai_outputs", exist_ok=True)
rc2,out2,err2 = run("ls /workspace/autohdr/outputs/ai_outputs/cmp_*.jpg", timeout=30)
files = [l.strip() for l in out2.splitlines() if l.strip().endswith(".jpg")]
got=0
for rp in files[:6]:
    name = os.path.basename(rp)
    try:
        get(rp, f"outputs/ai_outputs/{name}"); got+=1
    except Exception as e:
        print("dl fail", name, e)
print(f"downloaded {got} comparison images")
