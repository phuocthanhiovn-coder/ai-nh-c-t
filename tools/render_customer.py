from tools.gpu_ssh import run, get, put
import os

DRIVER = r'''
import sys, os, glob, hashlib, cv2, numpy as np, torch
sys.path.insert(0, "/workspace/autohdr"); os.chdir("/workspace/autohdr")
from ai_engine.specialists.auto_enhance.gpu.model_v2 import HDRNetV2
m = HDRNetV2(grid_bins=16, grid_size=16, proxy_res=384, width=24).cuda().eval()
m.load_state_dict(torch.load("checkpoints/adv_best.pt", map_location="cuda"))
befs = sorted(os.path.basename(p) for p in glob.glob("data/pairs/before/*.jpg"))
val = [f for f in befs if (int(hashlib.md5(f.encode()).hexdigest(),16)%1000)/1000.0 < 0.12]
os.makedirs("outputs/customer_preview", exist_ok=True)
for f in val[:8]:
    b = cv2.imread(f"data/pairs/before/{f}")
    h,w = b.shape[:2]
    proxy = cv2.resize(b, (384,384), interpolation=cv2.INTER_AREA)
    bt = torch.from_numpy(cv2.cvtColor(b,cv2.COLOR_BGR2RGB).transpose(2,0,1).copy()).float().unsqueeze(0).cuda()/255.
    pt = torch.from_numpy(cv2.cvtColor(proxy,cv2.COLOR_BGR2RGB).transpose(2,0,1).copy()).float().unsqueeze(0).cuda()/255.
    with torch.no_grad(): out,_ = m(pt, bt)
    o = cv2.cvtColor((out.squeeze(0).cpu().numpy().transpose(1,2,0)*255).clip(0,255).astype("uint8"), cv2.COLOR_RGB2BGR)
    # AI-only full res (what customer receives as the edited photo)
    cv2.imwrite(f"outputs/customer_preview/edited_{f}", o, [cv2.IMWRITE_JPEG_QUALITY,95])
    # before->AI 2-column (value demo, no AutoHDR column)
    panel = np.hstack([b, o]); th=760; tw=int(panel.shape[1]*th/panel.shape[0])
    panel = cv2.resize(panel,(tw,th))
    cv2.putText(panel,"BEFORE",(12,36),0,1.0,(255,255,255),3)
    cv2.putText(panel,"BEFORE",(12,36),0,1.0,(30,30,30),1)
    cv2.putText(panel,"AFTER",(tw//2+12,36),0,1.0,(255,255,255),3)
    cv2.putText(panel,"AFTER",(tw//2+12,36),0,1.0,(30,30,30),1)
    cv2.imwrite(f"outputs/customer_preview/beforeafter_{f}", panel, [cv2.IMWRITE_JPEG_QUALITY,92])
print("DONE")
'''
with open("outputs/render_customer_driver.py","w",newline="\n",encoding="utf-8") as fp:
    fp.write(DRIVER)
put("outputs/render_customer_driver.py","/workspace/autohdr/render_customer_driver.py")
rc,out,err = run("cd /workspace/autohdr && PYTHONPATH=. /opt/conda/bin/python render_customer_driver.py 2>&1 | tail -5", timeout=180)
print(out)
if err.strip(): print("STDERR:", err[:400])

os.makedirs("outputs/customer_preview", exist_ok=True)
rc2,out2,err2 = run("ls /workspace/autohdr/outputs/customer_preview/*.jpg", timeout=30)
files = [l.strip() for l in out2.splitlines() if l.strip().endswith(".jpg")]
got=0
for rp in files:
    try:
        get(rp, f"outputs/customer_preview/{os.path.basename(rp)}"); got+=1
    except Exception as e:
        print("dl fail", os.path.basename(rp), e)
print(f"downloaded {got} files to outputs/customer_preview/")
