import os, subprocess
from tools.gpu_ssh import run, put

# 1) pack current 178 pairs
print("packing 178 pairs...")
os.chdir("C:/Users/Administrator/Desktop/autohdr")
subprocess.run(["python", "-m", "ai_engine.specialists.auto_enhance.pack_dataset"], check=False)
zips = sorted([f for f in os.listdir("outputs") if f.startswith("dataset_v") and f.endswith(".zip")])
zp = "outputs/" + zips[-1]
print("dataset zip:", zp, os.path.getsize(zp)//1024//1024, "MB")

# 2) upload + unzip on box (replace data/pairs)
put(zp, "/workspace/autohdr/dataset_new.zip")
rc, out, err = run(
    "cd /workspace/autohdr && rm -rf ds_new && mkdir ds_new && unzip -q -o dataset_new.zip -d ds_new && "
    "rm -rf data/pairs && mkdir -p data/pairs && "
    "cp -r $(find ds_new -type d -name before|head -1) data/pairs/before && "
    "cp -r $(find ds_new -type d -name after|head -1) data/pairs/after && "
    "echo pairs: $(ls data/pairs/before|wc -l)",
    timeout=120,
)
print(out); print("err:", err[:200])

# 3) write retrain script (winner big + rich loss on 178, 550 ep, periodic best-val save)
DRIVER = '''import sys, os
sys.path.insert(0, "/workspace/autohdr"); os.chdir("/workspace/autohdr")
from ai_engine.specialists.auto_enhance.gpu.train_sweep import train_one
cfg = dict(data_dir="data/pairs", epochs=550, batch_size=8, crop=256,
    grid_bins=16, grid_size=16, proxy_res=384, width=24,
    loss=dict(w_l1=1.0, w_lab=0.5, w_perc=0.12),
    lr=3e-4, seed=42, amp=True, device="cuda", out="checkpoints/final178.pt")
res = train_one(cfg)
import torch
meta=dict(grid_bins=16,grid_size=16,proxy_res=384,width=24,guidance_hidden=16)
meta["cfg"]=dict(meta); torch.save(meta,"checkpoints/final178.pt.meta")
print("FINAL178 best_val=", res["best_val"], flush=True)
print("FINAL178_COMPLETE_MARKER", flush=True)
'''
with open("outputs/retrain178_driver.py","w",newline="\n",encoding="utf-8") as f:
    f.write(DRIVER)
put("outputs/retrain178_driver.py","/workspace/autohdr/retrain178_driver.py")

run("pkill -f train_gpu 2>/dev/null; pkill -f phase2_driver 2>/dev/null; tmux kill-session -t p2 2>/dev/null; sleep 2; echo cleaned", timeout=25)
rc2, out2, err2 = run("cd /workspace/autohdr && tmux new-session -d -s r178 'cd /workspace/autohdr && PYTHONPATH=. /opt/conda/bin/python retrain178_driver.py 2>&1 | tee outputs/sweep/r178.log'", timeout=20)
print("launch rc:", rc2, "err:", err2[:150])
rc3, out3, err3 = run("sleep 6; tmux ls; cd /workspace/autohdr; pgrep -af retrain178 | head -1; tail -3 outputs/sweep/r178.log 2>/dev/null", timeout=40)
print(out3)
