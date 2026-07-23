from tools.gpu_ssh import run, get
import os

# 1) write correct .meta for adv_best (winner "big" arch), then render
FIX = r'''
cd /workspace/autohdr
/opt/conda/bin/python - <<'PY'
import torch
cfg = dict(grid_bins=16, grid_size=16, proxy_res=384, width=24, guidance_hidden=16)
meta = dict(cfg); meta["cfg"]=dict(cfg); meta["config"]=dict(cfg); meta["model"]=dict(cfg)
torch.save(meta, "checkpoints/adv_best.pt.meta")
print("meta written:", cfg)
PY
PYTHONPATH=. /opt/conda/bin/python -m ai_engine.specialists.auto_enhance.gpu.eval_visual \
  --ckpt checkpoints/adv_best.pt --n 8 --out outputs/sweep_eval/adv_best_big.jpg 2>&1 | tail -12
'''
rc, out, err = run(FIX, timeout=200)
print(out)
if err.strip():
    print("STDERR:", err[:600])

os.makedirs("outputs/sweep_eval", exist_ok=True)
try:
    get("/workspace/autohdr/outputs/sweep_eval/adv_best_big.jpg", "outputs/sweep_eval/adv_best_big.jpg")
    print("DOWNLOADED")
except Exception as e:
    print("download failed:", e)
