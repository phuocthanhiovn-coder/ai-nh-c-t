"""Phong train tren box (detached, ghi log file). Chay SAU khi deploy xong.
Config 'big' = kien truc thang sweep: grid_bins8 grid_size16 proxy_res384 width24.
Loss L1 + Lab (mau) + VGG perceptual nhe. AMP bat tren cuda.
"""
import json
from tools.gpu_ssh import run

REMOTE = "/workspace/autohdr"

CFG = {
    "run_name": "v4_big_color",
    "data_dir": "data",
    "epochs": 400,
    "batch_size": 8,
    "lr": 3e-4,
    "crop": 256,
    "grid_bins": 8,
    "grid_size": 16,
    "proxy_res": 384,
    "width": 24,
    "guidance_hidden": 16,
    "loss": {"w_l1": 1.0, "w_lab": 0.4, "w_perc": 0.08},
    "seed": 42,
    "amp": True,
    "num_workers": 8,
    "val_frac": 0.12,
    "device": "cuda",
    "out": "checkpoints/sweep/v4_big_color.pt",
}

cfg_json = json.dumps(CFG)
# ghi cfg ra file de tranh loi quote, roi phong bang setsid + python -u -> logfile
launch = f'''
cd {REMOTE}
cat > train_cfg.json <<'JSON'
{cfg_json}
JSON
: > train.log
setsid /opt/conda/bin/python -u -m ai_engine.specialists.auto_enhance.gpu.train_sweep \\
  --json "$(cat train_cfg.json)" >> train.log 2>&1 < /dev/null &
sleep 8
echo "=== PID ==="; pgrep -f train_sweep || echo none
echo "=== log dau ==="; head -30 train.log
'''
rc, out, err = run(launch, timeout=120)
print(out)
if err.strip():
    print("STDERR:", err.strip()[:400])
