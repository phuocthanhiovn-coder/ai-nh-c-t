"""Smoke test trainer co cache: 3 epoch A_base, do thoi gian/epoch."""
import json
from tools.gpu_ssh import run

REMOTE = "/workspace/autohdr"
cfg = {
    "run_name": "_smoke", "data_dir": "data", "epochs": 3, "batch_size": 8,
    "crop": 256, "grid_bins": 8, "grid_size": 16, "proxy_res": 384, "width": 24,
    "amp": True, "num_workers": 1, "device": "cuda", "cache_ram": True, "cache_cap": 1280,
    "loss": {"w_l1": 1.0, "w_lab": 0.4, "w_perc": 0.08},
    "out": "checkpoints/sweep/_smoke.pt",
}
script = f'''cd {REMOTE}
cat > _smoke.json <<'JSON'
{json.dumps(cfg)}
JSON
/opt/conda/bin/python -u -m ai_engine.specialists.auto_enhance.gpu.train_sweep --json "$(cat _smoke.json)" 2>&1 | grep -E "RAM-cached|Epoch|Error|Traceback|cache_ram" | head -12'''
rc, out, err = run(script, timeout=180)
print(out)
if err.strip():
    print("STDERR:", err.strip()[:600])
