"""Vong 3: tinh chinh QUAN QUAN. 2 bien the song song, dai hon + nhich Lab
de chua kem bao hoa/am. Chay: python -m tools.gpu_champion launch|status
"""
import sys
import json
from tools.gpu_ssh import run

REMOTE = "/workspace/autohdr"
COMMON = dict(data_dir="data", seed=42, amp=True, device="cuda", val_frac=0.12,
              epochs=260, cache_ram=True, cache_cap=1280, num_workers=1)

CONFIGS = [
    # CH_C: cong thuc C_bigcrop (thang sweep) - dai hon + Lab 0.5->0.6
    dict(run_name="CH_C", grid_bins=8, grid_size=16, proxy_res=384, width=24,
         crop=512, batch_size=6, lr=3e-4,
         loss={"w_l1": 1.0, "w_lab": 0.6, "w_perc": 0.08}),
    # CH_D: cong thuc D_bigmodel (dung luong cao) - dai hon + Lab 0.6
    dict(run_name="CH_D", grid_bins=8, grid_size=32, proxy_res=512, width=32,
         crop=384, batch_size=6, lr=2.5e-4,
         loss={"w_l1": 1.0, "w_lab": 0.6, "w_perc": 0.08}),
]


def full_cfg(c):
    cfg = dict(COMMON); cfg.update(c)
    cfg["out"] = f"checkpoints/sweep/{c['run_name']}.pt"
    return cfg


def launch():
    parts = [f"cd {REMOTE}", "mkdir -p logs checkpoints/sweep"]
    for c in CONFIGS:
        cfg = full_cfg(c); name = cfg["run_name"]
        parts.append(f"cat > logs/{name}.json <<'JSON'\n{json.dumps(cfg)}\nJSON")
        parts.append(f": > logs/{name}.log")
        parts.append(
            f'setsid /opt/conda/bin/python -u -m ai_engine.specialists.auto_enhance.gpu.train_sweep '
            f'--json "$(cat logs/{name}.json)" >> logs/{name}.log 2>&1 < /dev/null &')
        parts.append("sleep 2")
    parts.append("sleep 8")
    parts.append("echo '=== count ==='; pgrep -c -f train_sweep")
    parts.append("nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader")
    rc, out, err = run("\n".join(parts), timeout=120)
    print(out)
    if err.strip():
        print("STDERR:", err.strip()[:400])


def status():
    lines = [f"cd {REMOTE}", "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader"]
    for c in CONFIGS:
        n = c["run_name"]
        lines.append(f"echo '--- {n} ---'")
        lines.append(f"grep -E 'Epoch|Done run' logs/{n}.log 2>/dev/null | tail -1 || echo none")
    lines.append("echo alive:; pgrep -c -f train_sweep")
    rc, out, err = run("\n".join(lines), timeout=90)
    print(out)


if __name__ == "__main__":
    op = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"launch": launch, "status": status}.get(op, status)()
