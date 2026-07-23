"""Sweep SONG SONG nhieu config tren cung 1 V100-32GB de ep can GPU.
Moi config: log rieng logs/<name>.log, ckpt checkpoints/sweep/<name>.pt.
Chay: python -m tools.gpu_sweep launch   (phong ca wave)
      python -m tools.gpu_sweep status   (xem tien do tat ca)
"""
import sys
import json
from tools.gpu_ssh import run

REMOTE = "/workspace/autohdr"

# --- 5 config da dang, moi con tra loi 1 cau hoi ve chat luong ---
COMMON = dict(data_dir="data", seed=42, amp=True, device="cuda",
              val_frac=0.12, epochs=160, cache_ram=True, cache_cap=1280)

CONFIGS = [
    # A: baseline big (kien truc thang sweep cu) - moc so sanh
    dict(run_name="A_base", grid_bins=8, grid_size=16, proxy_res=384, width=24,
         crop=256, batch_size=8, num_workers=1, lr=3e-4,
         loss={"w_l1": 1.0, "w_lab": 0.4, "w_perc": 0.08}),
    # B: keo mau MANH (khach che bac mau) - Lab cao, perc thap
    dict(run_name="B_color", grid_bins=8, grid_size=16, proxy_res=384, width=24,
         crop=256, batch_size=8, num_workers=1, lr=3e-4,
         loss={"w_l1": 1.0, "w_lab": 0.9, "w_perc": 0.05}),
    # C: crop TO 512 - ngu canh rong hon cho tone, GPU ban hon
    dict(run_name="C_bigcrop", grid_bins=8, grid_size=16, proxy_res=384, width=24,
         crop=512, batch_size=6, num_workers=1, lr=3e-4,
         loss={"w_l1": 1.0, "w_lab": 0.5, "w_perc": 0.08}),
    # D: model TO (width32, grid32, proxy512) - test dung luong voi 723 cap
    dict(run_name="D_bigmodel", grid_bins=8, grid_size=32, proxy_res=512, width=32,
         crop=384, batch_size=6, num_workers=1, lr=2.5e-4,
         loss={"w_l1": 1.0, "w_lab": 0.5, "w_perc": 0.08}),
    # E: luoi tong MIN (grid_bins16) - phan giai tone tot hon cho HDR
    dict(run_name="E_deepgrid", grid_bins=16, grid_size=16, proxy_res=384, width=24,
         crop=256, batch_size=8, num_workers=1, lr=3e-4,
         loss={"w_l1": 1.0, "w_lab": 0.5, "w_perc": 0.08, "w_char": 0.2}),
]


def full_cfg(c):
    cfg = dict(COMMON)
    cfg.update(c)
    cfg["out"] = f"checkpoints/sweep/{c['run_name']}.pt"
    return cfg


def launch():
    parts = [f"cd {REMOTE}", "mkdir -p logs checkpoints/sweep outputs/eval"]
    for c in CONFIGS:
        cfg = full_cfg(c)
        name = cfg["run_name"]
        cj = json.dumps(cfg)
        parts.append(f"cat > logs/{name}.json <<'JSON'\n{cj}\nJSON")
        parts.append(f": > logs/{name}.log")
        parts.append(
            f'setsid /opt/conda/bin/python -u -m ai_engine.specialists.auto_enhance.gpu.train_sweep '
            f'--json "$(cat logs/{name}.json)" >> logs/{name}.log 2>&1 < /dev/null &')
        parts.append("sleep 2")
    parts.append("sleep 10")
    parts.append("echo '=== PIDS ==='; pgrep -af train_sweep | grep -oE 'run_name.{0,20}' | sort | uniq -c")
    parts.append("echo '=== GPU ==='; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader")
    parts.append("echo '=== count ==='; pgrep -c -f train_sweep")
    script = "\n".join(parts)
    rc, out, err = run(script, timeout=180)
    print(out)
    if err.strip():
        print("STDERR:", err.strip()[:500])


def status():
    lines = [f"cd {REMOTE}", "echo '=== GPU ==='",
             "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader"]
    for c in CONFIGS:
        name = c["run_name"]
        lines.append(f"echo '--- {name} ---'")
        lines.append(f"grep -E 'Epoch|Done run' logs/{name}.log 2>/dev/null | tail -1 || echo '(chua co log)'")
        lines.append(f"grep -oE 'val_total=[0-9.]+' logs/{name}.log 2>/dev/null | sort -t= -k2 -n | head -1 | sed 's/^/  best /' || true")
    lines.append("echo '=== alive ==='; pgrep -c -f train_sweep")
    rc, out, err = run("\n".join(lines), timeout=90)
    print(out)
    if err.strip():
        print("STDERR:", err.strip()[:400])


if __name__ == "__main__":
    op = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"launch": launch, "status": status}.get(op, status)()
