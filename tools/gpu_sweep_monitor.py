"""Theo doi sweep 5 config: doc tien do, keo ckpt best dinh ky, bao khi ca 5 xong."""
import os
import time
from tools.gpu_ssh import run, get

REMOTE = "/workspace/autohdr"
NAMES = ["A_base", "B_color", "C_bigcrop", "D_bigmodel", "E_deepgrid"]
LOCAL_CKPT = "C:/Users/Administrator/Desktop/autohdr/checkpoints/gpu"
os.makedirs(LOCAL_CKPT, exist_ok=True)


def snapshot():
    lines = [f"cd {REMOTE}"]
    for n in NAMES:
        lines.append(f"echo '@@{n}'")
        lines.append(f"grep -E 'Epoch|Done run' logs/{n}.log 2>/dev/null | tail -1 || echo none")
        lines.append(f"grep -oE 'val_total=[0-9.]+' logs/{n}.log 2>/dev/null | sort -t= -k2 -n | head -1 || echo none")
    lines.append("echo '@@ALIVE'")
    lines.append("pgrep -c -f train_sweep || echo 0")
    rc, out, err = run("\n".join(lines), timeout=90)
    return out


def pull_all():
    ok = 0
    for n in NAMES:
        try:
            get(f"{REMOTE}/checkpoints/sweep/{n}.pt", f"{LOCAL_CKPT}/{n}.pt")
            get(f"{REMOTE}/checkpoints/sweep/{n}.pt.meta", f"{LOCAL_CKPT}/{n}.pt.meta")
            ok += 1
        except Exception:
            pass
    return ok


for i in range(1, 60):  # ~60 * 150s = 150 min max
    out = snapshot()
    # parse
    blocks = out.split("@@")
    prog = {}
    alive = "?"
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        head, _, rest = b.partition("\n")
        if head == "ALIVE":
            alive = rest.strip()
        elif head in NAMES:
            ln = rest.strip().splitlines()
            prog[head] = ln[0] if ln else "none"
    done_count = sum(1 for n in NAMES if "Done run" in prog.get(n, ""))
    # rut gon dong Epoch
    def short(s):
        import re
        m = re.search(r"Epoch (\d+)/(\d+).*val_total=([0-9.]+)", s)
        if m:
            return f"ep{m.group(1)}/{m.group(2)} val={m.group(3)}"
        return "Done" if "Done" in s else s[:30]
    row = " | ".join(f"{n}:{short(prog.get(n,'none'))}" for n in NAMES)
    pulled = pull_all() if i % 3 == 0 else "-"
    print(f"[sw {i}] alive={alive} done={done_count}/5 pull={pulled}", flush=True)
    print(f"        {row}", flush=True)
    if done_count == 5 or alive == "0":
        print("=== SWEEP XONG (hoac het process) ===", flush=True)
        print("pulled final:", pull_all(), flush=True)
        break
    time.sleep(150)
