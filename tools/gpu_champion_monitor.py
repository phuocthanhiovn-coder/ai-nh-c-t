"""Theo doi vong tinh chinh CH_C/CH_D: keo ckpt dinh ky, bao khi ca 2 xong."""
import os
import re
import time
from tools.gpu_ssh import run, get

REMOTE = "/workspace/autohdr"
NAMES = ["CH_C", "CH_D"]
LOCAL = "C:/Users/Administrator/Desktop/autohdr/checkpoints/gpu"
os.makedirs(LOCAL, exist_ok=True)


def pull():
    ok = 0
    for n in NAMES:
        try:
            get(f"{REMOTE}/checkpoints/sweep/{n}.pt", f"{LOCAL}/{n}.pt")
            get(f"{REMOTE}/checkpoints/sweep/{n}.pt.meta", f"{LOCAL}/{n}.pt.meta")
            ok += 1
        except Exception:
            pass
    return ok


for i in range(1, 50):
    lines = [f"cd {REMOTE}"]
    for n in NAMES:
        lines.append(f"echo @@{n}")
        lines.append(f"grep -E 'Epoch|Done run' logs/{n}.log 2>/dev/null | tail -1 || echo none")
    lines.append("echo @@ALIVE")
    lines.append("pgrep -c -f train_sweep || echo 0")
    rc, out, err = run("\n".join(lines), timeout=90)
    blocks = out.split("@@")
    prog = {}; alive = "?"
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        head, _, rest = b.partition("\n")
        if head == "ALIVE":
            alive = rest.strip()
        elif head in NAMES:
            prog[head] = rest.strip().splitlines()[0] if rest.strip() else "none"
    done = sum(1 for n in NAMES if "Done run" in prog.get(n, ""))

    def short(s):
        m = re.search(r"Epoch (\d+)/(\d+).*val_l1=([0-9.]+)", s)
        return f"ep{m.group(1)}/{m.group(2)} l1={m.group(3)}" if m else ("Done" if "Done" in s else s[:24])
    row = " | ".join(f"{n}:{short(prog.get(n,'none'))}" for n in NAMES)
    pulled = pull() if i % 2 == 0 else "-"
    print(f"[ch {i}] alive={alive} done={done}/2 pull={pulled} | {row}", flush=True)
    if done == 2 or alive == "0":
        print("=== CHAMPION XONG ===", flush=True)
        print("pulled final:", pull(), flush=True)
        break
    time.sleep(150)
