"""Theo doi train tren box: doc log, keo checkpoint best ve dinh ky, bao khi xong."""
import os
import re
import time
from tools.gpu_ssh import run, get

REMOTE = "/workspace/autohdr"
CKPT_REMOTE = f"{REMOTE}/checkpoints/sweep/v4_big_color.pt"
CKPT_LOCAL = "C:/Users/Administrator/Desktop/autohdr/checkpoints/gpu/v4_big_color.pt"
META_REMOTE = CKPT_REMOTE + ".meta"
META_LOCAL = CKPT_LOCAL + ".meta"

os.makedirs(os.path.dirname(CKPT_LOCAL), exist_ok=True)


def last_epoch():
    rc, out, err = run(
        f"cd {REMOTE} && grep -E 'Epoch|Done run' train.log | tail -3", timeout=40)
    return out.strip()


def pull():
    try:
        get(CKPT_REMOTE, CKPT_LOCAL)
        get(META_REMOTE, META_LOCAL)
        return os.path.getsize(CKPT_LOCAL)
    except Exception as e:
        return f"pull-fail {e}"


for i in range(1, 40):  # ~40 * 150s = 100 min max
    tail = last_epoch()
    line = tail.splitlines()[-1] if tail else "(no log)"
    done = "Done run" in tail
    # keo checkpoint moi ~10 vong (moi ~5 phut)
    sz = pull() if i % 2 == 0 else "-"
    print(f"[mon {i}] {line} | ckpt={sz}", flush=True)
    if done:
        print("=== TRAIN XONG ===", flush=True)
        print(pull(), flush=True)
        break
    time.sleep(150)
