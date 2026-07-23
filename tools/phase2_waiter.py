import os
import time
from tools.gpu_ssh import run, get

os.makedirs("checkpoints/gpu", exist_ok=True)
MARKER = "P2_COMPLETE_MARKER"
for i in range(120):
    try:
        rc, out, err = run(
            "cd /workspace/autohdr && grep -c '%s' outputs/sweep/p2_all.log 2>/dev/null; "
            "echo ---; grep -oE '\\-> [a-z]+ best_val=[0-9.]+' outputs/sweep/p2_all.log 2>/dev/null; "
            "echo ---CUR---; grep -E 'Epoch|P2 ' outputs/sweep/p2_all.log 2>/dev/null | tail -2" % MARKER,
            timeout=40,
        )
    except Exception as e:
        print(f"[poll {i}] ssh err {e}"); time.sleep(90); continue
    marker = out.strip().split("\n")[0].strip() if out.strip() else "0"
    cur = out.split("---CUR---")[-1].strip() if "---CUR---" in out else ""
    done = out.split("---")[1] if "---" in out else ""
    ndone = len([x for x in done.splitlines() if "best_val" in x])
    print(f"[poll {i}] runs_done={ndone} | {cur[:120]}")
    if marker not in ("0", ""):
        try:
            get("/workspace/autohdr/checkpoints/p2_best.pt", "checkpoints/gpu/p2_best.pt")
            get("/workspace/autohdr/checkpoints/p2_best.pt.meta", "checkpoints/gpu/p2_best.pt.meta")
            print("=== P2 COMPLETE, p2_best.pt downloaded ===")
        except Exception as e:
            print("dl failed:", e)
        break
    time.sleep(120)
else:
    print("=== waiter timeout ===")
