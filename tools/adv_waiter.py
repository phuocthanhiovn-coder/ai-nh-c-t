import os
import re
import time
from tools.gpu_ssh import run, get

os.makedirs("checkpoints/gpu", exist_ok=True)
downloaded = set()
MARKER = "ADV_SWEEP_COMPLETE_MARKER"

for i in range(90):  # up to ~3h at 120s
    try:
        rc, out, err = run(
            "cd /workspace/autohdr && "
            "grep -c '%s' outputs/sweep/adv_all.log 2>/dev/null; echo '---DONE---'; "
            "grep -oE '\\-> [a-z]+ best_val=[0-9.]+' outputs/sweep/adv_all.log 2>/dev/null; echo '---CUR---'; "
            "tail -2 outputs/sweep/adv_all.log 2>/dev/null" % MARKER,
            timeout=40,
        )
    except Exception as e:
        print(f"[poll {i}] ssh err {e}"); time.sleep(90); continue

    marker = out.strip().split("\n")[0].strip() if out.strip() else "0"
    done_block = out.split("---DONE---")[-1].split("---CUR---")[0] if "---DONE---" in out else ""
    cur = out.split("---CUR---")[-1].strip() if "---CUR---" in out else ""

    # download any config that has reported best_val (i.e., finished)
    for m in re.finditer(r"-> ([a-z]+) best_val=([0-9.]+)", done_block):
        name = m.group(1)
        if name in downloaded:
            continue
        try:
            get(f"/workspace/autohdr/checkpoints/adv_{name}.pt", f"checkpoints/gpu/adv_{name}.pt")
            try:
                get(f"/workspace/autohdr/checkpoints/adv_{name}.pt.meta", f"checkpoints/gpu/adv_{name}.pt.meta")
            except Exception:
                pass
            downloaded.add(name)
            print(f"[poll {i}] DOWNLOADED adv_{name}.pt (val={m.group(2)})")
        except Exception as e:
            print(f"[poll {i}] download {name} failed: {e}")

    print(f"[poll {i}] done_configs={len(downloaded)} | {cur[:120]}")
    if marker not in ("0", ""):
        # grab adv_best.pt too
        try:
            get("/workspace/autohdr/checkpoints/adv_best.pt", "checkpoints/gpu/adv_best.pt")
            try:
                get("/workspace/autohdr/checkpoints/adv_best.pt.meta", "checkpoints/gpu/adv_best.pt.meta")
            except Exception:
                pass
            print("=== ADV SWEEP COMPLETE, adv_best.pt downloaded ===")
        except Exception as e:
            print("best download failed:", e)
        break
    time.sleep(120)
else:
    print("=== waiter timeout ===")
print("DOWNLOADED:", sorted(downloaded))
