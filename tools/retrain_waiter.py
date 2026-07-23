import os, time
from tools.gpu_ssh import run, get

os.makedirs("checkpoints/gpu", exist_ok=True)
MARKER = "FINAL178_COMPLETE_MARKER"
last_grab = 0
for i in range(80):  # ~4h at 180s
    try:
        rc, out, err = run(
            "cd /workspace/autohdr && grep -c '%s' outputs/sweep/r178.log 2>/dev/null; "
            "echo ---; ls -la checkpoints/final178*.pt 2>/dev/null; "
            "echo ---CUR---; grep -E 'Epoch|FINAL178' outputs/sweep/r178.log 2>/dev/null | tail -2" % MARKER,
            timeout=40,
        )
    except Exception as e:
        print(f"[poll {i}] ssh err {e}"); time.sleep(120); continue
    first = out.strip().split("\n")[0].strip() if out.strip() else "0"
    marker = first if first.isdigit() else "0"   # chi coi la marker neu la SO tu grep -c
    cur = out.split("---CUR---")[-1].strip() if "---CUR---" in out else ""
    has_ckpt = "final178" in out
    # grab periodically (every ~6min) if checkpoint exists
    if has_ckpt and (time.time() - last_grab > 300 or marker not in ("0","")):
        for nm in ["final178.pt", "final178_best.pt", "final178.pt.meta"]:
            try:
                get(f"/workspace/autohdr/checkpoints/{nm}", f"checkpoints/gpu/{nm}")
            except Exception:
                pass
        last_grab = time.time()
        print(f"[poll {i}] grabbed final178 | {cur[:100]}")
    else:
        print(f"[poll {i}] {cur[:110]}")
    if marker not in ("0", ""):
        print("=== RETRAIN178 COMPLETE, final178 grabbed ===")
        break
    time.sleep(180)
else:
    print("=== waiter timeout ===")
