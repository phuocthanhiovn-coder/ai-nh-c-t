import os
from tools.gpu_ssh import run, get

os.makedirs("checkpoints/gpu", exist_ok=True)

# 1) phase 2 status
rc, out, err = run(
    "cd /workspace/autohdr && echo '=== p2 log tail ==='; tail -6 outputs/sweep/p2_all.log 2>/dev/null; "
    "echo '=== checkpoints ==='; ls -la checkpoints/*.pt 2>/dev/null; "
    "echo '=== leaderboards ==='; cat outputs/sweep/*leaderboard*.csv 2>/dev/null",
    timeout=60,
)
print(out)
if err.strip():
    print("ERR:", err[:300])

# 2) list all .pt and download each (+.meta)
rc2, out2, err2 = run("ls /workspace/autohdr/checkpoints/*.pt 2>/dev/null", timeout=30)
pts = [l.strip() for l in out2.splitlines() if l.strip().endswith(".pt")]
print(f"\n=== downloading {len(pts)} checkpoints ===")
got = 0
for rp in pts:
    name = os.path.basename(rp)
    try:
        get(rp, f"checkpoints/gpu/{name}")
        got += 1
        try:
            get(rp + ".meta", f"checkpoints/gpu/{name}.meta")
        except Exception:
            pass
        print(f"  got {name}")
    except Exception as e:
        print(f"  FAIL {name}: {e}")

# also grab leaderboards + logs
for f in ["outputs/sweep/adv_leaderboard.csv", "outputs/sweep/p2_leaderboard.csv",
          "outputs/sweep/adv_all.log", "outputs/sweep/p2_all.log"]:
    try:
        get(f"/workspace/autohdr/{f}", f"outputs/gpu_logs/{os.path.basename(f)}")
    except Exception:
        pass
print(f"\nDONE: {got}/{len(pts)} checkpoints downloaded to checkpoints/gpu/")
