import time
from tools.gpu_ssh import run

MARKER = "SWEEP_COMPLETE_MARKER"
for i in range(120):  # up to ~3h at 90s
    try:
        rc, out, err = run(
            "cd /workspace/autohdr && "
            "grep -c '%s' outputs/sweep/sweep_all.log 2>/dev/null; "
            "echo '---CFG---'; ls outputs/sweep/*.log 2>/dev/null | wc -l; "
            "echo '---LB---'; cat outputs/sweep/leaderboard.csv 2>/dev/null; "
            "echo '---NOW---'; pgrep -af train_gpu | head -1 | grep -oE 'sw_[a-z0-9_]+|final_s[0-9]+' | head -1" % MARKER,
            timeout=40,
        )
    except Exception as e:
        print(f"[poll {i}] ssh err {e}")
        time.sleep(60)
        continue
    done = out.strip().split("\n")[0].strip() if out.strip() else "0"
    if done not in ("0", ""):
        print("=== SWEEP COMPLETE ===")
        print(out)
        break
    # progress line
    lb = out.split("---LB---")[-1].split("---NOW---")[0].strip() if "---LB---" in out else ""
    nrows = len([l for l in lb.splitlines() if "," in l]) - 1
    now = out.split("---NOW---")[-1].strip() if "---NOW---" in out else "?"
    print(f"[poll {i}] configs done: {max(nrows,0)} | running: {now}")
    time.sleep(90)
else:
    print("=== TIMEOUT waiting for sweep ===")
