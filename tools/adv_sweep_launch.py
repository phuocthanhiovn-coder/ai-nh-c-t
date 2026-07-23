from tools.gpu_ssh import run, put

# Box-side driver: race a few BIG configs (Lab+perceptual), keep best-val.
DRIVER = r'''
import sys, time, json, csv, os, shutil
sys.path.insert(0, "/workspace/autohdr")
os.chdir("/workspace/autohdr")
from ai_engine.specialists.auto_enhance.gpu.train_sweep import train_one

# lr=3e-4 / crop=256 learned well in the fast sweep. Now scale MODEL SIZE up.
LOSS = dict(w_l1=1.0, w_lab=0.3, w_perc=0.06)
CONFIGS = [
    dict(name="big",     grid_bins=16, grid_size=16, proxy_res=384, width=24, epochs=450),
    dict(name="bigger",  grid_bins=16, grid_size=32, proxy_res=384, width=32, epochs=450),
    dict(name="biggest", grid_bins=16, grid_size=32, proxy_res=512, width=32, epochs=400),
]
LB = "outputs/sweep/adv_leaderboard.csv"
os.makedirs("outputs/sweep", exist_ok=True)
rows = []
best = None
for c in CONFIGS:
    cfg = dict(
        data_dir="data/pairs", epochs=c["epochs"], batch_size=8, crop=256,
        grid_bins=c["grid_bins"], grid_size=c["grid_size"], proxy_res=c["proxy_res"], width=c["width"],
        loss=LOSS, lr=3e-4, seed=42, amp=True, device="cuda",
        out=f"checkpoints/adv_{c['name']}.pt",
    )
    print(f"===== ADV CONFIG {c['name']} grid={c['grid_bins']}/{c['grid_size']} proxy={c['proxy_res']} width={c['width']} ep={c['epochs']} =====", flush=True)
    t0 = time.time()
    try:
        res = train_one(cfg)
        bv = float(res["best_val"]); wall = round(time.time()-t0, 1)
        rows.append((c["name"], c["grid_bins"], c["grid_size"], c["proxy_res"], c["width"], bv, wall))
        print(f"  -> {c['name']} best_val={bv:.5f} wall={wall}s", flush=True)
        if best is None or bv < best[1]:
            best = (c["name"], bv, cfg["out"])
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  -> {c['name']} FAILED: {e}", flush=True)

with open(LB, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["name","grid_bins","grid_size","proxy_res","width","best_val","wall_s"])
    for r in sorted(rows, key=lambda x: x[5]): w.writerow(r)
print("===== ADV LEADERBOARD =====", flush=True)
for r in sorted(rows, key=lambda x: x[5]): print(r, flush=True)
if best:
    shutil.copy(best[2], "checkpoints/adv_best.pt")
    # also save its .meta if present
    if os.path.exists(best[2]+".meta"): shutil.copy(best[2]+".meta", "checkpoints/adv_best.pt.meta")
    print(f"BEST = {best[0]} (val={best[1]:.5f}) -> checkpoints/adv_best.pt", flush=True)
print("ADV_SWEEP_COMPLETE_MARKER", flush=True)
'''

with open("outputs/adv_sweep_driver.py", "w", newline="\n", encoding="utf-8") as f:
    f.write(DRIVER)
put("outputs/adv_sweep_driver.py", "/workspace/autohdr/adv_sweep_driver.py")
print("driver uploaded")

# kill simple sweep, launch advanced in tmux (proven pattern: tee, no kill-server)
run("pkill -f train_gpu 2>/dev/null; tmux kill-session -t sweep 2>/dev/null; sleep 2; echo cleaned", timeout=25)
rc, out, err = run("cd /workspace/autohdr && tmux new-session -d -s adv 'cd /workspace/autohdr && PYTHONPATH=. /opt/conda/bin/python adv_sweep_driver.py 2>&1 | tee outputs/sweep/adv_all.log'", timeout=20)
print("launch rc:", rc, "err:", err[:150])
rc2, out2, err2 = run("sleep 6; tmux ls; cd /workspace/autohdr; pgrep -af adv_sweep_driver | head -1; tail -4 outputs/sweep/adv_all.log 2>/dev/null", timeout=40)
print(out2); print("err:", err2[:150])
