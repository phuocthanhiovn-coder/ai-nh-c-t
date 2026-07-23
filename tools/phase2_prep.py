from tools.gpu_ssh import run, put

# Phase 2: winner arch, RICHER color losses to fight the washed/desaturated look. Time-bounded.
DRIVER = r'''
import sys, os, time, csv, shutil
sys.path.insert(0, "/workspace/autohdr")
os.chdir("/workspace/autohdr")
from ai_engine.specialists.auto_enhance.gpu.train_sweep import train_one

# winner from size sweep = "big" (grid16/16, proxy384, width24). Read to be safe.
GB, GS, PR, WD = 16, 16, 384, 24
try:
    rows=[]
    for r in csv.DictReader(open("outputs/sweep/adv_leaderboard.csv")):
        rows.append(r)
    rows.sort(key=lambda r: float(r["best_val"])); w=rows[0]
    GB, GS, PR, WD = int(w["grid_bins"]), int(w["grid_size"]), int(w["proxy_res"]), int(w["width"])
    print("winner arch:", GB, GS, PR, WD, flush=True)
except Exception as e:
    print("using default winner arch:", e, flush=True)

def cfg(name, wlab, wperc, seed, epochs):
    return dict(name=name, c=dict(
        data_dir="data/pairs", epochs=epochs, batch_size=8, crop=256,
        grid_bins=GB, grid_size=GS, proxy_res=PR, width=WD,
        loss=dict(w_l1=1.0, w_lab=wlab, w_perc=wperc),
        lr=3e-4, seed=seed, amp=True, device="cuda", out=f"checkpoints/p2_{name}.pt"))

# Focus: crank Lab (color fidelity) + perceptual (contrast/detail) to fight desaturation.
RUNS = [
    cfg("rich",   0.60, 0.12, 42, 800),
    cfg("perchi", 0.35, 0.22, 42, 800),
]
LB="outputs/sweep/p2_leaderboard.csv"; results=[]; best=None
for R in RUNS:
    print(f"===== P2 {R['name']} wlab={R['c']['loss']['w_lab']} wperc={R['c']['loss']['w_perc']} ep={R['c']['epochs']} =====", flush=True)
    t0=time.time()
    try:
        res=train_one(R["c"]); bv=float(res["best_val"]); wall=round(time.time()-t0,1)
        results.append((R["name"], bv, wall)); print(f"  -> {R['name']} best_val={bv:.5f} wall={wall}s", flush=True)
        if best is None or bv<best[1]: best=(R["name"], bv, R["c"]["out"])
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"  -> {R['name']} FAILED {e}", flush=True)
with open(LB,"w",newline="") as f:
    wr=csv.writer(f); wr.writerow(["name","best_val","wall_s"])
    for r in sorted(results,key=lambda x:x[1]): wr.writerow(r)
# also record the arch in the winner's meta so eval reads it right
if best:
    shutil.copy(best[2],"checkpoints/p2_best.pt")
    import torch
    meta=dict(grid_bins=GB,grid_size=GS,proxy_res=PR,width=WD,guidance_hidden=16)
    meta["cfg"]=dict(meta); meta["config"]=dict(meta); meta["model"]=dict(meta)
    torch.save(meta,"checkpoints/p2_best.pt.meta")
    print(f"P2 BEST = {best[0]} val={best[1]:.5f} -> checkpoints/p2_best.pt", flush=True)
print("===== P2 LEADERBOARD =====", flush=True)
for r in sorted(results,key=lambda x:x[1]): print(r, flush=True)
print("P2_COMPLETE_MARKER", flush=True)
'''
with open("outputs/phase2_driver.py", "w", newline="\n", encoding="utf-8") as f:
    f.write(DRIVER)
put("outputs/phase2_driver.py", "/workspace/autohdr/phase2_driver.py")
print("phase2 driver (richer losses, 2x800ep) uploaded")

# launch in tmux (proven pattern)
run("pkill -f train_gpu 2>/dev/null; pkill -f adv_sweep_driver 2>/dev/null; tmux kill-session -t adv 2>/dev/null; tmux kill-session -t p2 2>/dev/null; sleep 2; echo cleaned", timeout=25)
rc, out, err = run("cd /workspace/autohdr && tmux new-session -d -s p2 'cd /workspace/autohdr && PYTHONPATH=. /opt/conda/bin/python phase2_driver.py 2>&1 | tee outputs/sweep/p2_all.log'", timeout=20)
print("launch rc:", rc, "err:", err[:150])
rc2, out2, err2 = run("sleep 6; tmux ls; cd /workspace/autohdr; pgrep -af phase2_driver | head -1; tail -3 outputs/sweep/p2_all.log 2>/dev/null", timeout=40)
print(out2); print("err:", err2[:150])
