"""Ghi sweep_box.sh (LF), upload len box, launch trong tmux."""
from tools.gpu_ssh import run, put

SWEEP = """#!/bin/bash
cd /workspace/autohdr
export PYTHONPATH=.
PY=/opt/conda/bin/python
mkdir -p checkpoints outputs/sweep
LB=outputs/sweep/leaderboard.csv
echo "name,lr,loss,crop,epochs,best_val" > $LB

run_cfg () {
  local name=$1 lr=$2 loss=$3 crop=$4 epochs=$5 seed=$6
  local extra=""
  [ "$loss" = "charbonnier" ] && extra="--charbonnier"
  echo "===== CONFIG $name (lr=$lr loss=$loss crop=$crop ep=$epochs seed=$seed) ====="
  $PY -m ai_engine.specialists.auto_enhance.train_gpu \\
    --epochs $epochs --batch-size 8 --crop $crop --lr $lr $extra \\
    --num-workers 8 --seed $seed --save-every 200 \\
    --out checkpoints/sw_${name}.pt --device cuda > outputs/sweep/${name}.log 2>&1
  local bv=$(grep -oE "best_val=[0-9.]+" outputs/sweep/${name}.log | tail -1 | cut -d= -f2)
  echo "$name,$lr,$loss,$crop,$epochs,$bv" >> $LB
  echo "  -> $name best_val=$bv"
}

run_cfg c1_lr3e4_char256    3e-4 charbonnier 256 400 42
run_cfg c2_lr1e3_char256    1e-3 charbonnier 256 400 42
run_cfg c3_lr1e4_char256    1e-4 charbonnier 256 400 42
run_cfg c4_lr3e4_L1_256     3e-4 l1          256 400 42
run_cfg c5_lr3e4_char384    3e-4 charbonnier 384 400 42
run_cfg c6_lr6e4_char320    6e-4 charbonnier 320 400 42
run_cfg c7_lr3e4_char256_s7 3e-4 charbonnier 256 400 7

echo "===== SWEEP ROUND DONE ====="
cat $LB
BEST=$(tail -n +2 $LB | sort -t, -k6 -g | head -1)
echo "BEST row: $BEST"
BLR=$(echo $BEST | cut -d, -f2); BLOSS=$(echo $BEST | cut -d, -f3); BCROP=$(echo $BEST | cut -d, -f4)
BEXTRA=""; [ "$BLOSS" = "charbonnier" ] && BEXTRA="--charbonnier"
echo "===== FINAL LONG TRAIN (best cfg, 1500ep, seeds 42/7/123) ====="
for s in 42 7 123; do
  $PY -m ai_engine.specialists.auto_enhance.train_gpu \\
    --epochs 1500 --batch-size 8 --crop $BCROP --lr $BLR $BEXTRA \\
    --num-workers 8 --seed $s --save-every 300 \\
    --out checkpoints/final_s${s}.pt --device cuda > outputs/sweep/final_s${s}.log 2>&1
  fbv=$(grep -oE "best_val=[0-9.]+" outputs/sweep/final_s${s}.log | tail -1 | cut -d= -f2)
  echo "final_s${s},$BLR,$BLOSS,$BCROP,1500,$fbv" >> $LB
  echo "  -> final_s${s} best_val=$fbv"
done
echo "===== ALL DONE ====="
tail -n +2 $LB | sort -t, -k6 -g
echo "SWEEP_COMPLETE_MARKER"
"""

with open("outputs/sweep_box.sh", "w", newline="\n", encoding="utf-8") as f:
    f.write(SWEEP)
print("local sweep_box.sh bytes:", len(SWEEP))

put("outputs/sweep_box.sh", "/workspace/autohdr/sweep_box.sh")
print("uploaded")

rc, out, err = run("cd /workspace/autohdr && pkill -f train_gpu 2>/dev/null; tmux kill-server 2>/dev/null; sleep 1; chmod +x sweep_box.sh; wc -c sweep_box.sh; tmux new-session -d -s sweep 'cd /workspace/autohdr && bash sweep_box.sh 2>&1 | tee outputs/sweep/sweep_all.log'; sleep 3; tmux ls", timeout=30)
print(out)
if err.strip():
    print("STDERR:", err[:400])
print("EXIT", rc)
