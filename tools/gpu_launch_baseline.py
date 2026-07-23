from tools.gpu_ssh import run

# Step 1: create the run script + ensure tmux
CMD1 = r'''
cd /workspace/autohdr
which tmux >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq tmux) >/dev/null 2>&1
which tmux && echo "tmux ok"
cat > run_baseline.sh <<'SH'
#!/bin/bash
cd /workspace/autohdr
export PYTHONPATH=.
/opt/conda/bin/python -m ai_engine.specialists.auto_enhance.train_gpu \
  --epochs 800 --batch-size 8 --crop 256 --lr 3e-4 --num-workers 8 --charbonnier \
  --save-every 25 --out checkpoints/baseline.pt --device cuda
SH
chmod +x run_baseline.sh
echo "script bytes: $(wc -c < run_baseline.sh)"
'''
rc, out, err = run(CMD1, timeout=180)
print("=== setup ===")
print(out)
if err.strip():
    print("STDERR:", err[:400])

# Step 2: launch in tmux (returns immediately)
CMD2 = r'''
cd /workspace/autohdr
tmux kill-session -t train 2>/dev/null || true
tmux new-session -d -s train 'cd /workspace/autohdr && bash run_baseline.sh 2>&1 | tee outputs/baseline_train.log'
sleep 2
tmux ls
'''
rc2, out2, err2 = run(CMD2, timeout=30)
print("=== launch ===")
print(out2)
if err2.strip():
    print("STDERR:", err2[:400])
print("EXIT", rc2)
