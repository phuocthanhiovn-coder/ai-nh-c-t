from tools.gpu_ssh import run

# clean any lingering training + old sweep session
run("pkill -f train_gpu 2>/dev/null; tmux kill-session -t sweep 2>/dev/null; tmux kill-session -t train 2>/dev/null; sleep 2; echo cleaned", timeout=20)

# launch with the EXACT pattern that worked for baseline (tee, no kill-server)
rc, out, err = run("cd /workspace/autohdr && tmux new-session -d -s sweep 'cd /workspace/autohdr && bash sweep_box.sh 2>&1 | tee outputs/sweep/sweep_all.log'", timeout=20)
print("launch rc:", rc, "err:", err[:200])

# verify
rc2, out2, err2 = run("sleep 5; tmux ls; cd /workspace/autohdr; pgrep -af train_gpu | head -1; tail -3 outputs/sweep/sweep_all.log 2>/dev/null; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader", timeout=40)
print("=== verify ==="); print(out2); print("err:", err2[:200])
