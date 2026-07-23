from tools.gpu_ssh import run
import time

# 1) show GPU processes + kill stale python, free GPU
CMD1 = r'''
echo "== nvidia-smi procs =="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || echo none
echo "== python procs =="
pgrep -af python | head
echo "== killing stale =="
tmux kill-server 2>/dev/null; pkill -9 -f retrain178 2>/dev/null; pkill -9 -f train_sweep 2>/dev/null; pkill -9 -f phase2 2>/dev/null
sleep 3
nvidia-smi --query-gpu=memory.used --format=csv,noheader
'''
rc, out, err = run(CMD1, timeout=60)
print("=== cleanup ==="); print(out)

# 2) run driver to a logfile in background via setsid, wait, read log
CMD2 = r'''
cd /workspace/autohdr
setsid bash -c 'PYTHONPATH=. /opt/conda/bin/python -u retrain178_driver.py > outputs/sweep/r178.log 2>&1' </dev/null >/dev/null 2>&1 &
echo launched
'''
rc2, out2, err2 = run(CMD2, timeout=20)
print("=== launch ==="); print(out2, err2[:200])
