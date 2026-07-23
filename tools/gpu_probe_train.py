from tools.gpu_ssh import run

CMD = r'''
cd /workspace/autohdr
echo "== struct check =="
find ai_engine/specialists/auto_enhance -name 'train_gpu.py' -o -name 'model.py' | head
echo "== 2-epoch GPU timing probe =="
mkdir -p checkpoints outputs
PYTHONPATH=. /opt/conda/bin/python -m ai_engine.specialists.auto_enhance.train_gpu \
  --epochs 2 --batch-size 8 --crop 256 --val-frac 0.12 \
  --out checkpoints/probe.pt --device auto 2>&1 | tail -25
echo "== gpu util snapshot =="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
'''
rc, out, err = run(CMD, timeout=600)
print(out)
if err.strip():
    print("STDERR:", err[:1500])
print("EXIT", rc)
