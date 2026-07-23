import os
from tools.gpu_ssh import run, put

LOCAL = "ai_engine/specialists/auto_enhance/gpu"
REMOTE = "/workspace/autohdr/ai_engine/specialists/auto_enhance/gpu"

run(f"mkdir -p {REMOTE}", timeout=20)
n = 0
for f in os.listdir(LOCAL):
    if f.endswith(".py"):
        put(os.path.join(LOCAL, f), f"{REMOTE}/{f}")
        n += 1
print(f"uploaded {n} py files to box gpu/")

# quick import check on box (conda python)
rc, out, err = run(
    "cd /workspace/autohdr && PYTHONPATH=. /opt/conda/bin/python -c "
    "\"import ai_engine.specialists.auto_enhance.gpu.model_v2 as m; "
    "import ai_engine.specialists.auto_enhance.gpu.losses as l; "
    "import ai_engine.specialists.auto_enhance.gpu.train_sweep as t; "
    "print('kit import OK')\"",
    timeout=60,
)
print(out)
if err.strip():
    print("STDERR:", err[:600])
