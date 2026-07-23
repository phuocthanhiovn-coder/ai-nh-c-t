from tools.gpu_ssh import run, put

# re-upload fixed train_sweep, then smoke
put("ai_engine/specialists/auto_enhance/gpu/train_sweep.py",
    "/workspace/autohdr/ai_engine/specialists/auto_enhance/gpu/train_sweep.py")
print("re-uploaded fixed train_sweep.py")

rc, out, err = run(
    "cd /workspace/autohdr && PYTHONPATH=. timeout 400 /opt/conda/bin/python kit_smoke_driver.py 2>&1 | tail -30",
    timeout=430,
)
print(out)
if err.strip():
    print("STDERR:", err[:800])
print("EXIT", rc)
