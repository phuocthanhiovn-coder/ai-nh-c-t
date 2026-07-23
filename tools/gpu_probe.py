from tools.gpu_ssh import run

CMD = r'''
echo "== conda torch/cuda =="
/opt/conda/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
PY
echo "== deps =="
/opt/conda/bin/python -m pip install -q "numpy<2" opencv-python-headless scikit-image 2>&1 | tail -2
/opt/conda/bin/python - <<'PY'
import numpy, cv2
print("numpy", numpy.__version__, "cv2", cv2.__version__)
PY
mkdir -p /workspace/autohdr && echo "workspace ready"
nproc; df -h / | tail -1
'''
rc, out, err = run(CMD, timeout=300)
print(out)
if err.strip():
    print("STDERR:", err[:400])
