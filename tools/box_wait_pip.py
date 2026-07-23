"""Cho pip tren box cai xong (poll 20s) roi bao ket qua."""
import time

from tools.gpu_ssh import run

CHECK = ("tail -1 /root/pip.log; "
         "python3 -c 'import torch, cv2; print(\"DEPS_OK\", torch.__version__, torch.cuda.is_available())' 2>&1 | tail -1")

if __name__ == "__main__":
    for i in range(45):
        rc, out, err = run(CHECK, timeout=60)
        line = out.strip().splitlines()[-1] if out.strip() else ""
        if "DEPS_OK" in line:
            print(out.strip())
            break
        time.sleep(20)
    else:
        print("TIMEOUT — trang thai cuoi:")
        print(out)
