"""Upload + chay cmp_chf.py tren box, in ket qua."""
import os

from tools.gpu_ssh import run, put

if __name__ == "__main__":
    put(r"C:\Users\ADMINI~1\AppData\Local\Temp\claude\C--Users-Administrator-Desktop-autohdr\0a5f658e-9466-4be1-94b2-80fedeebb6c9\scratchpad\cmp_chf.py", "/root/autohdr/cmp_chf.py")
    rc, out, err = run("cd /root/autohdr && python3 cmp_chf.py 2>&1 | tail -6", timeout=900)
    print(out)
    if err.strip():
        print("STDERR:", err[-300:])
