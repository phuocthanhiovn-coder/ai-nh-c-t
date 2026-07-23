"""Xem trang thai train tren box: process, log cuoi, GPU util."""
from tools.gpu_ssh import run

CMD = (
    "pgrep -af launch_chf; "
    "tail -3 /root/autohdr/train_chf.log 2>/dev/null; "
    "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
)

if __name__ == "__main__":
    rc, out, err = run(CMD, timeout=45)
    print(out)
    if err.strip():
        print("STDERR:", err[-300:])
