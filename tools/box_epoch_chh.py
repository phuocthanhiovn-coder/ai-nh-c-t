"""In 1 dong trang thai train CH_H tren box (cho monitor local goi dinh ky)."""
from tools.gpu_ssh import run

if __name__ == "__main__":
    try:
        rc, out, err = run(
            "alive=$(pgrep -c -f 'tools.launch_chh'); "
            "line=$(grep -E 'Epoch|Done|RESULT|Error|Traceback|out of memory|Killed' "
            "/root/autohdr/train_chh.log | tail -1); "
            "echo \"alive=$alive | $line\"",
            timeout=40,
        )
        print(out.strip())
    except Exception as e:
        print(f"SSH_FAIL {type(e).__name__}: {e}")
