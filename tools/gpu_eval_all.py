"""Render contact sheet [before|AI|target] cho tat ca config trong sweep + keo ve local.
Chay SAU khi sweep xong: python -m tools.gpu_eval_all
"""
import os
from tools.gpu_ssh import run, get

REMOTE = "/workspace/autohdr"
NAMES = ["A_base", "B_color", "C_bigcrop", "D_bigmodel", "E_deepgrid"]
LOCAL = "C:/Users/Administrator/Desktop/autohdr/outputs/sweep_eval"
os.makedirs(LOCAL, exist_ok=True)
N = 10  # so anh val render moi config (nhat quan giua cac config vi split co dinh)


def main():
    # render tren box
    parts = [f"cd {REMOTE}", "mkdir -p outputs/eval"]
    for n in NAMES:
        parts.append(f"echo '=== eval {n} ==='")
        parts.append(
            f"test -f checkpoints/sweep/{n}.pt && "
            f"/opt/conda/bin/python -u -m ai_engine.specialists.auto_enhance.gpu.eval_box "
            f"--ckpt checkpoints/sweep/{n}.pt --n {N} --out outputs/eval/{n}.jpg "
            f"2>&1 | grep -E 'mean_L1|cfg|khong' || echo 'SKIP {n} (no ckpt)'")
    rc, out, err = run("\n".join(parts), timeout=600)
    print(out)
    if err.strip():
        print("STDERR:", err.strip()[:400])

    # keo ve
    print("\n=== keo contact sheet ve local ===")
    for n in NAMES:
        try:
            get(f"{REMOTE}/outputs/eval/{n}.jpg", f"{LOCAL}/{n}.jpg")
            sz = os.path.getsize(f"{LOCAL}/{n}.jpg")
            print(f"  {n}.jpg  {sz//1024} KB -> {LOCAL}/{n}.jpg")
        except Exception as e:
            print(f"  {n}.jpg FAIL: {e}")


if __name__ == "__main__":
    main()
