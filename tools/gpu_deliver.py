"""Xuat bo anh giao khach (full-res q100 4:4:4) tu 1 checkpoint + keo ve local.
Chay: python -m tools.gpu_deliver <ckpt_name>   (vd: C_bigcrop)
"""
import os
import sys
from tools.gpu_ssh import run, get

REMOTE = "/workspace/autohdr"
LOCAL_ROOT = "C:/Users/Administrator/Desktop/autohdr/delivery"


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "C_bigcrop"
    n = sys.argv[2] if len(sys.argv) > 2 else "24"
    outdir = f"delivery_{ckpt}"
    cmpdir = outdir + "_compare"

    # 1) render tren box
    cmd = (f"cd {REMOTE} && rm -rf {outdir} {cmpdir} && "
           f"/opt/conda/bin/python -u -m ai_engine.specialists.auto_enhance.gpu.render_delivery "
           f"--ckpt checkpoints/sweep/{ckpt}.pt --val-only --biggest --side-by-side --grade "
           f"--n {n} --outdir {outdir} 2>&1 | tail -30")
    rc, out, err = run(cmd, timeout=600)
    print(out)
    if err.strip():
        print("STDERR:", err.strip()[:400])

    # 2) liet ke + keo ve
    local_ai = f"{LOCAL_ROOT}/{ckpt}"
    local_cmp = f"{LOCAL_ROOT}/{ckpt}_compare"
    os.makedirs(local_ai, exist_ok=True)
    os.makedirs(local_cmp, exist_ok=True)

    rc, lst, _ = run(f"cd {REMOTE}/{outdir} && ls *.jpg 2>/dev/null", timeout=60)
    files = [f.strip() for f in lst.splitlines() if f.strip().endswith(".jpg")]
    print(f"\n=== keo {len(files)} anh AI ve {local_ai} ===")
    tot = 0
    for f in files:
        try:
            get(f"{REMOTE}/{outdir}/{f}", f"{local_ai}/{f}")
            kb = os.path.getsize(f"{local_ai}/{f}") / 1024
            tot += kb
        except Exception as e:
            print(f"  FAIL {f}: {e}")
    print(f"  tong {tot/1024:.1f} MB, TB {tot/max(1,len(files)):.0f} KB/anh")

    rc, lst2, _ = run(f"cd {REMOTE}/{cmpdir} && ls *.jpg 2>/dev/null", timeout=60)
    cfiles = [f.strip() for f in lst2.splitlines() if f.strip().endswith(".jpg")]
    print(f"=== keo {len(cfiles)} anh SO SANH ve {local_cmp} ===")
    for f in cfiles:
        try:
            get(f"{REMOTE}/{cmpdir}/{f}", f"{local_cmp}/{f}")
        except Exception:
            pass
    print(f"[+] XONG. Anh AI: {local_ai}  |  So sanh: {local_cmp}")


if __name__ == "__main__":
    main()
