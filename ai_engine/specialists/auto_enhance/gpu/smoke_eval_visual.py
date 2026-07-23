"""
CPU smoke test for eval_visual.py.

Builds a fresh TINY random-init HDRNetV2, saves its state_dict + a .meta
sidecar (so config discovery is exercised), then runs eval_visual over 4
held-out val images and verifies the contact sheet was written and is a
valid, non-empty image (cv2.imread shape). L1/dE will be BAD (random init) -
that only tests the harness, not quality.
"""
import os
import subprocess
import sys

import cv2
import torch

from .model_v2 import HDRNetV2

cv2.setNumThreads(2)
torch.set_num_threads(2)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SCRATCH = os.path.join(REPO, "outputs", "sweep_eval")
CKPT = os.path.join(SCRATCH, "_smoke_rand.pt")
OUT = os.path.join(SCRATCH, "_smoke_rand.jpg")

# Tiny config -> fast on CPU. Non-default values prove .meta discovery works.
CFG = dict(grid_bins=4, grid_size=8, proxy_res=128, width=8, guidance_hidden=8)


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    torch.manual_seed(0)

    model = HDRNetV2(**CFG)
    torch.save(model.state_dict(), CKPT)
    # Nest the knobs under 'args' like a trainer would, to exercise the probe.
    torch.save({"epoch": 1, "best_val": 9.99, "args": dict(CFG)}, CKPT + ".meta")
    print(f"[smoke] wrote random-init ckpt {CKPT} + .meta  cfg={CFG}")

    cmd = [
        sys.executable, "-m",
        "ai_engine.specialists.auto_enhance.gpu.eval_visual",
        "--ckpt", CKPT, "--n", "4", "--out", OUT,
        "--proc-width", "512", "--cell-width", "384", "--device", "cpu",
    ]
    print("[smoke] running:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    print("---- eval_visual stdout ----")
    print(r.stdout)
    if r.stderr.strip():
        print("---- eval_visual stderr ----")
        print(r.stderr)
    if r.returncode != 0:
        print(f"[smoke] FAIL: eval_visual exited {r.returncode}")
        sys.exit(1)

    assert os.path.exists(OUT), f"contact sheet not written: {OUT}"
    img = cv2.imread(OUT)
    assert img is not None, f"cv2.imread returned None (invalid image): {OUT}"
    assert img.ndim == 3 and img.shape[2] == 3, f"bad shape: {img.shape}"
    assert img.shape[0] > 0 and img.shape[1] > 0, f"empty image: {img.shape}"
    print(f"[smoke] PASS: contact sheet written {OUT}  shape={img.shape}")


if __name__ == "__main__":
    main()
