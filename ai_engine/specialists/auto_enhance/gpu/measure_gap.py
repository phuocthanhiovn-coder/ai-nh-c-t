"""Do CHINH XAC do lech giua anh AI (model) va anh AutoHDR (target) tren val set,
tach rieng TRONG NHA vs NGOAI TROI. Cho biet can keo bao hoa/am/tuong phan bao nhieu.
Chay: python -m ai_engine.specialists.auto_enhance.gpu.measure_gap --ckpt checkpoints/sweep/C_bigcrop.pt
"""
import os
import argparse

import cv2
import numpy as np
import torch

from .model_v2 import HDRNetV2
from .train_sweep import split_filenames

_KEYS = ("grid_bins", "grid_size", "proxy_res", "width", "guidance_hidden")
_DEF = dict(grid_bins=8, grid_size=16, proxy_res=256, width=16, guidance_hidden=16)

# ten file goi y ngoai troi
OUTDOOR = ("drone", "after_pool", "dji", "aerial", "_ext")


def load_cfg(meta, device):
    cfg = dict(_DEF)
    if os.path.exists(meta):
        try:
            m = torch.load(meta, map_location=device)
            mk = m.get("model_kwargs") if isinstance(m, dict) else None
            if isinstance(mk, dict):
                for k in _KEYS:
                    if k in mk and mk[k] is not None:
                        cfg[k] = int(mk[k])
        except Exception:
            pass
    return cfg


def run_model(model, before_bgr, device, cap=768):
    h, w = before_bgr.shape[:2]
    if max(h, w) > cap:
        s = cap / max(h, w)
        before_bgr = cv2.resize(before_bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(before_bgr.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device) / 255.0
    proxy = HDRNetV2.make_proxy(t, model.proxy_res)
    with torch.no_grad():
        out, _ = model(proxy, t)
    out = (out.squeeze(0).clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return before_bgr, out


def stats(bgr):
    """Tra ve (sat_mean, L_mean, L_std, a_mean, b_mean) - Lab & HSV."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    sat = hsv[:, :, 1].mean()
    L = lab[:, :, 0]
    a = lab[:, :, 1] - 128.0  # >0 do (red), <0 xanh la
    b = lab[:, :, 2] - 128.0  # >0 vang (am), <0 xanh duong (lanh)
    return sat, L.mean(), L.std(), a.mean(), b.mean()


def agg(rows):
    if not rows:
        return None
    A = np.array(rows)
    return A.mean(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n", type=int, default=97)
    args = ap.parse_args()

    cv2.setNumThreads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_cfg(args.ckpt + ".meta", device)
    model = HDRNetV2(**cfg).to(device)
    st = torch.load(args.ckpt, map_location=device)
    if isinstance(st, dict) and "state_dict" in st:
        st = st["state_dict"]
    model.load_state_dict(st); model.eval()

    _, val = split_filenames(args.data_dir, 0.12)
    val = sorted(val)[: args.n]
    bd = os.path.join(args.data_dir, "before")
    ad = os.path.join(args.data_dir, "after")

    # gap = target - ai  (duong = can KEO LEN)
    gap_in, gap_out = [], []
    for name in val:
        b = cv2.imread(os.path.join(bd, name)); a = cv2.imread(os.path.join(ad, name))
        if b is None or a is None:
            continue
        b_proc, out = run_model(model, b, device)
        a_m = cv2.resize(a, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_AREA)
        so = stats(out); sa = stats(a_m)
        # ty le sat & contrast, delta warmth/cast
        row = [
            sa[0] / max(1e-6, so[0]),      # sat ratio target/ai (>1 = AI thieu bao hoa)
            sa[2] / max(1e-6, so[2]),      # contrast(Lstd) ratio (>1 = AI thieu tuong phan)
            sa[3] - so[3],                 # a delta (target-ai): >0 AI thieu do (thien xanh la)
            sa[4] - so[4],                 # b delta: >0 AI thieu vang/am (thien lanh)
            sa[1] - so[1],                 # L delta: >0 AI toi hon; <0 AI sang qua
        ]
        (gap_out if name.lower().startswith(OUTDOOR) else gap_in).append(row)

    def show(tag, g):
        m = agg(g)
        if m is None:
            print(f"  {tag}: (khong co anh)")
            return
        print(f"  {tag} (n={len(g)}): sat x{m[0]:.3f} | contrast x{m[1]:.3f} | "
              f"a(do) {m[2]:+.2f} | b(am) {m[3]:+.2f} | L {m[4]:+.2f}")

    print(f"[*] GAP = target(AutoHDR) - AI  |  >0 nghia la AI CON THIEU, can keo len")
    show("TRONG NHA", gap_in)
    show("NGOAI TROI", gap_out)
    show("TAT CA", gap_in + gap_out)
    print("[i] sat x1.15 = can tang bao hoa 15%; b(am)+3 = can am them 3 don vi Lab; "
          "contrast x1.1 = tang tuong phan 10%")


if __name__ == "__main__":
    main()
