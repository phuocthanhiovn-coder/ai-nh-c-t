"""Eval nhanh tren box: render [before | AI | after(target)] cho N anh VAL.
Dung BGR xuyen suot (KHOP train_sweep) + split cua train_sweep (khong can train_gpu).
Chay: python -m ai_engine.specialists.auto_enhance.gpu.eval_box --ckpt <pt> --n 10 --out eval.jpg
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


def load_cfg(meta_path, device):
    cfg = dict(_DEF)
    if not os.path.exists(meta_path):
        return cfg
    try:
        m = torch.load(meta_path, map_location=device)
    except Exception:
        return cfg
    mk = m.get("model_kwargs") if isinstance(m, dict) else None
    if isinstance(mk, dict):
        for k in _KEYS:
            if k in mk and mk[k] is not None:
                cfg[k] = int(mk[k])
    return cfg


def cap(img, w):
    if w and img.shape[1] > w:
        h = int(round(img.shape[0] * w / img.shape[1]))
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img


def run_one(model, before_bgr, device, proc_w):
    before_bgr = cap(before_bgr, proc_w)
    # BGR [0,1] -> tensor (KHONG chuyen RGB; train_sweep train o BGR)
    t = torch.from_numpy(before_bgr.transpose(2, 0, 1).copy()).float().unsqueeze(0) / 255.0
    t = t.to(device)
    proxy = HDRNetV2.make_proxy(t, model.proxy_res)
    with torch.no_grad():
        out, _ = model(proxy, t)
    out = (out.squeeze(0).clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return before_bgr, out


def label(img, txt, w):
    h = int(round(img.shape[0] * w / img.shape[1]))
    s = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    strip = np.full((28, w, 3), 20, np.uint8)
    cv2.putText(strip, txt, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([strip, s])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="eval.jpg")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--proc-width", type=int, default=768)
    ap.add_argument("--cell-width", type=int, default=460)
    args = ap.parse_args()

    cv2.setNumThreads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_cfg(args.ckpt + ".meta", device)
    print("[*] cfg", cfg, "device", device)

    model = HDRNetV2(**cfg).to(device)
    state = torch.load(args.ckpt, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    _, val = split_filenames(args.data_dir, 0.12)
    val = sorted(val)[: args.n]
    bd = os.path.join(args.data_dir, "before")
    ad = os.path.join(args.data_dir, "after")

    rows, l1s = [], []
    for name in val:
        b = cv2.imread(os.path.join(bd, name))
        a = cv2.imread(os.path.join(ad, name))
        if b is None or a is None:
            continue
        b_proc, out = run_one(model, b, device, args.proc_width)
        a_m = cv2.resize(a, (b_proc.shape[1], b_proc.shape[0]), interpolation=cv2.INTER_AREA)
        l1 = float(np.mean(np.abs(out.astype(np.float32) - a_m.astype(np.float32)) / 255.0))
        l1s.append(l1)
        cw = args.cell_width
        row = np.hstack([
            label(b_proc, f"BEFORE {name[:24]}", cw),
            np.full((label(b_proc, "", cw).shape[0], 4, 3), 40, np.uint8),
            label(out, f"AI  L1={l1:.3f}", cw),
            np.full((label(b_proc, "", cw).shape[0], 4, 3), 40, np.uint8),
            label(a_m, "AUTOHDR (target)", cw),
        ])
        rows.append(row)
        print(f"    {name:<28} L1={l1:.4f}")

    if not rows:
        print("[!] khong render duoc")
        return
    maxw = max(r.shape[1] for r in rows)
    rows = [r if r.shape[1] == maxw else np.hstack([r, np.full((r.shape[0], maxw - r.shape[1], 3), 20, np.uint8)]) for r in rows]
    sheet = np.vstack([np.vstack([r, np.full((5, maxw, 3), 40, np.uint8)]) for r in rows])
    cv2.imwrite(args.out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[+] {args.out} shape={sheet.shape} mean_L1={np.mean(l1s):.4f}")


if __name__ == "__main__":
    main()
