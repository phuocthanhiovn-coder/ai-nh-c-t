"""Render [before | model output | AutoHDR target] cho N anh val (unseen), tu checkpoint da tai ve.
Dung: python -m tools.eval_local <checkpoint.pt> <out.jpg> [N]
"""
import sys
import os
import hashlib
import glob
import cv2
import numpy as np
import torch

sys.path.insert(0, "C:/Users/Administrator/Desktop/autohdr")
cv2.setNumThreads(2)
from ai_engine.specialists.auto_enhance.model import HDRNet


def val_files(pairs="C:/Users/Administrator/Desktop/autohdr/data/pairs", val_frac=0.12):
    befs = sorted(os.path.basename(p) for p in glob.glob(pairs + "/before/*.jpg"))
    val = [f for f in befs if (int(hashlib.md5(f.encode()).hexdigest(), 16) % 1000) / 1000.0 < val_frac]
    return pairs, val


def run(ckpt, out_path, n=8):
    dev = "cpu"
    model = HDRNet().to(dev)
    sd = torch.load(ckpt, map_location=dev)
    model.load_state_dict(sd)
    model.eval()
    pairs, val = val_files()
    val = val[:n]
    rows = []
    l1s = []
    for f in val:
        b = cv2.imread(f"{pairs}/before/{f}")
        a = cv2.imread(f"{pairs}/after/{f}")
        if b is None or a is None:
            continue
        h, w = b.shape[:2]
        proxy = cv2.resize(b, (256, 256), interpolation=cv2.INTER_AREA)
        bt = torch.from_numpy(cv2.cvtColor(b, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        pt = torch.from_numpy(cv2.cvtColor(proxy, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            out, _ = model(pt, bt)
        o = (out.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        o = cv2.cvtColor(o, cv2.COLOR_RGB2BGR)
        l1s.append(float(np.abs(o.astype(float) - a.astype(float)).mean()))
        # resize a to match
        a2 = cv2.resize(a, (w, h))
        panel = np.hstack([b, o, a2])
        ph = 360
        pw = int(panel.shape[1] * ph / panel.shape[0])
        panel = cv2.resize(panel, (pw, ph))
        cv2.putText(panel, "BEFORE", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 255), 2)
        cv2.putText(panel, "MODEL", (pw // 3 + 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 255, 60), 2)
        cv2.putText(panel, "AUTOHDR", (2 * pw // 3 + 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 160, 60), 2)
        rows.append(panel)
    if rows:
        maxw = max(r.shape[1] for r in rows)
        rows = [cv2.copyMakeBorder(r, 0, 0, 0, maxw - r.shape[1], cv2.BORDER_CONSTANT) for r in rows]
        sheet = np.vstack(rows)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[eval] {ckpt}: mean L1 to AutoHDR target = {np.mean(l1s):.2f} over {len(l1s)} val imgs")
    print(f"[eval] contact sheet -> {out_path}")


if __name__ == "__main__":
    ckpt = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "outputs/eval_local.jpg"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    run(ckpt, out, n)
