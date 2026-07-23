"""Giao anh FULL-RES cho khach: ap operator quan quan o DO PHAN GIAI GOC.
Xuat JPEG q100 + chroma 4:4:4 (khong mat mau) -> file nhieu MB, sac net.
CHUA het lo khach che "anh nen vai tram KB / vo pixel".

Chay: python -m ai_engine.specialists.auto_enhance.gpu.render_delivery \
        --ckpt checkpoints/sweep/<champion>.pt --outdir delivery_out --val-only --n 20
"""
import os
import argparse

import cv2
import numpy as np
import torch

from .model_v2 import HDRNetV2
from .train_sweep import split_filenames
from .finish_grade import grade_auto

_KEYS = ("grid_bins", "grid_size", "proxy_res", "width", "guidance_hidden")
_DEF = dict(grid_bins=8, grid_size=16, proxy_res=256, width=16, guidance_hidden=16)


def load_cfg(meta_path, device):
    cfg = dict(_DEF)
    if os.path.exists(meta_path):
        try:
            m = torch.load(meta_path, map_location=device)
            mk = m.get("model_kwargs") if isinstance(m, dict) else None
            if isinstance(mk, dict):
                for k in _KEYS:
                    if k in mk and mk[k] is not None:
                        cfg[k] = int(mk[k])
        except Exception:
            pass
    return cfg


def apply_fullres(model, before_bgr, device):
    """Ap operator o DUNG do phan giai goc (operator doc lap phan giai)."""
    t = torch.from_numpy(before_bgr.transpose(2, 0, 1).copy()).float().unsqueeze(0) / 255.0
    t = t.to(device)
    proxy = HDRNetV2.make_proxy(t, model.proxy_res)
    with torch.no_grad():
        out, _ = model(proxy, t)
    out = (out.squeeze(0).clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return out


def save_hq(path, img):
    """JPEG q100 + chroma 4:4:4 (khong subsample -> khong mat mau)."""
    params = [cv2.IMWRITE_JPEG_QUALITY, 100]
    if hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR"):
        params += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
    cv2.imwrite(path, img, params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--outdir", default="delivery_out")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--val-only", action="store_true",
                    help="chi dung anh VAL (model chua tung thay -> demo trung thuc)")
    ap.add_argument("--biggest", action="store_true",
                    help="uu tien anh do phan giai cao nhat")
    ap.add_argument("--side-by-side", action="store_true",
                    help="them ban ghep before|after de so sanh")
    ap.add_argument("--grade", action="store_true",
                    help="ap tang finish (bao hoa/am/tuong phan/khu nhieu) - scene-aware")
    args = ap.parse_args()

    cv2.setNumThreads(6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_cfg(args.ckpt + ".meta", device)
    print(f"[*] cfg {cfg} device {device}")

    model = HDRNetV2(**cfg).to(device)
    state = torch.load(args.ckpt, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    bd = os.path.join(args.data_dir, "before")
    if args.val_only:
        _, names = split_filenames(args.data_dir, 0.12)
    else:
        names = os.listdir(bd)
    names = [n for n in names if os.path.exists(os.path.join(bd, n))]

    if args.biggest:
        def side(n):
            im = cv2.imread(os.path.join(bd, n))
            return max(im.shape[:2]) if im is not None else 0
        names = sorted(names, key=side, reverse=True)
    else:
        names = sorted(names)
    names = names[: args.n]

    os.makedirs(args.outdir, exist_ok=True)
    if args.side_by_side:
        os.makedirs(args.outdir + "_compare", exist_ok=True)

    total_kb = 0
    for name in names:
        b = cv2.imread(os.path.join(bd, name))
        if b is None:
            continue
        out = apply_fullres(model, b, device)
        if args.grade:
            out = grade_auto(out, name)
        stem = os.path.splitext(name)[0]
        outp = os.path.join(args.outdir, f"{stem}_AI.jpg")
        save_hq(outp, out)
        kb = os.path.getsize(outp) / 1024
        total_kb += kb
        h, w = out.shape[:2]
        print(f"  {stem}_AI.jpg  {w}x{h}  {kb:.0f} KB")
        if args.side_by_side:
            cmp = np.hstack([b, np.full((b.shape[0], 8, 3), 255, np.uint8), out])
            save_hq(os.path.join(args.outdir + "_compare", f"{stem}_cmp.jpg"), cmp)
    print(f"[+] {len(names)} anh -> {args.outdir}  (tong {total_kb/1024:.1f} MB, TB {total_kb/max(1,len(names)):.0f} KB/anh)")


if __name__ == "__main__":
    main()
