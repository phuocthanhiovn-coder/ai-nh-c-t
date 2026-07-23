"""
Task GPU - Visual QC harness for trained auto_enhance HDRNetV2 checkpoints.

WHY: val L1 alone does not tell the architect whether the model matched the
AutoHDR "look". This renders a contact sheet of

        [ before | model output | AutoHDR target(after) ]

rows over N HELD-OUT val images so the gu can be judged BY EYE, and also prints
a quick numeric gu-match score (mean L1 + mean CIE76 Lab-dE) between output and
target across the val set.

Standalone add-on. Does NOT modify ..model / ..train / ..infer / ..dataset. It
reuses two canonical pieces so it never drifts from training:
  - HDRNetV2                 (.model_v2)   - the operator-not-pixel net.
  - split_filenames(...)     (..train_gpu) - the md5(filename)%1000 hash split.
    Any sweep trainer keys its val set off this SAME function, so eval judges on
    exactly the images training never saw. (No train_sweep.py exists yet at time
    of writing; when it lands it uses this same split, so this stays correct.)

OPERATOR CONTRACT (immutable): the net emits a bilateral grid of affine coeffs
and slices it differentiably; output size == input size. We run inference at a
capped processing width (--proc-width) purely to bound CPU memory / time for QC
- the operator is resolution-independent, so a 1024-px QC view is representative
of the full-res result. Metrics are computed at that processing resolution
(before and after are downscaled identically, so the comparison stays fair).

CUDA: eval runs under torch.no_grad(); on cuda it optionally autocasts for
speed. This box is CPU-only (torch 2.13.0+cpu) so the CUDA path is UNVERIFIED
locally - see the smoke report. AMP training wiring is a trainer concern, not
here.

Config discovery: the model shape (grid_bins/grid_size/proxy_res/width/
guidance_hidden) is read from the '<ckpt>.meta' sidecar if present, else v2
defaults. We probe the sidecar top-level and common nests (args/config/
model_config/model) so it works whatever the eventual sweep trainer stores.

CLI:
    python -m ai_engine.specialists.auto_enhance.gpu.eval_visual \
        --ckpt checkpoints/sweep/run1.pt --n 8 \
        --out outputs/sweep_eval/run1.jpg
"""
import os
import sys
import argparse

import cv2
import numpy as np
import torch

from .model_v2 import HDRNetV2
from .losses import bgr_to_lab

# Canonical hash split shared with the trainer(s).
from ..train_gpu import split_filenames


# Keys that parameterize HDRNetV2. Defaults mirror the model_v2 constructor.
_CFG_KEYS = ("grid_bins", "grid_size", "proxy_res", "width", "guidance_hidden")
_CFG_DEFAULTS = dict(grid_bins=8, grid_size=16, proxy_res=256, width=16, guidance_hidden=16)


# ---------------------------------------------------------------------------
# Meta sidecar -> model config
# ---------------------------------------------------------------------------
def infer_model_cfg(ckpt_path, device):
    """Return (cfg_dict, source_str). Reads '<ckpt>.meta' if present and mines
    the HDRNetV2 knobs from its top level or a nested args/config/model_config/
    model dict; anything missing falls back to the v2 default."""
    cfg = dict(_CFG_DEFAULTS)
    meta_path = ckpt_path + ".meta"
    if not os.path.exists(meta_path):
        return cfg, "defaults (no .meta sidecar)"

    try:
        meta = torch.load(meta_path, map_location=device)
    except Exception as e:  # noqa: BLE001 - a broken sidecar must not kill eval
        print(f"[!] Doc meta that bai ({meta_path}): {e} -> dung defaults")
        return cfg, "defaults (.meta unreadable)"

    if not isinstance(meta, dict):
        return cfg, "defaults (.meta not a dict)"

    # Candidate namespaces to search, most specific last so they win.
    candidates = [meta]
    for nest in ("args", "config", "model_config", "model", "hparams"):
        sub = meta.get(nest)
        if isinstance(sub, dict):
            candidates.append(sub)

    found = {}
    for ns in candidates:
        for k in _CFG_KEYS:
            if k in ns and ns[k] is not None:
                found[k] = ns[k]
    cfg.update({k: int(found[k]) for k in found})

    if found:
        src = f"{meta_path} (keys: {', '.join(sorted(found))})"
    else:
        src = f"{meta_path} (no model keys found -> defaults)"
    return cfg, src


def load_model(ckpt_path, cfg, device):
    model = HDRNetV2(**cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state and not any(
        k.startswith(("predictor", "guidance_net")) for k in state
    ):
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference + metrics for one pair
# ---------------------------------------------------------------------------
def _cap_width(img, proc_width):
    """Downscale so the WIDTH <= proc_width (area filter), keeping aspect.
    proc_width<=0 disables capping."""
    if proc_width and proc_width > 0 and img.shape[1] > proc_width:
        h, w = img.shape[:2]
        new_h = max(1, int(round(h * proc_width / w)))
        img = cv2.resize(img, (proc_width, new_h), interpolation=cv2.INTER_AREA)
    return img


def run_pair(model, before_bgr, after_bgr, device, proc_width, use_autocast):
    """Run the operator on `before` and return (out_bgr, after_bgr_matched,
    l1, de) all at the processing resolution. Inference follows the training
    convention (RGB, proxy via area-downsample)."""
    before_bgr = _cap_width(before_bgr, proc_width)
    # Match target to the processed input size so metrics compare like-for-like.
    if after_bgr.shape[:2] != before_bgr.shape[:2]:
        after_bgr = cv2.resize(after_bgr, (before_bgr.shape[1], before_bgr.shape[0]),
                               interpolation=cv2.INTER_AREA)

    before_rgb = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2RGB)
    full = torch.from_numpy(before_rgb.transpose(2, 0, 1).copy()).float().unsqueeze(0) / 255.0
    full = full.to(device)
    proxy = HDRNetV2.make_proxy(full, model.proxy_res)

    with torch.no_grad():
        if use_autocast:
            with torch.autocast(device_type="cuda"):
                out, _ = model(proxy, full)
            out = out.float()
        else:
            out, _ = model(proxy, full)

    out_rgb = (out.squeeze(0).clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255.0)
    out_rgb = out_rgb.astype(np.uint8)
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

    # Metrics on CPU tensors, BGR [0,1] (Lab helper expects BGR order).
    out_t = torch.from_numpy(out_bgr.transpose(2, 0, 1).copy()).float().unsqueeze(0) / 255.0
    aft_t = torch.from_numpy(after_bgr.transpose(2, 0, 1).copy()).float().unsqueeze(0) / 255.0
    l1 = torch.mean(torch.abs(out_t - aft_t)).item()

    lab_o = bgr_to_lab(out_t)
    lab_a = bgr_to_lab(aft_t)
    de = torch.sqrt(torch.clamp(((lab_o - lab_a) ** 2).sum(dim=1), min=0.0)).mean().item()

    return out_bgr, after_bgr, l1, de


# ---------------------------------------------------------------------------
# Contact sheet assembly
# ---------------------------------------------------------------------------
_STRIP_H = 26
_SEP = 4
_SEP_COLOR = 30
_BG = 18


def _labeled_cell(img_bgr, label, cell_w):
    """Resize img to width cell_w (keep aspect) and stack a dark caption strip
    on top. Returns a uint8 BGR cell of width cell_w."""
    h, w = img_bgr.shape[:2]
    cell_h = max(1, int(round(h * cell_w / w)))
    small = cv2.resize(img_bgr, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
    strip = np.full((_STRIP_H, cell_w, 3), _BG, np.uint8)
    cv2.putText(strip, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([strip, small])


def _row(before_bgr, out_bgr, after_bgr, name, l1, de, cell_w):
    """One contact-sheet row: [before | output | after] with a filename +
    per-image metric caption on each output cell."""
    cells = [
        _labeled_cell(before_bgr, f"BEFORE  {name}", cell_w),
        _labeled_cell(out_bgr, f"OUTPUT  L1={l1:.4f} dE={de:.2f}", cell_w),
        _labeled_cell(after_bgr, "AFTER (AutoHDR target)", cell_w),
    ]
    # Equalize heights (aspect should match, but guard rounding drift).
    max_h = max(c.shape[0] for c in cells)
    for i, c in enumerate(cells):
        if c.shape[0] != max_h:
            pad = np.full((max_h - c.shape[0], cell_w, 3), _BG, np.uint8)
            cells[i] = np.vstack([c, pad])
    vsep = np.full((max_h, _SEP, 3), _SEP_COLOR, np.uint8)
    return np.hstack([cells[0], vsep, cells[1], vsep, cells[2]])


def _title_bar(text, total_w):
    bar = np.full((34, total_w, 3), _BG, np.uint8)
    cv2.putText(bar, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return bar


def build_contact_sheet(rows, title):
    total_w = rows[0].shape[1]
    hsep = np.full((_SEP, total_w, 3), _SEP_COLOR, np.uint8)
    stacked = []
    for i, r in enumerate(rows):
        if i:
            stacked.append(hsep)
        stacked.append(r)
    body = np.vstack(stacked)
    return np.vstack([_title_bar(title, total_w), body])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Visual QC contact sheet for HDRNetV2 checkpoints")
    parser.add_argument("--ckpt", required=True, help="path to a .pt state_dict (reads <ckpt>.meta if present)")
    parser.add_argument("--n", type=int, default=8, help="number of held-out val images to render")
    parser.add_argument("--out", default=None,
                        help="output jpg path (default outputs/sweep_eval/<ckpt-stem>.jpg)")
    parser.add_argument("--data-dir", default="data/pairs")
    parser.add_argument("--val-frac", type=float, default=0.12,
                        help="must match the trainer's val_frac so the split lines up")
    parser.add_argument("--proc-width", type=int, default=1024,
                        help="cap input width for inference/metrics (<=0 = full-res)")
    parser.add_argument("--cell-width", type=int, default=512, help="panel cell width in px")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()

    # Box constraint: bounded CPU threads.
    cv2.setNumThreads(2)
    torch.set_num_threads(2)

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else "cpu" if args.device == "auto" else args.device)
    use_autocast = device.type == "cuda"
    print(f"[*] Device: {device}")

    if not os.path.exists(args.ckpt):
        print(f"[!] Khong tim thay checkpoint: {args.ckpt}")
        sys.exit(1)

    # --- model ---
    cfg, cfg_src = infer_model_cfg(args.ckpt, device)
    print(f"[*] Model cfg: {cfg}")
    print(f"[*] Cfg source: {cfg_src}")
    model = load_model(args.ckpt, cfg, device)

    # --- held-out val split (SAME hash as trainer) ---
    _, val_files = split_filenames(args.data_dir, args.val_frac)
    val_files = sorted(val_files)
    if not val_files:
        print(f"[!] Val split rong (val_frac={args.val_frac}, data-dir={args.data_dir}).")
        sys.exit(1)
    picked = val_files[: args.n]
    print(f"[+] {len(val_files)} val images total; rendering {len(picked)}: {picked}")

    before_dir = os.path.join(args.data_dir, "before")
    after_dir = os.path.join(args.data_dir, "after")

    # --- output path ---
    out_path = args.out
    if out_path is None:
        stem = os.path.splitext(os.path.basename(args.ckpt))[0]
        out_path = os.path.join("outputs", "sweep_eval", f"{stem}.jpg")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # --- per-image loop ---
    rows, l1s, des = [], [], []
    for name in picked:
        b = cv2.imread(os.path.join(before_dir, name))
        a = cv2.imread(os.path.join(after_dir, name))
        if b is None or a is None:
            print(f"[!] Bo qua (khong doc duoc cap): {name}")
            continue
        out_bgr, a_matched, l1, de = run_pair(model, b, a, device, args.proc_width, use_autocast)
        l1s.append(l1)
        des.append(de)
        # Row uses the processed 'before' so all three cells share resolution.
        b_proc = _cap_width(b, args.proc_width)
        rows.append(_row(b_proc, out_bgr, a_matched, name, l1, de, args.cell_width))
        print(f"    {name:<28} L1={l1:.4f}  dE={de:.2f}")

    if not rows:
        print("[!] Khong render duoc anh nao.")
        sys.exit(1)

    mean_l1 = float(np.mean(l1s))
    mean_de = float(np.mean(des))
    stem = os.path.splitext(os.path.basename(args.ckpt))[0]
    title = (f"{stem}  |  cfg={cfg}  |  n={len(rows)}  "
             f"|  mean L1={mean_l1:.4f}  mean Lab-dE={mean_de:.2f}")
    sheet = build_contact_sheet(rows, title)

    ok = cv2.imwrite(out_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg_quality)])
    if not ok:
        print(f"[!] cv2.imwrite that bai: {out_path}")
        sys.exit(1)

    print("=" * 68)
    print(f"[+] Contact sheet: {out_path}  (shape={sheet.shape})")
    print(f"[+] gu-match over {len(rows)} val imgs: mean L1={mean_l1:.4f}  mean Lab-dE={mean_de:.2f}")
    print("=" * 68)


if __name__ == "__main__":
    main()
