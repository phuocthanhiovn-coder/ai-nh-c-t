"""Train HybridNet (model ghép 30/07) — tái dùng hạ tầng train_sweep.

Khác train_nm: thân NAFNet đã pretrain nên lr THẤP (1e-5 ~ 5e-5), và có tùy
chọn đóng băng thân vài epoch đầu để 2 đầu operator ổn định trước.

Box/Colab: python -u -m tools.launch_hybrid
"""
import csv
import os
import time

import torch
from torch.utils.data import DataLoader

from .losses import CombinedLoss
from .model_hybrid import HybridNet
from .train_sweep import (CropPairDataset, SWEEP_CSV_DIR, _run_epoch,
                          atomic_torch_save, lr_for_epoch, meta_path_for,
                          split_filenames)


def train_hybrid(cfg: dict) -> dict:
    # FIX 30/07: thieu gieo seed -> A/B "1 bien duy nhat" bi nhieu boi khoi tao va
    # thu tu crop ngau nhien (luat L2 task 27 bi vo hieu hoa).
    import random as _rnd
    import numpy as _np
    _seed = int(cfg.get("seed", 1234))
    _rnd.seed(_seed); _np.random.seed(_seed); torch.manual_seed(_seed)
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    data_dir = cfg["data_dir"]
    crop = int(cfg.get("crop", 512))
    proxy_res = int(cfg.get("proxy_res", 512))
    batch_size = int(cfg.get("batch_size", 2))
    epochs = int(cfg.get("epochs", 120))
    lr = float(cfg.get("lr", 3e-5))
    val_frac = float(cfg.get("val_frac", 0.12))
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    num_workers = int(cfg.get("num_workers", 2))
    out_path = cfg.get("out", "checkpoints/sweep/HY_A.pt")
    loss_cfg = dict(cfg.get("loss") or {"w_l1": 1.0})
    freeze_epochs = int(cfg.get("freeze_trunk_epochs", 5))
    kw = dict(width=int(cfg.get("width", 32)), lut_dim=int(cfg.get("lut_dim", 17)),
              n_basis=int(cfg.get("n_basis", 6)), proxy_res=proxy_res,
              gain_clip=float(cfg.get("gain_clip", 3.0)),
              gain_blur=int(cfg.get("gain_blur", 8)),
              nafnet_ckpt=cfg.get("nafnet_ckpt"))

    train_files, val_files = split_filenames(data_dir, val_frac)
    print(f"[HY] {len(train_files)} train / {len(val_files)} val", flush=True)
    train_ds = CropPairDataset(data_dir, train_files, crop, proxy_res, is_train=True,
                               cache_ram=True, cache_cap=int(cfg.get("cache_cap", 120)))
    val_ds = (CropPairDataset(data_dir, val_files, crop, proxy_res, is_train=False,
                              cache_ram=True,
                              cache_cap=int(cfg.get('cache_cap', 1280))) if val_files else None)
    lk = dict(num_workers=num_workers, drop_last=False, pin_memory=device.type == "cuda")
    if num_workers > 0:
        lk["persistent_workers"] = True
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **lk)
    val_loader = (DataLoader(val_ds, batch_size=batch_size, shuffle=False, **lk)
                  if val_ds is not None else None)

    model = HybridNet(**kw).to(device)
    print(f"[HY] params={sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    lab_weights = tuple(loss_cfg.get("lab_weights", (1.0, 1.0, 1.0)))
    criterion = CombinedLoss(
        w_l1=float(loss_cfg.get("w_l1", 0.0)), w_char=float(loss_cfg.get("w_char", 0.0)),
        w_lab=float(loss_cfg.get("w_lab", 0.0)), w_perc=float(loss_cfg.get("w_perc", 0.0)),
        lab_weights=lab_weights, w_hi=float(loss_cfg.get("w_hi", 0.0)),
        hi_gamma=float(loss_cfg.get("hi_gamma", 2.0)), w_dark=float(loss_cfg.get("w_dark", 0.0)),
        dark_thresh=float(loss_cfg.get("dark_thresh", 0.28)),
        w_color=float(loss_cfg.get("w_color", 0.0)), w_lc=float(loss_cfg.get("w_lc", 0.0)),
    ).to(device)

    w_tv = float(cfg.get("lambda_smooth", 1e-4))
    w_mn = float(cfg.get("lambda_monotonicity", 10.0))

    class _WithLutReg(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, out, tgt):
            # _run_epoch cua train_sweep unpack `loss, terms = criterion(...)`
            # -> PHAI tra CAP (fix 30/07, cung ho voi loi cell E5:
            # "'tuple' object has no attribute 'backward'").
            loss, terms = self.base(out, tgt)
            tv, mn = model.lut_regularizers()
            total = loss + w_tv * tv + w_mn * mn
            terms = dict(terms or {})
            terms.update({"lut_tv": float(tv.detach()), "lut_mn": float(mn.detach())})
            return total, terms

    reg_crit = _WithLutReg(criterion).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    warmup = min(5, max(1, epochs // 10))

    os.makedirs(SWEEP_CSV_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    history_csv = os.path.join(SWEEP_CSV_DIR,
                               os.path.splitext(os.path.basename(out_path))[0] + ".csv")
    with open(history_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_total", "val_total", "val_l1", "lr", "sec"])

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        # dong bang than may epoch dau -> 2 dau operator on dinh truoc, khong pha
        # kien thuc pretrain BDS
        frozen = epoch <= freeze_epochs
        for p in model.trunk.parameters():
            p.requires_grad = not frozen
        cur_lr = lr_for_epoch(epoch, epochs, lr, warmup)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr
        t0 = time.time()
        tr, _ = _run_epoch(model, train_loader, device, reg_crit, optimizer, scaler, use_amp, True)
        if val_loader is not None:
            # FIX 30/07: val PHAI dung loss SACH. Dung reg_crit thi best_val co the
            # "tot len" chi vi basis LUT co lai (w_mn=10, khong phu thuoc anh) ->
            # checkpoint duoc chon theo regularizer chu khong theo chat luong anh.
            vt, vl = _run_epoch(model, val_loader, device, criterion, optimizer, scaler, use_amp, False)
        else:
            vt, vl = float("nan"), float("nan")
        sec = time.time() - t0
        tag = " [than dong bang]" if frozen else ""
        print(f"  Epoch {epoch:03d}/{epochs:03d} | train={tr:.6f} val={vt:.6f} "
              f"val_l1={vl:.6f} lr={cur_lr:.3e} {sec:.1f}s{tag}", flush=True)
        with open(history_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{tr:.6f}", f"{vt:.6f}", f"{vl:.6f}",
                                    f"{cur_lr:.8f}", f"{sec:.3f}"])
        if (val_loader is None) or (vt < best_val):
            best_val = vt if val_loader is not None else best_val
            atomic_torch_save(model.state_dict(), out_path)
            atomic_torch_save({"epoch": epoch, "best_val": best_val, "cfg": cfg,
                               "hybrid_kwargs": {k: v for k, v in kw.items()
                                                 if k != "nafnet_ckpt"},
                               "loss": loss_cfg}, meta_path_for(out_path))
    if not os.path.exists(out_path):
        atomic_torch_save(model.state_dict(), out_path)
    return {"best_val": best_val, "ckpt": out_path, "history_csv": history_csv}
