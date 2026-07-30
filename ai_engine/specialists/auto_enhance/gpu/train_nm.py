"""Train NMNet (kiến trúc mới 29/07, E3) — tái dùng hạ tầng train_sweep,
cùng loss CH_N/CH_O để so sánh CÔNG BẰNG (1 biến: kiến trúc).

Box: cd /root/autohdr && setsid /opt/conda/bin/python -u -m tools.launch_nm > train_nm.log 2>&1 < /dev/null &
"""
import csv
import os
import time

import torch
from torch.utils.data import DataLoader

from .losses import CombinedLoss
from .model_nm import NMNet
from .train_sweep import (CropPairDataset, SWEEP_CSV_DIR, _run_epoch,
                          atomic_torch_save, lr_for_epoch, meta_path_for,
                          split_filenames)


def train_nm(cfg: dict) -> dict:
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    data_dir = cfg["data_dir"]
    crop = int(cfg.get("crop", 512))
    proxy_res = int(cfg.get("proxy_res", 448))
    batch_size = int(cfg.get("batch_size", 4))
    epochs = int(cfg.get("epochs", 300))
    lr = float(cfg.get("lr", 2e-4))
    val_frac = float(cfg.get("val_frac", 0.12))
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    num_workers = int(cfg.get("num_workers", 4))
    out_path = cfg.get("out", "checkpoints/sweep/NM_A.pt")
    loss_cfg = dict(cfg.get("loss") or {"w_l1": 1.0})
    nm_kwargs = dict(base=int(cfg.get("base", 32)), n_basis=int(cfg.get("n_basis", 6)),
                     lut_dim=int(cfg.get("lut_dim", 17)), proxy_res=proxy_res,
                     gain_amp=float(cfg.get("gain_amp", 1.4)))

    train_files, val_files = split_filenames(data_dir, val_frac)
    print(f"[NM] {len(train_files)} train / {len(val_files)} val")
    train_ds = CropPairDataset(data_dir, train_files, crop, proxy_res, is_train=True,
                               cache_ram=bool(cfg.get("cache_ram", True)),
                               cache_cap=int(cfg.get("cache_cap", 120)))
    val_ds = (CropPairDataset(data_dir, val_files, crop, proxy_res, is_train=False,
                              cache_ram=True, cache_cap=40) if val_files else None)
    pin = device.type == "cuda"
    lk = dict(num_workers=num_workers, drop_last=False, pin_memory=pin)
    if num_workers > 0:
        lk["persistent_workers"] = True
        lk["prefetch_factor"] = 4
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **lk)
    val_loader = (DataLoader(val_ds, batch_size=batch_size, shuffle=False, **lk)
                  if val_ds is not None else None)

    model = NMNet(**nm_kwargs).to(device)
    print(f"[NM] params={sum(p.numel() for p in model.parameters())} kwargs={nm_kwargs}")

    lab_weights = tuple(loss_cfg.get("lab_weights", (1.0, 1.0, 1.0)))
    criterion = CombinedLoss(
        w_l1=float(loss_cfg.get("w_l1", 0.0)), w_char=float(loss_cfg.get("w_char", 0.0)),
        w_lab=float(loss_cfg.get("w_lab", 0.0)), w_perc=float(loss_cfg.get("w_perc", 0.0)),
        lab_weights=lab_weights, w_hi=float(loss_cfg.get("w_hi", 0.0)),
        hi_gamma=float(loss_cfg.get("hi_gamma", 2.0)), w_dark=float(loss_cfg.get("w_dark", 0.0)),
        dark_thresh=float(loss_cfg.get("dark_thresh", 0.28)),
        w_color=float(loss_cfg.get("w_color", 0.0)), w_lc=float(loss_cfg.get("w_lc", 0.0)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    warmup = min(5, max(1, epochs // 10))

    # KY LUAT LUT (30/07, hoc tu repo goc HuiZeng/Image-Adaptive-3DLUT trong
    # external/): tv smooth 1e-4 + monotonicity 10.0 — thieu 2 cai nay chinh la
    # ly do CH_LUT 25/07 ra artifact (vet sang, glow gat). Boc criterion lai de
    # khong phai sua _run_epoch dung chung voi cac model khac.
    w_tv = float(cfg.get("lambda_smooth", 1e-4))
    w_mn = float(cfg.get("lambda_monotonicity", 10.0))

    class _WithLutReg(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, out, tgt):
            # _run_epoch unpack `loss, terms = criterion(...)` -> PHAI tra CAP
            loss, terms = self.base(out, tgt)
            tv, mn = model.lut_regularizers()
            total = loss + w_tv * tv + w_mn * mn
            terms = dict(terms or {})
            terms.update({"lut_tv": float(tv.detach()), "lut_mn": float(mn.detach())})
            return total, terms

    reg_crit = _WithLutReg(criterion).to(device)

    os.makedirs(SWEEP_CSV_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    history_csv = os.path.join(SWEEP_CSV_DIR, os.path.splitext(os.path.basename(out_path))[0] + ".csv")
    with open(history_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_total", "val_total", "val_l1", "lr", "sec"])

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        cur_lr = lr_for_epoch(epoch, epochs, lr, warmup)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr
        t0 = time.time()
        tr, _ = _run_epoch(model, train_loader, device, reg_crit, optimizer, scaler, use_amp, True)
        if val_loader is not None:
            vt, vl = _run_epoch(model, val_loader, device, criterion, optimizer, scaler, use_amp, False)
        else:
            vt, vl = float("nan"), float("nan")
        sec = time.time() - t0
        print(f"  Epoch {epoch:03d}/{epochs:03d} | train_total={tr:.6f} val_total={vt:.6f} "
              f"val_l1={vl:.6f} lr={cur_lr:.3e} time={sec:.2f}s")
        with open(history_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{tr:.6f}", f"{vt:.6f}", f"{vl:.6f}", f"{cur_lr:.8f}", f"{sec:.3f}"])
        improved = (val_loader is None) or (vt < best_val)
        if improved:
            best_val = vt if val_loader is not None else best_val
            atomic_torch_save(model.state_dict(), out_path)
            atomic_torch_save({"epoch": epoch, "best_val": best_val, "cfg": cfg,
                               "nm_kwargs": nm_kwargs, "loss": loss_cfg}, meta_path_for(out_path))
    if not os.path.exists(out_path):
        atomic_torch_save(model.state_dict(), out_path)
    return {"best_val": best_val, "ckpt": out_path, "history_csv": history_csv}
