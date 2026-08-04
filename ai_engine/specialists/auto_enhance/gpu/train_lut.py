"""Train Image-Adaptive 3D-LUT (25/07 huong moi [2]) tai dung ha tang train_sweep.
Cung LOSS MOI (l1 nhe + color + lc) nhu CH_M de A/B cong bang grid vs LUT.

Box: cd /root/autohdr && setsid /opt/conda/bin/python -u -m tools.launch_lut > train_lut.log 2>&1 < /dev/null &
"""
import csv
import os
import time

import torch
from torch.utils.data import DataLoader

from .losses import CombinedLoss
from .model_lut import Image3DLUT
from .train_sweep import (CropPairDataset, SWEEP_CSV_DIR, _run_epoch,
                          atomic_torch_save, lr_for_epoch, meta_path_for,
                          split_filenames)


def train_lut(cfg: dict) -> dict:
    # ⭐ 04/08 (ra soat vong 7) — GIEO SEED. train_nm.py da gieo tu 31/07 nhung
    # train_lut thi khong (luat L16: ban va dung, ap thieu nhanh). Khong gieo thi hai
    # lan chay CUNG cau hinh cho hai ket qua khac nhau, tuc moi phep A/B "mot bien"
    # tren dong LUT deu khong tai lap duoc — khong the biet chenh lech den tu bien
    # dang thu hay tu may man khoi tao/xao tron.
    import random as _random
    _seed = int(cfg.get("seed", 42))
    _random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    torch.cuda.manual_seed_all(_seed)

    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    data_dir = cfg["data_dir"]
    crop = int(cfg.get("crop", 512))
    proxy_res = int(cfg.get("proxy_res", 256))
    batch_size = int(cfg.get("batch_size", 4))
    epochs = int(cfg.get("epochs", 200))
    lr = float(cfg.get("lr", 1e-4))
    val_frac = float(cfg.get("val_frac", 0.15))
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    num_workers = int(cfg.get("num_workers", 4))
    cache_ram = bool(cfg.get("cache_ram", True))
    cache_cap = int(cfg.get("cache_cap", 60))
    out_path = cfg.get("out", "checkpoints/sweep/CH_LUT.pt")
    loss_cfg = dict(cfg.get("loss") or {"w_l1": 1.0})
    lut_kwargs = dict(n_basis=int(cfg.get("n_basis", 3)),
                      lut_dim=int(cfg.get("lut_dim", 33)),
                      backbone_res=int(cfg.get("backbone_res", 256)))

    train_files, val_files = split_filenames(data_dir, val_frac)
    print(f"[LUT] {len(train_files)} train / {len(val_files)} val (val_frac={val_frac})")
    print(f"[LUT] val files: {val_files}")

    train_ds = CropPairDataset(data_dir, train_files, crop, proxy_res, is_train=True,
                               cache_ram=cache_ram, cache_cap=cache_cap)
    val_ds = (CropPairDataset(data_dir, val_files, crop, proxy_res, is_train=False,
                              cache_ram=cache_ram, cache_cap=cache_cap) if val_files else None)
    pin = device.type == "cuda"
    lk = dict(num_workers=num_workers, drop_last=False, pin_memory=pin)
    if num_workers > 0:
        lk["persistent_workers"] = True
        lk["prefetch_factor"] = 4
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **lk)
    val_loader = (DataLoader(val_ds, batch_size=batch_size, shuffle=False, **lk)
                  if val_ds is not None else None)

    model = Image3DLUT(**lut_kwargs).to(device)
    print(f"[LUT] params={sum(p.numel() for p in model.parameters())} kwargs={lut_kwargs}")

    lab_weights = tuple(loss_cfg.get("lab_weights", (1.0, 1.0, 1.0)))
    # FIX 02/08 (luat L16 — va phai ap cho CA N nhanh): w_chroma/w_sharp/w_clip truoc
    # day THIEU o day y het train_sweep.py, trong khi tools/launch_lut.py DANG khai
    # "w_chroma": 2.0 -> CH_LUT da train voi term mau = 0.0 im lang. Ket luan
    # "3D-LUT underfit nen loai" (25/07) vi the KHONG con hieu luc.
    _KNOWN_LOSS_KEYS = {
        "w_l1", "w_char", "w_lab", "w_perc", "lab_weights", "w_hi", "hi_gamma",
        "w_dark", "dark_thresh", "w_color", "w_lc", "w_chroma", "w_sharp", "w_clip",
    }
    _unknown = sorted(set(loss_cfg) - _KNOWN_LOSS_KEYS)
    if _unknown:
        raise ValueError(f"loss_cfg co khoa la {_unknown}: CombinedLoss KHONG nhan nen "
                         f"se bi bo qua IM LANG. Khoa hop le: {sorted(_KNOWN_LOSS_KEYS)}")
    criterion = CombinedLoss(
        w_l1=float(loss_cfg.get("w_l1", 0.0)), w_char=float(loss_cfg.get("w_char", 0.0)),
        w_lab=float(loss_cfg.get("w_lab", 0.0)), w_perc=float(loss_cfg.get("w_perc", 0.0)),
        lab_weights=lab_weights, w_hi=float(loss_cfg.get("w_hi", 0.0)),
        hi_gamma=float(loss_cfg.get("hi_gamma", 2.0)), w_dark=float(loss_cfg.get("w_dark", 0.0)),
        dark_thresh=float(loss_cfg.get("dark_thresh", 0.28)),
        w_color=float(loss_cfg.get("w_color", 0.0)), w_lc=float(loss_cfg.get("w_lc", 0.0)),
        w_chroma=float(loss_cfg.get("w_chroma", 0.0)),   # <-- 02/08: truoc day THIEU
        w_sharp=float(loss_cfg.get("w_sharp", 0.0)),     # <-- 02/08: truoc day THIEU
        w_clip=float(loss_cfg.get("w_clip", 0.0)),       # <-- 02/08: truoc day THIEU
    ).to(device)
    print("[loss] dang chay: " + ", ".join(
        f"{k}={getattr(criterion, k):g}" for k in
        ("w_l1", "w_char", "w_lab", "w_perc", "w_hi", "w_dark", "w_color", "w_lc",
         "w_chroma", "w_sharp", "w_clip") if getattr(criterion, k, 0.0)), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    warmup = min(5, max(1, epochs // 10))

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
        tr, _ = _run_epoch(model, train_loader, device, criterion, optimizer, scaler, use_amp, True)
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
                               "lut_kwargs": lut_kwargs, "loss": loss_cfg}, meta_path_for(out_path))
    if not os.path.exists(out_path):
        atomic_torch_save(model.state_dict(), out_path)
    return {"best_val": best_val, "ckpt": out_path, "history_csv": history_csv}
