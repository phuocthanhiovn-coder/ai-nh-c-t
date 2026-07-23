"""
Task 20 - GPU-ready trainer for the auto_enhance HDRNet.

Standalone: does NOT modify train.py/model.py/dataset.py/infer.py.
Model architecture is imported unchanged from .model (HDRNet) so any
checkpoint saved here is a plain state_dict loadable by the existing
infer.py without changes.

Crop-based training: full-res before/after pairs vary in resolution,
so batch_size>1 is impossible on whole images. Instead we take a random
--crop x --crop window from the (pixel-aligned) before/after pair each
step, and build the 256x256 proxy FROM that crop (not from the whole
image). All crops share a common shape, so batching works.
"""
import os
import csv
import math
import random
import hashlib
import argparse
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .model import HDRNet

IMG_EXTS = (".jpg", ".jpeg", ".png")
LOG_CSV_PATH = os.path.join("outputs", "train_gpu_log.csv")


# ---------------------------------------------------------------------------
# Deterministic split
# ---------------------------------------------------------------------------
def split_filenames(data_dir, val_frac):
    """Stable train/val split keyed off md5(filename) - independent of run
    order, dataset size, or shuffling, so it stays the same across runs and
    across dataset growth (a file's bucket never changes)."""
    before_dir = os.path.join(data_dir, "before")
    after_dir = os.path.join(data_dir, "after")
    names = sorted(f for f in os.listdir(before_dir) if f.lower().endswith(IMG_EXTS))
    names = [n for n in names if os.path.exists(os.path.join(after_dir, n))]

    train_files, val_files = [], []
    threshold = int(round(val_frac * 1000))
    for n in names:
        bucket = int(hashlib.md5(n.encode("utf-8")).hexdigest(), 16) % 1000
        (val_files if bucket < threshold else train_files).append(n)
    return train_files, val_files


# ---------------------------------------------------------------------------
# Crop dataset
# ---------------------------------------------------------------------------
class CropHDRDataset(Dataset):
    """Loads a before/after pair, takes a crop_size x crop_size window
    (random for train, centered for val), builds the 256x256 proxy from
    that crop, and returns (before_crop, proxy, after_crop) tensors in [0,1]."""

    def __init__(self, data_dir, filenames, crop_size, is_train):
        self.before_dir = os.path.join(data_dir, "before")
        self.after_dir = os.path.join(data_dir, "after")
        self.filenames = filenames
        self.crop_size = crop_size
        self.is_train = is_train

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        before_path = os.path.join(self.before_dir, filename)
        after_path = os.path.join(self.after_dir, filename)

        before_img = cv2.imread(before_path)
        after_img = cv2.imread(after_path)
        if before_img is None or after_img is None:
            raise FileNotFoundError(f"Khong doc duoc cap anh: {filename}")

        before_img = cv2.cvtColor(before_img, cv2.COLOR_BGR2RGB)
        after_img = cv2.cvtColor(after_img, cv2.COLOR_BGR2RGB)

        c = self.crop_size
        h, w = before_img.shape[:2]

        # Upscale first if the source image is smaller than the requested crop.
        if h < c or w < c:
            scale = c / min(h, w)
            new_w, new_h = max(c, int(round(w * scale))), max(c, int(round(h * scale)))
            before_img = cv2.resize(before_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            after_img = cv2.resize(after_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            h, w = new_h, new_w

        if self.is_train:
            y = random.randint(0, h - c)
            x = random.randint(0, w - c)
        else:
            # Centered, deterministic crop so val loss is comparable epoch to epoch.
            y = (h - c) // 2
            x = (w - c) // 2

        before_crop = before_img[y : y + c, x : x + c]
        after_crop = after_img[y : y + c, x : x + c]

        if self.is_train and random.random() > 0.5:
            before_crop = cv2.flip(before_crop, 1)
            after_crop = cv2.flip(after_crop, 1)

        proxy = cv2.resize(before_crop, (256, 256), interpolation=cv2.INTER_AREA)

        before_t = torch.from_numpy(before_crop.transpose(2, 0, 1).copy()).float() / 255.0
        after_t = torch.from_numpy(after_crop.transpose(2, 0, 1).copy()).float() / 255.0
        proxy_t = torch.from_numpy(proxy.transpose(2, 0, 1).copy()).float() / 255.0
        return before_t, proxy_t, after_t


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def charbonnier_loss(pred, target, eps=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))


# ---------------------------------------------------------------------------
# LR schedule: linear warmup -> cosine decay, recomputed off args.epochs so
# it also makes sense after a --resume that extends the total epoch count.
# ---------------------------------------------------------------------------
def lr_for_epoch(epoch, total_epochs, base_lr, warmup_epochs):
    if epoch <= warmup_epochs:
        return base_lr * epoch / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
    progress = min(1.0, progress)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Atomic checkpoint I/O
# ---------------------------------------------------------------------------
def atomic_torch_save(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def best_path_for(out_path):
    root, ext = os.path.splitext(out_path)
    return f"{root}_best{ext}"


def meta_path_for(out_path):
    return out_path + ".meta"


def load_resume(resume_path, model, optimizer, device):
    """resume_path holds a PLAIN state_dict (same format infer.py loads).
    Optimizer/epoch/best_val live in a sidecar '<resume_path>.meta' file so
    the primary checkpoint stays infer.py-compatible with zero changes there."""
    state_dict = torch.load(resume_path, map_location=device)
    model.load_state_dict(state_dict)

    start_epoch = 0
    best_val = float("inf")
    meta_path = meta_path_for(resume_path)
    if os.path.exists(meta_path):
        meta = torch.load(meta_path, map_location=device)
        if "optimizer" in meta:
            optimizer.load_state_dict(meta["optimizer"])
        start_epoch = meta.get("epoch", 0)
        best_val = meta.get("best_val", float("inf"))
    else:
        print(f"[!] Khong tim thay meta sidecar '{meta_path}' - chi resume duoc trong so, "
              f"optimizer/epoch se reset ve 0.")
    return start_epoch, best_val


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------
def append_csv_log(epoch, train_loss, val_loss, lr, seconds):
    os.makedirs(os.path.dirname(LOG_CSV_PATH), exist_ok=True)
    write_header = not os.path.exists(LOG_CSV_PATH)
    with open(LOG_CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "train_l1", "val_l1", "lr", "seconds"])
        writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{lr:.8f}", f"{seconds:.3f}"])


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def run_epoch(model, loader, device, optimizer, criterion, train):
    model.train(train)
    total_loss = 0.0
    n = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for before, proxy, after in loader:
            before = before.to(device)
            proxy = proxy.to(device)
            after = after.to(device)

            if train:
                optimizer.zero_grad()
            output, _ = model(proxy, before)
            loss = criterion(output, after)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * before.size(0)
            n += before.size(0)
    return total_loss / max(1, n)


def main():
    parser = argparse.ArgumentParser(description="HDRNet GPU-ready trainer (Task 20)")
    parser.add_argument("--data-dir", default="data/pairs")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-frac", type=float, default=0.12)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--out", type=str, default="checkpoints/auto_enhance_v2.pt")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--crop", type=int, default=1024)
    parser.add_argument("--charbonnier", action="store_true")
    parser.add_argument("--save-every", type=int, default=1, help="checkpoint cadence in epochs")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Production-machine constraint: keep CPU thread usage bounded so this
    # can run alongside other work on the box.
    cv2.setNumThreads(2)
    torch.set_num_threads(2)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[*] Device: {device}")

    train_files, val_files = split_filenames(args.data_dir, args.val_frac)
    print(f"[+] {len(train_files)} train / {len(val_files)} val pairs (val_frac={args.val_frac})")
    print(f"[+] Val files (stable across runs): {val_files}")

    if not train_files:
        print("[!] Khong co du lieu train. Kiem tra --data-dir.")
        return

    train_ds = CropHDRDataset(args.data_dir, train_files, args.crop, is_train=True)
    val_ds = CropHDRDataset(args.data_dir, val_files, args.crop, is_train=False) if val_files else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=False)
    val_loader = (DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, drop_last=False)
                  if val_ds is not None else None)

    model = HDRNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = charbonnier_loss if args.charbonnier else nn.L1Loss()

    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        start_epoch, best_val = load_resume(args.resume, model, optimizer, device)
        print(f"[*] Resumed from {args.resume}: start_epoch={start_epoch}, best_val={best_val:.6f}")

    warmup_epochs = min(5, max(1, args.epochs // 10))

    if start_epoch >= args.epochs:
        print(f"[*] start_epoch ({start_epoch}) >= --epochs ({args.epochs}), khong con gi de train.")
        return

    for epoch in range(start_epoch + 1, args.epochs + 1):
        lr = lr_for_epoch(epoch, args.epochs, args.lr, warmup_epochs)
        for g in optimizer.param_groups:
            g["lr"] = lr

        t0 = time.time()
        train_loss = run_epoch(model, train_loader, device, optimizer, criterion, train=True)
        val_loss = run_epoch(model, val_loader, device, optimizer, criterion, train=False) if val_loader else float("nan")
        elapsed = time.time() - t0

        print(f"  Epoch {epoch:04d}/{args.epochs:04d} | train_l1={train_loss:.6f} "
              f"val_l1={val_loss:.6f} lr={lr:.6e} time={elapsed:.2f}s")

        is_best = val_loader is not None and val_loss < best_val
        if is_best:
            best_val = val_loss

        if epoch % args.save_every == 0 or epoch == args.epochs or is_best:
            atomic_torch_save(model.state_dict(), args.out)
            atomic_torch_save(
                {"optimizer": optimizer.state_dict(), "epoch": epoch, "best_val": best_val, "args": vars(args)},
                meta_path_for(args.out),
            )
            if is_best:
                atomic_torch_save(model.state_dict(), best_path_for(args.out))

        append_csv_log(epoch, train_loss, val_loss, lr, elapsed)

    print(f"[+] Xong. Checkpoint cuoi: {args.out} | Best-val checkpoint: {best_path_for(args.out)} (best_val={best_val:.6f})")


if __name__ == "__main__":
    main()
