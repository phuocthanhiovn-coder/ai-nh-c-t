"""
Task GPU - Single-config trainer / sweep unit for the auto_enhance HDRNetV2.

Standalone add-on. Does NOT modify model.py/train.py/dataset.py/infer.py or the
already-shipped gpu/model_v2.py / gpu/losses.py; it only IMPORTS from them:

    from .model_v2 import HDRNetV2       # operator-not-pixel net (parameterizable)
    from .losses     import CombinedLoss # weighted L1/Charbonnier/Lab/VGG loss

WHAT THIS IS
    `train_one(cfg: dict) -> dict(best_val, last_val, ckpt, history_csv)` trains
    ONE hyper-parameter configuration end to end and is the atom a sweep driver
    calls in a loop. It is also runnable from the CLI:

        python -m ai_engine.specialists.auto_enhance.gpu.train_sweep --json '<cfg json>'

CORE PRINCIPLE (immutable, inherited from model_v2): the net predicts an
OPERATOR (a bilateral grid of affine coeffs) on a small proxy and slices it
differentiably onto the full-res crop. This trainer never asks the net for
pixels directly -- it feeds (proxy, before_crop) and supervises the sliced
output against after_crop.

CHANNEL ORDER: images are kept in BGR float32 [0,1] end to end (cv2's native
order, and exactly what losses.py documents/expects -- LabLoss and VGGPerceptual
assume BGR and do their own internal flip). We deliberately do NOT cvtColor to
RGB the way the v1 train_gpu.py did; HDRNetV2 is channel-order agnostic, so BGR
throughout keeps the Lab/perceptual losses honest.

CROP-BASED BATCHING: full-res before/after pairs vary in size, so whole-image
batching is impossible. Each sample is a crop x crop window from the aligned
pair (random for train, centered for val); the proxy is built FROM that crop via
HDRNetV2.make_proxy(...). All crops share a shape, so batch_size>1 works.

SPLIT: deterministic train/val split keyed on md5(filename) with val_frac=0.12 --
the SAME scheme as ..train_gpu.split_filenames (reused by import when available,
with an identical inline fallback), so a file's bucket never moves across runs or
dataset growth.

CUDA NOTE: AMP (autocast + GradScaler) is wired but ENGAGES ONLY when
device=='cuda' AND cfg['amp'] is true. On this CPU-only box the AMP path is
UNVERIFIED; the CPU path (scaler disabled, plain fp32) is what the smoke test
exercises. GradScaler is constructed with enabled=<use_amp> so the exact same
code runs on CPU as a no-op.
"""
import os
import csv
import json
import math
import time
import random
import hashlib
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .model_v2 import HDRNetV2
from .losses import CombinedLoss

IMG_EXTS = (".jpg", ".jpeg", ".png")
DEFAULT_DATA_DIR = os.path.join("data", "pairs")
SWEEP_CSV_DIR = os.path.join("outputs", "sweep")
DEFAULT_VAL_FRAC = 0.12

# Keys HDRNetV2.__init__ accepts, so we can hand it exactly cfg's model knobs.
_MODEL_KEYS = ("grid_bins", "grid_size", "proxy_res", "width", "guidance_hidden")


# ---------------------------------------------------------------------------
# Deterministic split - reuse ..train_gpu.split_filenames when importable, else
# an identical inline copy (same md5(filename) bucketing, so buckets match).
# ---------------------------------------------------------------------------
try:  # pragma: no cover - trivial import shim
    from ..train_gpu import split_filenames as _split_filenames
except Exception:  # pragma: no cover
    _split_filenames = None


def split_filenames(data_dir, val_frac=DEFAULT_VAL_FRAC):
    """Stable train/val split keyed off md5(filename) % 1000 < val_frac*1000.
    Independent of run order / dataset size, identical to train_gpu.py."""
    if _split_filenames is not None:
        return _split_filenames(data_dir, val_frac)
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
# Crop dataset - BGR [0,1], proxy built from the crop via HDRNetV2.make_proxy.
# ---------------------------------------------------------------------------
class CropPairDataset(Dataset):
    """Returns (before_crop [3,c,c], proxy [3,pr,pr], after_crop [3,c,c]) in
    BGR float32 [0,1]. Random crop + random hflip for train; centered crop and
    no flip for val (so val loss is comparable epoch to epoch)."""

    def __init__(self, data_dir, filenames, crop_size, proxy_res, is_train,
                 cache_ram=False, cache_cap=1280):
        self.before_dir = os.path.join(data_dir, "before")
        self.after_dir = os.path.join(data_dir, "after")
        self.filenames = filenames
        self.crop_size = int(crop_size)
        self.proxy_res = int(proxy_res)
        self.is_train = is_train
        self.cache_cap = int(cache_cap)
        # RAM cache: decode + downscale ONCE (longest side <= cache_cap) so
        # __getitem__ is a pure slice+augment. Kills the JPEG-decode bottleneck
        # that starves the GPU when many configs share the CPUs. The operator
        # net predicts a low-res bilateral grid, so a 1280-capped source does
        # not hurt operator quality (eval/delivery still apply at full-res).
        self.cache = {}
        if cache_ram:
            for fn in self.filenames:
                b = cv2.imread(os.path.join(self.before_dir, fn), cv2.IMREAD_COLOR)
                a = cv2.imread(os.path.join(self.after_dir, fn), cv2.IMREAD_COLOR)
                if b is None or a is None:
                    continue
                self.cache[fn] = (self._cap(b), self._cap(a))

    def _cap(self, img):
        h, w = img.shape[:2]
        m = max(h, w)
        if self.cache_cap and m > self.cache_cap:
            s = self.cache_cap / m
            img = cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                             interpolation=cv2.INTER_AREA)
        return img

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        if filename in self.cache:
            before_img, after_img = self.cache[filename]
        else:
            # cv2 loads BGR; we keep BGR (see module docstring).
            before_img = cv2.imread(os.path.join(self.before_dir, filename), cv2.IMREAD_COLOR)
            after_img = cv2.imread(os.path.join(self.after_dir, filename), cv2.IMREAD_COLOR)
            if before_img is None or after_img is None:
                raise FileNotFoundError(f"Khong doc duoc cap anh: {filename}")

        c = self.crop_size
        h, w = before_img.shape[:2]

        # Upscale first if the source is smaller than the requested crop, so a
        # crop x crop window always exists (matches train_gpu.py behaviour).
        if h < c or w < c:
            scale = c / min(h, w)
            new_w = max(c, int(round(w * scale)))
            new_h = max(c, int(round(h * scale)))
            before_img = cv2.resize(before_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            after_img = cv2.resize(after_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            h, w = new_h, new_w

        if self.is_train:
            y = random.randint(0, h - c)
            x = random.randint(0, w - c)
        else:
            y = (h - c) // 2
            x = (w - c) // 2

        before_crop = before_img[y:y + c, x:x + c]
        after_crop = after_img[y:y + c, x:x + c]

        # Random horizontal flip (train only), applied identically to both.
        if self.is_train and random.random() > 0.5:
            before_crop = cv2.flip(before_crop, 1)
            after_crop = cv2.flip(after_crop, 1)

        before_t = torch.from_numpy(before_crop.transpose(2, 0, 1).copy()).float() / 255.0
        after_t = torch.from_numpy(after_crop.transpose(2, 0, 1).copy()).float() / 255.0

        # Build the proxy FROM the crop, with the model's own differentiable
        # area-resample helper. make_proxy returns [1,3,pr,pr]; drop the batch.
        proxy_t = HDRNetV2.make_proxy(before_t, self.proxy_res).squeeze(0)
        return before_t, proxy_t, after_t


# ---------------------------------------------------------------------------
# LR schedule: linear warmup -> cosine decay (epoch is 1-based).
# ---------------------------------------------------------------------------
def lr_for_epoch(epoch, total_epochs, base_lr, warmup_epochs):
    if epoch <= warmup_epochs:
        return base_lr * epoch / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
    progress = min(1.0, progress)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Atomic checkpoint I/O + sidecar helpers.
# ---------------------------------------------------------------------------
def atomic_torch_save(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def meta_path_for(ckpt_path):
    return ckpt_path + ".meta"


# ---------------------------------------------------------------------------
# Config plumbing.
# ---------------------------------------------------------------------------
def _resolve_device(spec):
    if spec in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _model_kwargs(cfg):
    return {k: cfg[k] for k in _MODEL_KEYS if k in cfg and cfg[k] is not None}


def _run_name(cfg, out_path):
    if cfg.get("run_name"):
        return str(cfg["run_name"])
    base = os.path.splitext(os.path.basename(out_path))[0]
    return base or "run"


def _default_out(cfg):
    """Deterministic ckpt path when cfg omits 'out'."""
    tag = "gb{grid_bins}_gs{grid_size}_p{proxy_res}_w{width}_c{crop}".format(
        grid_bins=cfg.get("grid_bins", 8), grid_size=cfg.get("grid_size", 16),
        proxy_res=cfg.get("proxy_res", 256), width=cfg.get("width", 16),
        crop=cfg.get("crop", 512),
    )
    return os.path.join("checkpoints", "sweep", f"auto_enhance_{tag}.pt")


# ---------------------------------------------------------------------------
# Train / eval one epoch.
# ---------------------------------------------------------------------------
def _run_epoch(model, loader, device, criterion, optimizer, scaler, use_amp, train):
    """Returns (avg_total_loss, avg_raw_l1). raw_l1 is plain F.l1_loss (loss-weight
    agnostic) so val_l1 is comparable across configs with different loss weights."""
    model.train(train)
    tot_loss = 0.0
    tot_l1 = 0.0
    n = 0
    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for before, proxy, after in loader:
            before = before.to(device, non_blocking=True)
            proxy = proxy.to(device, non_blocking=True)
            after = after.to(device, non_blocking=True)
            bs = before.size(0)

            if train:
                optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda", enabled=True):
                    output, _ = model(proxy, before)
                    loss, _terms = criterion(output, after)
            else:
                output, _ = model(proxy, before)
                loss, _terms = criterion(output, after)

            if train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            with torch.no_grad():
                raw_l1 = F.l1_loss(output.float(), after.float())
            tot_loss += float(loss.item()) * bs
            tot_l1 += float(raw_l1.item()) * bs
            n += bs

    denom = max(1, n)
    return tot_loss / denom, tot_l1 / denom


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------
def train_one(cfg: dict) -> dict:
    """Train ONE config. Returns dict(best_val, last_val, ckpt, history_csv).

    cfg keys (all optional except where noted):
        data_dir   (default 'data/pairs')
        epochs     (default 100)
        batch_size (default 4)
        lr         (default 3e-4)
        crop       (default 512)      random(train)/center(val) crop size
        grid_bins, grid_size, proxy_res, width, guidance_hidden -> HDRNetV2
        loss       dict of w_l1/w_char/w_lab/w_perc (default {'w_l1': 1.0})
        seed       (default 42)
        amp        bool (default False) - only engages on cuda
        out        ckpt path (default checkpoints/sweep/auto_enhance_<tag>.pt)
        device     'auto'|'cuda'|'cpu' (default 'auto')
        val_frac   (default 0.12)
        num_workers(default 0)
    """
    cfg = dict(cfg)  # shallow copy; never mutate caller's dict

    # Thread caps per project rule (safe to run beside other work on the box).
    cv2.setNumThreads(2)
    torch.set_num_threads(2)

    data_dir = cfg.get("data_dir") or DEFAULT_DATA_DIR
    epochs = int(cfg.get("epochs", 100))
    batch_size = int(cfg.get("batch_size", 4))
    lr = float(cfg.get("lr", 3e-4))
    crop = int(cfg.get("crop", 512))
    proxy_res = int(cfg.get("proxy_res", 256))
    seed = int(cfg.get("seed", 42))
    val_frac = float(cfg.get("val_frac", DEFAULT_VAL_FRAC))
    num_workers = int(cfg.get("num_workers", 0))
    cache_ram = bool(cfg.get("cache_ram", False))
    cache_cap = int(cfg.get("cache_cap", 1280))
    loss_cfg = dict(cfg.get("loss") or {"w_l1": 1.0})

    out_path = cfg.get("out") or _default_out(cfg)
    run_name = _run_name(cfg, out_path)
    history_csv = os.path.join(SWEEP_CSV_DIR, f"{run_name}.csv")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = _resolve_device(cfg.get("device", "auto"))
    use_amp = bool(cfg.get("amp", False)) and device.type == "cuda"
    # Fixed-size crops -> let cudnn autotune the fastest conv algorithms.
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"[*] run='{run_name}'  device={device}  amp={use_amp}  "
          f"(requested amp={bool(cfg.get('amp', False))})  cache_ram={cache_ram}")

    # ---- split ----
    train_files, val_files = split_filenames(data_dir, val_frac)
    print(f"[+] {len(train_files)} train / {len(val_files)} val pairs "
          f"(val_frac={val_frac})")
    print(f"[+] Val files (stable across runs): {val_files}")
    if not train_files:
        raise RuntimeError(f"Khong co du lieu train tai '{data_dir}'.")

    # ---- data ----
    mk = _model_kwargs(cfg)
    train_ds = CropPairDataset(data_dir, train_files, crop, proxy_res, is_train=True,
                               cache_ram=cache_ram, cache_cap=cache_cap)
    val_ds = (CropPairDataset(data_dir, val_files, crop, proxy_res, is_train=False,
                              cache_ram=cache_ram, cache_cap=cache_cap)
              if val_files else None)
    if cache_ram:
        print(f"[+] RAM-cached {len(train_ds.cache)} train + "
              f"{len(val_ds.cache) if val_ds else 0} val imgs (cap {cache_cap}px)")
    pin = device.type == "cuda"
    loader_kw = dict(num_workers=num_workers, drop_last=False, pin_memory=pin)
    if num_workers > 0:
        loader_kw["persistent_workers"] = True
        loader_kw["prefetch_factor"] = 4
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kw)
    val_loader = (DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kw)
                  if val_ds is not None else None)

    # ---- model / loss / optim ----
    model = HDRNetV2(**mk).to(device)

    # Warm-start (fine-tune) from an existing checkpoint if requested. Lets us
    # continue from the champion (e.g. CH_C) toward a new anti-washout objective
    # instead of retraining from scratch. model_kwargs MUST match the ckpt.
    init_ckpt = cfg.get("init_ckpt")
    if init_ckpt:
        sd = torch.load(init_ckpt, map_location=device)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[+] Warm-start tu '{init_ckpt}' "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")

    # lab_weights=(w_L, w_a, w_b): chroma-heavy weighting (a,b >> L) fights
    # washout (lost saturation) more directly than a uniform w_lab bump, which
    # was already tried and barely helped. w_hi = highlight-protection weight.
    lab_weights = tuple(loss_cfg.get("lab_weights", (1.0, 1.0, 1.0)))
    criterion = CombinedLoss(
        w_l1=float(loss_cfg.get("w_l1", 0.0)),
        w_char=float(loss_cfg.get("w_char", 0.0)),
        w_lab=float(loss_cfg.get("w_lab", 0.0)),
        w_perc=float(loss_cfg.get("w_perc", 0.0)),
        lab_weights=lab_weights,
        w_hi=float(loss_cfg.get("w_hi", 0.0)),
        hi_gamma=float(loss_cfg.get("hi_gamma", 2.0)),
        w_dark=float(loss_cfg.get("w_dark", 0.0)),
        dark_thresh=float(loss_cfg.get("dark_thresh", 0.28)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # torch 2.2.x compat: GradScaler lives under torch.cuda.amp (torch.amp.GradScaler is 2.3+)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    warmup_epochs = min(5, max(1, epochs // 10))

    # ---- CSV header ----
    os.makedirs(SWEEP_CSV_DIR, exist_ok=True)
    with open(history_csv, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_total", "val_total", "val_l1", "lr", "sec"])

    best_val = float("inf")
    last_val = float("nan")

    for epoch in range(1, epochs + 1):
        cur_lr = lr_for_epoch(epoch, epochs, lr, warmup_epochs)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        t0 = time.time()
        train_total, _train_l1 = _run_epoch(
            model, train_loader, device, criterion, optimizer, scaler, use_amp, train=True)

        if val_loader is not None:
            val_total, val_l1 = _run_epoch(
                model, val_loader, device, criterion, optimizer, scaler, use_amp, train=False)
        else:
            val_total, val_l1 = float("nan"), float("nan")
        sec = time.time() - t0
        last_val = val_total

        print(f"  Epoch {epoch:03d}/{epochs:03d} | train_total={train_total:.6f} "
              f"val_total={val_total:.6f} val_l1={val_l1:.6f} "
              f"lr={cur_lr:.3e} time={sec:.2f}s")

        with open(history_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{train_total:.6f}", f"{val_total:.6f}",
                f"{val_l1:.6f}", f"{cur_lr:.8f}", f"{sec:.3f}"])

        # Best-val checkpoint (plain model_v2 state_dict) + reloadable meta.
        improved = val_loader is not None and val_total < best_val
        if val_loader is None:
            improved = True  # no val: keep saving latest as "best"
        if improved:
            best_val = val_total if val_loader is not None else best_val
            atomic_torch_save(model.state_dict(), out_path)
            atomic_torch_save(
                {
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val": best_val,
                    "cfg": cfg,
                    "model_kwargs": mk,
                    "loss": loss_cfg,
                },
                meta_path_for(out_path),
            )

    # If a val set existed but never improved off inf (shouldn't happen), still
    # guarantee a checkpoint on disk.
    if not os.path.exists(out_path):
        atomic_torch_save(model.state_dict(), out_path)
        atomic_torch_save(
            {"optimizer": optimizer.state_dict(), "epoch": epochs,
             "best_val": best_val, "cfg": cfg, "model_kwargs": mk, "loss": loss_cfg},
            meta_path_for(out_path),
        )

    result = {
        "best_val": best_val,
        "last_val": last_val,
        "ckpt": out_path,
        "history_csv": history_csv,
    }
    print(f"[+] Done run='{run_name}': best_val={best_val:.6f} "
          f"last_val={last_val:.6f}\n    ckpt={out_path}\n    csv={history_csv}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train ONE HDRNetV2 config (sweep unit).")
    parser.add_argument("--json", type=str, required=True,
                        help="JSON dict of the config (see train_one docstring).")
    args = parser.parse_args()
    cfg = json.loads(args.json)
    result = train_one(cfg)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
