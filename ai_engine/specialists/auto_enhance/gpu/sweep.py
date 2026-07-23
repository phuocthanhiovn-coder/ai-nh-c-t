"""
Sweep driver for the auto_enhance HDRNetV2 — squeeze one rented 3-hour V100.

Standalone add-on. Does NOT modify ..model / ..train / ..infer / ..dataset,
and only reads/writes under this package + outputs/ + checkpoints/.

WHAT THIS FILE DOES
    1. Holds a hand-curated CONFIGS list (16 points) that explores lr / loss
       combo / grid resolution / proxy_res / width — baseline first, then one
       axis varied at a time, plus 3 'kitchen-sink' strong configs. It is NOT a
       full cartesian product (that would be 3*4*3*2*2 = 144 runs — no chance in
       3 hours); it is a sensible hand-picked path.
    2. run(): trains every config (via train_one), writes a best_val-sorted
       leaderboard to outputs/sweep/leaderboard.csv, and copies the OVERALL best
       checkpoint to checkpoints/auto_enhance_v2_best.pt. Honors --time-budget-min:
       it stops LAUNCHING new configs once the budget is spent, but always lets
       the config already running finish.
    3. FINAL phase: takes the top config, retrains it LONG (epochs * --final-mult)
       with 3 different seeds, and keeps the best of the seeds as
       checkpoints/auto_enhance_v2_final.pt. Guarded by remaining time — it skips
       seeds it cannot afford rather than blowing the budget.
    4. Prints a clear leaderboard at the end.

DEPENDENCY — the trainer interface (ai_engine/specialists/auto_enhance/gpu/train_sweep.py)
    This driver calls a single function, kept deliberately narrow so the two
    files can be written independently:

        train_one(cfg: dict) -> dict

    INPUT cfg (this driver fills EVERY key below before calling):
        name        str    unique run label (safe as a filename)
        lr          float   base learning rate
        epochs      int     number of epochs to train THIS config
        crop        int     full-res crop size for batching (proxy built FROM it)
        batch_size  int
        val_frac    float   fraction held out for validation (hash-stable split)
        seed        int
        proxy_res   int     proxy square resolution fed to the coeff predictor
        grid_bins   int     bilateral-grid luminance bins
        grid_size   int     bilateral-grid spatial size (gh == gw)
        width       int     channel-width knob of HDRNetV2
        loss        str     human label of the loss combo (for logs/CSV only)
        w_l1        float   CombinedLoss weight — plain L1
        w_char      float   CombinedLoss weight — Charbonnier
        w_lab       float   CombinedLoss weight — CIE-Lab
        w_perc      float   CombinedLoss weight — VGG perceptual
        lab_weights tuple   (w_L, w_a, w_b) for LabLoss
        data_dir    str     dataset root (contains before/ after/)
        device      str     'auto' | 'cuda' | 'cpu'
        out         str     where train_one should save this run's checkpoint(s);
                            the returned ckpt_path is authoritative.

    RETURN dict (this driver reads these keys, defensively):
        best_val    float   REQUIRED. Best validation loss reached. Lower better.
                            May be inf if the config had no val data.
        ckpt_path   str     Path to the best-val checkpoint (a PLAIN state_dict,
                            infer.py-loadable). If missing, this driver falls
                            back to cfg['out'].
        (any other keys — e.g. epochs_run, seconds — are ignored but welcome.)

    train_one is responsible for actually building HDRNetV2 from the cfg knobs,
    building CombinedLoss from the w_* weights, training, and atomically saving
    its best checkpoint. This driver only orchestrates, ranks, copies, and times.

CUDA NOTE: this driver is pure Python/stdlib + shutil/csv and has no device code
of its own; the CUDA/AMP path lives entirely inside train_one. On THIS box
(CPU-only, torch 2.13.0+cpu) the real end-to-end smoke is DEFERRED because
train_sweep.py does not exist yet; the driver's own loop/leaderboard/copy logic
IS exercised on CPU with an injected stub train_one (see smoke_sweep.py). Be
explicit about that in any report.
"""
import os
import csv
import time
import shutil
import argparse

# The real trainer may not exist yet while this driver is being written. Import
# it lazily/softly so `--list` and stub-injected smoke tests still work, and so a
# missing dependency yields a clear message instead of an import crash.
try:
    from .train_sweep import train_one as _train_one
except Exception:  # pragma: no cover - dependency may be absent during authoring
    _train_one = None

# Reference the module-global by name inside run() so tests can monkeypatch
# `sweep.train_one` (or pass train_fn=) without the real dependency present.
train_one = _train_one


# ---------------------------------------------------------------------------
# Loss combos — each returns the CombinedLoss weight block plus a label.
# w_perc = 0.05 for every perceptual combo, matching the task spec. w_lab = 0.5
# is a deliberate, modest choice: LabLoss already divides L by 100 so its
# magnitude sits near the pixel-L1 scale; 0.5 nudges color fidelity without
# letting it dominate the L1/Charbonnier fidelity term. Tune here if the sweep
# shows Lab is under/over-weighted.
# ---------------------------------------------------------------------------
_W_PERC = 0.05
_W_LAB = 0.5


def _L1():
    return dict(loss="L1", w_l1=1.0, w_char=0.0, w_lab=0.0, w_perc=0.0)


def _L1_LAB():
    return dict(loss="L1+Lab", w_l1=1.0, w_char=0.0, w_lab=_W_LAB, w_perc=0.0)


def _L1_LAB_PERC():
    return dict(loss="L1+Lab+Perc", w_l1=1.0, w_char=0.0, w_lab=_W_LAB, w_perc=_W_PERC)


def _CHAR_LAB_PERC():
    return dict(loss="Char+Lab+Perc", w_l1=0.0, w_char=1.0, w_lab=_W_LAB, w_perc=_W_PERC)


def _cfg(name, lr, loss_fn, grid_bins, grid_size, proxy_res, width, **extra):
    c = dict(name=name, lr=lr, grid_bins=grid_bins, grid_size=grid_size,
             proxy_res=proxy_res, width=width)
    c.update(loss_fn())
    c.update(extra)  # optional per-config overrides (epochs, crop, ...)
    return c


# ---------------------------------------------------------------------------
# The 16 curated configs.
#   Axes: lr {1e-4, 3e-4, 1e-3}; loss {L1, +Lab, +Lab+Perc, Char+Lab+Perc};
#         grid {bins8/size16, bins16/size16, bins8/size32};
#         proxy_res {256, 384}; width {16, 24}.
#   Strategy: baseline first; then vary ONE axis at a time off the baseline;
#   then a few sensible pairings; then 3 kitchen-sink strong configs.
# ---------------------------------------------------------------------------
CONFIGS = [
    # --- baseline -------------------------------------------------------------
    _cfg("baseline",          3e-4, _L1,            8, 16, 256, 16),
    # --- vary lr --------------------------------------------------------------
    _cfg("lr_low",            1e-4, _L1,            8, 16, 256, 16),
    _cfg("lr_high",           1e-3, _L1,            8, 16, 256, 16),
    # --- vary loss ------------------------------------------------------------
    _cfg("loss_lab",          3e-4, _L1_LAB,        8, 16, 256, 16),
    _cfg("loss_lab_perc",     3e-4, _L1_LAB_PERC,   8, 16, 256, 16),
    _cfg("loss_char_lab_perc",3e-4, _CHAR_LAB_PERC, 8, 16, 256, 16),
    # --- vary grid ------------------------------------------------------------
    _cfg("grid_bins16",       3e-4, _L1,           16, 16, 256, 16),
    _cfg("grid_size32",       3e-4, _L1,            8, 32, 256, 16),
    # --- vary proxy_res -------------------------------------------------------
    _cfg("proxy384",          3e-4, _L1,            8, 16, 384, 16),
    # --- vary width -----------------------------------------------------------
    _cfg("width24",           3e-4, _L1,            8, 16, 256, 24),
    # --- sensible pairings (still one idea per config) ------------------------
    _cfg("lab_lr_low",        1e-4, _L1_LAB,        8, 16, 256, 16),
    _cfg("grid32_proxy384",   3e-4, _L1,            8, 32, 384, 16),
    _cfg("width24_lab",       3e-4, _L1_LAB,        8, 16, 256, 24),
    # --- kitchen sink: strong, higher-capacity combos -------------------------
    _cfg("ks_char_big",       3e-4, _CHAR_LAB_PERC,16, 16, 384, 24),
    _cfg("ks_lab_grid32",     1e-3, _L1_LAB,        8, 32, 384, 24),
    _cfg("ks_perc_wide",      3e-4, _L1_LAB_PERC,  16, 16, 256, 24),
]

# Tiny 2-config list for the CPU smoke test (epochs=2, crop=128, batch=1).
MINI_CONFIGS = [
    _cfg("mini_l1",  3e-4, _L1,     8, 16, 256, 16, epochs=2, crop=128, batch_size=1),
    _cfg("mini_lab", 3e-4, _L1_LAB, 8, 16, 256, 16, epochs=2, crop=128, batch_size=1),
]

# Final destinations for the two "winner" checkpoints.
BEST_CKPT = os.path.join("checkpoints", "auto_enhance_v2_best.pt")
FINAL_CKPT = os.path.join("checkpoints", "auto_enhance_v2_final.pt")

# CSV columns for the leaderboard.
_CSV_COLS = ["rank", "name", "best_val", "loss", "lr", "grid_bins", "grid_size",
             "proxy_res", "width", "epochs", "crop", "batch_size", "seconds", "ckpt_path"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def finalize_cfg(cfg, args):
    """Fill every key train_one expects, taking per-config overrides first and
    global CLI defaults otherwise. Also assigns a per-config checkpoint path."""
    c = dict(cfg)
    c.setdefault("epochs", args.epochs)
    c.setdefault("crop", args.crop)
    c.setdefault("batch_size", args.batch_size)
    c.setdefault("val_frac", args.val_frac)
    c.setdefault("seed", args.seed)
    c.setdefault("lab_weights", (1.0, 1.0, 1.0))
    c["data_dir"] = args.data_dir
    c["device"] = args.device
    c["out"] = os.path.join(args.out_dir, "ckpts", f"{c['name']}.pt")
    return c


def _copy_ckpt(src, dst):
    """Copy a checkpoint to a stable destination, atomically-ish (tmp+replace).
    Returns True on success. Missing/None src is a no-op warning."""
    if not src or not os.path.exists(src):
        print(f"[!] Khong tim thay checkpoint nguon de copy: {src!r} -> {dst} (bo qua)")
        return False
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    tmp = dst + ".tmp"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    print(f"[+] Copied best checkpoint: {src} -> {dst}")
    return True


def write_leaderboard(results, out_dir):
    """results: list of dicts already sorted best-first. Writes CSV, returns path."""
    path = os.path.join(out_dir, "leaderboard.csv")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_CSV_COLS)
        for rank, r in enumerate(results, start=1):
            c = r["cfg"]
            w.writerow([
                rank, c["name"], f"{r['best_val']:.6f}", c["loss"], f"{c['lr']:.0e}",
                c["grid_bins"], c["grid_size"], c["proxy_res"], c["width"],
                c["epochs"], c["crop"], c["batch_size"], f"{r['seconds']:.1f}",
                r["ckpt_path"] or "",
            ])
    return path


def print_leaderboard(results):
    print("\n" + "=" * 88)
    print("  SWEEP LEADERBOARD (best validation loss first)")
    print("=" * 88)
    header = f"  {'#':>2}  {'name':<20} {'best_val':>10}  {'loss':<14} {'lr':>6}  {'grid':>8} {'pxy':>4} {'w':>3} {'sec':>7}"
    print(header)
    print("  " + "-" * 84)
    for rank, r in enumerate(results, start=1):
        c = r["cfg"]
        grid = f"{c['grid_bins']}x{c['grid_size']}"
        print(f"  {rank:>2}  {c['name']:<20} {r['best_val']:>10.6f}  {c['loss']:<14} "
              f"{c['lr']:>6.0e}  {grid:>8} {c['proxy_res']:>4} {c['width']:>3} {r['seconds']:>7.1f}")
    print("=" * 88)


# ---------------------------------------------------------------------------
# One measured training call
# ---------------------------------------------------------------------------
def _train_measured(cfg, train_fn):
    """Call train_fn(cfg), time it, and normalize the result into
    (best_val: float, ckpt_path: str|None, seconds: float)."""
    t = time.monotonic()
    res = train_fn(cfg)
    seconds = time.monotonic() - t
    if not isinstance(res, dict) or "best_val" not in res:
        raise ValueError(
            f"train_one({cfg['name']}) must return a dict with 'best_val'; got {type(res)}")
    best_val = float(res["best_val"])
    ckpt_path = res.get("ckpt_path") or cfg.get("out")
    return best_val, ckpt_path, seconds


# ---------------------------------------------------------------------------
# Sweep + final phases
# ---------------------------------------------------------------------------
def run(args, train_fn=None):
    """Drive the whole sweep. train_fn overrides the module-global train_one
    (used by the smoke test to inject a stub). Returns the sorted results list,
    or None if the dependency is unavailable (DEFERRED)."""
    train_fn = train_fn or train_one
    if train_fn is None:
        print("[DEFERRED] train_sweep.py::train_one khong ton tai — khong the chay sweep that.\n"
              "           Viet train_sweep.py roi chay lai, hoac chay smoke_sweep.py (stub) de "
              "kiem tra driver logic.")
        return None

    configs = MINI_CONFIGS if args.smoke else CONFIGS
    out_dir = args.out_dir
    os.makedirs(os.path.join(out_dir, "ckpts"), exist_ok=True)

    t0 = time.monotonic()
    budget_s = args.time_budget_min * 60.0
    print(f"[*] Sweep start: {len(configs)} config(s), time budget = "
          f"{args.time_budget_min:.0f} min, epochs/config = {args.epochs} "
          f"(smoke={args.smoke})")

    results = []
    for i, raw in enumerate(configs, start=1):
        elapsed = time.monotonic() - t0
        if elapsed >= budget_s:
            print(f"[*] Het ngan sach thoi gian sau {i-1}/{len(configs)} config "
                  f"({elapsed/60:.1f} >= {args.time_budget_min:.0f} min) — dung khong khoi dong config moi.")
            break

        cfg = finalize_cfg(raw, args)
        print(f"\n[{i}/{len(configs)}] === {cfg['name']} === "
              f"lr={cfg['lr']:.0e} loss={cfg['loss']} "
              f"grid={cfg['grid_bins']}x{cfg['grid_size']} proxy={cfg['proxy_res']} "
              f"width={cfg['width']} epochs={cfg['epochs']} "
              f"(elapsed {elapsed/60:.1f} min)")
        try:
            best_val, ckpt_path, seconds = _train_measured(cfg, train_fn)
        except Exception as e:
            print(f"[!] Config '{cfg['name']}' LOI, bo qua: {e!r}")
            continue

        print(f"    -> best_val={best_val:.6f}  ckpt={ckpt_path}  ({seconds:.1f}s)")
        results.append({"cfg": cfg, "best_val": best_val,
                        "ckpt_path": ckpt_path, "seconds": seconds})

    if not results:
        print("[!] Khong config nao chay xong — khong co leaderboard.")
        return results

    results.sort(key=lambda r: r["best_val"])
    lb_path = write_leaderboard(results, out_dir)
    print_leaderboard(results)
    print(f"[+] Leaderboard: {lb_path}")

    # Copy the overall best config's checkpoint to the stable v2_best path.
    _copy_ckpt(results[0]["ckpt_path"], BEST_CKPT)

    # --- FINAL phase ---------------------------------------------------------
    if args.smoke:
        print("[*] Smoke mode: bo qua FINAL phase (chi kiem tra vong sweep + leaderboard + copy best).")
        return results
    if args.no_final:
        print("[*] --no-final: bo qua FINAL phase.")
        return results

    final_phase(results[0], t0, budget_s, args, train_fn)
    return results


def final_phase(top, t0, budget_s, args, train_fn):
    """Retrain the winning config LONG (epochs * final-mult) across several seeds;
    keep the best seed as FINAL_CKPT. Skips seeds that don't fit remaining time."""
    base_cfg = top["cfg"]
    long_epochs = base_cfg["epochs"] * args.final_mult
    # First estimate scales the sweep run's wall-time by the epoch multiplier;
    # after the first long run completes we replace it with a measured number.
    est_per_seed = top["seconds"] * args.final_mult
    seeds = args.final_seeds

    print("\n" + "=" * 88)
    print(f"  FINAL PHASE — retrain winner '{base_cfg['name']}' LONG "
          f"({long_epochs} epochs) x {len(seeds)} seed(s) {seeds}")
    print("=" * 88)

    best = None  # (best_val, ckpt_path, seed)
    for seed in seeds:
        time_left = budget_s - (time.monotonic() - t0)
        need = est_per_seed * args.final_safety
        if time_left < need:
            print(f"[*] Bo qua seed {seed}: con {time_left/60:.1f} min < can ~{need/60:.1f} min "
                  f"(uoc luong). Giu ket qua cac seed da xong.")
            break

        cfg = dict(base_cfg)
        cfg["seed"] = seed
        cfg["epochs"] = long_epochs
        cfg["name"] = f"{base_cfg['name']}_final_s{seed}"
        cfg["out"] = os.path.join(args.out_dir, "ckpts", f"{cfg['name']}.pt")
        print(f"\n[final] seed={seed} epochs={long_epochs} "
              f"(con ~{time_left/60:.1f} min, uoc ~{est_per_seed/60:.1f} min/seed)")
        try:
            best_val, ckpt_path, seconds = _train_measured(cfg, train_fn)
        except Exception as e:
            print(f"[!] Final seed {seed} LOI, bo qua: {e!r}")
            continue

        est_per_seed = seconds  # refine estimate with a real measurement
        print(f"    -> best_val={best_val:.6f}  ckpt={ckpt_path}  ({seconds:.1f}s)")
        if best is None or best_val < best[0]:
            best = (best_val, ckpt_path, seed)

    if best is None:
        print("[!] Khong seed FINAL nao chay xong — khong tao duoc auto_enhance_v2_final.pt.")
        return
    print(f"\n[+] FINAL winner: seed={best[2]}  best_val={best[0]:.6f}")
    _copy_ckpt(best[1], FINAL_CKPT)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_seeds(s):
    return [int(x) for x in str(s).split(",") if str(x).strip() != ""]


def build_parser():
    p = argparse.ArgumentParser(description="HDRNetV2 sweep driver (3-hour GPU squeeze)")
    p.add_argument("--data-dir", default="data/pairs")
    p.add_argument("--out-dir", default=os.path.join("outputs", "sweep"))
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    # epochs sized so ONE config is ~5-8 min on a V100-32GB (UNMEASURED here —
    # the HDRNet predictor is light; a rented box should time --epochs 2 first).
    p.add_argument("--epochs", type=int, default=300,
                   help="epochs per sweep config (default 300 ~= 5-8 min/config on a V100, UNVERIFIED)")
    p.add_argument("--crop", type=int, default=512, help="full-res crop size for batching")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--val-frac", type=float, default=0.12)
    p.add_argument("--seed", type=int, default=42, help="default seed for the sweep phase")
    p.add_argument("--time-budget-min", type=float, default=170.0,
                   help="stop LAUNCHING new configs past this many minutes (current one always finishes)")
    p.add_argument("--final-mult", type=int, default=4, help="FINAL epochs = sweep epochs * this")
    p.add_argument("--final-seeds", type=_parse_seeds, default=[1, 2, 3],
                   help="comma-separated seeds for the FINAL retrain, e.g. 1,2,3")
    p.add_argument("--final-safety", type=float, default=1.10,
                   help="require this * estimated-seed-time of remaining budget before starting a FINAL seed")
    p.add_argument("--no-final", action="store_true", help="run the sweep but skip the FINAL phase")
    p.add_argument("--smoke", action="store_true",
                   help="2-config mini sweep (epochs=2, crop=128) — validates the driver only")
    p.add_argument("--list", action="store_true", help="print the configs and exit (no training)")
    return p


def main():
    args = build_parser().parse_args()
    if args.list:
        cfgs = MINI_CONFIGS if args.smoke else CONFIGS
        print(f"{len(cfgs)} config(s):")
        for i, c in enumerate(cfgs, 1):
            print(f"  {i:>2}. {c['name']:<20} lr={c['lr']:.0e} loss={c['loss']:<14} "
                  f"grid={c['grid_bins']}x{c['grid_size']} proxy={c['proxy_res']} width={c['width']}")
        return
    run(args)


if __name__ == "__main__":
    main()
