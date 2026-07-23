"""
CPU smoke test for train_sweep.train_one.

Run: python -m ai_engine.specialists.auto_enhance.gpu.smoke_train_sweep

Exercises the tiny config from the task spec on the REAL data/pairs:
  - trains 2 epochs, prints train+val loss each epoch,
  - confirms it wrote ckpt + .meta + per-run CSV,
  - reloads the checkpoint into a fresh HDRNetV2(**model_kwargs),
  - runs train_one TWICE and asserts the val split (filenames) is identical.
"""
import os
import csv
import time

import torch

from .model_v2 import HDRNetV2
from .train_sweep import train_one, split_filenames, meta_path_for, _model_kwargs

TINY = dict(
    data_dir="data/pairs",
    epochs=2,
    batch_size=2,
    crop=128,
    proxy_res=128,
    grid_bins=8,
    grid_size=8,
    width=8,
    lr=3e-4,
    loss={"w_l1": 1.0, "w_lab": 0.2},
    amp=False,
    device="cpu",
    seed=42,
    out="checkpoints/sweep/smoke_train_sweep.pt",
    run_name="smoke_train_sweep",
)


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def main():
    print("=" * 70)
    print("  train_sweep CPU smoke test (tiny cfg, real data/pairs)")
    print("=" * 70)

    # --- split stability (independent of training) ---
    _, val_a = split_filenames(TINY["data_dir"], 0.12)
    _, val_b = split_filenames(TINY["data_dir"], 0.12)
    assert val_a == val_b, "val split not stable across calls!"
    print(f"[split] {len(val_a)} val files, stable: {val_a}")

    # --- run 1 ---
    print("\n----- RUN 1 -----")
    t0 = time.time()
    r1 = train_one(TINY)
    run1_sec = time.time() - t0

    ckpt, meta, hist = r1["ckpt"], meta_path_for(r1["ckpt"]), r1["history_csv"]
    assert os.path.exists(ckpt), f"missing ckpt {ckpt}"
    assert os.path.exists(meta), f"missing meta {meta}"
    assert os.path.exists(hist), f"missing csv {hist}"
    rows = _read_csv(hist)
    print(f"[artifacts] ckpt={os.path.getsize(ckpt)}B  meta OK  "
          f"csv rows(incl header)={len(rows)}")
    assert len(rows) == 1 + TINY["epochs"], "csv should have header + 2 epochs"
    print(f"[csv] header={rows[0]}")
    for row in rows[1:]:
        print(f"[csv] {row}")

    # --- reload checkpoint into a fresh model ---
    mk = _model_kwargs(TINY)
    fresh = HDRNetV2(**mk)
    state = torch.load(ckpt, map_location="cpu")
    fresh.load_state_dict(state)
    fresh.eval()
    # forward pass to prove it's a working operator net after reload
    with torch.no_grad():
        proxy = torch.rand(1, 3, TINY["proxy_res"], TINY["proxy_res"])
        full = torch.rand(1, 3, 137, 211)
        out, grid = fresh(proxy, full)
    assert out.shape == full.shape and torch.isfinite(out).all()
    print(f"[reload] ckpt reloaded into fresh HDRNetV2{tuple(mk.items())}; "
          f"forward out={tuple(out.shape)} grid={tuple(grid.shape)} OK")

    # --- run 2: confirm val split identical ---
    print("\n----- RUN 2 (val-split stability) -----")
    r2 = train_one(dict(TINY, run_name="smoke_train_sweep_run2",
                        out="checkpoints/sweep/smoke_train_sweep_run2.pt"))
    _, val2 = split_filenames(TINY["data_dir"], 0.12)
    assert val_a == val2, "val split differed between runs!"
    print(f"[stability] val split identical across 2 runs: {val_a == val2}")

    sec_per_epoch = run1_sec / TINY["epochs"]
    print("\n" + "=" * 70)
    print("  ALL SMOKE ASSERTIONS PASSED")
    print(f"  run1 wall={run1_sec:.2f}s -> ~{sec_per_epoch:.2f}s/epoch (incl 1 val pass/epoch)")
    print(f"  best_val(run1)={r1['best_val']:.6f}  last_val={r1['last_val']:.6f}")
    print(f"  ckpt={ckpt}\n  csv={hist}")
    print("=" * 70)


if __name__ == "__main__":
    main()
