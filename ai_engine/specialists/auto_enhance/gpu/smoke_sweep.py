"""
CPU smoke for the sweep DRIVER (sweep.py).

The REAL end-to-end smoke (calling the actual train_one) is DEFERRED: the
dependency ai_engine/specialists/auto_enhance/gpu/train_sweep.py does not exist
yet. To still exercise the driver's own logic — config loop, timing, best_val
sort, leaderboard CSV, best-checkpoint copy, and the smoke's skip-FINAL branch —
this injects a STUB train_one that writes a tiny real checkpoint and returns a
deterministic best_val. This proves the orchestration works; it proves NOTHING
about training quality.

Run: python -m ai_engine.specialists.auto_enhance.gpu.smoke_sweep
"""
import os
import csv
import argparse

import cv2
import torch

from . import sweep

cv2.setNumThreads(2)
torch.set_num_threads(2)


def _stub_train_one(cfg):
    """Fake trainer: saves a tiny plain state_dict at cfg['out'] (so the copy
    step has a real file to move) and returns a deterministic best_val derived
    from the config name, so the leaderboard ordering is reproducible."""
    out = cfg["out"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    # A tiny but real torch checkpoint (plain state_dict, like train_gpu.py).
    torch.save({"stub_weight": torch.zeros(1)}, out)
    # Deterministic pseudo-loss: 'mini_lab' should beat 'mini_l1' to show sorting.
    fake = {"mini_l1": 0.42, "mini_lab": 0.31}.get(cfg["name"], 0.5)
    return {"best_val": fake, "ckpt_path": out, "epochs_run": cfg["epochs"]}


def main():
    args = sweep.build_parser().parse_args([
        "--smoke",
        "--time-budget-min", "100000",   # effectively unlimited
        "--out-dir", os.path.join("outputs", "sweep_smoke"),
        "--device", "cpu",
    ])

    print("=" * 88)
    print("  SWEEP DRIVER CPU SMOKE (stub train_one — REAL train_one DEFERRED: train_sweep.py absent)")
    print("=" * 88)

    results = sweep.run(args, train_fn=_stub_train_one)

    # --- verify the acceptance criteria from the task -------------------------
    lb = os.path.join(args.out_dir, "leaderboard.csv")
    assert results is not None, "run() returned None (dependency path taken unexpectedly)"
    assert os.path.exists(lb), f"leaderboard.csv not written at {lb}"
    with open(lb, newline="") as f:
        rows = list(csv.reader(f))
    data_rows = rows[1:]
    assert len(data_rows) == 2, f"expected 2 leaderboard rows, got {len(data_rows)}"
    # sorted best-first: row 1 must be the lower best_val (mini_lab=0.31)
    assert data_rows[0][1] == "mini_lab", f"expected mini_lab ranked #1, got {data_rows[0][1]}"
    assert float(data_rows[0][2]) < float(data_rows[1][2]), "leaderboard not sorted ascending by best_val"
    assert os.path.exists(sweep.BEST_CKPT), f"best checkpoint not copied to {sweep.BEST_CKPT}"

    print("\n[VERIFY] leaderboard.csv rows =", len(data_rows), "(expected 2)  OK")
    print("[VERIFY] rank#1 =", data_rows[0][1], "best_val =", data_rows[0][2], " OK")
    print("[VERIFY] best ckpt copied ->", sweep.BEST_CKPT,
          f"({os.path.getsize(sweep.BEST_CKPT)} bytes)  OK")
    print("\nALL SWEEP-DRIVER SMOKE ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
