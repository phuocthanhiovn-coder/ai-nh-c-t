"""
CPU smoke test for gpu/losses.py.

Run from the autohdr project root:
    python ai_engine/specialists/auto_enhance/gpu/smoke_losses.py

Checks:
  1. Four weight combos (incl. one with perceptual -> downloads VGG). Prints
     each term + total; asserts finite and total > 0.
  2. Gradient flows: total.backward() on a leaf pred (requires_grad=True).
  3. Honest cross-check: torch Lab vs cv2.cvtColor on a solid color patch.
"""
import os
import sys

import cv2
import numpy as np
import torch

# Machine constraints from CLAUDE.md.
cv2.setNumThreads(2)
torch.set_num_threads(2)

# Allow direct-run import of the sibling module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from losses import CombinedLoss, bgr_to_lab, charbonnier  # noqa: E402


def _assert_finite(name, t):
    assert torch.isfinite(t).all(), f"{name} is not finite: {t}"


def test_combos():
    print("=" * 64)
    print("  SMOKE: CombinedLoss weight combos (CPU)")
    print("=" * 64)
    torch.manual_seed(0)

    combos = [
        {"w_l1": 1.0},                                              # plain L1
        {"w_l1": 1.0, "w_char": 0.5},                              # L1 + charbonnier
        {"w_l1": 1.0, "w_lab": 0.5},                              # L1 + Lab color
        {"w_l1": 1.0, "w_char": 0.25, "w_lab": 0.25, "w_perc": 0.1},  # + VGG perceptual
    ]

    for i, cfg in enumerate(combos, 1):
        pred = torch.rand(2, 3, 64, 64, requires_grad=True)
        target = torch.rand(2, 3, 64, 64)

        print(f"\n--- Combo {i}: {cfg} ---")
        if "w_perc" in cfg:
            print("    (this combo builds VGGPerceptual -> may download ~528 MB)")

        crit = CombinedLoss(**cfg)
        total, terms = crit(pred, target)

        for k, v in terms.items():
            print(f"    {k:6s} = {v.item():.6f}")
            _assert_finite(k, v)

        assert total.item() > 0.0, f"total not > 0: {total.item()}"

        # Gradient must flow to the leaf pred.
        total.backward()
        assert pred.grad is not None, "pred.grad is None -- no grad flow"
        assert torch.isfinite(pred.grad).all(), "pred.grad has non-finite values"
        gnorm = pred.grad.norm().item()
        assert gnorm > 0.0, "pred.grad norm is 0 -- gradient did not flow"
        print(f"    grad OK: pred.grad.norm() = {gnorm:.6f}")

    print("\n[OK] All combos finite, total>0, grad flows.")


def test_charbonnier():
    a = torch.zeros(1, 3, 8, 8)
    b = torch.zeros(1, 3, 8, 8)
    v = charbonnier(a, b).item()
    # equal inputs -> ~eps
    assert abs(v - 1e-3) < 1e-4, f"charbonnier of equal inputs should be ~eps, got {v}"
    print(f"\n[OK] charbonnier(equal)={v:.6f} ~= eps (1e-3)")


def test_lab_vs_cv2():
    print("\n" + "=" * 64)
    print("  HONEST CHECK: torch Lab vs cv2.cvtColor (solid patches)")
    print("=" * 64)
    patches_bgr = [
        (0.20, 0.50, 0.80),   # B,G,R
        (0.90, 0.10, 0.30),
        (0.50, 0.50, 0.50),
        (0.02, 0.02, 0.02),   # near-black (below sRGB linear knee)
        (0.99, 0.99, 0.10),
    ]
    max_abs = {"L": 0.0, "a": 0.0, "b": 0.0}
    for bgr in patches_bgr:
        # cv2 path: float32 BGR [0,1] -> Lab (L 0..100, a/b -127..127)
        img = np.zeros((4, 4, 3), dtype=np.float32)
        img[:, :, 0] = bgr[0]
        img[:, :, 1] = bgr[1]
        img[:, :, 2] = bgr[2]
        cv_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        cv_vals = cv_lab[0, 0]  # L,a,b

        # torch path
        t = torch.tensor(bgr, dtype=torch.float32).view(1, 3, 1, 1)
        my_lab = bgr_to_lab(t)[0, :, 0, 0].numpy()

        dL = abs(cv_vals[0] - my_lab[0])
        da = abs(cv_vals[1] - my_lab[1])
        db = abs(cv_vals[2] - my_lab[2])
        max_abs["L"] = max(max_abs["L"], dL)
        max_abs["a"] = max(max_abs["a"], da)
        max_abs["b"] = max(max_abs["b"], db)
        print(f"  BGR{bgr}: cv2=({cv_vals[0]:7.3f},{cv_vals[1]:8.3f},{cv_vals[2]:8.3f}) "
              f"torch=({my_lab[0]:7.3f},{my_lab[1]:8.3f},{my_lab[2]:8.3f}) "
              f"|d|=({dL:.3f},{da:.3f},{db:.3f})")

    print(f"\n  Max abs diff over patches: L={max_abs['L']:.3f} "
          f"a={max_abs['a']:.3f} b={max_abs['b']:.3f}")
    print("  (L on 0..100, a/b on ~-127..127 scale)")


if __name__ == "__main__":
    test_charbonnier()
    test_combos()
    test_lab_vs_cv2()
    print("\n[DONE] smoke_losses.py finished.")
