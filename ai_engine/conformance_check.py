"""
KIEM DINH KHUNG XUONG: moi specialist phai tuan hop dong operator
    apply(img float32 [0,1] HxWx3 BGR, params) -> cung shape, float32, huu han, ~[0,1].
Chay: python -m ai_engine.conformance_check
"""
import importlib
import traceback
import numpy as np
import cv2

cv2.setNumThreads(2)

# (module path, ten ham apply) — cap nhat khi them con moi
SPECIALISTS = [
    ("ai_engine.specialists.white_balance.wb", "apply"),
    ("ai_engine.specialists.straighten.straighten", "apply"),
    ("ai_engine.specialists.denoise_sharpen.ds", "apply"),
    ("ai_engine.specialists.grass_green.grass", "apply"),
    ("ai_engine.specialists.sky_replace.replace", "apply"),
    ("ai_engine.specialists.window_pull.pull", "apply"),
    ("ai_engine.specialists.harsh_sun.tone_map", "apply"),
    ("ai_engine.specialists.finish_detail.finish", "apply"),
    ("ai_engine.specialists.vibrance.vib", "apply"),
    ("ai_engine.specialists.shadow_light.light", "apply"),
]


def check_one(modpath, fnname):
    rng = np.random.RandomState(1234)
    # anh test 2 kich thuoc khac nhau + 1 anh chan/le de bat loi resize
    for (h, w) in [(120, 160), (161, 97)]:
        img = rng.rand(h, w, 3).astype(np.float32)
        mod = importlib.import_module(modpath)
        fn = getattr(mod, fnname)
        out = fn(img.copy(), {})
        assert isinstance(out, np.ndarray), f"{modpath}: output khong phai ndarray"
        assert out.shape == img.shape, f"{modpath}: shape {out.shape} != input {img.shape}"
        assert out.dtype == np.float32, f"{modpath}: dtype {out.dtype} != float32"
        assert np.isfinite(out).all(), f"{modpath}: co gia tri NaN/Inf"
        assert out.min() >= -1e-3 and out.max() <= 1.0 + 1e-3, \
            f"{modpath}: range [{out.min():.3f},{out.max():.3f}] ngoai [0,1]"
    return True


def main():
    print("=" * 64)
    print("  KIEM DINH HOP DONG OPERATOR (conformance)")
    print("=" * 64)
    ok, fail = 0, 0
    for modpath, fnname in SPECIALISTS:
        short = modpath.split(".")[-2]
        try:
            check_one(modpath, fnname)
            print(f"  [PASS] {short:16} apply() dung hop dong")
            ok += 1
        except ModuleNotFoundError:
            print(f"  [SKIP] {short:16} chua co module (worker dang lam?)")
        except Exception as e:
            print(f"  [FAIL] {short:16} {e}")
            traceback.print_exc()
            fail += 1
    print("-" * 64)
    print(f"  KET QUA: {ok} PASS / {fail} FAIL")
    return fail == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
