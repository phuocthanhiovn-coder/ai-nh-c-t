# -*- coding: utf-8 -*-
"""Tự cân bằng trắng / khử ám màu (auto white balance).

Cách dùng:
    python -m ai_engine.specialists.white_balance.white_balance --test [--sample <path.jpg>]
    python -m ai_engine.specialists.white_balance.white_balance --in <folder> --out <outdir> [--method combined] [--strength 0.8]
"""
import argparse
import glob
import os

import cv2
import numpy as np

cv2.setNumThreads(2)

_EPS = 1e-6


def _luma(bgr_f32):
    """Luma Rec.709 từ ảnh BGR float32."""
    return (0.0722 * bgr_f32[:, :, 0]
            + 0.7152 * bgr_f32[:, :, 1]
            + 0.2126 * bgr_f32[:, :, 2])


def _gains_grayworld(img):
    """Gain mỗi kênh theo giả định gray-world: mean 3 kênh bằng nhau."""
    means = img.reshape(-1, 3).mean(axis=0)
    target = means.mean()
    return target / np.maximum(means, _EPS)


def _gains_whitepatch(img, top_frac=0.01):
    """Gain theo white-patch: vùng sáng nhất ~1% làm trắng chuẩn."""
    luma = _luma(img)
    thresh = np.quantile(luma, 1.0 - top_frac)
    mask = luma >= thresh
    if mask.sum() < 16:  # ảnh quá phẳng, không đủ điểm sáng
        return np.ones(3, np.float64)
    means = img[mask].mean(axis=0)
    target = means.mean()
    return target / np.maximum(means, _EPS)


def _apply_gains(img, gains):
    """Áp gain kênh rồi chuẩn hoá lại luma để không đổi độ sáng tổng thể."""
    out = img * gains.reshape(1, 1, 3)
    luma_before = float(_luma(img).mean())
    luma_after = float(_luma(out).mean())
    if luma_after > _EPS:
        out *= luma_before / luma_after
    return np.clip(out, 0.0, 1.0)


def auto_wb(bgr, method="grayworld", strength=1.0):
    """Khử ám màu. bgr uint8/float; trả về cùng dtype/shape.

    - method="grayworld": mean 3 kênh bằng nhau.
    - method="whitepatch": top ~1% sáng nhất làm trắng chuẩn.
    - method="combined": trung bình gain 2 cách trên.
    - strength (0..1): pha trộn gốc (0) -> sửa hoàn toàn (1).
    Luma tổng thể được chuẩn hoá lại sau khi scale màu.
    """
    if bgr is None:
        return None
    src_dtype = bgr.dtype
    img = bgr.astype(np.float32)
    if src_dtype == np.uint8:
        img /= 255.0
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("auto_wb: can anh BGR 3 kenh, nhan shape %s" % (bgr.shape,))

    if method == "grayworld":
        gains = _gains_grayworld(img)
    elif method == "whitepatch":
        gains = _gains_whitepatch(img)
    elif method == "combined":
        gains = 0.5 * (_gains_grayworld(img) + _gains_whitepatch(img))
    else:
        raise ValueError("auto_wb: method la khong hop le: %r" % (method,))

    corrected = _apply_gains(img, gains)
    strength = float(np.clip(strength, 0.0, 1.0))
    out = np.clip((1.0 - strength) * img + strength * corrected, 0.0, 1.0)

    if src_dtype == np.uint8:
        return (out * 255.0 + 0.5).astype(np.uint8)
    return out.astype(src_dtype)


def measure_cast(bgr):
    """Đo ám màu: trả {'a': mean_a, 'b': mean_b} trong Lab (đã trừ 128). Gần 0 = trung tính."""
    if bgr is None:
        return None
    img = bgr
    if img.dtype != np.uint8:
        img = (np.clip(img.astype(np.float32), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return {"a": float(lab[:, :, 1].mean()) - 128.0,
            "b": float(lab[:, :, 2].mean()) - 128.0}


def _put_label(img, text):
    """Ghi nhãn chữ lên góc trên-trái ảnh (nền đen mờ)."""
    out = img.copy()
    scale = max(0.6, out.shape[1] / 1400.0)
    thick = max(1, int(round(scale * 2)))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(out, (0, 0), (tw + 20, th + base + 20), (0, 0, 0), -1)
    cv2.putText(out, text, (10, th + 10), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def _run_test(sample_path):
    if sample_path is None:
        candidates = sorted(glob.glob(os.path.join("data", "pairs", "before", "*.jpg")))
        if not candidates:
            print("Khong tim thay anh mau trong data/pairs/before/")
            return 1
        sample_path = candidates[0]
    orig = cv2.imread(sample_path)
    if orig is None:
        print("Khong doc duoc anh: %s" % sample_path)
        return 1

    # Tạo ám xanh giả: nhân kênh B x1.18
    cast = orig.astype(np.float32)
    cast[:, :, 0] *= 1.18
    cast = np.clip(cast, 0, 255).astype(np.uint8)

    fixed = auto_wb(cast, method="combined", strength=1.0)

    print("Anh mau: %s" % sample_path)
    for name, im in (("GOC", orig), ("AM XANH", cast), ("DA SUA", fixed)):
        m = measure_cast(im)
        print("  %-8s a=%+.2f  b=%+.2f" % (name, m["a"], m["b"]))

    strip = np.hstack([_put_label(orig, "GOC"),
                       _put_label(cast, "AM XANH"),
                       _put_label(fixed, "DA SUA")])
    out_path = os.path.join("outputs", "wb_test.jpg")
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(out_path, strip, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("Da luu: %s" % out_path)
    print("TASK DONE")
    return 0


def _run_batch(in_dir, out_dir, method, strength):
    paths = sorted(glob.glob(os.path.join(in_dir, "*.jpg"))
                   + glob.glob(os.path.join(in_dir, "*.jpeg"))
                   + glob.glob(os.path.join(in_dir, "*.png")))
    if not paths:
        print("Khong co anh trong: %s" % in_dir)
        return 1
    os.makedirs(out_dir, exist_ok=True)
    n_ok = 0
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print("  BO QUA (khong doc duoc): %s" % p)
            continue
        before = measure_cast(img)
        out = auto_wb(img, method=method, strength=strength)
        after = measure_cast(out)
        dst = os.path.join(out_dir, os.path.basename(p))
        cv2.imwrite(dst, out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print("  %s  cast a:%+.2f->%+.2f  b:%+.2f->%+.2f" % (
            os.path.basename(p), before["a"], after["a"], before["b"], after["b"]))
        n_ok += 1
    print("Xong %d/%d anh -> %s" % (n_ok, len(paths), out_dir))
    print("TASK DONE")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Tu can bang trang / khu am mau")
    ap.add_argument("--test", action="store_true", help="chay tu-test voi anh am xanh gia")
    ap.add_argument("--sample", default=None, help="anh mau cho --test")
    ap.add_argument("--in", dest="in_dir", default=None, help="thu muc anh vao")
    ap.add_argument("--out", dest="out_dir", default=None, help="thu muc anh ra")
    ap.add_argument("--method", default="combined",
                    choices=["grayworld", "whitepatch", "combined"])
    ap.add_argument("--strength", type=float, default=0.8)
    args = ap.parse_args()

    if args.test:
        raise SystemExit(_run_test(args.sample))
    if args.in_dir and args.out_dir:
        raise SystemExit(_run_batch(args.in_dir, args.out_dir, args.method, args.strength))
    ap.print_help()


if __name__ == "__main__":
    main()
