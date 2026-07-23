"""
Smoke-test that cung la acceptance test cho ai_engine/core/quality.py (Task 13).

Chay: python -m ai_engine.core.test_quality
"""

import os
import sys

import cv2
import numpy as np

cv2.setNumThreads(2)

from ai_engine.core.quality import (
    apply_color_on_lowfreq,
    composite_mask,
    guided_upsample,
    merge_frequency,
    read_image_16,
    split_frequency,
    to_linear,
    to_srgb,
    write_image,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAMPLE_IMAGE = os.path.join(REPO_ROOT, "data", "pairs", "before", "20260703-DSC1105.jpg")
OUT_DIR = os.path.join(REPO_ROOT, "outputs", "core_samples")


def test_srgb_roundtrip():
    rng = np.random.default_rng(42)
    x = rng.random((256, 256, 3), dtype=np.float32)
    y = to_srgb(to_linear(x))
    err = float(np.max(np.abs(y - x)))
    print(f"[srgb_roundtrip] max err = {err:.3e}")
    assert err < 1e-4, f"srgb roundtrip err too high: {err}"


def test_split_merge_identity():
    rng = np.random.default_rng(7)
    img = rng.random((300, 400, 3), dtype=np.float32)
    low, high = split_frequency(img, sigma=6.0)
    recon_no_clip = low + high
    err_no_clip = float(np.max(np.abs(recon_no_clip - img)))
    print(f"[split_merge_identity] max err (truoc clip) = {err_no_clip:.3e}")
    assert err_no_clip < 1e-6, f"split/merge khong phai identity: {err_no_clip}"

    merged = merge_frequency(low, high)
    err_after_clip = float(np.max(np.abs(merged - img)))
    print(f"[split_merge_identity] max err (sau merge_frequency, co clip) = {err_after_clip:.3e}")
    assert err_after_clip < 1e-6


def _gradient_magnitude(bgr_float01):
    gray = cv2.cvtColor((np.clip(bgr_float01, 0, 1) * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def test_apply_color_on_lowfreq(img):
    identity_out = apply_color_on_lowfreq(img, lambda low: low, sigma=8)
    err = float(np.max(np.abs(identity_out - img)))
    print(f"[apply_color_on_lowfreq identity] max err = {err:.3e}")
    assert err < 1e-4, f"identity transform doi anh: {err}"

    def brighten(low):
        # tang sang bang cong (exposure lift) thay vi nhan gain manh:
        # phep cong khong doi gradient cua low, chi mat mot chut o vung
        # da gan bao hoa (>1 bi clip trong merge_frequency).
        return np.clip(low + 0.04, 0.0, 1.0)

    bright_out = apply_color_on_lowfreq(img, brighten, sigma=8)

    gray_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    edges = cv2.Canny(gray_u8, 80, 160)
    edge_mask = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1) > 0
    n_edge_px = int(edge_mask.sum())
    assert n_edge_px > 0, "khong tim thay canh nao trong anh mau de kiem tra"

    grad_before = _gradient_magnitude(img)
    grad_after = _gradient_magnitude(bright_out)

    avg_before = float(grad_before[edge_mask].mean())
    avg_after = float(grad_after[edge_mask].mean())
    ratio = avg_after / avg_before
    print(
        f"[apply_color_on_lowfreq brighten] n_edge_px={n_edge_px} "
        f"grad_avg_before={avg_before:.3f} grad_avg_after={avg_after:.3f} ratio={ratio:.4f}"
    )
    assert ratio >= 0.98, f"canh mat net qua nhieu sau brighten: ratio={ratio:.4f}"


def test_guided_upsample_demo(img):
    os.makedirs(OUT_DIR, exist_ok=True)
    h, w = img.shape[:2]

    small_w = 128
    small_h = max(1, round(h * small_w / w))
    small_mask = np.zeros((small_h, small_w), dtype=np.float32)
    small_mask[: small_h // 3, :] = 1.0
    small_mask = cv2.GaussianBlur(small_mask, (0, 0), sigmaX=3)

    up = guided_upsample(small_mask, img)
    assert up.shape[:2] == (h, w), f"guided_upsample sai size: {up.shape[:2]} != {(h, w)}"

    # do "bam canh": gradient cua mask upsample phai lon o vung co canh that trong guide
    gray_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    guide_edges = cv2.Canny(gray_u8, 80, 160) > 0
    mask_grad = np.abs(cv2.Sobel(up, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(
        cv2.Sobel(up, cv2.CV_32F, 0, 1, ksize=3)
    )
    band = np.zeros((h, w), dtype=bool)
    band[max(0, h // 3 - 40) : h // 3 + 40, :] = True
    on_guide_edge = band & guide_edges
    off_guide_edge = band & (~guide_edges)
    if on_guide_edge.sum() > 0 and off_guide_edge.sum() > 0:
        grad_on = float(mask_grad[on_guide_edge].mean())
        grad_off = float(mask_grad[off_guide_edge].mean())
        print(f"[guided_upsample] grad tren canh guide={grad_on:.4f} vs ngoai canh={grad_off:.4f}")

    vis = (np.clip(img, 0, 1) * 255).astype(np.uint8).copy()
    red = np.zeros_like(vis)
    red[:, :, 2] = 255
    alpha = np.clip(up, 0, 1)[:, :, None]
    vis_f = vis.astype(np.float32) * (1 - 0.5 * alpha) + red.astype(np.float32) * (0.5 * alpha)
    vis_u8 = np.clip(vis_f, 0, 255).astype(np.uint8)

    demo_path = os.path.join(OUT_DIR, "jbu_demo.jpg")
    ok = cv2.imwrite(demo_path, vis_u8, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    assert ok, f"khong ghi duoc {demo_path}"
    print(f"[guided_upsample] da luu demo: {demo_path} size={vis_u8.shape[1]}x{vis_u8.shape[0]}")
    return demo_path


def test_composite_mask(img):
    h, w = img.shape[:2]
    edited = np.clip(img * 1.2, 0, 1).astype(np.float32)
    small_mask = np.zeros((64, 64), dtype=np.float32)
    small_mask[:32, :] = 1.0
    out = composite_mask(img, edited, small_mask, feather_px=10)
    assert out.shape == img.shape, f"composite_mask sai shape: {out.shape} != {img.shape}"
    print(f"[composite_mask] shape OK = {out.shape}")


def test_write_read_roundtrip(img):
    os.makedirs(OUT_DIR, exist_ok=True)
    png_path = os.path.join(OUT_DIR, "_rw_roundtrip_test.png")
    jpg_path = os.path.join(OUT_DIR, "_rw_roundtrip_test.jpg")

    write_image(png_path, img, quality=95)
    back_png = read_image_16(png_path)
    assert back_png.shape[:2] == img.shape[:2]
    err_png = float(np.max(np.abs(back_png - img)))
    print(f"[write_read_roundtrip] PNG16 max err = {err_png:.3e} (roundoff, ok neu nho)")

    write_image(jpg_path, img, quality=95)
    back_jpg = read_image_16(jpg_path)
    assert back_jpg.shape[:2] == img.shape[:2]
    print(f"[write_read_roundtrip] JPEG size OK = {back_jpg.shape[1]}x{back_jpg.shape[0]}")

    os.remove(png_path)
    os.remove(jpg_path)


def main():
    print("=== Task 13 core quality lib — test_quality.py ===")
    print(f"cv2.getNumThreads() = {cv2.getNumThreads()}")

    test_srgb_roundtrip()
    test_split_merge_identity()

    assert os.path.exists(SAMPLE_IMAGE), f"khong tim thay anh sample: {SAMPLE_IMAGE}"
    img = read_image_16(SAMPLE_IMAGE)
    print(f"[load sample] {SAMPLE_IMAGE} -> shape={img.shape} dtype={img.dtype}")

    test_apply_color_on_lowfreq(img)
    demo_path = test_guided_upsample_demo(img)
    test_composite_mask(img)
    test_write_read_roundtrip(img)

    print("=== ALL TESTS PASSED ===")
    print(f"Xem anh demo bam-canh tai: {demo_path}")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"TEST FAILED: {e}")
        sys.exit(1)
