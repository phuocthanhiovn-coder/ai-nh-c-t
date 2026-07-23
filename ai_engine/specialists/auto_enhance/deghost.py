# -*- coding: utf-8 -*-
"""Trộn bracket CÓ KHỬ BÓNG MA (deghost).

Vật di chuyển giữa các tấm bracket (người, rèm, cây) làm fusion Mertens bị
"ghost" (vật mờ/nhân đôi). Module này phát hiện vùng chuyển động so với 1 tấm
tham chiếu, rồi ở vùng đó CHỈ lấy pixel từ tấm tham chiếu (đã bù sáng cho khớp
kết quả fusion) -> hết ghost, phần tĩnh vẫn hưởng lợi fusion. Thuần CPU.

CLI:
  python -m ai_engine.specialists.auto_enhance.deghost --test [--sample <path.jpg>]
"""

import argparse
import glob
import os
import sys

# Console Windows mặc định cp1252 không in được tiếng Việt -> ép UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

import cv2

cv2.setNumThreads(2)

import numpy as np

# Ngưỡng chênh lệch xám (sau khi bù sáng) để coi là chuyển động.
_MOTION_THRESH = 25


def _read_image(path: str) -> np.ndarray:
    """Đọc ảnh BGR bằng cv2, raise lỗi rõ ràng nếu không đọc được."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {path}")
    return img


def _match_histogram_gray(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Khớp histogram ảnh xám src về phân bố của ref (bù chênh phơi sáng)."""
    s_cdf = np.cumsum(np.bincount(src.ravel(), minlength=256)).astype(np.float64)
    r_cdf = np.cumsum(np.bincount(ref.ravel(), minlength=256)).astype(np.float64)
    s_cdf /= s_cdf[-1]
    r_cdf /= r_cdf[-1]
    lut = np.interp(s_cdf, r_cdf, np.arange(256)).astype(np.uint8)
    return lut[src]


def _motion_mask(gray_img: np.ndarray, gray_ref: np.ndarray) -> np.ndarray:
    """Mask uint8 (0/255) vùng chuyển động giữa 1 tấm và tấm tham chiếu."""
    matched = _match_histogram_gray(gray_img, gray_ref)
    # Blur nhẹ trước khi trừ để không dính nhiễu hạt.
    a = cv2.GaussianBlur(matched, (5, 5), 0).astype(np.int16)
    b = cv2.GaussianBlur(gray_ref, (5, 5), 0).astype(np.int16)
    mask = (np.abs(a - b) > _MOTION_THRESH).astype(np.uint8) * 255
    # Mở để bỏ đốm lẻ, giãn để phủ trọn mép vật chuyển động.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=2)
    return mask


def merge_deghost(paths: list, ref_index: int = None) -> np.ndarray:
    """Trộn bracket nhưng ở vùng CHUYỂN ĐỘNG chỉ lấy 1 tấm tham chiếu (tránh ghost).

    1. Đọc + đồng nhất size + AlignMTB (như bracket_merge).
    2. ref mặc định = tấm có độ sáng trung vị (phơi sáng 'giữa').
    3. Mỗi tấm khác ref: khớp histogram xám về ref rồi |diff| > ngưỡng -> mask motion.
    4. Fusion Mertens toàn ảnh; ở vùng motion thay bằng pixel ref (đã bù sáng
       theo tỉ lệ blur cục bộ cho khớp fusion), mép mask làm mịn GaussianBlur.
    5. Trả BGR uint8 cùng kích thước ảnh đầu.
    """
    if not paths or len(paths) < 2:
        raise ValueError(
            f"Bracket cần >= 2 ảnh, nhận được {0 if not paths else len(paths)}"
        )

    imgs = [_read_image(p) for p in paths]

    # Đồng nhất kích thước theo ảnh đầu tiên.
    h0, w0 = imgs[0].shape[:2]
    imgs = [
        img if img.shape[:2] == (h0, w0)
        else cv2.resize(img, (w0, h0), interpolation=cv2.INTER_AREA)
        for img in imgs
    ]

    try:
        cv2.createAlignMTB().process(imgs, imgs)
    except cv2.error as e:
        print(f"[deghost] CẢNH BÁO: align MTB lỗi ({e}), dùng ảnh gốc.")

    grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]

    # Chọn tấm tham chiếu: độ sáng trung bình nằm giữa (trung vị) các tấm.
    if ref_index is None:
        means = [float(g.mean()) for g in grays]
        ref_index = int(np.argsort(means)[len(means) // 2])
    if not 0 <= ref_index < len(imgs):
        raise ValueError(f"ref_index={ref_index} ngoài phạm vi 0..{len(imgs) - 1}")
    ref = imgs[ref_index]
    gray_ref = grays[ref_index]

    # Hợp nhất mask chuyển động của mọi tấm so với ref.
    motion = np.zeros((h0, w0), np.uint8)
    for i, g in enumerate(grays):
        if i == ref_index:
            continue
        motion = np.maximum(motion, _motion_mask(g, gray_ref))

    fused = cv2.createMergeMertens().process(imgs)  # float32, xấp xỉ [0,1]
    fused = np.clip(fused * 255.0, 0.0, 255.0)

    # Bù sáng ref về mức fusion bằng tỉ lệ blur cục bộ (mép ghép không lộ).
    ref_f = ref.astype(np.float32)
    gain = cv2.GaussianBlur(fused, (0, 0), 25) / (
        cv2.GaussianBlur(ref_f, (0, 0), 25) + 1.0
    )
    ref_adj = np.clip(ref_f * np.clip(gain, 0.5, 2.0), 0.0, 255.0)

    # Trộn mềm theo mask đã làm mịn mép.
    alpha = cv2.GaussianBlur(motion, (21, 21), 0).astype(np.float32) / 255.0
    alpha = alpha[..., None]
    out = fused * (1.0 - alpha) + ref_adj * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


# ----------------------------- CLI + tự test ------------------------------


def _label(img: np.ndarray, text: str) -> np.ndarray:
    """Vẽ nhãn chữ lên góc trên-trái ảnh (nền đen chữ trắng cho dễ đọc)."""
    out = img.copy()
    scale = max(0.6, out.shape[1] / 800.0)
    thick = max(1, int(round(scale * 2)))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(out, (0, 0), (tw + 20, th + base + 20), (0, 0, 0), -1)
    cv2.putText(out, text, (10, th + 10), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def _run_test(sample: str) -> None:
    """Bracket tổng hợp + 'vật di chuyển' giả -> so merge thường vs deghost."""
    from ai_engine.specialists.auto_enhance.bracket_merge import merge_brackets

    if sample is None:
        candidates = sorted(glob.glob(os.path.join("data", "pairs", "before", "*.jpg")))
        if not candidates:
            raise ValueError("Không tìm thấy ảnh mẫu trong data/pairs/before/*.jpg "
                             "— hãy truyền --sample <path.jpg>.")
        sample = candidates[0]
    print(f"[deghost] Ảnh mẫu: {sample}")

    base = _read_image(sample)
    h, w = base.shape[:2]
    f = base.astype(np.float32)

    # 3 bản phơi sáng + hình vuông sáng ("vật di chuyển") ở vị trí KHÁC NHAU.
    gains = (0.4, 1.0, 2.4)
    xs = (100, 300, 500)
    side = max(60, min(h, w) // 8)
    y = h // 2 - side // 2
    trio = []
    tmp_dir = os.path.join("outputs", "_deghost_test_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    for i, (g, x) in enumerate(zip(gains, xs)):
        img = np.clip(f * g, 0, 255).astype(np.uint8)
        x = min(x, w - side - 1)
        cv2.rectangle(img, (x, y), (x + side, y + side), (240, 240, 240), -1)
        p = os.path.join(tmp_dir, f"exp_{i}.png")
        cv2.imwrite(p, img)
        trio.append(p)

    plain = merge_brackets(trio, align=True)
    clean = merge_deghost(trio)

    strip = cv2.hconcat([
        _label(plain, "MERGE THUONG (co ghost)"),
        _label(clean, "DEGHOST (sach)"),
    ])
    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "deghost_test.jpg")
    if not cv2.imwrite(out_path, strip, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise IOError(f"Không ghi được {out_path}")

    diff = np.abs(plain.astype(np.int16) - clean.astype(np.int16)).max(axis=2)
    diff_pct = 100.0 * float((diff > 10).sum()) / diff.size
    print(f"[deghost] Shape kết quả: {clean.shape} (khớp gốc: {clean.shape[:2] == (h, w)})")
    print(f"[deghost] % pixel khác biệt (>10) giữa merge thường và deghost: {diff_pct:.2f}%")
    print(f"[deghost] Đã lưu dải so sánh: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Trộn bracket có khử bóng ma (deghost).")
    ap.add_argument("--test", action="store_true", help="Chạy tự test bracket + vật di chuyển giả")
    ap.add_argument("--sample", default=None, help="Ảnh mẫu cho --test")
    args = ap.parse_args()

    if args.test:
        _run_test(args.sample)
        print("TASK DONE")
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
