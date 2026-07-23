"""
Chạy thử con "DỌC THẲNG" trên 6 ảnh mẫu (3 từ data/pairs/before/, 3 từ data/review/before/).
Lưu outputs/straighten_samples/<tên>.jpg = ghép ngang [gốc | đã nắn] (downscale 1500px).
In góc nghiêng ước lượng + có nắn hay identity cho từng ảnh.

Chỉ ĐỌC data/, KHÔNG ghi vào đó.
"""

import os
import sys

import cv2
import numpy as np

cv2.setNumThreads(2)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import straighten  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PAIRS_BEFORE = os.path.join(ROOT, "data", "pairs", "before")
REVIEW_BEFORE = os.path.join(ROOT, "data", "review", "before")
OUT_DIR = os.path.join(ROOT, "outputs", "straighten_samples")

VIEW_DIM = 1500


def list_jpgs(folder, n):
    names = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    return [os.path.join(folder, f) for f in names[:n]]


# Chọn cố định để bao quát cả 3 tình huống: ảnh đã thẳng (identity), ảnh nghiêng
# rõ được nắn (rectify), và ảnh có nghiêng nhưng bị an-toàn-gate từ chối (identity).
# (Xác định bằng cách quét toàn bộ 92 ảnh mẫu qua analyze() trước khi chọn.)
FIXED_PAIRS_BEFORE = ["20260703-DSC1105.jpg", "_ML_1421.jpg", "20260703-DSC1132.jpg"]
FIXED_REVIEW_BEFORE = ["_ML_1661.jpg", "_ML_1393.jpg", "20260703-DSC1197.jpg"]


def pick_fixed_or_fallback(folder, fixed_names, n):
    paths = []
    for name in fixed_names:
        p = os.path.join(folder, name)
        if os.path.exists(p):
            paths.append(p)
    if len(paths) < n:
        for p in list_jpgs(folder, n):
            if p not in paths:
                paths.append(p)
            if len(paths) >= n:
                break
    return paths[:n]


def resize_view(img, target_dim=VIEW_DIM):
    h, w = img.shape[:2]
    scale = target_dim / max(h, w)
    if scale >= 1.0:
        return img.copy()
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    samples = pick_fixed_or_fallback(PAIRS_BEFORE, FIXED_PAIRS_BEFORE, 3) + \
        pick_fixed_or_fallback(REVIEW_BEFORE, FIXED_REVIEW_BEFORE, 3)
    if len(samples) < 6:
        print(f"CANH BAO: chi tim thay {len(samples)} anh mau (can 6).")

    print(f"{'ten anh':40s} {'goc(deg)':>10s} {'so duong':>10s} {'trang thai':>12s}")
    print("-" * 76)

    for path in samples:
        name = os.path.basename(path)
        img_u8 = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_u8 is None:
            print(f"{name:40s}  LOI DOC ANH")
            continue

        img_f32 = img_u8.astype(np.float32) / 255.0

        diag = straighten.analyze(img_f32)
        rectified = straighten.apply(img_f32, {"strength": 1.0, "k1": 0.0})

        status = "NAN (rectify)" if diag["applied"] else "IDENTITY"
        print(f"{name:40s} {diag['angle_deg']:10.2f} {diag['num_lines']:10d} {status:>12s}")

        rectified_u8 = np.clip(rectified * 255.0, 0, 255).astype(np.uint8)

        left = resize_view(img_u8)
        right = resize_view(rectified_u8)
        # đảm bảo cùng chiều cao để hstack (do làm tròn resize có thể lệch 1px)
        h = min(left.shape[0], right.shape[0])
        left = left[:h]
        right = right[:h]

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(left, "Before", (20, 40), font, 1.0, (0, 0, 255), 2)
        label_right = f"After ({status}) {diag['angle_deg']:.1f}deg"
        cv2.putText(right, label_right, (20, 40), font, 0.9, (0, 255, 0), 2)

        canvas = np.hstack((left, right))
        out_path = os.path.join(OUT_DIR, os.path.splitext(name)[0] + ".jpg")
        cv2.imwrite(out_path, canvas)

        # kiểm tra output apply() giữ nguyên kích thước input (đúng full-res, không resize ngầm)
        assert rectified.shape == img_f32.shape, f"Sai shape output cho {name}: {rectified.shape} != {img_f32.shape}"

    print(f"\nDa luu {len(samples)} anh so sanh vao {OUT_DIR}")


if __name__ == "__main__":
    main()
