# -*- coding: utf-8 -*-
"""Pipeline giao ảnh HDR: bracket -> merge (Mertens) -> operator màu CH_C -> grade -> full-res q100.

Nối 3 mảnh đã có: bracket_merge (fusion), render_delivery (áp operator full-res
+ lưu q100 4:4:4), finish_grade (tăng bão hòa/ấm/tương phản scene-aware).
Chạy thuần CPU.

CLI:
  python -m ai_engine.specialists.auto_enhance.bracket_deliver --test [--sample <path.jpg>]
  python -m ai_engine.specialists.auto_enhance.bracket_deliver --folder <dir> --group-size N --out <outdir>

LƯU Ý TRUNG THỰC: bracket của --test là TỔNG HỢP từ 1 ảnh (nhân 0.35/1.0/2.6)
nên KHÔNG chứng minh lợi ích HDR thật — chỉ chứng minh đường ống chạy thông.
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
import torch

torch.set_num_threads(2)

from ai_engine.specialists.auto_enhance.gpu.model_v2 import HDRNetV2
from ai_engine.specialists.auto_enhance.gpu.render_delivery import load_cfg, apply_fullres, save_hq
from ai_engine.specialists.auto_enhance.gpu.finish_grade import grade_auto
from ai_engine.specialists.auto_enhance.bracket_merge import merge_brackets, group_brackets


def load_model(ckpt="checkpoints/gpu/CH_C.pt", device=None):
    """Nạp checkpoint CH_C + cấu hình từ .meta -> (model eval, device cpu)."""
    device = device or torch.device("cpu")
    cfg = load_cfg(ckpt + ".meta", device)
    model = HDRNetV2(**cfg).to(device)
    st = torch.load(ckpt, map_location=device)
    if isinstance(st, dict) and "state_dict" in st:
        st = st["state_dict"]
    model.load_state_dict(st); model.eval()
    return model, device


def deliver_bracket(paths: list, model, device, grade: bool = True) -> np.ndarray:
    """1 bracket (list path) -> merge -> operator CH_C -> (grade) -> ảnh BGR uint8 full-res."""
    merged = merge_brackets(paths)
    ai = apply_fullres(model, merged, device)
    return grade_auto(ai, paths[0]) if grade else ai


def deliver_folder(folder, out_dir, group_size=0, grade=True, ckpt="checkpoints/gpu/CH_C.pt"):
    """Gom bracket trong folder -> deliver từng cái -> lưu hdr_NNN.jpg q100 4:4:4."""
    model, device = load_model(ckpt)
    groups = group_brackets(folder, group_size=group_size)
    os.makedirs(out_dir, exist_ok=True)
    n_ok = 0
    for i, grp in enumerate(groups):
        if len(grp) < 2:
            print(f"[bracket_deliver] Bỏ qua bracket {i} (chỉ {len(grp)} ảnh): {grp}")
            continue
        try:
            out = deliver_bracket(grp, model, device, grade=grade)
        except Exception as e:
            print(f"[bracket_deliver] LỖI bracket {i} ({grp}): {e}")
            continue
        out_path = os.path.join(out_dir, f"hdr_{n_ok:03d}.jpg")
        save_hq(out_path, out)
        if not os.path.exists(out_path):
            raise IOError(f"Không ghi được {out_path}")
        kb = os.path.getsize(out_path) / 1024
        h, w = out.shape[:2]
        print(f"[bracket_deliver] Bracket {i} ({len(grp)} ảnh) -> {out_path}  {w}x{h}  {kb:.0f} KB")
        n_ok += 1
    print(f"[bracket_deliver] Xong: {n_ok}/{len(groups)} bracket -> {out_dir}")
    return n_ok


# ----------------------------- CLI + tự test ------------------------------


def _label(img: np.ndarray, text: str) -> np.ndarray:
    """Vẽ nhãn chữ góc trên-trái (nền đen chữ trắng)."""
    out = img.copy()
    scale = max(0.6, out.shape[1] / 800.0)
    thick = max(1, int(round(scale * 2)))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(out, (0, 0), (tw + 20, th + base + 20), (0, 0, 0), -1)
    cv2.putText(out, text, (10, th + 10), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def _run_test(sample, ckpt):
    """Bracket TỔNG HỢP (x0.35/1.0/2.6) từ 1 ảnh -> merge -> CH_C -> grade -> dải so sánh.

    Bracket tổng hợp chỉ chứng minh đường ống chạy thông, KHÔNG chứng minh lợi ích HDR thật.
    """
    if sample is None:
        candidates = sorted(glob.glob(os.path.join("data", "pairs", "before", "*.jpg")))
        if not candidates:
            raise ValueError("Không tìm thấy ảnh mẫu trong data/pairs/before/*.jpg "
                             "— hãy truyền --sample <path.jpg>.")
        sample = candidates[0]
    print(f"[bracket_deliver] Ảnh mẫu: {sample}")

    base = cv2.imread(sample, cv2.IMREAD_COLOR)
    if base is None:
        raise ValueError(f"Không đọc được ảnh: {sample}")
    f = base.astype(np.float32)
    dark = np.clip(f * 0.35, 0, 255).astype(np.uint8)
    bright = np.clip(f * 2.6, 0, 255).astype(np.uint8)

    # Ghi 3 bản phơi sáng ra file tạm để đi đúng đường merge_brackets(paths).
    tmp_dir = os.path.join("outputs", "_bracket_deliver_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    trio = []
    for name, img in (("dark", dark), ("mid", base), ("bright", bright)):
        p = os.path.join(tmp_dir, f"{name}.png")
        if not cv2.imwrite(p, img):
            raise IOError(f"Không ghi được file tạm {p}")
        trio.append(p)

    print("[bracket_deliver] Nạp model CH_C (CPU)...")
    model, device = load_model(ckpt)

    merged = merge_brackets(trio)
    final = deliver_bracket(trio, model, device, grade=True)

    strip = cv2.hconcat([
        _label(base, "VUA (goc)"),
        _label(merged, "DA MERGE"),
        _label(final, "MERGE+AI+GRADE"),
    ])
    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "bracket_deliver_test.jpg")
    if not cv2.imwrite(out_path, strip, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise IOError(f"Không ghi được {out_path}")

    kb = os.path.getsize(out_path) / 1024
    print(f"[bracket_deliver] Shape ảnh cuối (MERGE+AI+GRADE): {final.shape}")
    print(f"[bracket_deliver] Dải so sánh: {out_path}  ({kb:.0f} KB)")
    print("[bracket_deliver] LƯU Ý: bracket test là TỔNG HỢP từ 1 ảnh — chỉ chứng minh "
          "đường ống merge->AI->grade chạy thông, KHÔNG chứng minh lợi ích HDR thật.")
    print("TASK DONE")


def main():
    ap = argparse.ArgumentParser(
        description="Giao ảnh HDR: bracket -> merge -> operator CH_C -> grade -> q100.")
    ap.add_argument("--test", action="store_true", help="Tự test với bracket tổng hợp")
    ap.add_argument("--sample", default=None, help="Ảnh mẫu cho --test")
    ap.add_argument("--folder", default=None, help="Folder ảnh cần gom bracket + deliver")
    ap.add_argument("--group-size", type=int, default=0,
                    help="Số ảnh mỗi bracket (0 = tự đoán theo EXIF)")
    ap.add_argument("--out", default="outputs/bracket_deliver", help="Thư mục lưu ảnh ra")
    ap.add_argument("--no-grade", action="store_true", help="Bỏ bước grade")
    ap.add_argument("--ckpt", default="checkpoints/gpu/CH_C.pt", help="Checkpoint model màu")
    args = ap.parse_args()

    if args.test:
        _run_test(args.sample, args.ckpt)
    elif args.folder:
        deliver_folder(args.folder, args.out, group_size=args.group_size,
                       grade=not args.no_grade, ckpt=args.ckpt)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
