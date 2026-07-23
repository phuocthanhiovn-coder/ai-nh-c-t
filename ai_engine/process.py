# -*- coding: utf-8 -*-
"""Entry point product: nhận folder ảnh khách BẤT KỲ -> giao ảnh đã chỉnh.

Ảnh đơn (mặc định): mỗi ảnh -> operator CH_C full-res -> grade -> q100 4:4:4.
Bracket (--brackets N): gom N ảnh/bộ -> merge Mertens -> operator -> grade.
Không phụ thuộc data/pairs (chỉ dùng cho --selftest).

CLI:
  python -m ai_engine.process --in <folder> --out <folder> [--brackets N] [--no-grade]
  python -m ai_engine.process --selftest
"""

import argparse
import glob
import os
import shutil
import sys

# Console Windows mặc định cp1252 không in được tiếng Việt -> ép UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

import cv2

cv2.setNumThreads(2)

import torch

torch.set_num_threads(2)

from ai_engine.specialists.auto_enhance.gpu.render_delivery import apply_fullres, save_hq
from ai_engine.specialists.auto_enhance.gpu.finish_grade import grade_auto
from ai_engine.specialists.auto_enhance.bracket_deliver import load_model, deliver_bracket
from ai_engine.specialists.auto_enhance.bracket_merge import group_brackets

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")


def _list_images(folder):
    """Liệt kê ảnh trong folder (không đệ quy), sort tên."""
    return sorted(
        p for p in glob.glob(os.path.join(folder, "*"))
        if os.path.isfile(p) and p.lower().endswith(_IMG_EXTS)
    )


def _report(out_path):
    """In tên, WxH, KB của 1 file vừa lưu; trả về KB."""
    img = cv2.imread(out_path)
    h, w = (img.shape[:2] if img is not None else (0, 0))
    kb = os.path.getsize(out_path) / 1024
    print(f"  {os.path.basename(out_path)}  {w}x{h}  {kb:.0f} KB")
    return kb


def process_folder(in_dir, out_dir, brackets=1, grade=True,
                   ckpt="checkpoints/gpu/CH_C.pt"):
    """Nhận folder ảnh khách -> lưu ảnh đã chỉnh vào out_dir (q100 4:4:4, giữ res gốc).

    brackets == 1: mỗi ảnh đơn -> operator -> grade -> <ten_goc>_edited.jpg.
    brackets  > 1: gom bộ N ảnh -> merge -> operator -> grade -> <ten_dau>_hdr.jpg.
    Trả về số ảnh giao thành công.
    """
    device = torch.device("cpu")
    print(f"[process] Nạp model {ckpt} (CPU)...")
    model, device = load_model(ckpt, device)
    os.makedirs(out_dir, exist_ok=True)

    n_ok, total_kb = 0, 0.0

    if brackets <= 1:
        paths = _list_images(in_dir)
        if not paths:
            print(f"[process] Không có ảnh trong {in_dir}")
            return 0
        for p in paths:
            try:
                bgr = cv2.imread(p, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError("cv2 không đọc được ảnh")
                ai = apply_fullres(model, bgr, device)
                out = grade_auto(ai, p) if grade else ai
                stem = os.path.splitext(os.path.basename(p))[0]
                out_path = os.path.join(out_dir, f"{stem}_edited.jpg")
                save_hq(out_path, out)
                total_kb += _report(out_path)
                n_ok += 1
            except Exception as e:
                print(f"[process] CẢNH BÁO: bỏ qua {p} ({e})")
    else:
        # Bracket: cho phép ảnh nằm ở in_dir hoặc trong các subfolder cấp 1.
        folders = [in_dir] + sorted(
            d for d in glob.glob(os.path.join(in_dir, "*")) if os.path.isdir(d)
        )
        groups = []
        for d in folders:
            if _list_images(d):
                groups.extend(group_brackets(d, group_size=brackets))
        if not groups:
            print(f"[process] Không có ảnh trong {in_dir}")
            return 0
        for grp in groups:
            try:
                if len(grp) < 2:
                    raise ValueError(f"bộ chỉ có {len(grp)} ảnh")
                out = deliver_bracket(grp, model, device, grade=grade)
                stem = os.path.splitext(os.path.basename(grp[0]))[0]
                out_path = os.path.join(out_dir, f"{stem}_hdr.jpg")
                save_hq(out_path, out)
                total_kb += _report(out_path)
                n_ok += 1
            except Exception as e:
                print(f"[process] CẢNH BÁO: bỏ qua bộ {grp} ({e})")

    print(f"[process] Xong: {n_ok} ảnh -> {out_dir}  (tổng {total_kb / 1024:.1f} MB)")
    return n_ok


def _selftest():
    """Copy 2 ảnh mẫu -> process_folder -> kiểm tra 2 file *_edited.jpg hợp lệ."""
    src = sorted(glob.glob(os.path.join("data", "pairs", "before", "*.jpg")))[:2]
    if len(src) < 2:
        print("SELFTEST FAIL: cần 2 ảnh mẫu trong data/pairs/before/")
        print("TASK DONE")
        return

    in_dir = os.path.join("outputs", "_proc_selftest_in")
    out_dir = os.path.join("outputs", "_proc_selftest_out")
    for d in (in_dir, out_dir):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    for p in src:
        shutil.copy2(p, in_dir)

    process_folder(in_dir, out_dir, brackets=1)

    outs = sorted(glob.glob(os.path.join(out_dir, "*_edited.jpg")))
    fails = []
    if len(outs) != 2:
        fails.append(f"cần đúng 2 file *_edited.jpg, có {len(outs)}")
    for p in outs:
        if cv2.imread(p) is None:
            fails.append(f"file hỏng (cv2 không mở được): {p}")
        kb = os.path.getsize(p) / 1024
        if kb <= 200:
            fails.append(f"file quá nhỏ ({kb:.0f} KB <= 200): {p}")

    if fails:
        print("SELFTEST FAIL:")
        for f in fails:
            print(f"  - {f}")
    else:
        print("SELFTEST PASS")
    print("TASK DONE")


def main():
    ap = argparse.ArgumentParser(
        description="Nhận folder ảnh khách -> giao ảnh đã chỉnh (q100 4:4:4, giữ res gốc).")
    ap.add_argument("--in", dest="in_dir", default=None, help="Folder ảnh khách")
    ap.add_argument("--out", dest="out_dir", default=None, help="Folder lưu ảnh ra")
    ap.add_argument("--brackets", type=int, default=1,
                    help="Số ảnh mỗi bộ bracket (1 = ảnh đơn)")
    ap.add_argument("--no-grade", action="store_true", help="Tắt bước grade")
    ap.add_argument("--ckpt", default="checkpoints/gpu/CH_C.pt", help="Checkpoint model")
    ap.add_argument("--selftest", action="store_true", help="Tự test rồi thoát")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if not args.in_dir or not args.out_dir:
        ap.print_help()
        sys.exit(1)
    process_folder(args.in_dir, args.out_dir, brackets=args.brackets,
                   grade=not args.no_grade, ckpt=args.ckpt)


if __name__ == "__main__":
    main()
