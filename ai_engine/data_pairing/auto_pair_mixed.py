"""
TỰ ĐỘNG tách before/after từ 1 folder TRỘN (chưa sửa + đã sửa lẫn lộn) và ghép cặp.
Xử lý các dạng thật:
  - RAW + JPG lẫn nhau: RAW=before, JPG=after.
  - 2 subfolder (nguồn vs đã-giao): folder có tên ảnh là TẬP CON khớp = after; folder kia = before.
  - Ghép theo TÊN FILE chính xác (bỏ đuôi + hậu tố edit).
Với JPG-đã-đơn (không cần Mertens), pair thẳng + gate Edge-NCC (+undistort rescue) rồi thêm vào data/pairs.
Chạy: python -m ai_engine.data_pairing.auto_pair_mixed <folder> <job_prefix>
"""
import os
import sys
import re
import shutil
import cv2
import numpy as np

cv2.setNumThreads(2)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ai_engine.data_pairing.ingest import align_before_after  # noqa: E402
from ai_engine.data_pairing.undistort import estimate_undistort  # noqa: E402

RAW_EXTS = (".cr3", ".arw", ".dng", ".nef", ".cr2")
JPG_EXTS = (".jpg", ".jpeg", ".png")
ALIGN_OK = 0.50
EDIT_SUFFIX = re.compile(r"(_edited|_edit|_final|_hdr|_replaced_camera)$", re.I)


def norm(name):
    base = os.path.splitext(name)[0]
    return EDIT_SUFFIX.sub("", base)


def list_imgs(d, exts):
    out = []
    for root, _, files in os.walk(d):
        for f in files:
            if f.lower().endswith(exts):
                out.append(os.path.join(root, f))
    return out


def detect_before_after(folder):
    """Trả về (before_files, after_files) — list đường dẫn."""
    raws = list_imgs(folder, RAW_EXTS)
    jpgs = list_imgs(folder, JPG_EXTS)
    # Case 1: co RAW -> RAW=before, JPG=after
    if raws:
        return raws, jpgs
    # Case 2: toan JPG -> chia theo subfolder truc tiep
    subdirs = [os.path.join(folder, d) for d in os.listdir(folder)
               if os.path.isdir(os.path.join(folder, d))]
    subdirs = [d for d in subdirs if list_imgs(d, JPG_EXTS)]
    if len(subdirs) == 2:
        a, b = subdirs
        na = {norm(os.path.basename(f)) for f in list_imgs(a, JPG_EXTS)}
        nb = {norm(os.path.basename(f)) for f in list_imgs(b, JPG_EXTS)}
        # folder nho hon + la tap con cua folder kia => after (da chon loc)
        if len(na) <= len(nb) and len(na & nb) >= 0.5 * len(na):
            return list_imgs(b, JPG_EXTS), list_imgs(a, JPG_EXTS)
        if len(nb) <= len(na) and len(na & nb) >= 0.5 * len(nb):
            return list_imgs(a, JPG_EXTS), list_imgs(b, JPG_EXTS)
    # fallback: khong ro -> tra rong
    return [], []


def main(folder, prefix):
    before_files, after_files = detect_before_after(folder)
    print(f"[auto-split] before(nguon)={len(before_files)}  after(da sua)={len(after_files)}")
    if not before_files or not after_files:
        print("[!] Khong tach duoc before/after tu cau truc nay.")
        return

    before_map = {norm(os.path.basename(f)): f for f in before_files}
    pairs_dir_b = "data/pairs/before"
    pairs_dir_a = "data/pairs/after"
    os.makedirs(pairs_dir_b, exist_ok=True)
    os.makedirs(pairs_dir_a, exist_ok=True)

    matched = 0
    clean = 0
    rescued = 0
    skipped = 0
    for af in after_files:
        key = norm(os.path.basename(af))
        bf = before_map.get(key)
        if bf is None:
            skipped += 1
            continue
        matched += 1
        a_img = cv2.imread(af)
        b_img = cv2.imread(bf)
        if a_img is None or b_img is None:
            skipped += 1
            continue
        # dua ve cung do phan giai (after la chuan giao hang)
        h, w = a_img.shape[:2]
        b_res = cv2.resize(b_img, (w, h), interpolation=cv2.INTER_AREA)
        score, aligned = align_before_after(b_res, a_img)
        out_name = f"{prefix}_{key}.jpg"
        if score >= ALIGN_OK:
            cv2.imwrite(os.path.join(pairs_dir_b, out_name), aligned, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(os.path.join(pairs_dir_a, out_name), a_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            clean += 1
        else:
            # thu undistort rescue
            try:
                k1, und, ncc2 = estimate_undistort(b_res, a_img)
                if ncc2 >= ALIGN_OK:
                    cv2.imwrite(os.path.join(pairs_dir_b, out_name), und, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    cv2.imwrite(os.path.join(pairs_dir_a, out_name), a_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    rescued += 1
                else:
                    skipped += 1
            except Exception:
                skipped += 1
        if matched % 20 == 0:
            print(f"  ...{matched} matched, {clean} clean, {rescued} rescued")

    total = len([f for f in os.listdir(pairs_dir_b) if f.lower().endswith((".jpg", ".png"))])
    print(f"\n[+] {prefix}: matched={matched} clean={clean} rescued={rescued} skipped={skipped}")
    print(f"[+] Tong cap trong data/pairs/before hien tai: {total}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "mixed")
