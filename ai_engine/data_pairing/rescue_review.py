import os
import re
import csv
import shutil

import cv2
import numpy as np

cv2.setNumThreads(2)

from .config import REVIEW_DIR, PAIRS_COLOR_DIR, REPORT_CSV_PATH, ALIGN_SCORE_THRESHOLD
from .undistort import estimate_undistort

RESCUE_REPORT_PATH = "data/rescue_report.csv"
RESCUE_SAMPLES_DIR = "outputs/rescue_samples"
MAX_SAMPLES = 10
SAMPLE_MAX_WIDTH = 1500


def build_filename(job_name, prefix, number):
    """Tái tạo đúng quy tắc đặt tên file trong ingest.py::run_ingest."""
    prefix_clean = re.sub(r'[^a-zA-Z0-9_-]', '_', prefix)
    if job_name:
        return f"{job_name}_{prefix_clean}{number}.jpg"
    return f"{prefix_clean}{number}.jpg"


def load_ncc_old_map():
    """Đọc report.csv, trả về dict filename -> align_score (ncc_old) cho các dòng align_low."""
    ncc_map = {}
    if not os.path.exists(REPORT_CSV_PATH):
        return ncc_map
    with open(REPORT_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("align_status") != "align_low":
                continue
            filename = build_filename(row.get("job_name", ""), row["prefix"], row["number"])
            try:
                ncc_map[filename] = float(row["align_score"])
            except (KeyError, ValueError):
                pass
    return ncc_map


def save_rescue_sample(before_undistorted, after_img, output_path, max_width=SAMPLE_MAX_WIDTH):
    """Ghép ngang [before_undistorted | after], downscale để dễ xem."""
    h_b, w_b = before_undistorted.shape[:2]
    h_a, w_a = after_img.shape[:2]
    target_h = min(h_b, h_a)

    def resize_to_h(img, h):
        w = int(img.shape[1] * (h / img.shape[0]))
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    before_r = resize_to_h(before_undistorted, target_h)
    after_r = resize_to_h(after_img, target_h)

    canvas = np.hstack((before_r, after_r))
    ch, cw = canvas.shape[:2]
    if cw > max_width:
        scale = max_width / cw
        canvas = cv2.resize(canvas, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_AREA)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])


def run_rescue():
    review_before_dir = os.path.join(REVIEW_DIR, "before")
    review_after_dir = os.path.join(REVIEW_DIR, "after")
    pairs_before_dir = os.path.join(PAIRS_COLOR_DIR, "before")
    pairs_after_dir = os.path.join(PAIRS_COLOR_DIR, "after")

    os.makedirs(pairs_before_dir, exist_ok=True)
    os.makedirs(pairs_after_dir, exist_ok=True)
    os.makedirs(RESCUE_SAMPLES_DIR, exist_ok=True)

    ncc_old_map = load_ncc_old_map()

    filenames = sorted(
        f for f in os.listdir(review_before_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and os.path.exists(os.path.join(review_after_dir, f))
    )

    print(f"[*] Tìm thấy {len(filenames)} cặp trong data/review/ để thử cứu.")

    rescued_count = 0
    report_rows = []
    samples_saved = 0

    for idx, filename in enumerate(filenames):
        before_path = os.path.join(review_before_dir, filename)
        after_path = os.path.join(review_after_dir, filename)

        before_img = cv2.imread(before_path)
        after_img = cv2.imread(after_path)

        if before_img is None or after_img is None:
            print(f"  [✗] Lỗi đọc file: {filename}")
            continue

        ncc_old = ncc_old_map.get(filename, float("nan"))

        try:
            k1, aligned_before, ncc_new = estimate_undistort(before_img, after_img)
        except Exception as e:
            print(f"  [✗] LỖI khi xử lý {filename}: {e}")
            continue

        rescued = ncc_new >= ALIGN_SCORE_THRESHOLD

        status = ""
        if rescued:
            dest_before = os.path.join(pairs_before_dir, filename)
            dest_after = os.path.join(pairs_after_dir, filename)
            if os.path.exists(dest_before) or os.path.exists(dest_after):
                status = "SKIPPED (trùng tên với pairs hiện có)"
                print(f"  [!] CẢNH BÁO: {filename} đã tồn tại trong data/pairs/, bỏ qua ghi đè.")
                rescued = False
            else:
                cv2.imwrite(dest_before, aligned_before, [cv2.IMWRITE_JPEG_QUALITY, 95])
                shutil.copy(after_path, dest_after)
                rescued_count += 1
                status = "RESCUED"

                if samples_saved < MAX_SAMPLES:
                    sample_path = os.path.join(RESCUE_SAMPLES_DIR, f"sample_{filename}")
                    save_rescue_sample(aligned_before, after_img, sample_path)
                    samples_saved += 1
        else:
            status = "kept in review"

        print(f"  [{idx+1}/{len(filenames)}] {filename}: k1={k1:.4f} ncc_old={ncc_old:.4f} "
              f"ncc_new={ncc_new:.4f} -> {status}")

        report_rows.append({
            "filename": filename,
            "k1": f"{k1:.4f}",
            "ncc_old": f"{ncc_old:.4f}",
            "ncc_new": f"{ncc_new:.4f}",
            "rescued": str(rescued),
        })

    with open(RESCUE_REPORT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "k1", "ncc_old", "ncc_new", "rescued"])
        writer.writeheader()
        for row in report_rows:
            writer.writerow(row)

    print(f"\n[+] HOÀN THÀNH: {rescued_count}/{len(filenames)} rescued")
    print(f"[+] Báo cáo lưu tại: {RESCUE_REPORT_PATH}")
    print(f"[+] Samples lưu tại: {RESCUE_SAMPLES_DIR} ({samples_saved} ảnh)")
    return rescued_count, len(filenames)


if __name__ == "__main__":
    run_rescue()
