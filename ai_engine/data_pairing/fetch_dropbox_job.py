import os
import shutil
import argparse

import cv2

cv2.setNumThreads(2)

from ai_engine.data_pairing.fetch_job import (
    download_dropbox_folder,
    check_disk_space,
    is_job_done,
    mark_job_done,
)
from ai_engine.data_pairing.ingest import run_ingest

RAW_INCOMING_DIR = "data/raw_incoming"
AFTER_POOL_DIR = "data/raw_incoming/after_pool"


def fetch_and_ingest(job_name, url, after_root=AFTER_POOL_DIR, keep_raw=False, force=False):
    """Tải 1 link Dropbox chứa RAW/before, ghép với POOL after đã tải sẵn, rồi xóa RAW.

    Khác fetch_job.py (cần cả 2 link): ở đây after là pool local dùng chung,
    vì chủ đưa 6 link before + 8 link after KHÔNG có ánh xạ 1-1.
    """
    if not check_disk_space(40):
        print("[✗] DỪNG: Ổ đĩa dưới 40GB trống.")
        return False

    # Khoa nhan dang = URL Dropbox (duy nhat moi job). KHONG dung after_root vi moi job
    # Dropbox chung 1 after_pool -> se trung nhau het sau job dau tien (bug da gap 2026-07-14).
    if not force and is_job_done(job_name, url, url):
        print(f"[*] BỎ QUA: Job '{job_name}' đã xử lý. Dùng --force để chạy lại.")
        return True

    if not os.path.isdir(after_root) or not any(os.scandir(after_root)):
        print(f"[✗] DỪNG: Pool after '{after_root}' trống — tải after trước đã.")
        return False

    dest = os.path.join(RAW_INCOMING_DIR, job_name)
    if os.path.exists(dest):
        shutil.rmtree(dest, ignore_errors=True)

    if not download_dropbox_folder(url, dest):
        print("[✗] Tải Dropbox thất bại. Giữ thư mục dở để debug.")
        return False

    print(f"[*] Ingest job '{job_name}' (before={dest}, after=POOL {after_root})...")
    try:
        run_ingest(reset=False, before_root=dest, after_root=after_root, job_name=job_name)
    except Exception as e:
        print(f"[✗] Lỗi ingest: {e}. Giữ RAW để debug.")
        return False

    if not keep_raw:
        print(f"[*] Xóa RAW incoming: {dest}")
        shutil.rmtree(dest, ignore_errors=True)

    mark_job_done(job_name, url, url)
    print(f"[✓] JOB {job_name} XONG.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tải 1 job Dropbox (before RAW) và ghép với after-pool local")
    parser.add_argument("--name", required=True, help="Tên job, vd db01")
    parser.add_argument("--url", required=True, help="Link Dropbox folder chứa RAW before")
    parser.add_argument("--after-root", default=AFTER_POOL_DIR)
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    ok = fetch_and_ingest(args.name, args.url, args.after_root, args.keep_raw, args.force)
    raise SystemExit(0 if ok else 1)
