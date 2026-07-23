"""
Tai 1 folder Google Drive chua RAW/before, ghep voi POOL after local (da tai san),
roi xoa RAW. Song sinh voi fetch_dropbox_job.py nhung nguon before = Google Drive.
Dung cho lo data moi 2026-07-14 (before=Drive, after=Dropbox pool).
"""
import os
import shutil
import argparse

import cv2

cv2.setNumThreads(2)

from ai_engine.data_pairing.fetch_job import (
    download_gdrive_folder,
    check_disk_space,
    is_job_done,
    mark_job_done,
)
from ai_engine.data_pairing.ingest import run_ingest

RAW_INCOMING_DIR = "data/raw_incoming"


def fetch_and_ingest(job_name, url, after_root, keep_raw=False, force=False):
    if not check_disk_space(40):
        print("[X] DUNG: o dia duoi 40GB trong.")
        return False
    if not force and is_job_done(job_name, url, url):
        print(f"[*] BO QUA: job '{job_name}' da xu ly.")
        return True
    if not os.path.isdir(after_root) or not any(os.scandir(after_root)):
        print(f"[X] DUNG: pool after '{after_root}' trong.")
        return False

    dest = os.path.join(RAW_INCOMING_DIR, job_name)
    if os.path.exists(dest):
        shutil.rmtree(dest, ignore_errors=True)

    if not download_gdrive_folder(url, dest):
        print("[X] Tai Google Drive that bai. Giu thu muc do de debug.")
        return False

    print(f"[*] Ingest job '{job_name}' (before={dest}, after=POOL {after_root})...")
    try:
        run_ingest(reset=False, before_root=dest, after_root=after_root, job_name=job_name)
    except Exception as e:
        print(f"[X] Loi ingest: {e}. Giu RAW de debug.")
        return False

    if not keep_raw:
        print(f"[*] Xoa RAW incoming: {dest}")
        shutil.rmtree(dest, ignore_errors=True)

    mark_job_done(job_name, url, url)
    print(f"[OK] JOB {job_name} XONG.")
    return True


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--after-root", required=True)
    p.add_argument("--keep-raw", action="store_true")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    ok = fetch_and_ingest(a.name, a.url, a.after_root, a.keep_raw, a.force)
    raise SystemExit(0 if ok else 1)
