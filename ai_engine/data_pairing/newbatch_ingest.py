"""
Orchestrator lo data MOI 2026-07-14 (before=Google Drive RAW, after=Dropbox JPG).
- Tai 7 folder after -> data/newbatch/after_pool/ (moi link 1 subfolder).
- Lap 23 folder before Drive: tai RAW -> ingest vs after_pool -> xoa RAW.
- Rescue + dedup.
Chay all-in-Python de tranh bay pipe PowerShell. Moi buoc try/except, 1 loi khong giet ca batch.
Chay: python -m ai_engine.data_pairing.newbatch_ingest
"""
import os
import shutil
import traceback

import cv2

cv2.setNumThreads(3)

from ai_engine.data_pairing.fetch_job import (
    download_dropbox_folder, download_gdrive_folder, check_disk_space,
)
from ai_engine.data_pairing.ingest import run_ingest
from ai_engine.data_pairing import rescue_review, dedup_pairs

NB = "data/newbatch"
AFTER_POOL = os.path.join(NB, "after_pool")
BEFORE_LINKS = os.path.join(NB, "before_links.txt")
AFTER_LINKS = os.path.join(NB, "after_links.txt")
RAW_INCOMING = "data/raw_incoming"


def log(msg):
    print(msg, flush=True)


def count_imgs(d, exts):
    n = 0
    for root, _, files in os.walk(d):
        n += sum(1 for f in files if f.lower().endswith(exts))
    return n


def main():
    os.makedirs(AFTER_POOL, exist_ok=True)
    after_links = [l.strip() for l in open(AFTER_LINKS, encoding="utf-8") if l.strip()]
    before_links = [l.strip() for l in open(BEFORE_LINKS, encoding="utf-8") if l.strip()]
    # bo trung link before
    seen = set(); before_links = [x for x in before_links if not (x in seen or seen.add(x))]

    log(f"=== LO MOI: {len(after_links)} after (Dropbox) + {len(before_links)} before (Drive) ===")

    # 1) TAI AFTER POOL
    log("\n--- Tai AFTER pool (Dropbox JPG) ---")
    for i, url in enumerate(after_links, 1):
        sub = os.path.join(AFTER_POOL, f"a{i:02d}")
        if os.path.isdir(sub) and count_imgs(sub, (".jpg", ".jpeg", ".png")) > 0:
            log(f"  a{i:02d}: da co, bo qua")
            continue
        try:
            ok = download_dropbox_folder(url, sub)
            log(f"  a{i:02d}: download={ok}, jpg={count_imgs(sub, ('.jpg','.jpeg','.png'))}")
        except Exception as e:
            log(f"  a{i:02d}: LOI {e}")
    total_after = count_imgs(AFTER_POOL, (".jpg", ".jpeg", ".png"))
    log(f"  => AFTER pool: {total_after} anh")
    if total_after == 0:
        log("[X] After pool trong -> dung.")
        return

    # 2) LAP BEFORE (Drive RAW) -> ingest -> purge
    log("\n--- Ingest tung BEFORE (Drive RAW) ---")
    for i, url in enumerate(before_links, 1):
        name = f"nb{i:02d}"
        free = shutil.disk_usage(".").free / (1024**3)
        log(f"\n[{i}/{len(before_links)}] {name} | disk {free:.1f}GB")
        if free < 45:
            log("  STOP: disk < 45GB")
            break
        dest = os.path.join(RAW_INCOMING, name)
        if os.path.exists(dest):
            shutil.rmtree(dest, ignore_errors=True)
        try:
            if not download_gdrive_folder(url, dest):
                log(f"  {name}: tai Drive that bai, bo qua")
                continue
            raws = count_imgs(dest, (".cr3", ".dng", ".arw", ".nef", ".cr2"))
            log(f"  {name}: RAW={raws}")
            run_ingest(reset=False, before_root=dest, after_root=AFTER_POOL, job_name=name)
        except Exception as e:
            log(f"  {name}: LOI {e}")
            traceback.print_exc()
        finally:
            shutil.rmtree(dest, ignore_errors=True)  # luon xoa RAW giai phong dia

    # 3) RESCUE + DEDUP
    log("\n--- RESCUE (undistort cuu cap align_low) ---")
    try:
        rescue_review.run_rescue()
    except Exception as e:
        log(f"  rescue LOI: {e}")
    log("\n--- DEDUP ---")
    try:
        dedup_pairs.main()
    except Exception as e:
        log(f"  dedup LOI: {e}")

    b = len([f for f in os.listdir("data/pairs/before") if f.lower().endswith((".jpg", ".png"))])
    log(f"\n=== XONG. Tong cap sach: {b} ===")


if __name__ == "__main__":
    main()
