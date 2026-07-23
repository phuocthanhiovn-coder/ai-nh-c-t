"""
Lo data MOI 2026-07-14 — BAN SUA VAI (chu dan nham nhan 2 file txt):
  THUC TE: Dropbox = RAW before (.CR3/.ARW) | Google Drive = JPG after (AutoHDR da chinh).
Flow:
  P1: tai 24 link Drive -> data/newbatch/after_pool2/gdXX (JPG). Log folder 0-file de retry.
  P2: gom RAW da tai nham vao after_pool/aXX -> lam before jobs; tai not link Dropbox thieu.
  P3: ingest tung before job vs after_pool2 -> purge RAW.
  P4: rescue + dedup. In tong ket.
Chay: python -m ai_engine.data_pairing.newbatch_ingest2
"""
import os
import shutil
import traceback

import cv2

cv2.setNumThreads(3)

from ai_engine.data_pairing.fetch_job import (
    download_dropbox_folder, download_gdrive_folder,
)
from ai_engine.data_pairing.ingest import run_ingest
from ai_engine.data_pairing import rescue_review, dedup_pairs

NB = "data/newbatch"
OLD_AFTER_POOL = os.path.join(NB, "after_pool")      # thuc ra chua RAW before
AFTER_POOL2 = os.path.join(NB, "after_pool2")         # JPG after that (tu Drive)
BEFORE_LINKS = os.path.join(NB, "before_links.txt")   # 24 link DRIVE = AFTER that
AFTER_LINKS = os.path.join(NB, "after_links.txt")     # 7 link DROPBOX = BEFORE that
RAW_INCOMING = "data/raw_incoming"
RAW_EXTS = (".cr3", ".dng", ".arw", ".nef", ".cr2")
JPG_EXTS = (".jpg", ".jpeg", ".png")


def log(m):
    print(m, flush=True)


def count_imgs(d, exts):
    n = 0
    for root, _, files in os.walk(d):
        n += sum(1 for f in files if f.lower().endswith(exts))
    return n


def free_gb():
    return shutil.disk_usage(".").free / (1024 ** 3)


def main():
    os.makedirs(AFTER_POOL2, exist_ok=True)
    # utf-8-sig: PowerShell 5.1 ghi file kem BOM -> phai strip keo link dau tien hong URL
    drive_links = [l.strip() for l in open(BEFORE_LINKS, encoding="utf-8-sig") if l.strip()]
    seen = set(); drive_links = [x for x in drive_links if not (x in seen or seen.add(x))]
    dropbox_links = [l.strip() for l in open(AFTER_LINKS, encoding="utf-8-sig") if l.strip()]

    log(f"=== SUA VAI: {len(drive_links)} Drive(AFTER jpg) + {len(dropbox_links)} Dropbox(BEFORE raw) ===")

    # P1: AFTER pool tu Drive
    log("\n--- P1: tai AFTER pool (Drive JPG) ---")
    empty = []
    for i, url in enumerate(drive_links, 1):
        sub = os.path.join(AFTER_POOL2, f"gd{i:02d}")
        if os.path.isdir(sub) and count_imgs(sub, JPG_EXTS) > 0:
            log(f"  gd{i:02d}: da co {count_imgs(sub, JPG_EXTS)} jpg, bo qua")
            continue
        try:
            download_gdrive_folder(url, sub)
        except Exception as e:
            log(f"  gd{i:02d}: LOI {e}")
        n = count_imgs(sub, JPG_EXTS)
        log(f"  gd{i:02d}: jpg={n}")
        if n == 0:
            empty.append(f"gd{i:02d}")
    total_after = count_imgs(AFTER_POOL2, JPG_EXTS)
    log(f"  => AFTER pool2: {total_after} jpg | folder rong (rate-limit?): {empty}")
    if total_after == 0:
        log("[X] After pool2 trong -> dung (Drive rate-limit? thu lai sau).")
        return

    # P2: gom before jobs (RAW)
    log("\n--- P2: gom BEFORE jobs (Dropbox RAW) ---")
    before_jobs = []
    # 2a: RAW da tai nham trong after_pool/aXX -> move sang raw_incoming
    if os.path.isdir(OLD_AFTER_POOL):
        for d in sorted(os.listdir(OLD_AFTER_POOL)):
            src = os.path.join(OLD_AFTER_POOL, d)
            if os.path.isdir(src) and count_imgs(src, RAW_EXTS) > 0:
                dst = os.path.join(RAW_INCOMING, f"nb_{d}")
                if os.path.exists(dst):
                    shutil.rmtree(dst, ignore_errors=True)
                shutil.move(src, dst)
                before_jobs.append(dst)
                log(f"  move {src} -> {dst} ({count_imgs(dst, RAW_EXTS)} RAW)")
    # 2b: tai not cac link Dropbox chua co RAW tuong ung
    have = len(before_jobs)
    for i, url in enumerate(dropbox_links, 1):
        name = f"nb_a{i:02d}"
        dst = os.path.join(RAW_INCOMING, name)
        if any(os.path.basename(b) == name for b in before_jobs):
            continue
        # neu i <= so job da move (a02-a04 -> nb_a02...), kiem tra ton tai
        if os.path.isdir(dst) and count_imgs(dst, RAW_EXTS) > 0:
            before_jobs.append(dst); continue
        if free_gb() < 45:
            log(f"  {name}: SKIP (disk {free_gb():.1f}GB < 45)")
            continue
        try:
            ok = download_dropbox_folder(url, dst)
            n = count_imgs(dst, RAW_EXTS)
            log(f"  {name}: download={ok}, RAW={n}")
            if ok and n > 0:
                before_jobs.append(dst)
            else:
                shutil.rmtree(dst, ignore_errors=True)
        except Exception as e:
            log(f"  {name}: LOI {e}")
            shutil.rmtree(dst, ignore_errors=True)

    log(f"  => {len(before_jobs)} before jobs: {[os.path.basename(b) for b in before_jobs]}")

    # P3: ingest tung job -> purge RAW
    log("\n--- P3: ingest ---")
    for dest in before_jobs:
        name = os.path.basename(dest)
        log(f"\n[{name}] disk {free_gb():.1f}GB")
        try:
            run_ingest(reset=False, before_root=dest, after_root=AFTER_POOL2, job_name=name)
        except Exception as e:
            log(f"  {name}: LOI ingest {e}")
            traceback.print_exc()
        finally:
            shutil.rmtree(dest, ignore_errors=True)
            log(f"  {name}: RAW purged")

    # P4: rescue + dedup
    log("\n--- P4: RESCUE ---")
    try:
        rescue_review.run_rescue()
    except Exception as e:
        log(f"  rescue LOI: {e}")
    log("\n--- P4: DEDUP ---")
    try:
        dedup_pairs.main()
    except Exception as e:
        log(f"  dedup LOI: {e}")

    b = len([f for f in os.listdir("data/pairs/before") if f.lower().endswith((".jpg", ".png"))])
    log(f"\n=== XONG LO MOI (sua vai). Tong cap sach: {b} ===")


if __name__ == "__main__":
    main()
