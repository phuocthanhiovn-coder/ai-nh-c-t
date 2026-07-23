"""
MỘT LỆNH onboard data: link/folder BẤT KỲ -> tách before/after -> ghép -> align+rescue -> dedup -> pack.
Tự nhận diện cấu trúc (không đoán mò):
  - Có RAW (.CR3/.ARW/...) -> đường ingest bracket (run_ingest, Mertens merge).
  - Toàn JPG, 2 subfolder / tên khớp -> auto_pair_mixed (ghép theo tên).
Nguồn: --url (Drive qua rclone / Dropbox qua zip) HOẶC --dir (folder đã tải sẵn).

Chay:
  python -m ai_engine.data_pairing.ingest_any --dir <folder> --name <job> [--pack]
  python -m ai_engine.data_pairing.ingest_any --url <link>  --name <job> [--pack]
"""
import os
import sys
import argparse
import shutil
import subprocess

import cv2

cv2.setNumThreads(3)

from ai_engine.data_pairing.ingest import run_ingest
from ai_engine.data_pairing import rescue_review, dedup_pairs
from ai_engine.data_pairing import auto_pair_mixed
from ai_engine.data_pairing.fetch_job import download_dropbox_folder

RAW_EXTS = (".cr3", ".arw", ".dng", ".nef", ".cr2", ".orf", ".rw2")
JPG_EXTS = (".jpg", ".jpeg", ".png")
RCLONE = "C:/Users/Administrator/Desktop/autohdr/tools/rclone.exe"
STAGE = "data/raw_incoming"


def log(m):
    print(m, flush=True)


def count(d, exts):
    n = 0
    for _, _, files in os.walk(d):
        n += sum(1 for f in files if f.lower().endswith(exts))
    return n


def pairs_now():
    return len([f for f in os.listdir("data/pairs/before")
                if f.lower().endswith((".jpg", ".png"))]) if os.path.isdir("data/pairs/before") else 0


def download(url, dest):
    os.makedirs(dest, exist_ok=True)
    if "drive.google.com" in url:
        m = None
        import re
        mm = re.search(r"/folders/([A-Za-z0-9_-]+)", url)
        if mm:
            fid = mm.group(1)
            log(f"[dl] rclone Drive folder {fid[:10]}...")
            subprocess.run([RCLONE, "copy", "gdrive:", dest, "--drive-root-folder-id", fid,
                            "--transfers", "8", "--drive-acknowledge-abuse", "--ignore-existing"],
                           check=False)
            return count(dest, JPG_EXTS + RAW_EXTS) > 0
        log("[dl] Drive URL khong parse duoc folder id")
        return False
    elif "dropbox.com" in url:
        log("[dl] Dropbox zip...")
        return download_dropbox_folder(url, dest)
    log(f"[dl] nguon la khong ho tro: {url}")
    return False


def process(folder, name, do_pack=False):
    before0 = pairs_now()
    raws = count(folder, RAW_EXTS)
    jpgs = count(folder, JPG_EXTS)
    log(f"[detect] {folder}: {raws} RAW, {jpgs} JPG")

    if raws > 0:
        # đường bracket: cần before(RAW)+after(JPG). Nếu cả 2 nằm trong 1 folder,
        # tách theo subfolder có RAW = before, subfolder JPG = after.
        subdirs = [os.path.join(folder, d) for d in os.listdir(folder)
                   if os.path.isdir(os.path.join(folder, d))]
        before_root = after_root = None
        raw_subs = [d for d in subdirs if count(d, RAW_EXTS) > 0]
        jpg_subs = [d for d in subdirs if count(d, JPG_EXTS) > 0 and count(d, RAW_EXTS) == 0]
        if raw_subs and jpg_subs:
            before_root, after_root = raw_subs[0], jpg_subs[0]
        elif count(folder, RAW_EXTS) > 0 and count(folder, JPG_EXTS) > 0:
            # RAW + JPG lẫn trong cùng folder phẳng -> ingest tự lọc theo đuôi
            before_root = after_root = folder
        if before_root and after_root:
            log(f"[route] BRACKET ingest (before={before_root}, after={after_root})")
            try:
                run_ingest(reset=False, before_root=before_root, after_root=after_root, job_name=name)
            except Exception as e:
                log(f"[!] run_ingest loi: {e}")
        else:
            log("[!] Co RAW nhung khong tach duoc before/after ro rang.")
    else:
        # toàn JPG -> auto_pair_mixed (2 subfolder / tên khớp)
        log("[route] AUTO-PAIR (JPG, ghep theo ten)")
        try:
            auto_pair_mixed.main(folder, name)
        except Exception as e:
            log(f"[!] auto_pair_mixed loi: {e}")

    # rescue + dedup
    log("[post] rescue (undistort cuu cap align_low)...")
    try:
        rescue_review.run_rescue()
    except Exception as e:
        log(f"[!] rescue loi: {e}")
    log("[post] dedup...")
    try:
        dedup_pairs.main()
    except Exception as e:
        log(f"[!] dedup loi: {e}")

    added = pairs_now() - before0
    log(f"\n[+] JOB '{name}': +{added} cap sach | tong = {pairs_now()}")

    if do_pack:
        log("[pack] dong goi dataset...")
        subprocess.run([sys.executable, "-m", "ai_engine.specialists.auto_enhance.pack_dataset"], check=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=None)
    p.add_argument("--dir", default=None)
    p.add_argument("--name", required=True)
    p.add_argument("--pack", action="store_true")
    p.add_argument("--keep-raw", action="store_true")
    a = p.parse_args()

    if a.dir:
        process(a.dir, a.name, a.pack)
    elif a.url:
        free = shutil.disk_usage(".").free / (1024**3)
        if free < 45:
            log(f"[X] Disk {free:.0f}GB < 45GB, don bot truoc.")
            return
        dest = os.path.join(STAGE, a.name)
        if os.path.exists(dest):
            shutil.rmtree(dest, ignore_errors=True)
        if not download(a.url, dest):
            log("[X] Tai that bai.")
            return
        process(dest, a.name, a.pack)
        if not a.keep_raw:
            shutil.rmtree(dest, ignore_errors=True)
            log("[cleanup] xoa RAW incoming.")
    else:
        log("Can --url hoac --dir")


if __name__ == "__main__":
    main()
