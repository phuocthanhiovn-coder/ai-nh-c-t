"""
Xu ly HANG LOAT tu datalist.csv (cot A = before/chua sua, cot B = after/da sua).
Moi job: tai A->before, B->after -> route (RAW bracket / JPG ghep ten) -> xoa RAW.
Cuoi: rescue + dedup. Co checkpoint (jobs_done_sheet.txt) de chay lai bo qua job da xong.
Chay: python -m ai_engine.data_pairing.ingest_sheet [--limit N] [--start K]
"""
import os
import sys
import csv
import re
import shutil
import argparse

import cv2
import numpy as np

cv2.setNumThreads(3)

from ai_engine.data_pairing.fetch_job import download_dropbox_folder, download_gdrive_folder
from ai_engine.data_pairing.ingest import run_ingest, align_before_after
from ai_engine.data_pairing.undistort import estimate_undistort
from ai_engine.data_pairing import rescue_review, dedup_pairs

CSV = "data/newbatch/datalist.csv"
STAGE = "data/raw_incoming/sheet"
DONE = "data/newbatch/jobs_done_sheet.txt"
RAW_EXTS = (".cr3", ".arw", ".dng", ".nef", ".cr2", ".orf", ".rw2")
JPG_EXTS = (".jpg", ".jpeg", ".png")
ALIGN_OK = 0.50
EDIT_SUFFIX = re.compile(r"(_edited|_edit|_final|_hdr|_replaced_camera|-edited)$", re.I)


def log(m):
    print(m, flush=True)


def norm(name):
    return EDIT_SUFFIX.sub("", os.path.splitext(name)[0]).lower()


def imgs(d, exts):
    out = []
    for root, _, files in os.walk(d):
        for f in files:
            if f.lower().endswith(exts):
                out.append(os.path.join(root, f))
    return out


def free_gb():
    return shutil.disk_usage(".").free / (1024**3)


def download(url, dest):
    os.makedirs(dest, exist_ok=True)
    if "drive.google" in url:
        m = re.search(r"/folders/([A-Za-z0-9_-]+)", url)
        if not m:
            return False
        import subprocess
        rc = "C:/Users/Administrator/Desktop/autohdr/tools/rclone.exe"
        subprocess.run([rc, "copy", "gdrive:", dest, "--drive-root-folder-id", m.group(1),
                        "--transfers", "8", "--drive-acknowledge-abuse", "--ignore-existing"],
                       check=False, timeout=1800)
        if not imgs(dest, JPG_EXTS + RAW_EXTS):
            _extract_archives(dest)  # job kieu k001: RAW nen trong .zip
        return len(imgs(dest, JPG_EXTS + RAW_EXTS)) > 0
    elif "dropbox" in url:
        ok = download_dropbox_folder(url, dest)
        if not ok or not imgs(dest, JPG_EXTS + RAW_EXTS):
            _extract_archives(dest)
        return len(imgs(dest, JPG_EXTS + RAW_EXTS)) > 0
    return False  # wetransfer & unknown: skip


def _extract_archives(dest):
    """Giai nen moi .zip trong dest (de quy 2 vong cho zip long zip), xoa zip sau khi bung."""
    import zipfile
    for _round in range(2):
        found = False
        for root, _, files in os.walk(dest):
            for f in files:
                if not f.lower().endswith(".zip"):
                    continue
                p = os.path.join(root, f)
                try:
                    with zipfile.ZipFile(p) as z:
                        z.extractall(root)
                    os.remove(p)
                    found = True
                    log(f"    [zip] bung {f}")
                except Exception as e:
                    log(f"    [zip] LOI {f}: {e}")
        if not found:
            break


def pair_jpg(before_dir, after_dir, prefix):
    """Ghep before(JPG) <-> after(JPG) theo ten file + gate align (+undistort rescue)."""
    bmap = {norm(os.path.basename(f)): f for f in imgs(before_dir, JPG_EXTS)}
    os.makedirs("data/pairs/before", exist_ok=True)
    os.makedirs("data/pairs/after", exist_ok=True)
    clean = rescued = 0
    for af in imgs(after_dir, JPG_EXTS):
        bf = bmap.get(norm(os.path.basename(af)))
        if not bf:
            continue
        a = cv2.imread(af)
        b = cv2.imread(bf)
        if a is None or b is None:
            continue
        h, w = a.shape[:2]
        b_res = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
        out_name = f"{prefix}_{norm(os.path.basename(af))}.jpg"
        dst_b = os.path.join("data/pairs/before", out_name)
        if os.path.exists(dst_b):
            continue
        score, aligned = align_before_after(b_res, a)
        if score >= ALIGN_OK:
            cv2.imwrite(dst_b, aligned, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(os.path.join("data/pairs/after", out_name), a, [cv2.IMWRITE_JPEG_QUALITY, 95])
            clean += 1
        else:
            try:
                _, und, ncc2 = estimate_undistort(b_res, a)
                if ncc2 >= ALIGN_OK:
                    cv2.imwrite(dst_b, und, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    cv2.imwrite(os.path.join("data/pairs/after", out_name), a, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    rescued += 1
            except Exception:
                pass
    return clean + rescued


def pairs_now():
    d = "data/pairs/before"
    return len([f for f in os.listdir(d) if f.lower().endswith((".jpg", ".png"))]) if os.path.isdir(d) else 0


def load_done():
    if not os.path.exists(DONE):
        return set()
    return set(l.strip() for l in open(DONE, encoding="utf-8") if l.strip())


def mark_done(job):
    with open(DONE, "a", encoding="utf-8") as f:
        f.write(job + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--no-post", action="store_true", help="bo qua rescue+dedup cuoi")
    ap.add_argument("--prefix", default="j", help="prefix ten job/cap (batch moi dung prefix khac de khong trung ten cap cu)")
    a = ap.parse_args()

    rows = []
    with open(CSV, encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) >= 2 and row[0].strip() and row[1].strip():
                rows.append((row[0].strip(), row[1].strip()))

    done = load_done()
    log(f"=== {len(rows)} job | da xong {len(done)} | chay tu {a.start} toi da {a.limit} ===")
    processed = 0
    for i, (burl, aurl) in enumerate(rows):
        if i < a.start:
            continue
        if processed >= a.limit:
            break
        name = f"{a.prefix}{i:03d}"
        if name in done:
            continue
        if "wetransfer" in burl or "wetransfer" in aurl:
            log(f"[{name}] SKIP wetransfer"); mark_done(name); continue
        if free_gb() < 45:
            log(f"[{name}] STOP disk {free_gb():.0f}GB < 45"); break

        before0 = pairs_now()
        jobdir = os.path.join(STAGE, name)
        bdir = os.path.join(jobdir, "before")
        adir = os.path.join(jobdir, "after")
        if os.path.exists(jobdir):
            shutil.rmtree(jobdir, ignore_errors=True)
        log(f"\n[{name}] disk {free_gb():.0f}GB | tai before+after...")
        try:
            ok_b = download(burl, bdir)
            ok_a = download(aurl, adir)
            if not ok_b or not ok_a:
                # KHONG mark_done — de lan chay lai (sau khi sua link/tool) tu retry.
                log(f"[{name}] tai that bai (before={ok_b} after={ok_a}), bo qua (se retry lan sau)")
                shutil.rmtree(jobdir, ignore_errors=True)
                continue
            nraw = len(imgs(bdir, RAW_EXTS))
            log(f"[{name}] before: {nraw} RAW / {len(imgs(bdir, JPG_EXTS))} JPG | after: {len(imgs(adir, JPG_EXTS))} JPG")
            if nraw > 0:
                run_ingest(reset=False, before_root=bdir, after_root=adir, job_name=name)
            else:
                added = pair_jpg(bdir, adir, name)
                log(f"[{name}] JPG-pair: +{added}")
        except Exception as e:
            import traceback
            log(f"[{name}] LOI: {e}")
            traceback.print_exc()
        finally:
            shutil.rmtree(jobdir, ignore_errors=True)
        mark_done(name)
        processed += 1
        log(f"[{name}] xong. pairs: {before0} -> {pairs_now()}")

    if not a.no_post:
        log("\n=== RESCUE ===");
        try: rescue_review.run_rescue()
        except Exception as e: log(f"rescue loi {e}")
        log("=== DEDUP ===")
        try: dedup_pairs.main()
        except Exception as e: log(f"dedup loi {e}")
    log(f"\n=== XONG batch. Tong cap sach: {pairs_now()} ===")


if __name__ == "__main__":
    main()
