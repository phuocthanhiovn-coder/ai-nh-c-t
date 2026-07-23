"""
Ghep tong lo moi 2026-07-14 — data TRON: ca Drive lan Dropbox deu co folder RAW (before)
lan folder JPG (after). Bo giả định phan vai theo nguon.
Chien luoc:
  - AFTER pool = MOI folder chua JPG (gom vao 1 cho de ingest quet).
  - BEFORE jobs = MOI folder chua RAW (.CR3/.ARW/.DNG/.NEF).
  - Ingest tung before-job vs after-pool, ghep theo TEN FILE (gate align-NCC chan ghep nham).
  - rescue + dedup cuoi.
Chay: python -m ai_engine.data_pairing.ingest_mixed
"""
import os
import shutil
import traceback

import cv2

cv2.setNumThreads(3)

from ai_engine.data_pairing.ingest import run_ingest
from ai_engine.data_pairing import rescue_review, dedup_pairs

NB = "data/newbatch"
SCAN_ROOTS = [os.path.join(NB, "after_pool"), os.path.join(NB, "after_pool2")]
AFTER_POOL = os.path.join(NB, "all_after")   # gom tat ca JPG vao day (symlink-free: chi ingest doc)
RAW_EXTS = (".cr3", ".arw", ".dng", ".nef", ".cr2")
JPG_EXTS = (".jpg", ".jpeg", ".png")


def log(m):
    print(m, flush=True)


def has_ext(d, exts):
    for _, _, files in os.walk(d):
        if any(f.lower().endswith(exts) for f in files):
            return True
    return False


def count_ext(d, exts):
    n = 0
    for _, _, files in os.walk(d):
        n += sum(1 for f in files if f.lower().endswith(exts))
    return n


def main():
    # 1) Gom tat ca folder JPG -> all_after (copy vao 1 pool phang de ingest quet nhanh)
    os.makedirs(AFTER_POOL, exist_ok=True)
    jpg_copied = 0
    before_dirs = []
    for root in SCAN_ROOTS:
        if not os.path.isdir(root):
            continue
        for sub in sorted(os.listdir(root)):
            p = os.path.join(root, sub)
            if not os.path.isdir(p):
                continue
            raw_n = count_ext(p, RAW_EXTS)
            jpg_n = count_ext(p, JPG_EXTS)
            if raw_n > 0:
                before_dirs.append((f"{os.path.basename(root)}_{sub}", p))
            if jpg_n > 0:
                # copy jpg (flat, tranh trung ten bang prefix folder)
                for r, _, files in os.walk(p):
                    for f in files:
                        if f.lower().endswith(JPG_EXTS):
                            src = os.path.join(r, f)
                            dst = os.path.join(AFTER_POOL, f)  # giu nguyen ten de ghep prefix+so
                            if not os.path.exists(dst):
                                shutil.copy2(src, dst)
                                jpg_copied += 1
    total_after = count_ext(AFTER_POOL, JPG_EXTS)
    log(f"=== AFTER pool: {total_after} jpg (copy moi {jpg_copied}) | BEFORE jobs: {len(before_dirs)} ===")
    for name, p in before_dirs:
        log(f"   before {name}: {count_ext(p, RAW_EXTS)} RAW")
    if total_after == 0 or not before_dirs:
        log("[X] Thieu after hoac before -> dung.")
        return

    # 2) Ingest tung before-job vs after-pool
    for name, dest in before_dirs:
        log(f"\n[{name}] ingest...")
        try:
            run_ingest(reset=False, before_root=dest, after_root=AFTER_POOL, job_name=name)
        except Exception as e:
            log(f"  {name}: LOI {e}")
            traceback.print_exc()

    # 3) rescue + dedup
    log("\n--- RESCUE ---")
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
