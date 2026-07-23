"""
Khu trung lap cap train: db01__ML_X.jpg vs _ML_X.jpg = CUNG canh (942 house
tai ingest tu 2 nguon). Giu ban co Edge-NCC (before vs after da luu) cao hon,
chuyen ban thua sang data/pairs_dedup_removed/ (KHONG xoa).
"""
import os
import shutil
import cv2
import numpy as np

cv2.setNumThreads(2)

PAIRS = "data/pairs"
REMOVED = "data/pairs_dedup_removed"


def edge_ncc(before_path, after_path):
    b = cv2.imread(before_path, cv2.IMREAD_GRAYSCALE)
    a = cv2.imread(after_path, cv2.IMREAD_GRAYSCALE)
    if b is None or a is None:
        return -1.0
    h = min(b.shape[0], a.shape[0]); w = min(b.shape[1], a.shape[1])
    b = cv2.resize(b, (w, h)); a = cv2.resize(a, (w, h))
    eb = cv2.Sobel(b, cv2.CV_32F, 1, 1, ksize=3)
    ea = cv2.Sobel(a, cv2.CV_32F, 1, 1, ksize=3)
    eb = (eb - eb.mean()) / (eb.std() + 1e-6)
    ea = (ea - ea.mean()) / (ea.std() + 1e-6)
    return float((eb * ea).mean())


def main():
    before_dir = os.path.join(PAIRS, "before")
    files = [f for f in os.listdir(before_dir) if f.lower().endswith((".jpg", ".png"))]
    db01 = [f for f in files if f.startswith("db01_")]

    os.makedirs(os.path.join(REMOVED, "before"), exist_ok=True)
    os.makedirs(os.path.join(REMOVED, "after"), exist_ok=True)

    moved = 0
    for f in db01:
        orig = f[len("db01_"):]  # db01__ML_1710.jpg -> _ML_1710.jpg
        if orig not in files:
            continue  # db01 canh moi, khong trung -> giu
        ncc_db01 = edge_ncc(os.path.join(PAIRS, "before", f), os.path.join(PAIRS, "after", f))
        ncc_orig = edge_ncc(os.path.join(PAIRS, "before", orig), os.path.join(PAIRS, "after", orig))
        loser = f if ncc_db01 < ncc_orig else orig
        keeper = orig if loser == f else f
        for sub in ("before", "after"):
            src = os.path.join(PAIRS, sub, loser)
            if os.path.exists(src):
                shutil.move(src, os.path.join(REMOVED, sub, loser))
        moved += 1
        print(f"  {orig}: db01={ncc_db01:.3f} vs orig={ncc_orig:.3f} -> GIU {keeper}, chuyen {loser}")

    remain = len([f for f in os.listdir(before_dir) if f.lower().endswith((".jpg", ".png"))])
    print(f"\n[+] Da khu {moved} canh trung. Con lai {remain} cap sach (khong trung).")


if __name__ == "__main__":
    main()
