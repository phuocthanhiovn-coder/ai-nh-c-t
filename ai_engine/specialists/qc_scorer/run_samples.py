"""
Cham toan bo data/pairs/before/ VA data/pairs/after/ bang qc.score().
Xuat outputs/qc_report.csv (ten, cac diem, flags, needs_human, before/after).
In top 5 anh te nhat (overall thap nhat) + ti le after > before (bai tu kiem chung thang do).

Chi DOC data/, ghi outputs/qc_report.csv.
"""

import csv
import os
import sys

import cv2
import numpy as np

cv2.setNumThreads(2)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qc  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BEFORE_DIR = os.path.join(ROOT, "data", "pairs", "before")
AFTER_DIR = os.path.join(ROOT, "data", "pairs", "after")
OUT_CSV = os.path.join(ROOT, "outputs", "qc_report.csv")
IMG_EXTS = (".jpg", ".jpeg", ".png")

FIELDS = [
    "name", "which", "overall",
    "blur_score", "exposure_score", "tilt_score", "color_cast_score", "noise_score",
    "flags", "needs_human",
]


def list_images(d):
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXTS))


def score_file(path):
    img_u8 = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_u8 is None:
        return None
    img = img_u8.astype(np.float32) / 255.0
    return qc.score(img)


def main():
    before_names = list_images(BEFORE_DIR)
    after_names = list_images(AFTER_DIR)
    common = sorted(set(before_names) & set(after_names))

    if not common:
        print("KHONG co anh chung ten giua before/after.")
        return

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    rows = []
    crashes = []
    pair_scores = []  # (name, overall_before, overall_after)

    for name in common:
        for which, folder in (("before", BEFORE_DIR), ("after", AFTER_DIR)):
            path = os.path.join(folder, name)
            try:
                r = score_file(path)
            except Exception as e:
                crashes.append((name, which, str(e)))
                continue
            if r is None:
                crashes.append((name, which, "cv2.imread failed"))
                continue
            rows.append({
                "name": name,
                "which": which,
                "overall": round(r["overall"], 2),
                "blur_score": round(r["blur_score"], 2),
                "exposure_score": round(r["exposure_score"], 2),
                "tilt_score": round(r["tilt_score"], 2),
                "color_cast_score": round(r["color_cast_score"], 2),
                "noise_score": round(r["noise_score"], 2),
                "flags": ";".join(r["flags"]),
                "needs_human": r["needs_human"],
            })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    by_key = {(r["name"], r["which"]): r for r in rows}
    for name in common:
        b = by_key.get((name, "before"))
        a = by_key.get((name, "after"))
        if b is not None and a is not None:
            pair_scores.append((name, b["overall"], a["overall"]))

    n_pairs = len(pair_scores)
    n_after_better = sum(1 for _, ob, oa in pair_scores if oa > ob)
    ratio = (n_after_better / n_pairs * 100.0) if n_pairs else 0.0

    print(f"Tong so anh cham: {len(rows)} (crash: {len(crashes)})")
    print(f"So cap before/after so sanh duoc: {n_pairs}")
    print(f"Ti le after > before: {n_after_better}/{n_pairs} = {ratio:.1f}%")
    print(f"Da luu CSV: {OUT_CSV}\n")

    if crashes:
        print("=== CAC ANH CRASH / LOI DOC ===")
        for name, which, err in crashes:
            print(f"  {name} ({which}): {err}")
        print()

    worst = sorted(rows, key=lambda r: r["overall"])[:5]
    print("=== TOP 5 ANH TE NHAT (overall thap nhat) ===")
    for r in worst:
        print(
            f"  [{r['which']:6s}] {r['name']:35s} overall={r['overall']:6.1f}"
            f"  flags={r['flags'] or '-'}  needs_human={r['needs_human']}"
        )

    # in chi tiet vai cap co after < before (thang do sai o dau) de tien review
    regress = sorted(
        [(name, ob, oa) for name, ob, oa in pair_scores if oa <= ob],
        key=lambda t: t[2] - t[1],
    )
    if regress:
        print(f"\n=== CAP AFTER <= BEFORE ({len(regress)} cap, in toi da 10) ===")
        for name, ob, oa in regress[:10]:
            print(f"  {name:35s} before={ob:6.1f}  after={oa:6.1f}  delta={oa-ob:+.1f}")


if __name__ == "__main__":
    main()
