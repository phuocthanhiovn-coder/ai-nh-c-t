"""Calib lai nguong QC scorer tren TOAN BO cap sach hien co (940 cap, 22/07).

Chay qc_scorer.score() tren ca before (anh xau biet truoc) va after (anh dep
chuan AutoHDR) -> phan bo diem 2 nhom -> de xuat nguong tach (percentile).
Ket qua ghi outputs/qc_calib_940.csv + in bang tom tat. Chi doc anh, khong sua
gi — architect xem bang + vai anh bien roi mo mat quyet dinh nguong.
"""
import csv
import os

import cv2
import numpy as np

cv2.setNumThreads(3)

from ai_engine.specialists.qc_scorer.qc import score as qc_score

PAIRS = "data/pairs"
OUT_CSV = "outputs/qc_calib_940.csv"


def main():
    names = sorted(os.listdir(os.path.join(PAIRS, "before")))
    rows = []
    for i, n in enumerate(names):
        for side in ("before", "after"):
            p = os.path.join(PAIRS, side, n)
            img = cv2.imread(p)
            if img is None:
                continue
            r = qc_score(img.astype(np.float32) / 255.0)
            rows.append({"name": n, "side": side, **{k: round(float(v), 4)
                        for k, v in r.items() if isinstance(v, (int, float, np.floating))}})
        if (i + 1) % 100 == 0:
            print(f"{i+1}/{len(names)}", flush=True)

    if not rows:
        print("khong co anh"); return
    keys = sorted({k for r in rows for k in r})
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)

    import statistics as st
    for metric in [k for k in keys if k not in ("name", "side")]:
        b = [r[metric] for r in rows if r["side"] == "before" and metric in r]
        a = [r[metric] for r in rows if r["side"] == "after" and metric in r]
        if not b or not a:
            continue
        print(f"{metric:<22} before p10/p50/p90: "
              f"{np.percentile(b,10):7.2f}/{np.percentile(b,50):7.2f}/{np.percentile(b,90):7.2f}"
              f"   after: {np.percentile(a,10):7.2f}/{np.percentile(a,50):7.2f}/{np.percentile(a,90):7.2f}")
    print("DONE ->", OUT_CSV, flush=True)


if __name__ == "__main__":
    main()
