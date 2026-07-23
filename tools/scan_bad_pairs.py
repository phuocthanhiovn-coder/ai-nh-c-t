"""Quet 940 cap tim BEFORE mo duc bat thuong (chup xuyen kinh / phan chieu / suong).
Dau hieu: den cua before bi nang cao (p5 luma cao — haze nang den) + tuong phan
thap hon after nhieu. Chi FLAG + xuat contact sheet de architect NHIN roi moi
cach ly — khong tu xoa.
"""
import os

import cv2
import numpy as np

cv2.setNumThreads(3)

PAIRS = "data/pairs"
OUT = "outputs/bad_pair_scan"


def main():
    os.makedirs(OUT, exist_ok=True)
    names = sorted(os.listdir(os.path.join(PAIRS, "before")))
    flagged = []
    for i, n in enumerate(names):
        b = cv2.imread(os.path.join(PAIRS, "before", n), cv2.IMREAD_GRAYSCALE)
        a = cv2.imread(os.path.join(PAIRS, "after", n), cv2.IMREAD_GRAYSCALE)
        if b is None or a is None:
            continue
        p5b = np.percentile(b, 5)
        stdb, stda = b.std(), a.std()
        ratio = stdb / max(stda, 1e-3)
        # haze: den bi nang (p5 cao) + tuong phan sut manh so voi after
        if p5b > 55 and ratio < 0.62:
            flagged.append((n, p5b, ratio))
        if (i + 1) % 200 == 0:
            print(f"{i+1}/{len(names)}", flush=True)

    flagged.sort(key=lambda x: x[2])
    print(f"\nFLAG {len(flagged)} cap nghi hong:")
    for n, p5b, r in flagged[:30]:
        print(f"  {n:<40} p5_before={p5b:5.1f} contrast_ratio={r:.3f}")

    # contact sheet 12 cap nghi nhat de nhin
    rows = []
    for n, _, _ in flagged[:12]:
        b = cv2.imread(os.path.join(PAIRS, "before", n))
        a = cv2.imread(os.path.join(PAIRS, "after", n))
        h = 260
        pb = cv2.resize(b, (int(b.shape[1]*h/b.shape[0]), h))
        pa = cv2.resize(a, (int(a.shape[1]*h/a.shape[0]), h))
        row = np.hstack([pb, pa])
        row = np.ascontiguousarray(row)
        cv2.rectangle(row, (0, 0), (row.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(row, n, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        rows.append(row)
    if rows:
        w = max(r.shape[1] for r in rows)
        sheet = np.vstack([cv2.copyMakeBorder(r, 0, 4, 0, w - r.shape[1],
                                              cv2.BORDER_CONSTANT) for r in rows])
        cv2.imwrite(os.path.join(OUT, "flagged_sheet.jpg"), sheet,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    with open(os.path.join(OUT, "flagged.txt"), "w", encoding="utf-8") as f:
        for n, p5b, r in flagged:
            f.write(f"{n}\t{p5b:.1f}\t{r:.3f}\n")
    print("sheet + danh sach ->", OUT)


if __name__ == "__main__":
    main()
