"""
Cham mot bo anh da gan-nhan-bang-mat (LOOK truoc, hardcode nhan sau) bang
classify.classify(). In predicted scene + confidence + do chinh xac.

Nhan trong LABELS duoi day duoc gan sau khi mo tung anh bang Read tool va
nhin that (xem bao cao cuoi file task 23-scene-classify.md de biet chi tiet
tung anh). Chi DOC data/, ghi outputs/scene_samples/report.csv.
"""

import csv
import os
import sys

import cv2
import numpy as np

cv2.setNumThreads(2)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import classify  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(ROOT, "outputs", "scene_samples")
OUT_CSV = os.path.join(OUT_DIR, "report.csv")

PAIRS_BEFORE = os.path.join(ROOT, "data", "pairs", "before")
MIXED_PROBE = os.path.join(ROOT, "data", "newbatch", "mixed_probe", "01-RAW-Photos")

# (duong dan tuong doi tu ROOT, nhan mat-nhin-thay) — xem bao cao trong task file.
LABELS = [
    # --- noi that (interior) ---
    (os.path.join(MIXED_PROBE, "DSC01341.JPG"), "interior"),           # hanh lang toi, cua, guong
    (os.path.join(MIXED_PROBE, "DSC01420.JPG"), "interior"),           # phong toi, cua so thay cay/toa nha
    (os.path.join(PAIRS_BEFORE, "20260703-DSC1105.jpg"), "interior"),  # phong khach, cua so thay skyline NYC
    (os.path.join(PAIRS_BEFORE, "_ML_1622.jpg"), "interior"),          # phong ngu/tu do, khong troi
    (os.path.join(PAIRS_BEFORE, "after_pool2_gd09_783A9524.jpg"), "interior"),  # bep/khach, cua kinh lon thay skyline
    (os.path.join(PAIRS_BEFORE, "drone01_DSC01408.jpg"), "interior"),  # hanh lang tu ke (ten file "drone01" nhung noi dung noi that)
    (os.path.join(PAIRS_BEFORE, "db01__ML_1633.jpg"), "interior"),
    (os.path.join(PAIRS_BEFORE, "20260703-DSC1132.jpg"), "interior"),
    (os.path.join(PAIRS_BEFORE, "20260703-DSC1161.jpg"), "interior"),
    # --- ngoai troi mat dat (exterior_ground) ---
    (os.path.join(MIXED_PROBE, "DSC01500.JPG"), "exterior_ground"),    # ban cong nhin ra duong/cay/troi
    (os.path.join(MIXED_PROBE, "DSC01580.JPG"), "exterior_ground"),    # cay hoa sat nha, macro ngoai troi
    (os.path.join(MIXED_PROBE, "DSC01604.JPG"), "exterior_ground"),    # mat tien cua chinh + bui cay
    (os.path.join(PAIRS_BEFORE, "drone01_DSC01602.jpg"), "exterior_ground"),  # mat tien cua chinh (frame gan DSC01604)
    (os.path.join(MIXED_PROBE, "DJI_20260714050834_0940_D.JPG"), "exterior_ground"),  # nha rieng le, may bay chup ngang tam mat
    # --- aerial (nadir / oblique cao) ---
    (os.path.join(MIXED_PROBE, "DJI_20260714050700_0921_D.JPG"), "aerial"),  # nhin thang xuong nha/duong, KHONG troi
    (os.path.join(MIXED_PROBE, "DJI_20260714050744_0930_D.JPG"), "aerial"),  # oblique cao, toan canh khu dan cu/cang
    (os.path.join(MIXED_PROBE, "DJI_20260714050931_0950_D.JPG"), "aerial"),  # oblique cao, canh toan thanh pho/song/nui
]


def load_float01(path):
    img_u8 = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_u8 is None:
        return None
    return img_u8.astype(np.float32) / 255.0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = []
    n_correct = 0
    n_total = 0
    misclassified = []
    crashes = []

    for path, label in LABELS:
        rel = os.path.relpath(path, ROOT)
        if not os.path.isfile(path):
            crashes.append((rel, "khong tim thay file"))
            continue
        img = load_float01(path)
        if img is None:
            crashes.append((rel, "cv2.imread that bai"))
            continue

        try:
            result = classify.classify(img)
        except Exception as e:
            crashes.append((rel, f"classify() raise: {e}"))
            continue

        pred = result["scene"]
        conf = result["confidence"]
        correct = (pred == label)
        n_total += 1
        if correct:
            n_correct += 1
        else:
            misclassified.append((rel, label, pred, conf))

        sig = result["signals"]
        rows.append({
            "path": rel,
            "label": label,
            "pred": pred,
            "confidence": conf,
            "correct": correct,
            "sky_fraction": round(sig.get("sky_fraction", -1), 4),
            "sky_touch_top": round(sig.get("sky_touch_top", -1), 4),
            "top_edge_density": round(sig.get("top_edge_density", -1), 4),
            "bottom_edge_density": round(sig.get("bottom_edge_density", -1), 4),
            "horizon_present": sig.get("horizon_present"),
            "vline_count": sig.get("vline_count"),
            "scores": sig.get("scores"),
        })
        print(f"{rel:60s} label={label:16s} pred={pred:16s} conf={conf:.3f}  {'OK' if correct else 'MISS'}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    acc = (n_correct / n_total * 100.0) if n_total else 0.0
    print(f"\nDo chinh xac: {n_correct}/{n_total} = {acc:.1f}%")
    print(f"Da luu CSV: {OUT_CSV}")

    if misclassified:
        print("\n=== SAI NHAN ===")
        for rel, label, pred, conf in misclassified:
            print(f"  {rel:60s} that={label:16s} du_doan={pred:16s} conf={conf:.3f}")

    if crashes:
        print("\n=== LOI / KHONG DOC DUOC ===")
        for rel, err in crashes:
            print(f"  {rel}: {err}")


if __name__ == "__main__":
    main()
