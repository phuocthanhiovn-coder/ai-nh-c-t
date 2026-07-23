"""Tim anh ngoai that mat dat nang gat trong data/pairs (troi sang chay + bong do)."""
import glob, os, cv2, numpy as np

cands = []
for p in glob.glob("data/pairs/before/*.jpg"):
    img = cv2.imread(p)
    if img is None:
        continue
    h, w = img.shape[:2]
    lum = 0.114*img[:,:,0] + 0.587*img[:,:,1] + 0.299*img[:,:,2]
    lum = lum/255.0
    # ngoai that nang gat = phan TREN anh rat sang (troi chay) + co vung toi (bong)
    top = lum[:h//3]
    blown = float((top > 0.90).mean())     # troi chay o phan tren
    dark = float((lum < 0.15).mean())       # bong do
    dyn = float(np.percentile(lum, 99) - np.percentile(lum, 2))
    # diem "nang gat ngoai that": troi tren chay + bong + dai rong
    score = blown*2 + dark + dyn
    # loai drone (drone = sang deu, it bong sau) - drone01 prefix
    is_drone = os.path.basename(p).startswith("drone")
    cands.append((score, blown, dark, dyn, is_drone, os.path.basename(p)))

cands.sort(reverse=True)
print("=== TOP ung vien ngoai that nang gat (khong drone) ===")
n = 0
for s, bl, dk, dy, dr, name in cands:
    if dr:
        continue
    print(f"  {name}: score={s:.2f} blown_top={bl:.2f} dark={dk:.2f} dyn={dy:.2f}")
    n += 1
    if n >= 8:
        break
print(f"\nTong: {len(cands)} anh | drone: {sum(1 for c in cands if c[4])}")
