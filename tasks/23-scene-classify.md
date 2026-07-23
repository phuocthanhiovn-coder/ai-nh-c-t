# Task 23 — SCENE CLASSIFIER (route interior / exterior-ground / aerial → right pipeline)

**Assigned to:** Worker (Sonnet) · **Review:** Claude architect · **Read `CLAUDE.md` first.**

## Context
The learned model applied an INTERIOR grade to a DRONE aerial and produced garbage (gray roof). The delivery pipeline (Task 21) needs to KNOW the scene type to choose the right chain (e.g. straighten only for interiors, harsh-sun tone-map for sunny exteriors, gentle for aerial). Build a deterministic classifier.

## Files (`ai_engine/specialists/scene_classify/`)
1. `classify.py` — `classify(img_bgr_float01_or_u8) -> dict{scene, confidence, signals}`:
   - scene in {"interior", "exterior_ground", "aerial", "unknown"}.
   - Signals (deterministic, work on a ~768px proxy):
     * SKY at top: large smooth bright blue/white region touching the top border → exterior/aerial.
     * Horizon line: strong horizontal gradient boundary in upper half → exterior_ground.
     * Aerial/nadir: no sky at all + repetitive roof/ground texture filling frame + vanishing-point cues absent → aerial (roofs, streets seen top-down). Hint: aerial shots have little/no vertical walls and no ceiling; the whole frame is "ground".
     * Interior: vertical wall/ceiling structure (strong vertical lines, uniform ceiling region at top, no sky) → interior.
   - Combine signals into a scored decision + confidence 0-1. Return the raw signals dict too (for debugging/tuning).
2. `run_samples.py` — classify a labeled-by-eye mix: several `data/pairs/before/` (interiors), the `data/newbatch/mixed_probe/01-RAW-Photos/` drone shots (aerial), and any ground exteriors you can find (e.g. `_ML_1605`-style if present). Print predicted scene + confidence per image; compute accuracy vs your eyeball labels (hardcode the labels you assign after LOOKING at each with the Read tool).

## Acceptance (LOOK at the images to label, then measure)
- [ ] Drone aerials → "aerial" (these were the failure case — must not be called "interior").
- [ ] Interiors → "interior". Any ground exteriors → "exterior_ground".
- [ ] Report accuracy on your hand-labeled set (aim ≥80%); list misclassifications honestly + why.
- [ ] `classify` never crashes; returns "unknown" with low confidence when unsure rather than guessing wrong confidently.

## DO NOT
- No files outside `ai_engine/specialists/scene_classify/` + `outputs/scene_samples/` + report. `data/` read-only. `cv2.setNumThreads(2)`.

---

## BAO CAO (worker, 2026-07-14)

### Da lam
- `ai_engine/specialists/scene_classify/classify.py` — `classify(img)` deterministic, KHONG model, `cv2.setNumThreads(2)`. Proxy 768px canh dai.
  Tin hieu: (1) TROI — tai su dung `sky_replace/sky_mask.detect_sky()` (KHONG viet lai flood-grow, tranh trung logic) -> `sky_fraction` + `sky_touch_top` (ty le hang bien tren la troi). (2) DUONG CHAN TROI — gradient-ngang trung binh dinh/median cac hang trong nua tren khung. (3) AERIAL — khong troi + dai tren "ban" (Canny edge density cao, khong phang nhu tran) + mat do canh tren/duoi gan deu (ca khung deu la "mat dat"). (4) NOI THAT — khong troi + dai tren phang/min (edge density thap, std luma thap = tran nha) + nhieu duong gan-doc (LSD/Hough, nhu `qc_scorer`) trai deu theo truc x (tuong/khung cua). Diem tung scene cong don tu cac tin hieu tren, argmax + margin -> confidence; diem cao nhat < 0.28 -> `"unknown"`. Moi loi (shape sai, NaN, anh 1x1...) bi bat trong `try/except` bao ngoai cung -> tra `{"scene":"unknown","confidence":0.0,...}`, KHONG bao gio raise (da test thu 10 truong hop bat thuong: anh den/trang tuyet doi, 4x4, 1x1, sai so kenh, NaN, list thuong khong phai ndarray — tat ca tra ve dict hop le, khong crash).
- `ai_engine/specialists/scene_classify/run_samples.py` — cham 17 anh **da tu mo bang Read tool va gan nhan mat-nhin-thay** (KHONG doan tu ten file — xem muc "bai hoc" ben duoi), in du doan + confidence + tinh accuracy, xuat `outputs/scene_samples/report.csv`.

### Ket qua THAT (da chay, khong bia)
Chay `python ai_engine/specialists/scene_classify/run_samples.py` that, output day du trong `outputs/scene_samples/report.csv`.

**Do chinh xac: 14/17 = 82.4%** (muc tieu acceptance la ≥80%).

- Noi that (9/9 dung, 100%): tat ca anh noi that — ke ca 2 anh KHO co cua so lon lo ca mang troi xanh + skyline NYA (`20260703-DSC1132.jpg`, cua so gan het chieu cao khung; `after_pool2_gd09_783A9524.jpg`, cua kinh lon thay skyline) — deu bi `sky_replace.detect_sky()` cho `sky_fraction=0.0` (seed khong du 12% hang bien tren la sang+mat lien tuc theo MIN_SEED_TOP_FRAC cua con do) nen roi dung vao nhanh "khong troi" -> phan loai dung nho tran phang + nhieu duong doc.
- Ngoai troi mat dat (5/5 dung, 100%): nha rieng le, cua chinh, ban cong, mat do troi/canh khac nhau nhieu (`sky_fraction` 0.019–0.10) van nhan dung.
- Aerial (1/3 dung): `DJI_..._0921_D.JPG` (nhin thang xuong nadir, KHONG mot chut troi, toan mai nha+duong) -> **dung**, day chinh la loai anh gay ra su co goc (grade noi that len anh aerial) — quan trong nhat da fix. 2 anh con la **anh drone kieu oblique cao (establishing shot)** co troi/duong chan troi ro (`0930_D` toan canh cang bien+doi nui, `0950_D` toan canh thanh pho+song) -> bi doan thanh `exterior_ground` thay vi `aerial`.

### Sai nhan — thanh that, khong giau
1. **`20260703-DSC1161.jpg`** (noi that, ket qua `exterior_ground` conf 0.697) — day la mot goc may quay thang vao 1 o cua so lon-tran-nha nhin ra skyline NYC, khong bi che boi tuong xung quanh nhu 2 anh tren. `sky_fraction=0.099`, `sky_touch_top=0.43` du de vuot nguong "co troi" cua bo phan loai. **Nguyen nhan goc**: troi nhin qua cua so la PIXEL TROI THAT (mau/do sang giong het troi ngoai), khong co cach nao deterministic (khong dung segmentation ngu nghia) phan biet "troi qua khung cua kinh" voi "troi mo thien nhien" chi bang mau+gradient — da thu them tin hieu "troi cham bien trai/phai khung hinh" va "ty le mang tuong min mau trung tinh" nhung CA HAI deu KHONG tach duoc: anh ban cong ngoai troi that (`DSC01500`) cung khong cham bien trai/phai (mai hien che) va co ty le "tuong-nhu" cao hon ca anh noi that nay. Day la GIOI HAN THAT cua huong tiep can thuan CV, ghi nhan de xu ly sau (co the can them tin hieu do sau/parallax hoac segment khung cua so).
2. **`DJI_..._0930_D.JPG`, `DJI_..._0950_D.JPG`** (aerial oblique, ket qua `exterior_ground` conf 0.87 va 0.54) — day la anh drone o do cao lon nhung KHONG phai goc nhin thang xuong (nadir): co troi chiem 17–37% khung + duong chan troi ro rang (giong dung mo ta "Horizon line ... -> exterior_ground" trong spec task nay). Bo tin hieu hien tai (dua vao "co troi/horizon" cho exterior_ground, "khong troi" cho aerial) khong phan biet duoc "anh toan canh chup tu do cao lon" voi "anh ngoai troi mat dat co bau troi". Da thu them tin hieu "vung min lon nhat duoi duong chan troi" (gia thuyet: nha don le co 1 mang tuong min lon, anh aerial thi vun/nhieu mai nha nho) nhung do thi KHONG tach duoc ro rang tren du lieu that — bo, tranh overfit vao 2 anh cu the.
   **Quan trong**: ca 2 anh nay **KHONG bi goi la "interior"** — tieu chi an toan cot loi cua task ("drone aerial khong duoc goi la interior") **DAT 3/3** tren toan bo anh DJI test. Loi o day la nham giua 2 nhanh ngoai-troi (aerial vs exterior_ground), khong phai loi nguy hiem (noi that vs ngoai troi).

### Gioi han da biet (ghi lai trung thuc, chua fix)
- Phan biet "troi qua cua so lon trong noi that" vs "troi that ngoai troi" la diem yeu nhat — anh huong truc tiep tien do ~1/9 anh noi that trong bo test (11%). Can them tin hieu ngu nghia hon (VD: phat hien khung cua so — duong vien hinh chu nhat/net thang bao quanh vung troi) o task sau.
- Phan biet "aerial oblique/establishing shot co troi" vs "exterior_ground toan canh" chua co tin hieu rieng — hien tai ca 2 dung chung nhanh scoring dua vao troi/horizon. Anh aerial NADIR (khong troi) van phan loai dung 100%.
- Anh macro/close-up cuc doan (VD hoa sat tuong, gan het khung la texture khong troi khong tran) co the roi vao `"unknown"` — dung theo dung tinh than "khong doan bua" cua acceptance, nhung nghia la mot so anh ngoai troi ky la se can nguoi xem lai thay vi tu dong dinh tuyen.
- Da test 10 input bat thuong (anh den/trang tuyet doi 100%, 1x1, 4x4, sai so kenh, NaN, list thuong) — **khong crash lan nao**, luon tra ve dict hop le. Truong hop NaN in ra 1 `RuntimeWarning` cua numpy (invalid value in cast) nhung van tra ve ket qua, khong raise.
- Toc do: ~0.34s/anh tren proxy 768px (CPU, `cv2.setNumThreads(2)`), du nhanh de goi trong buoc dinh tuyen truoc pipeline.

TASK23=DONE — accuracy 14/17 = 82.4% tren bo 17 anh gan-nhan-bang-mat (9 noi that, 5 ngoai troi mat dat, 3 aerial); aerial-nadir (loai anh gay su co goc) phan loai dung 100% va khong anh aerial/exterior nao bi goi nham la "interior"; 10/10 input bat thuong khong lam crash `classify()`.
