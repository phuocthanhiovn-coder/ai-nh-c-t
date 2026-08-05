import os
import re
import csv
import shutil
import argparse
import rawpy
import cv2
import numpy as np
from .config import (
    PAIRS_COLOR_DIR, PAIRS_SKY_DIR, PAIRS_REMOVAL_DIR, UNMATCHED_DIR,
    OUTPUTS_SAMPLES_DIR, REPORT_CSV_PATH, INLIER_THRESHOLD, MIN_MATCH_COUNT, MIN_INLIER_RATIO,
    REVIEW_DIR, ALIGN_SCORE_THRESHOLD
)
from .align import extract_and_match
from .classify import classify_edit_type

def parse_prefix_number(filepath):
    """
    Trích xuất prefix (phần chữ trước số) và number (số cuối cùng) của file.
    Ví dụ: _ML_1682.CR3 -> prefix='_ML_', number=1682
           DSC01682.ARW -> prefix='DSC0', number=1682
           20260703-DSC04562.jpg -> prefix='20260703-DSC0', number=4562
    """
    filename = os.path.basename(filepath)
    name, _ = os.path.splitext(filename)
    match = re.search(r'^(.*?)(\d+)$', name)
    if match:
        prefix = match.group(1)
        number = int(match.group(2))
        return prefix, number
    return None, None

def get_all_files(root_dir, extensions):
    """Tìm tất cả các file có đuôi mở rộng trong extensions đệ quy."""
    file_list = []
    if not os.path.exists(root_dir):
        return file_list
    for root, _, files in os.walk(root_dir):
        for f in files:
            if any(f.lower().endswith(ext) for ext in extensions):
                file_list.append(os.path.join(root, f))
    return file_list

def resize_to_max(img, max_dim=2048):
    """Downscale ảnh giữ nguyên tỷ lệ sao cho cạnh dài nhất bằng max_dim."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    if h > w:
        new_h = max_dim
        new_w = int(w * (max_dim / h))
    else:
        new_w = max_dim
        new_h = int(h * (max_dim / w))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

# ---------------------------------------------------------------------------
# DO CAN KHOP HINH HOC — DUNG CHUNG (04/08, ra soat vong 7)
#
# Ba ham duoi day TRUOC DAY nam LONG BEN TRONG `align_bracket_images`, nen
# `align_before_after` — ham sinh TOAN BO data/pairs — khong dung lai duoc va phai
# tu cham bang Edge-NCC. Do lai tren chinh 20 cap dung 04/08: Edge-NCC KHONG do duoc
# do can khop. Dai diem cua nhom LECH >4px la 0.500-0.582; dai diem cua nhom CAN KHOP
# <2px la 0.537-0.771 — HAI DAI CHONG NHAU HOAN TOAN. Co cap lech 62px van an 0.571,
# cao hon 5/6 cap can khop duoi 1px.
# Nay nang len muc module de MOT ban cai dat phuc vu ca hai bai toan (luat L16:
# "va phai ap cho CA N nhanh, hoac hop nhat con 1").
# ---------------------------------------------------------------------------
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# Hai nguong nay TRUOC DAY nam cuc bo trong `align_bracket_images`. Sau khi nang ba
# ham do len muc module, chung phai len theo — neu khong `_do_hinh_hoc` se khong tim
# thay ten nao. (Da bi dung NameError o lan chay dau tien, sua tai cho.)
TAU_VUNG = 0.05      # response toi thieu de TIN mot vung do
MIN_VUNG = 3         # phai co it nhat 3 vung tin cay moi ket luan

def _de_so(bgr):
    """Dua mot tam bracket ve thang do sang chung DE SO SANH (khong doi anh goc).

    Keo theo phan vi 1%-99.5% (ben voi diem chay va vung den tit) roi CLAHE.
    Nho vay tam toi co du muc xam de ORB thay feature va de edge-NCC co nghia.
    """
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lo, hi = np.percentile(g, (1.0, 99.5))
    if hi - lo >= 1.0:
        g = np.clip((g - lo) * (255.0 / (hi - lo)), 0.0, 255.0)
    return clahe.apply(g.astype(np.uint8))

def _chon_vung(g, P=768):
    """Chon 5 o do: trong MOI GOC PHAN TU + tam, lay o co KET CAU cao nhat.

    Goc anh noi that thuong la mang tuong/tran PHANG (do duoc: std 7.47) —
    tuong quan pha khong bam duoc, tra ve rac 100-400px. Chon theo ket cau
    thi tam toi co 5/5 vung tin cay (23/24 tam) thay vi 2-3/5 khi lay goc
    co dinh, va bien dang do duoc cua nhom chan may giam 2.56 -> 1.48.
    """
    H, W = g.shape[:2]
    P = int(min(P, H // 2, W // 2))
    ra = []
    for gy in (0, 1):
        for gx in (0, 1):
            y0, x0 = gy * (H // 2), gx * (W // 2)
            dy, dx = max(0, H // 2 - P), max(0, W // 2 - P)
            tot, best = -1.0, (y0, x0)
            for fy in (0.05, 0.5, 0.95):
                for fx in (0.05, 0.5, 0.95):
                    y = min(max(0, y0 + int(dy * fy)), H - P)
                    x = min(max(0, x0 + int(dx * fx)), W - P)
                    s = float(g[y:y + P:8, x:x + P:8].std())   # lay mau thua cho nhanh
                    if s > tot:
                        tot, best = s, (y, x)
            ra.append(best)
    ra.append(((H - P) // 2, (W - P) // 2))                    # tam anh
    return ra, P

_HANN = {}

def _do_hinh_hoc(a_gray, b_gray, vung, P):
    """Tach DICH (tinh tien) khoi BIEN DANG (xoay/phong) — 03/08.

    VI SAO TACH (do duoc, khong doan): AlignMTB chay ngay sau cong nay SUA
    SACH tinh tien toi 70px ke ca tren tam TOI NHAT (8.26->2.06, 70.14->2.27),
    nhung KHONG sua duoc xoay/phong (0.3do: 14.69 -> 19.81, TE HON).
    => Chan tinh tien = mat tam toi OAN, doi lay zero loi ich.
    => Chi BIEN DANG moi dang chan.

    v_k = vec-to dich do tai vung k (chi lay vung co response >= TAU).
    t   = trung vi cua v_k  ->  DICH = |t|
    BIEN DANG = max |v_k - t|   (tinh tien deu thi ve 0; xoay/phong thi khong)

    Phan bo do tren 79 tam chan may that / 20 bracket:
        BIEN DANG  p50 0.83 · p90 1.87   (bo cuc nay)
        xoay 0.3do p5 6.99 · phong 1.005 p5 5.96   -> tach sach, bien 2.0x/2.6x
    (04/08: da xoa "max 1.48" — p90 1.87 > max 1.48 la BAT KHA, mot trong hai so
     do sai. Khong biet so nao sai nen bo han so khong tin duoc thay vi giu lai.)

    ⚠️ cv2.phaseCorrelate(a, b, window) GHI DE a va b TAI CHO khi canh o trung
    getOptimalDFTSize (768 va 1024 deu trung). Nen phai TU nhan cua so Hann roi
    goi KHONG truyen `window` — da kiem tuong duong tung chu so.

    Tra ve (dich_px, bien_dang_px, so_vung_tin_cay).
    """
    if P not in _HANN:
        _HANN[P] = cv2.createHanningWindow((P, P), cv2.CV_32F)
    win = _HANN[P]
    vs = []
    for (y, x) in vung:
        pa = np.ascontiguousarray(a_gray[y:y + P, x:x + P].astype(np.float32) * win)
        pb = np.ascontiguousarray(b_gray[y:y + P, x:x + P].astype(np.float32) * win)
        (dx, dy), resp = cv2.phaseCorrelate(pa, pb)
        if resp >= TAU_VUNG:
            vs.append((dx, dy, resp))
    if len(vs) < MIN_VUNG:
        return None, None, len(vs)
    n_tin = len(vs)
    # 04/08 (ra soat vong 7) — BO VUNG YEU NHAT, KHONG HA DO NHAY CUA PHEP DO.
    # Chan doan cua ban va truoc do LA DUNG: 3 tam bi loai oan deu do MOT vung duy
    # nhat, va no luon la vung co RESPONSE THAP NHAT (lech 15.7px trong khi 4 vung
    # kia nhat tri trong 0.65px). Nhung CACH CHUA thi nham: `dv[-2]` bo qua lech
    # LON NHAT bat ke no den tu dau, tuc ha do nhay cua CA phep do chu khong loai
    # rieng vung rac. Do lai tren 20 anh that 6000x4000 (moi anh tu lam tham chieu,
    # nhieu loan bang warpAffine):
    #     nhieu loan   | max (cu)      | dv[-2]        | bo-vung-yeu (NAY)
    #     xoay 0.15do  | 20/20 (1.78x) | 15/20 (1.37x) | 20/20 (1.67x)
    #     xoay 0.20do  | 20/20 (2.37x) | 18/20 (1.83x) | 20/20 (2.24x)
    #     xoay 0.30do  | 20/20 (3.55x) | 20/20 (2.76x) | 20/20 (3.10x)
    #     phong 1.003  | 20/20 (2.04x) | 18/20 (1.58x) | 20/20 (1.93x)
    #     phong 1.005  | 20/20 (3.38x) | 20/20 (2.64x) | 20/20 (3.23x)
    # (ngoac = bien an toan p50 so voi T_BIEN_DANG; script do:
    #  scratchpad/do_bien_dang.py, trich _de_so/_chon_vung bang ast tu chinh file
    #  nay nen khong the troi khoi ma that.)
    # ✅ DA KIEM TREN 53 BRACKET THAT (04/08 — 269 ARW develop full-res, ~80 phut):
    #     do duoc 171/209 tam (khong tinh tam tham chieu)
    #     cong CU  (max)          giu 170/171
    #     cong MOI (bo vung yeu)  giu 171/171
    #     tam duoc CUU them : 1  — FP104756.ARW, luma 17.3, bien dang 5.48 -> 0.39
    #     tam bi LOT them   : 0  <-- tinh chat an toan then chot, KHONG co regression
    # ⚠️ DINH CHINH: BAN_GIAO ghi cong cu loai OAN 3 tam. Do lai tren ca 53 bracket chi
    # thay 1. Con so 3 KHONG tai lap duoc — dung trich dan lai no.
    # (38/209 tam khong do duoc vi it hon MIN_VUNG vung tin cay; chung di theo duong
    #  "cuu tam bi loai" rieng o cuoi ham, khong qua phep so nay.)
    # Cong BIEN DANG nay la lop chong anh ma them 25/07 sau khi chu chi 2 anh hong
    # (227A1947 ghost / 227A2152 vet tia) — KHONG duoc lam yeu de doi lay tam toi.
    vs.sort(key=lambda r: r[2])                  # response tang dan
    if len(vs) > MIN_VUNG:
        vs = vs[1:]                              # bo dung MOT vung yeu nhat
    V = np.asarray([(dx, dy) for dx, dy, _ in vs], np.float32)
    t = np.median(V, axis=0)
    dich = float(np.hypot(t[0], t[1]))
    bien_dang = float(np.max(np.hypot(V[:, 0] - t[0], V[:, 1] - t[1])))
    return dich, bien_dang, n_tin


def align_bracket_images(images):
    """Căn khớp các frame bracket về frame THAM CHIẾU bằng SIMILARITY (scale+xoay+dịch)
    ORB+RANSAC CÓ CHẶN, frame không khớp/khớp bậy thì LOẠI (veto). Xấu nhất còn 1
    frame tham chiếu (vẫn là before hợp lệ) — pipeline KHÔNG BAO GIỜ nhả rác.

    VÌ SAO similarity thay homography (sửa 25/07/2026 — sau khi chủ chỉ 2 ảnh merge
    HỎNG: k000_227A1947 ghost chồng đôi, 227A2152 VỠ THÀNH VỆT TIA PHÓNG XẠ):
    homography 8-DOF khi khớp bậy sẽ BẮN 4 GÓC RA VÔ CỰC → warp + BORDER_REPLICATE
    kéo thành vệt tia. Similarity 4-DOF (estimateAffinePartial2D) KHÔNG có thành
    phần phối cảnh nên KHÔNG THỂ tạo vệt tia — về hình học là bất khả. Thêm cổng
    lành mạnh (scale 0.85–1.18, |xoay|≤12°, dịch≤25% cạnh) loại luôn fit suy biến,
    và siết veto NCC 0.30→0.45 (ghost lọt qua 0.30). Bản cũ dùng homography + veto
    0.30 vẫn để lọt cả 2 lỗi trên.
    """
    if len(images) <= 1:
        return images

    # Tham chiếu: frame median-luma gần 128 nhất (đủ sáng, nhiều feature).
    meds = [abs(float(np.median(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))) - 128.0)
            for im in images]
    ref_i = int(np.argmin(meds))
    ref = images[ref_i]
    h, w = ref.shape[:2]
    max_shift = 0.25 * max(h, w)

    # ⭐ FIX 03/08 — LOI DA GIET MOI TAM ANH CUA SO CUA DU AN.
    # Do duoc tren bracket that fp104505 (5 tam, sang TB 1.7 / 5.8 / 19.1 / 51.0 / 110.7):
    #   TRUOC FIX: cong veto giu 2/5 tam — chi con [110.7, 51.0].
    #   BA TAM TOI NHAT BI LOAI SACH, trong do FP104505 la tam DUY NHAT giu duoc
    #   canh ngoai cua so (vung cua so: sang TB 0.20, CHI 0.04% diem chay).
    #   Ket qua: vung cua so trong anh gop CHAY 53.77% -> trang bech, mat sach nha
    #   hang xom / cay / mai. Day dung la loi khach che tu 14/07 va ta da chua nham
    #   bang tone suot mot thang.
    # NGUYEN NHAN: ca ORB lan edge-NCC deu so hai tam LECH NHAU 6+ KHAU PHOI SANG.
    # Tam toi co luma TB 1.7/255 — gan nhu khong con muc xam nao de dung feature,
    # va NCC voi tam sang 110.7 luon gan 0. Chu thich cu o duoi tung ghi nhan dieu
    # nay ("tam RAT TOI ... it/khong feature -> veto") nhung chon cach VUT tam do
    # thay vi sua phep so sanh. Vut tam toi = vut cua so.
    # CACH SUA: keo tung tam ve CUNG THANG DO SANG truoc khi SO SANH. Chi dung cho
    # khau do-va-cham; anh tra ve van la PIXEL GOC da warp, khong he bi keo sang.

    # (04/08 da xoa ham noi bo _ncc_bracket: khong con loi goi nao sau khi cong NCC
    #  duoc thay bang cong hinh hoc. Docstring cua no con khang dinh "ban Laplacian
    #  nhan 4/4 tam" — do lai thi la 0/4, tuc chinh chu thich do cung sai.)

    # 03/08: CHAN tam khac kich thuoc. Ham nay gia dinh moi tam cung khung (bracket
    # tu mot lan bam may). Neu goi voi lo tron anh doc/ngang thi cat o theo toa do
    # tam tham chieu se ra mang RONG -> ValueError kho hieu o tan trong _do_hinh_hoc.
    # Da xay ra that khi script goi gom bracket sai (tron nhieu phong). Loai som + noi ro.
    khac_co = {k for k, im in enumerate(images) if im.shape[:2] != ref.shape[:2]}
    if khac_co:
        print(f"[bracket] LOAI {len(khac_co)} tam khac kich thuoc voi tam tham chieu "
              f"{ref.shape[1]}x{ref.shape[0]} — bracket phai cung khung. Neu con so nay "
              f"lon, rat co the khau GOM BRACKET dang sai.", flush=True)
        images = [im for k, im in enumerate(images) if k not in khac_co]
        if len(images) <= 1:
            return images
        ref_i = min(range(len(images)),
                    key=lambda k: abs(float(np.median(
                        cv2.cvtColor(images[k], cv2.COLOR_BGR2GRAY))) - 128.0))
        ref = images[ref_i]
        h, w = ref.shape[:2]
        max_shift = 0.25 * max(h, w)

    gray_ref = _de_so(ref)
    vung_do, P_do = _chon_vung(gray_ref)        # chon 1 lan cho ca bracket (~1.6s)

    # NGUONG DA HIEU CHINH BANG SO (03/08) — do tren 53 bracket that gom bang
    # groups_by_after(), 127 tam chan may + 3 nhom doi chung (dich 10/30/60px,
    # phong 1.005/1.01, xoay 0.3/0.5/1.0 do). Nghiem thu dau-cuoi 10 bracket:
    #   chan may that     -> giu 49/49 tam (100%), gom MOI tam toi luma 1.24-6.6
    #   xe 1 tam 60px     -> bat dung 10/10
    #   xoay 1 tam 0.5do  -> bat dung 10/10
    #   phong 1 tam 1.005 -> bat dung 10/10
    # (TAU_VUNG / MIN_VUNG da nang len muc module — dung chung voi
    #  align_before_after; xem khoi DO CAN KHOP HINH HOC o dau file.)
    T_BIEN_DANG = 3.0    # px — xoay/phong. Chan may that: max 1.48 | xoay 0.3do: p5 6.99
    T_DICH = 15.0        # px — tinh tien. MTB sua duoc; chan may that p99 10.34, max 11.24
    out = [ref.copy()]  # tham chiếu luôn có mặt
    # 04/08 (ra soat vong 7): GIU LAI phep do truoc MTB, song song voi `out`.
    # Vong hoan tac o cuoi ham truoc day do LAI tu dau cho moi tam, mat 6.5s/bracket
    # de ra dung con so da tinh o duoi (dong `dich, bd, nv = _do_hinh_hoc(...)`).
    # Tam tham chieu la phep TU tuong quan nen luon (0, 0) — khong can do.
    do_truoc = [(0.0, 0.0)]
    # (04/08 da xoa `dich_da_khop`: no chi duoc GHI, khong noi nao DOC. Phep suy
    #  "may co rung khong" tu cac tam khop duoc da bi bo tu 03/08 vi sai — tam bi
    #  xe la tam KHONG khop duoc nen khong de lai bang chung gi. Nay do truc tiep.)
    da_nhan = {ref_i}   # 03/08: theo doi theo CHI SO (so sanh `is` sai vi anh da bi warp)
    for i, im in enumerate(images):
        if i == ref_i:
            continue
        gray = _de_so(im)          # 03/08: keo ve cung thang sang truoc khi do feature
        # 03/08: tham so CUC BO rong tay hon config chung (2000/15/5.0) — do duoc
        # tren bracket that fp104505: bo chat khop 1/4 tam (cua so chay 21.03%),
        # bo rong khop 4/4 tam (chay 14.47%). Tam KHOP DUOC thi duoc NAN dung cho;
        # tam chi "cuu" thi giu nguyen vi tri cu, nen khop duoc luon tot hon.
        # KHONG sua config.py vi MIN_MATCH_COUNT/INLIER_THRESHOLD con dung cho
        # khau ghep cap before/after — doi o do se gay hoi quy cho ca dataset.
        BR_FEATURES, BR_MIN_MATCH, BR_RANSAC, BR_TOP = 4000, 12, 4.0, 500
        res = extract_and_match(gray, gray_ref,
                                max_features=BR_FEATURES, min_match=BR_MIN_MATCH)
        if res is None:
            continue
        kp1, kp2, matches = res
        # tam RAT TOI (no_auto_bright -> luma ~2) it/khong feature -> matches None.
        # KHONG con la an tu hinh: tam bi loai o day van co the duoc cuu o duoi
        # neu chan may dung yen (xem khoi FIX 03/08 cuoi ham).
        if matches is None or len(matches) < BR_MIN_MATCH:
            continue
        matches = sorted(matches, key=lambda m: m.distance)[:BR_TOP]
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        # SIMILARITY 4-DOF (scale+xoay+dịch) — KHÔNG phối cảnh nên không vệt tia.
        M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC,
                                              ransacReprojThreshold=BR_RANSAC)
        if M is None or mask is None or int(mask.sum()) < BR_MIN_MATCH:
            continue
        # Cổng lành mạnh: chặn fit suy biến (scale/xoay/dịch phi lý).
        scale = float(np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2))
        angle = abs(float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))))
        shift = float(np.hypot(M[0, 2], M[1, 2]))
        if not (0.85 <= scale <= 1.18) or angle > 12.0 or shift > max_shift:
            continue
        # LANCZOS4: warp bilinear làm MỀM ảnh mỗi lần resample; Lanczos giữ nét.
        warped = cv2.warpAffine(im, M, (w, h), flags=cv2.INTER_LANCZOS4,
                                borderMode=cv2.BORDER_REPLICATE)
        # Kiểm chứng sau warp bằng NCC cạnh — siết 0.45 chống ghost.
        # 03/08: PHAI cham tren ban DA KEO VE CUNG THANG SANG. Ban cu cham thang
        # tren pixel goc, nen tam toi (luma TB 1.7) luon cho NCC ~0 va bi veto —
        # do la co che da vut mat tam giu cua so o MOI bracket cua du an.
        # NCC canh chi "bat bien tone" khi hai anh cung thang do sang; lech 6 khau
        # thi khong con bat bien nua.
        # 03/08: cong HINH HOC thay cong NCC. Tam da nan van phai chung minh no
        # THAT SU nam dung cho — do 5 vung, khong tin rieng ket qua fit.
        dich, bd, nv = _do_hinh_hoc(_de_so(warped), gray_ref, vung_do, P_do)
        if bd is None or bd > T_BIEN_DANG or dich > T_DICH:
            continue
        out.append(warped)
        do_truoc.append((dich, bd))
        da_nhan.add(i)

    # ⭐ CUU TAM BI LOAI KHI CHAN MAY DUNG YEN (fix 03/08).
    # Van de: tam bracket toi nhat co noi that DEN TUYET DOI va chi cua so sang;
    # tam tham chieu thi nguoc lai — noi that ro, cua so chay. Hai tam gan nhu
    # KHONG CO CAU TRUC CHUNG nao de ORB khop, nen tam toi luon bi veto du no
    # nam dung vi tri. Do tren bracket that fp104505: giu 2/5 tam, mat ca 3 tam
    # toi — trong do co tam DUY NHAT giu duoc canh ngoai cua so.
    #
    # Nhung veto van CAN THIET de chong anh ma (them 25/07 sau khi chu chi 2 anh
    # hong). Nen KHONG duoc bo — phai phan biet "tam khac phoi sang" voi "may rung".
    # Cach phan biet: neu cac tam DA KHOP deu dich rat it thi may dung yen (anh
    # BDS gan nhu luon chup chan may) -> tam bi loai cung dang o dung cho, giu lai
    # va de AlignMTB (bat bien do sang theo thiet ke) don not vai pixel con lai.
    # Neu co tam dich nhieu -> may co rung -> giu nguyen hanh vi veto cu.
    #
    # Do duoc sau fix, vung cua so anh fp104505:
    #   giu 2/5 tam -> chay 53.77%, tuong phan 0.191
    #   giu 5/5 tam -> chay 25.31%, tuong phan 0.224   (AutoHDR: 0.28%, 0.234)
    # ⚠️ SUA 03/08 (lan 2) — ban dau toi suy ra "may dung yen" TU CAC TAM KHOP DUOC.
    # Sai: tam bi xe la tam KHONG KHOP DUOC, nen no khong de lai bang chung gi va
    # bi cuu OAN. Kiem hoi quy bat duoc: xe 1 tam 60px van giu 5/5 => pha luon
    # chuc nang chong anh ma. Nay DO TRUC TIEP tung tam duoc cuu bang tuong quan
    # pha tren ban da keo ve cung thang sang — bat bien do sang, do duoc dich that.
    bi_loai = [(k, images[k]) for k in range(len(images)) if k not in da_nhan]
    cuu = 0
    for k, im in bi_loai:
        dich, bd, nv = _do_hinh_hoc(_de_so(im), gray_ref, vung_do, P_do)
        if bd is None:
            print(f"[bracket] tam #{k} chi co {nv}/{MIN_VUNG} vung tin cay -> LOAI "
                  f"(khong du bang chung de ket luan)", flush=True)
        elif bd > T_BIEN_DANG or dich > T_DICH:
            print(f"[bracket] tam #{k} khong khop duoc VA bien dang {bd:.2f}px / dich "
                  f"{dich:.2f}px -> LOAI (chong anh ma)", flush=True)
        else:
            out.append(im.copy())
            do_truoc.append((dich, bd))
            cuu += 1
    if cuu:
        # 04/08: bo ve "AlignMTB sua duoc". Cau do KHONG dung duoc nua — chinh khoi MTB
        # ben duoi co vong hoan tac, va tren tam luma <= 6 no kich hoat THAT (do duoc:
        # 3/14 bracket). Noi "MTB sua duoc" roi ngay sau do hoan tac MTB la tu mau
        # thuan; nhat ky phai noi dung thu da lam.
        print(f"[bracket] giu {cuu}/{len(bi_loai)} tam khong khop duoc: bien dang duoi "
              f"{T_BIEN_DANG}px va dich duoi {T_DICH}px -> chung nam DUNG CHO trong "
              f"sai so do duoc", flush=True)

    # MTB cuối chỉ dọn dịch nhỏ còn sót giữa các frame ĐÃ khớp similarity.
    if len(out) >= 2:
        # 🔴 FIX 04/08 (ra soat vong 7) — SUA NGUYEN NHAN, khong chi chan trieu chung.
        #
        # Trieu chung do duoc: MTB day vang chinh tam TOI vua duoc cuu.
        #   fp104630 tam luma 2.5 : TRUOC 0.15px -> SAU 59.45px
        # Quet 14/53 bracket: 3 bracket bi. Anh gop fp104505 lech TB 4.91/255,
        # 11.78% diem lech >8 muc, 2.66% >30 muc, max 234.
        #
        # NGUYEN NHAN: cv2.AlignMTB::process neo vao PHAN TU GIUA danh sach
        # (out[len//2]). `out` xep [ref, cac tam ORB khop duoc..., cac tam duoc CUU...]
        # ma tam duoc cuu chinh la cac tam TOI NHAT va bi noi vao CUOI. Nen khi so tam
        # duoc cuu >= mot nua (vi du chi 1 tam khop duoc ORB: out = [ref, m, r, r, r]
        # -> neo = out[2] = tam toi nhat), MTB lay ban do nguong-trung-vi cua mot tam
        # gan den = NHIEU, roi tinh dich RAC cho moi tam khac.
        #
        # CACH SUA: dat chinh TAM THAM CHIEU vao vi tri neo len//2 truoc khi goi MTB.
        # Tam tham chieu la tam co luma trung vi gan 128 nhat (chon o dau ham) nen ban
        # do nguong-trung-vi cua no co nghia, VA no dung la tam ma moi phep do hinh hoc
        # ben duoi lay lam goc — hai dieu can cho MTB chay dung.
        # Vong hoan tac ben duoi VAN GIU lam luoi an toan lop hai (MTB khong dang tin
        # tren tam luma <= 6 ke ca khi neo dung; doi chung: MTB sua duoc tam toi toi
        # 60px, that bai o 70px — dung tran 2^max_bits = 64).
        #
        # ⚠️ Chu thich cu o day dan "fp104505 out[3]: 1.94 -> 88.70px" lam vi du. Con
        # so do do TRUOC khi doi thong ke bien dang (`max` -> `dv[-2]` -> bo-vung-yeu),
        # ma phep doi do lam thay ca thanh phan LAN thu tu cua `out`, nen vi du do
        # khong con tai lap duoc voi ma hien tai. Vi du con tai lap duoc: fp104630.
        _piv = len(out) // 2
        thu_tu = list(range(1, len(out)))
        thu_tu.insert(_piv, 0)               # tam tham chieu roi dung vao vi tri neo
        sap = [out[k] for k in thu_tu]
        truoc_mtb = [im.copy() for im in sap]
        try:
            # `try` chi bao DUNG loi goi MTB. Ban cu bao ca vong hoan tac ben duoi, nen
            # mot loi cv2 giua vong se nuot bang `pass` va de lai `out` o trang thai
            # TRON (nua truoc MTB, nua sau) ma khong in mot dong nao — dung khuon mau
            # "loi im lang" ma luat L12 goi la nguyen nhan giet du an.
            cv2.createAlignMTB().process(sap, sap)
        except cv2.error as e:
            print(f"[bracket] AlignMTB that bai ({e}) -> giu nguyen pixel truoc MTB",
                  flush=True)
            sap = truoc_mtb
        hoan = 0
        for pos, k in enumerate(thu_tu):
            if k == 0:
                continue                     # tam tham chieu: tu tuong quan, khong the te hon
            d0, b0 = do_truoc[k]             # da do o tren, khong do lai (tiet kiem 6.5s)
            d1, b1, _ = _do_hinh_hoc(_de_so(sap[pos]), gray_ref, vung_do, P_do)
            # ngach 1.0px de khong lat qua lat lai vi nhieu do
            if d1 is None or d1 > d0 + 1.0 or b1 > b0 + 1.0:
                sap[pos] = truoc_mtb[pos]
                hoan += 1
        if hoan:
            print(f"[bracket] AlignMTB lam TE HON {hoan}/{len(sap)} tam -> tra ve "
                  f"pixel TRUOC MTB (chong anh ma)", flush=True)
        for pos, k in enumerate(thu_tu):     # tra lai thu tu cu cua `out`
            out[k] = sap[pos]
    return out

def process_cr3_to_rgb(cr3_path, no_auto_bright=False):
    """Develop RAW (CR3/DNG/ARW) FULL-SIZE (24/07: bỏ half_size — đây là 1 trong 2
    thủ phạm làm ảnh 'before' nát nét, đo được lap 12 vs AutoHDR 146). half_size
    hủy MỘT NỬA độ phân giải ngay từ develop — không cứu lại được ở bước sau.
    Thêm khử nhiễu nhẹ + sharpen chuẩn của rawpy để nét như máy ảnh xuất.

    26/07 GỐC BỆNH MERGE: tấm THÀNH VIÊN BRACKET phải develop với
    no_auto_bright=True — auto-bright SAN BẰNG phơi sáng mọi tấm (~165 luma)
    → vứt sạch dynamic range (cửa sổ/đèn cháy) TRƯỚC khi Mertens gộp. Ảnh ĐƠN
    (không bracket) giữ auto-bright như cũ."""
    with rawpy.imread(cr3_path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            half_size=False,                 # FULL RES (truoc: True = nua cỡ)
            output_bps=8,
            no_auto_bright=no_auto_bright,
            fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Light,
        )
        return rgb

def merge_exposures(images):
    """Gộp các ảnh phơi sáng (bracket) bằng Mertens Exposure Fusion.

    1 ảnh (các frame khác bị veto ở align_bracket_images) -> trả thẳng ảnh đó."""
    if len(images) == 1:
        return images[0].copy()
    merge_mertens = cv2.createMergeMertens()
    fusion = merge_mertens.process(images)
    # 04/08 (ra soat vong 7) — HAI LOI TRONG MOT DONG, ca hai deu im lang.
    #
    # (1) LAM TRON, khong CAT. `astype(np.uint8)` cat ve 0 (truncate), nen moi diem
    #     anh bi ha trung binh NUA MUC (-0.5/255) mot cach he thong. Dong nay chay
    #     cho MOI anh before cua MOI dataset va MOI anh vao cua BENCH-10, tuc lech
    #     do di vao toan bo du lieu hoc va toan bo bang diem.
    #
    # (2) BAO CAO ti le bi cat. cv2.MergeMertens tra ve gia tri VUOT KHOI [0,1] (no
    #     tai dung anh bang kim tu thap Laplacian nen co rung/overshoot o canh tuong
    #     phan cao). Ban cu cat thang, khong nen mem, khong noi mot loi — nen neu
    #     khau gop lam hong dai sang cua so thi khong ai thay. Nay do va IN ra khi
    #     vuot nguong dang chu y.
    # ⚠️ KHONG duoc "chua" bang cach chia cho max: da thu, luma tut 76.8 -> 41.1 vi
    #    mot vai diem rung keo ca anh xuong. Cat van la dung; cai thieu la BAO CAO.
    tren = float(np.mean(fusion > 1.0)) * 100.0
    duoi = float(np.mean(fusion < 0.0)) * 100.0
    if tren > 0.5 or duoi > 0.5:
        print(f"[merge] Mertens tra ra ngoai [0,1]: {tren:.2f}% diem >1.0, "
              f"{duoi:.2f}% diem <0.0 (max {float(fusion.max()):.3f}, "
              f"min {float(fusion.min()):.3f}) -> bi CAT PHANG. Neu con so nay lon "
              f"o anh co cua so thi dai sang cua so dang bi mat o day.", flush=True)
    fusion_8bit = np.clip(np.rint(fusion * 255.0), 0, 255).astype(np.uint8)
    return fusion_8bit

def save_sample_visual(before_merged, aligned_before, after, output_path, edit_type):
    """Lưu ảnh mẫu panorama [before-merge | after | diff] ở kích thước nhỏ (max width ~1500px)."""
    target_w = 500
    target_h = int(500 * after.shape[0] / after.shape[1])
    
    before_resized = cv2.resize(before_merged, (target_w, target_h), interpolation=cv2.INTER_AREA)
    aligned_resized = cv2.resize(aligned_before, (target_w, target_h), interpolation=cv2.INTER_AREA)
    after_resized = cv2.resize(after, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    diff = cv2.absdiff(after_resized, aligned_resized)
    diff_enhanced = cv2.multiply(diff, 3)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(before_resized, "Before Merged", (20, 40), font, 0.8, (0, 0, 255), 2)
    cv2.putText(after_resized, f"After ({edit_type.upper()})", (20, 40), font, 0.8, (255, 0, 0), 2)
    cv2.putText(diff_enhanced, "Diff Map x3", (20, 40), font, 0.8, (0, 255, 255), 2)
    
    canvas = np.hstack((before_resized, after_resized, diff_enhanced))
    cv2.imwrite(output_path, canvas)

def compute_edge_gradient(gray_img):
    """Tính toán bản đồ cạnh Sobel gradient để bất biến với sự thay đổi tông màu."""
    sobelx = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(sobelx, sobely)
    grad_norm = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return grad_norm

def compute_edge_ncc(aligned_gray, after_gray):
    """Tính toán Normalized Cross-Correlation (NCC) trên bản đồ cạnh Sobel sử dụng TM_CCOEFF_NORMED."""
    grad_aligned = compute_edge_gradient(aligned_gray)
    grad_after = compute_edge_gradient(after_gray)
    
    res = cv2.matchTemplate(grad_aligned, grad_after, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return float(max_val)

def align_before_after(before_img, after_img):
    """
    Align ảnh before_img khớp khít với after_img sử dụng RANSAC Homography + ECC 512px.
    """
    gray_before = cv2.cvtColor(before_img, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(after_img, cv2.COLOR_BGR2GRAY)
    h, w = after_img.shape[:2]
    
    res = extract_and_match(gray_before, gray_after)
    # FIX 04/08: extract_and_match tra ve TUPLE (None, None, None) khi that bai,
    # KHONG BAO GIO tra bare None -> phep kiem `res is None` cu la MA CHET, roi
    # `len(matches)` nem TypeError. Ham nay sinh TOAN BO data/pairs/before, va
    # `run_ingest` bat Exception rong nen cap anh bi VUT AM THAM thay vi ve REVIEW
    # voi nhan align_low. Dung loai anh toi/it feature ma ca ban va hom nay nham toi.
    # (Chu thich trong align.py tung ghi "ban va tai GOC" — thuc te moi va
    #  verify_and_align, bo sot chinh ham nay.)
    if res is None or res[2] is None:
        return 0.0, before_img.copy()

    kp1, kp2, matches = res
    if len(matches) < MIN_MATCH_COUNT:
        return 0.0, before_img.copy()
        
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, INLIER_THRESHOLD)
    if H is None or mask is None:
        return 0.0, before_img.copy()

    # 04/08 (ra soat vong 7) — CONG LANH MANH cho homography. Vi pham L16 dien hinh:
    # align_bracket_images:262-267 DA co cong nay tu 25/07 (sau khi chu chi 2 anh hong
    # 227A1947 ghost / 227A2152 vet tia), nhung ham NAY — ham sinh TOAN BO
    # data/pairs/before — thi chua bao gio co. Ban va dung nhung chi ap cho mot nhanh.
    #
    # Co che hong: fit suy bien ban 4 goc ra rat xa; warpPerspective +
    # BORDER_REPLICATE keo pixel bien thanh VET TIA / DAI MEP GIA. Do 04/08: 45% anh
    # before cua bo 20 cap moi co dai mep gia.
    #
    # Kiem bang ANH XA 4 GOC chu khong doc rieng khoi 2x2 goc trai tren: voi
    # homography 8-DOF thi chinh thanh phan phoi canh H[2,0]/H[2,1] moi la thu sinh
    # vet tia, ma no khong the hien trong khoi 2x2.
    _goc = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    try:
        _goc_moi = cv2.perspectiveTransform(_goc, H).reshape(-1, 2)
    except cv2.error:
        return 0.0, before_img.copy()
    _g0 = _goc.reshape(-1, 2)
    if not np.all(np.isfinite(_goc_moi)):
        print("[pair] homography ban goc ra VO CUC -> VETO (chong vet tia)", flush=True)
        return 0.0, before_img.copy()
    _dich_goc = float(np.max(np.hypot(_goc_moi[:, 0] - _g0[:, 0],
                                      _goc_moi[:, 1] - _g0[:, 1])))
    if _dich_goc > 0.25 * max(h, w):
        print(f"[pair] goc anh dich {_dich_goc:.0f}px > 25% canh -> VETO "
              f"(fit suy bien, se keo dai mep gia)", flush=True)
        return 0.0, before_img.copy()
    # Cung 3 nguong da hieu chinh cua nhanh bracket: scale 0.85-1.18, |xoay| <= 12do.
    _scale = float(np.sqrt(H[0, 0] ** 2 + H[0, 1] ** 2))
    _angle = abs(float(np.degrees(np.arctan2(H[1, 0], H[0, 0]))))
    if not (0.85 <= _scale <= 1.18) or _angle > 12.0:
        print(f"[pair] homography phi ly (scale {_scale:.3f}, xoay {_angle:.1f}do) "
              f"-> VETO", flush=True)
        return 0.0, before_img.copy()

    aligned_before_rough = cv2.warpPerspective(before_img, H, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # ECC Refine trên 512px
    aligned_gray_rough = cv2.cvtColor(aligned_before_rough, cv2.COLOR_BGR2GRAY)
    grad_before = compute_edge_gradient(aligned_gray_rough)
    grad_after = compute_edge_gradient(gray_after)
    
    target_dim = 512
    h_g, w_g = grad_after.shape[:2]
    scale_factor = target_dim / max(h_g, w_g)
    
    w_small = int(w_g * scale_factor)
    h_small = int(h_g * scale_factor)
    
    grad_before_small = cv2.resize(grad_before, (w_small, h_small), interpolation=cv2.INTER_AREA)
    grad_after_small = cv2.resize(grad_after, (w_small, h_small), interpolation=cv2.INTER_AREA)
    
    warp_matrix = np.eye(3, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, 1e-3)
    
    aligned_before_final = aligned_before_rough
    
    try:
        _, warp_matrix = cv2.findTransformECC(grad_after_small, grad_before_small, warp_matrix, cv2.MOTION_HOMOGRAPHY, criteria, None, 1)
        s = 1.0 / scale_factor
        warp_matrix_large = warp_matrix.copy()
        warp_matrix_large[0, 2] = warp_matrix[0, 2] * s
        warp_matrix_large[1, 2] = warp_matrix[1, 2] * s
        warp_matrix_large[2, 0] = warp_matrix[2, 0] / s
        warp_matrix_large[2, 1] = warp_matrix[2, 1] / s
        
        # 04/08: cong lanh manh LAN HAI. ECC tinh chinh tren anh 512px roi phong
        # nguoc len full-res, nen mot nghiem ECC troi nhe o 512 thanh dich lon o
        # 6000px. Neu no ban goc di qua 5% canh thi giu ban warp THO (da qua cong
        # o tren) thay vi tin ket qua tinh chinh.
        _gm = cv2.perspectiveTransform(_goc, np.linalg.inv(warp_matrix_large)) \
            .reshape(-1, 2) if abs(np.linalg.det(warp_matrix_large)) > 1e-9 else None
        if _gm is None or not np.all(np.isfinite(_gm)) or float(np.max(np.hypot(
                _gm[:, 0] - _g0[:, 0], _gm[:, 1] - _g0[:, 1]))) > 0.05 * max(h, w):
            raise cv2.error("ECC homography ban goc di qua xa -> giu ban warp tho")

        aligned_before_final = cv2.warpPerspective(
            aligned_before_rough, warp_matrix_large, (w, h),
            flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE)
    except cv2.error:
        # Fallback Affine
        warp_matrix_affine = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp_matrix_affine = cv2.findTransformECC(grad_after_small, grad_before_small, warp_matrix_affine, cv2.MOTION_AFFINE, criteria, None, 1)
            s = 1.0 / scale_factor
            warp_matrix_affine_large = warp_matrix_affine.copy()
            warp_matrix_affine_large[0, 2] = warp_matrix_affine[0, 2] * s
            warp_matrix_affine_large[1, 2] = warp_matrix_affine[1, 2] * s
            
            aligned_before_final = cv2.warpAffine(
                aligned_before_rough, warp_matrix_affine_large, (w, h),
                flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE)
        except cv2.error:
            pass
            
    # ⭐ 04/08 (ra soat vong 7) — THAY THUOC, khong chi chinh nguong.
    #
    # Ban cu: `align_score = compute_edge_ncc(final_gray, gray_after)`. HAI LOI CHONG
    # NHAU, deu la loi CHAN:
    #
    # (1) Edge-NCC KHONG DO DUOC do can khop. Do tren chinh 20 cap dung 04/08: dai
    #     diem cua nhom LECH >4px@2048 la 0.500-0.582, dai diem cua nhom CAN KHOP
    #     <2px la 0.537-0.771 — HAI DAI CHONG NHAU HOAN TOAN. Co cap lech 62px van an
    #     0.571, cao hon 5/6 cap can khop duoi 1px. Tuc cong cu xep cap HONG len tren
    #     cap TOT. Model duoc day mot anh xa BAT KHA o nhung vung lech hang chuc pixel;
    #     L1 giai bai toan bat kha bang cach hoi quy ve trung binh = anh bot/phang —
    #     dung co che tasks/29 muc D3 mo ta, va dung trieu chung chu che suot 1 thang.
    #
    # (2) Diem Edge-NCC CO GIAN THEO DO PHAN GIAI. Cung cap anh: 0.560 @full-res /
    #     0.629 @3000 / 0.677 @2048 / 0.755 @1024 / 0.793 @512. Hang so
    #     ALIGN_SCORE_THRESHOLD=0.50 duoc hieu chinh khi MOI duong deu cham o 2048,
    #     nhung build_real_pairs_fullres.py cham o 6000x4000 — o do 0.50 khat khe
    #     tuong duong 0.64 @2048. Do la ly do 33/53 cap bi loai, 17 trong so do nam
    #     sat duoi nguong (0.44-0.499).
    #
    # ⚠️ HA NGUONG LA SAI (da can nhac va bac bo): vi (1), ha nguong chi nhan them cap
    # LECH NANG. Phai doi THUOC, khong phai doi vach.
    #
    # THUOC MOI: dung dung cap ham da hieu chinh cho bracket (_chon_vung +
    # _do_hinh_hoc) — tach DICH khoi BIEN DANG, do 5 vung chon theo ket cau, loc theo
    # response. Do o CO CO DINH 2048 nen khong con co gian theo do phan giai.
    # Anh xa ve thang diem cu de moi ben goi va chinh ALIGN_SCORE_THRESHOLD=0.50 van
    # dung nguyen: r = lech/nguong; r<=1 -> diem = 1 - 0.5r (nam trong [0.5, 1.0]);
    # r>1 -> diem = 0.5/r (nam trong (0, 0.5)). Ham don dieu giam, lien tuc tai r=1,
    # va moc 0.50 gio DUNG BANG ranh gioi dat/truot — mot con so co nghia that.
    _T_CAP = 3.0        # px @2048. Bracket dung 3.0 cho bien dang; cap before/after
                        # KHAT KHE HON o DICH vi khong co AlignMTB nao chay sau de don.
    _s = 2048.0 / max(aligned_before_final.shape[:2])
    if _s < 1.0:
        _a = cv2.resize(aligned_before_final, None, fx=_s, fy=_s,
                        interpolation=cv2.INTER_AREA)
        _b = cv2.resize(after_img, None, fx=_s, fy=_s, interpolation=cv2.INTER_AREA)
    else:
        _a, _b = aligned_before_final, after_img
    _ga, _gb = _de_so(_a), _de_so(_b)
    _vung, _P = _chon_vung(_gb)
    _dich, _bd, _nv = _do_hinh_hoc(_ga, _gb, _vung, _P)
    if _bd is None:
        # Khong du bang chung -> KHONG duoc coi la dat. Noi ro thay vi tra diem cao.
        print(f"[pair] chi co {_nv}/{MIN_VUNG} vung tin cay -> khong ket luan duoc "
              f"do can khop, xep vao REVIEW", flush=True)
        return 0.0, aligned_before_final
    _r = max(_dich, _bd) / _T_CAP
    align_score = (1.0 - 0.5 * _r) if _r <= 1.0 else (0.5 / _r)

    return float(align_score), aligned_before_final

def is_pair_processed(output_name):
    """Kiểm tra xem cặp ảnh này đã được xử lý và lưu ở bất kỳ thư mục đầu ra nào chưa."""
    for d in [PAIRS_COLOR_DIR, PAIRS_SKY_DIR, PAIRS_REMOVAL_DIR, REVIEW_DIR]:
        if os.path.exists(os.path.join(d, "before", output_name)):
            return True
    return False

def load_existing_report():
    """Nạp báo cáo cũ report.csv dưới dạng dict để cập nhật không trùng lặp."""
    existing_data = {}
    if not os.path.exists(REPORT_CSV_PATH):
        return existing_data
    try:
        with open(REPORT_CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                has_job_name = "job_name" in header
                for row in reader:
                    if len(row) < 7:
                        continue
                    if has_job_name:
                        job, prefix, num = row[0], row[1], int(row[2])
                        existing_data[(job, prefix, num)] = row
                    else:
                        # Bản cũ không có job_name
                        prefix = row[0]
                        try:
                            num = int(row[1])
                            # Thêm job_name mặc định là rỗng
                            existing_data[("", prefix, num)] = [""] + row
                        except ValueError:
                            continue
    except Exception as e:
        print(f"[!] CẢNH BÁO khi nạp report cũ: {str(e)}")
    return existing_data

def run_ingest(reset=False, before_root="data/raw/before", after_root="data/raw/after", job_name=None):
    """Chạy toàn bộ pipeline ingest raw brackets theo prefix và number."""
    print("=" * 60)
    print("  AutoHDR Dataset Ingestion - Prefix-Based Pairing")
    if job_name:
        print(f"  [Job Name]: {job_name}")
    print("=" * 60)
    
    if reset:
        print("[*] Tùy chọn --reset được bật. Đang dọn dẹp dữ liệu cũ...")
        for d in [PAIRS_COLOR_DIR, PAIRS_SKY_DIR, PAIRS_REMOVAL_DIR, UNMATCHED_DIR, OUTPUTS_SAMPLES_DIR, REVIEW_DIR]:
            if os.path.exists(d):
                shutil.rmtree(d)
        if os.path.exists(REPORT_CSV_PATH):
            os.remove(REPORT_CSV_PATH)
            
    for d in [PAIRS_COLOR_DIR, PAIRS_SKY_DIR, PAIRS_REMOVAL_DIR, UNMATCHED_DIR, OUTPUTS_SAMPLES_DIR, REVIEW_DIR]:
        if d != OUTPUTS_SAMPLES_DIR:
            os.makedirs(os.path.join(d, "before"), exist_ok=True)
            os.makedirs(os.path.join(d, "after"), exist_ok=True)
        else:
            os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.dirname(REPORT_CSV_PATH), exist_ok=True)
    
    report_dict = {} if reset else load_existing_report()
    
    print(f"[*] Đang quét các file thô tại: {before_root} và {after_root}...")
    # Đọc đệ quy .cr3, .dng, .arw
    before_paths = get_all_files(before_root, [".cr3", ".dng", ".arw"])
    after_paths = get_all_files(after_root, [".jpg", ".jpeg"])
    
    print(f"[+] Tìm thấy {len(before_paths)} file RAW (.CR3/.DNG/.ARW)")
    print(f"[+] Tìm thấy {len(after_paths)} file AFTER (.jpg/.jpeg)")
    
    # Indexing theo prefix
    before_index = {}
    for path in before_paths:
        prefix, num = parse_prefix_number(path)
        if prefix is not None and num is not None:
            before_index.setdefault(prefix, []).append((num, path))
            
    after_index = {}
    for path in after_paths:
        prefix, num = parse_prefix_number(path)
        if prefix is not None and num is not None:
            after_index.setdefault(prefix, []).append((num, path))
            
    # Sắp xếp số thứ tự tăng dần
    for k in before_index:
        before_index[k].sort(key=lambda x: x[0])
    for k in after_index:
        after_index[k].sort(key=lambda x: x[0])
        
    print(f"[+] Indexing thành công: {len(before_index)} prefix máy ở BEFORE, {len(after_index)} prefix ở AFTER.")
    
    matched_pairs_count = 0
    skipped_pairs_count = 0
    unmatched_after = []
    
    job_key = job_name if job_name else ""
    
    # 3. Gom và xử lý từng nhóm prefix tương ứng
    for prefix, after_list in after_index.items():
        if prefix not in before_index:
            print(f"[!] CẢNH BÁO: Prefix '{prefix}' có trong AFTER nhưng không có tương ứng ở BEFORE.")
            for num, path in after_list:
                unmatched_after.append((prefix, num, path))
            continue
            
        before_list = before_index[prefix]
        print(f"\n[*] Đang xử lý nhóm prefix: {prefix} ({len(after_list)} ảnh sau, {len(before_list)} ảnh thô)")
        
        prefix_clean = re.sub(r'[^a-zA-Z0-9_-]', '_', prefix)
        
        for idx, (num, after_path) in enumerate(after_list):
            start_num = num
            end_num = after_list[idx + 1][0] if idx + 1 < len(after_list) else float('inf')
            
            if job_name:
                output_name = f"{job_name}_{prefix_clean}{num}.jpg"
            else:
                output_name = f"{prefix_clean}{num}.jpg"
            
            if not reset and is_pair_processed(output_name):
                skipped_pairs_count += 1
                matched_pairs_count += 1
                continue
                
            bracket_files = [path for b_num, path in before_list if start_num <= b_num < end_num]
            bracket_size = len(bracket_files)
            
            if bracket_size == 0:
                print(f"  [✗] Không tìm thấy file RAW nào cho after {num} (prefix={prefix}).")
                unmatched_after.append((prefix, num, after_path))
                continue
                
            print(f"  [-] Cảnh after {num}: tìm thấy bộ bracket gồm {bracket_size} ảnh RAW.")
            
            try:
                raw_images = []
                for cr3_path in bracket_files:
                    # 26/07: bracket member -> GIU phoi sang that (fix goc benh merge)
                    rgb = process_cr3_to_rgb(cr3_path, no_auto_bright=(bracket_size >= 2))
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    raw_images.append(bgr)
                    
                aligned_raws = align_bracket_images(raw_images)
                before_merged = merge_exposures(aligned_raws)
                
                after_img = cv2.imread(after_path)
                if after_img is None:
                    print(f"  [✗] Lỗi không đọc được ảnh after: {after_path}")
                    continue
                    
                # 25/07: giữ 2048 cho cặp TRAIN (model xem proxy 384; full-res chỉ
                # tốn disk 4× mà KHÔNG nét hơn — đo được: mờ đến từ GỘP BRACKET
                # (warp+Mertens), không phải độ phân giải. Nét thật = detail_restore
                # ở khâu GIAO, không phải ở ingest). half_size=False vẫn giữ vì
                # develop full rồi downscale 2048 sắc hơn develop nửa cỡ.
                before_merged_resized = resize_to_max(before_merged, 2048)
                after_img_resized = resize_to_max(after_img, 2048)

                align_score, aligned_before = align_before_after(before_merged_resized, after_img_resized)
                edit_type, confidence = classify_edit_type(aligned_before, after_img_resized)
                
                if align_score >= ALIGN_SCORE_THRESHOLD:
                    align_status = "align_ok"
                    if edit_type == "sky":
                        dest_dir = PAIRS_SKY_DIR
                    elif edit_type == "removal":
                        dest_dir = PAIRS_REMOVAL_DIR
                    else:
                        dest_dir = PAIRS_COLOR_DIR
                else:
                    align_status = "align_low"
                    dest_dir = REVIEW_DIR
                    
                print(f"    [✓] Căn chỉnh: {align_status} (align_score: {align_score:.4f}) | Phân loại: {edit_type} (conf: {confidence:.2f})")
                
                cv2.imwrite(os.path.join(dest_dir, "before", output_name), aligned_before, [cv2.IMWRITE_JPEG_QUALITY, 95,
                     cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
                     cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])
                cv2.imwrite(os.path.join(dest_dir, "after", output_name), after_img_resized, [cv2.IMWRITE_JPEG_QUALITY, 95,
                     cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
                     cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])
                
                if job_name:
                    sample_path = os.path.join(OUTPUTS_SAMPLES_DIR, f"sample_{job_name}_{prefix_clean}{num}.jpg")
                else:
                    sample_path = os.path.join(OUTPUTS_SAMPLES_DIR, f"sample_{prefix_clean}{num}.jpg")
                save_sample_visual(before_merged, aligned_before, after_img, sample_path, edit_type)
                
                report_dict[(job_key, prefix, num)] = [
                    job_key, prefix, num, bracket_size, edit_type, f"{confidence:.4f}", align_status, f"{align_score:.4f}"
                ]
                matched_pairs_count += 1
                
            except Exception as e:
                print(f"  [✗] LỖI khi xử lý cảnh {num} (prefix={prefix}): {str(e)}")
                
    print("\n[*] Xử lý ảnh không khớp...")
    for prefix, num, path in unmatched_after:
        prefix_clean = re.sub(r'[^a-zA-Z0-9_-]', '_', prefix)
        if job_name:
            dest_name = f"{job_name}_{prefix_clean}{num}.jpg"
        else:
            dest_name = f"{prefix_clean}{num}.jpg"
        dest_path = os.path.join(UNMATCHED_DIR, "after", dest_name)
        shutil.copy(path, dest_path)
        
    with open(REPORT_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["job_name", "prefix", "number", "bracket_size", "edit_type", "confidence", "align_status", "align_score"])
        sorted_keys = sorted(list(report_dict.keys()), key=lambda x: (x[0], x[1], x[2]))
        for key in sorted_keys:
            writer.writerow(report_dict[key])
            
    total_rows = list(report_dict.values())
    train_count = sum(1 for r in total_rows if r[6] == "align_ok")
    review_count = sum(1 for r in total_rows if r[6] == "align_low")
    
    print(f"\n[+] INGEST HOÀN THÀNH!")
    print(f"  - Tổng số cặp thành công tích lũy: {len(total_rows)}")
    print(f"  - Số cặp đủ điều kiện TRAIN (align_ok): {train_count}")
    print(f"  - Số cặp cần REVIEW (align_low): {review_count}")
    print(f"  - Đã xử lý mới trong lượt này: {matched_pairs_count - skipped_pairs_count} cặp (Bỏ qua đã có: {skipped_pairs_count} cặp).")
    print(f"  - Số ảnh sau unmatched: {len(unmatched_after)}")
    print(f"  - Báo cáo lưu tại: {REPORT_CSV_PATH}")
    return len(total_rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoHDR Ingest Engine - Prefix-Based Pairing")
    parser.add_argument("--reset", action="store_true", help="Xóa sạch dữ liệu cũ và ingest lại từ đầu")
    parser.add_argument("--before-root", type=str, default="data/raw/before", help="Thư mục chứa ảnh raw before")
    parser.add_argument("--after-root", type=str, default="data/raw/after", help="Thư mục chứa ảnh after")
    parser.add_argument("--job-name", type=str, default=None, help="Tên của job tải về")
    args = parser.parse_args()
    run_ingest(reset=args.reset, before_root=args.before_root, after_root=args.after_root, job_name=args.job_name)
