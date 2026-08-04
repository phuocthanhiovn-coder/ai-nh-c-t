"""KIEM TRA TRUOC KHI TRAIN — dung mot lan cho ca du an (02/08/2026).

VI SAO CO FILE NAY
------------------
18 doi model, 1 thang, khong tien bo. Ra soat 02/08 tim ra 4 tang hong DOC LAP,
va CA 4 deu KHONG THE phat hien bang cach nhin anh — chung chi lo ra khi doc ma:

  1. `w_chroma`/`w_sharp`/`w_clip` khong duoc truyen vao loss -> 3 ham phat them
     vao de chua "nhat mau" va "mo" CHUA BAO GIO CHAY.
  2. `cache_cap=60` bi hieu la so anh -> model hoc tren anh 45x45 pixel.
  3. BENCH-10 trung 8/10 canh voi tap train -> "cham mu" thuc chat cham tri nho.
  4. val chia bang bam TEN FILE, 53 cap deu MOT can nha -> val la bai thi mo sach.

Ca 4 deu la loi IM LANG: khong exception, khong canh bao, so do van dep.
File nay bien chung thanh LOI DUNG MAY. Nguyen tac ky su: thi nghiem khong tai
lap duoc thi khong phai thi nghiem, va tien GPU khong duoc tieu tren ha tang hong.

CACH DUNG (bat buoc, dat NGAY TRUOC vong train):

    from .preflight import chay_het, ghi_ho_so
    chay_het(criterion=criterion, mau_pred=pred, mau_tgt=tgt,
             train_files=tr, val_files=va, data_dir=data_dir,
             bench_dir="data/bench10/input", crop_thuc_te=crop_hieu_dung)
    ghi_ho_so(ckpt_path, cfg)

Moi ham tra ve list loi (rong = dat). `chay_het` gom lai va `raise` neu co loi.
Muon bo qua mot muc (chi khi da hieu ro): truyen ten muc vao `bo_qua=[...]`,
viec bo qua se duoc IN RA de sau nay con truy.
"""
import json
import os
import re
import subprocess
import sys

_NGAY = "02/08/2026"
# Anh giao cho khach: canh dai 6024 (job that). Doi so nay khi doi may anh.
DO_PHAN_GIAI_GIAO_HANG = 6024


# ---------------------------------------------------------------------------
def _ma_job(ten_file):
    """Suy ma JOB (can nha / buoi chup) tu ten file cap anh.

    Quy uoc ten trong du an: 'hr_fp104505.jpg', 'j055_0123.jpg', 'k002_17.jpg'.
    Job = phan chu dung truoc cum so cuoi:
        hr_fp104505 -> 'hr_fp'   (ca 53 cap real deu CUNG MOT job — dung nhu that)
        j055_0123   -> 'j055_'
    """
    goc = os.path.splitext(os.path.basename(ten_file))[0]
    m = re.match(r"^(.*?)(\d+)$", goc)
    return (m.group(1) if m else goc).lower()


def _canh_dai(duong_dan):
    import cv2
    im = cv2.imread(duong_dan)
    return 0 if im is None else max(im.shape[:2])


# ---------------------------------------------------------------------------
def kiem_loss(criterion, mau_pred, mau_tgt):
    """MOI thanh phan loss phai co gradient KHAC 0 tren mot batch that.

    Day la kiem tra da bat duoc tham hoa 8 ngay trong 60 giay neu co tu dau:
    term duoc cau hinh nhung khong truyen vao -> gia tri 0.000, gradient 0.
    """
    import torch
    loi = []
    p = mau_pred.detach().clone().requires_grad_(True)
    tong, terms = criterion(p, mau_tgt.detach())
    if not torch.isfinite(tong):
        return ["loss tong khong huu han (NaN/Inf) ngay tu batch dau"]

    for ten, gt in sorted(terms.items()):
        if ten == "total":
            continue
        v = float(gt.detach()) if hasattr(gt, "detach") else float(gt)
        if abs(v) < 1e-12:
            loi.append(f"loss term '{ten}' co gia tri 0.000 -> term CHET, "
                       f"khong day model gi ca (dung loi 25/07-01/08)")

    tong.backward()
    if p.grad is None or float(p.grad.abs().sum()) == 0.0:
        loi.append("toan bo loss khong sinh gradient — model se khong hoc gi")

    bat = [k for k in ("w_l1", "w_char", "w_lab", "w_perc", "w_hi", "w_dark",
                       "w_color", "w_lc", "w_chroma", "w_sharp", "w_clip")
           if getattr(criterion, k, 0.0)]
    chay = {k for k in terms if k != "total"}
    thieu = [k for k in bat if k.replace("w_", "") not in chay]
    if thieu:
        loi.append(f"trong so {thieu} khac 0 nhung KHONG co term tuong ung trong "
                   f"ket qua loss -> bi nuot im lang")
    return loi


def kiem_data(train_files, val_files, bench_dir=None):
    """val khong rong · train/val KHAC JOB · bench KHONG giao train."""
    loi = []
    if not train_files:
        loi.append("tap train RONG")
    if not val_files:
        loi.append("tap val RONG -> val_loss se la 0.0 va 'best checkpoint' se la "
                   "trong so gan ngau nhien o epoch 1 (loi train_sweep.py:323)")
        return loi

    job_tr = {_ma_job(f) for f in train_files}
    job_va = {_ma_job(f) for f in val_files}
    chung = sorted(job_tr & job_va)
    if chung:
        loi.append(
            f"train va val DUNG CHUNG job {chung[:5]} ({len(chung)} job): cung can nha, "
            f"cung buoi chup, cung phien chinh cua AutoHDR -> val la BAI THI MO SACH, "
            f"moi val_loss deu lac quan gia. Phai chia theo JOB, khong theo ten file.")
    if len(job_tr) < 2:
        loi.append(
            f"toan bo tap train chi co {len(job_tr)} job ({sorted(job_tr)}): "
            f"KHONG the tach val co y nghia tu mot can nha duy nhat. Can data cua "
            f"it nhat 2 can nha khac nhau truoc khi bat ky val_loss nao dang tin.")

    if bench_dir and os.path.isdir(bench_dir):
        khoa = lambda f: os.path.splitext(os.path.basename(f))[0].lower()
        b = {khoa(f) for f in os.listdir(bench_dir)
             if f.lower().endswith((".jpg", ".png"))}
        t = {khoa(f) for f in train_files}
        giao = sorted(b & t)
        if giao:
            loi.append(
                f"BENCH co {len(giao)}/{len(b)} canh NAM TRONG tap train {giao[:5]}: "
                f"cham mu tren anh model da hoc thuoc = cham TRI NHO, khong phai "
                f"chat luong (loi phat hien 02/08).")
    return loi


def kiem_do_phan_giai(data_dir, crop_thuc_te, mau=8):
    """Anh model THUC SU nhin thay phai cung thang voi anh giao khach."""
    loi = []
    bd = os.path.join(data_dir, "before")
    if not os.path.isdir(bd):
        return [f"khong thay thu muc {bd}"]
    fs = sorted(f for f in os.listdir(bd) if f.lower().endswith((".jpg", ".png")))[:mau]
    if not fs:
        return [f"{bd} khong co anh nao"]

    canh = [_canh_dai(os.path.join(bd, f)) for f in fs]
    tb = sum(canh) / len(canh)
    ty_le = DO_PHAN_GIAI_GIAO_HANG / max(tb, 1.0)
    if ty_le > 1.5:
        loi.append(
            f"anh train canh dai ~{tb:.0f}px nhung anh giao khach {DO_PHAN_GIAI_GIAO_HANG}px "
            f"= {ty_le:.1f} lan/canh ({ty_le ** 2:.0f} lan dien tich). Day mot dang thi mot neo "
            f"— KHONG ham phat nao sua duoc (loi phat hien 02/08).")
    if crop_thuc_te and crop_thuc_te > tb:
        loi.append(
            f"crop={crop_thuc_te}px LON HON canh dai anh ({tb:.0f}px) -> anh se bi PHONG "
            f"NGUOC len roi moi cat: model hoc tren anh mo (dung loi cache_cap 30/07).")
    return loi


def kiem_thuoc_do(src, target):
    """Thuoc do phai duoc tu kiem O DUNG CO ANH se dung de ket luan."""
    from .metrics import tu_kiem
    try:
        ok = tu_kiem(src, target, verbose=False)
    except Exception as e:
        return [f"thuoc do nem loi khi tu kiem: {type(e).__name__}: {e}"]
    if not ok:
        return [f"thuoc do KHONG dat tu kiem o co anh {max(src.shape[:2])}px. "
                f"Luu y: cac ca thu phu thuoc kich thuoc (ca 'lam mo' dung blur 21px) "
                f"va phu thuoc cap anh (ca 'den trang' sai khi anh vao am mau nang). "
                f"Phai hieu ro TAI SAO truoc khi bo qua."]
    return []


# ---------------------------------------------------------------------------
def ghi_ho_so(ckpt_path, cfg, them=None):
    """Ghi <ckpt>.meta.json — checkpoint khong co ho so coi nhu khong ton tai.

    Hien 10/11 checkpoint cu khong con ho so nao, nen `audit_lich_su.py` phai
    DOAN kien truc tu hinh dang tensor va 3 doi dau bi bo qua im lang.
    """
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        commit = "?"
    ho_so = {"ngay": _NGAY, "git_commit": commit,
             "python": sys.version.split()[0], "cfg": cfg}
    if them:
        ho_so.update(them)
    p = str(ckpt_path) + ".meta.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(ho_so, f, ensure_ascii=False, indent=1, default=str)
    print(f"[preflight] da ghi ho so: {p}", flush=True)
    return p


def chay_het(criterion=None, mau_pred=None, mau_tgt=None,
             train_files=None, val_files=None, data_dir=None, bench_dir=None,
             crop_thuc_te=None, src=None, target=None, bo_qua=()):
    """Chay het cac muc kiem tra. Co loi -> RAISE, khong train.

    `bo_qua`: ten muc duoc mien ("loss"/"data"/"do_phan_giai"/"thuoc_do").
    Chi dung khi DA HIEU RO ly do; viec bo qua duoc in ra de sau con truy.
    """
    muc = {}
    if criterion is not None and mau_pred is not None:
        muc["loss"] = kiem_loss(criterion, mau_pred, mau_tgt)
    if train_files is not None:
        muc["data"] = kiem_data(train_files, val_files or [], bench_dir)
    if data_dir:
        muc["do_phan_giai"] = kiem_do_phan_giai(data_dir, crop_thuc_te)
    if src is not None and target is not None:
        muc["thuoc_do"] = kiem_thuoc_do(src, target)

    print("\n" + "=" * 66, flush=True)
    print("KIEM TRA TRUOC KHI TRAIN", flush=True)
    print("=" * 66, flush=True)
    tat_ca = []
    for ten, ls in muc.items():
        if ten in bo_qua:
            print(f"  [BO QUA] {ten} ({len(ls)} canh bao) — nguoi chay tu quyet", flush=True)
            continue
        if ls:
            for l in ls:
                print(f"  [HONG] {ten}: {l}", flush=True)
            tat_ca += [f"{ten}: {l}" for l in ls]
        else:
            print(f"  [DAT ] {ten}", flush=True)
    print("=" * 66 + "\n", flush=True)

    if tat_ca:
        raise RuntimeError(
            "KHONG DUOC TRAIN — " + str(len(tat_ca)) + " muc hong:\n  - "
            + "\n  - ".join(tat_ca)
            + "\n\nDay khong phai canh bao. Ca 4 tang hong tim ra 02/08 deu IM LANG "
              "va deu khien mot lan thue GPU thanh vo nghia. Sua roi hay chay lai.")
    return True
