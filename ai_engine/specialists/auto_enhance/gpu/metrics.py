"""THUOC DO CHAT LUONG — 3 truc, co TU KIEM (01/08/2026).

Vi sao co file nay: thuoc cu (sai lech trung binh tung diem anh) da lua chinh
toi hai lan. Dem thu bang 6 truong hop biet truoc dap an thi no truot 3:

    bien anh thanh DEN TRANG    -> thuoc cu cho -0.9%  (gan nhu KHONG phat mat mau)
    lam MO anh                  -> thuoc cu cho +1.9%  (THUONG cho viec lam mo!)
    phep cong tru tho so 6 dong -> 36.2%, trong khi model 29 TRIEU tham so 41.7%

Do la dung loi "L1 mu truoc mat mau" da tung phat hien trong du an nay roi MAC
LAI khi dung thuoc moi. Chu cham 0/10 trong khi thuoc bao 41.7% — nguoi dung la chu.

Cac file eval_visual.py / eval_box.py cham bang L1 + dE GOP CHUNG nen dinh dung
lo hong do: anh den trang giu nguyen do sang, gop lai thi mat sach mau chi bi
tru ~9 diem thay vi 21. Dung file nay cho moi quyet dinh dua tren so.

Thuoc nay tach 3 truc de khong the bu tru cho nhau, va di kem `tu_kiem()` chay
lai dung nhung truong hop do. QUY TAC: doi cong thuc thi phai chay `tu_kiem()`
va no phai qua HET, truoc khi tin bat ky con so nao.

    sang     100 = bang AutoHDR, 0 = nhu anh goc, AM = te hon khong lam gi
    mau      nhu tren, do RIENG sac mau (a,b trong CIE-Lab), khong gop do sang
    chi_tiet ti le nang luong canh so voi AutoHDR, CHI xet vung co canh THAT
             (30% diem anh manh nhat cua anh dich). 1.00 = bang. <1 mo, >1 qua tay.
"""
import cv2
import numpy as np

_LAB_SCALE = np.array([100.0 / 255.0, 1.0, 1.0], np.float32)
_LAB_SHIFT = np.array([0.0, -128.0, -128.0], np.float32)


def _lab(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32) * _LAB_SCALE + _LAB_SHIFT


def _err_sang(a, b):
    return float(np.abs(_lab(a)[:, :, 0] - _lab(b)[:, :, 0]).mean())


def _err_mau(a, b):
    A, B = _lab(a), _lab(b)
    return float(np.sqrt((A[:, :, 1] - B[:, :, 1]) ** 2 + (A[:, :, 2] - B[:, :, 2]) ** 2).mean())


# 04/08 (ra soat vong 7) — DO NET O CANH DAI CO DINH.
# Laplacian tinh tren pixel THO nen gia tri co gian theo do phan giai: cung mot anh
# do duoc chi_tiet 0.51@400px / 0.41@800 / 0.29@1600 / 0.21@3000 (chenh 2.8 lan), va
# fp104790 do duoc 323.6@full-res / 1124.0@2048 / 1863.7@1024 (chenh 5.8 lan).
# Hau qua: audit_lich_su cham o 800 con box_sweep_ft cham o 1600 -> HAI BANG DIEM
# KHONG SO DUOC VOI NHAU, ma ca hai deu duoc dung de xep hang model.
_CANH_RES = 1024


def _ve_thang_do(bgr):
    """Dua ve canh dai co dinh de phep do net khong phu thuoc co anh."""
    s = _CANH_RES / max(bgr.shape[:2])
    if s >= 1.0:
        return bgr
    return cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def _canh(bgr):
    g = cv2.cvtColor(_ve_thang_do(bgr), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return np.abs(cv2.Laplacian(g, cv2.CV_32F))


def _chroma_tb(bgr):
    """Do bao hoa trung binh (khoang cach chroma toi truc xam) trong CIE-Lab."""
    L = _lab(bgr)
    return float(np.sqrt(L[:, :, 1] ** 2 + L[:, :, 2] ** 2).mean())


def do_ba_truc(pred, target, src, canh_pct=70):
    """pred/target/src: BGR uint8 CUNG kich thuoc. Tra ve (sang, mau, chi_tiet)."""
    sang = 100.0 * (1.0 - _err_sang(pred, target) / max(_err_sang(src, target), 1e-6))
    mau = 100.0 * (1.0 - _err_mau(pred, target) / max(_err_mau(src, target), 1e-6))

    # ⭐ 04/08 — PHAT MAT CAU TRUC SANG (cung goc benh voi truc mau ben duoi).
    # Ca hai truc deu chuan hoa theo sai so cua ANH GOC, nen mot anh XAM DAC (moi diem
    # = 128, khong con mot chi tiet nao) van an +22.9% diem SANG: neu anh goc toi hon
    # anh dich nhieu, thi "xam giua" tinh co gan trung binh anh dich hon la anh goc.
    # Do la ca thu "xam tit" ma chinh tu_kiem da co san va VAN DANG TRUOT.
    # Nay tru theo phan TUONG QUAN cau truc bi mat so voi anh goc — chon moc la anh
    # goc (khong phai 1.0) de hai ca neo van dung y nguyen:
    #     pred = src    -> khong tru gi   -> "khong lam gi" van dung 0.0
    #     pred = target -> corr = 1 >= src -> khong tru gi -> van 100.0
    #     xam dac       -> corr = 0        -> tru 100 diem
    def _corr_L(x, y):
        a = _lab(x)[:, :, 0].ravel(); b = _lab(y)[:, :, 0].ravel()
        a = a - a.mean(); b = b - b.mean()
        d = float(np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()))
        return float((a * b).sum() / d) if d > 1e-9 else 0.0

    _c_pred, _c_src = _corr_L(pred, target), _corr_L(src, target)
    if _c_src > 1e-6 and _c_pred < _c_src:
        sang -= 100.0 * (_c_src - _c_pred) / _c_src

    # ⭐ 04/08 — PHAT RIENG CHO "MAT BAO HOA".
    # Cong thuc tren chuan hoa theo sai so cua ANH GOC, nen mot anh DEN TRANG (chroma
    # = 0) van co the an diem DUONG: neu anh goc lech mau nhieu theo mot huong thi
    # "khong co mau gi ca" tinh co gan anh dich hon la "lech mau". Do tren bench10:
    # anh den trang duoc THUONG diem tren 6/10 canh. Day dung la lo hong ma chinh
    # docstring dau file noi thuoc CU mac phai (-0.9% cho anh den trang) — thuoc moi
    # MAC LAI theo mot duong khac.
    # Nay: pred nhat mau hon target thi tru thang theo ti le bao hoa con thieu.
    c_pred, c_tg = _chroma_tb(pred), _chroma_tb(target)
    if c_tg > 1e-6 and c_pred < c_tg:
        mau -= 100.0 * (1.0 - c_pred / c_tg)

    ct = _canh(target)
    m = ct > np.percentile(ct, canh_pct)
    cp = _canh(pred)
    ti_le = float(cp[m].mean() / max(ct[m].mean(), 1e-6))

    # ⭐ 04/08 — CHONG LACH BANG RAC NHIEU.
    # `chi_tiet` cu la ti le do LON nang luong canh, khong quan tam canh do co NAM
    # DUNG CHO khong. Rac nhieu Gauss sigma=10 lam ti le vot len 1.38 — vuot ca muc
    # "hoan hao" 1.00 — trong khi anh that te di. Do dung la cai bay da giet NAF_FT5.
    # Nhan them TUONG QUAN cuc bo giua ban do canh cua pred va target: nhieu thi
    # KHONG tuong quan (corr -> 0) nen bi keo xuong, con lam net qua tay thi van
    # tuong quan cao nen giu ti le > 1 (dung y nghia "qua tay" da ghi trong docstring).
    a = cp[m] - cp[m].mean()
    b = ct[m] - ct[m].mean()
    mau_so = float(np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()))
    corr = float((a * b).sum() / mau_so) if mau_so > 1e-9 else 0.0
    chi_tiet = ti_le * float(np.clip(corr, 0.0, 1.0))
    return sang, mau, chi_tiet


def tu_kiem(src, target, verbose=True):
    """Chay lai cac truong hop biet truoc dap an. Tra ve True neu thuoc con dung.

    Bat buoc chay sau moi lan sua cong thuc — day chinh la cai da bat duoc
    thuoc cu (den trang -0.9%, lam mo +1.9%).
    """
    # 04/08 (ra soat vong 7) — THEM 4 CA THU CON THIEU.
    # Bo ca thu cu KHONG co truong hop "rac nhieu", dung cai bay da giet NAF_FT5:
    # nhieu Gauss sigma=10 lam chi_tiet vot len 1.38 (vuot ca muc hoan hao 1.00) ma
    # thuoc van khen. Nay bat buoc phai bat duoc.
    rng = np.random.default_rng(0)                 # gieo co dinh -> ca thu tai lap duoc
    def _rac(sigma):
        n = rng.normal(0.0, sigma, target.shape).astype(np.float32)
        return np.clip(target.astype(np.float32) + n, 0, 255).astype(np.uint8)

    _blur_tg = cv2.GaussianBlur(target, (0, 0), 3)
    cases = {
        "tra ve dung anh dich": target,
        "khong lam gi": src,
        "bien thanh den trang": cv2.cvtColor(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY),
                                             cv2.COLOR_GRAY2BGR),
        "lam mo anh": cv2.GaussianBlur(src, (21, 21), 0),
        "xam tit": np.full_like(src, 128),
        "dich + rac nhieu s=5": _rac(5.0),
        "dich + rac nhieu s=10": _rac(10.0),
        "dich + rac nhieu s=20": _rac(20.0),
        # lam net QUA TAY: unsharp manh tren chinh anh dich
        "dich + lam net qua tay": np.clip(
            target.astype(np.float32) * 2.2 - _blur_tg.astype(np.float32) * 1.2,
            0, 255).astype(np.uint8),
        # nhat mau mot nua: kiem rieng duong "mat bao hoa"
        "dich + nhat mau mot nua": cv2.cvtColor(
            (lambda h: (h.__setitem__((slice(None), slice(None), 1),
                                      (h[:, :, 1] * 0.5).astype(np.uint8)) or h))(
                cv2.cvtColor(target, cv2.COLOR_BGR2HSV).copy()),
            cv2.COLOR_HSV2BGR),
    }
    got = {k: do_ba_truc(v, target, src) for k, v in cases.items()}
    checks = [
        ("tra ve dung anh dich", lambda s, m, d: s > 99 and m > 99 and abs(d - 1) < 0.01),
        ("khong lam gi", lambda s, m, d: abs(s) < 0.01 and abs(m) < 0.01),
        ("bien thanh den trang", lambda s, m, d: m < -5),        # PHAI phat nang mat mau
        ("lam mo anh", lambda s, m, d: d < 0.2),                 # PHAI bat duoc anh mo
        ("xam tit", lambda s, m, d: s < 0 and m < 0 and d < 0.05),
        # rac nhieu KHONG duoc coi la "nhieu chi tiet hon": chi_tiet phai <= 1.05
        ("dich + rac nhieu s=5", lambda s, m, d: d <= 1.05),
        ("dich + rac nhieu s=10", lambda s, m, d: d <= 1.05),
        ("dich + rac nhieu s=20", lambda s, m, d: d <= 1.05),
        # lam net qua tay PHAI lo ra (chi_tiet > 1), khong duoc khen
        ("dich + lam net qua tay", lambda s, m, d: d > 1.05),
        # nhat mau PHAI bi tru diem mau
        ("dich + nhat mau mot nua", lambda s, m, d: m < 60),
    ]
    ok = True
    for name, cond in checks:
        s, m, d = got[name]
        passed = bool(cond(s, m, d))
        ok = ok and passed
        if verbose:
            print(f"  [{'OK  ' if passed else 'HONG'}] {name:24s} "
                  f"sang {s:7.1f}%  mau {m:7.1f}%  chi_tiet {d:5.2f}")
    return ok
