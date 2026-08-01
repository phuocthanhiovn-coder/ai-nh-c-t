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


def _canh(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return np.abs(cv2.Laplacian(g, cv2.CV_32F))


def do_ba_truc(pred, target, src, canh_pct=70):
    """pred/target/src: BGR uint8 CUNG kich thuoc. Tra ve (sang, mau, chi_tiet)."""
    sang = 100.0 * (1.0 - _err_sang(pred, target) / max(_err_sang(src, target), 1e-6))
    mau = 100.0 * (1.0 - _err_mau(pred, target) / max(_err_mau(src, target), 1e-6))
    ct = _canh(target)
    m = ct > np.percentile(ct, canh_pct)
    chi_tiet = float(_canh(pred)[m].mean() / max(ct[m].mean(), 1e-6))
    return sang, mau, chi_tiet


def tu_kiem(src, target, verbose=True):
    """Chay lai cac truong hop biet truoc dap an. Tra ve True neu thuoc con dung.

    Bat buoc chay sau moi lan sua cong thuc — day chinh la cai da bat duoc
    thuoc cu (den trang -0.9%, lam mo +1.9%).
    """
    cases = {
        "tra ve dung anh dich": target,
        "khong lam gi": src,
        "bien thanh den trang": cv2.cvtColor(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY),
                                             cv2.COLOR_GRAY2BGR),
        "lam mo anh": cv2.GaussianBlur(src, (21, 21), 0),
        "xam tit": np.full_like(src, 128),
    }
    got = {k: do_ba_truc(v, target, src) for k, v in cases.items()}
    checks = [
        ("tra ve dung anh dich", lambda s, m, d: s > 99 and m > 99 and abs(d - 1) < 0.01),
        ("khong lam gi", lambda s, m, d: abs(s) < 0.01 and abs(m) < 0.01),
        ("bien thanh den trang", lambda s, m, d: m < -5),        # PHAI phat nang mat mau
        ("lam mo anh", lambda s, m, d: d < 0.2),                 # PHAI bat duoc anh mo
        ("xam tit", lambda s, m, d: s < 0 and m < 0 and d < 0.05),
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
