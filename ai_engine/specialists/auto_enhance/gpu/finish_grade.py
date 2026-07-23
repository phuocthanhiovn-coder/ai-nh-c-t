"""Tang FINISH tat dinh sau operator AI: chua "mo bot / nhiem mau / thieu bao hoa / thieu am".
Scene-aware: NGOAI TROI keo bao hoa + am manh hon TRONG NHA (theo so do measure_gap).
THU TU quan trong: khu nhieu chroma TRUOC -> roi moi tang bao hoa (khong khuech dai nhieu).

grade(bgr, preset) -> bgr. __main__ render so sanh 4 cot [goc|AI|AI+grade|target] de soi mat.
"""
import os
import argparse

import cv2
import numpy as np


# Preset hieu chinh ~80-85% do lech do duoc (tranh qua tay), tinh chinh bang mat.
PRESETS = {
    # trong nha: thieu bao hoa 18% + tuong phan + khu mu (black point) chua "mo bot"
    "indoor":  dict(sat=1.26, warm_b=0.3, warm_a=0.5, contrast=1.10, clarity=0.22, black=8, chroma_dn=2, sharpen=0.85),
    # ngoai troi that su (troi/cay): bao hoa manh + am, nhung khong qua tay
    "outdoor": dict(sat=1.36, warm_b=2.2, warm_a=1.5, contrast=1.06, clarity=0.15, black=9, chroma_dn=2, sharpen=0.70),
}


def _scene(bgr):
    """Ngoai troi = co TROI XANH lon o phia tren. Chi dua vao mau xanh troi:
    - go am: R>B  -> loai
    - tuong trang: R~=B -> loai
    - troi xanh: B>R ro + sang -> AN.  Bao thu: khong chac -> indoor (an toan)."""
    arr = bgr.astype(np.int16)
    b, r = arr[:, :, 0], arr[:, :, 2]
    v = arr.max(axis=2)
    H = bgr.shape[0]
    top = slice(0, int(H * 0.30))
    blue_sky = ((b[top] > r[top] + 14) & (v[top] > 150))
    return "outdoor" if blue_sky.mean() > 0.22 else "indoor"


def grade(bgr, sat=1.15, warm_b=0.0, warm_a=0.0, contrast=1.06, clarity=0.15,
          black=0, chroma_dn=2, sharpen=0.0):
    img = bgr.copy()

    # 1) KHU NHIEU CHROMA (fix "nhiem mau") - lam TRUOC khi tang bao hoa
    if chroma_dn > 0:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab)
        d = int(chroma_dn) * 2 + 1
        a = cv2.bilateralFilter(a, d, chroma_dn * 12, chroma_dn * 4)
        b = cv2.bilateralFilter(b, d, chroma_dn * 12, chroma_dn * 4)
        img = cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_LAB2BGR)

    # 1b) KEO DIEM DEN (khu mu/duc "mo bot"): map [black,255] -> [0,255]
    if black > 0:
        x = img.astype(np.float32)
        x = np.clip((x - black) * (255.0 / (255.0 - black)), 0, 255)
        img = x.astype(np.uint8)

    # 2) BAO HOA + AM/CAST trong Lab (nhan chroma quanh 0 = tang bao hoa; cong = dich mau)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    a = (lab[:, :, 1] - 128.0) * sat + warm_a
    b = (lab[:, :, 2] - 128.0) * sat + warm_b
    lab[:, :, 1] = np.clip(a + 128.0, 0, 255)
    lab[:, :, 2] = np.clip(b + 128.0, 0, 255)
    img = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # 3) TUONG PHAN nhe (S-curve quanh 0.5) - chua "mo bot"
    if abs(contrast - 1.0) > 1e-3:
        x = img.astype(np.float32) / 255.0
        x = np.clip((x - 0.5) * contrast + 0.5, 0, 1)
        img = (x * 255.0).astype(np.uint8)

    # 4) CLARITY (do trong): unsharp ban kinh lon tren L de tang tuong phan cuc bo, khong gat
    if clarity > 0:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0].astype(np.float32)
        blur = cv2.GaussianBlur(L, (0, 0), 9)
        L = np.clip(L + clarity * (L - blur), 0, 255)
        lab[:, :, 0] = L.astype(np.uint8)
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 5) LAM NET (fix "mo/blurry"): unsharp ban kinh NHO tren L (chi tiet), khong cham chroma
    #    -> tang do net nhu AutoHDR ma khong len mau via.
    if sharpen > 0:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0].astype(np.float32)
        blur = cv2.GaussianBlur(L, (0, 0), 1.2)
        high = L - blur
        # chong halo: gioi han bien do vien
        high = np.clip(high, -18, 18)
        L = np.clip(L + sharpen * high, 0, 255)
        lab[:, :, 0] = L.astype(np.uint8)
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return img


def grade_auto(bgr, name=None):
    """Nhan dien scene theo NOI DUNG cua chinh anh -> chon preset."""
    return grade(bgr, **PRESETS[_scene(bgr)])


def protect_highlights(before_bgr, ai_bgr, thresh_ai=236, thresh_orig=230, feather=9):
    """DO-NO-HARM: vung AI THOI TRANG (chay) nhung anh GOC con chi tiet -> keo chi tiet
    goc ve, chong 'cua so chay / tuong chay' do model lam sang qua tay.
    before_bgr va ai_bgr phai cung kich thuoc (operator giu nguyen size)."""
    if before_bgr.shape[:2] != ai_bgr.shape[:2]:
        before_bgr = cv2.resize(before_bgr, (ai_bgr.shape[1], ai_bgr.shape[0]),
                                interpolation=cv2.INTER_AREA)
    ai_l = cv2.cvtColor(ai_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    or_l = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # chi phuc hoi noi AI chay MA goc con giu duoc (goc < nguong)
    mask = ((ai_l > thresh_ai) & (or_l < thresh_orig)).astype(np.float32)
    if mask.sum() < 1:
        return ai_bgr
    mask = cv2.GaussianBlur(mask, (0, 0), feather)
    mask = np.clip(mask, 0, 1)[:, :, None]
    out = ai_bgr.astype(np.float32) * (1 - mask) + before_bgr.astype(np.float32) * mask
    return np.clip(out, 0, 255).astype(np.uint8)


# --------- render so sanh 4 cot de soi mat ---------
def _cell(img, label, w):
    h = int(round(img.shape[0] * w / img.shape[1]))
    s = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    strip = np.full((28, w, 3), 20, np.uint8)
    cv2.putText(strip, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([strip, s])


def main():
    import torch
    from .model_v2 import HDRNetV2
    from .train_sweep import split_filenames
    from .measure_gap import load_cfg, run_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="outputs/eval/grade_compare.jpg")
    ap.add_argument("--cell-width", type=int, default=380)
    args = ap.parse_args()

    cv2.setNumThreads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_cfg(args.ckpt + ".meta", device)
    model = HDRNetV2(**cfg).to(device)
    st = torch.load(args.ckpt, map_location=device)
    if isinstance(st, dict) and "state_dict" in st:
        st = st["state_dict"]
    model.load_state_dict(st); model.eval()

    _, val = split_filenames(args.data_dir, 0.12)
    val = sorted(val)
    # chon xen ke: vai anh co "drone" trong ten (co the co aerial that) + vai anh noi that
    hint_out = [n for n in val if n.lower().startswith(("drone", "dji"))][:4]
    hint_in = [n for n in val if not n.lower().startswith(("drone", "dji"))][:4]
    picked = hint_in + hint_out
    bd = os.path.join(args.data_dir, "before"); ad = os.path.join(args.data_dir, "after")

    rows = []
    cw = args.cell_width
    for name in picked[: args.n if args.n else len(picked)]:
        b = cv2.imread(os.path.join(bd, name)); a = cv2.imread(os.path.join(ad, name))
        if b is None or a is None:
            continue
        b_proc, ai = run_model(model, b, device)
        sc = _scene(ai)  # nhan dien theo NOI DUNG
        g = grade(ai, **PRESETS[sc])
        a_m = cv2.resize(a, (ai.shape[1], ai.shape[0]), interpolation=cv2.INTER_AREA)
        row = np.hstack([
            _cell(b_proc, f"GOC {name[:18]}", cw),
            _cell(ai, "AI (chua grade)", cw),
            _cell(g, f"AI+GRADE [{sc}]", cw),
            _cell(a_m, "AUTOHDR target", cw),
        ])
        rows.append(row)
    maxw = max(r.shape[1] for r in rows)
    rows = [r if r.shape[1] == maxw else np.hstack([r, np.full((r.shape[0], maxw-r.shape[1], 3), 20, np.uint8)]) for r in rows]
    sheet = np.vstack([np.vstack([r, np.full((5, maxw, 3), 40, np.uint8)]) for r in rows])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[+] {args.out} shape={sheet.shape}")


if __name__ == "__main__":
    main()
