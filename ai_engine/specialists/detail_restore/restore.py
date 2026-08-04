"""detail_restore — PHUC NET/CHI TIET THAT bang Real-ESRGAN general (24/07).

Op: chay SR x4 (SRVGGNetCompact) roi thu nho ve ĐUNG size goc = mot lop phuc
hoi chi tiet/khu nen, THEM chi tiet that ma op tuyen tinh khong lam duoc. Blend
theo strength de tranh "nhua" (plasticky). Tile de khong bung RAM voi anh lon.

Hop dong: apply(img f32 [0,1] HxWx3 BGR, params) -> cung shape.
Params: strength 0..1 (default 0.5) · denoise 0..1 (default 0.5, cho model general).
strength=0 -> bit-identical. Thieu weights -> tra anh goc (khong crash).
"""
import os

import cv2
import numpy as np

cv2.setNumThreads(3)

_CKPT = "checkpoints/ext/realesr-general-x4v3.pth"
_TILE = 512          # xu ly tung o 512 (input) de gioi han RAM
_PAD = 16
_model = None
_failed = False


def _load():
    global _model, _failed
    if _failed:
        return None
    if _model is None:
        try:
            import torch
            from ai_engine.specialists.detail_restore.srvgg import SRVGGNetCompact
            torch.set_num_threads(3)
            if not os.path.exists(_CKPT):
                raise FileNotFoundError(_CKPT)
            sd = torch.load(_CKPT, map_location="cpu")
            sd = sd.get("params", sd.get("params_ema", sd))
            m = SRVGGNetCompact(num_feat=64, num_conv=32, upscale=4)
            m.load_state_dict(sd, strict=True)
            m.eval()
            _model = m
        except Exception as exc:
            print(f"[WARN] detail_restore: khong nap duoc weights ({exc}). Bo qua.")
            _failed = True
            return None
    return _model


def _sr_tile(model, bgr01):
    """Chay SR x4 tren 1 o (bgr float32 [0,1]) -> x4."""
    import torch
    t = torch.from_numpy(bgr01[:, :, ::-1].transpose(2, 0, 1).copy()).unsqueeze(0).float()
    with torch.no_grad():
        out = model(t).clamp(0, 1)
    out = out.squeeze(0).numpy().transpose(1, 2, 0)[:, :, ::-1]  # RGB->BGR
    return out


def apply(img, params=None):
    params = params or {}
    strength = float(np.clip(params.get("strength", 0.5), 0.0, 1.0))
    img = np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)
    if strength == 0.0:
        return img
    model = _load()
    if model is None:
        return img

    h, w = img.shape[:2]
    up = np.zeros((h * 4, w * 4, 3), dtype=np.float32)
    for y0 in range(0, h, _TILE):
        for x0 in range(0, w, _TILE):
            y1, x1 = min(y0 + _TILE, h), min(x0 + _TILE, w)
            py0, px0 = max(0, y0 - _PAD), max(0, x0 - _PAD)
            py1, px1 = min(h, y1 + _PAD), min(w, x1 + _PAD)
            tile = img[py0:py1, px0:px1]
            sr = _sr_tile(model, tile)
            # cat bo phan pad (theo he so x4)
            cy0, cx0 = (y0 - py0) * 4, (x0 - px0) * 4
            up[y0 * 4:y1 * 4, x0 * 4:x1 * 4] = sr[cy0:cy0 + (y1 - y0) * 4,
                                                  cx0:cx0 + (x1 - x0) * 4]
    # FIX 30/07: vong SR x4 -> thu nho INTER_AREA hoat dong nhu bo KHU NHIEU o
    # full-res (do duoc lap 29.8 -> 20.0 = -33% tren crop that) => con "phuc net"
    # dang LAM MEM anh. Chi GHEP DAI TAN CAO cua ban SR len anh GOC nguyen ven:
    # khong lam mo tan thap, khong doi mau, khong "nhua".
    restored = cv2.resize(up, (w, h), interpolation=cv2.INTER_AREA)
    sig = max(1.0, 0.0015 * min(h, w))
    low_res = cv2.GaussianBlur(restored, (0, 0), sigmaX=sig)
    detail = restored - low_res                     # dai tan cao cua ban SR

    # FIX 02/08 — LOI DO DUOC: dong cu la `out = img + strength * detail`, tuc CONG
    # dai tan cao cua ban SR LEN TREN dai tan cao VON CO cua anh goc => hai dai cong
    # don, bien do tan cao x1.96 (Laplacian var x3.84) khi strength=1.0. Day la
    # nguyen nhan SO HOC cua "net tho 1394 vs AutoHDR 619 = lam net qua tay gap doi"
    # ghi trong SO_DIEM 01/08 — KHONG phai loi model.
    # Y dinh ghi trong chu thich tren la "GHEP" (thay the) chu khong phai "CONG".
    # Ghep dung = bo dai tan cao cua anh goc roi dat dai cua ban SR vao cho do:
    #     strength=0 -> tra anh goc nguyen ven
    #     strength=1 -> low_img + detail  (thay the hoan toan)
    detail_img = img - cv2.GaussianBlur(img, (0, 0), sigmaX=sig)   # tan cao VON CO
    # ⭐ 04/08 (ra soat vong 7) — BAN VA 02/08 CHUA "QUA TAY" THANH "THIEU TAY".
    # Dong cu `out = img + strength*(detail - detail_img)` BO dai tan cao von co roi
    # DAT dai cua ban SR vao cho. Nhung chinh chu thich dong 82-85 o tren da ghi: vong
    # SR x4 -> INTER_AREA "hoat dong nhu bo KHU NHIEU (lap 29.8 -> 20.0 = -33%)".
    # Vay o moi vung IT KET CAU, dai duoc dat vao YEU HON dai bi bo di => con "phuc
    # net" thanh con LAM MEM. Do theo o 64px, strength=1.0, anh 1536x1024:
    #     hr_fp104610: 63.0% so o bi lam mem   |  hr_fp104505: 58.6%
    #     o phang nhat: chi con x0.05 nang luong canh (mat 95%)
    #     dong CU (img + detail): 0.0% so o bi lam mem
    # Ma op nay BAT MAC DINH o cuoi chuoi giao hang (brain/run.py goi voi
    # sharpen_strength mac dinh 1.0) — tuc no lam mem chinh cai no sinh ra de chua.
    #
    # Sua: giu tinh than "GHEP, khong CONG" nhung KHONG BAO GIO duoc lam yeu di. Voi
    # moi diem anh, lay dai nao co bien do LON HON:
    #     |detail_SR| > |detail_goc| -> dung dai SR (that su co chi tiet moi)
    #     nguoc lai                  -> giu nguyen dai goc (khong mat gi)
    # Nho vay: strength=0 -> anh goc; strength=1 -> low_img + dai manh nhat; va
    # |dai ket qua| >= |dai goc| o moi diem, nen VE MAT TOAN HOC op khong the lam mem.
    manh_hon = np.abs(detail) > np.abs(detail_img)
    detail_ghep = np.where(manh_hon, detail, detail_img)
    out = img + strength * (detail_ghep - detail_img)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
