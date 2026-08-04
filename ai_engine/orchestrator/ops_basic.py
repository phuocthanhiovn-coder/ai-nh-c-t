"""Operator xac dinh (numpy/cv2), float32 [0,1] HxWx3 BGR, full-res, khong resize/re-encode.

Hop dong: apply(img: np.ndarray float32 [0,1] HxWx3 BGR, params: dict) -> np.ndarray cung shape/dtype.
"""
import numpy as np
import cv2

_GAMMA = 2.2


def _clip01(img):
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def brightness(img, params):
    """Exposure +-stops, gamma-aware (linear-light multiply)."""
    amount = float(params.get("amount", 0.0))
    lin = np.power(np.clip(img, 0.0, 1.0), _GAMMA)
    lin = lin * (2.0 ** amount)
    out = np.power(np.clip(lin, 0.0, 1.0), 1.0 / _GAMMA)
    return _clip01(out)


def contrast(img, params):
    """Xoay quanh gia tri trung vi (median luminance)."""
    amount = float(params.get("amount", 0.0))
    factor = max(0.0, 1.0 + amount)
    median = float(np.median(img))
    out = median + (img - median) * factor
    return _clip01(out)


def saturation(img, params):
    """Scale kenh S trong HSV."""
    amount = float(params.get("amount", 0.0))
    factor = max(0.0, 1.0 + amount)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0.0, 1.0)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return _clip01(out)


def temperature(img, params):
    """Dich kenh R/B de am hon (+) hoac lanh hon (-)."""
    amount = float(params.get("amount", 0.0))
    out = img.copy()
    shift = amount * 0.15
    out[..., 2] = out[..., 2] + shift       # R (BGR order index 2) len khi am hon
    out[..., 0] = out[..., 0] - shift       # B giam khi am hon
    return _clip01(out)


def shadows_lift(img, params):
    """Nang vung toi, tone-curve. amount 0..1."""
    amount = float(params.get("amount", 0.0))
    weight = np.clip(1.0 - img * 2.0, 0.0, 1.0)
    out = img + amount * weight * 0.5
    return _clip01(out)


def highlights_recover(img, params):
    """Ha vung chay. amount 0..1."""
    amount = float(params.get("amount", 0.0))
    weight = np.clip((img - 0.5) * 2.0, 0.0, 1.0)
    out = img - amount * weight * 0.5
    return _clip01(out)


def white_balance(img, params):
    """Gray-world white balance, strength 0..1 blend voi anh goc."""
    strength = float(params.get("strength", 1.0))
    means = img.reshape(-1, 3).mean(axis=0)
    means = np.clip(means, 1e-6, None)
    target = float(means.mean())
    scale = target / means
    scale_final = 1.0 + strength * (scale - 1.0)
    out = img * scale_final.reshape(1, 1, 3)
    return _clip01(out)


def sharpen(img, params):
    """Unsharp mask nhe, amount <= 0.5."""
    amount = float(params.get("amount", 0.2))
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
    out = img + amount * (img - blurred)
    return _clip01(out)


_auto_enhance_model = None
_auto_enhance_device = None
# (04/08: da xoa `_auto_enhance_failed`. No la co "tat op ca phien" — dat True mot
#  lan roi khong bao gio reset, khien moi anh sau do lang le ra anh goc. Nay loi cau
#  hinh thi nem len, loi inference thi canh bao TUNG LAN va dem vao BO_QUA_LAN.)
_auto_enhance_arch = None

from pathlib import Path as _Path

# 04/08 (ra soat vong 7) — DUONG DAN TUYET DOI, neo theo vi tri file nay.
# Ban cu la duong dan TUONG DOI: chay tu bat ky cwd nao khac goc repo thi
# os.path.exists() tra False -> roi ve arch "v1" + checkpoints/auto_enhance.pt (cung
# tuong doi, cung khong ton tai) -> FileNotFoundError -> op bi bo IM LANG. Da kiem
# chung: tu cwd C:/Users/Administrator thi
# exists('checkpoints/auto_enhance_config.json') = False. Tuc chay dich vu tu thu muc
# khac la MAT TOAN BO model ma khong mot loi bao nao.
_AUTO_ENHANCE_CONFIG = str(
    _Path(__file__).resolve().parents[2] / "checkpoints" / "auto_enhance_config.json"
)

# Dem so lan op bi bo qua, de ben goi ghi nhat ky DUNG SU THAT thay vi ghi "da chinh"
# cho ca 500 anh. Xem chu thich trong `auto_enhance`.
BO_QUA_LAN = 0


def model_complete():
    """Model doi CH_N+ co TU LO sang-deu/mau/cua so hay khong.

    04/08 (ra soat vong 7) — MOT NGUON SU THAT cho ca 4 duong giao hang.
    Truoc day CHI `brain/run.py` doc co khoa nay; `process.py`, `webapp/app.py`,
    `delivery/deliver.py`, `bracket_deliver.py` deu KHONG doc. Hau qua do duoc tren
    cung mot anh (hr_fp104610, cung checkpoint CH_N):
        brain (chuoi toi gian)      : p50 luma 176.1 / sat 19.1
        deliver.py --use-model      : p50 luma 185.5 / sat 21.8
        -> lech trung binh 9.72/255, 90.3% so diem anh lech >8 muc
        process.py / webapp         -> lech 2.33/255, 4.15% diem lech >8 muc
    Tuc chu duyet anh tren webapp roi giao khach bang deliver.py la khach nhan mot
    tam KHAC HAN tam vua duyet.
    Ly do goc: cum op bu (shadow_light/vibrance/dark_clean/grade_auto) duoc tune cho
    model DOI CU; voi CH_N+ chinh CLAUDE.md goi chung la "THUOC DOC" (den chay mat
    van, cua so phu mu — bang chung outputs/minimal_vs_full.jpg).
    """
    import json
    import os
    if not os.path.exists(_AUTO_ENHANCE_CONFIG):
        raise FileNotFoundError(
            f"Khong thay {_AUTO_ENHANCE_CONFIG} — khong duoc DOAN model_complete. "
            f"Doan sai la chay cum op bu len model CH_N+ (thuoc doc)."
        )
    with open(_AUTO_ENHANCE_CONFIG, encoding="utf-8") as f:
        cfg = json.load(f)
    if "model_complete" not in cfg:
        raise KeyError(f"{_AUTO_ENHANCE_CONFIG} thieu khoa 'model_complete'.")
    return bool(cfg["model_complete"])


def _load_auto_enhance():
    """Doc checkpoints/auto_enhance_config.json -> (model, device, arch).

    arch "v1": HDRNet pilot (infer.process_image, duong RGB, proxy 256).
    arch "v2": HDRNetV2 (BGR THANG nhu luc train/eval tren box GPU, make_proxy).
    Khong co config -> fallback v1 + checkpoints/auto_enhance.pt (hanh vi cu).
    """
    import json
    import os
    import torch

    torch.set_num_threads(3)

    # 04/08 — CAU HINH SAI PHAI LAM DUNG MAY (luat so 1 trong BAN_GIAO muc 6).
    # Ban cu: thieu file config -> AM THAM roi ve arch "v1" + checkpoint pilot (ban
    # BAC MAU da bi loai tu 22/07). Tuc mot loi go phim trong duong dan se lang le
    # doi model dang ban sang mot model khac han, va khong ai biet.
    # `.get(khoa, mac_dinh)` chinh la cai bay ma du an tu goi la "da giet du an".
    if not os.path.exists(_AUTO_ENHANCE_CONFIG):
        raise FileNotFoundError(
            f"Khong thay cau hinh model: {_AUTO_ENHANCE_CONFIG}. KHONG tu dong roi ve "
            f"model pilot v1 nua (ban do bac mau, da loai 22/07) — cau hinh sai phai "
            f"dung may chu khong duoc doi model im lang."
        )
    with open(_AUTO_ENHANCE_CONFIG, encoding="utf-8") as f:
        cfg = json.load(f)
    thieu = [k for k in ("arch", "checkpoint") if k not in cfg]
    if thieu:
        raise KeyError(
            f"{_AUTO_ENHANCE_CONFIG} thieu khoa bat buoc {thieu}. Phai ghi ro, khong "
            f"duoc de mac dinh ngam."
        )
    arch = cfg["arch"]
    ckpt = cfg["checkpoint"]
    kwargs = cfg.get("model_kwargs", {}) or {}
    if arch not in ("v1", "v2"):
        raise ValueError(f"arch={arch!r} khong hop le (chi 'v1' hoac 'v2').")
    # Duong dan checkpoint trong config la tuong doi so voi GOC REPO, khong phai cwd.
    if not os.path.isabs(ckpt):
        ckpt = str(_Path(__file__).resolve().parents[2] / ckpt)

    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)

    device = torch.device("cpu")
    if arch == "v2":
        from ai_engine.specialists.auto_enhance.gpu.model_v2 import HDRNetV2

        model = HDRNetV2(**kwargs).to(device)
    else:
        from ai_engine.specialists.auto_enhance.model import HDRNet

        model = HDRNet().to(device)

    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model, device, arch


def auto_enhance(img, params):
    """Chinh dep toan dien theo model hoc tu data (config: checkpoints/auto_enhance_config.json).

    Loi/thieu checkpoint -> bo qua (tra ve anh goc), khong crash pipeline.
    """
    global _auto_enhance_model, _auto_enhance_device, _auto_enhance_arch

    # 04/08 (ra soat vong 7) — GO BAY IM LANG.
    #
    # Ban cu: `_auto_enhance_failed` la bien module, dat True o hai cho va KHONG BAO
    # GIO reset. Sau lan loi dau tien, moi loi goi sau deu thoat ngay o day va tra ve
    # anh GOC, khong in them mot dong nao. Trong khi do MOI ben goi ghi nhat ky VO
    # DIEU KIEN: engine.py `applied.append(op_name)`, deliver.py:148, brain/run.py:60.
    # Kich ban da tai lap duoc: giao 500 anh, checkpoint hong o anh dau -> 500 anh ra
    # deu la anh GOC CHUA CHINH, chi co DUNG MOT dong [WARN] o dau log, con 500 dong
    # nhat ky deu ghi "auto_enhance da ap". Khong ai biet cho toi khi khach phan hoi.
    #
    # Nay: (a) loi CAU HINH/CHECKPOINT thi NEM LEN — cau hinh sai phai lam dung may
    # (luat so 1), khong duoc lang le giao anh chua chinh; (b) loi INFERENCE (het RAM
    # tren mot anh cu the...) thi van bo qua anh do de khong do ca lo, nhung CANH BAO
    # MOI LAN va dem vao BO_QUA_LAN de ben goi ghi nhat ky dung su that.
    global BO_QUA_LAN

    if _auto_enhance_model is None:
        # KHONG bat Exception o day nua: de loi cau hinh noi len tan ben goi.
        _auto_enhance_model, _auto_enhance_device, _auto_enhance_arch = _load_auto_enhance()

    try:
        if _auto_enhance_arch == "v2":
            import torch
            from ai_engine.specialists.auto_enhance.gpu.model_v2 import HDRNetV2

            proxy_res = getattr(_auto_enhance_model, "proxy_res", 384)
            t = torch.from_numpy(
                np.clip(img, 0.0, 1.0).transpose(2, 0, 1).copy()
            ).unsqueeze(0).float().to(_auto_enhance_device)
            proxy = HDRNetV2.make_proxy(t, proxy_res)
            with torch.no_grad():
                out_t, _grid = _auto_enhance_model(proxy, t)
            out = out_t.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
            return _clip01(out)

        from ai_engine.specialists.auto_enhance.infer import process_image

        img_u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        out_u8, _grid_shape = process_image(_auto_enhance_model, img_u8, _auto_enhance_device)
        out = out_u8.astype(np.float32) / 255.0
        return _clip01(out)
    except Exception as exc:
        # CANH BAO MOI LAN (khong tat op ca phien nua) + dem lai de ben goi biet.
        BO_QUA_LAN += 1
        print(f"[WARN] auto_enhance: loi khi chay inference ({exc}). BO QUA anh nay "
              f"-> ANH RA LA ANH GOC CHUA CHINH (lan thu {BO_QUA_LAN}). Nhat ky cua "
              f"ben goi co the van ghi 'da chinh' — doi chieu BO_QUA_LAN truoc khi "
              f"tin bao cao.", flush=True)
        return img
