"""brain.run — NÃO v1.5: khám 2 lần + tay theo vùng (chuẩn hóa từ demo 24/07).

Chuỗi: khám ảnh gốc → denoise + model màu → KHÁM LẠI ảnh giữa chuỗi →
kê toa vùng (thắp góc chỉ trên kiến trúc, rửa bùn tha đồ vật) → cửa sổ theo
mask mắt → nắn dọc → phục nét. Trả (ảnh, hồ sơ toa đầy đủ để giải thích).

CLI:  python -m ai_engine.brain.run --in <anh> --out <anh_ra>
"""
import argparse
import json

import cv2
import numpy as np

cv2.setNumThreads(3)

from ai_engine.brain.diagnose import diagnose
from ai_engine.orchestrator.registry import REGISTRY
from ai_engine.orchestrator.region_apply import (region_apply, build_arch_mask)
from ai_engine.specialists.shadow_light import light as _sl


def process(img, sharpen=None, sharpen_strength=1.0):
    """THICH UNG theo do net dau vao (25/07 — bai hoc job that): op tune cho data
    MEM se NAU HONG input SAC (tuong loang lo, cua so nhu tranh ve). Do lap dau
    vao: SAC (>=80, bracket that) -> profile NHE (khong detail_restore, finish
    nhe); MEM (<80, data cu/anh don) -> profile MANH (co detail_restore).
    sharpen=None -> tu quyet theo do net; True/False -> ep."""
    R = REGISTRY
    record = {"steps": []}

    # do net dau vao -> chon profile
    _g0 = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    in_lap = cv2.Laplacian(_g0, cv2.CV_64F).var()
    sharp_input = in_lap >= 80.0
    if sharpen is None:
        sharpen = not sharp_input        # input da sac thi KHONG detail_restore
    record["input_lap"] = round(float(in_lap), 1)
    record["profile"] = "gentle" if sharp_input else "strong"

    d0 = diagnose(img)                       # kham lan 1 (anh goc, co mat segment)
    record["diagnosis_before"] = {k: v for k, v in d0.items() if not k.startswith("_")}
    masks = d0.get("_masks")

    # 26/07: model đời CH_N+ (train trên data merge ĐÚNG) tự lo sáng-đều/màu →
    # cụm op bù (shadow_light/vibrance/dark_clean) thành CHỒNG LIỀU (đèn cháy,
    # cửa sổ mù — bằng chứng outputs/minimal_vs_full.jpg). Cờ model_complete
    # trong auto_enhance_config.json bật chuỗi TỐI GIẢN.
    model_complete = False
    try:
        with open("checkpoints/auto_enhance_config.json", "r", encoding="utf-8") as _f:
            model_complete = bool(json.load(_f).get("model_complete", False))
    except Exception:
        pass
    record["model_complete"] = model_complete

    out = R["denoise"]["fn"](img, {"denoise_strength": 0.35, "sharpen_amount": 0.0})
    record["steps"].append({"op": "denoise", "reason": "khu nhieu truoc model"})
    out = R["auto_enhance"]["fn"](out, {})
    record["steps"].append({"op": "auto_enhance", "reason": "mau/tone (model CH hien hanh)"})

    d1 = diagnose(out, with_masks=False)     # kham lan 2 (sau model)
    record["diagnosis_mid"] = {k: v for k, v in d1.items() if not k.startswith("_")}

    # NAO v2 (25/07 dem): LIEU DATA-RUT-RA. Khai thac lieu toi uu tren 905 cap
    # (tools/mine_doses) roi hoc predictor tuyen tinh -> predictor CHI NGANG
    # baseline (9 features khong du tin hieu doan lieu rieng tung anh; std cao).
    # NHUNG data lo ra su that lon: lieu tay CU QUA NANG (sl 0.5 vs 0.16 toi uu,
    # vb 0.75 vs 0.36) -> dung "sang qua nang / mau khong tu nhien" chu che nhieu
    # vong. => dung LIEU TRUNG BINH TOI UU lam mac dinh (nhe hon nhieu). Predictor
    # giu file nhung TAT (khong hon baseline). Per-image dose can feature giau hon
    # / model phi tuyen — de sau.
    learned = {"sl": 0.16, "wh": 0.30, "vb": 0.36, "dc": 0.34, "bk": 0.58}
    record["learned_doses"] = learned

    if model_complete:
        record["steps"].append({"op": "minimal-chain",
                                "reason": "model CH_N+ tu lo sang-deu/mau; bo op bu chong lieu"})
    elif masks is not None:
        from ai_engine.brain.material_grade import _warm_gate
        from ai_engine.specialists.segment_room.seg import segment_fine
        # 25/07 (vòng chấm 10): trần/dầm GỖ ẤM không được thắp trắng như tường.
        arch = build_arch_mask(masks) * (1.0 - _warm_gate(out))
        # 25/07 (vòng chấm 12): NỘI THẤT trong tối (ghế/sofa/bàn) cũng phải được
        # thắp — trước bị né oan vì mask chỉ có kiến trúc. Lò/TV vẫn miễn nhiễm
        # (không nằm trong wood/fabric).
        mats_pre = segment_fine(img)
        furn = np.clip(mats_pre.get("wood", 0) + mats_pre.get("fabric", 0), 0, 1)
        lift_mask = np.clip(arch + 0.85 * furn, 0.0, 1.0)
        need = max(0.0, min(1.0, (d1["dark_frac"] - 0.02) / 0.20))
        if learned:
            amt = learned["sl"]
        else:
            amt = round(0.5 + 0.5 * need, 2) if need > 0.05 else 0.0
        if amt > 0.05:
            out = region_apply(out, _sl.apply, {"amount": amt}, lift_mask)
            record["steps"].append({"op": "shadow_light@arch+furniture", "amount": amt,
                                    "reason": f"vung toi sau model {d1['dark_frac']:.0%}, thap kien truc + noi that (ne do dien)"})
            # KHU VANG goc vua thap: lam mat nhe vung toi da nang (chong "goc vang")
            y_mid = np.maximum(out @ np.array([0.0722, 0.7152, 0.2126], dtype=np.float32), 1e-4)
            dark_w = np.clip((0.42 - y_mid) / 0.30, 0, 1)
            cool_mask = np.clip(lift_mask * dark_w, 0, 1)
            if float(cool_mask.mean()) > 0.01:
                out = region_apply(out, R["temperature"]["fn"], {"amount": -0.07}, cool_mask)
                record["steps"].append({"op": "cool@dark", "reason": "khu vang vung toi vua thap"})
        # rua bun: chi kien truc (do vat/go giu mau). Input SAC -> vibrance nhe.
        if sharp_input:
            dc = 0.15
            vib_params = {"whites": 0.3, "vibrance": 0.25, "dark_clean": 0.0}
        elif learned:
            dc = learned["dc"]
            vib_params = {"whites": learned["wh"], "vibrance": learned["vb"], "dark_clean": 0.0}
        else:
            dc = round(min(0.7, max(0.0, (d1["dark_sat"] - 45) / 50)), 2)
            vib_params = {"whites": 0.4, "vibrance": 0.75, "dark_clean": 0.0}
        out = R["vibrance"]["fn"](out, vib_params)
        if dc > 0.05:
            out = region_apply(out, R["vibrance"]["fn"],
                               {"whites": 0.0, "vibrance": 0.0, "dark_clean": dc}, arch)
        record["steps"].append({"op": "vibrance(+dark_clean@arch)", "dark_clean": dc,
                                "reason": f"bun vung toi sau model {d1['dark_sat']:.0f}"})
    else:
        out = R["shadow_light"]["fn"](out, {"amount": 0.35})
        out = R["vibrance"]["fn"](out, {"whites": 0.45, "vibrance": 0.7, "dark_clean": 0.35})
        record["steps"].append({"op": "fallback-global", "reason": "mat segment loi"})

    # CUA SO (26/07, chu chi ca fp104610): model THOI TRANG + am xanh cua so ->
    # mat nha/cay; AutoHDR keo phoi sang XUONG lo ro. SegFormer mu (mask 2 dot) nen
    # truoc day chi "bao ve" cai da hong. MAT-MOI gsam (Grounding DINO + SAM2) bat
    # dung cua so -> keo phoi sang tu ANH GOC (bracket merge co noi dung nha). Tinh
    # mask o day; GHEP sau finish (de finish khong lam canh ngoai thanh "tranh ve").
    from ai_engine.brain.window_recover import get_window_mask
    _before_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    win_mask_gsam = get_window_mask(_before_u8)
    if win_mask_gsam is not None:
        record["steps"].append({"op": "eye:gsam-window",
                                "frac": round(float(win_mask_gsam.mean()), 4)})
    elif not sharp_input and (d0.get("frac_window", -1.0) >= 0.01 or d0.get("frac_window", -1.0) < 0.0):
        s = round(min(0.95, 0.5 + max(d0.get("frac_window", 0.1), 0.1) * 2.5), 2)
        out = R["window_pull"]["fn"](out, {"strength": s, "saturation_boost": 0.5})
        record["steps"].append({"op": "window_pull(fallback)", "strength": s})

    if d0["scene"] in ("interior", "exterior_ground", "general"):
        out = R["straighten"]["fn"](out, {"strength": 1.0})
        record["steps"].append({"op": "straighten", "reason": f"scene={d0['scene']}"})

    # finish_detail: input SAC chi cham nhe (tranh loang lo/gion); MEM day manh.
    if sharp_input:
        fd = {"clarity": 0.35, "detail": 0.25, "black": learned["bk"]}
    else:
        fd = {"clarity": 0.8, "detail": 1.0, "black": learned["bk"]}
    out = R["finish_detail"]["fn"](out, fd)
    record["steps"].append({"op": "finish_detail", **fd, "reason": f"profile={record['profile']}"})

    # KEO CUA SO XUONG — CO DIEU KIEN (26/07): CH_N+ da giu cua so tot; recover
    # grey-world de len chi lam MU (bang chung minimal_vs_full.jpg). Chi recover
    # khi vung cua so trong OUTPUT that su CHAY (luma cao + sat thap).
    if win_mask_gsam is not None:
        sel = win_mask_gsam > 0.5
        if int(sel.sum()) > 100:
            wl = float((out @ np.array([0.114, 0.587, 0.299], dtype=np.float32))[sel].mean())
            wsat = float(np.abs(out[sel] - out[sel].mean(axis=-1, keepdims=True)).mean())
            if wl > 0.82 and wsat < 0.03:      # cua so van chay -> moi keo
                from ai_engine.brain.window_recover import recover_windows
                out = recover_windows(img, out, win_mask_gsam)
                record["steps"].append({"op": "window_recover",
                                        "reason": f"cua so con chay (luma {wl:.2f}) -> keo tu anh goc"})
            else:
                record["steps"].append({"op": "window_keep",
                                        "reason": f"model giu cua so tot (luma {wl:.2f}) -> khong dong them"})

    # TANG CHAT LIEU (25/07) — BO khi model_complete (op tune cho model cu).
    if not model_complete:
        try:
            from ai_engine.specialists.segment_room.seg import segment_fine
            from ai_engine.brain.material_grade import apply_material_grade
            mats = segment_fine(img)
            mat_log = []
            out = apply_material_grade(out, mats=mats, record=mat_log)
            record["steps"].extend(mat_log)
        except Exception as e:
            record["steps"].append({"op": "material:SKIP", "reason": str(e)[:100]})

    # PHUC NET CUOI (detail_restore, Real-ESRGAN) — tri "mo khong net nhu HDR".
    if sharpen:
        out = R["detail_restore"]["fn"](out, {"strength": sharpen_strength})
        record["steps"].append({"op": "detail_restore", "strength": sharpen_strength,
                                "reason": "phuc net that (Real-ESRGAN), keo lap gan AutoHDR"})
    return out, record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="outp", required=True)
    a = ap.parse_args()
    img = cv2.imread(a.inp).astype(np.float32) / 255.0
    out, record = process(img)
    assert out.shape == img.shape, "VI PHAM kich thuoc"
    cv2.imwrite(a.outp, (out * 255).clip(0, 255).astype(np.uint8),
                [cv2.IMWRITE_JPEG_QUALITY, 100,
                 cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])
    print(json.dumps(record, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
