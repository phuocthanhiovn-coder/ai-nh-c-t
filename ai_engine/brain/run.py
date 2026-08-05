"""brain.run — NÃO v1.5: khám 2 lần + tay theo vùng (chuẩn hóa từ demo 24/07).

Chuỗi: khám ảnh gốc → denoise + model màu → KHÁM LẠI ảnh giữa chuỗi →
kê toa vùng (thắp góc chỉ trên kiến trúc, rửa bùn tha đồ vật) → cửa sổ theo
mask mắt → nắn dọc → phục nét. Trả (ảnh, hồ sơ toa đầy đủ để giải thích).

CLI:  python -m ai_engine.brain.run --in <anh> --out <anh_ra>
"""
import argparse
import json
from pathlib import Path

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

    # ⭐ 04/08 (ra soat vong 7) — HAI LOI TRONG CONG PHUC NET.
    #
    # (1) DO NHAM ANH. `sharpen` quyet dinh co chay detail_restore o CUOI chuoi, nhung
    #     lai duoc tinh tu anh DAU VAO. Bracket da gop thi luon sac (fp104505 lap 84.8)
    #     nen cong luon ket luan "input sac, khong can phuc net" — trong khi chinh
    #     MODEL o giua chuoi moi la thu pha net. Do o co anh khach nhan: vao 209 ->
    #     ra 126, tuc mat 40% do net CUA CHINH ANH GOC, ma cong khong he hay biet.
    #
    # (2) NGUONG 80 CO GIAN THEO DO PHAN GIAI. Cung anh fp104790 do duoc 323.6
    #     @full-res / 1124.0 @2048 / 1863.7 @1024 — chenh 5.8 lan. O 2048 ca 10/10 anh
    #     BENCH-10 deu >=80 (profile "gentle"); o co giao that 6024 chi con 5/10, tuc
    #     5 anh nhay sang profile "strong" — duong CHUA BAO GIO duoc tune o co do.
    #
    # Sua: do tren ban resize ve CANH DAI CO DINH (nguong moi co nghia that), va dua
    # quyet dinh phuc net xuong SAU khi model chay, so anh RA voi anh VAO.
    _LAP_RES = 1600

    def _do_net(x):
        g = cv2.cvtColor((np.clip(x, 0, 1) * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        s = _LAP_RES / max(g.shape[:2])
        if s < 1.0:
            g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        return float(cv2.Laplacian(g, cv2.CV_64F).var())

    in_lap = _do_net(img)
    sharp_input = in_lap >= 80.0
    record["input_lap"] = round(in_lap, 1)
    record["profile"] = "gentle" if sharp_input else "strong"
    _sharpen_ep = sharpen            # None = de chuoi tu quyet SAU khi model chay

    d0 = diagnose(img)                       # kham lan 1 (anh goc, co mat segment)
    record["diagnosis_before"] = {k: v for k, v in d0.items() if not k.startswith("_")}
    masks = d0.get("_masks")

    # 26/07: model đời CH_N+ (train trên data merge ĐÚNG) tự lo sáng-đều/màu →
    # cụm op bù (shadow_light/vibrance/dark_clean) thành CHỒNG LIỀU (đèn cháy,
    # cửa sổ mù — bằng chứng outputs/minimal_vs_full.jpg). Cờ model_complete
    # trong auto_enhance_config.json bật chuỗi TỐI GIẢN.
    # 04/08 (ra soat vong 7) — DUONG DAN TUYET DOI + BO except TRON.
    # Ban cu doc bang duong dan TUONG DOI trong `try/except Exception: pass`. Chay tu
    # bat ky cwd nao khac goc repo -> khong mo duoc file -> model_complete AM THAM ve
    # False -> brain chay CHUOI DAY DU (shadow_light + vibrance + dark_clean) len
    # model CH_N. Ma chinh CLAUDE.md ghi cum op do la "THUOC DOC" voi CH_N+: den chay
    # mat van, cua so phu mu. Tuc chi doi thu muc chay la anh giao khach xau di, khong
    # mot dong canh bao nao.
    _CFG = Path(__file__).resolve().parents[2] / "checkpoints" / "auto_enhance_config.json"
    if not _CFG.exists():
        raise FileNotFoundError(
            f"Khong thay {_CFG}. KHONG doan `model_complete=False` nua — doan sai la "
            f"chay cum op bu len model CH_N+ (thuoc doc). Cau hinh sai phai dung may."
        )
    with open(_CFG, "r", encoding="utf-8") as _f:
        _cfg = json.load(_f)
    if "model_complete" not in _cfg:
        raise KeyError(f"{_CFG} thieu khoa 'model_complete' — phai ghi ro True/False.")
    model_complete = bool(_cfg["model_complete"])
    record["model_complete"] = model_complete

    out = R["denoise"]["fn"](img, {"denoise_strength": 0.35, "sharpen_amount": 0.0})
    record["steps"].append({"op": "denoise", "reason": "khu nhieu truoc model"})
    # ⭐ 04/08 (sua lai trong ngay) — MOC SO SANH PHAI DO SAU DENOISE.
    # Ban va sang nay so `mid_lap` (sau model) voi `in_lap` (do tren ANH GOC), nhung
    # `denoise` chay TRUOC model va tu no da an mat mot phan do net. Nen ti so
    # mid_lap/in_lap gom CA phan denoise, trong khi nguong lai la 0.90 — dung bang muc
    # ma rieng denoise gay ra. Ket qua: quy nham thu pham cho model, va bat
    # `detail_restore` (op nang nhat chuoi, ~8 phut/5MP tren CPU) trong khi theo ghi
    # chep 01/08 no doi lay 0.17->0.42 chi tiet nhung MAT 5.3 diem sang + 1.6 diem mau
    # (vi pham L11). Nay lay moc NGAY SAU denoise de ti so chi phan anh phan MODEL.
    lap_sau_denoise = _do_net(out)
    record["lap_sau_denoise"] = round(lap_sau_denoise, 1)
    out = R["auto_enhance"]["fn"](out, {})
    record["steps"].append({"op": "auto_enhance", "reason": "mau/tone (model CH hien hanh)"})

    d1 = diagnose(out, with_masks=False)     # kham lan 2 (sau model)
    record["diagnosis_mid"] = {k: v for k, v in d1.items() if not k.startswith("_")}

    # 04/08 — QUYET DINH PHUC NET O DAY, khong phai o dau chuoi (xem chu thich tren).
    # Cau hoi dung la "MODEL co lam mem anh khong", chu khong phai "anh vao co sac
    # khong". Do ca hai tren cung mot thang (canh dai 1600) nen so sanh duoc.
    mid_lap = _do_net(out)
    record["mid_lap"] = round(mid_lap, 1)
    # 04/08 (sua lai trong ngay): moc la ban SAU DENOISE, khong phai anh goc — xem
    # chu thich o cho tinh `lap_sau_denoise`.
    _moc = lap_sau_denoise
    record["lap_ratio"] = round(mid_lap / _moc, 3) if _moc > 0 else None
    if _sharpen_ep is None:
        # Nguong 0.90: RIENG MODEL lam mat >10% nang luong canh thi moi phuc net.
        sharpen = mid_lap < _moc * 0.90
        record["steps"].append({
            "op": "quyet-dinh-phuc-net",
            "reason": (f"net anh goc {in_lap:.0f} -> sau denoise {_moc:.0f} "
                       f"-> sau model {mid_lap:.0f} "
                       f"(rieng model {mid_lap / _moc:.2f}x); "
                       + ("MODEL lam mem -> bat detail_restore" if sharpen
                          else "model giu duoc net -> khong phuc net")),
        })
    else:
        sharpen = bool(_sharpen_ep)

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
    # ⚠️ FIX 30/07: DOC THANG PHAI CHAY TRUOC khi tinh mask. Truoc day mask (cua so,
    # vat lieu) tinh tren anh GOC roi ap len anh DA XOAY-ZOOM -> lech 60-480 pixel tren
    # anh 6024px => mau canh ngoai to len TUONG canh cua so ("cua so nhu tranh ve"),
    # cong thuc go to len vat khac ("do vat cai sang cai toi").
    if d0["scene"] in ("interior", "exterior_ground", "general"):
        # FIX 30/07 (dot 2): tinh hinh hoc MOT LAN tren anh GOC roi ap cho ca hai.
        # Goi apply() 2 lan lam analyze() do duong thang tren 2 anh khac tong mau ->
        # 2 goc khac nhau (dich toi 445px, co ca ca 1 ben xoay 1 ben khong).
        from ai_engine.specialists.straighten.straighten import apply as _st
        img, _H = _st(img, {"strength": 1.0, "return_h": True})
        out = _st(out, {"strength": 1.0, "h_override": _H}) if _H is not None else out
        record["steps"].append({"op": "straighten(dau chuoi, H dung chung)",
                                "reason": f"scene={d0['scene']}", "applied": _H is not None})
    _before_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    win_mask_gsam = get_window_mask(_before_u8)
    # ⭐ 04/08 (ra soat vong 7) — CONG LANH MANH CHO MASK "MAT".
    # Grounding DINO co the tra ve mot box phu GAN TRON KHUNG HINH tren anh KHONG co
    # cua so nao (no van co gang khop prompt "window"). Khi do mask phu ~100% va moi
    # thao tac "theo vung cua so" bien thanh thao tac TOAN ANH — dung loai loi ma
    # nguyen tac "tay theo vung" sinh ra de tranh. Nghiem trong hon: nhanh du phong
    # `window_pull` o `elif` ben duoi tro thanh MA CHET, vi mask rong luon khac None.
    # Nguong 35%: cua so that hiem khi chiem qua 1/3 khung anh noi that; qua muc do
    # thi gan nhu chac chan la mat bat nham.
    _mask_bi_bo = False
    if win_mask_gsam is not None:
        _frac = float(win_mask_gsam.mean())
        if _frac > 0.35:
            _mask_bi_bo = True
            record["steps"].append({
                "op": "eye:gsam-window:BO",
                "frac": round(_frac, 4),
                "reason": f"mask phu {_frac*100:.0f}% khung hinh > 35% -> gan nhu chac "
                          f"chan mat bat nham, bo mask va de nhanh du phong chay",
            })
            win_mask_gsam = None
    if win_mask_gsam is not None:
        record["steps"].append({"op": "eye:gsam-window",
                                "frac": round(float(win_mask_gsam.mean()), 4)})
    # ⭐ 04/08 (sua lai trong ngay) — BO DIEU KIEN `not sharp_input`.
    # Do net cua ANH VAO khong lien quan gi den chuyen cua so co can xu ly hay khong.
    # Ma anh giao hang that (bracket da gop, full-res) LUON sac: do duoc 102.0 va
    # 183.8 tren hai anh that, deu >= 80 => `sharp_input` luon True => than nhanh nay
    # CHUA BAO GIO chay tren dung loai anh di giao khach.
    # Hau qua ket hop voi cong 35% them sang nay: khi mat gsam bat nham (mask phu
    # ~100%) thi mask bi bo dung nhu thiet ke, nhung sau do KHONG duong nao xu ly cua
    # so — khong window_pull, khong window_recover — va khong mot dong canh bao.
    # Tuc ban va chan mot loi roi de lai mot lo khac to hon.
    elif _mask_bi_bo or (d0.get("frac_window", -1.0) >= 0.01
                         or d0.get("frac_window", -1.0) < 0.0):
        s = round(min(0.95, 0.5 + max(d0.get("frac_window", 0.1), 0.1) * 2.5), 2)
        out = R["window_pull"]["fn"](out, {"strength": s, "saturation_boost": 0.5})
        record["steps"].append({"op": "window_pull(fallback)", "strength": s})

    # (straighten da chay o DAU chuoi — xem fix 30/07 phia tren)

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
