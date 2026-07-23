"""FULL-RES DELIVERY PIPELINE (Task 21).

Fix loi khach hang #1: output bi nen/mem. Nguyen tac (CLAUDE.md):
  - MASTER full-res KHONG BAO GIO bi resize. Moi op ap truc tiep len master.
  - Uoc luong THAM SO (WB gains, tone amounts) chay tren PROXY nho — dung
    kien truc "operator khong pixel": proxy -> params -> ap len master.
  - Export JPEG quality 100 (hoac PNG neu duoi file .png). Chi nen 8-bit
    dung 1 lan o buoc ghi file cuoi.

Chain mac dinh (khong co auto_enhance — model con washout, chi bat qua
opts["use_model"]):
  auto_white_balance -> denoise (nhe) -> straighten (bo qua aerial)
  -> highlights_recover + shadows_lift (adaptive theo dynamic range)
  -> saturation (nhe) -> sharpen (nhe, chong halo qua ds.sharpen)

CLI:
  python -m ai_engine.delivery.deliver --in <file|dir> --out <file|dir> [--png] [--use-model]
"""

import argparse
import os
import time

import cv2
import numpy as np

cv2.setNumThreads(3)

from ai_engine.orchestrator.registry import REGISTRY, clamp_params
from ai_engine.specialists.white_balance import wb as wb_mod
from ai_engine.specialists.scene_classify import classify as scene_mod
from ai_engine.specialists.harsh_sun import tone_map as harsh_mod  # noqa: F401 (via registry)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PROXY_DIM = 1024

# Folder cohesion: gain WB tung anh bi kep ve [ref - DELTA, ref + DELTA]
# quanh gain tham chieu cua listing (median tren subset dai dien).
WB_GAIN_CLAMP_DELTA = 0.10
WB_STRENGTH = 0.8
TARGET_MEDIAN = 0.42

# Adaptive tone: amount ti le voi phan tram pixel chay/ toi, co tran an toan.
HL_LUMA_THRESH = 0.90
SH_LUMA_THRESH = 0.15
HL_SCALE, HL_MAX = 3.0, 0.40
SH_SCALE, SH_MAX = 2.5, 0.30

SATURATION_AMOUNT = 0.12
DENOISE_STRENGTH = 0.25
SHARPEN_AMOUNT = 0.30


def _luma(img):
    return 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]


def _proxy(img, dim=PROXY_DIM):
    """Ban sao nho CHI de uoc luong tham so. Master khong bao gio di qua day."""
    h, w = img.shape[:2]
    scale = dim / max(h, w)
    if scale >= 1.0:
        return img
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _is_aerial(path):
    """Heuristic don gian: anh drone DJI dat ten DJI_*. Du an chua co scene
    classifier (roster nhom A) — khi co thi thay heuristic nay."""
    return os.path.basename(str(path)).upper().startswith("DJI")


def _run_op(img, op_name, params):
    """Ap 1 op tu registry, params LUON qua clamp_params (khong bao gio rac)."""
    entry = REGISTRY[op_name]
    return entry["fn"](img, clamp_params(op_name, params))


def _estimate_wb_gains(img_full):
    """Uoc luong gain WB tren proxy (re, ket qua ~ giong het full-res)."""
    return wb_mod.estimate_wb_gains(_proxy(img_full))


def _clamp_gains_to_ref(gains, ref, delta=WB_GAIN_CLAMP_DELTA):
    out = {}
    for ch in ("b", "g", "r"):
        out[ch] = float(np.clip(gains[ch], ref[ch] - delta, ref[ch] + delta))
    return out


def _estimate_tone(img_full):
    """Adaptive tone tren proxy CUA ANH DA can trang/exposure: amount ti le
    voi ty le pixel chay (luma>0.90) va pixel toi (luma<0.15)."""
    lum = _luma(_proxy(img_full))
    hi_frac = float(np.mean(lum > HL_LUMA_THRESH))
    lo_frac = float(np.mean(lum < SH_LUMA_THRESH))
    highlights = float(np.clip(hi_frac * HL_SCALE, 0.0, HL_MAX))
    shadows = float(np.clip(lo_frac * SH_SCALE, 0.0, SH_MAX))
    return highlights, shadows


def deliver_image(in_path, out_path, scene=None, opts=None):
    """Doc anh GOC full-res -> ap chain operator full-res -> ghi JPEG q100
    (hoac PNG neu out_path .png). Output BAT BUOC cung kich thuoc input.

    opts:
      use_model     : bool — them auto_enhance cuoi chain (mac dinh TAT, model washout).
      wb_ref_gains  : dict {b,g,r} — gain tham chieu cua listing (folder mode);
                      gain anh nay bi kep ve ref +- WB_GAIN_CLAMP_DELTA.
    """
    opts = opts or {}
    t0 = time.time()

    img_u8 = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
    if img_u8 is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {in_path}")
    in_shape = img_u8.shape[:2]
    img = img_u8.astype(np.float32) / 255.0

    scene_conf = 1.0
    if scene is None:
        try:
            _cls = scene_mod.classify(_proxy(img))
            scene = _cls.get("scene", "general")
            scene_conf = float(_cls.get("confidence", 0.0))
        except Exception:
            scene = "aerial" if _is_aerial(in_path) else "general"
            scene_conf = 0.0

    # 24/07: phan loai KHONG CHAC (conf<0.5) thi khong duoc quyen chan window_pull —
    # k001 (bep!) bi gan exterior_ground conf 0.296 -> mat window_pull -> chu che
    # "cua so khong thay cay". window_pull tu gate bang mask noi bo cua no.
    allow_window_pull = (scene in ("interior", "general")) or scene_conf < 0.5

    applied = []

    # ===== DUONG MODEL (23/07): model hoc anh THO -> AutoHDR, nen phai an anh
    # tho TRUOC. Chuoi cu (wb->tone->saturation roi moi model) = double-processing
    # -> mau NHAT + vung den posterize xanh (chu du an che 23/07, bang chung
    # outputs/compare_chf/stove_stages.jpg). Model lo mau/tone; cac op sau chi lam
    # thu model khong lam: cua so, nan doc, phuc net. =====
    if opts.get("use_model"):
        img = _run_op(img, "denoise", {"denoise_strength": DENOISE_STRENGTH, "sharpen_amount": 0.0})
        applied.append(("denoise", {"denoise_strength": DENOISE_STRENGTH}))
        img = _run_op(img, "auto_enhance", {})
        applied.append(("auto_enhance", {}))
        # Thap sang goc khuat (flambient) — cai "nhat" chu/khach thay thuc chat la
        # THIEU SANG vung toi, khong phai thieu mau (chroma Lab ta >= target, do 23/07).
        # 24/07: ha lua ca cum op (0.7->0.35...) — vong 5 chu cham: 3 op cung day
        # sang chong nhau -> "sang ma mo duc", goc o vang, cua so chay. Muc nhe =
        # sach truoc, phan con lai de CH_G hoc.
        img = _run_op(img, "shadow_light", {"amount": 0.35})
        applied.append(("shadow_light", {"amount": 0.35}))
        # Bu 2 diem model con hut so AutoHDR (do 23/07, 4 vung chu khoanh):
        # tuong trang hoi toi + mau decor nhat. vibrance = nang trang khong clip
        # + day mau chon loc (accent-aware, san lon giu nha).
        img = _run_op(img, "vibrance", {"whites": 0.45, "vibrance": 0.7, "dark_clean": 0.35})
        applied.append(("vibrance", {"whites": 0.45, "vibrance": 0.7, "dark_clean": 0.35}))
        if allow_window_pull:
            img = _run_op(img, "window_pull", {"strength": 0.9, "saturation_boost": 0.45})
            applied.append(("window_pull", {"strength": 0.9}))
        if scene in ("interior", "exterior_ground", "general"):
            img = _run_op(img, "straighten", {"strength": 1.0})
            applied.append(("straighten", {"strength": 1.0}))
        img = _run_op(img, "finish_detail", {"clarity": 0.8, "detail": 1.0, "black": 0.75})
        applied.append(("finish_detail", {"clarity": 0.8, "detail": 1.0, "black": 0.75}))

        out_shape = img.shape[:2]
        assert out_shape == in_shape, (
            f"VI PHAM: kich thuoc doi {in_shape} -> {out_shape} ({in_path})")
        return _save_output(img, in_path, out_path, in_shape, applied, scene, t0)

    # 1) Can trang + auto exposure. Gain uoc luong tren proxy; folder mode kep
    #    gain ve tham chieu listing de ca bo anh dong mau.
    gains = _estimate_wb_gains(img)
    ref = opts.get("wb_ref_gains")
    if ref is not None:
        gains = _clamp_gains_to_ref(gains, ref)
    img = wb_mod.apply_wb(img, gains, WB_STRENGTH)
    # auto_exposure 1 lan bi ket gamma clamp (0.6) voi anh RAT toi -> median dung
    # duoi target, listing khong dong deu. Lap toi da 3 lan den khi median vao
    # vung target (gain stretch lan 2+ ~1.0 nen chi gamma tac dung — an toan).
    for _ in range(3):
        img, _ = wb_mod.auto_exposure(img, {"exposure": "auto", "target_median": TARGET_MEDIAN})
        if float(np.median(_luma(_proxy(img)))) >= TARGET_MEDIAN - 0.05:
            break
    applied.append(("auto_white_balance", {"gains": {k: round(v, 3) for k, v in gains.items()}}))

    # 1b) Cua so chay trang (chi interior): phuc hoi ngoai canh NGAY SAU exposure,
    #     TRUOC cac op tone — de auto_enhance/finish_detail hoa mau phan da cuu.
    #     window_pull tu gate (khong cua so / mask nghi ngo -> tra nguyen anh).
    #     Tri "canh ngoai cua so trang bech" (khach che 14/07 "lay cua so chua dep";
    #     doi chieu target AutoHDR 22/07: ho giu troi xanh/thanh pho qua kinh).
    #     Probe 22/07 (outputs/deliver_v2/probe_wp_cmp.jpg): .9/.45 cho troi XANH +
    #     gach do ro, ngang muc 1.0/.6 (da bao hoa) — chon .9/.45 con du phong.
    if allow_window_pull:
        img = _run_op(img, "window_pull", {"strength": 0.9, "saturation_boost": 0.45})
        applied.append(("window_pull", {"strength": 0.9}))

    # 2) Khu nhieu nhe (chua sharpen — sharpen la buoc cuoi).
    img = _run_op(img, "denoise", {"denoise_strength": DENOISE_STRENGTH, "sharpen_amount": 0.0})
    applied.append(("denoise", {"denoise_strength": DENOISE_STRENGTH}))

    # 3) Doc thang — chi anh co truc doc (noi that / mat tien ngoai troi).
    #    Bo qua aerial (goc drone hoi tu la dung) + unknown (khong chac).
    if scene in ("interior", "exterior_ground", "general"):
        img = _run_op(img, "straighten", {"strength": 1.0})
        applied.append(("straighten", {"strength": 1.0}))

    # 4) Tone theo dynamic range thuc te sau WB/exposure.
    #    Dai sang RONG (nang gat/cua so chay) -> harsh_sun (Durand tone-map,
    #    giu mau giau, khong halo). Binh thuong -> adaptive nhe.
    highlights, shadows = _estimate_tone(img)
    high_dr = highlights > 0.05 or shadows > 0.12
    if high_dr:
        hs = float(np.clip(max(highlights / HL_MAX, shadows / SH_MAX), 0.4, 0.9))
        img = _run_op(img, "harsh_sun", {"strength": hs, "highlight_recover": 0.8,
                                         "shadow_lift": 0.5, "sat_restore": 0.5})
        applied.append(("harsh_sun", {"strength": round(hs, 2)}))
    else:
        if highlights > 0.01:
            img = _run_op(img, "highlights_recover", {"amount": highlights})
            applied.append(("highlights_recover", {"amount": round(highlights, 3)}))
        if shadows > 0.01:
            img = _run_op(img, "shadows_lift", {"amount": shadows})
            applied.append(("shadows_lift", {"amount": round(shadows, 3)}))

    # 5) Mau ruc nhe.
    img = _run_op(img, "saturation", {"amount": SATURATION_AMOUNT})
    applied.append(("saturation", {"amount": SATURATION_AMOUNT}))

    # 6) Model mau/tone (CH_E) TRUOC buoc hoan thien net — tone xong moi phuc net.
    if opts.get("use_model"):
        img = _run_op(img, "auto_enhance", {})
        applied.append(("auto_enhance", {}))

    # 7) Hoan thien: phuc net + vi tuong phan + diem den (finish_detail, guided
    #    filter khong halo). Thay cho sharpen nhe cu — tri dut diem "anh mo bot"
    #    (khach che 14/07, 16/07; chan doan outputs/diagnose_blur 22/07).
    img = _run_op(img, "finish_detail", {"clarity": 0.8, "detail": 1.0, "black": 0.5})
    applied.append(("finish_detail", {"clarity": 0.8, "detail": 1.0, "black": 0.5}))

    out_shape = img.shape[:2]
    assert out_shape == in_shape, (
        f"VI PHAM: kich thuoc doi {in_shape} -> {out_shape} ({in_path})")

    return _save_output(img, in_path, out_path, in_shape, applied, scene, t0)


def _save_output(img, in_path, out_path, in_shape, applied, scene, t0):
    """Xuat full-res JPEG q100 4:4:4 (hoac PNG) + tra metadata. Dung chung cho
    ca duong model lan duong deterministic."""
    out_shape = img.shape[:2]
    out_u8 = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)

    out_dir = os.path.dirname(str(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ext = os.path.splitext(str(out_path))[1].lower()
    if ext == ".png":
        ok = cv2.imwrite(str(out_path), out_u8)
    else:
        # q100 + chroma 4:4:4 (mac dinh cv2 la 4:2:0 ke ca q100 -> nhoe canh mau)
        ok = cv2.imwrite(str(out_path), out_u8, [
            cv2.IMWRITE_JPEG_QUALITY, 100,
            cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
        ])
    if not ok:
        raise IOError(f"Khong ghi duoc anh: {out_path}")

    return {
        "in_path": str(in_path),
        "out_path": str(out_path),
        "in_shape": in_shape,
        "out_shape": out_shape,
        "out_bytes": os.path.getsize(str(out_path)),
        "scene": scene,
        "applied": applied,
        "secs": time.time() - t0,
    }


def _list_images(in_dir):
    names = sorted(os.listdir(in_dir))
    return [os.path.join(in_dir, n) for n in names
            if os.path.splitext(n)[1].lower() in IMG_EXTS]


def estimate_listing_wb(files, max_subset=7):
    """Gain WB tham chieu cua ca listing: median gain tren subset dai dien
    (toi da max_subset anh cach deu). Uoc luong tren proxy — chi doc/decode,
    khong dung den master."""
    if not files:
        raise ValueError("Folder khong co anh nao.")
    step = max(1, len(files) // max_subset)
    subset = files[::step][:max_subset]
    gains_b, gains_r = [], []
    for f in subset:
        img_u8 = cv2.imread(f, cv2.IMREAD_COLOR)
        if img_u8 is None:
            continue
        g = wb_mod.estimate_wb_gains(_proxy(img_u8.astype(np.float32) / 255.0))
        gains_b.append(g["b"])
        gains_r.append(g["r"])
    if not gains_b:
        raise ValueError("Khong doc duoc anh nao trong subset dai dien.")
    return {"b": float(np.median(gains_b)), "g": 1.0, "r": float(np.median(gains_r))}


def deliver_folder(in_dir, out_dir, opts=None):
    """Xu ly ca listing. Dong bo mau: gain WB tung anh bi kep ve median cua
    subset dai dien (+- WB_GAIN_CLAMP_DELTA) — chon per-image-clamped thay vi
    1 gain chung cho tat ca vi listing tron ca noi that (den vang) lan aerial
    (anh sang troi): 1 gain cung se sai it nhat 1 nhom. Exposure chuan hoa
    per-image ve cung target_median (0.42) -> do sang dong deu."""
    opts = dict(opts or {})
    files = _list_images(in_dir)
    if not files:
        raise ValueError(f"Khong tim thay anh trong {in_dir}")

    ref = estimate_listing_wb(files)
    opts["wb_ref_gains"] = ref
    print(f"[deliver] {len(files)} anh, gain WB tham chieu listing: "
          f"b={ref['b']:.3f} r={ref['r']:.3f} (kep +-{WB_GAIN_CLAMP_DELTA})")

    os.makedirs(out_dir, exist_ok=True)
    out_ext = ".png" if opts.get("png") else ".jpg"

    results = []
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        out_path = os.path.join(out_dir, stem + out_ext)
        info = deliver_image(f, out_path, opts=opts)
        h_in, w_in = info["in_shape"]
        h_out, w_out = info["out_shape"]
        print(f"  {os.path.basename(f)}: in {w_in}x{h_in} -> out {w_out}x{h_out}, "
              f"{info['out_bytes'] / 1024.0:.0f} KB, {info['secs']:.1f}s, scene={info['scene']}")
        results.append(info)
    return results


def main():
    ap = argparse.ArgumentParser(description="Full-res delivery pipeline (Task 21)")
    ap.add_argument("--in", dest="inp", required=True, help="File anh hoac folder listing")
    ap.add_argument("--out", dest="out", required=True, help="File hoac folder output")
    ap.add_argument("--png", action="store_true", help="Xuat PNG thay vi JPEG q100")
    ap.add_argument("--use-model", action="store_true",
                    help="Them auto_enhance (model hoc) — mac dinh TAT vi con washout")
    args = ap.parse_args()

    opts = {"png": args.png, "use_model": args.use_model}

    if os.path.isdir(args.inp):
        results = deliver_folder(args.inp, args.out, opts)
        total = sum(r["secs"] for r in results)
        print(f"[deliver] XONG {len(results)} anh, tong {total:.1f}s")
    else:
        out_path = args.out
        if args.png and os.path.splitext(out_path)[1].lower() != ".png":
            out_path = os.path.splitext(out_path)[0] + ".png"
        info = deliver_image(args.inp, out_path, opts=opts)
        h_in, w_in = info["in_shape"]
        h_out, w_out = info["out_shape"]
        print(f"[deliver] {args.inp}: in {w_in}x{h_in} -> out {w_out}x{h_out}, "
              f"{info['out_bytes'] / 1024.0:.0f} KB, {info['secs']:.1f}s, "
              f"ops={[a[0] for a in info['applied']]}")


if __name__ == "__main__":
    main()
