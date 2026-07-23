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


def process(img):
    R = REGISTRY
    record = {"steps": []}

    d0 = diagnose(img)                       # kham lan 1 (anh goc, co mat segment)
    record["diagnosis_before"] = {k: v for k, v in d0.items() if not k.startswith("_")}
    masks = d0.get("_masks")

    out = R["denoise"]["fn"](img, {"denoise_strength": 0.35, "sharpen_amount": 0.0})
    record["steps"].append({"op": "denoise", "reason": "khu nhieu truoc model"})
    out = R["auto_enhance"]["fn"](out, {})
    record["steps"].append({"op": "auto_enhance", "reason": "mau/tone (model CH hien hanh)"})

    d1 = diagnose(out, with_masks=False)     # kham lan 2 (sau model)
    record["diagnosis_mid"] = {k: v for k, v in d1.items() if not k.startswith("_")}

    if masks is not None:
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
        if need > 0.05:
            amt = round(0.5 + 0.5 * need, 2)   # 25/07: nang cong thuc (san con toi nhe)
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
        # rua bun: chi kien truc (do vat/go giu mau)
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

    fw = d0.get("frac_window", -1.0)
    if fw >= 0.01 or fw < 0.0:
        s = round(min(0.95, 0.5 + max(fw, 0.1) * 2.5), 2)
        out = R["window_pull"]["fn"](out, {"strength": s, "saturation_boost": 0.5})
        record["steps"].append({"op": "window_pull", "strength": s,
                                "reason": f"mat thay cua so {max(fw,0):.0%}"})

    if d0["scene"] in ("interior", "exterior_ground", "general"):
        out = R["straighten"]["fn"](out, {"strength": 1.0})
        record["steps"].append({"op": "straighten", "reason": f"scene={d0['scene']}"})

    out = R["finish_detail"]["fn"](out, {"clarity": 0.8, "detail": 1.0, "black": 0.7})
    record["steps"].append({"op": "finish_detail", "reason": "net + den sau chot"})

    # TANG CHAT LIEU (25/07): mask tinh tren ANH GOC (mat nhin truoc khi model
    # lam sang — TV loa sau model bi mat dau, bug bat duoc 25/07), ap len ket qua.
    try:
        from ai_engine.specialists.segment_room.seg import segment_fine
        from ai_engine.brain.material_grade import apply_material_grade
        mats = segment_fine(img)
        mat_log = []
        out = apply_material_grade(out, mats=mats, record=mat_log)
        record["steps"].extend(mat_log)
    except Exception as e:
        record["steps"].append({"op": "material:SKIP", "reason": str(e)[:100]})
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
