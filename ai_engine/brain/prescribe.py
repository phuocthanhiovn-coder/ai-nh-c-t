"""brain.prescribe — KÊ TOA riêng từng ảnh từ bệnh án (não v1: luật rõ ràng).

Khác chuỗi cứng cũ (deliver.py): tham số từng op được SCALE theo số đo của
chính ảnh đó, và MỖI dòng toa kèm LÝ DO — chủ dự án chê ảnh nào là truy được
"vì sao não quyết thế" ngay.

Trả: {"plan": [{"op","params","reason"}...], "notes": [...]}.
Engine thi hành vẫn clamp mọi param (van an toàn giữ nguyên).
"""


def prescribe(d):
    plan = []
    notes = []

    def add(op, params, reason):
        plan.append({"op": op, "params": params, "reason": reason})

    add("denoise", {"denoise_strength": 0.35, "sharpen_amount": 0.0},
        "khu nhieu nhe truoc model")

    add("auto_enhance", {}, "mau/tone chinh (model CH hien hanh, an anh tho)")

    # thap sang goc: scale theo dien tich vung toi thuc do
    dark_need = max(0.0, min(1.0, (d["dark_frac"] - 0.03) / 0.25))
    if dark_need > 0.05:
        amt = round(0.15 + 0.45 * dark_need, 2)
        add("shadow_light", {"amount": amt},
            f"vung toi chiem {d['dark_frac']:.0%} -> san sang muc {amt}")
    else:
        notes.append("anh du sang deu — bo qua shadow_light")

    # vibrance: dark_clean chi khi bong toi thuc su ban (sat cao)
    dc = 0.0
    if d["dark_sat"] > 55:
        dc = round(min(0.7, (d["dark_sat"] - 45) / 50), 2)
    add("vibrance", {"whites": 0.45, "vibrance": 0.7, "dark_clean": dc},
        f"trang+mau accent; dark_clean={dc} (bun vung toi do duoc {d['dark_sat']:.0f})")

    # window_pull: quyet bang MAT (mask cua so) thay vi doan theo scene
    fw = d.get("frac_window", -1.0)
    if fw >= 0.01:
        strength = round(min(0.95, 0.5 + fw * 2.5), 2)
        add("window_pull", {"strength": strength, "saturation_boost": 0.45},
            f"mat thay cua so {fw:.0%} khung hinh -> pull muc {strength}")
    elif fw < 0.0 and d["scene"] in ("interior", "general"):
        add("window_pull", {"strength": 0.9, "saturation_boost": 0.45},
            "mat segment loi — fallback theo scene")
    else:
        notes.append(f"khong thay cua so (mask {fw:.1%}) — bo window_pull")

    if d["scene"] in ("interior", "exterior_ground", "general"):
        add("straighten", {"strength": 1.0}, f"scene={d['scene']} -> nan doc")
    else:
        notes.append(f"scene={d['scene']} — khong nan doc (aerial/khong chac)")

    add("finish_detail", {"clarity": 0.8, "detail": 1.0, "black": 0.75},
        "phuc net + vi tuong phan + diem den (chot)")

    return {"plan": plan, "notes": notes}
