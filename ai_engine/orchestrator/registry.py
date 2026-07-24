"""REGISTRY: ten op -> {fn, desc, params (schema)}.

Cac con AI (basic + specialist) cam vao day, mien tuan thu hop dong
    apply(img_f32_bgr_01, params) -> img_f32_bgr_01 cung shape.

Schema param ho tro 3 kieu:
  - float: {"type":"float","min":..,"max":..,"default":..}
  - enum : {"type":"enum","choices":[..],"default":..}
  - bool : {"type":"bool","default":..}
"""
from . import ops_basic

# --- Cac specialist THAT (moi con 1 thu muc, da qua conformance_check) ---
from ai_engine.specialists.straighten import straighten as sp_straighten
from ai_engine.specialists.white_balance import wb as sp_wb
from ai_engine.specialists.denoise_sharpen import ds as sp_ds
from ai_engine.specialists.grass_green import grass as sp_grass
from ai_engine.specialists.sky_replace import replace as sp_sky
from ai_engine.specialists.window_pull import pull as sp_window
from ai_engine.specialists.harsh_sun import tone_map as sp_harsh
from ai_engine.specialists.finish_detail import finish as sp_finish
from ai_engine.specialists.vibrance import vib as sp_vib
from ai_engine.specialists.shadow_light import light as sp_shadowlight
from ai_engine.specialists.detail_restore import restore as sp_detail

REGISTRY = {
    # ---- Op co ban (ops_basic) ----
    "brightness": {
        "fn": ops_basic.brightness,
        "desc": "Tang/giam do sang (exposure, gamma-aware).",
        "params": {"amount": {"type": "float", "min": -1.0, "max": 1.0, "default": 0.0}},
    },
    "contrast": {
        "fn": ops_basic.contrast,
        "desc": "Tang/giam tuong phan quanh gia tri trung vi.",
        "params": {"amount": {"type": "float", "min": -1.0, "max": 1.0, "default": 0.0}},
    },
    "saturation": {
        "fn": ops_basic.saturation,
        "desc": "Tang/giam do rue ro (HSV saturation scale).",
        "params": {"amount": {"type": "float", "min": -1.0, "max": 1.0, "default": 0.0}},
    },
    "temperature": {
        "fn": ops_basic.temperature,
        "desc": "Am hon (+) hoac lanh hon (-) (dich kenh R/B).",
        "params": {"amount": {"type": "float", "min": -1.0, "max": 1.0, "default": 0.0}},
    },
    "shadows_lift": {
        "fn": ops_basic.shadows_lift,
        "desc": "Nang chi tiet vung toi.",
        "params": {"amount": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.0}},
    },
    "highlights_recover": {
        "fn": ops_basic.highlights_recover,
        "desc": "Ha vung chay/qua sang.",
        "params": {"amount": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.0}},
    },
    "sharpen": {
        "fn": ops_basic.sharpen,
        "desc": "Phuc net nhe (unsharp mask co ban).",
        "params": {"amount": {"type": "float", "min": 0.0, "max": 0.5, "default": 0.2}},
    },
    "auto_enhance": {
        "fn": ops_basic.auto_enhance,
        "desc": "Tu dong chinh dep toan dien (HDRnet hoc tu data), bo qua neu checkpoint loi/thieu.",
        "params": {},
    },

    # ---- Specialist THAT ----
    "auto_white_balance": {
        "fn": sp_wb.apply,
        "desc": "Can trang lai (khu am vang den/am xanh cua so) + tu dong exposure, giu gu sang-airy BDS.",
        "params": {
            "wb_strength": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.8},
            "target_median": {"type": "float", "min": 0.30, "max": 0.75, "default": 0.42},
        },
    },
    "straighten": {
        "fn": sp_straighten.apply,
        "desc": "Nan cac duong doc ve thang dung (kien truc); tu tra ve nguyen anh neu khong an toan.",
        "params": {
            "strength": {"type": "float", "min": 0.0, "max": 1.0, "default": 1.0},
            "k1": {"type": "float", "min": -0.30, "max": 0.10, "default": 0.0},
        },
    },
    "denoise": {
        "fn": sp_ds.apply,
        "desc": "Khu nhieu vung phang (edge-aware) roi phuc net co gate texture, khong halo.",
        "params": {
            "denoise_strength": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.35},
            "sharpen_amount": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.3},
        },
    },
    "grass_green": {
        "fn": sp_grass.apply,
        "desc": "Lam co/cay xanh tuoi hon CHI trong vung co (mask mem), khong dung noi that/be tong.",
        "params": {"strength": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.7}},
    },
    "sky_replace": {
        "fn": sp_sky.apply,
        "desc": "Thay troi (mask bam mai nha/canh cay + harmonize), tu bo qua neu anh khong co troi.",
        "params": {
            "sky": {"type": "enum", "choices": ["blue", "golden", "dusk", "hazy"], "default": "blue"},
            "strength": {"type": "float", "min": 0.0, "max": 1.0, "default": 1.0},
            "harmonize": {"type": "bool", "default": True},
        },
    },
    "window_pull": {
        "fn": sp_window.apply,
        "desc": "Can sang cua so chay trang: phuc hoi canh ben ngoai (toa nha/troi) thay ro, noi that giu nguyen.",
        "params": {
            "strength": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.7},
            "saturation_boost": {"type": "float", "min": 0.0, "max": 0.6, "default": 0.25},
        },
    },
    "detail_restore": {
        "fn": sp_detail.apply,
        "desc": "Phuc net/chi tiet THAT bang Real-ESRGAN general (BSD-3, ban duoc). CHAM tren CPU — bat theo y (HD/premium) hoac chay GPU khi giao lo.",
        "params": {"strength": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5}},
    },
    "shadow_light": {
        "fn": sp_shadowlight.apply,
        "desc": "Thap sang vung toi/goc khuat kieu flambient (giu den that, khong lech mau) — tri 'goc khuat khong duoc chinh'.",
        "params": {"amount": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.6}},
    },
    "vibrance": {
        "fn": sp_vib.apply,
        "desc": "Nang trang (khong clip) + vibrance chon loc (day mau no-vua, tha trung tinh/da-no) — bu 'nhat mau, thieu sang' sau model.",
        "params": {
            "whites": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
            "vibrance": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
            "dark_clean": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.0},
        },
    },
    "finish_detail": {
        "fn": sp_finish.apply,
        "desc": "Hoan thien: phuc net + vi tuong phan (guided filter, khong halo) + diem den sau — tri 'anh mo bot'.",
        "params": {
            "clarity": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
            "detail": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.6},
            "black": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.35},
        },
    },
    "harsh_sun": {
        "fn": sp_harsh.apply,
        "desc": "Nen dai sang cho anh nang gat (keo lai vung chay + mo bong sau) giu mau giau, khong halo.",
        "params": {
            "strength": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.7},
            "highlight_recover": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.8},
            "shadow_lift": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
            "local_contrast": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.3},
            "sat_restore": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.4},
        },
    },
}


def get_registry_summary():
    """List dict mo ta op + schema, dung cho system prompt cua planner."""
    return [{"op": name, "desc": e["desc"], "params": e["params"]} for name, e in REGISTRY.items()]


def clamp_params(op_name, params):
    """Chuan hoa params ve dung schema (float clamp / enum whitelist / bool coerce), dien default."""
    schema = REGISTRY[op_name]["params"]
    clamped = {}
    for pname, pschema in schema.items():
        ptype = pschema.get("type", "float")
        val = params.get(pname, pschema["default"])
        if ptype == "float":
            try:
                val = float(val)
            except (TypeError, ValueError):
                val = pschema["default"]
            val = max(pschema["min"], min(pschema["max"], val))
        elif ptype == "enum":
            if val not in pschema["choices"]:
                val = pschema["default"]
        elif ptype == "bool":
            val = bool(val)
        clamped[pname] = val
    return clamped
