"""material_grade — CHỈNH THEO CHẤT LIỆU TỪNG MÓN ĐỒ (24/07/2026).

VÌ SAO: chủ dự án chốt sau 8 vòng: "AutoHDR mọi đồ vật đều xử lý rất tốt;
ta thì mọi ảnh một bệnh lặp lại — sửa toàn bộ bức ảnh". Model màu (lưới 24×24)
không thể chỉnh vật < ~85px. Tầng này dùng mắt-150-lớp + tay-theo-vùng để mỗi
nhóm chất liệu nhận đúng công thức của nó.

CÔNG THỨC (deterministic, liều bảo thủ, có KHÓA MÀU kép để không nhuộm nhầm):
  dark_appliance (lò/TV/tủ lạnh): đen sâu + trung tính + bóng  — dark_clean cao, black cao
  wood (bàn/mặt bếp/tủ ngăn kéo): giữ ấm + vân nét — CHỈ nơi pixel đang ấm sẵn
  fabric (sofa/gối/rèm/thảm): màu vải sống lại — saturation nhẹ
  plant (cây/hoa): tươi — dùng grass_green sẵn có, mask-scope
  fixture_white (sink/bồn): trắng sạch sáng

An toàn: mask mềm + feather (region_apply), mọi op vẫn qua clamp; nhóm không
xuất hiện trong ảnh (mask ~0) tự bỏ qua, chi phí ~0.
"""
import cv2
import numpy as np

cv2.setNumThreads(3)

from ai_engine.orchestrator.registry import REGISTRY
from ai_engine.orchestrator.region_apply import region_apply
from ai_engine.specialists.segment_room.seg import segment_fine

_MIN_FRAC = 0.005  # nhom < 0.5% khung hinh -> bo qua


def _warm_gate(img):
    """Mask [0,1]: pixel dang co mau AM (hue cam-do, sat du) — khoa cho cong thuc go."""
    hsv = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    h, s = hsv[..., 0].astype(np.float32), hsv[..., 1].astype(np.float32) / 255.0
    hue_ok = ((h >= 5) & (h <= 30)).astype(np.float32)
    sat_ok = np.clip((s - 0.12) / 0.10, 0, 1)
    return cv2.GaussianBlur(hue_ok * sat_ok, (0, 0), 9)


def apply_material_grade(img, mats=None, record=None):
    """img float32 [0,1] BGR. mats: dict mask tu segment_fine (None -> tu goi).
    Tra anh da chinh theo chat lieu; record (list) duoc ghi buoc + ly do."""
    R = REGISTRY
    if mats is None:
        mats = segment_fine(img)
    out = img
    log = record if record is not None else []

    def frac(m):
        return float(m.mean())

    m = mats.get("dark_appliance")
    if m is not None and frac(m) > _MIN_FRAC:
        def _appliance(x, p):
            y = R["vibrance"]["fn"](x, {"whites": 0.0, "vibrance": 0.0, "dark_clean": 0.8})
            return R["finish_detail"]["fn"](y, {"clarity": 0.5, "detail": 0.6, "black": 0.9})
        out = region_apply(out, _appliance, {}, m, feather_sigma=8)
        log.append({"op": "material:dark_appliance", "frac": round(frac(m), 3),
                    "reason": "lo/TV/tu lanh -> den sau trung tinh bong"})

    m = mats.get("wood")
    if m is not None and frac(m) > _MIN_FRAC:
        gate = _warm_gate(out)
        m_gated = np.clip(m * gate, 0.0, 1.0)
        if frac(m_gated) > _MIN_FRAC / 2:
            def _wood(x, p):
                # 25/07 ha lieu (chu che "van go khong tu nhien"): am nhe, van net
                y = R["temperature"]["fn"](x, {"amount": 0.05})
                y = R["saturation"]["fn"](y, {"amount": 0.08})
                return R["finish_detail"]["fn"](y, {"clarity": 0.4, "detail": 0.8, "black": 0.0})
            out = region_apply(out, _wood, {}, m_gated, feather_sigma=14)
            log.append({"op": "material:wood", "frac": round(frac(m_gated), 3),
                        "reason": "go dang am -> am NHE + van net (lieu tu nhien)"})

    m = mats.get("window_glass")
    if m is not None and frac(m) > 0.01:
        def _winview(x, p):
            # khu-mu + nen nang gat + TO TROI trong khung kinh (25/07 vong cham 10)
            y = R["highlights_recover"]["fn"](x, {"amount": 0.35})       # nen nang phan chieu
            y = R["finish_detail"]["fn"](y, {"clarity": 0.7, "detail": 0.5, "black": 0.4})
            y = R["saturation"]["fn"](y, {"amount": 0.18})
            # to troi: pixel XANH DUONG sang trong kinh -> dam mau, ha sang nhe
            hsv = cv2.cvtColor((np.clip(y, 0, 1) * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
            hch = hsv[..., 0].astype(np.float32)
            v = hsv[..., 2].astype(np.float32) / 255.0
            skyish = ((hch >= 95) & (hch <= 135) & (v > 0.55)).astype(np.float32)
            skyish = cv2.GaussianBlur(skyish, (0, 0), 5)
            hsv2 = hsv.astype(np.float32)
            hsv2[..., 1] = np.clip(hsv2[..., 1] + skyish * 60, 0, 255)   # dam mau troi
            hsv2[..., 2] = np.clip(hsv2[..., 2] * (1.0 - skyish * 0.10), 0, 255)
            return cv2.cvtColor(hsv2.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
        out = region_apply(out, _winview, {}, m, feather_sigma=6)
        log.append({"op": "material:window_view", "frac": round(frac(m), 3),
                    "reason": "khung kinh: nen nang + khu mu + TO TROI dam"})

    # 25/07: TAT MAC DINH — ban blob-fill ra mieng den lech/nham (tv4_j055),
    # can quad-fit hinh chu nhat man hinh moi ship duoc. Bat lai khi lam xong.
    _ENABLE_SCREEN_FILL = False
    m = mats.get("screen")
    if _ENABLE_SCREEN_FILL and m is not None and frac(m) > 0.004:
        def _screen_fill(x, p):
            """LAP MAN TV (roster nhom D, 25/07): man dang phan chieu sang ->
            den tat nhu AutoHDR. Giu 12%% shading goc lam do bong glossy,
            khu mau ve trung tinh."""
            y = x @ np.array([0.0722, 0.7152, 0.2126], dtype=np.float32)
            gray = (y * 0.12)[..., None]                       # toi ~ man tat
            gloss = np.clip(y - np.percentile(y, 85), 0, 1)[..., None] * 0.25
            return np.clip(gray + gloss, 0.0, 1.0).astype(np.float32).repeat(3, axis=2)
        # mask cung + KHOA HINH HOC (25/07: mask tung bam nham vung kinh phan
        # chieu -> lap den thanh vet loang; chi chap nhan khoi GON, CHU NHAT,
        # ty le man hinh 1.1-2.6, dac >=60%)
        m_hard = (m > 0.30).astype(np.uint8)
        m_hard = cv2.erode(m_hard, np.ones((5, 5), np.uint8))
        m_ok = np.zeros_like(m_hard, dtype=np.float32)
        n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(m_hard, 8)
        kept = 0
        for i in range(1, n_lbl):
            x, y, bw, bh, area = stats[i]
            if area < 0.0015 * m_hard.size:
                continue
            solidity = area / max(bw * bh, 1)
            aspect = bw / max(bh, 1)
            if solidity >= 0.60 and 1.1 <= aspect <= 2.6:
                m_ok[lbl == i] = 1.0
                kept += 1
        if kept > 0:
            out = region_apply(out, _screen_fill, {}, m_ok, feather_sigma=3)
            log.append({"op": "material:screen_fill", "frac": round(float(m_ok.mean()), 3),
                        "reason": f"man TV that ({kept} khoi qua khoa hinh hoc) -> lap den glossy"})

    m = mats.get("art")
    if m is not None and frac(m) > 0.003:
        def _art(x, p):
            # tranh/poster/guong: tuong phan cuc bo + giu highlight nang (khong nen)
            return R["finish_detail"]["fn"](x, {"clarity": 0.7, "detail": 0.7, "black": 0.5})
        out = region_apply(out, _art, {}, m, feather_sigma=5)
        log.append({"op": "material:art", "frac": round(frac(m), 3),
                    "reason": "tranh/TV treo -> tuong phan anh nang ro"})

    m = mats.get("fabric")
    if m is not None and frac(m) > _MIN_FRAC:
        out = region_apply(out, R["saturation"]["fn"], {"amount": 0.14}, m, feather_sigma=12)
        log.append({"op": "material:fabric", "frac": round(frac(m), 3),
                    "reason": "vai/sofa/rem -> mau song lai"})

    m = mats.get("plant")
    if m is not None and frac(m) > _MIN_FRAC:
        def _plant(x, p):
            # 25/07 v2: nang sang + LAM MAT nhe (chu che "cay con vang") + to xanh
            y = R["brightness"]["fn"](x, {"amount": 0.22})
            y = R["temperature"]["fn"](y, {"amount": -0.05})
            return R["grass_green"]["fn"](y, {"strength": 0.45})
        out = region_apply(out, _plant, {}, m, feather_sigma=10)
        log.append({"op": "material:plant", "frac": round(frac(m), 3),
                    "reason": "cay/hoa -> sang len + xanh tuoi (lieu xanh ha 0.6->0.45 chong neon)"})

    m = mats.get("fixture_white")
    if m is not None and frac(m) > _MIN_FRAC:
        out = region_apply(out, R["vibrance"]["fn"],
                           {"whites": 0.7, "vibrance": 0.0, "dark_clean": 0.3}, m, feather_sigma=10)
        log.append({"op": "material:fixture_white", "frac": round(frac(m), 3),
                    "reason": "sink/bon -> trang sach sang"})
    return out
