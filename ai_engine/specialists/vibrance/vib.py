"""vibrance — nâng TRẮNG + VIBRANCE chọn lọc (23/07/2026).

VÌ SAO: chủ dự án khoanh 4 vùng trên k001 (outputs/compare_chf/region_check.jpg):
so target AutoHDR, output model bị (a) tường trắng TỐI hơn ~17 luma ("thiếu sáng"),
(b) màu trang trí (gạch hoa) NHẠT hơn ~37%% saturation. Gốc: loss chống-bạc-màu
kìm kênh sáng + lưới luma không tách "gạch màu" khỏi "tường trắng" cùng độ sáng.
Con này bù deterministic sau model:

  1. whites: nâng vùng sáng theo đường cong tiệm cận (luma' = luma + w*ss*(1-luma))
     — càng gần 1 nâng càng ít, KHÔNG BAO GIỜ clip cứng; vùng tối/midtone không đổi.
  2. vibrance: tăng bão hòa CHỌN LỌC — mạnh với màu no-vừa (gạch hoa ~0.15-0.5),
     gần 0 với trung tính (tường, s<~0.06 — không khuếch màu nhiễu) và giảm dần
     với màu đã no (sàn terracotta) nhờ hệ số (1-s).

Toàn per-pixel, không op không gian -> không halo. whites=vibrance=0 -> identity.
Hợp đồng: apply(img float32 [0,1] HxWx3 BGR, params) -> cùng shape.
"""
import cv2
import numpy as np

cv2.setNumThreads(3)

_LUMA_W = np.array([0.0722, 0.7152, 0.2126], dtype=np.float32)  # BGR

_WHITES_LO, _WHITES_HI = 0.55, 0.92   # vùng luma bắt đầu/đạt max nâng trắng
_WHITES_MAX = 0.55                    # tỷ lệ tối đa của (1-luma) được nâng tại whites=1
_VIB_NEUTRAL_S = 0.06                 # dưới mức này coi là trung tính, không đụng
# Cú hích BÃO HÒA HÌNH CHUÔNG (fit 4 điểm neo đo trên k001 vs target, 23/07 —
# tools/hue_probe.py): đẩy mạnh nhất quanh s~0.23 (màu trang trí no-vừa),
# giảm dần về 2 phía (trung tính không đụng, sàn đã no chỉ +nhẹ đúng gu AutoHDR).
_VIB_BUMP_MU = 0.23
_VIB_BUMP_SIGMA = 0.09
_VIB_GLOBAL_AMP = 0.05                # phần đẩy chung nhẹ (sàn/mảng lớn)
_VIB_ACCENT_AMP = 0.14                # phần cộng thêm cho màu NỔI cục bộ (hoa văn/decor)


def _smoothstep(x, lo, hi):
    t = np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def apply(img, params=None):
    params = params or {}
    whites = float(np.clip(params.get("whites", 0.5), 0.0, 1.0))
    vibrance = float(np.clip(params.get("vibrance", 0.5), 0.0, 1.0))
    dark_clean = float(np.clip(params.get("dark_clean", 0.0), 0.0, 1.0))

    img = np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)
    if whites == 0.0 and vibrance == 0.0 and dark_clean == 0.0:
        return img

    out = img
    if whites > 0.0:
        y = out @ _LUMA_W
        ss = _smoothstep(y, _WHITES_LO, _WHITES_HI)
        y_new = y + whites * _WHITES_MAX * ss * (1.0 - y)
        ratio = np.clip(y_new / np.maximum(y, 1e-4), 1.0, 1.6)
        out = out * ratio[..., None]

    if vibrance > 0.0:
        hsv = cv2.cvtColor(np.clip(out, 0.0, 1.0), cv2.COLOR_BGR2HSV)
        s = hsv[..., 1]
        gate = _smoothstep(s, _VIB_NEUTRAL_S, 0.12)  # tha trung tính (tường/trần)
        bump = np.exp(-((s - _VIB_BUMP_MU) ** 2) / (2.0 * _VIB_BUMP_SIGMA ** 2))
        # ACCENT (23/07): màu NỔI giữa vùng trung tính (hoa văn/đồ decor) được đẩy
        # mạnh; mảng màu LỚN đồng đều (sàn gỗ) chỉ ăn phần global nhẹ — vì cùng
        # mức s mà AutoHDR xử khác nhau tùy ngữ cảnh (đo tools/hue_probe.py).
        h_img, w_img = s.shape
        r = max(24, min(h_img, w_img) // 8)
        s_local = cv2.boxFilter(s, -1, (2 * r + 1, 2 * r + 1))
        accent = _smoothstep(np.clip(s - s_local, 0.0, 1.0), 0.02, 0.12)
        amp = _VIB_GLOBAL_AMP + _VIB_ACCENT_AMP * accent
        hsv[..., 1] = np.clip(s + vibrance * amp * gate * bump, 0.0, 1.0)
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if dark_clean > 0.0:
        # DEN SACH (23/07): vung toi cua AutoHDR TRUNG TINH (sat 32-70) trong khi
        # model tra ra bong toi NAU BUN (sat 67-89, do tren v4) -> "nhat/duc".
        # Ha bao hoa vung toi ve trung tinh, muot theo luma, khong dung midtone.
        hsv = cv2.cvtColor(np.clip(out, 0.0, 1.0), cv2.COLOR_BGR2HSV)
        y = out @ _LUMA_W
        dark_w = 1.0 - _smoothstep(y, 0.15, 0.42)  # phu het dai bong toi do duoc (luma 25-80/255)
        hsv[..., 1] = hsv[..., 1] * (1.0 - dark_clean * 0.55 * dark_w)
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return np.clip(out, 0.0, 1.0).astype(np.float32)
