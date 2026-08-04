"""shadow_light v2 — SAN PHẲNG ÁNH SÁNG THEO VÙNG kiểu flambient (23/07/2026).

TIẾN HÓA TỪ v1 (nâng theo band luma pixel): chủ dự án vẫn thấy "chỗ sáng chỗ
tối" trong khi AutoHDR "mọi góc đều sáng". Đo 23/07 (tools/learn_tone_curve.py,
48 cặp): phân bố luma TOÀN CỤC của model đã trùng target (lệch ±7/255) — khác
biệt là KHÔNG GIAN: vùng tối của họ chỉ nằm trên đồ vật đen thật, của ta loang
thành mảng phòng. Phải san lớp CHIẾU SÁNG (illumination), không phải curve pixel.

CÁCH LÀM (Retinex/Durand — cùng họ với harsh_sun nhưng chiều ngược):
  1. log-luma -> tách BASE (chiếu sáng vùng, guided filter bán kính ~6%% cạnh
     ngắn — edge-aware nên không halo) và DETAIL (kết cấu).
  2. Neo mức "phòng sáng" = percentile 65 của base. Vùng base DƯỚI neo được kéo
     lên (nén khoảng cách còn k_dark ~35%% tại amount=1); vùng trên neo giữ ~nguyên.
  3. SÀN BẢO VỆ ĐEN THẬT: base dưới ~0.08 (đồ vật đen: lò, TV) không kéo —
     đen sâu giữ nguyên, chỉ mảng phòng thiếu sáng được thắp.
  4. Ghép lại base' + detail -> ratio luma -> nhân 3 kênh (không lệch màu).

amount=0 -> bit-identical. Hợp đồng: apply(img f32 [0,1] HxWx3 BGR, params) -> cùng shape.
"""
import cv2
import numpy as np

cv2.setNumThreads(3)

_LUMA_W = np.array([0.0722, 0.7152, 0.2126], dtype=np.float32)

_RADIUS_FRAC = 0.06        # bán kính guided filter theo cạnh ngắn
_GUIDED_EPS = 0.02 ** 2
_ANCHOR_PCT = 65           # "mức phòng sáng" = p65 cua base
_K_DARK_MIN = 0.35         # tại amount=1: vùng tối chỉ còn 35% khoảng cách tới neo
_K_BRIGHT = 0.92           # vùng sáng hơn neo gần như giữ nguyên
_BLACK_FLOOR = 0.16        # base (luma tuyến tính) dưới mức này = đồ vật đen thật, không kéo
                           # (0.08 để lọt lò/TV đen bóng ~0.1-0.15 — bị kéo nâu, v8 23/07)
_LOG_EPS = 1e-3
_MAX_GAIN = 3.2            # trần gain an toàn


def _smoothstep(x, lo, hi):
    t = np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _guided(src, radius, eps):
    r = max(4, int(radius))
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
        return cv2.ximgproc.guidedFilter(guide=src, src=src, radius=r, eps=eps)
    k = (2 * r + 1, 2 * r + 1)
    m = cv2.boxFilter(src, -1, k)
    mm = cv2.boxFilter(src * src, -1, k)
    var = np.maximum(mm - m * m, 0.0)
    a = var / (var + eps)
    b = m - a * m
    return cv2.boxFilter(a, -1, k) * src + cv2.boxFilter(b, -1, k)


def apply(img, params=None):
    params = params or {}
    amount = float(np.clip(params.get("amount", 0.6), 0.0, 1.0))

    img = np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)
    if amount == 0.0:
        return img

    h, w = img.shape[:2]
    y = np.maximum(img @ _LUMA_W, _LOG_EPS)
    L = np.log2(y)

    # ⭐ 04/08 (ra soat vong 7) — SAI THANG DO: bo loc guided chay tren log2-luma.
    # `L = log2(y)` voi y in [_LOG_EPS, 1] nen L trai khoang 10 DON VI ([-9.97, 0]),
    # trong khi `_GUIDED_EPS = 0.0004` la con so tune cho thang [0,1]. Voi guided
    # filter, eps la nguong "coi chenh lech nay la NHIEU hay la CANH" — dat eps nho
    # hon thang tin hieu 4 bac do lam bo loc bam sat tung chi tiet, tuc base = L va
    # detail = 0. Do that: ti le detail/tin hieu = 0.0010 (gan bang khong).
    # => Phep tach Retinex ma ca op nay dua vao KHONG HE XAY RA. Op suy bien thanh
    #    mot phep bien doi tone toan cuc; MAE o amount=0.35 chi 0.0062.
    # Chuan hoa L ve [0,1] truoc khi loc (dung cach harsh_sun da lam) roi nhan nguoc
    # lai — do lai cho ti le detail 0.0109, gap ~11 lan.
    #
    # ⚠️ PHAI DOC: `_K_DARK_MIN`, `_BLACK_FLOOR` va lieu `sl=0.16` trong brain/run.py
    # deu duoc tune KHI OP DANG SUY BIEN. Nay op chay dung thi cung mot lieu se manh
    # hon truoc. Production hien tai KHONG bi anh huong (model_complete=True nen brain
    # bo qua op nay), nhung duong `deliver.py --use-model` co goi shadow_light(0.35) —
    # phai tune lai truoc khi dung duong do de giao hang.
    _lo = float(L.min())
    _rng = max(float(L.max()) - _lo, 1e-6)
    base = _guided((L - _lo) / _rng, _RADIUS_FRAC * min(h, w), _GUIDED_EPS) * _rng + _lo
    detail = L - base

    anchor = np.percentile(base, _ANCHOR_PCT)
    k_dark = 1.0 - amount * (1.0 - _K_DARK_MIN)
    delta = base - anchor
    # FIX 30/07: _K_BRIGHT co dinh 0.92 -> nen vung SANG 8% NGAY CA khi amount~0
    # (khong lien tuc tai 0: amount=0 la identity, amount=0.001 mat 8% highlight)
    # -> cua so mat ~0.32EV, highlight bet/bac. Cho k_bright ti le theo amount.
    k_bright = 1.0 - amount * (1.0 - _K_BRIGHT)
    compressed = np.where(delta < 0.0, delta * k_dark, delta * k_bright)

    # Bảo vệ đen thật: blend về base gốc khi base (tuyến tính) dưới sàn.
    base_lin = np.exp2(base)
    protect = _smoothstep(base_lin, _BLACK_FLOOR * 0.5, _BLACK_FLOOR * 1.8)
    base_new = base * (1.0 - protect) + (anchor + compressed) * protect

    y_new = np.exp2(base_new + detail)
    gain = np.clip(y_new / y, 1.0 / _MAX_GAIN, _MAX_GAIN)
    out = img * gain[..., None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)
