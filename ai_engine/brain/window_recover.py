"""Window recovery (26/07): kéo phơi sáng CỬA SỔ xuống để lộ cảnh ngoài (nhà/cây)
như AutoHDR, dùng mask mắt-mới (Grounded-SAM-2). Nguồn nội dung = ẢNH GỐC (bracket
merge có nhà), KHÔNG phải output model (đã thổi trắng). Ghép lại vào output.

Chủ dự án chỉ (26/07): model thổi trắng + ám xanh cửa sổ → mất nhà+cây; AutoHDR
kéo xuống lộ rõ. Nội dung nhà CÓ trong ảnh gốc → cứu được.
"""
import cv2
import numpy as np


def _clahe_grid(shape, frac=0.015):
    """FIX 30/07: luoi CLAHE co dinh 8x8 -> o 753x503px tren anh 6024 =>
    chenh phoi sang 2.5 LAN giua cac o = dung benh 'cho sang cho toi'.
    Cho o ~1.5% canh ngan de CLAHE tro lai dung nghia 'tuong phan cuc bo'."""
    h, w = shape[:2]
    side = max(8.0, frac * min(h, w))
    return (max(2, int(round(w / side))), max(2, int(round(h / side))))


def _recover_exterior(before):
    """Kéo highlight cửa sổ xuống + tăng tương phản cục bộ + trung tính cast."""
    b = np.clip(before, 0, 1).astype(np.float32)
    luma = b @ np.array([0.114, 0.587, 0.299], dtype=np.float32)
    # kéo vùng SÁNG xuống mạnh hơn (cửa sổ cháy), vùng đã tối giữ nguyên
    w_hi = np.clip((luma - 0.45) / 0.55, 0, 1)[..., None]
    recov = b * (1.0 - 0.55 * w_hi)
    # local contrast (CLAHE trên L) — lộ vân vách/mái nhà
    lab = cv2.cvtColor((np.clip(recov, 0, 1) * 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(1.6, _clahe_grid(lab.shape)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0


def recover_windows(before, out, win_mask, feather=None):
    """before/out float[0,1] BGR HxWx3; win_mask float[0,1] HxW (kính cửa sổ).
    Trả out với vùng cửa sổ = cảnh ngoài đã kéo xuống (trung tính, lộ nhà/cây)."""
    if win_mask is None or float(win_mask.max()) <= 0:
        return out
    b = np.clip(before, 0, 1).astype(np.float32)
    # BAO VE vat SANG-AM tren be cua (den/fixture): nha ngoai trung tinh/lanh, den
    # AM (R>B) + rat sang -> giu nguyen (khong keo xuong -> khong nhoe nhu proof cu).
    lum = b @ np.array([0.114, 0.587, 0.299], dtype=np.float32)
    warm = b[..., 2] - b[..., 0]                      # R - B
    keep = np.clip((lum - 0.62) / 0.38, 0, 1) * np.clip(warm / 0.12, 0, 1)
    eff = np.clip(win_mask.astype(np.float32) * (1.0 - keep), 0, 1)

    recov = _recover_exterior(before)
    # FIX 30/07: BO grey-world o day. Vung cua so la cho DUY NHAT ma gia dinh
    # "trung binh vung = xam" CHAC CHAN sai (troi xanh, cay xanh, hoang hon cam).
    # Do duoc: mau canh ngoai mat 55% chroma; troi xanh thuan bi ep ve xam hoan toan.
    if feather is None:   # FIX 30/07: 3px = 0.05% anh 6024 -> duong seam cung
        feather = max(3.0, 0.005 * min(eff.shape[:2]))
    fmask = cv2.GaussianBlur(eff, (0, 0), feather)[..., None]
    return np.clip(out * (1.0 - fmask) + recov * fmask, 0, 1)


def get_window_mask(before_u8):
    """Mask cửa sổ bằng mắt-mới gsam (Grounded-SAM-2). Lỗi/không có -> None."""
    try:
        from ai_engine.specialists.segment_room.gsam import window_mask
        m = window_mask(before_u8)
        return m if float(m.max()) > 0 else None
    except Exception:
        return None
