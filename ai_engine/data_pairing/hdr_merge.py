"""hdr_merge — GỘP BRACKET có KÉO CỬA SỔ từ tấm tối nhất (25/07).

Vấn đề: Mertens fusion để vùng cửa sổ CHÁY TRẮNG (trọng số ưu tiên frame phơi
nội thất). Nhưng tấm phơi TỐI NHẤT trong bracket có ĐỦ chi tiết ngoài trời
(trời/cây/nhà đúng sáng). Con này: Mertens cho nền + nơi cháy trắng thì HÒA LẠI
pixel từ tấm tối (đã khớp độ sáng ranh giới) -> cửa sổ hiện cảnh ngoài TỰ NHIÊN,
không cần window_pull thô post-hoc.

merge_with_window(frames_bgr_uint8) -> merged BGR uint8.
frames: list ảnh BGR đã align, KHÁC phơi sáng (thứ tự bất kỳ).
"""
import cv2
import numpy as np

cv2.setNumThreads(3)

_LUMA = np.array([0.114, 0.587, 0.299], dtype=np.float32)  # BGR


def _luma(img01):
    return img01 @ _LUMA


def merge_with_window(frames):
    """frames: list HxWx3 BGR uint8 (đã align). Trả merged uint8."""
    fs = [f.astype(np.float32) / 255.0 for f in frames]
    # 1) Mertens fusion chuẩn cho nền
    merge = cv2.createMergeMertens()
    base = np.clip(merge.process([(f * 255).astype(np.uint8) for f in fs]), 0, 1)

    # 2) tìm tấm TỐI NHẤT (mean luma nhỏ nhất) — chứa chi tiết ngoài trời
    means = [float(_luma(f).mean()) for f in fs]
    dark = fs[int(np.argmin(means))]

    # 3) mask vùng base CHÁY / gần cháy (cửa sổ): luma cao + tấm tối CÓ chi tiết
    #    (tấm tối ở vùng đó KHÔNG cháy -> tức là ngoài trời, kéo lại được)
    yb = _luma(base)
    yd = _luma(dark)
    blown = np.clip((yb - 0.82) / 0.15, 0, 1)          # base bắt đầu cháy từ 0.82
    dark_has = np.clip((0.75 - yd) / 0.25, 0, 1)        # tấm tối vùng đó chưa cháy
    win_mask = blown * dark_has
    win_mask = cv2.GaussianBlur(win_mask, (0, 0), 3)

    if float(win_mask.mean()) < 0.0005:
        return (base * 255).clip(0, 255).astype(np.uint8)

    # 4) khớp độ sáng tấm tối cho vừa ranh giới cửa sổ rồi HÒA vào chỗ cháy.
    #    Nâng nhẹ tấm tối (nó tối) để cảnh ngoài đủ sáng tự nhiên, không xám.
    dark_lift = np.clip(dark * 1.6, 0, 1)               # kéo sáng cảnh ngoài
    m3 = win_mask[..., None]
    out = base * (1 - m3) + dark_lift * m3
    return (np.clip(out, 0, 1) * 255).clip(0, 255).astype(np.uint8)
