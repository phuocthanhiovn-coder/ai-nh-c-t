"""region_apply — TAY THEO VÙNG (24/07/2026, chuẩn hóa từ demo brain_demo_cmp_v2).

VÌ SAO: chủ dự án chốt bệnh "AI không phân biệt đồ vật". Op toàn cục phải kìm
liều vì sợ vạ lây (thắp góc thì lò nâu, rửa bùn thì gỗ phai). Cơ chế này cho
phép áp BẤT KỲ op nào theo mask ngữ nghĩa (segment_room) + feather — đồ vật
nào không thuộc vùng thì bit-identical.

API:
    region_apply(img, op_fn, params, mask, feather_sigma=25)
      mask: float [0,1] HxW — 1 = áp trọn, 0 = giữ nguyên ảnh gốc.
    build_arch_mask(masks)  — tường+sàn+trần (kiến trúc phòng)
    build_object_mask(masks) — đồ vật

Van an toàn: mask được clip [0,1] + feather Gauss để không lộ ranh giới;
op_fn vẫn là operator chuẩn (được clamp bởi engine như thường).
"""
import cv2
import numpy as np

cv2.setNumThreads(3)


def region_apply(img, op_fn, params, mask, feather_sigma=25):
    img = np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)
    m = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    if m.shape != img.shape[:2]:
        m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    if feather_sigma > 0:
        m = cv2.GaussianBlur(m, (0, 0), feather_sigma)
    out_full = op_fn(img, params)
    m3 = m[..., None]
    return np.clip(img * (1.0 - m3) + out_full * m3, 0.0, 1.0).astype(np.float32)


def build_arch_mask(masks):
    """Tường + sàn + trần — vùng 'kiến trúc phòng' (được phép thắp sáng mạnh)."""
    return np.clip(masks["wall"] + masks["floor"] + masks["ceiling"], 0.0, 1.0)


def build_object_mask(masks):
    """Đồ vật — vùng giữ màu/tương phản, miễn nhiễm thắp sáng + rửa bùn."""
    return np.clip(masks["object"], 0.0, 1.0)
