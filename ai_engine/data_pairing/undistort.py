import cv2
import numpy as np

cv2.setNumThreads(2)

from .ingest import align_before_after

SMALL_DIM = 512


def resize_small(img, target_dim=SMALL_DIM):
    """Downscale ảnh giữ nguyên tỷ lệ sao cho cạnh dài nhất bằng target_dim."""
    h, w = img.shape[:2]
    scale = target_dim / max(h, w)
    if scale >= 1.0:
        return img.copy()
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def undistort_k1(img_bgr, k1):
    """Áp mô hình méo radial đơn giản (k1 only) lên ảnh, camera matrix giả định
    fx=fy=w, cx=w/2, cy=h/2 (không có thông tin calib thật)."""
    if k1 == 0.0:
        return img_bgr.copy()
    h, w = img_bgr.shape[:2]
    camera_matrix = np.array([
        [w, 0, w / 2.0],
        [0, w, h / 2.0],
        [0, 0, 1.0],
    ], dtype=np.float64)
    dist_coeffs = np.array([k1, 0.0, 0.0, 0.0], dtype=np.float64)
    return cv2.undistort(img_bgr, camera_matrix, dist_coeffs)


def _search_k1(before_small, after_small, k1_values, best_k1=None, best_score=-1.0):
    """Chạy align_before_after cho mỗi k1 trong danh sách, trả về (best_k1, best_score)."""
    for k1 in k1_values:
        k1 = round(float(k1), 4)
        undistorted = undistort_k1(before_small, k1)
        score, _ = align_before_after(undistorted, after_small)
        if score > best_score:
            best_score = score
            best_k1 = k1
    return best_k1, best_score


def estimate_undistort(before_bgr, after_bgr):
    """
    Grid-search hệ số méo k1 để cứu cặp align_low, dùng lại align_before_after
    (ORB+RANSAC+ECC+Edge-NCC) từ ingest.py, không viết lại logic đã duyệt.

    Trả về (k1_best, before_undistorted_aligned_full_res, score_full_res).
    """
    before_small = resize_small(before_bgr, SMALL_DIM)
    after_small = resize_small(after_bgr, SMALL_DIM)

    # 1) Grid search thô: [-0.30, +0.10] bước 0.02
    coarse_values = np.arange(-0.30, 0.10 + 1e-9, 0.02)
    best_k1, best_score = _search_k1(before_small, after_small, coarse_values)

    # 2) Tinh chỉnh bước 0.005 quanh đỉnh (+-0.02 quanh best_k1 thô)
    fine_values = np.arange(best_k1 - 0.02, best_k1 + 0.02 + 1e-9, 0.005)
    best_k1, best_score = _search_k1(before_small, after_small, fine_values, best_k1, best_score)

    # 3) Áp k1 tốt nhất lên before ở độ phân giải "full" (kích thước gốc truyền vào,
    #    ở review đã là JPG ~2048px) rồi align lại bằng đúng hàm align_before_after
    #    (tái tạo ORB+ECC ở độ phân giải full thay vì cố rescale ma trận homography
    #    tính ở bản 512px nhỏ — đơn giản hơn, không đổi kết quả cuối, và không đụng
    #    logic đã duyệt trong ingest.py/align.py).
    before_full_undistorted = undistort_k1(before_bgr, best_k1)
    final_score, aligned_before_final = align_before_after(before_full_undistorted, after_bgr)

    return best_k1, aligned_before_final, final_score


if __name__ == "__main__":
    print("Undistort estimation module loaded.")
