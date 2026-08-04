import os

# Đường dẫn dữ liệu
RAW_BEFORE_DIR = "data/raw/before"
RAW_AFTER_DIR = "data/raw/after"

PAIRS_COLOR_DIR = "data/pairs"
PAIRS_SKY_DIR = "data/pairs_sky"
PAIRS_REMOVAL_DIR = "data/pairs_removal"
UNMATCHED_DIR = "data/unmatched"
OUTPUTS_SAMPLES_DIR = "outputs/pair_samples"
REPORT_CSV_PATH = "data/report.csv"

# Cấu hình Gom Bracket
BRACKET_TIME_THRESHOLD = 5.0  # Khoảng cách giây giữa các ảnh trong cùng bộ bracket (nếu có EXIF)
BRACKET_DIFF_THRESHOLD = 0.95  # Độ tương đồng SSIM tối thiểu để gom bracket khi không có EXIF

# Cấu hình Tìm ứng viên (Shortlist)
PHASH_THRESHOLD = 15  # Khoảng cách Hamming tối đa để xếp vào nhóm ứng viên trùng cấu trúc
THUMBNAIL_SIZE = (64, 64)

# Cấu hình Căn khớp (Verify & Align)
MAX_FEATURES = 2000
MIN_MATCH_COUNT = 15  # Số lượng matches tối thiểu để chạy Homography
INLIER_THRESHOLD = 5.0  # RANSAC reprojection threshold
MIN_INLIER_RATIO = 0.25  # Tỉ lệ inliers / matches tối thiểu để công nhận là khớp

# Review directory cho các ảnh lệch
REVIEW_DIR = "data/review"
# 04/08 (ra soat vong 7) — Y NGHIA CUA CON SO NAY DA DOI, gia tri thi khong.
#
# TRUOC: nguong Edge-NCC. Hai loi CHAN: (a) Edge-NCC khong do duoc do can khop (dai
# diem cua cap lech va cap khop CHONG NHAU hoan toan; cap lech 62px van an 0.571);
# (b) diem co gian theo do phan giai (cung cap anh: 0.560 @full-res vs 0.677 @2048),
# nen hang so hieu chinh o 2048 nay bi build_real_pairs_fullres.py ap o 6000x4000 —
# o do no khat khe tuong duong 0.64, lam 33/53 cap bi loai (17 cap nam sat duoi vach).
#
# NAY: `align_before_after` do bang HINH HOC (tach dich/bien dang, 5 vung chon theo
# ket cau) o CO CO DINH 2048, roi anh xa sao cho 0.50 DUNG BANG ranh gioi dat/truot.
# Nen giu nguyen 0.50 va tu day con so co nghia that; muon khat khe hon thi NANG len
# (vd 0.75 ~ lech 1.5px@2048), muon rong hon thi HA xuong.
# ⚠️ KHONG con phu thuoc do phan giai anh vao — do dung o dau cung ra cung mot so.
ALIGN_SCORE_THRESHOLD = 0.50  # 0.50 = lech dung 3.0px @2048; 1.00 = trung khit tuyet doi
