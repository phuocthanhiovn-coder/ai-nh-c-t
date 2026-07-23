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
ALIGN_SCORE_THRESHOLD = 0.50  # Ngưỡng khớp cạnh tối thiểu để đưa vào tập train (phát hiện bằng Edge-NCC)
