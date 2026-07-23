import cv2
import numpy as np
from .config import MAX_FEATURES, MIN_MATCH_COUNT, INLIER_THRESHOLD, MIN_INLIER_RATIO

def extract_and_match(img1_gray, img2_gray):
    """Trích xuất và so khớp keypoints sử dụng ORB."""
    orb = cv2.ORB_create(MAX_FEATURES)
    kp1, des1 = orb.detectAndCompute(img1_gray, None)
    kp2, des2 = orb.detectAndCompute(img2_gray, None)
    
    if des1 is None or des2 is None or len(des1) < MIN_MATCH_COUNT or len(des2) < MIN_MATCH_COUNT:
        return None, None, None
        
    # Sử dụng BFMatcher với khoảng cách Hamming (vì ORB là binary descriptor)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Sắp xếp matches theo khoảng cách
    matches = sorted(matches, key=lambda x: x.distance)
    
    return kp1, kp2, matches

def verify_and_align(before_path, after_path):
    """
    Xác minh xem before và after có khớp cấu trúc không bằng Homography RANSAC.
    Nếu khớp, trả về (True, aligned_before_img, inlier_ratio, homography_matrix).
    Nếu không khớp, trả về (False, None, 0, None).
    """
    img_before = cv2.imread(before_path)
    img_after = cv2.imread(after_path)
    
    if img_before is None or img_after is None:
        return False, None, 0, None
        
    gray_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)
    
    # So khớp keypoints
    res = extract_and_match(gray_before, gray_after)
    if res is None:
        return False, None, 0, None
        
    kp1, kp2, matches = res
    if len(matches) < MIN_MATCH_COUNT:
        return False, None, 0, None
        
    # Chuyển đổi keypoints sang tọa độ float32
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Tính toán Homography sử dụng RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, INLIER_THRESHOLD)
    
    if H is None or mask is None:
        return False, None, 0, None
        
    # Tính số lượng inliers
    inliers_count = np.sum(mask)
    total_matches = len(matches)
    inlier_ratio = inliers_count / total_matches
    
    # Kiểm tra điều kiện chất lượng khớp
    if inliers_count >= MIN_MATCH_COUNT and inlier_ratio >= MIN_INLIER_RATIO:
        # Căn khớp ảnh before theo coordinates của after
        height, width, _ = img_after.shape
        aligned_before = cv2.warpPerspective(img_before, H, (width, height), borderMode=cv2.BORDER_REPLICATE)
        return True, aligned_before, inlier_ratio, H
        
    return False, None, inlier_ratio, None

if __name__ == "__main__":
    print("Alignment and Verification module loaded.")
