import cv2
import numpy as np

def normalize_and_get_diff(aligned_before, after):
    """
    Chuẩn hóa độ sáng của aligned_before về gần với after trên từng kênh màu,
    sau đó tính toán bản đồ khác biệt (diff map) 3 kênh và lấy giá trị lớn nhất.
    """
    normalized_channels = []
    
    # Chuẩn hóa trên từng kênh màu B, G, R
    for i in range(3):
        ch_before = aligned_before[:, :, i]
        ch_after = after[:, :, i]
        
        mean_b, std_b = cv2.meanStdDev(ch_before)
        mean_a, std_a = cv2.meanStdDev(ch_after)
        
        mean_b = mean_b[0][0]
        std_b = std_b[0][0]
        mean_a = mean_a[0][0]
        std_a = std_a[0][0]
        
        if std_b < 1e-4:
            std_b = 1.0
            
        norm_ch = ((ch_before - mean_b) * (std_a / std_b) + mean_a)
        norm_ch = np.clip(norm_ch, 0, 255).astype(np.uint8)
        normalized_channels.append(norm_ch)
        
    normalized_before = cv2.merge(normalized_channels)
    
    # Tính toán trị tuyệt đối của hiệu hai ảnh màu
    diff_color = cv2.absdiff(after, normalized_before)
    
    # Lấy giá trị lớn nhất của sự thay đổi trong 3 kênh màu tại mỗi pixel
    diff = np.max(diff_color, axis=2)
    return diff

def classify_edit_type(aligned_before, after):
    """
    Phân loại kiểu chỉnh sửa dựa trên phân tích diff map.
    Trả về (edit_type, confidence).
    - color: chỉnh sửa màu sắc toàn cục.
    - sky: thay đổi bầu trời (nhiều thay đổi ở phần trên ảnh).
    - removal: xóa hoặc thêm vật thể cục bộ.
    """
    diff = normalize_and_get_diff(aligned_before, after)
    
    # Ngưỡng hóa để tìm các vùng thay đổi rõ rệt
    _, thresh = cv2.threshold(diff, 65, 255, cv2.THRESH_BINARY)
    
    # Áp dụng phép mở (Opening) để loại bỏ nhiễu đường mảnh do lệch pixel 1-2px ở các cạnh sắc nét
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Tính diện tích các vùng thay đổi
    h, w = thresh.shape
    total_pixels = h * w
    changed_pixels = np.sum(thresh == 255)
    changed_ratio = changed_pixels / total_pixels
    
    # 1. Kiểm tra Sky Replacement
    # Bầu trời thường ở 1/2 phía trên của bức ảnh
    top_half = thresh[0:int(h*0.5), :]
    bottom_half = thresh[int(h*0.5):h, :]
    
    top_changed = np.sum(top_half == 255)
    bottom_changed = np.sum(bottom_half == 255)
    
    # Nếu phần lớn thay đổi nằm ở nửa trên và chiếm tỉ lệ đáng kể của nửa trên đó
    top_half_pixels = top_half.size
    top_ratio = top_changed / top_half_pixels if top_half_pixels > 0 else 0
    
    # 2. Kiểm tra Object Removal
    # Tìm các contours (đốm thay đổi)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    large_blobs = 0
    for c in contours:
        area = cv2.contourArea(c)
        # Nếu có đốm thay đổi kích thước trung bình/lớn cục bộ (ví dụ chậu cây cảnh bị xóa)
        if area > (total_pixels * 0.002) and area < (total_pixels * 0.10):
            large_blobs += 1
            
    # Phân tích heuristics để phân loại
    # print(f"DEBUG: changed_ratio={changed_ratio:.5f}, large_blobs={large_blobs}")
    # for c in contours:
    #     print(f"  Contour area: {cv2.contourArea(c)}")
        
    if changed_ratio < 0.001:
        # Hầu như không có thay đổi lớn cục bộ hoặc thay trời -> Chỉnh màu toàn cục
        return "color", 0.95
        
    if top_ratio > 0.25 and (top_changed > bottom_changed * 1.8):
        # Thay đổi nhiều ở nửa trên -> Sky Replacement
        confidence = min(0.5 + (top_ratio * 0.5), 0.99)
        return "sky", float(confidence)
        
    elif 0 < large_blobs <= 2 and changed_ratio < 0.05:
        # Thay đổi cục bộ ở một vài đốm nhỏ -> Object Removal
        confidence = 0.85
        return "removal", float(confidence)
        
    else:
        # Các thay đổi khác -> Color-only
        confidence = 0.90 if changed_ratio < 0.3 else 0.70
        return "color", float(confidence)

if __name__ == "__main__":
    print("Classifier module loaded.")
