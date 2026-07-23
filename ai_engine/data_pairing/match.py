import cv2
import numpy as np

def calculate_phash(image_path, hash_size=8):
    """
    Tính toán Perceptual Hash (pHash) cho một bức ảnh.
    Giúp so sánh cấu trúc thô nhanh chóng.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # 1. Resize ảnh về kích thước nhỏ (32x32) để giảm chi tiết nhiễu
    resized = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    
    # 2. Chuyển sang kiểu float32 để tính DCT
    dct = cv2.dct(np.float32(resized))
    
    # 3. Lấy vùng tần số thấp (8x8 ở góc trên bên trái, bỏ hệ số DC ở vị trí 0,0)
    dct_low = dct[0:hash_size, 0:hash_size]
    
    # 4. Tính giá trị trung bình
    avg = np.mean(dct_low)
    
    # 5. Tạo chuỗi bit dựa trên việc so sánh với giá trị trung bình
    diff = dct_low > avg
    
    # Trả về hash dưới dạng mảng boolean phẳng
    return diff.flatten()

def hamming_distance(hash1, hash2):
    """Tính khoảng cách Hamming giữa 2 pHash."""
    if hash1 is None or hash2 is None:
        return 999
    return np.count_nonzero(hash1 != hash2)

def find_shortlist(before_path, after_paths, threshold=15):
    """
    Tìm danh sách các ảnh 'after' ứng viên có cấu trúc tương đồng với 'before'.
    Trả về danh sách các tuple (after_path, score) được sắp xếp từ giống nhất đến ít giống nhất.
    """
    before_hash = calculate_phash(before_path)
    if before_hash is None:
        return []
        
    candidates = []
    for after_path in after_paths:
        after_hash = calculate_phash(after_path)
        if after_hash is not None:
            dist = hamming_distance(before_hash, after_hash)
            if dist <= threshold:
                candidates.append((after_path, dist))
                
    # Sắp xếp theo khoảng cách Hamming tăng dần (càng nhỏ càng giống)
    candidates.sort(key=lambda x: x[1])
    return candidates

if __name__ == "__main__":
    print("Match Shortlist module loaded.")
