import os
import cv2
import numpy as np
from PIL import Image
import exifread
from datetime import datetime
import shutil

def get_image_time(file_path):
    """Đọc EXIF để lấy thời gian chụp của ảnh."""
    try:
        with open(file_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
            if 'EXIF DateTimeOriginal' in tags:
                time_str = str(tags['EXIF DateTimeOriginal'])
                return datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
    except Exception:
        pass
    # Fallback: sử dụng mtime của file
    return datetime.fromtimestamp(os.path.getmtime(file_path))

def calculate_brightness(image_path):
    """Tính độ sáng trung bình của ảnh."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 127
    return np.mean(img)

from .match import calculate_phash, hamming_distance

def group_brackets(image_paths, time_threshold=5.0):
    """
    Gom các ảnh chụp cùng một cảnh dựa trên thời gian chụp và độ tương đồng cấu trúc.
    Trả về danh sách các nhóm bracket.
    """
    if not image_paths:
        return []
    
    # Sắp xếp các ảnh theo thời gian chụp
    img_data = []
    for path in image_paths:
        t = get_image_time(path)
        img_data.append((path, t))
    
    img_data.sort(key=lambda x: x[1])
    
    # Tính pHash trước cho tất cả các file để tránh tính lặp lại
    hashes = {}
    for path in image_paths:
        hashes[path] = calculate_phash(path)
        
    groups = []
    
    for path, t in img_data:
        # Tìm xem ảnh này có thể gộp vào nhóm nào hiện có không
        placed = False
        h_path = hashes[path]
        
        for group in groups:
            # So sánh với ảnh đầu tiên trong nhóm
            ref_path = group[0]
            ref_time = get_image_time(ref_path)
            
            # Kiểm tra khoảng cách thời gian và cấu trúc pHash
            if abs((t - ref_time).total_seconds()) <= time_threshold:
                ref_hash = hashes[ref_path]
                if h_path is not None and ref_hash is not None:
                    # Khoảng cách pHash nhỏ (<=10) nghĩa là cấu trúc giống hệt nhau (chỉ khác phơi sáng)
                    if hamming_distance(h_path, ref_hash) <= 10:
                        group.append(path)
                        placed = True
                        break
        
        if not placed:
            # Tạo nhóm mới
            groups.append([path])
            
    return groups

def select_representative(group):
    """
    Chọn tấm đại diện từ nhóm bracket.
    Tấm đại diện là tấm có độ sáng gần với trung vị (median) nhất, 
    tránh bị quá tối (under-exposed) hoặc quá sáng (over-exposed).
    """
    if len(group) == 1:
        return group[0]
        
    brightness_list = []
    for path in group:
        b = calculate_brightness(path)
        brightness_list.append((path, b))
        
    # Sắp xếp theo độ sáng
    brightness_list.sort(key=lambda x: x[1])
    
    # Chọn tấm ở vị trí giữa (median) làm đại diện phơi sáng chuẩn
    mid_idx = len(brightness_list) // 2
    return brightness_list[mid_idx][0]

if __name__ == "__main__":
    print("Bracket Grouper module loaded.")
