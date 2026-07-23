import os
import cv2
import numpy as np
import random
import time
from datetime import datetime
from .config import RAW_BEFORE_DIR, RAW_AFTER_DIR

def draw_mock_scene(scene_id):
    """Vẽ một cảnh bất động sản giả lập đơn giản với nhiều chi tiết hình học để ORB dễ bắt keypoint."""
    img = np.ones((600, 800, 3), dtype=np.uint8) * 180  # Tường xám sáng
    
    # Vẽ sàn nhà gỗ
    cv2.rectangle(img, (0, 450), (800, 600), (40, 60, 100), -1)
    for x in range(0, 800, 50):
        cv2.line(img, (x, 450), (x + int((x-400)*0.5), 600), (20, 30, 50), 2)
        
    # Với scene 3 & 4 (sky replacement), ta vẽ bầu trời lớn bao trọn 1/3 phía trên ảnh (giả lập ngoại cảnh)
    if scene_id in [3, 4]:
        cv2.rectangle(img, (0, 0), (800, 220), (235, 206, 135), -1)  # Bầu trời xanh lớn phía trên
        # Vẽ mây
        cv2.circle(img, (200, 100), 50, (255, 255, 255), -1)
        cv2.circle(img, (300, 90), 60, (255, 255, 255), -1)
        cv2.circle(img, (600, 110), 40, (255, 255, 255), -1)
        
        # Vẽ một ngôi nhà đơn giản ở dưới
        cv2.rectangle(img, (150, 220), (650, 450), (120, 120, 120), -1)
        # Mái nhà
        pts = np.array([[100, 220], [400, 120], [700, 220]], np.int32)
        cv2.fillPoly(img, [pts], (80, 50, 50))
    else:
        # Vẽ cửa sổ lớn thông thường cho các cảnh khác
        cv2.rectangle(img, (100, 50), (450, 350), (100, 100, 100), 10)  # Khung cửa
        cv2.rectangle(img, (110, 60), (440, 340), (235, 206, 135), -1)  # Bầu trời xanh ban ngày
        # Vẽ mây
        cv2.circle(img, (200, 120), 40, (255, 255, 255), -1)
        cv2.circle(img, (240, 120), 50, (255, 255, 255), -1)
        cv2.circle(img, (280, 120), 40, (255, 255, 255), -1)
        
    # Các chi tiết đặc trưng cho từng scene
    if scene_id == 1:
        cv2.rectangle(img, (550, 100), (700, 250), (30, 30, 150), -1)  # Tranh xanh lá
        cv2.circle(img, (625, 175), 30, (255, 255, 255), 2)
    elif scene_id == 2:
        cv2.rectangle(img, (550, 100), (700, 250), (150, 30, 30), -1)  # Tranh đỏ
        cv2.rectangle(img, (570, 120), (680, 230), (255, 255, 255), 2)
    elif scene_id == 3:
        # Cảnh có cột ống khói trên mái nhà
        cv2.rectangle(img, (200, 100), (260, 220), (40, 40, 40), -1)
    elif scene_id == 4:
        # Cảnh có cửa ra vào nhỏ màu đỏ
        cv2.rectangle(img, (350, 300), (450, 450), (30, 30, 150), -1)
    elif scene_id == 5:
        # Vật thể cụ thể để xóa (Một chậu cây cảnh màu đỏ rực đặt trên sàn gỗ bên phải)
        cv2.rectangle(img, (550, 380), (700, 450), (80, 120, 100), -1)  # Kệ bàn
        cv2.circle(img, (625, 340), 25, (0, 0, 220), -1)  # Bình hoa đỏ rực ở góc dưới
        cv2.line(img, (625, 340), (625, 380), (10, 10, 10), 4)
        
    return img

def apply_color_edit(img):
    """Mô phỏng chỉnh màu (Color enhancement) đơn giản."""
    # Tăng độ tương phản và bão hòa màu nhẹ
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced

def apply_sky_replacement(img):
    """Mô phỏng thay đổi bầu trời (Sky replacement) bằng cách đổi bầu trời xanh -> cam hoàng hôn."""
    res = img.copy()
    # Tô trực tiếp màu cam hoàng hôn vào vùng 1/3 phía trên ảnh (y: 0-220) nơi chứa bầu trời
    res[0:220, :] = (50, 100, 240)  # BGR cam đỏ
    
    # Chỉnh tông màu ấm nhẹ cho phần còn lại
    res = apply_color_edit(res)
    return res

def apply_object_removal(img):
    """Mô phỏng xóa vật thể (Xóa bình hoa đỏ ở cảnh 5)."""
    res = img.copy()
    # Vẽ đè màu bức tường xám (180, 180, 180) đè lên khu vực bình hoa (x: 590-660, y: 310-375)
    cv2.rectangle(res, (590, 310), (660, 375), (180, 180, 180), -1)
    return apply_color_edit(res)

def apply_geometric_distortion(img):
    """Mô phỏng lệch khung hình/xoay nhẹ do không dùng tripod hoàn hảo."""
    h, w, _ = img.shape
    # Tạo ma trận xoay và dịch chuyển nhẹ (thực tế tripod chỉ lệch cực nhỏ)
    angle = random.uniform(-0.3, 0.3)  # Xoay dưới 0.3 độ
    tx = random.uniform(-2.0, 2.0)     # Dịch chuyển x
    ty = random.uniform(-2.0, 2.0)     # Dịch chuyển y
    
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    
    distorted = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return distorted

def generate_mock_dataset():
    """Tạo bộ dataset test giả lập."""
    print("[*] Đang tạo dữ liệu test giả lập...")
    os.makedirs(RAW_BEFORE_DIR, exist_ok=True)
    os.makedirs(RAW_AFTER_DIR, exist_ok=True)
    
    # 5 cảnh tương ứng với 5 cặp chính
    for scene_id in range(1, 6):
        base_img = draw_mock_scene(scene_id)
        
        # 1. Tạo bracket cho BEFORE (3 tấm độ sáng khác nhau)
        # Giả lập thời gian chụp tăng dần: mỗi scene cách nhau 60 giây, mỗi bracket cách nhau 1 giây
        scene_base_time = time.time() + (scene_id * 60)
        
        # Tấm 1: Rất tối (Under-exposed)
        img_dark = (base_img * 0.4).astype(np.uint8)
        dark_path = os.path.join(RAW_BEFORE_DIR, f"scene_{scene_id}_under.png")
        cv2.imwrite(dark_path, img_dark)
        os.utime(dark_path, (scene_base_time, scene_base_time))
        
        # Tấm 2: Sáng chuẩn (Normal-exposed)
        normal_path = os.path.join(RAW_BEFORE_DIR, f"scene_{scene_id}_normal.png")
        cv2.imwrite(normal_path, base_img)
        os.utime(normal_path, (scene_base_time + 1, scene_base_time + 1))
        
        # Tấm 3: Rất sáng (Over-exposed)
        img_bright = np.clip(base_img * 1.8, 0, 255).astype(np.uint8)
        bright_path = os.path.join(RAW_BEFORE_DIR, f"scene_{scene_id}_over.png")
        cv2.imwrite(bright_path, img_bright)
        os.utime(bright_path, (scene_base_time + 2, scene_base_time + 2))
        
        # 2. Tạo AFTER tương ứng với các biến đổi khác nhau
        if scene_id in [1, 2]:
            # Cảnh 1 & 2: Chỉnh màu sắc toàn cục
            after_img = apply_color_edit(base_img)
        elif scene_id in [3, 4]:
            # Cảnh 3 & 4: Thay đổi bầu trời
            after_img = apply_sky_replacement(base_img)
        else:
            # Cảnh 5: Xóa vật thể
            after_img = apply_object_removal(base_img)
            
        # Áp dụng lệch hình học
        after_img = apply_geometric_distortion(after_img)
        
        # Lưu ảnh After với tên ngẫu nhiên để test khả năng re-pair tự động
        random_name = f"edited_photo_{random.randint(1000, 9999)}.png"
        after_path = os.path.join(RAW_AFTER_DIR, random_name)
        cv2.imwrite(after_path, after_img)
        os.utime(after_path, (scene_base_time + 3, scene_base_time + 3))
        
    # 3. Tạo các ảnh nhiễu/lẻ (Unmatched)
    # 1 ảnh before lẻ
    before_noise = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(RAW_BEFORE_DIR, "noise_before.png"), before_noise)
    # 1 ảnh after lẻ
    after_noise = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(RAW_AFTER_DIR, "noise_after.png"), after_noise)
    
    print("[+] Tạo dữ liệu test giả lập thành công!")

if __name__ == "__main__":
    from datetime import datetime
    generate_mock_dataset()
