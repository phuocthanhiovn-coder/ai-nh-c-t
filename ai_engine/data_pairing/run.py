import os
import csv
import cv2
import numpy as np
import shutil
from .config import (
    RAW_BEFORE_DIR, RAW_AFTER_DIR, PAIRS_COLOR_DIR, PAIRS_SKY_DIR,
    PAIRS_REMOVAL_DIR, UNMATCHED_DIR, OUTPUTS_SAMPLES_DIR, REPORT_CSV_PATH,
    BRACKET_TIME_THRESHOLD, PHASH_THRESHOLD
)
from .bracket_group import group_brackets, select_representative
from .match import find_shortlist
from .align import verify_and_align
from .classify import classify_edit_type

def ensure_dirs():
    """Tạo các thư mục cần thiết nếu chưa có."""
    for d in [PAIRS_COLOR_DIR, PAIRS_SKY_DIR, PAIRS_REMOVAL_DIR, UNMATCHED_DIR, OUTPUTS_SAMPLES_DIR]:
        # Cần tạo cả thư mục con before và after bên trong
        if d != OUTPUTS_SAMPLES_DIR:
            os.makedirs(os.path.join(d, "before"), exist_ok=True)
            os.makedirs(os.path.join(d, "after"), exist_ok=True)
        else:
            os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.dirname(REPORT_CSV_PATH), exist_ok=True)

def save_sample_visual(before_path, aligned_before, after, output_path, edit_type):
    """
    Tạo và lưu ảnh so sánh mẫu dạng [before | after-aligned | diff-map]
    để người dùng kiểm tra trực quan.
    """
    img_before = cv2.imread(before_path)
    img_before = cv2.resize(img_before, (aligned_before.shape[1], aligned_before.shape[0]))
    
    # Tính diff map trực quan màu sắc
    diff = cv2.absdiff(after, aligned_before)
    # Tăng cường độ sáng của diff map lên để dễ quan sát
    diff_enhanced = cv2.multiply(diff, 3)
    
    # Đặt nhãn văn bản lên ảnh
    h, w, _ = after.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_before, "Original Before", (20, 40), font, 1.0, (0, 0, 255), 2)
    cv2.putText(aligned_before, "Aligned Before", (20, 40), font, 1.0, (0, 255, 0), 2)
    cv2.putText(after, f"After ({edit_type.upper()})", (20, 40), font, 1.0, (255, 0, 0), 2)
    
    # Ghép 4 tấm thành 1 tấm panorama ngang: [before | aligned_before | after | diff]
    canvas = np.hstack((img_before, aligned_before, after, diff_enhanced))
    cv2.imwrite(output_path, canvas)

def run_pairing():
    """Chạy toàn bộ pipeline ghép cặp dataset."""
    ensure_dirs()
    
    print("[*] Quét thư mục raw/before và raw/after...")
    if not os.path.exists(RAW_BEFORE_DIR) or not os.path.exists(RAW_AFTER_DIR):
        print("[!] Không tìm thấy thư mục raw/before hoặc raw/after. Hãy chạy self-test trước.")
        return
        
    before_files = [os.path.join(RAW_BEFORE_DIR, f) for f in os.listdir(RAW_BEFORE_DIR) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
    after_files = [os.path.join(RAW_AFTER_DIR, f) for f in os.listdir(RAW_AFTER_DIR) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
                   
    print(f"[*] Tìm thấy {len(before_files)} ảnh before và {len(after_files)} ảnh after thô.")
    
    # 1. Gom bracket
    print("[*] Đang gom nhóm bracket phơi sáng...")
    bracket_groups = group_brackets(before_files, BRACKET_TIME_THRESHOLD)
    representatives = [select_representative(g) for g in bracket_groups]
    print(f"[+] Đã gom được {len(bracket_groups)} nhóm bracket. Chọn {len(representatives)} ảnh đại diện.")
    
    matched_pairs = []
    matched_after_files = set()
    unmatched_before = []
    
    # 2. Xử lý khớp từng ảnh đại diện
    for before_path in representatives:
        filename = os.path.basename(before_path)
        print(f"\n[*] Đang tìm cặp cho: {filename}...")
        
        # Lấy danh sách after chưa bị khớp
        available_after = [f for f in after_files if f not in matched_after_files]
        
        # Tìm shortlist bằng pHash
        shortlist = find_shortlist(before_path, available_after, PHASH_THRESHOLD)
        
        # Tạo danh sách ứng viên (sau khi loại bỏ trùng trong shortlist)
        shortlist_paths = [path for path, _ in shortlist]
        fallback_candidates = [path for path in available_after if path not in shortlist_paths]
        
        # Danh sách tất cả các ứng viên thử nghiệm: shortlist trước, fallback sau
        candidates_to_try = [(path, "phash") for path in shortlist_paths] + [(path, "orb_fallback") for path in fallback_candidates]
        
        matched = False
        for after_path, method in candidates_to_try:
            # Verify bằng Homography RANSAC
            is_ok, aligned_before, inlier_ratio, H = verify_and_align(before_path, after_path)
            
            if is_ok:
                after_filename = os.path.basename(after_path)
                # Phân loại kiểu edit
                after_img = cv2.imread(after_path)
                edit_type, confidence = classify_edit_type(aligned_before, after_img)
                
                print(f"  [✓] Khớp khít với {after_filename} ({method})! Loai: {edit_type} (conf: {confidence:.2f}, ratio: {inlier_ratio:.2f})")
                
                # Xác định thư mục đích dựa trên loại edit
                if edit_type == "sky":
                    dest_dir = PAIRS_SKY_DIR
                elif edit_type == "removal":
                    dest_dir = PAIRS_REMOVAL_DIR
                else:
                    dest_dir = PAIRS_COLOR_DIR
                    
                # Lưu file aligned before và copy after sang với cùng tên mới thống nhất
                common_name = f"pair_{len(matched_pairs):04d}.png"
                
                cv2.imwrite(os.path.join(dest_dir, "before", common_name), aligned_before)
                shutil.copy(after_path, os.path.join(dest_dir, "after", common_name))
                
                # Lưu mẫu trực quan
                sample_path = os.path.join(OUTPUTS_SAMPLES_DIR, f"sample_{len(matched_pairs):04d}.png")
                save_sample_visual(before_path, aligned_before, after_img, sample_path, edit_type)
                
                # Ghi nhận kết quả
                matched_pairs.append({
                    "before_orig": filename,
                    "after_orig": after_filename,
                    "match_score": f"{inlier_ratio:.4f}",
                    "edit_type": edit_type,
                    "confidence": f"{confidence:.4f}",
                    "aligned_path": os.path.join(dest_dir, "before", common_name)
                })
                
                matched_after_files.add(after_path)
                matched = True
                break
                
        if not matched:
            print(f"  [✗] Không tìm thấy cặp khớp hợp lệ.")
            unmatched_before.append(before_path)
            
    # 3. Gom và xử lý ảnh lẻ không khớp
    print("\n[*] Xử lý các ảnh không tìm thấy cặp...")
    for unmatched_b in unmatched_before:
        name = os.path.basename(unmatched_b)
        shutil.copy(unmatched_b, os.path.join(UNMATCHED_DIR, "before", name))
        
    for after_path in after_files:
        if after_path not in matched_after_files:
            name = os.path.basename(after_path)
            shutil.copy(after_path, os.path.join(UNMATCHED_DIR, "after", name))
            
    # 4. Ghi file báo cáo csv
    with open(REPORT_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["before_original", "after_original", "match_score", "edit_type", "confidence", "aligned_output"])
        for p in matched_pairs:
            writer.writerow([
                p["before_orig"], p["after_orig"], p["match_score"], 
                p["edit_type"], p["confidence"], p["aligned_path"]
            ])
            
    print(f"\n[+] HOÀN THÀNH: Ghép thành công {len(matched_pairs)} cặp.")
    print(f"[+] Báo cáo lưu tại: {REPORT_CSV_PATH}")
    return matched_pairs

if __name__ == "__main__":
    run_pairing()
