import os
import re
import csv
import shutil
import argparse
import rawpy
import cv2
import numpy as np
from .config import (
    PAIRS_COLOR_DIR, PAIRS_SKY_DIR, PAIRS_REMOVAL_DIR, UNMATCHED_DIR,
    OUTPUTS_SAMPLES_DIR, REPORT_CSV_PATH, INLIER_THRESHOLD, MIN_MATCH_COUNT, MIN_INLIER_RATIO,
    REVIEW_DIR, ALIGN_SCORE_THRESHOLD
)
from .align import extract_and_match
from .classify import classify_edit_type

def parse_prefix_number(filepath):
    """
    Trích xuất prefix (phần chữ trước số) và number (số cuối cùng) của file.
    Ví dụ: _ML_1682.CR3 -> prefix='_ML_', number=1682
           DSC01682.ARW -> prefix='DSC0', number=1682
           20260703-DSC04562.jpg -> prefix='20260703-DSC0', number=4562
    """
    filename = os.path.basename(filepath)
    name, _ = os.path.splitext(filename)
    match = re.search(r'^(.*?)(\d+)$', name)
    if match:
        prefix = match.group(1)
        number = int(match.group(2))
        return prefix, number
    return None, None

def get_all_files(root_dir, extensions):
    """Tìm tất cả các file có đuôi mở rộng trong extensions đệ quy."""
    file_list = []
    if not os.path.exists(root_dir):
        return file_list
    for root, _, files in os.walk(root_dir):
        for f in files:
            if any(f.lower().endswith(ext) for ext in extensions):
                file_list.append(os.path.join(root, f))
    return file_list

def resize_to_max(img, max_dim=2048):
    """Downscale ảnh giữ nguyên tỷ lệ sao cho cạnh dài nhất bằng max_dim."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    if h > w:
        new_h = max_dim
        new_w = int(w * (max_dim / h))
    else:
        new_w = max_dim
        new_h = int(h * (max_dim / w))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def align_bracket_images(images):
    """Căn khớp các frame bracket về frame THAM CHIẾU bằng homography ORB+RANSAC,
    frame không khớp được thì LOẠI (veto) — rồi mới MTB dọn rung tay còn sót.

    VÌ SAO (sửa 22/07/2026): bản cũ chỉ dùng cv2.AlignMTB — chỉ chỉnh dịch nhỏ
    vài pixel. Job k000 (data mới) khách chụp các frame LỆCH VỊ TRÍ lớn →
    MTB bó tay → Mertens gộp ra ảnh MA chồng 2-3 lớp (xem
    outputs/review_check_k000_*.jpg), cả 34 cảnh rớt align_low. Homography +
    veto trị tận gốc; xấu nhất còn 1 frame tham chiếu (vẫn là before hợp lệ).
    """
    if len(images) <= 1:
        return images

    # Tham chiếu: frame median-luma gần 128 nhất (đủ sáng, nhiều feature).
    meds = [abs(float(np.median(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))) - 128.0)
            for im in images]
    ref_i = int(np.argmin(meds))
    ref = images[ref_i]
    h, w = ref.shape[:2]

    # CLAHE để frame thiếu/dư sáng vẫn ra feature cho ORB.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_ref = clahe.apply(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY))

    out = []
    for i, im in enumerate(images):
        if i == ref_i:
            out.append(im.copy())
            continue
        gray = clahe.apply(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
        res = extract_and_match(gray, gray_ref)
        if res is None:
            continue
        kp1, kp2, matches = res
        if len(matches) < MIN_MATCH_COUNT:
            continue
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, INLIER_THRESHOLD)
        if H is None or mask is None or int(mask.sum()) < MIN_MATCH_COUNT:
            continue
        warped = cv2.warpPerspective(im, H, (w, h), borderMode=cv2.BORDER_REPLICATE)
        # Kiểm chứng sau warp bằng NCC cạnh (bất biến tone) — dưới sàn thì veto.
        ncc = compute_edge_ncc(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY),
                               cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY))
        if ncc < 0.30:
            continue
        out.append(warped)

    # MTB cuối chỉ dọn dịch nhỏ còn sót giữa các frame ĐÃ khớp homography.
    if len(out) >= 2:
        try:
            cv2.createAlignMTB().process(out, out)
        except cv2.error:
            pass
    return out

def process_cr3_to_rgb(cr3_path):
    """Đọc file RAW (CR3/DNG/ARW) bằng rawpy và develop thành ảnh RGB ở kích thước một nửa để tối ưu tốc độ."""
    with rawpy.imread(cr3_path) as raw:
        rgb = raw.postprocess(use_camera_wb=True, half_size=True)
        return rgb

def merge_exposures(images):
    """Gộp các ảnh phơi sáng (bracket) bằng Mertens Exposure Fusion.

    1 ảnh (các frame khác bị veto ở align_bracket_images) -> trả thẳng ảnh đó."""
    if len(images) == 1:
        return images[0].copy()
    merge_mertens = cv2.createMergeMertens()
    fusion = merge_mertens.process(images)
    fusion_8bit = np.clip(fusion * 255, 0, 255).astype(np.uint8)
    return fusion_8bit

def save_sample_visual(before_merged, aligned_before, after, output_path, edit_type):
    """Lưu ảnh mẫu panorama [before-merge | after | diff] ở kích thước nhỏ (max width ~1500px)."""
    target_w = 500
    target_h = int(500 * after.shape[0] / after.shape[1])
    
    before_resized = cv2.resize(before_merged, (target_w, target_h), interpolation=cv2.INTER_AREA)
    aligned_resized = cv2.resize(aligned_before, (target_w, target_h), interpolation=cv2.INTER_AREA)
    after_resized = cv2.resize(after, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    diff = cv2.absdiff(after_resized, aligned_resized)
    diff_enhanced = cv2.multiply(diff, 3)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(before_resized, "Before Merged", (20, 40), font, 0.8, (0, 0, 255), 2)
    cv2.putText(after_resized, f"After ({edit_type.upper()})", (20, 40), font, 0.8, (255, 0, 0), 2)
    cv2.putText(diff_enhanced, "Diff Map x3", (20, 40), font, 0.8, (0, 255, 255), 2)
    
    canvas = np.hstack((before_resized, after_resized, diff_enhanced))
    cv2.imwrite(output_path, canvas)

def compute_edge_gradient(gray_img):
    """Tính toán bản đồ cạnh Sobel gradient để bất biến với sự thay đổi tông màu."""
    sobelx = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(sobelx, sobely)
    grad_norm = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return grad_norm

def compute_edge_ncc(aligned_gray, after_gray):
    """Tính toán Normalized Cross-Correlation (NCC) trên bản đồ cạnh Sobel sử dụng TM_CCOEFF_NORMED."""
    grad_aligned = compute_edge_gradient(aligned_gray)
    grad_after = compute_edge_gradient(after_gray)
    
    res = cv2.matchTemplate(grad_aligned, grad_after, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return float(max_val)

def align_before_after(before_img, after_img):
    """
    Align ảnh before_img khớp khít với after_img sử dụng RANSAC Homography + ECC 512px.
    """
    gray_before = cv2.cvtColor(before_img, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(after_img, cv2.COLOR_BGR2GRAY)
    h, w = after_img.shape[:2]
    
    res = extract_and_match(gray_before, gray_after)
    if res is None:
        return 0.0, before_img.copy()
        
    kp1, kp2, matches = res
    if len(matches) < MIN_MATCH_COUNT:
        return 0.0, before_img.copy()
        
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, INLIER_THRESHOLD)
    if H is None or mask is None:
        return 0.0, before_img.copy()
        
    aligned_before_rough = cv2.warpPerspective(before_img, H, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # ECC Refine trên 512px
    aligned_gray_rough = cv2.cvtColor(aligned_before_rough, cv2.COLOR_BGR2GRAY)
    grad_before = compute_edge_gradient(aligned_gray_rough)
    grad_after = compute_edge_gradient(gray_after)
    
    target_dim = 512
    h_g, w_g = grad_after.shape[:2]
    scale_factor = target_dim / max(h_g, w_g)
    
    w_small = int(w_g * scale_factor)
    h_small = int(h_g * scale_factor)
    
    grad_before_small = cv2.resize(grad_before, (w_small, h_small), interpolation=cv2.INTER_AREA)
    grad_after_small = cv2.resize(grad_after, (w_small, h_small), interpolation=cv2.INTER_AREA)
    
    warp_matrix = np.eye(3, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, 1e-3)
    
    aligned_before_final = aligned_before_rough
    
    try:
        _, warp_matrix = cv2.findTransformECC(grad_after_small, grad_before_small, warp_matrix, cv2.MOTION_HOMOGRAPHY, criteria, None, 1)
        s = 1.0 / scale_factor
        warp_matrix_large = warp_matrix.copy()
        warp_matrix_large[0, 2] = warp_matrix[0, 2] * s
        warp_matrix_large[1, 2] = warp_matrix[1, 2] * s
        warp_matrix_large[2, 0] = warp_matrix[2, 0] / s
        warp_matrix_large[2, 1] = warp_matrix[2, 1] / s
        
        aligned_before_final = cv2.warpPerspective(aligned_before_rough, warp_matrix_large, (w, h), borderMode=cv2.BORDER_REPLICATE)
    except cv2.error:
        # Fallback Affine
        warp_matrix_affine = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp_matrix_affine = cv2.findTransformECC(grad_after_small, grad_before_small, warp_matrix_affine, cv2.MOTION_AFFINE, criteria, None, 1)
            s = 1.0 / scale_factor
            warp_matrix_affine_large = warp_matrix_affine.copy()
            warp_matrix_affine_large[0, 2] = warp_matrix_affine[0, 2] * s
            warp_matrix_affine_large[1, 2] = warp_matrix_affine[1, 2] * s
            
            aligned_before_final = cv2.warpAffine(aligned_before_rough, warp_matrix_affine_large, (w, h), borderMode=cv2.BORDER_REPLICATE)
        except cv2.error:
            pass
            
    final_gray = cv2.cvtColor(aligned_before_final, cv2.COLOR_BGR2GRAY)
    align_score = compute_edge_ncc(final_gray, gray_after)
    
    return align_score, aligned_before_final

def is_pair_processed(output_name):
    """Kiểm tra xem cặp ảnh này đã được xử lý và lưu ở bất kỳ thư mục đầu ra nào chưa."""
    for d in [PAIRS_COLOR_DIR, PAIRS_SKY_DIR, PAIRS_REMOVAL_DIR, REVIEW_DIR]:
        if os.path.exists(os.path.join(d, "before", output_name)):
            return True
    return False

def load_existing_report():
    """Nạp báo cáo cũ report.csv dưới dạng dict để cập nhật không trùng lặp."""
    existing_data = {}
    if not os.path.exists(REPORT_CSV_PATH):
        return existing_data
    try:
        with open(REPORT_CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                has_job_name = "job_name" in header
                for row in reader:
                    if len(row) < 7:
                        continue
                    if has_job_name:
                        job, prefix, num = row[0], row[1], int(row[2])
                        existing_data[(job, prefix, num)] = row
                    else:
                        # Bản cũ không có job_name
                        prefix = row[0]
                        try:
                            num = int(row[1])
                            # Thêm job_name mặc định là rỗng
                            existing_data[("", prefix, num)] = [""] + row
                        except ValueError:
                            continue
    except Exception as e:
        print(f"[!] CẢNH BÁO khi nạp report cũ: {str(e)}")
    return existing_data

def run_ingest(reset=False, before_root="data/raw/before", after_root="data/raw/after", job_name=None):
    """Chạy toàn bộ pipeline ingest raw brackets theo prefix và number."""
    print("=" * 60)
    print("  AutoHDR Dataset Ingestion - Prefix-Based Pairing")
    if job_name:
        print(f"  [Job Name]: {job_name}")
    print("=" * 60)
    
    if reset:
        print("[*] Tùy chọn --reset được bật. Đang dọn dẹp dữ liệu cũ...")
        for d in [PAIRS_COLOR_DIR, PAIRS_SKY_DIR, PAIRS_REMOVAL_DIR, UNMATCHED_DIR, OUTPUTS_SAMPLES_DIR, REVIEW_DIR]:
            if os.path.exists(d):
                shutil.rmtree(d)
        if os.path.exists(REPORT_CSV_PATH):
            os.remove(REPORT_CSV_PATH)
            
    for d in [PAIRS_COLOR_DIR, PAIRS_SKY_DIR, PAIRS_REMOVAL_DIR, UNMATCHED_DIR, OUTPUTS_SAMPLES_DIR, REVIEW_DIR]:
        if d != OUTPUTS_SAMPLES_DIR:
            os.makedirs(os.path.join(d, "before"), exist_ok=True)
            os.makedirs(os.path.join(d, "after"), exist_ok=True)
        else:
            os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.dirname(REPORT_CSV_PATH), exist_ok=True)
    
    report_dict = {} if reset else load_existing_report()
    
    print(f"[*] Đang quét các file thô tại: {before_root} và {after_root}...")
    # Đọc đệ quy .cr3, .dng, .arw
    before_paths = get_all_files(before_root, [".cr3", ".dng", ".arw"])
    after_paths = get_all_files(after_root, [".jpg", ".jpeg"])
    
    print(f"[+] Tìm thấy {len(before_paths)} file RAW (.CR3/.DNG/.ARW)")
    print(f"[+] Tìm thấy {len(after_paths)} file AFTER (.jpg/.jpeg)")
    
    # Indexing theo prefix
    before_index = {}
    for path in before_paths:
        prefix, num = parse_prefix_number(path)
        if prefix is not None and num is not None:
            before_index.setdefault(prefix, []).append((num, path))
            
    after_index = {}
    for path in after_paths:
        prefix, num = parse_prefix_number(path)
        if prefix is not None and num is not None:
            after_index.setdefault(prefix, []).append((num, path))
            
    # Sắp xếp số thứ tự tăng dần
    for k in before_index:
        before_index[k].sort(key=lambda x: x[0])
    for k in after_index:
        after_index[k].sort(key=lambda x: x[0])
        
    print(f"[+] Indexing thành công: {len(before_index)} prefix máy ở BEFORE, {len(after_index)} prefix ở AFTER.")
    
    matched_pairs_count = 0
    skipped_pairs_count = 0
    unmatched_after = []
    
    job_key = job_name if job_name else ""
    
    # 3. Gom và xử lý từng nhóm prefix tương ứng
    for prefix, after_list in after_index.items():
        if prefix not in before_index:
            print(f"[!] CẢNH BÁO: Prefix '{prefix}' có trong AFTER nhưng không có tương ứng ở BEFORE.")
            for num, path in after_list:
                unmatched_after.append((prefix, num, path))
            continue
            
        before_list = before_index[prefix]
        print(f"\n[*] Đang xử lý nhóm prefix: {prefix} ({len(after_list)} ảnh sau, {len(before_list)} ảnh thô)")
        
        prefix_clean = re.sub(r'[^a-zA-Z0-9_-]', '_', prefix)
        
        for idx, (num, after_path) in enumerate(after_list):
            start_num = num
            end_num = after_list[idx + 1][0] if idx + 1 < len(after_list) else float('inf')
            
            if job_name:
                output_name = f"{job_name}_{prefix_clean}{num}.jpg"
            else:
                output_name = f"{prefix_clean}{num}.jpg"
            
            if not reset and is_pair_processed(output_name):
                skipped_pairs_count += 1
                matched_pairs_count += 1
                continue
                
            bracket_files = [path for b_num, path in before_list if start_num <= b_num < end_num]
            bracket_size = len(bracket_files)
            
            if bracket_size == 0:
                print(f"  [✗] Không tìm thấy file RAW nào cho after {num} (prefix={prefix}).")
                unmatched_after.append((prefix, num, after_path))
                continue
                
            print(f"  [-] Cảnh after {num}: tìm thấy bộ bracket gồm {bracket_size} ảnh RAW.")
            
            try:
                raw_images = []
                for cr3_path in bracket_files:
                    rgb = process_cr3_to_rgb(cr3_path)
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    raw_images.append(bgr)
                    
                aligned_raws = align_bracket_images(raw_images)
                before_merged = merge_exposures(aligned_raws)
                
                after_img = cv2.imread(after_path)
                if after_img is None:
                    print(f"  [✗] Lỗi không đọc được ảnh after: {after_path}")
                    continue
                    
                before_merged_resized = resize_to_max(before_merged, 2048)
                after_img_resized = resize_to_max(after_img, 2048)
                
                align_score, aligned_before = align_before_after(before_merged_resized, after_img_resized)
                edit_type, confidence = classify_edit_type(aligned_before, after_img_resized)
                
                if align_score >= ALIGN_SCORE_THRESHOLD:
                    align_status = "align_ok"
                    if edit_type == "sky":
                        dest_dir = PAIRS_SKY_DIR
                    elif edit_type == "removal":
                        dest_dir = PAIRS_REMOVAL_DIR
                    else:
                        dest_dir = PAIRS_COLOR_DIR
                else:
                    align_status = "align_low"
                    dest_dir = REVIEW_DIR
                    
                print(f"    [✓] Căn chỉnh: {align_status} (align_score: {align_score:.4f}) | Phân loại: {edit_type} (conf: {confidence:.2f})")
                
                cv2.imwrite(os.path.join(dest_dir, "before", output_name), aligned_before, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(os.path.join(dest_dir, "after", output_name), after_img_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                if job_name:
                    sample_path = os.path.join(OUTPUTS_SAMPLES_DIR, f"sample_{job_name}_{prefix_clean}{num}.jpg")
                else:
                    sample_path = os.path.join(OUTPUTS_SAMPLES_DIR, f"sample_{prefix_clean}{num}.jpg")
                save_sample_visual(before_merged, aligned_before, after_img, sample_path, edit_type)
                
                report_dict[(job_key, prefix, num)] = [
                    job_key, prefix, num, bracket_size, edit_type, f"{confidence:.4f}", align_status, f"{align_score:.4f}"
                ]
                matched_pairs_count += 1
                
            except Exception as e:
                print(f"  [✗] LỖI khi xử lý cảnh {num} (prefix={prefix}): {str(e)}")
                
    print("\n[*] Xử lý ảnh không khớp...")
    for prefix, num, path in unmatched_after:
        prefix_clean = re.sub(r'[^a-zA-Z0-9_-]', '_', prefix)
        if job_name:
            dest_name = f"{job_name}_{prefix_clean}{num}.jpg"
        else:
            dest_name = f"{prefix_clean}{num}.jpg"
        dest_path = os.path.join(UNMATCHED_DIR, "after", dest_name)
        shutil.copy(path, dest_path)
        
    with open(REPORT_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["job_name", "prefix", "number", "bracket_size", "edit_type", "confidence", "align_status", "align_score"])
        sorted_keys = sorted(list(report_dict.keys()), key=lambda x: (x[0], x[1], x[2]))
        for key in sorted_keys:
            writer.writerow(report_dict[key])
            
    total_rows = list(report_dict.values())
    train_count = sum(1 for r in total_rows if r[6] == "align_ok")
    review_count = sum(1 for r in total_rows if r[6] == "align_low")
    
    print(f"\n[+] INGEST HOÀN THÀNH!")
    print(f"  - Tổng số cặp thành công tích lũy: {len(total_rows)}")
    print(f"  - Số cặp đủ điều kiện TRAIN (align_ok): {train_count}")
    print(f"  - Số cặp cần REVIEW (align_low): {review_count}")
    print(f"  - Đã xử lý mới trong lượt này: {matched_pairs_count - skipped_pairs_count} cặp (Bỏ qua đã có: {skipped_pairs_count} cặp).")
    print(f"  - Số ảnh sau unmatched: {len(unmatched_after)}")
    print(f"  - Báo cáo lưu tại: {REPORT_CSV_PATH}")
    return len(total_rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoHDR Ingest Engine - Prefix-Based Pairing")
    parser.add_argument("--reset", action="store_true", help="Xóa sạch dữ liệu cũ và ingest lại từ đầu")
    parser.add_argument("--before-root", type=str, default="data/raw/before", help="Thư mục chứa ảnh raw before")
    parser.add_argument("--after-root", type=str, default="data/raw/after", help="Thư mục chứa ảnh after")
    parser.add_argument("--job-name", type=str, default=None, help="Tên của job tải về")
    args = parser.parse_args()
    run_ingest(reset=args.reset, before_root=args.before_root, after_root=args.after_root, job_name=args.job_name)
