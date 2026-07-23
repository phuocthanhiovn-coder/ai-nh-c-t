import os
import csv
from ai_engine.data_pairing.config import REPORT_CSV_PATH, PAIRS_COLOR_DIR, PAIRS_SKY_DIR, PAIRS_REMOVAL_DIR, UNMATCHED_DIR

def run_assertions():
    """Kiểm tra tính đúng đắn của pipeline ghép cặp."""
    print("\n[*] Đang chạy kiểm chứng (Assertions) kết quả self-test...")
    
    # 1. Kiểm tra sự tồn tại của file report.csv
    assert os.path.exists(REPORT_CSV_PATH), f"LỖI: Không tìm thấy file báo cáo {REPORT_CSV_PATH}"
    
    # Đọc báo cáo bằng csv module chuẩn
    rows = []
    with open(REPORT_CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            
    print(f"[+] Đọc báo cáo: tìm thấy {len(rows)} cặp được ghi nhận.")
    
    # 2. Assert số lượng cặp ghép được phải bằng đúng 5 (vì có 5 scene gốc)
    assert len(rows) == 5, f"LỖI: Số lượng cặp ghép được là {len(rows)}, yêu cầu đúng 5 cặp."
    
    # 3. Assert phân loại đúng loại edit cho từng scene
    for row in rows:
        before_orig = row['before_original']
        after_orig = row['after_original']
        edit_type = row['edit_type']
        
        print(f"  - Cặp: {before_orig} <-> {after_orig} | Loại edit phát hiện: {edit_type}")
        
        if "scene_1" in before_orig or "scene_2" in before_orig:
            assert edit_type == "color", f"LỖI: {before_orig} đáng nhẽ phải là color nhưng phát hiện là {edit_type}"
        elif "scene_3" in before_orig or "scene_4" in before_orig:
            assert edit_type == "sky", f"LỖI: {before_orig} đáng nhẽ phải là sky nhưng phát hiện là {edit_type}"
        elif "scene_5" in before_orig:
            assert edit_type == "removal", f"LỖI: {before_orig} đáng nhẽ phải là removal nhưng phát hiện là {edit_type}"
            
    # 4. Kiểm tra ảnh nhiễu đã được đưa vào unmatched chưa
    before_unmatched_dir = os.path.join(UNMATCHED_DIR, "before")
    after_unmatched_dir = os.path.join(UNMATCHED_DIR, "after")
    
    before_unmatched_files = os.listdir(before_unmatched_dir)
    after_unmatched_files = os.listdir(after_unmatched_dir)
    
    print(f"[+] Ảnh unmatched: Before={before_unmatched_files}, After={after_unmatched_files}")
    
    assert "noise_before.png" in before_unmatched_files, "LỖI: Không tìm thấy noise_before.png trong unmatched/before"
    assert "noise_after.png" in after_unmatched_files, "LỖI: Không tìm thấy noise_after.png trong unmatched/after"
    
    # 5. Kiểm tra các thư mục chứa cặp ghép thành công có tồn tại ảnh đầu ra không
    for d, expected_type in [(PAIRS_COLOR_DIR, "color"), (PAIRS_SKY_DIR, "sky"), (PAIRS_REMOVAL_DIR, "removal")]:
        before_dir = os.path.join(d, "before")
        after_dir = os.path.join(d, "after")
        
        b_files = os.listdir(before_dir)
        a_files = os.listdir(after_dir)
        
        assert len(b_files) == len(a_files), f"LỖI: Số lượng file before/after lệch nhau ở thư mục {d}"
        print(f"  - Thư mục {expected_type}: tìm thấy {len(b_files)} ảnh.")
        
    print("\n[✓] TẤT CẢ KIỂM TRA ĐỀU PASS (SMOKE TEST THÀNH CÔNG RỰC RỠ)!")

if __name__ == "__main__":
    run_assertions()
