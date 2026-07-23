import rawpy
import cv2
import os

def test_read_cr3(cr3_path):
    print(f"[*] Đang đọc thử file CR3: {cr3_path}...")
    if not os.path.exists(cr3_path):
        print(f"[!] File không tồn tại: {cr3_path}")
        return False
        
    try:
        with rawpy.imread(cr3_path) as raw:
            # Develop ảnh thô sang RGB 8-bit bằng LibRaw mặc định
            print("[*] Đang develop ảnh RAW...")
            rgb = raw.postprocess(use_camera_wb=True, half_size=True)
            print(f"[+] Đọc thành công! Kích thước ảnh: {rgb.shape}")
            
            # Lưu thử sang ảnh JPG để kiểm tra chất lượng
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            output_test = "outputs/cr3_test_out.jpg"
            os.makedirs("outputs", exist_ok=True)
            cv2.imwrite(output_test, bgr)
            print(f"[+] Đã lưu ảnh test thành công tại: {output_test}")
            return True
    except Exception as e:
        print(f"[!] Lỗi khi đọc file CR3: {str(e)}")
        return False

if __name__ == "__main__":
    test_path = "data/raw/before/_ML_1591.CR3"
    test_read_cr3(test_path)
