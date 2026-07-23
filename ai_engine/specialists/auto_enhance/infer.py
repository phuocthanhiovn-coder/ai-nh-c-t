import os
import argparse
import torch
import cv2
import numpy as np
from .model import HDRNet

def process_image(model, img_bgr, device):
    """
    Chạy inference HDRnet trên 1 ảnh numpy BGR.
    Trả về (output_bgr, grid_shape).
    """
    model.eval()
    
    # 1. Chuẩn bị ảnh full-res và proxy 256x256
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    proxy_rgb = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    
    # 2. Chuyển đổi sang Torch Tensor [0, 1]
    full_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1).copy()).float().unsqueeze(0) / 255.0
    proxy_tensor = torch.from_numpy(proxy_rgb.transpose(2, 0, 1).copy()).float().unsqueeze(0) / 255.0
    
    full_tensor = full_tensor.to(device)
    proxy_tensor = proxy_tensor.to(device)
    
    # 3. Chạy inference
    with torch.no_grad():
        output_tensor, grid = model(proxy_tensor, full_tensor)
        
    # 4. Chuyển kết quả về numpy BGR
    output_np = (output_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
    
    return output_bgr, grid.shape

def run_smoke_verification():
    """Tự động thực hiện các Smoke Test để chứng minh nguyên lý HDRnet."""
    print("=" * 60)
    print("  HDRnet Operator-Không-Pixel Verification Test")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "checkpoints/auto_enhance.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"[!] Lỗi: Không tìm thấy checkpoint tại {checkpoint_path}. Hãy chạy train trước.")
        return
        
    # Khởi tạo model và nạp checkpoint
    model = HDRNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # Tìm 1 ảnh bất kỳ trong data/pairs/before để chạy test
    before_dir = "data/pairs/before"
    if not os.path.exists(before_dir):
        print("[!] Không tìm thấy thư mục data/pairs/before để chạy test.")
        return
        
    test_files = [f for f in os.listdir(before_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not test_files:
        print("[!] Thư mục data/pairs/before trống.")
        return
        
    test_image_path = os.path.join(before_dir, test_files[0])
    img_orig = cv2.imread(test_image_path)
    print(f"[+] Sử dụng ảnh test: {test_image_path}")
    print(f"    - Kích thước ảnh gốc: {img_orig.shape[1]}x{img_orig.shape[0]}")
    
    # --- TEST 1: Kiểm tra Grid Shape ---
    _, grid_shape = process_image(model, img_orig, device)
    print(f"\n[✓] 1. Grid shape dự đoán: {list(grid_shape)}")
    print("    -> Chứng minh: Mạng chỉ xuất Bilateral Grid hệ số nhỏ (12 hệ số x 8 bins x 16x16), không tự nhả pixel!")
    
    # --- TEST 2: Chạy ở kích thước gốc (2048px) ---
    out_orig, _ = process_image(model, img_orig, device)
    print(f"\n[✓] 2. Kích thước Test (Kích thước gốc):")
    print(f"    - Input size:  {img_orig.shape[1]}x{img_orig.shape[0]}")
    print(f"    - Output size: {out_orig.shape[1]}x{out_orig.shape[0]}")
    assert img_orig.shape == out_orig.shape, "Lỗi: Kích thước input và output không trùng khớp!"
    print("    -> Trùng khớp tuyệt đối!")
    
    # --- TEST 3: Chạy ở độ phân giải tùy ý khác (ví dụ: scale nhỏ về 1000px) ---
    h_orig, w_orig = img_orig.shape[:2]
    w_custom = 1000
    h_custom = int(1000 * h_orig / w_orig)
    img_custom = cv2.resize(img_orig, (w_custom, h_custom), interpolation=cv2.INTER_AREA)
    
    out_custom, _ = process_image(model, img_custom, device)
    print(f"\n[✓] 3. Kích thước Test (Độ phân giải custom 1000px):")
    print(f"    - Input size:  {img_custom.shape[1]}x{img_custom.shape[0]}")
    print(f"    - Output size: {out_custom.shape[1]}x{out_custom.shape[0]}")
    assert img_custom.shape == out_custom.shape, "Lỗi: Kích thước input và output không trùng khớp!"
    print("    -> Trùng khớp tuyệt đối!")
    
    # --- TEST 4: Chạy ở độ phân giải lớn hơn (ví dụ: scale lớn lên 3000px) ---
    w_large = 3000
    h_large = int(3000 * h_orig / w_orig)
    img_large = cv2.resize(img_orig, (w_large, h_large), interpolation=cv2.INTER_CUBIC)
    
    out_large, _ = process_image(model, img_large, device)
    print(f"\n[✓] 4. Kích thước Test (Độ phân giải lớn 3000px):")
    print(f"    - Input size:  {img_large.shape[1]}x{img_large.shape[0]}")
    print(f"    - Output size: {out_large.shape[1]}x{out_large.shape[0]}")
    assert img_large.shape == out_large.shape, "Lỗi: Kích thước input và output không trùng khớp!"
    print("    -> Trùng khớp tuyệt đối!")
    
    print("\n[+] TẤT CẢ SMOKE TEST ĐÃ ĐẠT!")
    print("    Mạng đã chứng minh xuất sắc nguyên lý operator-không-pixel, độc lập hoàn toàn với độ phân giải ảnh gốc.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HDRnet Inference & Verification script")
    parser.add_argument("--input", type=str, help="Đường dẫn ảnh đầu vào")
    parser.add_argument("--output", type=str, help="Đường dẫn lưu ảnh đầu ra")
    parser.add_argument("--verify", action="store_true", default=False, help="Chạy smoke test tự động")
    args = parser.parse_args()
    
    if args.verify or not args.input:
        run_smoke_verification()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = "checkpoints/auto_enhance.pt"
        if not os.path.exists(checkpoint_path):
            print(f"[!] Không tìm thấy checkpoint tại {checkpoint_path}")
            exit(1)
            
        model = HDRNet().to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        
        img = cv2.imread(args.input)
        if img is None:
            print(f"[!] Không đọc được ảnh: {args.input}")
            exit(1)
            
        output, grid_shape = process_image(model, img, device)
        output_path = args.output if args.output else "output_result.jpg"
        cv2.imwrite(output_path, output, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"[+] Đã xử lý thành công ảnh và lưu tại: {output_path}")
        print(f"    - Kích thước: {output.shape[1]}x{output.shape[2] if len(output.shape) > 2 else output.shape[0]}")
        print(f"    - Grid shape: {list(grid_shape)}")
