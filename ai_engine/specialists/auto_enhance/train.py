import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import cv2
import numpy as np
from .model import HDRNet
from .dataset import HDRDataset

def save_smoke_compare(model, dataset, device, output_path, num_samples=3):
    """Lưu ảnh mẫu panorama [before | output | after] để kiểm chứng trực quan."""
    model.eval()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    samples = []
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            before, proxy, after = dataset[i]
            # Thêm batch dimension
            before_batch = before.unsqueeze(0).to(device)
            proxy_batch = proxy.unsqueeze(0).to(device)
            
            output_batch, _ = model(proxy_batch, before_batch)
            output = output_batch.squeeze(0).cpu()
            
            # Chuyển tensor sang numpy HWC uint8 RGB
            before_np = (before.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            output_np = (output.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            after_np = (after.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            
            # Đổi RGB sang BGR cho OpenCV
            before_bgr = cv2.cvtColor(before_np, cv2.COLOR_RGB2BGR)
            output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
            after_bgr = cv2.cvtColor(after_np, cv2.COLOR_RGB2BGR)
            
            # Đảm bảo 3 ảnh cùng kích thước
            h, w = after_bgr.shape[:2]
            before_bgr = cv2.resize(before_bgr, (w, h), interpolation=cv2.INTER_AREA)
            output_bgr = cv2.resize(output_bgr, (w, h), interpolation=cv2.INTER_AREA)
            
            # Vẽ chữ ghi chú
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(before_bgr, "Before", (30, 60), font, 1.5, (0, 0, 255), 3)
            cv2.putText(output_bgr, "Output (HDRnet)", (30, 60), font, 1.5, (0, 255, 0), 3)
            cv2.putText(after_bgr, "After (Target)", (30, 60), font, 1.5, (255, 0, 0), 3)
            
            # Ghép ngang
            canvas = np.hstack((before_bgr, output_bgr, after_bgr))
            # Downscale panorama về width 1500px để nhẹ
            target_w = 1500
            target_h = int(1500 * canvas.shape[0] / canvas.shape[1])
            canvas_small = cv2.resize(canvas, (target_w, target_h), interpolation=cv2.INTER_AREA)
            samples.append(canvas_small)
            
    # Ghép dọc các mẫu
    final_canvas = np.vstack(samples)
    cv2.imwrite(output_path, final_canvas)
    print(f"[+] Đã lưu ảnh so sánh kiểm chứng tại: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="HDRnet Model Training Pilot - Overfit 8 pairs")
    parser.add_argument("--epochs", type=int, default=150, help="Số lượng epochs training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (mặc định 1 vì các cặp ảnh khác kích thước, không batch chung được)")
    parser.add_argument("--smoke", action="store_true", help="Chạy nhanh smoke test vài epoch")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Đang sử dụng thiết bị: {device}")
    
    # 1. Load Dataset
    # Tắt augmentation lật ngang khi overfit 8 cặp để nó học vẹt nhanh nhất có thể
    dataset = HDRDataset(data_dir="data/pairs", is_train=False)
    if len(dataset) == 0:
        print("[!] Không tìm thấy dữ liệu trong data/pairs/before. Hãy đảm bảo chạy Task 02 trước.")
        return
        
    print(f"[+] Tìm thấy {len(dataset)} cặp ảnh sạch phục vụ overfit.")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # 2. Khởi tạo Model, Loss, Optimizer
    model = HDRNet().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    epochs = 10 if args.smoke else args.epochs
    print(f"[*] Bắt đầu huấn luyện overfit {len(dataset)} cặp trong {epochs} epochs...")
    
    model.train()
    first_loss = None
    last_loss = None
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for before, proxy, after in dataloader:
            before = before.to(device)
            proxy = proxy.to(device)
            after = after.to(device)
            
            optimizer.zero_grad()
            output, _ = model(proxy, before)
            loss = criterion(output, after)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * before.size(0)
            
        epoch_loss /= len(dataset)
        
        if epoch == 1:
            first_loss = epoch_loss
        last_loss = epoch_loss
        
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:03d}/{epochs:03d} | Loss (L1): {epoch_loss:.6f}")
            
    print(f"[+] Kết quả overfit: Loss ban đầu: {first_loss:.6f} -> Loss cuối cùng: {last_loss:.6f}")
    
    # 3. Lưu checkpoint
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "auto_enhance.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"[+] Đã lưu checkpoint tại: {checkpoint_path}")
    
    # 4. Lưu ảnh mẫu so sánh kiểm chứng
    save_smoke_compare(model, dataset, device, "outputs/smoke_compare.png", num_samples=3)

if __name__ == "__main__":
    main()
