import torch
import torch.nn as nn
import torch.nn.functional as F

class GuidanceMap(nn.Module):
    """
    Tạo Guidance Map (Bản đồ dẫn đường) 1 kênh [0, 1] từ ảnh full-res.
    Sử dụng Pointwise Conv 1x1 để học cách biến đổi ảnh 3 kênh màu RGB thành 1 kênh luminance tối ưu.
    """
    def __init__(self):
        super(GuidanceMap, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=1)
        self.conv2 = nn.Conv2d(16, 1, kernel_size=1)
        
    def forward(self, x):
        # x: [B, 3, H, W]
        out = F.relu(self.conv1(x))
        out = torch.sigmoid(self.conv2(out)) # Output ở dải [0, 1]
        return out

class CoefficientPredictor(nn.Module):
    """
    Mạng Conv chạy trên ảnh Proxy (256x256) để dự đoán Bilateral Grid chứa các hệ số affine.
    Đầu ra là Grid kích thước [B, 12, 8, 16, 16] (8 bins độ sáng, 16x16 không gian).
    """
    def __init__(self):
        super(CoefficientPredictor, self).__init__()
        # Conv layers để trích xuất đặc trưng
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1) # 128
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1) # 64
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # 32
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # 16
        
        # Nhánh local đặc trưng không gian
        self.local_conv = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        # Nhánh global đặc trưng toàn cục
        self.global_conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1) # 8
        self.global_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1) # 4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        
        # Gộp local + global trích xuất ra Grid hệ số
        self.fusion_conv = nn.Conv2d(256, 128, kernel_size=1)
        # 128 channels -> 12 * 8 = 96 channels (12 hệ số affine x 8 bins độ sáng)
        self.predict_conv = nn.Conv2d(128, 96, kernel_size=1)
        
    def forward(self, x):
        # x: proxy image [B, 3, 256, 256]
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3)) # [B, 128, 16, 16]
        
        # Trích xuất local
        local_feat = F.relu(self.local_conv(x4)) # [B, 128, 16, 16]
        
        # Trích xuất global
        g1 = F.relu(self.global_conv1(x4))
        g2 = F.relu(self.global_conv2(g1))
        g_flat = g2.view(g2.size(0), -1)
        g_fc1 = F.relu(self.fc1(g_flat))
        global_feat = F.relu(self.fc2(g_fc1)) # [B, 128]
        
        # Broadcast global feature và ghép với local
        global_feat_expanded = global_feat.view(global_feat.size(0), 128, 1, 1).expand(-1, -1, 16, 16)
        fusion = torch.cat([local_feat, global_feat_expanded], dim=1) # [B, 256, 16, 16]
        
        fusion_out = F.relu(self.fusion_conv(fusion))
        grid_flat = self.predict_conv(fusion_out) # [B, 96, 16, 16]
        
        # Reshape thành Bilateral Grid: [B, 12, 8, 16, 16]
        # 12: 12 coefficients, 8: luminance bins, 16: grid height, 16: grid width
        grid = grid_flat.view(grid_flat.size(0), 12, 8, 16, 16)
        return grid

class HDRNet(nn.Module):
    """
    Kiến trúc HDRnet hoàn chỉnh.
    Nhận proxy (256x256) + full_res_img.
    Dự đoán hệ số grid, slice grid theo guidance map của full_res_img, và áp dụng ma trận affine.
    """
    def __init__(self):
        super(HDRNet, self).__init__()
        self.predictor = CoefficientPredictor()
        self.guidance_net = GuidanceMap()
        
    def slice_grid(self, grid, guidance):
        """
        Khả vi Bilateral Slicing sử dụng torch.nn.functional.grid_sample.
        grid: [B, 12, 8, 16, 16]
        guidance: [B, 1, H, W] ở dải [0, 1]
        """
        B, C, D, gh, gw = grid.shape
        _, _, H, W = guidance.shape
        
        # 1. Tạo lưới tọa độ chuẩn hóa không gian X, Y về dải [-1, 1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=guidance.device),
            torch.linspace(-1, 1, W, device=guidance.device),
            indexing='ij'
        )
        
        # Reshape meshgrid sang [B, H, W, 1]
        grid_x = grid_x.view(1, H, W, 1).expand(B, -1, -1, -1)
        grid_y = grid_y.view(1, H, W, 1).expand(B, -1, -1, -1)
        
        # 2. Chuẩn hóa Guidance Map đóng vai trò tọa độ Z về dải [-1, 1]
        # Tọa độ Z tra cứu: z = guidance * 2 - 1
        grid_z = guidance.permute(0, 2, 3, 1) * 2.0 - 1.0 # [B, H, W, 1]
        
        # 3. Ghép thành Grid Grid-sample 3D: [B, H, W, 1, 3]
        # grid_sample 3D yêu cầu tọa độ định dạng (x, y, z) theo dải [-1, 1]
        query_grid = torch.cat([grid_x, grid_y, grid_z], dim=-1).unsqueeze(3) # [B, H, W, 1, 3]
        
        # 4. Thực hiện trilinear interpolation
        # grid_sample yêu cầu input là [B, C, D, H_g, W_g] và grid là [B, D_out, H_out, W_out, 3]
        # Ở đây ta coi D_out=H, H_out=W, W_out=1. query_grid là [B, H, W, 1, 3]
        sliced_flat = F.grid_sample(grid, query_grid, mode='bilinear', padding_mode='border', align_corners=True)
        # Sliced_flat: [B, 12, H, W, 1]
        
        sliced_grid = sliced_flat.squeeze(4) # [B, 12, H, W]
        return sliced_grid

    def apply_affine(self, sliced_grid, full_res_img):
        """
        Áp dụng ma trận affine 3x4 thu được lên ảnh gốc full-res.
        sliced_grid: [B, 12, H, W]
        full_res_img: [B, 3, H, W]
        """
        # Reshape sliced_grid sang dạng ma trận [B, 3, 4, H, W]
        # 12 coefficients được chia thành 3 hàng x 4 cột
        # Hàng 1: c0, c1, c2, c3
        # Hàng 2: c4, c5, c6, c7
        # Hàng 3: c8, c9, c10, c11
        coefficients = sliced_grid.view(sliced_grid.size(0), 3, 4, sliced_grid.size(2), sliced_grid.size(3))
        
        # Trích xuất ma trận màu 3x3 và vector tịnh tiến 3x1
        color_matrix = coefficients[:, :, :3, :, :] # [B, 3, 3, H, W]
        translation = coefficients[:, :, 3, :, :]   # [B, 3, H, W]
        
        # Nhân pointwise: output_rgb = color_matrix * input_rgb + translation
        # full_res_img: [B, 3, H, W]
        # Thêm chiều để nhân ma trận: [B, 1, 3, H, W] nhân với [B, 3, 3, H, W]
        input_unsqueezed = full_res_img.unsqueeze(1) # [B, 1, 3, H, W]
        
        # Nhân theo chiều channel: sum(input[B, 1, c_in, H, W] * matrix[B, c_out, c_in, H, W])
        transformed = torch.sum(input_unsqueezed * color_matrix, dim=2) # [B, 3, H, W]
        output = transformed + translation
        
        # Clip ảnh về dải [0, 1]
        return torch.clamp(output, 0.0, 1.0)

    def forward(self, proxy, full_res_img):
        # proxy: [B, 3, 256, 256]
        # full_res_img: [B, 3, H, W]
        
        # 1. Dự đoán Bilateral Grid từ proxy
        grid = self.predictor(proxy) # [B, 12, 8, 16, 16]
        
        # 2. Tạo Guidance Map từ full_res_img
        guidance = self.guidance_net(full_res_img) # [B, 1, H, W]
        
        # 3. Slice Grid lấy hệ số cục bộ tại mỗi pixel
        sliced_grid = self.slice_grid(grid, guidance) # [B, 12, H, W]
        
        # 4. Áp dụng hệ số lên ảnh gốc
        output = self.apply_affine(sliced_grid, full_res_img)
        
        return output, grid
