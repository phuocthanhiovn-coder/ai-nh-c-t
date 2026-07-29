"""NM — kiến trúc MỚI 29/07 (task 28 T3, thí nghiệm E3): thay bilateral grid.

Vì sao (0/10 vòng 1 SO_DIEM + khảo sát 3 mũi): HDRNet lưới trơn không xử được
"chỗ sáng chỗ tối" (cần gain KHÔNG GIAN thật) và màu giàu (cần LUT tự do theo hue).
NM = 2 tầng OPERATOR nối tiếp, đúng nguyên tắc "xuất operator không xuất pixel":

  1. GAIN chiếu sáng: U-Net nhỏ trên proxy (448) + ngữ cảnh toàn cục (avg-pool
     broadcast — cho model "nhìn cả phòng" khi quyết định vùng nào nâng/hạ)
     → log-gain 3 kênh ở res/4, exp(1.4·tanh) ∈ [0.25, 4.1]. Bản chất tần thấp
     → upsample an toàn lên full-res (train: bilinear; deploy: guided filter).
  2. MÀU: Image-Adaptive 3D-LUT (tái dùng model_lut, Apache-style tự viết) áp lên
     ảnh ĐÃ gain — LUT lo punch theo hue/vật liệu, gain lo sáng đều.

Chưa có mask ngữ nghĩa ở v1 (sẽ thêm kênh input ở v2 — cần cache mask gsam
cho toàn dataset). Interface khớp _run_epoch: forward(proxy, full) -> ảnh full.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_lut import _identity_lut


class _SE(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(ch, ch // 4), nn.ReLU(inplace=True),
                                nn.Linear(ch // 4, ch), nn.Sigmoid())

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return x * self.fc(s)[:, :, None, None]


def _block(ci, co, stride=1):
    return nn.Sequential(nn.Conv2d(ci, co, 3, stride, 1),
                         nn.GroupNorm(min(8, co), co),
                         nn.SiLU(inplace=True))


class NMNet(nn.Module):
    def __init__(self, base=32, n_basis=6, lut_dim=17, proxy_res=448, gain_amp=1.4,
                 **_ignore):
        super().__init__()
        self.proxy_res = proxy_res
        self.gain_amp = float(gain_amp)
        b = base
        # encoder proxy 448 -> /2 /4 /8 /16
        self.e1 = nn.Sequential(_block(3, b), _block(b, b))            # 448
        self.e2 = nn.Sequential(_block(b, b * 2, 2), _block(b * 2, b * 2), _SE(b * 2))    # 224
        self.e3 = nn.Sequential(_block(b * 2, b * 4, 2), _block(b * 4, b * 4), _SE(b * 4))  # 112
        self.e4 = nn.Sequential(_block(b * 4, b * 8, 2), _block(b * 8, b * 8), _SE(b * 8))  # 56
        # ngữ cảnh toàn cục: avg-pool bottleneck -> MLP -> broadcast cộng lại
        self.gctx = nn.Sequential(nn.Linear(b * 8, b * 8), nn.SiLU(inplace=True),
                                  nn.Linear(b * 8, b * 8))
        # decoder về /4 (112) cho gain map
        self.d3 = nn.Sequential(_block(b * 8 + b * 4, b * 4), _SE(b * 4))
        self.d2 = nn.Sequential(_block(b * 4 + b * 2, b * 2), _SE(b * 2))
        self.gain_head = nn.Conv2d(b * 2, 3, 3, 1, 1)
        nn.init.zeros_(self.gain_head.weight)
        nn.init.zeros_(self.gain_head.bias)          # khoi dau: gain = 1 (identity)
        # LUT head: vector toan cuc -> trong so basis
        self.lut_w = nn.Linear(b * 8, n_basis)
        nn.init.zeros_(self.lut_w.weight)
        with torch.no_grad():
            bias = torch.zeros(n_basis); bias[0] = 1.0
            self.lut_w.bias.copy_(bias)              # khoi dau: LUT identity
        luts = torch.zeros(n_basis, 3, lut_dim, lut_dim, lut_dim)
        luts[0] = _identity_lut(lut_dim)
        self.basis = nn.Parameter(luts)

    def _features(self, proxy):
        p = F.interpolate(proxy, size=(self.proxy_res, self.proxy_res),
                          mode="bilinear", align_corners=False)
        f1 = self.e1(p)
        f2 = self.e2(f1)
        f3 = self.e3(f2)
        f4 = self.e4(f3)
        g = F.adaptive_avg_pool2d(f4, 1).flatten(1)
        f4 = f4 + self.gctx(g)[:, :, None, None]
        d = F.interpolate(f4, size=f3.shape[2:], mode="bilinear", align_corners=False)
        d = self.d3(torch.cat([d, f3], 1))
        d = F.interpolate(d, size=f2.shape[2:], mode="bilinear", align_corners=False)
        d = self.d2(torch.cat([d, f2], 1))
        return d, g

    def gain_map(self, proxy):
        d, _ = self._features(proxy)
        return torch.exp(self.gain_amp * torch.tanh(self.gain_head(d)))

    def forward(self, proxy, full):
        d, g = self._features(proxy)
        gain = torch.exp(self.gain_amp * torch.tanh(self.gain_head(d)))   # (N,3,h,w)
        gain_up = F.interpolate(gain, size=full.shape[2:], mode="bilinear",
                                align_corners=False)
        x = (full * gain_up).clamp(0, 1)
        w = self.lut_w(g)                                                # (N,K)
        lut = torch.einsum("nk,kcdef->ncdef", w, self.basis)
        B, G, R = x[:, 0], x[:, 1], x[:, 2]
        grid = torch.stack([2 * B - 1, 2 * G - 1, 2 * R - 1], dim=-1).unsqueeze(1)
        out = F.grid_sample(lut, grid, mode="bilinear", padding_mode="border",
                            align_corners=True).squeeze(2)
        return out.clamp(0, 1)
