"""SRVGGNetCompact — kien truc nhe cua Real-ESRGAN (realesr-general-x4v3).

Nguon: xinntao/Real-ESRGAN (BSD-3-Clause — DUNG THUONG MAI DUOC). Viet lai toi
thieu de nap weights `checkpoints/ext/realesr-general-x4v3.pth` ma khong can
basicsr (basicsr loi import torchvision.transforms.functional_tensor tren
torchvision moi). Chi phuc vu inference.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32,
                 upscale=4, act_type="prelu"):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        # residual: bilinear-upsample dau vao roi cong (nhu Real-ESRGAN compact)
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base
