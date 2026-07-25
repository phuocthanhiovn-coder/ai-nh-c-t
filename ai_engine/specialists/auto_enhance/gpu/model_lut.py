"""Image-Adaptive 3D-LUT thuan PyTorch (25/07 — huong moi [2], thay bilateral grid).

Trien khai lai y tuong HuiZeng/Image-Adaptive-3DLUT (Apache) KHONG can CUDA op:
- N basis LUT hoc duoc (n_basis, 3, D, D, D), init 1 con IDENTITY + phan con lai ~0.
- backbone CNN nho tren PROXY -> N trong so -> tron thanh 1 LUT theo-anh.
- ap LUT len anh FULL bang F.grid_sample 3D (noi suy 3 tuyen, kha vi theo LUT).

VI SAO 3D-LUT > bilateral grid (chan doan workflow 25/07): grid tron-theo-thiet-ke;
LUT co D^3 o TU DO -> bieu dien duoc "punch" phu thuoc hue/vat lieu (gach day manh,
go/da giu nhe) ma grid KHONG the. La OPERATOR mau per-pixel deterministic: bat kha
doi/bia cau truc, giu dung kich thuoc -> HOP nguyen tac "operator khong pixel".

Kenh theo BGR (nhu HDRNetV2/eval). LUT truc: (channel, iR, iG, iB).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _identity_lut(dim):
    """LUT identity (3, D, D, D) kenh BGR: tai o (iR,iG,iB) tra (B=iB, G=iG, R=iR)."""
    r = torch.linspace(0, 1, dim)
    iR, iG, iB = torch.meshgrid(r, r, r, indexing="ij")  # moi (D,D,D)
    return torch.stack([iB, iG, iR], dim=0)              # (3,D,D,D) kenh B,G,R


class _Backbone(nn.Module):
    """CNN nho tren proxy -> vector dac trung -> trong so basis."""
    def __init__(self, n_basis, res=256):
        super().__init__()
        self.res = res
        ch = [3, 16, 32, 64, 128]
        blocks = []
        for i in range(len(ch) - 1):
            blocks += [nn.Conv2d(ch[i], ch[i + 1], 3, 2, 1),
                       nn.InstanceNorm2d(ch[i + 1], affine=True) if i > 0 else nn.Identity(),
                       nn.LeakyReLU(0.2, inplace=True)]
        self.conv = nn.Sequential(*blocks)
        self.head = nn.Linear(ch[-1], n_basis)

    def forward(self, proxy):
        x = F.interpolate(proxy, size=(self.res, self.res), mode="bilinear",
                          align_corners=False)
        x = self.conv(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)   # (N, C)
        return self.head(x)                          # (N, n_basis)


class Image3DLUT(nn.Module):
    def __init__(self, n_basis=3, lut_dim=33, backbone_res=256, **_ignore):
        super().__init__()
        self.n_basis = int(n_basis)
        self.dim = int(lut_dim)
        # basis[0] = identity; con lai init ~0 (bat dau = identity, hoc dan).
        base = torch.zeros(self.n_basis, 3, self.dim, self.dim, self.dim)
        base[0] = _identity_lut(self.dim)
        self.basis = nn.Parameter(base)
        self.backbone = _Backbone(self.n_basis, backbone_res)
        # KHOI DAU DUNG IDENTITY: zero head weight + bias=[1,0,0] -> w=[1,0,0] moi anh
        # -> output = identity (basis[0]) tuyet doi. Hoc dan khoi day (on dinh nhu warm-start).
        self.backbone.head.weight.data.zero_()
        self.backbone.head.bias.data.zero_()
        self.backbone.head.bias.data[0] = 1.0

    @staticmethod
    def make_proxy(img, res):
        """Khop API HDRNetV2.make_proxy: area-resample ve res vuong. img [3,H,W] hoac [N,3,H,W]."""
        if img.dim() == 3:
            img = img.unsqueeze(0)
        return F.interpolate(img, size=(int(res), int(res)), mode="area")

    def _combined_lut(self, weights):
        # weights (N, n_basis) -> LUT (N, 3, D, D, D)
        return torch.einsum("nk,kcxyz->ncxyz", weights, self.basis)

    def forward(self, proxy, full):
        w = self.backbone(proxy)                     # (N, n_basis)
        lut = self._combined_lut(w)                  # (N,3,D,D,D)
        N, _, H, W = full.shape
        B = full[:, 0]; G = full[:, 1]; R = full[:, 2]   # moi (N,H,W), [0,1]
        # grid_sample: coord (x,y,z) map (W=iB, H=iG, D=iR). Scale [0,1]->[-1,1].
        grid = torch.stack([2 * B - 1, 2 * G - 1, 2 * R - 1], dim=-1)  # (N,H,W,3)
        grid = grid.unsqueeze(1)                     # (N,1,H,W,3) D_out=1
        out = F.grid_sample(lut, grid, mode="bilinear",
                            padding_mode="border", align_corners=True)  # (N,3,1,H,W)
        out = out.squeeze(2).clamp(0, 1)             # (N,3,H,W)
        return out, lut
