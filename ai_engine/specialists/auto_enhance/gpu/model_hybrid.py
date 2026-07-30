"""HYBRID — model AI MỚI ghép từ 3 nguồn mã nguồn mở (30/07/2026).

Chủ dự án: "ghép các mã nguồn rời rạc đó thành mã nguồn ai mới đi".
Ghép có chủ đích, mỗi nguồn trị đúng 1 bệnh chủ chấm 0/10:

  [1] THÂN: NAFNet (megvii, MIT) — nạp weights `nafnet_realestate.pth`
      (SebRincon, Apache-2.0, fine-tune 577 cặp ảnh BĐS thật). Đây là phần
      "đã biết ảnh BĐS đẹp trông thế nào" — thứ model tự train từ số 0 của
      mình KHÔNG có, và là lý do bản của người khác sạch hơn dù ít data.
  [2] ĐẦU SÁNG: ý tưởng bản-đồ-chiếu-sáng của Retinexformer (MIT) — lấy TỈ LỆ
      giữa ảnh thân xuất ra và ảnh vào, LỌC TẦN THẤP → gain map. Trị "chỗ sáng
      chỗ tối" mà vẫn giữ đen (nhân, không cộng) và KHÔNG bịa chi tiết.
  [3] ĐẦU MÀU: 3D-LUT thích ứng của Image-Adaptive-3DLUT (HuiZeng, Apache) —
      basis LUT học được + trọng số đoán theo ảnh, KÈM 2 ràng buộc gốc mà bản
      tự viết 25/07 thiếu (tv smooth + monotonicity) → hết vệt/glow/đảo màu.

HỢP ĐỒNG (luật mới 30/07 của chủ): thân được TỰ DO sinh pixel ở PROXY; ảnh
giao hàng full-res CHỈ nhận operator (gain map + LUT) → nét gốc giữ nguyên,
đúng size, không bịa cấu trúc.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_lut import _identity_lut


# ─── NAFNet (megvii-research/NAFNet, MIT) — chép gọn phần cần, bỏ Local_Base ───
class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / torch.sqrt(var + self.eps)
        return y * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, 1, 0, bias=True)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, 1, 0, bias=True))
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.conv5(self.sg(self.conv4(self.norm2(y))))
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=32, middle_blk_num=12,
                 enc_blk_nums=(2, 2, 4, 8), dec_blk_nums=(2, 2, 2, 2)):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1, bias=True)
        self.encoders, self.decoders = nn.ModuleList(), nn.ModuleList()
        self.ups, self.downs = nn.ModuleList(), nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2
        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False),
                                          nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
        self.padder_size = 2 ** len(self.encoders)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h)), h, w

    def forward(self, inp):
        x, H, W = self.check_image_size(inp)
        inp_p = x
        x = self.intro(x)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        x = self.ending(x)
        x = x + inp_p
        return x[:, :, :H, :W]


# ─────────────────────────── HYBRID ───────────────────────────
class HybridNet(nn.Module):
    """Thân NAFNet (pretrain BĐS) → 2 đầu OPERATOR: gain map + 3D-LUT."""

    def __init__(self, width=32, lut_dim=17, n_basis=6, proxy_res=512,
                 gain_clip=3.0, gain_blur=8, nafnet_ckpt=None, **_ignore):
        super().__init__()
        self.proxy_res = int(proxy_res)
        self.gain_clip = float(gain_clip)
        self.gain_blur = int(gain_blur)
        self.trunk = NAFNet(img_channel=3, width=width, middle_blk_num=12,
                            enc_blk_nums=(2, 2, 4, 8), dec_blk_nums=(2, 2, 2, 2))
        if nafnet_ckpt:
            sd = torch.load(nafnet_ckpt, map_location="cpu")
            sd = sd.get("params", sd)
            missing, unexpected = self.trunk.load_state_dict(sd, strict=False)
            print(f"[hybrid] nap NAFNet pretrain: thieu {len(missing)} / thua {len(unexpected)}",
                  flush=True)
        # đầu LUT: CNN nhỏ trên ảnh thân xuất → trọng số basis
        ch = [3, 16, 32, 64, 128]
        blocks = []
        for i in range(len(ch) - 1):
            blocks += [nn.Conv2d(ch[i], ch[i + 1], 3, 2, 1),
                       nn.InstanceNorm2d(ch[i + 1], affine=True) if i else nn.Identity(),
                       nn.LeakyReLU(0.2, inplace=True)]
        self.lut_cnn = nn.Sequential(*blocks)
        self.lut_w = nn.Linear(ch[-1], n_basis)
        nn.init.normal_(self.lut_w.weight, 0.0, 0.02)   # xem model_nm.py: diem chet LUT
        with torch.no_grad():
            b = torch.zeros(n_basis); b[0] = 1.0
            self.lut_w.bias.copy_(b)                    # khởi đầu = LUT identity
        luts = torch.zeros(n_basis, 3, lut_dim, lut_dim, lut_dim)
        luts[0] = _identity_lut(lut_dim)
        self.basis = nn.Parameter(luts)

    def lut_regularizers(self):
        """tv smooth + monotonicity — 2 ràng buộc của repo gốc Image-Adaptive-3DLUT."""
        L = self.basis
        dif_r = L[..., :-1] - L[..., 1:]
        dif_g = L[..., :-1, :] - L[..., 1:, :]
        dif_b = L[..., :-1, :, :] - L[..., 1:, :, :]
        tv = (dif_r ** 2).mean() + (dif_g ** 2).mean() + (dif_b ** 2).mean()
        mn = F.relu(dif_r).mean() + F.relu(dif_g).mean() + F.relu(dif_b).mean()
        return tv, mn

    def _gain_from_trunk(self, proxy):
        """Retinex: tỉ lệ (thân xuất / vào) → lọc tần thấp = bản đồ chiếu sáng.
        Lọc tần thấp là chốt an toàn: mọi chi tiết/texture thân BỊA ra đều bị
        loại, chỉ còn thông tin 'vùng này nên sáng hơn/tối hơn bao nhiêu'."""
        # NAFNet pretrain la RGB, pipeline cua ta la BGR (fix 30/07): khong dao kenh
        # thi kien thuc pretrain bi ap sai kenh -> lech mau he thong.
        enh = self.trunk(proxy.flip(1)).flip(1)
        enh = enh + (enh.clamp(0, 1) - enh).detach()   # straight-through clamp
        # CLAMP TRUOC KHI LAM MO (fix 30/07): 1 pixel gan den (proxy~0) cho ratio ~500,
        # trung binh o 8x8 thanh ~8 -> clamp ve tran 3.0 -> CA VUNG bi keo sang toi da
        # => bong toi bi nang xam/bac mau (dung benh chu che). Clamp truoc, pool sau.
        ratio = ((enh + 0.02) / (proxy + 0.02)).clamp(1.0 / self.gain_clip, self.gain_clip)
        k = max(2, self.gain_blur)
        low = F.avg_pool2d(ratio, k, k, ceil_mode=True)
        low = F.interpolate(low, size=ratio.shape[2:], mode="bilinear", align_corners=False)
        return low, enh

    def forward(self, proxy, full):
        p = proxy if proxy.shape[-1] == self.proxy_res else F.interpolate(
            proxy, size=(self.proxy_res, self.proxy_res), mode="bilinear", align_corners=False)
        gain, enh = self._gain_from_trunk(p)
        gain_up = F.interpolate(gain, size=full.shape[2:], mode="bilinear", align_corners=False)
        x = full * gain_up
        x = x + (x.clamp(0, 1) - x).detach()          # straight-through clamp
        f = self.lut_cnn(enh)
        f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        w = self.lut_w(f)
        lut = torch.einsum("nk,kcdef->ncdef", w, self.basis)
        B, G, R = x[:, 0], x[:, 1], x[:, 2]
        grid = torch.stack([2 * B - 1, 2 * G - 1, 2 * R - 1], dim=-1).unsqueeze(1)
        out = F.grid_sample(lut, grid, mode="bilinear", padding_mode="border",
                            align_corners=True).squeeze(2)
        return out.clamp(0, 1), {"gain": gain, "enh": enh}
