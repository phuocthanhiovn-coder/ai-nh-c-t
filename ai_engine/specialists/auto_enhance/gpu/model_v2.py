"""
HDRNetV2 - parameterizable HDRNet for the auto_enhance specialist (Task GPU).

Same architecture FAMILY as ..model.HDRNet, but every capacity knob is a
constructor argument so a rented-GPU run can sweep grid resolution / proxy
resolution / channel width without editing code:

    __init__(grid_bins=8, grid_size=16, proxy_res=256, width=16, guidance_hidden=16)

CORE PRINCIPLE (immutable, unchanged from v1):
    The net predicts an OPERATOR, never pixels. The coefficient predictor runs
    on a small proxy and emits a bilateral grid of 3x4 affine coefficients
    [B, 12, grid_bins, grid_size, grid_size]. A pointwise guidance net maps the
    FULL-RES image to a 1-channel [0,1] guide. We differentiably slice the grid
    with F.grid_sample (trilinear) at every full-res pixel, apply the per-pixel
    3x4 affine to the full-res input, and clamp to [0,1]. Output size == input
    size at ANY H,W. No file in this package touches ..model / ..train / ..infer.

HOW IT ADAPTS TO THE KNOBS
    - proxy_res -> grid_size: the downsampling conv stack uses
      n_down = floor(log2(proxy_res / grid_size)) stride-2 convs, then a single
      F.adaptive_avg_pool2d(grid_size) snaps the spatial map to EXACTLY
      grid_size even when proxy_res/grid_size is not a power of two
      (e.g. 384/32 = 12 -> 3 stride-2 convs -> 48px -> pooled to 32).
    - width: channel counts are width, 2*width, 4*width, ... capped at 8*width,
      the global-branch FC hidden is 16*width.
    - grid_bins: only changes predict_conv out-channels (12 * grid_bins) and the
      reshape depth; slicing/affine are dimension-agnostic.
    - guidance_hidden: hidden width of the 1x1 guidance net.

BGR-AGNOSTIC: operates on any 3-channel [0,1] tensor (BGR or RGB); it never
assumes a channel order.

CUDA NOTE: this module is pure nn.Module / F.* ops and is CUDA-ready, but the
AMP autocast/GradScaler wiring lives in the trainer, not here. See
MODEL_V2_NOTES.md for the honest smoke-test report (CPU-verified; CUDA path
unverified locally on this CPU-only box).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GuidanceMapV2(nn.Module):
    """1-channel [0,1] guidance map from a full-res 3-channel image via two 1x1
    convs. guidance_hidden sets the hidden width."""

    def __init__(self, guidance_hidden=16):
        super().__init__()
        self.conv1 = nn.Conv2d(3, guidance_hidden, kernel_size=1)
        self.conv2 = nn.Conv2d(guidance_hidden, 1, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = torch.sigmoid(self.conv2(out))  # [B, 1, H, W] in [0, 1]
        return out


class CoefficientPredictorV2(nn.Module):
    """Runs on the proxy and predicts the bilateral grid
    [B, 12, grid_bins, grid_size, grid_size].

    The conv stack adapts to proxy_res/grid_size/width (see module docstring)."""

    def __init__(self, grid_bins=8, grid_size=16, proxy_res=256, width=16):
        super().__init__()
        self.grid_bins = grid_bins
        self.grid_size = grid_size
        self.proxy_res = proxy_res
        self.width = width

        # Number of stride-2 downsampling convs so the feature map lands at or
        # just above grid_size; a final adaptive pool snaps to exactly grid_size.
        ratio = proxy_res / float(grid_size)
        if ratio > 1.0:
            n_down = int(math.floor(math.log2(ratio)))
        else:
            n_down = 1
        n_down = max(1, n_down)
        self.n_down = n_down

        cap = 8 * width
        down_layers = []
        in_ch = 3
        self.down_channels = []
        for i in range(n_down):
            out_ch = min(width * (2 ** i), cap)
            down_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1))
            down_layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
            self.down_channels.append(out_ch)
        self.down = nn.Sequential(*down_layers)

        C = in_ch  # final feature channel count
        self.feat_channels = C

        # Local branch: keep grid_size resolution.
        self.local_conv = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1)

        # Global branch: two stride-2 convs, adaptive-pool to 4x4, then FC.
        # The 4x4 pool makes the FC input dim independent of grid_size.
        self.global_conv1 = nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1)
        self.global_conv2 = nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1)
        fc_dim = 16 * width
        self.fc1 = nn.Linear(C * 4 * 4, fc_dim)
        self.fc2 = nn.Linear(fc_dim, C)

        # Fuse local + broadcast(global), then predict the grid coefficients.
        self.fusion_conv = nn.Conv2d(2 * C, C, kernel_size=1)
        self.predict_conv = nn.Conv2d(C, 12 * grid_bins, kernel_size=1)

    def forward(self, x):
        # x: proxy [B, 3, proxy_res, proxy_res]
        B = x.size(0)
        gs = self.grid_size
        C = self.feat_channels

        feat = self.down(x)                              # [B, C, ~, ~]
        feat = F.adaptive_avg_pool2d(feat, gs)           # [B, C, gs, gs]

        local_feat = F.relu(self.local_conv(feat))       # [B, C, gs, gs]

        g = F.relu(self.global_conv1(feat))
        g = F.relu(self.global_conv2(g))
        g = F.adaptive_avg_pool2d(g, 4)                  # [B, C, 4, 4]
        g = g.reshape(B, -1)
        g = F.relu(self.fc1(g))
        global_feat = F.relu(self.fc2(g))                # [B, C]

        global_expanded = global_feat.view(B, C, 1, 1).expand(-1, -1, gs, gs)
        fusion = torch.cat([local_feat, global_expanded], dim=1)  # [B, 2C, gs, gs]

        fusion_out = F.relu(self.fusion_conv(fusion))
        grid_flat = self.predict_conv(fusion_out)        # [B, 12*grid_bins, gs, gs]

        # 12 affine coeffs (major) x grid_bins luminance bins (minor).
        grid = grid_flat.view(B, 12, self.grid_bins, gs, gs)
        return grid


class HDRNetV2(nn.Module):
    """Parameterizable HDRNet. Predicts a bilateral grid of affine coefficients
    on the proxy, slices it differentiably with the full-res guidance map, and
    applies a per-pixel 3x4 affine to the full-res input. Operator-not-pixel."""

    def __init__(self, grid_bins=8, grid_size=16, proxy_res=256, width=16, guidance_hidden=16):
        super().__init__()
        self.grid_bins = grid_bins
        self.grid_size = grid_size
        self.proxy_res = proxy_res
        self.width = width
        self.guidance_hidden = guidance_hidden

        self.predictor = CoefficientPredictorV2(
            grid_bins=grid_bins, grid_size=grid_size, proxy_res=proxy_res, width=width
        )
        self.guidance_net = GuidanceMapV2(guidance_hidden=guidance_hidden)

    @staticmethod
    def make_proxy(full_bgr_tensor, proxy_res):
        """Downsample a full-res 3-channel [0,1] tensor to a square proxy.

        Accepts [B, 3, H, W] or [3, H, W] (BGR or RGB - channel order agnostic).
        Uses F.interpolate(mode='area'), the differentiable analogue of cv2's
        INTER_AREA used elsewhere in this package. Returns [B, 3, proxy_res,
        proxy_res]."""
        if full_bgr_tensor.dim() == 3:
            full_bgr_tensor = full_bgr_tensor.unsqueeze(0)
        return F.interpolate(full_bgr_tensor, size=(proxy_res, proxy_res), mode="area")

    def slice_grid(self, grid, guidance):
        """Differentiable bilateral slicing via 3D F.grid_sample.
        grid: [B, 12, grid_bins, gh, gw]; guidance: [B, 1, H, W] in [0,1].
        Returns sliced coefficients [B, 12, H, W]."""
        B, C, D, gh, gw = grid.shape
        _, _, H, W = guidance.shape

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=guidance.device, dtype=guidance.dtype),
            torch.linspace(-1, 1, W, device=guidance.device, dtype=guidance.dtype),
            indexing="ij",
        )
        grid_x = grid_x.view(1, H, W, 1).expand(B, -1, -1, -1)
        grid_y = grid_y.view(1, H, W, 1).expand(B, -1, -1, -1)

        # Guidance value -> z lookup coordinate in [-1, 1].
        grid_z = guidance.permute(0, 2, 3, 1) * 2.0 - 1.0  # [B, H, W, 1]

        # grid_sample 3D expects coords ordered (x, y, z).
        query_grid = torch.cat([grid_x, grid_y, grid_z], dim=-1).unsqueeze(3)  # [B,H,W,1,3]

        sliced = F.grid_sample(
            grid, query_grid, mode="bilinear", padding_mode="border", align_corners=True
        )  # [B, 12, H, W, 1]
        return sliced.squeeze(4)  # [B, 12, H, W]

    def apply_affine(self, sliced_grid, full_res_img):
        """Apply the per-pixel 3x4 affine to the full-res image and clamp [0,1].
        sliced_grid: [B, 12, H, W]; full_res_img: [B, 3, H, W]."""
        B, _, H, W = sliced_grid.shape
        coeffs = sliced_grid.view(B, 3, 4, H, W)     # 3 rows x 4 cols per pixel
        color_matrix = coeffs[:, :, :3, :, :]        # [B, 3, 3, H, W]
        translation = coeffs[:, :, 3, :, :]          # [B, 3, H, W]

        inp = full_res_img.unsqueeze(1)              # [B, 1, 3, H, W]
        transformed = torch.sum(inp * color_matrix, dim=2)  # [B, 3, H, W]
        output = transformed + translation
        return torch.clamp(output, 0.0, 1.0)

    def forward(self, proxy, full_res_img):
        # proxy: [B, 3, proxy_res, proxy_res]; full_res_img: [B, 3, H, W]
        grid = self.predictor(proxy)                       # [B,12,grid_bins,gs,gs]
        guidance = self.guidance_net(full_res_img)         # [B, 1, H, W]
        sliced_grid = self.slice_grid(grid, guidance)      # [B, 12, H, W]
        output = self.apply_affine(sliced_grid, full_res_img)
        return output, grid


# ---------------------------------------------------------------------------
# CPU smoke test - run: python -m ai_engine.specialists.auto_enhance.gpu.model_v2
# ---------------------------------------------------------------------------
def _count_params(m):
    return sum(p.numel() for p in m.parameters())


def _smoke():
    import cv2  # local import: only needed to honor the thread-cap rule

    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    torch.manual_seed(0)

    configs = [
        ("default", dict()),
        ("bins16", dict(grid_bins=16, grid_size=16)),
        ("gs32_p384_w24", dict(grid_bins=8, grid_size=32, proxy_res=384, width=24)),
    ]
    full_sizes = [(200, 300), (137, 97)]  # (H, W): non-square, odd, tiny

    print("=" * 68)
    print("  HDRNetV2 CPU smoke test (operator-not-pixel, parameterizable)")
    print("=" * 68)

    results = []
    for name, kw in configs:
        model = HDRNetV2(**kw)
        model.eval()
        cfg = dict(
            grid_bins=model.grid_bins,
            grid_size=model.grid_size,
            proxy_res=model.proxy_res,
            width=model.width,
        )
        n_params = _count_params(model)
        results.append((name, n_params))
        print(f"\n[cfg] {name}: {cfg}")
        print(f"      params = {n_params:,}   "
              f"(predictor n_down={model.predictor.n_down}, "
              f"feat_ch={model.predictor.feat_channels}, "
              f"down_ch={model.predictor.down_channels})")

        proxy = torch.rand(1, 3, cfg["proxy_res"], cfg["proxy_res"])
        for (H, W) in full_sizes:
            full = torch.rand(1, 3, H, W)
            with torch.no_grad():
                out, grid = model(proxy, full)

            assert out.shape == full.shape, \
                f"{name} {H}x{W}: out {tuple(out.shape)} != full {tuple(full.shape)}"
            exp_grid = (1, 12, cfg["grid_bins"], cfg["grid_size"], cfg["grid_size"])
            assert tuple(grid.shape) == exp_grid, \
                f"{name} {H}x{W}: grid {tuple(grid.shape)} != {exp_grid}"
            assert torch.isfinite(out).all(), f"{name} {H}x{W}: non-finite output"
            assert torch.isfinite(grid).all(), f"{name} {H}x{W}: non-finite grid"
            assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0, \
                f"{name} {H}x{W}: out out-of-range [{float(out.min())},{float(out.max())}]"

            print(f"      full {H}x{W:>3} -> out {tuple(out.shape)}  "
                  f"grid {tuple(grid.shape)}  "
                  f"range=[{float(out.min()):.3f},{float(out.max()):.3f}]  OK")

        # make_proxy helper check: any H,W, both 4D and 3D input.
        p4 = HDRNetV2.make_proxy(torch.rand(1, 3, 211, 379), cfg["proxy_res"])
        p3 = HDRNetV2.make_proxy(torch.rand(3, 480, 640), cfg["proxy_res"])
        assert tuple(p4.shape) == (1, 3, cfg["proxy_res"], cfg["proxy_res"])
        assert tuple(p3.shape) == (1, 3, cfg["proxy_res"], cfg["proxy_res"])
        print(f"      make_proxy: 4D->{tuple(p4.shape)}  3D->{tuple(p3.shape)}  OK")

    print("\n" + "=" * 68)
    print("  ALL SMOKE ASSERTIONS PASSED")
    for name, n in results:
        print(f"    {name:>16}: {n:,} params")
    print("=" * 68)


if __name__ == "__main__":
    _smoke()
