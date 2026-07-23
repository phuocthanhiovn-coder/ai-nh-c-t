"""
Task 20 (GPU) - Advanced losses for image-to-image color/tone distillation.

Standalone add-on for the auto_enhance HDRNet. Does NOT modify the existing
model.py/train.py/dataset.py/infer.py; this only provides loss functions the
GPU trainer can opt into.

Convention (matches the rest of auto_enhance):
  - Tensors are [B, 3, H, W], BGR channel order, float in [0, 1].
    (The existing pipeline works in RGB internally after cv2 conversion, but
     the operator contract passes BGR float32 [0,1]; every loss here takes BGR
     and does its own flip where a specific channel order is required.)
  - All losses are differentiable w.r.t. `pred`.

Losses:
  - charbonnier: robust L1 variant.
  - LabLoss: perceptually-weighted color fidelity in CIE-Lab (D65), torch-native
    sRGB->linear->XYZ->Lab, matched against cv2.cvtColor to within tolerance.
  - VGGPerceptual: relu1_2 / relu2_2 / relu3_3 feature L1 (ImageNet-normalized).
  - CombinedLoss: weighted sum, builds only the requested sub-losses.

NOTE on VGG weights: VGGPerceptual lazy-loads torchvision vgg16 with
VGG16_Weights.DEFAULT. On a fresh rented box this downloads ~528 MB
(vgg16-397923af.pth) to the torch hub cache the first time it is constructed;
subsequent runs read from cache. CPU smoke tests below actually trigger this.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Charbonnier (robust L1)
# ---------------------------------------------------------------------------
def charbonnier(pred, target, eps=1e-3):
    """sqrt((pred-target)^2 + eps^2), averaged. Differentiable everywhere."""
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))


# ---------------------------------------------------------------------------
# Highlight protection (anti-washout / do-no-harm on bright regions)
# ---------------------------------------------------------------------------
def highlight_protection(pred, target, gamma=2.0):
    """Penalize ONLY over-brightening (pred > target), weighted toward bright
    regions of the target.

    Directly targets the #1 real client complaint on CH_C: the model brightens
    too hard -> blown windows, grey-washed warm floors, 'worse than original'.
    L1/Lab are symmetric so they don't specifically discourage this. Here only
    positive excursions (pred exceeding target) are charged, and the weight
    grows with target luminance^gamma so blowing already-bright pixels (windows,
    sky) hurts most. pred/target are BGR [0,1].
    """
    over = torch.relu(pred - target)                       # only over-brightening
    luma = (0.114 * target[:, 0:1]
            + 0.587 * target[:, 1:2]
            + 0.299 * target[:, 2:3])                      # BGR luminance
    w = luma.clamp(0.0, 1.0) ** gamma                      # emphasize highlights
    return (over * w).mean()


# ---------------------------------------------------------------------------
# Dark-region fidelity (24/07/2026 — CH_I)
# ---------------------------------------------------------------------------
def dark_fidelity(pred, target, thresh=0.28, l_weight=1.0, ab_weight=1.5):
    """Extra Lab-L1 CHI o vung TARGET toi: day 'den phai DUNG DO SAU (L) va
    TRUNG TINH (a/b ~ target)' — tri 2 chi tieu ket qua 3 doi model: p5 lech ~7-8
    va bong toi o vang/nau bun (chu du an che 6 vong lien tiep).

    Trong so w = ((thresh - luma_target)/thresh)^0.7 — muot, dark→1, sang→0.
    Chia /100 dua Lab ve thang ~[0,1] nhu cac term khac."""
    luma = (0.114 * target[:, 0:1]
            + 0.587 * target[:, 1:2]
            + 0.299 * target[:, 2:3])
    w = torch.clamp((thresh - luma) / thresh, 0.0, 1.0) ** 0.7
    lab_p = bgr_to_lab(pred)
    lab_t = bgr_to_lab(target)
    dl = (lab_p[:, 0:1] - lab_t[:, 0:1]).abs() * l_weight
    dab = (lab_p[:, 1:3] - lab_t[:, 1:3]).abs().mean(dim=1, keepdim=True) * ab_weight
    return ((dl + dab) / 100.0 * w).mean()


# ---------------------------------------------------------------------------
# CIE-Lab color loss (D65), torch-native & differentiable
# ---------------------------------------------------------------------------
# sRGB->XYZ matrix (D65), same coefficients OpenCV uses. Rows map RGB->XYZ.
_RGB2XYZ = torch.tensor(
    [
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ],
    dtype=torch.float32,
)
# D65 reference white used by OpenCV.
_WHITE_D65 = torch.tensor([0.950456, 1.0, 1.088754], dtype=torch.float32)
_LAB_EPS = 0.008856   # (6/29)^3
_LAB_KAPPA = 903.3    # 29^3 / 3^3


def _srgb_to_linear(c):
    """Inverse sRGB companding. c in [0,1] -> linear-light in [0,1]."""
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def bgr_to_lab(img_bgr):
    """
    Convert a BGR [0,1] image tensor [B,3,H,W] to CIE-Lab [B,3,H,W] (D65).
    L in [0,100], a/b roughly in [-127,127] -- same ranges cv2 produces for
    float32 input. Fully differentiable.
    """
    # BGR -> RGB
    r = img_bgr[:, 2:3]
    g = img_bgr[:, 1:2]
    b = img_bgr[:, 0:1]
    rgb = torch.cat([r, g, b], dim=1)

    lin = _srgb_to_linear(rgb.clamp(0.0, 1.0))

    m = _RGB2XYZ.to(device=img_bgr.device, dtype=img_bgr.dtype)   # [3,3]
    white = _WHITE_D65.to(device=img_bgr.device, dtype=img_bgr.dtype)

    # [B,3,H,W] linear RGB -> XYZ via einsum over channel dim.
    xyz = torch.einsum("oc,bchw->bohw", m, lin)
    xyz = xyz / white.view(1, 3, 1, 1)

    # f(t)
    def f(t):
        return torch.where(t > _LAB_EPS, torch.clamp(t, min=1e-8) ** (1.0 / 3.0),
                           (_LAB_KAPPA * t + 16.0) / 116.0)

    fx = f(xyz[:, 0:1])
    fy = f(xyz[:, 1:2])
    fz = f(xyz[:, 2:3])

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    bb = 200.0 * (fy - fz)
    return torch.cat([L, a, bb], dim=1)


class LabLoss(nn.Module):
    """
    L1 in CIE-Lab, with independent per-channel weights (L, a, b).
    L is on a 0..100 scale so it is internally divided by 100 to sit on a
    comparable footing with the chroma channels before weighting.
    """

    def __init__(self, w_l=1.0, w_a=1.0, w_b=1.0):
        super().__init__()
        self.w = (w_l, w_a, w_b)

    def forward(self, pred, target):
        lab_p = bgr_to_lab(pred)
        lab_t = bgr_to_lab(target)
        # Scale L down to ~[0,1] range like a/b (which are already ~[-1.3,1.3]
        # after /100). This keeps the three channels weight-comparable.
        diff = (lab_p - lab_t).abs() / 100.0
        wl, wa, wb = self.w
        loss = (wl * diff[:, 0:1] + wa * diff[:, 1:2] + wb * diff[:, 2:3]).mean()
        return loss


# ---------------------------------------------------------------------------
# VGG16 perceptual loss
# ---------------------------------------------------------------------------
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class VGGPerceptual(nn.Module):
    """
    Perceptual loss on VGG16 features at relu1_2, relu2_2, relu3_3.

    Input is BGR [0,1] [B,3,H,W]; internally flipped to RGB and ImageNet-
    normalized. VGG is frozen (eval + requires_grad_(False)) and lazy-loaded
    once on first forward. Downloading the weights (~528 MB) happens on first
    construction on a fresh machine.
    """

    def __init__(self):
        super().__init__()
        self.slices = None  # lazily built list of nn.Sequential blocks
        self.register_buffer("mean", _IMAGENET_MEAN.clone())
        self.register_buffer("std", _IMAGENET_STD.clone())

    def _build(self, device, dtype):
        from torchvision.models import vgg16, VGG16_Weights

        print("[VGGPerceptual] Loading vgg16 (VGG16_Weights.DEFAULT); "
              "first run downloads ~528 MB to the torch hub cache...")
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad_(False)

        # Split points at the ReLU after each target conv block:
        #   relu1_2 = features[:4], relu2_2 = features[4:9], relu3_3 = features[9:16]
        idx = [4, 9, 16]
        slices = nn.ModuleList()
        prev = 0
        for i in idx:
            slices.append(nn.Sequential(*[vgg[j] for j in range(prev, i)]))
            prev = i
        self.slices = slices.to(device=device, dtype=dtype)

    def _prep(self, img_bgr):
        # BGR -> RGB, then ImageNet normalize.
        rgb = img_bgr.flip(1)
        return (rgb - self.mean) / self.std

    def forward(self, pred, target):
        if self.slices is None:
            self._build(pred.device, pred.dtype)

        x = self._prep(pred)
        y = self._prep(target)
        loss = pred.new_zeros(())
        for blk in self.slices:
            x = blk(x)
            with torch.no_grad():
                y = blk(y)
            loss = loss + F.l1_loss(x, y)
        return loss


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------
class CombinedLoss(nn.Module):
    """
    Weighted sum of sub-losses. Only builds a sub-loss when its weight != 0,
    so e.g. VGG weights are not downloaded unless w_perc > 0.

    forward(pred, target) -> (total, terms_dict)
      pred/target: [B,3,H,W] BGR in [0,1].
      terms_dict holds each weighted contribution plus 'total'.
    """

    def __init__(self, w_l1=1.0, w_char=0.0, w_lab=0.0, w_perc=0.0,
                 lab_weights=(1.0, 1.0, 1.0), w_hi=0.0, hi_gamma=2.0,
                 w_dark=0.0, dark_thresh=0.28):
        super().__init__()
        self.w_l1 = float(w_l1)
        self.w_char = float(w_char)
        self.w_lab = float(w_lab)
        self.w_perc = float(w_perc)
        self.w_hi = float(w_hi)
        self.hi_gamma = float(hi_gamma)
        self.w_dark = float(w_dark)
        self.dark_thresh = float(dark_thresh)

        self.lab = LabLoss(*lab_weights) if self.w_lab != 0.0 else None
        self.perc = VGGPerceptual() if self.w_perc != 0.0 else None

    def forward(self, pred, target):
        terms = {}
        total = pred.new_zeros(())

        if self.w_l1 != 0.0:
            l1 = F.l1_loss(pred, target)
            terms["l1"] = self.w_l1 * l1
            total = total + terms["l1"]

        if self.w_char != 0.0:
            ch = charbonnier(pred, target)
            terms["char"] = self.w_char * ch
            total = total + terms["char"]

        if self.w_lab != 0.0:
            lab = self.lab(pred, target)
            terms["lab"] = self.w_lab * lab
            total = total + terms["lab"]

        if self.w_perc != 0.0:
            perc = self.perc(pred, target)
            terms["perc"] = self.w_perc * perc
            total = total + terms["perc"]

        if self.w_hi != 0.0:
            hi = highlight_protection(pred, target, self.hi_gamma)
            terms["hi"] = self.w_hi * hi
            total = total + terms["hi"]

        if self.w_dark != 0.0:
            dk = dark_fidelity(pred, target, self.dark_thresh)
            terms["dark"] = self.w_dark * dk
            total = total + terms["dark"]

        terms["total"] = total
        return total, terms
