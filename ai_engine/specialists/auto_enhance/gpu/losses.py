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
    # FIX 30/07: .mean() chia cho TOAN BO pixel trong khi w gan 0 khap noi ->
    # term chong chay sang chi con 0.26% gradient. Chuan hoa theo tong trong so.
    # B7: over co 3 kenh, w chi 1 kenh -> chia w.sum() lam term manh GAP 3 LAN so voi
    # danh nghia (do duoc dung 3.0x). B5: san mau so chong gradient vo han.
    denom = torch.clamp(3.0 * w.sum(), min=0.06 * w.numel())
    return (over * w).sum() / denom


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
    # FIX 30/07 (dot 2 — lan dau regex khong khop nen sot): .mean() chia cho TOAN BO
    # pixel trong khi w chi phu 0.5-17% -> term giu-den-sau bi loang 6-218 lan, chi
    # con 0.2% gradient. Chuan hoa theo tong trong so (do: |grad| tang 17.8 lan).
    # B5 (ra soat vong 6): w.sum() do duoc thap nhat 0.26 tren 534 crop that ->
    # gradient/pixel gap 967.450 lan L1. San mau so o muc 2% dien tich.
    denom = torch.clamp(w.sum(), min=0.02 * w.numel())
    return (((dl + dab) / 100.0) * w).sum() / denom


# ---------------------------------------------------------------------------
# Colorfulness + local-contrast regularizers (25/07/2026 — CH_M, huong moi)
# GOC benh "mau nhat/bot" = L1 hoi quy ve TRUNG BINH -> nhat + vi tuong phan thap
# (chan doan tu workflow nghien cuu 25/07). 2 term nay day THONG KE output ve
# khop target AutoHDR (punchy), thay vi de L1 keo ve trung binh nhat.
# ---------------------------------------------------------------------------
def colorfulness(img_bgr):
    """Hasler-Susstrunk colorfulness / anh (BGR [0,1]) -> (N,). Cao = mau song."""
    b, g, r = img_bgr[:, 0], img_bgr[:, 1], img_bgr[:, 2]  # moi (N,H,W)
    rg = r - g
    yb = 0.5 * (r + g) - b
    std_root = torch.sqrt(rg.var(dim=(1, 2)) + yb.var(dim=(1, 2)) + 1e-6)
    mean_root = torch.sqrt(rg.mean(dim=(1, 2)) ** 2 + yb.mean(dim=(1, 2)) ** 2 + 1e-6)
    return std_root + 0.3 * mean_root


def colorfulness_loss(pred, target):
    """Phat output NHAT hon target manh (relu, chong desaturation) + bam nhe 2 chieu.
    Target = AutoHDR (punchy) nen keo output len la dung huong."""
    cp, ct = colorfulness(pred), colorfulness(target)
    under = F.relu(ct - cp).mean()          # chi phat khi nhat hon target
    track = (cp - ct).abs().mean()          # bam sat 2 chieu (nhe)
    return under + 0.25 * track



# ---------------------------------------------------------------------------
# CHROMA THEO VUNG (31/07) — thay colorfulness_loss da bi TAT.
# Chu cham 0/10: "anh bac, mat mau". Nguyen nhan: tat w_color -> khong con gi
# ngan model lam nhat mau, ma L1+Lab ve ban chat la "doan trung binh" = xam.
# Loss cu co 2 loi (thuong am mau toan anh; keo NGUOC khi anh da du dam) nen phai
# SUA chu khong phai bo:
#   - tinh theo O 8x8 -> khong the "bu" bang cach lam dam mot goc
#   - dung chroma Lab (khong phai rg/yb) -> am mau toan anh KHONG duoc thuong
#   - CHI phat mot chieu (nhat hon target), khong keo nguoc
# ---------------------------------------------------------------------------
def chroma_tile_loss(pred, target, tiles=8):
    lab_p = bgr_to_lab(pred)
    lab_t = bgr_to_lab(target)
    cp = torch.sqrt(lab_p[:, 1:2] ** 2 + lab_p[:, 2:3] ** 2 + 1e-6)
    ct = torch.sqrt(lab_t[:, 1:2] ** 2 + lab_t[:, 2:3] ** 2 + 1e-6)
    mp = F.adaptive_avg_pool2d(cp, tiles)
    mt = F.adaptive_avg_pool2d(ct, tiles)
    # ⭐ 04/08 (ra soat vong 7) — THEM VE BAM NGUOC (nhe).
    # Ban cu chi `relu(mt - mp)`, tuc phat MOT CHIEU tuyet doi: chi phat khi pred NHAT
    # hon target. Do that: anh qua bao hoa GAP BA LAN target van cham 0.000000 — loss
    # nay hoan toan mu truoc viec over-cook mau, dung tat ma chu che nhieu vong
    # ("mau khong tu nhien", "neon").
    # Giu bat doi xung (day toi target manh hon la keo lui) nhung khong con mu: he so
    # 0.25 cho chieu nguoc, cung ti le ma colorfulness_loss dang dung.
    return (F.relu(mt - mp).mean() + 0.25 * F.relu(mp - mt).mean()) / 100.0

def clip_loss(pred, target, hi=0.96, lo=0.02):
    """PHAT RIENG viec DANG TRANG / BET DEN (01/08).

    Vi sao can rieng, khong dung highlight_protection: cong thuc do phat
    relu(pred - target) co trong so luma^2, tuc la phat MOI lan sang hon anh dich
    o vung sang. Voi w_hi=3.0 (FT5) no tao ra do lech mot chieu: vuot len tren
    ton 1 + 3*w, tut xuong duoi chi ton 1 -> diem toi uu bi keo XUONG DUOI anh
    dich. Do duoc tren FT5: chay trang ve 0.00% (AutoHDR 0.11%) nhung ANH TOI OM,
    am nau — dung loi chu che.

    Term nay chi kich hoat khi pred CHAM tran (>=hi) hoac CHAM day (<=lo) trong
    khi anh dich VAN CON chi tiet o do. Vung trung gian: gradient bang 0 tuyet
    doi -> khong the lam toi anh. Chia cho (1-hi) de dua ve thang [0,1].
    """
    m_hi = (target < hi).to(pred.dtype)
    m_lo = (target > lo).to(pred.dtype)
    blown = (torch.relu(pred - hi) / (1.0 - hi)) * m_hi
    crush = (torch.relu(lo - pred) / max(lo, 1e-6)) * m_lo
    return blown.mean() + crush.mean()


_LAP_K = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)



def sharp_tile_loss(pred, target, tiles=8, w_flat=2.0):
    """DO NET THEO VUNG (31/07) — lo hong GOC cua 15 doi model.

    Do duoc: pipeline lam GIAM net 30% (150->103) trong khi AutoHDR TANG gap 6 lan
    (150->893). Ly do: loss KHONG HE do do net (local_contrast cu bi bo vi co loi
    keo anh ve mo, chua thay bang gi), trong khi co 3 nguon lam mem (khu nhieu,
    warp, model). => model khong co ly do gi de giu net.

    Khac local_contrast cu: (a) theo O 8x8 chu khong phai 1 so cho ca anh -> khong
    the bu bang cach lam net mot goc; (b) CHI phat mot chieu (mem hon target),
    KHONG bao gio keo nguoc khi anh da net hon -> khong the lam mo anh.

    BAN VA 31/07 (luat L10) — ban dau CHI co ve thuong o tren, va NAF_FT5 da LACH:
    do net 1176 vs AutoHDR 619, nhung soi 100% thi do la LUOI SAN phu bau troi va
    tuong phang. Rac nhieu tan so cao re hon nhieu so voi dung canh that, ma
    Laplacian khong phan biet duoc hai thu. Nay tach lam 2 ve theo VUNG cua anh
    DICH:
      - vung CO CANH THAT  -> thuong net (nhu cu), o day nhieu khong an diem duoc
        vi da co san nang luong cao.
      - vung PHANG (troi, tuong) -> PHAT nang luong cao tan vuot qua anh dich.
        Day chinh la duong lach cu, gio thanh duong lo.
    w_flat=2.0: phat nang hon thuong de model khong "danh doi" san lay diem net.
    """
    k = _LAP_K.to(pred.dtype).to(pred.device)

    def lap(x):
        y = 0.114 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.299 * x[:, 2:3]
        return F.conv2d(y, k, padding=1).abs()

    lp, lt = lap(pred), lap(target)
    # "phang hay khong" doc tu anh DICH (khong doc tu pred — neu doc tu pred thi
    # model chi viec lam nhieu khap noi de tu tuyen bo "cho nao cung la canh").
    soft_t = F.avg_pool2d(lt, 9, stride=1, padding=4)
    thr = soft_t.mean(dim=(1, 2, 3), keepdim=True)
    edge_m = (soft_t >= thr).to(lt.dtype).detach()
    flat_m = 1.0 - edge_m

    # ve 1 — thuong net, chi tinh tren vung canh that, gop theo o cho on dinh
    ep = F.adaptive_avg_pool2d(lp * edge_m, tiles)
    et = F.adaptive_avg_pool2d(lt * edge_m, tiles)
    # ⭐ 04/08 (ra soat vong 7) — PHAT CA CHIEU VUOT MUC TRONG VUNG CANH.
    # Ban cu chi `relu(et - ep)`, tuc chi phat khi pred KEM net hon target trong vung
    # canh. Ve chong-san (`over`) chi gac vung PHANG. Nen rac nhieu rai DUNG TRONG
    # VUNG CANH van lot: no lam `ep` tang (thoat ve 1) ma khong cham ve 2.
    # Nay bam nhe chieu vuot muc ngay trong vung canh de "them nang luong bua" khong
    # con la duong lach. He so 0.25 giu tinh bat doi xung (uu tien chua MO hon la
    # chua QUA TAY), dung ti le voi chroma_tile_loss.
    under = F.relu(et - ep).mean() + 0.25 * F.relu(ep - et).mean()

    # ve 2 — phat san: vung phang ma pred cao tan hon target
    # ⭐ 04/08 (ra soat vong 7) — EP FP32 TRUOC KHI CONG DON.
    # Duoi AMP (torch.cuda.amp.autocast) cac tensor nay la fp16, ma fp16 chi bieu dien
    # toi 65504. `flat_m.sum()` tren mot batch crop 512x512 dem den hang tram nghin
    # diem -> TRAN SO -> inf. Phep chia cho inf cho 0, nen `over` = 0 va ve CHONG SAN
    # (chong lam net qua tay o vung phang) CHET HOAN TOAN — im lang, khong warning,
    # loss van ra so dep. Ca tu so lan mau so deu phai tinh o fp32.
    _fm = flat_m.float()
    over = (F.relu(lp - lt).float() * _fm).sum() / _fm.sum().clamp(min=1.0)
    return under + w_flat * over


def local_contrast_loss(pred, target):
    """Khop nang luong Laplacian (vi tuong phan) tren luma. Day output SAC len bang
    target (relu chong 'mo bot') + bam 2 chieu nhe. Guided-filter-free, tren luma
    nen khong lech mau."""
    def luma(x):
        return (0.114 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.299 * x[:, 2:3])
    k = _LAP_K.to(pred.dtype).to(pred.device)
    lp = F.conv2d(luma(pred), k, padding=1).abs().mean(dim=(1, 2, 3))
    lt = F.conv2d(luma(target), k, padding=1).abs().mean(dim=(1, 2, 3))
    under = F.relu(lt - lp).mean()
    track = (lp - lt).abs().mean()
    return under + 0.25 * track


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
        return torch.where(t > _LAB_EPS, torch.clamp(t, min=1e-6) ** (1.0 / 3.0),
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
        # FIX 31/07: mean/std la tensor thuong -> .to(device) cua Module KHONG doi
        # chung -> RuntimeError khi train tren GPU. Ep ve dung thiet bi cua input.
        m = self.mean.to(device=rgb.device, dtype=rgb.dtype)
        sd = self.std.to(device=rgb.device, dtype=rgb.dtype)
        return (rgb - m) / sd

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
                 w_dark=0.0, dark_thresh=0.28, w_color=0.0, w_lc=0.0, w_chroma=0.0, w_sharp=0.0,
                 w_clip=0.0):
        super().__init__()
        self.w_l1 = float(w_l1)
        self.w_char = float(w_char)
        self.w_lab = float(w_lab)
        self.w_perc = float(w_perc)
        self.w_hi = float(w_hi)
        self.hi_gamma = float(hi_gamma)
        self.w_dark = float(w_dark)
        self.dark_thresh = float(dark_thresh)
        self.w_color = float(w_color)
        self.w_lc = float(w_lc)
        self.w_chroma = float(w_chroma)
        self.w_sharp = float(w_sharp)
        self.w_clip = float(w_clip)

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

        if self.w_clip != 0.0:
            cl = clip_loss(pred, target)
            terms["clip"] = self.w_clip * cl
            total = total + terms["clip"]

        if self.w_color != 0.0:
            col = colorfulness_loss(pred, target)
            terms["color"] = self.w_color * col
            total = total + terms["color"]

        if self.w_lc != 0.0:
            lc = local_contrast_loss(pred, target)
            terms["lc"] = self.w_lc * lc
            total = total + terms["lc"]

        if self.w_chroma > 0:

            ch = chroma_tile_loss(pred, target)

            total = total + self.w_chroma * ch

            terms["chroma"] = self.w_chroma * float(ch.detach())
        if self.w_sharp > 0:
            sh = sharp_tile_loss(pred, target)
            total = total + self.w_sharp * sh
            terms["sharp"] = self.w_sharp * float(sh.detach())


        terms["total"] = total
        return total, terms
