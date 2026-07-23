"""Render before | CH_C(old) | CH_E(anti-washout) | target for eyeballing.
Operator-not-pixel: proxy -> grid -> apply to full-res before. BGR throughout.
Run on box: cd /root/autohdr && python3 -m tools.render_compare
"""
import os
import cv2
import numpy as np
import torch
from ai_engine.specialists.auto_enhance.gpu.model_v2 import HDRNetV2

torch.set_num_threads(4)
DEV = "cuda"
MK = dict(grid_bins=8, grid_size=16, proxy_res=384, width=24)

# Diverse val scenes (interior + drone/aerial + pool/exterior) — exactly the
# cases clients flagged for washout/over-brightening.
NAMES = [
    "_ML_1421.jpg",
    "j021_FP101671.jpg",
    "drone01_DSC01518.jpg",
    "j027_dji_0952.jpg",
    "after_pool2_gd09_783A9534.jpg",
    "j054_DSC4574.jpg",
]


def load(ckpt):
    m = HDRNetV2(**MK).to(DEV).eval()
    sd = torch.load(ckpt, map_location=DEV)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    m.load_state_dict(sd)
    return m


def apply_model(m, bgr):
    t = torch.from_numpy(bgr.transpose(2, 0, 1)).unsqueeze(0).float().to(DEV)
    proxy = HDRNetV2.make_proxy(t, 384)
    with torch.no_grad():
        out, _ = m(proxy, t)
    return out.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)


def panel(img, label, w=760):
    h = int(round(img.shape[0] * w / img.shape[1]))
    r = cv2.resize((img * 255).clip(0, 255).astype("uint8"), (w, h))
    cv2.rectangle(r, (0, 0), (w, 44), (0, 0, 0), -1)
    cv2.putText(r, label, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return r


def main():
    ch_c = load("checkpoints/gpu/CH_C.pt")
    ch_e = load("checkpoints/sweep/CH_E_antiwash.pt")
    os.makedirs("outputs/compare", exist_ok=True)
    for n in NAMES:
        bp, ap = f"data/before/{n}", f"data/after/{n}"
        if not (os.path.exists(bp) and os.path.exists(ap)):
            print("skip missing", n)
            continue
        b = cv2.imread(bp).astype("float32") / 255.0
        a = cv2.imread(ap).astype("float32") / 255.0
        if a.shape[:2] != b.shape[:2]:
            a = cv2.resize(a, (b.shape[1], b.shape[0]))
        c = apply_model(ch_c, b)
        e = apply_model(ch_e, b)
        row = np.hstack([
            panel(b, "BEFORE (merge)"),
            panel(c, "CH_C (old)"),
            panel(e, "CH_E anti-washout"),
            panel(a, "TARGET (AutoHDR)"),
        ])
        out = f"outputs/compare/cmp_{n}"
        cv2.imwrite(out, row, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        # quick numeric: mean brightness + saturation, to spot over-brightening
        def stats(x):
            hsv = cv2.cvtColor((x * 255).astype("uint8"), cv2.COLOR_BGR2HSV)
            return round(float(x.mean()), 3), round(float(hsv[:, :, 1].mean()), 1)
        print(f"{n}: before{stats(b)} CH_C{stats(c)} CH_E{stats(e)} target{stats(a)}")
    print("RENDER_DONE")


if __name__ == "__main__":
    main()
