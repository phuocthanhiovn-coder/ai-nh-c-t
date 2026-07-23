# HDRNetV2 — honest report

File: `ai_engine/specialists/auto_enhance/gpu/model_v2.py`. Does NOT touch
`model.py` / `train.py` / `infer.py` / `dataset.py`. Same operator-not-pixel
architecture family as `..model.HDRNet`, but `grid_bins`, `grid_size`,
`proxy_res`, `width`, `guidance_hidden` are constructor args.

## What adapts to the knobs
- **proxy_res → grid_size**: `n_down = floor(log2(proxy_res/grid_size))` stride-2
  convs, then one `F.adaptive_avg_pool2d(grid_size)` snaps the spatial map to
  EXACTLY `grid_size` even when the ratio is not a power of two
  (384/32 = 12 → 3 stride-2 convs → 48px → pooled to 32).
- **width**: channels = `width, 2·width, 4·width, …` capped at `8·width`; global
  FC hidden = `16·width`.
- **grid_bins**: only changes `predict_conv` out-channels (`12·grid_bins`) and the
  reshape depth. Slice + affine are dimension-agnostic.
- **guidance_hidden**: hidden width of the two-1×1-conv guidance net.
- **make_proxy(full, proxy_res)**: `F.interpolate(mode='area')` (differentiable
  analogue of the `cv2.INTER_AREA` used in dataset/infer); accepts `[B,3,H,W]` or
  `[3,H,W]`, BGR/RGB order-agnostic.

## CPU smoke test — REAL output (verified this session)
Run: `python ai_engine/specialists/auto_enhance/gpu/model_v2.py`

```
[cfg] default: {'grid_bins': 8, 'grid_size': 16, 'proxy_res': 256, 'width': 16}
      params = 1,142,993   (n_down=4, feat_ch=128, down_ch=[16, 32, 64, 128])
      full 200x300 -> out (1, 3, 200, 300)  grid (1, 12, 8, 16, 16)  range=[0.000,0.105]  OK
      full 137x 97 -> out (1, 3, 137, 97)  grid (1, 12, 8, 16, 16)  range=[0.000,0.104]  OK
[cfg] bins16: {'grid_bins': 16, 'grid_size': 16, 'proxy_res': 256, 'width': 16}
      params = 1,155,377
      full 200x300 -> out (1, 3, 200, 300)  grid (1, 12, 16, 16, 16)  OK
      full 137x 97 -> out (1, 3, 137, 97)  grid (1, 12, 16, 16, 16)  OK
[cfg] gs32_p384_w24: {'grid_bins': 8, 'grid_size': 32, 'proxy_res': 384, 'width': 24}
      params = 956,865   (n_down=3, feat_ch=96, down_ch=[24, 48, 96])
      full 200x300 -> out (1, 3, 200, 300)  grid (1, 12, 8, 32, 32)  OK
      full 137x 97 -> out (1, 3, 137, 97)  grid (1, 12, 8, 32, 32)  OK
ALL SMOKE ASSERTIONS PASSED
```
Assertions checked per (config × size): `out.shape == full.shape`, grid ==
`[1,12,grid_bins,grid_size,grid_size]`, all-finite, output in `[0,1]`, plus
`make_proxy` on 4D and 3D inputs.

## Brutally honest caveats
- **Output range is ~[0, 0.13], not spread across [0,1].** This is expected for
  UNTRAINED random weights: stacked ReLUs + small random affine coeffs push the
  per-pixel transform near zero. It only proves shape/finiteness/clamp validity,
  NOT visual quality. Do not read anything about enhancement quality into it.
- **CUDA path is UNVERIFIED.** This box is CPU-only (torch 2.13.0+cpu, no CUDA).
  The module is pure `nn`/`F.*` and should move to CUDA unchanged, but AMP
  autocast/GradScaler is a TRAINER concern and lives in `train_gpu.py`, not here;
  none of it was exercised on GPU.
- **`grid_size=32` roughly quadruples slice/affine memory vs 16** at full-res;
  fine on the target V100-32GB but untested at 4–6k-px real photos here.
- **`gs32_p384_w24` has FEWER params (957k) than default (1.14M)** despite being
  "bigger": with `n_down=3` the deepest channel is `4·width=96` vs default's
  `8·width=128`, and `predict_conv`/`fusion_conv` dominate the count. Widen with
  `width` if you want more capacity at that grid size.
- Grid channel layout is 12 (affine, major) × `grid_bins` (luminance, minor),
  matching v1's `view(B,12,bins,gs,gs)` — checkpoints are NOT interchangeable
  with v1 (different tensor shapes), which is intended.
