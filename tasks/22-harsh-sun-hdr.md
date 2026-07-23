# Task 22 — HARSH-SUN EXTERIOR specialist (deterministic local tone-mapping, the customer's hard case)

**Assigned to:** Worker (Fable 5) · **Review:** Claude architect · **Read `CLAUDE.md` first.**

## Context (the customer's hardest case)
Ground-level exterior real-estate photos shot in HARSH SUN have extreme dynamic range: blown-white sky/highlights + deep black shadows. This is exactly what HDR editing exists for and what AutoHDR does best. We have ZERO harsh-sun training pairs yet, so the LEARNED model can't do it — but a well-built DETERMINISTIC local tone-mapping can compress the dynamic range NOW, on a single image, no training needed. Build that.

## OPERATOR CONTRACT (same as all specialists)
`apply(img: np.ndarray float32 [0,1] HxWx3 BGR, params) -> np.ndarray` same shape. No resize/re-encode. Reuse `ai_engine/core/quality.py` (`to_linear`/`to_srgb`, `split_frequency`) where useful.

## Files (`ai_engine/specialists/harsh_sun/`)
1. `tone_map.py` — `apply(img, params)`:
   - Work in linear light (`to_linear`). Estimate a LOCAL luminance base via a large-sigma edge-aware blur (bilateral / guided filter — `cv2.ximgproc.guidedFilter` if present, else bilateral). 
   - Compress dynamic range: pull DOWN pixels far above the local base (recover blown highlights — sky, sunlit walls) and lift pixels far below (open shadows), via a curve applied on the base, then re-add high-frequency detail (frequency separation) so texture/edges stay crisp. This is HDR-like local tone mapping / "flambient" compression on ONE exposure.
   - Preserve/boost color: do the compression on luminance; recover chroma so colors don't wash (the learned model's failure — DON'T repeat it). A gentle saturation restore proportional to how much luminance was compressed.
   - params: `strength` (0-1, default 0.7), `highlight_recover` (0-1, default 0.8), `shadow_lift` (0-1, default 0.5), `local_contrast` (0-1, default 0.3), `sat_restore` (0-1, default 0.4).
   - Guard against halos at high-contrast edges (roofline against sky) — the edge-aware base + limited compression strength should prevent them; verify.
2. `run_samples.py` — since we lack real harsh-sun exteriors, SYNTHESIZE the test: take a few `data/pairs/before/` images and any exterior you can find, and ALSO create harsh-contrast test cases by boosting contrast/clipping a copy to simulate blown sky + crushed shadows, then run the tone-map. Save `outputs/harshsun_samples/<name>.jpg` = [input | tone-mapped] + a 100% crop at a high-contrast edge. Print dynamic-range before/after (p99-p1).

## Acceptance (run + LOOK)
- [ ] On a high-contrast input: blown/near-blown highlights recover visible detail; deep shadows open; NO halo at rooflines; colors stay RICH (not washed/gray). Output size == input.
- [ ] On a normal-exposure input: subtle, doesn't over-process (gate/strength keeps it natural).
- [ ] Honest report: dynamic-range compression numbers, worst case, halo check, limits. Note this is deterministic — will improve once we have harsh-sun training pairs to tune against.

## DO NOT
- No files outside `ai_engine/specialists/harsh_sun/` + `outputs/harshsun_samples/` + report. `data/` read-only. `cv2.setNumThreads(2)`.

---

## BÁO CÁO WORKER (Fable 5, 2026-07-14) — ĐÃ CHẠY THẬT, ĐÃ NHÌN TỪNG ẢNH

### Đã build
- `ai_engine/specialists/harsh_sun/tone_map.py` — `apply(img, params)` đúng hợp đồng operator. Thuật toán kiểu Durand: log2-luminance (linear-light, dùng `to_linear`/`to_srgb` của core) → base bằng `cv2.ximgproc.guidedFilter` (CÓ trên máy, OpenCV 5.0.0; radius ≈ 6% cạnh ngắn, fallback bilateral trên bản nhỏ nếu thiếu ximgproc) → detail = logL − base giữ nguyên 100% (cộng lại sau, gain 1+local_contrast·strength/2) → nén CHỈ trên base bằng 2 soft-knee quanh anchor CỐ ĐỊNH: vai highlight trên −1.25 EV (kéo xuống), chân shadow dưới −3.3 EV (nâng lên, trần 3 EV, guard đen-thật dưới ~−11 EV để letterbox/góc vignette không thành xám). Màu: scale kênh theo tỉ lệ luma + phục hồi saturation kiểu Mantiuk `(C/L)^s`, s tỉ lệ với số EV bị nén (chống washout). Gate tự động: %pixel cháy (≥0.95) và %pixel kịt (≤0.05) scale riêng 2 phía.
- `ai_engine/specialists/harsh_sun/run_samples.py` — quét 338 ảnh (data/pairs/before + data/review/before, read-only), xếp hạng "độ gắt", chạy top-5 + 1 ảnh phơi sáng chuẩn (test gate) + 2 SYNTH (contrast ×2.2 quanh median luma, clip 2 đầu). Panel = [input | output] + crop 100% tại cạnh mạnh nhất kề vùng cháy (soi halo).

### Số liệu THẬT (params mặc định, 8 case, `outputs/harshsun_samples/`)
| case | clip hi/lo % | DR p99−p1 | mean Δ |
|---|---|---|---|
| harsh1 _ML_36 (nội thất, 35% đen) | 3.8 / 35.0 | 1.000→0.960 | 0.029 |
| harsh2 783A9614 (phòng ngủ tối + cửa sổ cháy) | 5.0 / 23.1 | 1.000→0.966 | 0.046 |
| harsh3 783A9659 | 4.0 / 17.4 | 1.000→0.976 | 0.029 |
| harsh4 783A9639 | 5.3 / 9.0 | 0.996→0.974 | 0.032 |
| harsh5 DSC1161 (view phố nắng gắt) | 3.6 / 13.1 | 0.991→0.947 | 0.032 |
| normal DSC01584 (test gate) | 0.0 / 0.0 → gate 0/0 | 0.607→0.615 | **0.003** |
| synth1 DSC01584 (clip 2 đầu) | 5.5 / 27.7 | 0.998→0.851 | 0.043 |
| synth2 DSC01398 (hầm rượu, kịt bóng) | 0.0 / 34.5 | 0.494→0.509 | 0.031 |
- Runtime đo thật: 2.10s @ 2048×1365 (guided filter O(N), không phụ thuộc radius). Edge case: NaN=0 mọi case, `strength=0` → bit-identical, ảnh zeros/tiny không crash, output luôn đúng shape input.

### Nghiệm thu (tự nhìn từng panel bằng mắt)
- ✅ Ảnh gắt: bóng tối MỞ RA rõ (sàn gỗ hiện vân + màu ấm, ghế da cam hiện màu, kệ rượu synth2 đọc được), highlight cháy dịu xuống còn thấy cảnh ngoài cửa sổ; màu GIÀU không xám (sat_restore hoạt động — khác hẳn washout của auto_enhance).
- ✅ KHÔNG halo ở crop 100% cạnh tương phản cao (khung cửa sổ vs trời, roofline building harsh5) trên cả 8 panel.
- ✅ Ảnh phơi sáng chuẩn: gate 0/0, mean Δ=0.003 — gần như không đụng.
- ✅ Letterbox đen của pano harsh1 giữ ĐEN tuyệt đối (black guard).

### THẤT BẠI GIỮA CHỪNG (đã sửa, ghi lại làm bài học)
Bản đầu đặt midpoint nén theo percentile của base → ảnh có bóng kịt kéo midpoint xuống ~−7 EV → TOÀN ẢNH bị coi là highlight, kéo tối cả ảnh + cửa sổ xám bệch, shadow không mở (nhìn panel là thấy ngay, bảng số DR 1.000→0.358 trông "ấn tượng" nhưng ảnh HỎNG). Sửa: anchor cố định theo cảm nhận. Bài học lặp lại đúng CLAUDE.md: **số đẹp ≠ ảnh đẹp, phải nhìn**.

### Giới hạn thật (không giấu)
1. **Pixel cháy 100% không có data** — kéo xuống chỉ thành mảng sáng phẳng, không sinh được chi tiết (muốn có mây/trời thật phải sky_replace hoặc generative pha sau). Ảnh toàn trắng tinh sẽ bị dịu xuống ~0.18 (degenerate, không crash).
2. **Bóng kịt dưới ~sRGB 0.006 chủ đích KHÔNG nâng** (guard chống xám hoá đen thật + chống khuếch đại nhiễu) → case nghiền 93% ảnh về 0 là không cứu được — đúng bản chất, data đã mất.
3. Nâng shadow khuếch đại nhiễu vùng tối (chưa nối với denoise trong plan).
4. Trên ảnh synth bị clip lệch kênh, sat_restore có thể đẩy màu hơi gắt (bụi cỏ đỏ synth1).
5. Ngưỡng (anchor EV, gate, trần 3EV) chỉnh tay trên 8 case, **data thật KHÔNG có ảnh ngoại thất mặt đất nắng gắt đúng nghĩa** (top harshness toàn nội thất cửa sổ cháy + 1 view phố). Deterministic — sẽ tune lại khi có cặp harsh-sun thật.
6. Chưa vào `registry.py` + `conformance_check` (file ngoài phạm vi cho phép của task) — chờ architect duyệt rồi đăng ký.

**TASK22=DONE** — 8/8 case pass mắt + số: DR p99−p1 nén tới 0.998→0.851 (synth) / 1.000→0.960 (thật), ảnh chuẩn Δ=0.003 (gate giữ nguyên), 0 halo, 2.10s @ 2.8MP, output luôn đúng size.
