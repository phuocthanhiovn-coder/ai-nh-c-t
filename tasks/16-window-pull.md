# Task 16 — WINDOW PULL v0 (AutoHDR's most valuable feature — deterministic single-image version)

**Assigned to:** Worker W (Fable 5 on Max) · **Review:** Claude architect · **Read `CLAUDE.md` first.**

## Context — why this exists and what "window pull" means
In real-estate interiors, windows blow out white or wash pale because the exterior is many stops brighter. "Window pull" (flambient style) balances it: the exterior view stays visible/saturated through the glass while the interior keeps its brightness. It's the #1 thing RE agents pay editors for.
Classic window pull uses a dark bracket. **We do NOT have brackets here** — our `data/pairs/before/*.jpg` are Mertens exposure-fusion merges, which RETAIN partial window detail (not fully clipped). So v0 = LOCAL tone-mapping recovery: darken + saturate + recover contrast INSIDE window regions only, keep everything else bit-identical outside the feathered mask.

## OPERATOR CONTRACT (identical to other specialists)
`apply(img: np.ndarray float32 [0,1] HxWx3 BGR, params: dict) -> np.ndarray` same shape/dtype. No resize/re-encode.

## MUST REUSE `ai_engine/core/quality.py`: `guided_upsample`, `composite_mask`, `split_frequency`/`merge_frequency` where useful.

## Files (`ai_engine/specialists/window_pull/`)
1. `window_mask.py` — `detect_windows(img) -> (mask float [0,1], win_fraction)`:
   - Work on ~768px proxy. A window region = connected component that is (a) much brighter than the room: luma above ~p90 of image AND above interior median by a margin; (b) compact blob with reasonable area (0.5%–35% of frame), not touching ALL four borders (a whole blown wall isn't a window); (c) often bounded by strong straight edges (frames) — use edge density on the component boundary as a supporting vote, not a hard rule.
   - Merge nearby components (a window grid = many panes) with morphology close.
   - Refine to full-res with `guided_upsample` so the matte hugs the frame/mullions.
2. `pull.py` — `apply(img, params)`:
   - params: `strength` (0-1, default 0.7), `saturation_boost` (0-0.6, default 0.25).
   - Gate: `win_fraction < 0.005` or `> 0.5` → return unchanged.
   - Inside mask: recover the exterior — apply a highlight-compression tone curve in linear light (e.g., filmic/Reinhard-style on luma above a pivot ≈ interior median; pivot must ADAPT per image), multiply-in saturation boost (HSV S or Lab chroma) scaled by how much luminance was recovered; do it on LOW frequency only (`split_frequency`) and re-add original high frequency so glass reflections/mullions stay crisp.
   - Composite back with `composite_mask` + feather; pixels outside feathered mask bit-identical (assert like sky_replace does).
3. `run_samples.py` — auto-pick 6 images with windows from `data/pairs/before/` (rank by win_fraction), save `outputs/window_samples/<name>.jpg` = [original | mask viz | pulled] 1800px panels + one 100% crop of a window area appended. Print win_fraction + gated status per image.

## Acceptance (run + LOOK at images with Read tool)
- [ ] Window exteriors: visibly darker/more saturated with recovered sky/buildings; NOT gray/dirty, NOT halo at frames.
- [ ] Interior brightness unchanged (compare wall next to window before/after — must match).
- [ ] Image with no/tiny windows → unchanged (gate). Output size == input. Honest report at end of this file: per-image win_fraction, worst case, limits.

## DO NOT
- No files outside `ai_engine/specialists/window_pull/` + `outputs/window_samples/` + report here. `data/` read-only. `cv2.setNumThreads(2)`, one python process at a time.

---

## BÁO CÁO THỰC HIỆN — Worker W (Fable 5), 2026-07-14

### Đã làm gì
3 file trong `ai_engine/specialists/window_pull/`: `window_mask.py`, `pull.py`, `run_samples.py`. Tuân hợp đồng operator (float32 [0,1] BGR, cùng shape, không resize). Tái dùng `guided_upsample`, `composite_mask`, `split_frequency`/`merge_frequency`, `to_linear`/`to_srgb` từ `ai_engine/core/quality.py`, và `detect_sky` từ `sky_replace` (chỉ import, không sửa).

**Đã chạy thật `python -m ai_engine.specialists.window_pull.run_samples` (nhiều lần, lần cuối = trạng thái code hiện tại) và MỞ XEM từng ảnh kết quả + crop 100%.**

### Kết quả thật (lần chạy cuối, 51 ảnh sau khử trùng dbNN_)
| Ảnh | win_fraction | gated | max_diff_outside | Nhìn bằng mắt |
|---|---|---|---|---|
| db02_20260703-DSC1197 | 0.1566 | không | 0.0 | TỐT — skyline đậm màu, trời xanh, nội thất nguyên |
| 20260703-DSC1217 | 0.1172 | không | 0.0 | TỐT — cải thiện vừa, không artifact |
| _ML_1372 | 0.1203 | không | 0.0 | **TỐT NHẤT** — cửa sổ cháy trắng → thấy rõ tòa nhà gạch đỏ |
| 20260703-DSC1105 | 0.1111 | không | 0.0 | TỐT |
| 20260703-DSC1251 | 0.0894 | không | 0.0 | **XẤU NHẤT** — xem "Hạn chế" |
| 20260703-DSC1132 | 0.1088 | không | 0.0 | TỐT — có vệt xanh mint nhẹ ở trời góc trên |
| test gate tổng hợp (ảnh không cửa sổ) | 0.0000 | CÓ | bit-identical | PASS |
| test gate ảnh thật ít cửa sổ nhất (db01__ML_1661) | 0.0000 | CÓ | bit-identical | PASS (nhưng xem "Hạn chế" #2) |

Ngoài mask feather: **bit-identical 100%** (assert trong `apply` + đo lại độc lập trong run_samples, max_diff_outside = 0.0 ở cả 6 ảnh). Output đúng size input. Tốc độ ~14s/ảnh 2048px trên CPU (detect + apply).

### Quyết định thiết kế ngoài spec chữ-đen (đều do NHÌN ảnh hỏng mà thêm)
1. **Fill convex hull** mỗi blob giữ lại → pull đồng đều cả ô cửa (không loang lổ theo ngưỡng sáng). Tone curve có **knee** (pivot + 12% span) nên phần tường/nội thất lọt vào hull không bị tối đi.
2. **Gate ảnh ngoại thất**: vùng (sáng ∪ trời detect_sky) nối biên TRÊN khung phủ ≥35% hàng đầu → coi là trời lộ thiên, trả mask rỗng. Không có nó, ảnh ban công _ML_1542 bị pull MÂY thành vệt xám bẩn (đã thấy bằng mắt qua 3 vòng thử; veto từng blob bằng detect_sky KHÔNG đủ vì detect_sky bỏ sót mây).
3. **Yêu cầu ≥8% pixel gần clip (luma ≥0.93)** trong blob: chặn cửa/tường trắng nội thất.
4. **Veto nhiệt độ màu**: blob phải LẠNH hơn nội thất (median(B−R) chênh ≥ −0.015). Chặn máy giặt trắng bóng (_ML_1500, diff −0.027) và sàn gạch rooftop (_ML_1400, −0.039/−0.071) — cả hai từng bị pull sai, đã thấy bằng mắt.
5. **Local tone mapping**: gain tính từ ảnh nền bilateral (không phải luma trực tiếp) — Reinhard trực tiếp nghiền contrast vùng sáng còn ~7% → toà nhà trắng thành "sữa loãng". Trọng số theo chroma (làm mượt sigma 24) để trắng clip tinh khôi không bị kéo thành xám bẩn; sat boost cũng nhân trọng số chroma để không nhuộm màu vùng trung tính.
6. Spec nói edge vote "không phải luật cứng" — tôi vẫn đặt SÀN edge vote 0.15 (thoát nếu extent ≥0.8): mức tối thiểu để loại blob viền mềm (mây); không có sàn này fail acceptance "NOT gray/dirty".

### Hạn chế / bad case (TRUNG THỰC)
1. **DSC1251 (worst case)**: tòa nhà trắng gần clip hoàn toàn qua cửa sổ hành lang — kết quả hơi "kem/phẳng", cải thiện biên (không xám bẩn, không halo, nhưng cũng không đẹp hẳn). Bản chất: chỗ đã clip KHÔNG còn chi tiết để phục hồi — v0 đơn ảnh không bịa được; cần bracket tối hoặc model sinh (pha D). Cùng ảnh này 1 blob là **đèn chùm** bị nhận nhầm cửa sổ (sáng, lạnh trung tính) — hậu quả nhẹ (đèn vẫn sáng gần nguyên) nhưng là false positive thật.
2. **False negative có chủ đích**: db01__ML_1661 (cửa sổ nhìn ra cây xanh) bị veto nhiệt độ màu loại (lá cây ẤM, B−R −0.031 = cùng phía máy giặt). Đổi lấy an toàn: thà BỎ QUA cửa sổ lá cây (ảnh giữ nguyên) còn hơn pull nhầm máy giặt/sàn nhà. Ngưỡng −0.015 **mỏng manh** (cách cửa sổ thật gần nhất chỉ 0.007) — calib trên 51 ảnh, phải calib lại khi data đổi. Fix đúng lâu dài: model segment cửa sổ (nhóm A roster).
3. Gate ngoại thất che luôn trường hợp nội thất có cửa sổ chiếm ≥35% bề rộng SÁT mép trên khung (không gặp trong 51 ảnh).
4. DSC1132: vệt xanh mint nhẹ ở trời trong cửa sổ (sat boost khuếch đại màu cyan sẵn có). Nhẹ, chưa xử lý.
5. `win_fraction` phụ thuộc scale proxy nhẹ (scan 480px vs full 768px cho số hơi khác — xem log); gate dùng bản full nên nhất quán.
6. Chưa đăng ký vào `orchestrator/registry.py` + chạy conformance_check (file ngoài whitelist của task này — việc của architect khi duyệt). Khuyến nghị: giữ window_pull NGOÀI plan mặc định (như sky_replace) tới khi có scene classifier + segment cửa sổ bằng model.

### Files
- Code: `ai_engine/specialists/window_pull/{window_mask.py, pull.py, run_samples.py}`
- Ảnh: `outputs/window_samples/` — 6 panel [gốc|mask|pull]+crop, 4 crop 100% `debug_*_crop.png` (1197/1251/1372/1132)

**TASK16=DONE** — 51 ảnh quét, 6/6 sample pull sạch (max_diff_outside=0.0, đúng size), win_fraction 0.089–0.157, gate PASS (2/2 unchanged bit-identical), worst case DSC1251 ghi nhận trung thực.
