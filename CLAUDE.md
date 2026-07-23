# CLAUDE.md — Bộ não dự án: AI chỉnh ảnh Bất động sản (AutoHDR-clone)

> **File này để làm gì:** Context chung cho cả team. **Claude = kiến trúc sư + review + giao việc; worker (Claude phụ / model qua gateway anticode) = người code theo spec.** ĐỌC FILE NÀY TRƯỚC mỗi phiên. Việc cụ thể nằm trong thư mục `/tasks`.

---

## 1. Mục tiêu
Xây AI chỉnh ảnh bất động sản tự động, **chất lượng như AutoHDR**. Kiến trúc: **nhiều model nhỏ, mỗi con làm ĐÚNG 1 việc (~22 con)**, một **orchestrator** ghép lại. Dự án ĐỘC LẬP — KHÔNG đụng bất kỳ server/app nào khác trên máy này.

## 2. NGUYÊN TẮC BẤT BIẾN (vi phạm = làm sai, phải sửa)
1. **AI xuất OPERATOR, KHÔNG xuất pixel cuối.** Model chạy trên bản **proxy nhỏ** → đoán *tham số / mask / lưới (grid)* → một engine deterministic **áp lên ảnh MASTER full-res**. TUYỆT ĐỐI không để model generative nhả thẳng ảnh giao hàng (sẽ mất nét, sai size, nén bể — đúng lỗi của các web dùng banana thô).
2. **Mỗi con 1 việc.** Không gộp nhiều chức năng vào một model.
3. **Giữ chất lượng tuyệt đối.** Xử lý ở **full-res, 16-bit, linear-light** khi cần; **chỉ nén 8-bit ở bước export cuối**. Không downscale, không recompress ngầm. Output PHẢI đúng kích thước ảnh gốc.
4. **Model mở, tự host, fine-tune trên data riêng.** Không gọi API generative bên thứ ba.

## 3. Quy ước làm việc (architect ↔ worker)
- Mỗi việc = 1 file `tasks/NN-ten.md` gồm **spec + tiêu chí nghiệm thu (acceptance)**.
- **Worker:** code ĐÚNG spec. **KHÔNG tự ý đổi kiến trúc/nguyên tắc.** Mơ hồ → **HỎI, đừng đoán**.
- **Worker:** phải **TEST trước khi báo xong** (chạy được + smoke-test pass). Ghi kết quả thật (số liệu, đường dẫn ảnh) + **phần hạn chế/thất bại thành thật** vào cuối file task. Không báo "xong" khi chưa chạy.
- **Architect:** review code + **TỰ MỞ ẢNH NHÌN** → sửa/duyệt → giao task kế. Không duyệt bằng bảng số.
- Thư mục sạch, tên rõ. (Thư mục này KHÔNG phải git repo — chưa init.)
- **Gemini đã bị loại khỏi quy trình từ 08/07/2026** (chạy ingest làm max RAM cả máy VPS, chủ dự án yêu cầu dừng). `GEMINI.md` giữ lại làm lịch sử, không dùng nữa.
- **LUẬT MỚI 22/07/2026 (chủ dự án đặt, đợt data mới):**
  1. Claude **tự làm hoàn toàn**, không chờ hỏi từng bước.
  2. **GPU thuê chỉ khi THẬT SỰ cần** — phải báo chủ dự án và giải thích vì sao cần, chủ duyệt mới thuê. Việc gì CPU làm được thì làm CPU.
  3. **Hạn mức disk: 60 GB** cho toàn bộ dự án. Sau mỗi đợt việc (train, render, eval) phải **tự dọn**: xóa ảnh/artifact không còn giá trị (render nháp, checkpoint thua cuộc, log to). Giữ lại: `data/pairs/`, checkpoint tốt nhất, ảnh nghiệm thu tiêu biểu.

## 4. Ngăn xếp
Python 3.10+, PyTorch, OpenCV, scikit-image, NumPy, FastAPI (serve sau). Xem `requirements.txt`.

## 5. Cấu trúc thư mục (THỰC TẾ 2026-07-22)
```
C:\Users\Administrator\Desktop\autohdr\
  CLAUDE.md  GEMINI.md  requirements.txt
  tasks/                     # 22 spec (00..24, thiếu 10/14/19) — kết quả nghiệm thu ghi ở CUỐI mỗi file
  data/pairs/before|after/   # 723 cặp, cùng tên file
  ai_engine/
    core/quality.py          # lõi giữ chất lượng dùng chung (task 13)
    data_pairing/            # ingest CR3/Dropbox/Drive/Sheet, align, dedup, rescue (task 00/02/03/04)
    specialists/             # 11 con, mỗi con 1 thư mục
      auto_enhance/          # CPU: model.py=HDRNet | gpu/: model_v2.py=HDRNetV2 + train_sweep (train trên box thuê)
      white_balance/ straighten/ denoise_sharpen/ grass_green/
      sky_replace/ window_pull/ harsh_sun/ scene_classify/ qc_scorer/ object_removal/
    orchestrator/            # registry.py + ops_basic.py + planner + engine
    delivery/deliver.py      # xuất full-res JPEG q100 4:4:4 (task 21)
    service/ webapp/         # demo web local 127.0.0.1:8123 (task 24)
    report/                  # báo cáo HTML before/after
    conformance_check.py  integration_test.py  process.py
  tools/                     # script GPU/eval/render (gpu_*.py, render_*.py, rclone.exe)
  outputs/                   # ảnh kết quả, log, ảnh so sánh (RẤT nặng — dọn định kỳ)
  checkpoints/
    auto_enhance.pt          # HDRNet pilot CPU (14/07) — con đang được ops_basic nạp
    gpu/                     # 30+ checkpoint HDRNetV2 từ các đợt thuê GPU (CH_E_antiwash = tốt nhất)
```

## 6. Roster ~22 con (bức tranh lớn — build DẦN, KHÔNG cùng lúc)
`[CV]`=toán thuần không model · `[Mở]`=model mở fine-tune (SAM/DINO/LaMa/ESRGAN) · `[Học]`=train từ cặp data của chủ

✅ = đã có và dùng được · 🔄 = có nhưng chưa chín · ⬜ = chưa làm

**A. Nhìn & định tuyến:** ✅phân loại cảnh(23) · 🔄segment trời (mask CV trong sky_replace, chưa phải model) · ⬜segment cửa sổ · ⬜detect đồ lộn xộn · ⬜detect người/phản chiếu · ⬜detect TV · ⬜detect lò sưởi · ⬜segment sàn/tường/trần · 🔄segment cỏ (mask màu trong grass_green) · ⬜segment hồ/nước — hầu hết `[Mở]`
**B. Hình học/quang:** ✅align bracket · ✅khử ghost · ✅dọc thẳng perspective(06) · ✅sửa méo ống kính(04) — `[CV]`
**C. Màu/sáng/tone (LÕI, học từ data chủ):** ✅auto-enhance HDRNetV2 CH_F(01 — CH_F từ 23/07) · ✅cân trắng(07) · ✅window pull(16) · ✅khử nhiễu+phục nét(08) · ✅nén dải sáng nắng gắt(22) · ✅finish_detail(22/07 — trị "mờ bợt") · ✅vibrance(23/07 — trắng+màu accent+đen sạch) · ✅shadow_light(23/07 — thắp sáng góc flambient) · ⬜upscale — `[Học]`/`[Mở]`
**D. Sinh/ghép (GPU, sau):** 🔄xóa vật thể LaMa(18 — cần UI mask) · 🔄thay trời+harmonize(09/17) · ⬜twilight+glow · ⬜staging · ⬜lấp màn TV/lửa — `[Mở]`
**E. Trên cùng:** 🔄QC scorer(11 — cần calib lại) · ✅**Orchestrator**(05) · ✅giao hàng full-res(21) · ✅demo web(24)

## 7. Thứ tự làm (lộ trình)
- ✅ **Pha 0 XONG:** khung xương + pipeline deterministic + orchestrator + giao hàng full-res + demo web. Nguyên lý "operator không pixel" **đã chứng minh chạy thật** (task 21: 3000×2250 vào → 3000×2250 ra, 9 MB JPEG q100, crop 100% sắc nét).
- 🔄 **Pha 1 (ĐANG DỞ — nút thắt duy nhất):** con **auto_enhance** học màu/tone từ 723 cặp. Đã train được bản KHÔNG bạc màu (`CH_E_antiwash`, 18/07) nhưng **CHƯA nối vào engine** — xem mục 8.
- Pha kế: các con "mắt" (segment cửa sổ/trời/sàn bằng model mở) · mask-based (xóa đồ, thay trời chuẩn) · QC tự động chặn ảnh xấu · service thật.
- **Không mua GPU tới khi có doanh thu.** Máy này KHÔNG có GPU (`torch 2.13.0+cpu`, `cuda.is_available()==False`) — mọi thứ chạy CPU; train thật thuê box theo giờ, runbook ở `ai_engine/specialists/auto_enhance/gpu/REMOTE_GPU_RUNBOOK.md`.

## 8. TRẠNG THÁI (cập nhật 2026-07-22 — soát lại bằng cách chạy code + mở ảnh, không tin log cũ)

> **Dự án đang TẠM DỪNG từ 18/07/2026** — chủ dự án chuyển sang nhánh anticode.vn / bot VPS Telegram. Việc dở dang duy nhất: nối `CH_E_antiwash` vào engine (xem "VIỆC KẾ TIẾP").

### ⭐ KHUNG XƯƠNG ĐÃ DỰNG & KIỂM ĐỊNH (mọi con AI sau BÁM theo đây)
- **Hợp đồng operator (BẤT BIẾN):** mỗi specialist = 1 hàm `apply(img float32 [0,1] HxWx3 BGR, params: dict) -> cùng shape`. Không resize, không re-encode. Muốn thêm con mới: (1) viết `ai_engine/specialists/<ten>/`, (2) thêm entry vào `ai_engine/orchestrator/registry.py`, (3) chạy `python -m ai_engine.conformance_check` phải PASS.
- **Kiểm định tự động (chạy lại bất cứ lúc nào):**
  - `python -m ai_engine.conformance_check` — **đã chạy 22/07: 7 PASS / 0 FAIL** (white_balance, straighten, denoise_sharpen, grass_green, sky_replace, window_pull, harsh_sun).
  - `python -m ai_engine.integration_test` — lệnh→plan→specialist thật→ảnh full-res, QC before/after.
- **Orchestrator** (`ai_engine/orchestrator/`): CLI `--command "lệnh tiếng Việt/Anh"` → planner (LLM qua gateway anticode, fallback rule-based) → engine áp chuỗi op lên full-res. Engine LUÔN `clamp_params` (op rác bị bỏ, params rác bị clamp — không crash).
- **Registry hiện có:** 8 op cơ bản (brightness/contrast/saturation/temperature/shadows_lift/highlights_recover/sharpen/auto_enhance) + 7 specialist thật (auto_white_balance, straighten, denoise, grass_green, sky_replace, window_pull, harsh_sun).

### Các con CHÍN — dùng trong pipeline mặc định (PASS conformance + đã NHÌN ẢNH)
- `auto_white_balance` (07), `denoise` (08), `straighten` (06), `grass_green` (15). Hạ tầng: `qc_scorer` (11), `ai_engine/core/quality.py` (13).
- **Pipeline 4-op deterministic cho ảnh BĐS nội thất ĐẸP, dùng được ngay** (xem `outputs/integration/compare_A_*`).
- **Giao hàng full-res (21):** `ai_engine/delivery/deliver.py` — 3000×2250 vào → đúng 3000×2250 ra, JPEG q100 4:4:4 ~9 MB, ~8–10 s/ảnh CPU, crop 100% sắc nét không halo. **Đây là lời giải cho phàn nàn "ảnh bị nén còn 176 KB" của khách hồi 14/07.** Cảnh báo: `--use-model` (bật auto_enhance pilot) làm BẠC MÀU → mặc định TẮT.
- **Demo web (24):** `python -m ai_engine.service.run_dev` → `http://127.0.0.1:8123/` (upload, gõ lệnh, slider before/after, tải full-res). **Chỉ localhost, không auth — CẤM bind ra ngoài.**

### Các con CHÍN nhưng gọi RIÊNG theo lệnh (ngoài plan mặc định)
- `window_pull` (16): cửa sổ cháy → hiện cảnh ngoài, nội thất bit-identical. Ngưỡng veto màu calib mỏng (−0.015) — calib lại khi data đổi. Cửa sổ nhìn ra cây bị bỏ qua chủ đích.
- `sky_replace` (09+17 rework): mask flood-grow tolerance hẹp + luma floor + per-column cut → mặt tiền sạch TUYỆT ĐỐI (verified crop 100%), nội thất mask=0 tận gốc (MIN_SEED_TOP_FRAC=0.12 chặn trần nhà phẳng). Còn 2 hạn chế: (a) halo/vệt sáng nhẹ quanh cột-antenna MẢNH (fix thật = alpha matting pha sau); (b) mây texture bị under-cover (recall thấp chủ đích).
- `harsh_sun` (22, đã vào registry): nén dải sáng ảnh nắng gắt, giữ màu giàu, 0 halo, 2.1 s @ 2.8 MP; `strength=0` → bit-identical. **Ngưỡng chỉnh tay trên 8 case vì data thật KHÔNG có ảnh ngoại thất mặt đất nắng gắt** — đúng loại ảnh khách bảo "mới khó". Phải tune lại khi có cặp harsh-sun thật.
- `scene_classify` (23): 82.4% (14/17). An toàn cốt lõi ĐẠT: không ảnh drone/ngoại thất nào bị gọi nhầm là interior. Yếu nhất: "trời qua cửa sổ lớn" vs "trời thật" — giới hạn của CV thuần, cần segment khung cửa sổ ở task sau.
- `remove_objects` (18, LaMa CPU): cần mask_path — chưa vào registry, chờ thiết kế UI mask.
- Bracket: `bracket_merge.py` + `deghost.py` + `bracket_deliver.py` (gộp nhiều ảnh phơi sáng trước khi vào pipeline).

### ✅ auto_enhance ĐÃ NỐI VÀO ENGINE (22/07, nghiệm thu bằng mắt 10 ảnh)
- `ops_basic.auto_enhance` giờ đọc **`checkpoints/auto_enhance_config.json`** (arch + checkpoint + kwargs — muốn đổi model chỉ sửa file này). Đang trỏ: `HDRNetV2(8,16,384,24)` + `checkpoints/gpu/CH_E_antiwash.pt`. **V2 ăn BGR THẲNG** (như lúc train/eval trên box — pilot v1 mới đổi RGB, đừng lẫn). ~2 s/ảnh 2.8 MP CPU.
- Kiến trúc cũ vẫn còn: `model.py`=HDRNet pilot (bạc màu, chỉ dùng khi config arch="v1"). ~30 checkpoint cũ trong `checkpoints/gpu/`.
- Nghiệm thu 22/07 (`outputs/engine_che/`, `outputs/deliver_v2/contact_sheet.jpg` — 10 ảnh vs target): hết bạc màu, nội thất rất gần AutoHDR.
- QC scorer: ngưỡng calib từ hồi 49 cặp → **vẫn cần calib lại trên data đủ**.

### ✅ finish_detail — con MỚI 22/07, trị đúng bệnh "ảnh mờ bợt" (khách chê 3 lần)
- **Chẩn đoán bằng số** (`tools/diagnose_blur.py`, `outputs/diagnose_blur/`): output engine có năng lượng cạnh (Laplacian var) chỉ bằng **1/8–1/12** target AutoHDR, local contrast thấp hơn 10–35% → "mờ" không phải do màu mà do **thiếu bước phục nét + vi tương phản + hạ đen** mà AutoHDR làm rất mạnh.
- `ai_engine/specialists/finish_detail/finish.py`: tách lớp chi tiết bằng **guided filter** (edge-preserving → không halo, khác unsharp Gaussian), 2 tầng clarity+detail có gate chống khuếch đại nhiễu, boost trên luma (không lệch màu), điểm đen anchor cố định ≤0.05 (bài học task 22). `clarity=detail=black=0` → trả ảnh nguyên vẹn.
- Tune trên cặp thật (`tools/tune_finish.py`): bộ chọn `clarity=0.8, detail=1.0, black=0.5` — local contrast ĐUỔI KỊP target (10.20/10.46); độ nét cạnh còn ~1/4 target (giới hạn thật: ảnh before bị warp lúc ghép cặp nên mềm sẵn, không bịa vân được bằng op tuyến tính).
- Đã vào registry + conformance (**8 PASS / 0 FAIL** 22/07).

### Chuỗi giao hàng HIỆN TẠI (deliver.py — HAI ĐƯỜNG, cập nhật 23/07)
- **Đường MODEL (`--use-model`, khuyên dùng):** `denoise → auto_enhance(CH_F, ăn ảnh THÔ) → shadow_light(0.35) → vibrance(.45/.7/dark_clean .35) → window_pull(0.9/0.45, interior) → straighten → finish_detail(0.8/1.0/black .75)`.
  - **BÀI HỌC 24/07 (vòng chấm 5 của chủ): op chồng op thì ĐÁNH NHAU** — 3 op cùng đẩy sáng (shadow_light .7 + whites .8 + dark_clean .7) ra ảnh "sáng mà mờ đục", góc ố vàng (san sáng phóng to ánh đèn vàng trong góc — flash thật của flambient mới trắng được), cửa sổ cháy đè lên công window_pull, dark_clean ăn lẹm vân gỗ. Đã hạ về mức an toàn hiện tại (v10) — SẠCH ưu tiên hơn GIỐNG. **CẤM tăng lửa cụm op này thêm nữa; khoảng cách còn lại là việc của CH_G.**
  - **shadow_light (23/07, tiến hóa 2 đời trong 1 ngày):** PHÁT HIỆN THEN CHỐT — Lab chroma của ta ĐÃ ngang/hơn target → "nhạt" mà chủ+khách nói = **THIẾU SÁNG vùng tối/góc**, không phải thiếu màu (bao vòng tune saturation là đuổi nhầm hướng). Thêm nữa: phân bố luma TOÀN CỤC đã trùng target (learn_tone_curve, ±7/255) → khác biệt là KHÔNG GIAN ("chỗ sáng chỗ tối" vs "mọi góc đều sáng" — lời chủ). v1 = band-lift theo pixel (đỡ ít); **v2 hiện tại = san lớp CHIẾU SÁNG kiểu Retinex/flambient** (guided filter tách base, kéo vùng tối về p65, k_dark 0.35). **Bài học v8→v9: sàn bảo vệ đen 0.08 để lọt lò/TV đen bóng (~0.1–0.15) bị kéo NÂU — phải 0.16.** Giới hạn thật: op không phân biệt nổi "đồ đen" vs "góc tối" cùng mức sáng — ranh giới cuối cùng này là việc của model (CH_G).
  - **dark_clean (23/07 tối, sau nhận xét vòng 2 của chủ "đa số vẫn nhạt nặng"):** đo ra bóng tối model NÂU BÙN (sat vùng tối 67–89 vs target 32–70 — đen AutoHDR TRUNG TÍNH sạch) + đen nông (p5 75 vs 58). dark_clean hạ bão hòa vùng tối (luma<0.42, mượt) + black finish nâng 0.5→0.75. Kết quả sat vùng tối: 89→65, 77→64, 67→50 (target 70/49/32). Phần còn lại phải chờ CH_G.
  - **vibrance (con mới 23/07, chốt sau khi chủ khoanh 4 vùng lỗi trên k001):** model còn hụt 2 điểm so AutoHDR — tường trắng tối hơn ~17 luma ("thiếu sáng") + màu decor nhạt ~37% sat. Con này bù: nâng trắng tiệm cận (không clip) + đẩy bão hòa **accent-aware** (màu NỔI giữa vùng trung tính như gạch hoa được đẩy mạnh; mảng màu lớn như sàn gỗ chỉ +nhẹ — cùng mức s nhưng AutoHDR xử khác nhau theo ngữ cảnh, đo `tools/hue_probe.py`). Nghiệm thu: sàn k001 khớp 56.5/56.1, cửa sổ pool2 lần đầu ra trời xanh, `outputs/compare_chf/v4cmp_*.jpg`.
  **VÌ SAO model đứng ĐẦU (bài học 23/07):** model học ánh xạ (ảnh gộp thô → AutoHDR). Chuỗi cũ cho wb/tone/saturation chạy TRƯỚC model = đầu vào lệch phân phối → màu NHẠT (double-processing, chủ dự án tự thấy bằng mắt) + vùng đen posterize xanh-teal (lò nướng k001 — bằng chứng `outputs/compare_chf/stove_stages.jpg`). Sau khi đảo: saturation khớp target theo số (30.4/30.2, 29.1/28.2, 26.2/23.3) + hết artifact + lò đen trở lại.
- **Đường DETERMINISTIC (không model, fallback):** `auto_white_balance → window_pull → denoise → straighten → harsh_sun/tone → saturation → finish_detail`.
- Đúng size gốc mọi ảnh (test tới 5464×3640), ~8–21 s/ảnh CPU.
- **Fix gate window_pull 24/07:** scene_classify gán nhầm bếp k001 = exterior_ground (conf 0.296) → mất window_pull. Luật mới: phân loại conf<0.5 KHÔNG được chặn — window_pull tự gate bằng mask nội bộ. **Kết luận cửa sổ (soi crop 100% `outputs/compare_chf/v11_window_crop.jpg`):** cây ngoài cửa sổ của AutoHDR đến từ TẤM BRACKET TỐI (dữ liệu thật) — ảnh test của ta là ảnh đơn đã merge+thu nhỏ, pixel vùng đó đã mất, KHÔNG op nào cứu được. Muốn ngang họ ở cửa sổ: phải chạy từ bộ bracket gốc của job thật.

### KIẾN TRÚC MỚI 24/07 ĐÊM: MẮT → NÃO → TAY THEO VÙNG → CHẤT LIỆU (đột phá "phân biệt đồ vật")
Chủ dự án chốt bệnh gốc sau 8 vòng chấm: "AI không phân biệt được đồ vật — mọi ảnh một toa". Đã xây trọn tầng xử lý mới trong 1 đêm:
- **MẮT** `specialists/segment_room/` (SegFormer-B0 ADE20K, model mở tự host, 0.6 s/ảnh CPU): `segment()` = 5 nhóm (tường/sàn/trần/cửa sổ/đồ vật), `segment_fine()` = nhóm CHẤT LIỆU theo tên 150 lớp (dark_appliance/wood/fabric/plant/fixture_white — khớp theo id2label, không hardcode index).
- **NÃO** `ai_engine/brain/` (diagnose 2 lần: ảnh gốc + giữa chuỗi sau model; prescribe kê toa theo SỐ ĐO ảnh đó, mỗi dòng kèm LÝ DO; CLI `python -m ai_engine.brain.run`). Não tin MASK hơn scene_classify (k001 bị gán nhầm exterior conf 0.296 → não vẫn kê window_pull đúng nhờ mask 13%).
- **TAY THEO VÙNG** `orchestrator/region_apply.py`: áp bất kỳ op nào theo mask + feather. **Giải bài toán bế tắc "lò đen vs góc tối cùng độ sáng" bằng NGỮ NGHĨA**: thắp sáng 0.9 kịch tay CHỈ trên tường/sàn/trần, đồ vật miễn nhiễm.
- **CHẤT LIỆU** `brain/material_grade.py` (v2 25/07 sau vòng chấm 9): lò/TV→đen bóng trung tính · gỗ→ấm NHẸ+vân nét (hạ liều 25/07 vì "không tự nhiên"; khóa kép chống nhuộm tủ trắng) · vải→màu sống · cây→grass_green mask-scope · sứ→trắng sạch · **window_view→khử-mù trong khung kính (tòa nhà ngoài cửa lấy lại khối+màu — pool2 lột xác)** · **art→tranh/poster tương phản nắng**. v3 (25/07, vòng chấm 10): window_view thêm NÉN NẮNG + **TÔ TRỜI đậm trong khung kính** (pixel xanh dương sáng → đậm màu) · plant thêm NÂNG SÁNG (brightness 0.22, xanh hạ 0.45) · **trần/dầm gỗ ấm LOẠI khỏi mask thắp trắng** (brain/run.py nhân (1−warm_gate)). Nghiệm thu: `outputs/material_accept/` + `outputs/material_v2/v3_*.jpg`.
- **2 con nhóm D lộ rõ sau vòng chấm 10 (content-edit, chỉnh màu KHÔNG xử được):** (a) **xóa phản chiếu trên kính** (TV/tranh phản chiếu tòa nhà — AutoHDR khử sạch, ta giữ nguyên); (b) **lấp màn TV** (màn lóa sáng → đen tắt). Đây là 2 con kế tiếp của roster khi quay lại.
- Demo thuyết phục nhất: `outputs/brain_material_cmp.jpg` (lò đen bóng như AutoHDR, tường quanh không vạ lây).
- **Não chế độ Claude-qua-gateway (ảnh khó/premium):** đã chốt thiết kế với chủ (fallback luật, đo chi phí token trước khi bật) — chưa code.

### Model auto_enhance nền — dòng tiến hóa (chốt 24/07 đêm: CH_J)
- Dòng model: CH_E (chống bạc màu) → CH_F (940 cặp) → CH_G (mở trói kênh sáng: L .25→.55, w_hi .5→.2 — trắng cải thiện 37%, sáng đều hơn) → **CH_H: PHÁ TRẦN KIẾN TRÚC** `HDRNetV2(grid_bins=10, grid_size=24, proxy_res=448, width=32)` (lưới mịn 2.25×), train TỪ ĐẦU 240 epoch (val_l1 3 đời trước đứng im ~0.053 = trần kiến trúc, không phải loss).
- **Nghiệm thu CH_H 24/07** (`outputs/compare_chh/`): lần đầu 2 chỉ tiêu kẹt nhúc nhích — p5 đen 8.08→**6.67**, bùn tối 5.17→**4.30** (trắng giữ 2.4). Mắt: lò đen sâu bóng hơn, cửa sổ pool2 lần đầu ra TRỜI XANH từ model, phòng đều sáng. Config đã trỏ `checkpoints/gpu/CH_H.pt` + **kwargs kiến trúc MỚI trong config — đổi model nhớ đổi cả kwargs**.
- **CH_I:** CH_H + loss mới `dark_fidelity` (w_dark 0.6/thresh 0.28 — option trong `losses.py`/`train_sweep.py`): p5 đen 6.67→6.08, lò/vi sóng đen sạch hơn.
- **CH_J (BẢN ĐANG CHẠY, config đã trỏ):** CH_I fine-tune 100 epoch trên **912 cặp SẠCH** (sau cách ly 28 cặp độc). val_l1 0.0529 — thấp nhất mọi thời đại. So CH_I trên sample sạch: đen 3.83/4.00, trắng 0.58/0.75 (J thắng), bùn 3.05/2.37 (I nhỉnh) — mắt gần hòa, chọn J theo nguyên tắc data sạch. Tổng đêm 24/07: **4 phiên train G→H→I→J trên 2 box (~6h GPU)**, mỗi bậc đều đo được + nhìn được.
- Backup dataset: `gdrive:autohdr_kit/dataset_v5.zip`.
- **Fix sọc mép straighten 22/07:** warp cũ để BORDER_REPLICATE trần → mép thiếu nguồn thành SỌC KÉO (bằng chứng `outputs/deliver_v2/edge_stages.jpg`; đã lọt vào batch render trước mà không ai thấy vì contact sheet thu nhỏ — **bài học: kiểm mép ở 100%**). Fix: ghép cover-zoom (≤8%) quanh tâm vào chính homography, 1 lần warp; cần quá 8% → tự giảm strength, không được thì veto. Đã quét lại 10/10 ảnh: 0 sọc.

### VIỆC KẾ TIẾP (thứ tự, cập nhật đêm 22/07)
1. ✅ ~~Ingest 20 job~~ XONG: **940 cặp sạch** (chi tiết mục DATA).
2. **QC scorer phải REWORK, không phải chỉnh ngưỡng** — calib 940 cặp (`outputs/qc_calib_940.csv`) cho thấy blur MÙ (mọi ảnh ~100 — không bắt được đúng lỗi "mờ" khách chê), washout + noise chấm NGƯỢC (phạt gu sáng-airy + texture của chính AutoHDR). Spec đầy đủ: `tasks/25-qc-rework.md`.
3. ✅ ~~Train CH_F~~ XONG 23/07. **CH_G (việc số 1 lần thuê GPU tới — vá ngoài ĐÃ TỚI HẠN, chủ dự án xác nhận "vẫn nhạt nặng" sau 2 vòng vá):** gốc mọi lỗi còn lại (góc khuất không mở, TV/đen nhạt, tổng thể nhạt) = loss kìm kênh sáng. Sửa: `lab_weights` L 0.25→0.55 · `w_hi` 0.5→0.2 · giữ chroma a/b 1.5. Mục tiêu nghiệm thu đo được (từ v6 23/07): p5 luma khớp target ±5 · sat vùng tối (luma<80) ≤ target+10 · tường trắng luma ≥ target−5 — RỒI mới nhìn ảnh chốt. Warm-start từ CH_F, ~150 epoch ≈ 1.5h RTX 3060.
4. Ảnh chụp xuyên kính bị phản chiếu: tone op không cứu được (target AutoHDR xóa hẳn phản chiếu) — cần con riêng pha sau.
5. 296 cặp `data/review` chưa đạt: có thể cứu thêm bằng align mạnh hơn (pha sau).

### DATA (cập nhật 24/07 đêm — PHÁT HIỆN DATA ĐỘC)
- `data/pairs/` = **912 cặp SẠCH** (940 − 28 cách ly). **24/07 quét ra 28 cặp ĐỘC** (`tools/scan_bad_pairs.py`, bằng chứng `outputs/bad_pair_scan/*.jpg`): 26 cặp before bị PHỦ TÍM HỒNG toàn ảnh (lỗi develop RAW loạt máy 5O2A — job j002/003/004/005/026/056 hỏng GẦN TRỌN JOB) + 2 cặp chụp xuyên kính phản chiếu (j053/j054-type). Chúng dạy model phép biến đổi rác suốt từ 15/07 → nghi phạm góp phần "phòng có đồ vật không tiến bộ". Đã move vào `data/quarantine_pairs/` (cả local + box). **Bài học: ingest phải có bước quét màu-cast bất thường; heuristic: p5_before>55 & contrast_ratio<0.62 (+ nhìn mắt xác nhận).**
- Kèm `data/pairs_removal/` 34 cặp (data cho object_removal sau) + `data/review/` 296 cặp chưa đạt (cứu tiếp pha sau).
- Batch này phát hiện + vá 2 lỗi ingest: (a) gộp bracket chỉ AlignMTB → ảnh ma khi khách chụp lệch vị trí — vá bằng homography + veto frame (đã port sang cả `bracket_merge.py` phía giao hàng); (b) job nén RAW trong `.zip` bị coi là rỗng — vá tự bung. Ledger PS ghi UTF-8 BOM làm job chạy lại — ledger phải ghi ASCII.
- Pipeline: `tasks/02` ingest CR3 → `rescue_review` (undistort cứu cặp align_low) → `dedup_pairs` (khử trùng). Nguồn: `data_pairing/fetch_dropbox_job.py`, `fetch_drive_job.py`, `ingest_sheet.py`, `ingest_mixed.py` (link trộn cả ảnh sửa lẫn chưa sửa), `rclone.exe` trong `tools/`.
- Toàn bộ ảnh "after" đến từ AutoHDR → model đang học BẮT CHƯỚC gu AutoHDR. Đó là chủ đích.
- **BÀI HỌC:** KHÔNG đụng `data/pairs/` khi training đang chạy (dataset index lúc start → move file giữa chừng = crash).

### PHẢN HỒI KHÁCH HÀNG THẬT (14–17/07 — kim chỉ nam chất lượng)
- 14/07: "ảnh còn mờ bợt, sai màu, nhiễm màu, lấy cửa sổ chưa đẹp"; ảnh giao bị nén còn 176 KB, vỡ pixel khi phóng to → **đã fix bằng task 21**.
- 16/07: "vẫn còn mờ bợt với nhiễm màu; ngoài trời chưa đủ tươi màu, cần ấm hơn xíu — **nhìn chung tốt hơn hôm trước**".
- Khách nhấn mạnh: đừng test bằng ảnh drone (dễ), phải là **ảnh ngoài trời dưới mặt đất, nắng gắt** mới đánh giá được.
- 17/07 chủ dự án tự nhận: đa số ảnh vẫn mờ, chưa đẹp.

### BÀI HỌC VÀNG (lặp lại nhiều lần, đừng quên)
- **Worker BỊA report** (Task 09 khai thay trời 5 ảnh ngoại thất, thực tế 5/5 nội thất giữ nguyên). LUÔN tự nhìn ảnh.
- **Số đẹp ≠ ảnh đẹp** (Task 22: bảng DR 1.000→0.358 trông ấn tượng nhưng ảnh hỏng hoàn toàn).
- **CẤM demo/nghiệm thu/gửi khách bằng ảnh trong `data/pairs`** (phát hiện 22/07): ảnh đó bị pipeline làm-data làm MỀM 3 tầng chủ đích (RAW develop half-size → resize 2048 → warp align), lap_var chỉ 11–32 so với ~450 của ảnh gốc thật → mọi bản render từ đó đều "mờ" oan. Bằng chứng ngược: ảnh vào đúng cỡ gốc (j027 5464×3640) → bản giao lap 546, NÉT HƠN target AutoHDR (448). Demo phải dùng ảnh job nguyên gốc.

### LỊCH SỬ CHAT
Transcript các phiên cũ nằm ở `C:\Users\Administrator\.claude\projects\C--Users-Administrator-Downloads-autohdr\*.jsonl` (13 phiên, 04→22/07/2026) — **không** ở project key của Desktop, vì hồi đó chạy với cwd `Downloads\autohdr`. Trong đó trộn cả dự án anticode.vn.
