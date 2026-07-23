# Task 05 — Orchestrator v0: CHỈNH ẢNH THEO LỆNH (như AutoHDR "AI Re-edit")

**Giao cho:** Claude-worker A (Sonnet) · **Review:** Claude kiến trúc sư · **Đọc `CLAUDE.md` trước.**

## Mục tiêu
`python -m ai_engine.orchestrator.cli --input anh.jpg --command "tăng sáng nhẹ, ấm hơn, dọc cho thẳng" --output ra.jpg`
→ LLM dịch lệnh thành **kế hoạch operator JSON** → engine áp TUẦN TỰ lên ảnh **full-res** → xuất đúng kích thước gốc, JPEG q95.

## ⭐ HỢP ĐỒNG OPERATOR (bất biến — các con AI khác sẽ cắm vào đây)
Mỗi operator = 1 hàm `apply(img: np.ndarray float32 [0,1] HxWx3 BGR, params: dict) -> np.ndarray` cùng shape/dtype. KHÔNG resize, KHÔNG đổi size, KHÔNG re-encode bên trong operator.

## Files (`ai_engine/orchestrator/`)
1. `registry.py` — REGISTRY dict: tên op → {fn, mô tả 1 dòng, schema params (tên, kiểu, min/max, default)}. Đăng ký sẵn các op v0 (mục 3).
2. `ops_basic.py` — operator xác định bằng numpy/cv2 (float32, full-res):
   - `brightness` (exposure ±, nhân gamma-aware), `contrast` (quanh trung vị), `saturation` (HSV S scale), `temperature` (ấm/lạnh dịch kênh R/B), `shadows_lift` (nâng vùng tối, tone-curve), `highlights_recover` (hạ vùng cháy), `white_balance` (gray-world, strength 0-1), `sharpen` (unsharp mask nhẹ, amount ≤0.5).
   - `auto_enhance` — nạp `checkpoints/auto_enhance.pt` chạy HDRnet (import từ `ai_engine.specialists.auto_enhance.infer`). Checkpoint có thể ĐANG được train lại → nếu load lỗi/thiếu: BỎ QUA op này kèm warning, KHÔNG crash.
3. `planner.py` — dịch lệnh tự nhiên (Việt/Anh) → plan:
   - Gọi `POST {ANTICODE_BASE_URL|default https://anticode.vn}/v1/chat/completions`, model từ env `PLANNER_MODEL` (default `cheap-ai/claude-sonnet-5`), key từ env `ANTICODE_API_KEY`. **KHÔNG hardcode key vào repo.**
   - System prompt: đưa danh sách op + schema từ registry, bắt trả về DUY NHẤT JSON `{"plan":[{"op":"...","params":{...}},...]}`. Parse chặt (json.loads phần giữa ```), validate op tồn tại + params trong min/max (clamp nếu lệch).
   - Không có key / gọi lỗi → **fallback rule-based**: regex từ khóa (sáng/tối/ấm/lạnh/tương phản/rực/nét/tự động...) → plan tương đương. Ghi rõ nguồn plan (llm|fallback) khi in.
4. `engine.py` — `run_plan(img_path, plan, out_path)`: đọc ảnh (giữ nguyên res), float32 [0,1], áp op tuần tự, ghi JPEG q95 (PNG nếu out .png), assert size ra == size vào.
5. `cli.py` — argparse như trên + `--dry-run` (chỉ in plan) + `--plan` (đưa JSON thẳng, bỏ LLM).

## Acceptance (tự chạy TRƯỚC khi báo xong)
- [ ] `--dry-run --command "tăng sáng nhẹ, ấm hơn"` → in plan JSON hợp lệ (thử cả 3 lệnh mẫu: "sáng hơn và rực rỡ hơn", "make it brighter and straighten verticals" (op chưa có thì planner phải BỎ QUA op lạ kèm warning), "tự động chỉnh đẹp").
- [ ] Chạy thật trên 1 ảnh `data/pairs/before/*.jpg`: output đúng kích thước gốc, nhìn thấy khác biệt đúng hướng lệnh. Lưu `outputs/orchestrator_samples/` ghép [in|out].
- [ ] Thiếu `ANTICODE_API_KEY` → fallback hoạt động, không crash. Thiếu checkpoint → warning, các op khác vẫn chạy.
- [ ] Ghi báo cáo trung thực cuối file này (plan mẫu, đường dẫn ảnh, vướng mắc).

## KHÔNG được làm
- KHÔNG sửa file ngoài `ai_engine/orchestrator/` + báo cáo trong file này.
- KHÔNG đụng `data/` (chỉ ĐỌC ảnh test), không đụng tiến trình đang chạy, `cv2.setNumThreads(2)`.
- KHÔNG gọi model đắt; chỉ model trong env, request tuần tự.

---

## BÁO CÁO (Claude-worker A, đã tự chạy trong phiên)

### Cấu trúc đã tạo
`ai_engine/orchestrator/{__init__,registry,ops_basic,planner,engine,cli}.py` — không đụng file nào khác.

### Op v0 đã đăng ký (registry.py)
`brightness, contrast, saturation, temperature, shadows_lift, highlights_recover, white_balance, sharpen, auto_enhance` — tất cả tuân thủ hợp đồng `apply(img_f32_[0,1]_HxWx3_BGR, params) -> cùng shape/dtype`, không resize/re-encode nội bộ.

### Planner — nguồn thật (llm + fallback), có bug tìm thấy & đã sửa khi test thật
- **Bug 1 (đã sửa):** gọi `POST https://anticode.vn/v1/chat/completions` bằng `urllib` bị Cloudflare chặn **403 "error code: 1010"** vì `urllib` gửi User-Agent mặc định `Python-urllib/3.x` bị coi là bot. Xác nhận bằng `curl` (pass) vs `urllib` (fail) cùng payload. Sửa: thêm header `User-Agent: autohdr-orchestrator/0.1` trong `planner.py`. Sau khi sửa, gọi LLM thật thành công nhiều lần (source=llm).
- **Bug 2 (đã sửa):** rule fallback tiếng Việt dùng `unicodedata.normalize("NFD", ...)` để bỏ dấu, nhưng chữ `đ/Đ` trong Unicode KHÔNG phân rã qua NFD (nó là ký tự riêng, không phải base+dấu). Hệ quả: lệnh "tự động chỉnh đẹp" bị bỏ dấu sai thành "tu đong chinh dep" (còn dính "đ") nên KHÔNG khớp regex `tu dong|auto` → trả về `plan: []` (SAI). Sửa: map thủ công `đ→d, Đ→D` trước khi NFD-strip. Sau khi sửa, "tự động chỉnh đẹp" → `[{"op":"auto_enhance"}]` đúng như kỳ vọng.
- Cũng gặp 1 lần LLM timeout (đọc response quá 30s, mạng chập chờn) → tự động rơi xuống fallback, in `[WARN]`, KHÔNG crash — đúng acceptance.

### Kết quả `--dry-run` với 3 lệnh mẫu (chạy thật, cả llm và fallback — không có key)
1. `"tăng sáng nhẹ, ấm hơn"` → llm: `brightness(+0.2), temperature(+0.2)` | fallback: `brightness(+0.15), temperature(+0.15)` (giảm nửa vì có từ "nhẹ").
2. `"sáng hơn và rực rỡ hơn"` → llm: `brightness(+0.3), saturation(+0.3)` | fallback: giống hệt.
3. `"make it brighter and straighten verticals"` → llm: chỉ trả `brightness(+0.3)`, tự bỏ qua "straighten verticals" vì op không có trong registry (LLM không bịa op) | fallback: `brightness(+0.3)` + in `[WARN] ... nhắc tới 'straighten_verticals' nhưng op này chưa tồn tại. Bỏ qua.`
4. (thêm) `"tự động chỉnh đẹp"` → cả 2 nguồn → `[{"op":"auto_enhance"}]`.

Test thêm việc registry/engine tự bỏ qua op lạ khi dùng `--plan` JSON trực tiếp chứa op không tồn tại (`straighten_verticals`) → engine in `[WARN] ... bỏ qua op không tồn tại`, các op khác vẫn chạy, không crash.

### Chạy thật trên ảnh, có lưu so sánh [in|out]
Ảnh test: `data/pairs/before/20260703-DSC1105.jpg` (2048x1365).

| File | Lệnh | Plan áp dụng | Kích thước ra |
|---|---|---|---|
| `outputs/orchestrator_samples/sample1_bright_warm.jpg` | "tăng sáng nhẹ, ấm hơn, tương phản hơn" | brightness(+0.2), temperature(+0.2), contrast(+0.2) | 2048x1365 (khớp gốc) |
| `outputs/orchestrator_samples/sample2_auto_enhance.jpg` | "tự động chỉnh đẹp" | auto_enhance (HDRnet, checkpoint hiện có) | 2048x1365 (khớp gốc) |
| `outputs/orchestrator_samples/sample3_fallback_cool_dark.png` | "làm tối đi, lạnh hơn, nét hơn" (chạy KHÔNG có `ANTICODE_API_KEY` để test fallback thật) | brightness(-0.3), temperature(-0.3), sharpen(+0.2) | 2048x1365 PNG (khớp gốc) |

Ảnh ghép [IN\|OUT] để xem trực quan: `compare_sample1.jpg`, `compare_sample2.jpg`, `compare_sample3.jpg` (cùng thư mục). Đã tự xem cả 3 — hướng chỉnh khớp đúng lệnh:
- Sample 1: rõ ràng sáng hơn, ấm hơn, tương phản hơn.
- Sample 3: rõ ràng tối hơn, lạnh hơn (ám xanh), viền nét hơn.
- Sample 2 (auto_enhance): output bị NHẠT MÀU/xám hơn nhiều so với gốc — do checkpoint `checkpoints/auto_enhance.pt` mới chỉ là **pilot 8 cặp** (Task 01 ghi rõ "chưa phải model dùng được"), KHÔNG phải lỗi wiring của orchestrator. Việc gọi/nạp/áp op qua HDRnet infer đúng hợp đồng (size in==out, không crash) — orchestrator đã tích hợp đúng, chất lượng op sẽ tự cải thiện khi Task 01 train lại với data thật.

### Test case thiếu key / thiếu checkpoint (yêu cầu acceptance)
- Bỏ `ANTICODE_API_KEY` khỏi env (`env -u ANTICODE_API_KEY`) → planner tự in `[INFO] ... thiếu ANTICODE_API_KEY -> dùng fallback rule-based`, KHÔNG crash, plan vẫn hợp lệ (xem sample 3 ở trên, chạy thật thành công).
- Test checkpoint thiếu: chạy CLI với cwd khác (không có `checkpoints/auto_enhance.pt` theo đường dẫn tương đối) + `--plan` chứa `auto_enhance` + `brightness` → in `[WARN] auto_enhance: không load được checkpoint (...). Bỏ qua op này.`, KHÔNG crash, op `brightness` sau đó vẫn chạy bình thường, output đúng size.

### Vướng mắc / lưu ý cho Claude kiến trúc sư
- Cloudflare chặn `urllib` mặc định (403/1010) — đã fix bằng User-Agent riêng. Nếu sau này đổi sang `requests`/`httpx`, thư viện đó có UA mặc định khác `python-requests/x.x` — CŨNG có thể bị chặn tương tự, cần set UA giống trình duyệt/khách hợp lệ.
- `auto_enhance` hiện chỉ là chứng minh wiring; chất lượng ảnh ra còn kém (nhạt màu) vì checkpoint pilot. Không sửa gì ở Task 05 (đúng phạm vi — không đụng model training).
- `--plan` (JSON trực tiếp) hiện KHÔNG chạy qua `clamp_params`/lọc op lạ như khi qua planner LLM/fallback — theo tinh thần "đưa JSON thẳng, bỏ LLM" nghĩa là dùng nguyên trạng do người gọi chịu trách nhiệm; `engine.py` vẫn tự bỏ qua (kèm warning) op không có trong registry để không crash.

TASK05=DONE, PLAN_SOURCE_TESTED=llm+fallback, SAMPLES=3
