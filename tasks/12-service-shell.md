# Task 12 — SERVICE SHELL v0 (FastAPI: upload → edit-by-command → download, like AutoHDR's flow)

**Assigned to:** Worker V (Sonnet on Max) · **Review:** Claude architect · **Read `CLAUDE.md` first.**

## Goal
Wrap the orchestrator in a local HTTP service so the whole engine is callable like AutoHDR's backend: send a photo + a command, get the edited full-res photo back. This is the seed of Trụ 5 (vỏ).

## Files (`ai_engine/service/`)
1. `app.py` (FastAPI, already in requirements):
   - `GET /health` → `{"status":"ok","ops":<count>}`.
   - `GET /ops` → registry summary (from `ai_engine.orchestrator.registry.get_registry_summary`).
   - `POST /edit` (multipart): fields `image` (file), `command` (str, optional), `plan` (JSON str, optional — bypasses planner). Behavior: save upload to a temp file under `outputs/service_tmp/` (uuid name), build plan (planner if `command`, parse if `plan`, else default deterministic plan: auto_white_balance→denoise→straighten→grass_green), `run_plan`, return the edited image bytes (`image/jpeg`, q95) + headers `X-Plan-Applied` (op list) + `X-QC-Overall` (score from qc_scorer on the result). Clean up temp files after response.
   - `POST /qc` (multipart `image`) → full QC dict as JSON.
   - Errors: bad image → 400 JSON; planner failure → falls back internally (engine already safe). NEVER 500 for a bad op name.
2. `run_dev.py` — starts uvicorn on **127.0.0.1:8123** (LOCALHOST ONLY — this machine runs public production services; do not bind 0.0.0.0, do not touch ports 9000/9001/8080).
3. `test_service.py` — script that: starts the server as a subprocess, waits for /health, POSTs a real image from `data/pairs/before/` with command "tăng sáng nhẹ, ấm hơn" (planner may fall back rule-based without API key — fine), saves response to `outputs/service_samples/edited.jpg`, checks size == original, prints X-Plan-Applied + X-QC-Overall, calls /qc, then **terminates the server process** (nothing left listening). Run it for real.

## Acceptance
- [ ] `python -m ai_engine.service.test_service` passes end-to-end: health OK, /edit returns image with SAME dimensions, plan header sensible, /qc returns 6 scores, server killed at the end (verify no listener on 8123 after: `netstat -ano | findstr 8123` empty).
- [ ] LOOK at `outputs/service_samples/edited.jpg` (Read tool) — direction matches the command.
- [ ] Honest report at end of this file (include the exact curl equivalent for the user's docs).

## DO NOT
- No files outside `ai_engine/service/` + `outputs/service_samples|service_tmp/` + report here. LOCALHOST:8123 only, kill server after tests. `cv2.setNumThreads(2)`. `data/` read-only.

---

## BÁO CÁO (Worker V, 2026-07-14)

### Đã làm
- `ai_engine/service/app.py` — FastAPI app: `GET /health`, `GET /ops`, `POST /edit` (multipart `image` + optional `command`/`plan`), `POST /qc`. Ảnh upload lưu vào `outputs/service_tmp/<uuid>_in<ext>`, chạy `run_plan` ghi ra `<uuid>_out.jpg`, đọc bytes trả về rồi **luôn xoá cả 2 file tạm trong `finally`** (kể cả khi lỗi).
- `ai_engine/service/run_dev.py` — `uvicorn.run(..., host="127.0.0.1", port=8123)`. Không bind `0.0.0.0`, không đụng 9000/9001/8080 (đã kiểm `netstat` trước khi code: 3 port đó đang LISTENING bởi service khác trên máy, để nguyên).
- `ai_engine/service/test_service.py` — tự spawn server bằng subprocess, poll `/health` tới khi sẵn sàng, gọi `/edit` với ảnh thật + lệnh `"tăng sáng nhẹ, ấm hơn"`, lưu `outputs/service_samples/edited.jpg`, gọi `/qc`, rồi **luôn terminate server trong `finally`** kể cả khi có assertion fail giữa chừng.

### Quyết định khi spec mơ hồ (chọn phương án đơn giản nhất thoả acceptance)
- **Plan mặc định** (không có `command`/`plan`): dùng đúng 4 op + params mặc định của schema (`auto_white_balance→denoise→straighten→grass_green`), giống hệt `PLAN_DETERMINISTIC` trong `integration_test.py` — tái dùng pipeline đã "chín", không tự nghĩ params mới.
- **`plan` JSON bypass**: chấp nhận cả `{"plan":[...]}` lẫn list trần `[...]`, giống hệt logic `cli.py` để nhất quán CLI/HTTP. JSON hỏng hoặc không phải list → 400, không cho lọt xuống engine.
- **Không 500 khi op rác**: `engine.run_plan` đã tự bỏ qua op không tồn tại (in WARN) — không cần sửa gì thêm; endpoint chỉ bọc thêm `try/except Exception -> 400` quanh `run_plan` để chặn các lỗi bất ngờ khác (vd. plan là list nhưng phần tử không phải dict) không bao giờ biến thành 500.
- **`X-QC-Overall`**: tính QC trên ẢNH KẾT QUẢ đọc lại từ `out_path` (không tính trên buffer trong RAM) để chắc chắn số khớp với đúng bytes JPEG q95 thực sự trả về client.

### Chạy thật (trong phiên này)
```
python -m ai_engine.service.test_service
```
Kết quả in ra thật (không bịa):
```
[*] Anh test: data/pairs/before/_ML_1605.jpg (2048x1366)
[HEALTH] {'status': 'ok', 'ops': 13}
[OPS] 13 op dang ky: brightness, contrast, saturation, temperature, shadows_lift,
      highlights_recover, sharpen, auto_enhance, auto_white_balance, straighten,
      denoise, grass_green, sky_replace
[EDIT] status=200
[EDIT] X-Plan-Applied=brightness,temperature
[EDIT] X-QC-Overall=89.63
[EDIT] out_shape=(1366, 2048) in_shape=(1366, 2048) size_ok=True
[QC] status=200 overall=89.63 flags=['noisy'] needs_human=False has_6_scores=True
[BAD-IMAGE] status=400 (ky vong 400)               -> PASS
[BAD-OP-IN-PLAN] status=200 X-Plan-Applied=auto_white_balance  -> op rac bi bo qua, khong crash
[NETSTAT] LISTENING con lai tren 8123 sau khi kill: []   -> PASS (chi con TIME_WAIT tam thoi cua
      client cu, tu het sau vai chuc giay, KHONG phai server con song — da xac minh lai bang
      cach tu bind() thanh cong vao 127.0.0.1:8123 ngay sau khi test xong)
KET QUA: PASS
```
Lưu ý trung thực: `command="tăng sáng nhẹ, ấm hơn"` không có `ANTICODE_API_KEY` trong session này nên planner rơi vào **fallback rule-based** (đúng như spec cho phép: "planner may fall back rule-based ... fine"), khớp regex `sang`→brightness+0.3 và `am hon`→temperature+0.3, rồi bị hint "nhe" nhân 0.5 → plan thực tế là `brightness(+0.15), temperature(+0.15)`. **Không** chạy qua 4-op pipeline "chín" vì command không match op nào trong đó — đây là hành vi đúng của planner hiện tại, không phải bug của Task 12.

### Nhìn ảnh (Read tool, không đoán)
Đã mở cả `data/pairs/before/_ML_1605.jpg` (gốc) và `outputs/service_samples/edited.jpg` (kết quả) bằng mắt. **Khác biệt RẤT nhẹ, gần như không thấy bằng mắt thường** trên preview — đúng như kỳ vọng vì params chỉ 0.15/0.15 (do "nhẹ"). Đo pixel thật để xác nhận có đổi đúng hướng (không phải no-op):
```
mean R diff = +10.97   (ấm hơn: đỏ tăng)
mean B diff = -0.66    (ấm hơn: xanh dương giảm nhẹ)
mean luma diff = +5.23 (sáng hơn)
```
→ Hướng khớp lệnh ("sáng nhẹ, ấm hơn"): sáng lên + ấm lên, biên độ nhỏ đúng với mild-hint "nhẹ". Không phải ảnh bị lỗi/no-op, chỉ là thay đổi tinh tế theo đúng thiết kế planner hiện tại — nếu muốn thấy rõ hơn cần lệnh không có từ "nhẹ" hoặc dùng plan JSON trực tiếp.

### Giới hạn / rủi ro đã biết
- `test_service.py` dùng `requests` (đã có sẵn trong venv, KHÔNG có trong `requirements.txt`) — nếu môi trường khác thiếu `requests`, script sẽ ImportError. Không thêm vào requirements.txt vì ngoài phạm vi file được phép sửa của Task 12; ghi chú lại đây để Claude cân nhắc thêm sau.
- `/edit` hiện luôn trả JPEG q95 kể cả khi ảnh gốc là PNG (đúng spec: "return the edited image bytes (image/jpeg, q95)") — nghĩa là PNG lossless bị nén JPEG ở bước trả về; đây là spec, không phải lỗi, nhưng cần lưu ý nếu sau này cần giữ PNG lossless cho use-case khác.
- Chưa test tải đồng thời (concurrent requests) — spec Task 12 không yêu cầu, và máy chỉ chạy "one python process at a time" nên không nên test song song ở đây.

### curl tương đương (cho docs)
```bash
# Health
curl http://127.0.0.1:8123/health

# Danh sách op
curl http://127.0.0.1:8123/ops

# Edit theo lenh tu nhien
curl -X POST http://127.0.0.1:8123/edit \
  -F "image=@data/pairs/before/_ML_1605.jpg" \
  -F "command=tăng sáng nhẹ, ấm hơn" \
  -o edited.jpg -D -   # -D - de in ca header X-Plan-Applied / X-QC-Overall

# Edit bang plan JSON truc tiep (bo qua planner)
curl -X POST http://127.0.0.1:8123/edit \
  -F "image=@data/pairs/before/_ML_1605.jpg" \
  -F 'plan={"plan":[{"op":"auto_white_balance","params":{}},{"op":"denoise","params":{}}]}' \
  -o edited.jpg

# QC rieng
curl -X POST http://127.0.0.1:8123/qc -F "image=@outputs/service_samples/edited.jpg"
```

TASK12=DONE — test_service.py: 13 ops registered, /edit 200 (2048x1366→2048x1366, size giữ nguyên), X-QC-Overall=89.63, /qc trả đủ 6 điểm (overall=89.63, flags=['noisy']), ảnh xấu →400, op rác trong plan bị bỏ qua an toàn (200), server bị kill sạch (0 LISTENING trên 8123 sau khi test xong).
