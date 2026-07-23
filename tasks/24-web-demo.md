# Task 24 — WEB DEMO UI (upload a photo → see before/after, for showing customers)

**Assigned to:** Worker (Sonnet) · **Review:** Claude architect · **Read `CLAUDE.md` first.**

## Context
The owner needs to show customers results. There's already a FastAPI service (`ai_engine/service/app.py`, localhost:8123, endpoints /health /ops /edit /qc). Build a single self-contained HTML page that talks to it: upload an image, get the edited full-res result, show a before/after slider. Purely for local demos.

## Files (`ai_engine/service/`)
1. Add a `GET /` route to `app.py` (or a new `web.py` router mounted in app) that serves `ai_engine/service/static/index.html`. Keep the existing endpoints intact.
2. `static/index.html` — ONE self-contained file (inline CSS/JS, no external CDNs — the machine may be offline):
   - Drag-drop / file-picker to upload an image.
   - Optional text box for a command (Vietnamese ok) → sends to `/edit` as `command`; if empty, uses default deterministic plan.
   - Shows a BEFORE/AFTER comparison slider (draggable divider) at good size. Also show the X-Plan-Applied and X-QC-Overall headers returned.
   - A download button for the edited full-res result.
   - Clean, simple, professional look (this is shown to real-estate customers). Light theme, readable.
3. Update `test_service.py` (or add a note) — no need to fully test the browser, but verify `GET /` returns 200 and the HTML, and that a `/edit` round-trip still works headless.

## Acceptance (run for real)
- [ ] Start the server (localhost:8123), `curl http://127.0.0.1:8123/` returns the HTML (200). Confirm the page references /edit correctly.
- [ ] `/edit` still works (existing test passes). Kill server after.
- [ ] Describe (in the report) exactly how the owner opens it: start server command + open http://127.0.0.1:8123/ in a browser. Include a screenshot IF you can render one; otherwise describe the layout.
- [ ] Honest report: what works, what's stubbed, browser-compat notes.

## DO NOT
- No files outside `ai_engine/service/` + report. No external CDN/network deps in the HTML. localhost only. Keep existing endpoints working.

---

## BÁO CÁO (worker, 2026-07-14)

### Đã làm
- `ai_engine/service/app.py`: thêm `GET /` (FileResponse trả `static/index.html`, `media_type=text/html`). Không đụng `/health` `/ops` `/edit` `/qc`.
- `ai_engine/service/static/index.html`: 1 file duy nhất, CSS+JS inline, **không** CDN/font ngoài. Có: dropzone kéo-thả + bấm chọn file (validate JPG/PNG ở client), ô nhập lệnh tiếng Việt tuỳ chọn (rỗng → dùng `DEFAULT_PLAN` 4 op như cũ), nút "Chỉnh ảnh" gọi `POST /edit` bằng `fetch`+`FormData`, slider before/after kéo bằng chuột/touch hoặc click-to-set, hiển thị `X-Plan-Applied` / `X-QC-Overall` / kích thước ảnh (đọc từ `naturalWidth/Height` của ảnh SAU, không tự tính lại), nút tải ảnh full-res (`<a download>` trên blob URL, không convert lại nên byte-for-byte đúng response từ server), nút "Chỉnh ảnh khác" reset form. Theme sáng, layout 1 cột, chữ tiếng Việt không dấu trong JS string literals để tránh lỗi encode khi mở bằng notepad, nhưng HTML hiển thị (text node) dùng tiếng Việt có dấu bình thường (UTF-8, đã test hiển thị đúng trong Edge).
- `ai_engine/service/test_service.py`: thêm bước `GET /` giữa `/ops` và `/edit` — kiểm `status==200`, `content-type` chứa `text/html`, và HTML có chứa chuỗi `/edit`.

### Đã CHẠY THẬT (không suy diễn)
1. `python -m ai_engine.service.test_service` (server con tự bật/tắt) — **PASS** cả 2 lần chạy liên tiếp. Log lần cuối:
   - `[ROOT] status=200 is_html=True references_edit=True`
   - `[EDIT] status=200 X-Plan-Applied=brightness,temperature X-QC-Overall=89.63` (lần có COMMAND tiếng Việt "tăng sáng nhẹ, ấm hơn")
   - `out_shape=(1366, 2048) in_shape=(1366, 2048) size_ok=True`
   - `[QC] overall=89.63 has_6_scores=True`
   - `[BAD-IMAGE] status=400`, `[BAD-OP-IN-PLAN] status=200` (op rác bị bỏ qua, không crash)
   - Netstat sau kill: không còn dòng LISTENING trên 8123.
2. Bật server thủ công (`python -m ai_engine.service.run_dev`), `curl http://127.0.0.1:8123/` → `HTTP_STATUS:200 CONTENT_TYPE:text/html; charset=utf-8 SIZE:12048`, xác nhận HTML chứa đúng 1 tham chiếu `/edit`.
3. `curl -X POST /edit` không kèm `command` (plan mặc định) trên ảnh thật `data/pairs/before/_ML_1605.jpg` (2048×1366) → headers `x-plan-applied: auto_white_balance,denoise,straighten,grass_green`, `x-qc-overall: 89.25`; **đã đọc lại file JPG trả về bằng cv2, shape=(1366, 2048, 3) đúng full-res gốc**, không bị resize/nén lại.
4. **Đã tự nhìn ảnh** (Read tool, không tin số liệu): before/after của ảnh mặt tiền phố — sau khi áp plan mặc định, ảnh sáng/trắng cân hơn, mảng cây bụi xanh hơn (grass_green), không méo, không mất nét — khớp với hành vi pipeline 4-op đã biết ở `outputs/integration/`.
5. **Test UI thật qua trình duyệt** (không chỉ curl): dùng Selenium + msedgedriver (Edge headless có sẵn trên máy, `C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe`) — mở `http://127.0.0.1:8123/`, gán file thật vào `<input type=file>`, bấm nút "Chỉnh ảnh" thật (JS thật chạy trong trình duyệt thật, không mock), đợi `#compareWrap` hiện ra, chụp screenshot. Kết quả: status text "Xong. Kéo con trượt để so sánh Trước/Sau.", `planApplied=auto_white_balance,denoise,straighten,grass_green`, `qcOverall=89.25`, kích thước hiển thị đúng "2048 x 1366 px". Đã xem 2 screenshot (trang rỗng lúc mới mở, và trang sau khi chỉnh ảnh với slider trước/sau + 2 nhãn TRƯỚC/SAU + nút tải/chỉnh ảnh khác) — layout đúng như spec, không vỡ giao diện, không lỗi console gây crash script.
6. Dọn dẹp: kill server thủ công (`taskkill /F /PID`), xác nhận `netstat` không còn LISTENING trên 8123; không còn tiến trình `msedge.exe`/`msedgedriver.exe` treo lại sau khi `driver.quit()`.

### Cách chủ mở trang (thật, đã test đúng các bước này)
```
cd C:\Users\Administrator\Desktop\autohdr
python -m ai_engine.service.run_dev
```
rồi mở trình duyệt bất kỳ (đã test Edge; Chrome/Firefox cùng engine JS chuẩn nên tương thích) tới `http://127.0.0.1:8123/`. Kéo/thả ảnh hoặc bấm vùng chấm chấm để chọn file, (tuỳ chọn) gõ lệnh chỉnh sửa, bấm "Chỉnh ảnh", đợi vài giây, kéo mũi tên ⇔ ở giữa ảnh để so Trước/Sau, bấm "Tải ảnh đã chỉnh (full-res)" để lưu JPG kết quả. Đóng server bằng Ctrl+C ở terminal.

### Thật thà: hạn chế / chưa làm
- **Không có test tự động cho việc kéo-thả file** (`dragenter`/`drop` event) — Selenium không giả lập được native OS drag-drop dễ dàng nên chỉ test qua đường `input[type=file].send_keys(...)`, tức đường "bấm để chọn file". Cơ chế JS xử lý cả 2 đường giống nhau (cùng gọi `chooseFile()`) nên rủi ro thấp, nhưng bản thân sự kiện `dragstart`/`drop` từ OS thật (kéo file từ Explorer) CHƯA được click-test bằng tay/chuột thật trong phiên này.
- **Chưa test trên Chrome/Firefox thật**, chỉ test trên Edge (Chromium engine) — do đó không có bất ngờ lớn vì code không dùng API riêng của Edge, nhưng chưa xác nhận thực tế.
- Ảnh trước/sau trong slider được nạp lại 2 lần khác nhau ở client (ảnh gốc qua `URL.createObjectURL(selectedFile)`, ảnh sau qua blob response) — nếu ảnh gốc rất lớn (nhiều chục MB, ảnh RAW-derived cỡ lớn), trình duyệt vẫn tải cả 2 bản base64/blob vào bộ nhớ RAM tab; chưa test với ảnh > 20MB, có thể chậm trên máy yếu (không phải máy production này).
- QC/Plan hiển thị lấy trực tiếp từ header response — nếu sau này đổi cấu trúc header ở `/edit` (ví dụ đổi tên `X-Plan-Applied`), trang sẽ hiện "—"/"(không rõ)" im lặng chứ không báo lỗi rõ ràng; chấp nhận được cho bản demo nội bộ.
- Trang không có xác thực (auth) — đúng theo phạm vi task (demo local, localhost only), nhưng lưu ý nếu sau này có ý định mở cổng ra ngoài KHÔNG được làm vậy nếu chưa thêm auth (CLAUDE.md cũng cấm bind ra ngoài localhost).
- Không sửa `run_dev.py` — vẫn bind `127.0.0.1:8123` như cũ, không đổi.
