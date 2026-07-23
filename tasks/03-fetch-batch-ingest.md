# Task 03 — Tải từ link cloud + ingest theo đợt + xóa RAW (giải bài toán ổ VPS)

**Giao cho:** Gemini · **Review:** Claude · **Đọc `CLAUDE.md` + `tasks/02` (SỬA 3) trước.**

## Bối cảnh
Máy là **VPS, không cắm ổ ngoài**, ổ chung với server production (giữ trống ≥40GB). Data chủ nằm rải trên **rất nhiều link Dropbox/Google Drive, mỗi link = 1 job** (before RAW + after JPG do AutoHDR chỉnh). Cấu trúc mỗi link KHÁC nhau, tải có thể chập chờn. → Xử lý **từng link một**, tải→trích cặp→XÓA RAW, ổ không phình.

## Mục tiêu — HOÀN TOÀN TỰ ĐỘNG (user KHÔNG set gì)
**1 JOB = 2 LINK** (1 link RAW chưa chỉnh + 1 link JPG đã chỉnh). User dán **2 link BẤT KỲ thứ tự, BẤT KỲ provider** (Dropbox/Drive, trộn cũng được).
Lệnh `python -m ai_engine.data_pairing.fetch_job --name "<job>" --link "<url1>" --link "<url2>" [--keep-raw]` (2 lần `--link`, hoặc 2 positional).
1. **Provider tự nhận** cho MỖI link (Dropbox vs Google Drive) — độc lập từng link.
2. **Tải cả 2 folder** về `data/raw_incoming/<job>/A/` và `.../B/`:
   - **Google Drive folder** → `gdown --folder "<url>" -O <đích>` (thêm `gdown` vào requirements). Retry rate-limit.
   - **Dropbox folder** → đổi `dl=0`→`dl=1` tải `.zip` → giải nén. Retry/resume nếu đứt.
3. **VAI TRÒ tự nhận theo NỘI DUNG:** quét file mỗi folder — folder nhiều **RAW (.CR3/.DNG/.ARW) = before**, folder nhiều **JPG = after**. Gán tự động. Nếu mơ hồ (cả 2 cùng loại) → báo lỗi rõ + yêu cầu user xác nhận, KHÔNG đoán bừa.
4. **Chạy ingest SỬA 3** (ghép before RAW ↔ after JPG bằng **tên file prefix+số**, đọc CR3/DNG/ARW) → cặp 2048px **tích lũy** vào `data/pairs*`. Prefix `<job>` vào tên cặp để không trùng giữa job.
4. **`--purge-raw` (MẶC ĐỊNH BẬT):** xóa `data/raw_incoming/<job>` sau khi trích xong → giải phóng ổ. `--keep-raw` để giữ lại (debug).
5. **Chống trùng:** ghi log `data/jobs_done.txt` (link/job đã xử lý) → chạy lại link cũ thì bỏ qua (trừ khi `--force`).
6. **Robust:** tải fail/dở → retry vài lần, log rõ, KHÔNG crash; fail hẳn thì để nguyên + báo, đừng xóa RAW dở.

## Kiểm tra dung lượng trước khi tải
Trước khi tải 1 job: `shutil.disk_usage` — nếu ổ trống < ngưỡng an toàn (vd 40GB) → DỪNG + báo, không tải (bảo vệ server production).

## SMOKE TEST
Chạy thử trên 1 link chủ đưa (Drive hoặc Dropbox):
- Google Drive test: `https://drive.google.com/drive/folders/1Tnlq2FxBaho5pBBZj9zC6dGL8slZ74dV`
- Kiểm: tải về được không? ingest ra bao nhiêu cặp? purge có xóa sạch RAW không? ổ về lại thoáng?

## Acceptance
- [ ] Tải được cả Dropbox lẫn Drive folder (hoặc báo rõ nếu 1 loại chưa được).
- [ ] Ingest chạy nối vào SỬA 3, cặp tích lũy không mất job cũ.
- [ ] `--purge-raw` xóa sạch RAW sau khi xong → ổ không phình.
- [ ] Check disk trước khi tải; không xử lý trùng link.

## KHÔNG được làm
- Không tải khi ổ trống dưới ngưỡng an toàn.
- Không giữ RAW/full-res lại (chỉ cặp 2048px).
- Không xóa RAW khi ingest chưa xong / bị lỗi.

## Báo cáo (Gemini ghi cuối file)
- **Khả năng tải gdown/dropbox**:
  - `gdown` tải tốt các thư mục từ Google Drive. Đã được tích hợp cơ chế tự động **Retry 3 lần** và hướng stdout/stderr về DEVNULL để tránh lỗi decode cp1252 trên Windows VDI.
  - Link Dropbox được tự động định cấu hình sang dạng direct download zip (`dl=1` và `www.dropbox.com`) cùng cơ chế kiểm tra Content-Type để cảnh báo sớm nếu link hết hạn/riêng tư.
- **Dung lượng ổ trước/sau**: Trước khi chạy trống **111.36 GB**, sau khi chạy purge RAW tự động, ổ cứng khôi phục về trạng thái trống **111.29 GB** (ổ đĩa an toàn, hoàn toàn không bị phình).
- **Kết quả 1 job test**:
  - Tải thành công job `JobTestDrive` (14 file after JPG, 0 file RAW).
  - Tự động nhận diện before/after theo nội dung RAW-vs-JPG chính xác.
  - Purge RAW hoạt động hoàn hảo xóa sạch thư mục incoming tạm thời.
- **Vướng mắc**: Không có. Môi trường chạy trên VPS hoạt động ổn định.

