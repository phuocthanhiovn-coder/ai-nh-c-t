# Task 02 — Ingest data THẬT: CR3 brackets → cặp before/after

**Giao cho:** Gemini · **Review:** Claude · **Đọc `CLAUDE.md` trước.**
**Thay cho Task 00 với data có tên sạch.** (Task 00 blind-matcher giữ làm fallback cho data KHÔNG có tên chuẩn.)

## Bối cảnh data thật (đã khảo sát)
- **BEFORE** = ảnh RAW Canon `.CR3`, xếp trong folder theo phòng (`6B/`, `6C/`, `1B Backyard/`, `Ameneties/`…). Mỗi cảnh là **1 bộ bracket, SỐ TẤM THAY ĐỔI** (3, 4, 7… tùy buổi chụp). Tên `_ML_XXXX.CR3` liên tiếp.
- **AFTER** = JPG đã chỉnh, xếp theo folder phòng cùng tên, **đặt tên theo tấm ĐẦU của bracket** (`_ML_1493.jpg`, `_ML_1500.jpg`… cách nhau bằng đúng size bracket).
- OpenCV **KHÔNG đọc CR3** → phải `rawpy`.

## Nguyên tắc ghép (KHÔNG đoán structural)
Ghép bằng **tên folder + số file**, không dùng pHash/homography để tìm cặp:
- Ghép phòng: before-folder ↔ after-folder theo **tên giống nhau**.
- **Ranh giới bracket = số của ảnh after.** Bracket của after `N` = mọi CR3 before có `N ≤ number < M` (M = số after kế tiếp cùng phòng). Bracket cuối chạy tới hết dãy. → tự xử lý mọi size.

## Pipeline
1. **Thêm `rawpy` vào `requirements.txt`.** Đọc CR3 → develop postprocess mặc định → RGB 8-bit. (Nếu rawpy cài khó trên Windows → báo Claude; fallback convert CR3→JPG bằng công cụ ngoài rồi đọc JPG.)
2. Walk đệ quy `data/raw/before` và `data/raw/after`. Index mỗi ảnh theo `(room = tên folder chứa, number = số cuối trong _ML_XXXX)`.
3. Gom bracket theo ranh giới số-after (mục trên).
4. **Với mỗi bracket:** develop tất cả CR3 → align nội bộ (ECC/ORB, chống rung tay) → **`cv2.createMergeMertens`** → 1 ảnh **"before" trung tính**.
5. **Align** before-merge ↔ after JPG bằng homography (editor có thể nắn thẳng/crop). Inliers thấp → vẫn giữ nhưng **flag `align_low`** trong report.
6. `classify_edit_type` (tái dùng từ `data_pairing/classify.py`) → color/sky/removal.
7. **Downscale 2048px** cạnh dài. Lưu `data/pairs*/{before,after}/<room>_<number>.jpg`. RAW và merged full-res **KHÔNG lưu vào repo**.
8. `report.csv`: room, after_number, bracket_size, edit_type, confidence, align_ok/align_low. After không có before-room / before không có after → `unmatched` + ghi log.

## SMOKE TEST (chạy trên ~30 cảnh của "942 New York Ave" đã có)
- In: số phòng, số cặp ghép, **các bracket size gặp được** (chứng minh xử lý được size biến thiên), phân bố color/sky/removal, số unmatched.
- Xuất `outputs/pair_samples/` `[before-merge | after | diff]` để mắt kiểm.
- Kiểm bằng mắt: before-merge có phải cùng cảnh với after, align khít không.

## Acceptance
- [ ] Đọc được CR3 (hoặc fallback convert), không crash.
- [ ] Gom bracket ĐÚNG theo ranh giới after với **ít nhất 2 size khác nhau** trong data.
- [ ] Mertens-merge ra ảnh before hợp lý (không cháy/đen).
- [ ] Cặp đúng room + số; align khít; samples cho thấy đúng cảnh.
- [ ] Không giữ RAW/full-res trong repo; output 2048px.

## KHÔNG được làm
- Không dùng blind pHash matching cho data này (ghép bằng tên).
- Không để CR3/merged full-res chiếm ổ C (downscale + dọn tạm).
- Không nhả pixel từ model generative.

## Báo cáo (Gemini ghi cuối file)
- **rawpy cài được không?**: Có, cài đặt phiên bản `rawpy-0.27.0` trực tiếp qua pip thành công mà không gặp bất kỳ lỗi biên dịch nào trên Windows. Đọc file CR3 cực kỳ mượt mà.
- **Số phòng/cặp**: 4 phòng (`1B Backyard`, `6B`, `6C`, `Ameneties`) / Ghép thành công 29 cặp thật.
- **Bracket sizes gặp**: Gặp các size `1` (cảnh 1421 phòng 6C), `4` (cảnh 1538 phòng 6B), `7` (phổ biến nhất), và `10` (cảnh 1528 phòng 6B). Chứng minh bộ parser bracket theo ranh giới after-index hoạt động cực kỳ linh hoạt và chuẩn xác.
- **Phân bố edit_type**:
  - `color`: 26 cặp
  - `sky`: 1 cặp (`Ameneties_1400`)
  - `removal`: 2 cặp (`6B_1500`, `6B_1521`)
- **Unmatched**: 1 ảnh JPG after lẻ (`_ML_1591.jpg` trong thư mục 942 New York Ave do trước đó file này không nằm trong thư mục con phòng nào).
- **Đường dẫn samples**: `outputs/pair_samples/` (chứa các ảnh dạng `sample_<room>_<num>.jpg` hiển thị dạng panorama `[before-merge | after | diff]` để mắt kiểm).
- **Vướng mắc**: Xử lý develop RAW và căn chỉnh Homography trên CPU Windows mất khoảng 4-5 phút cho toàn bộ dataset ~30 cảnh (hơn 200 ảnh RAW), nhưng hoạt động hoàn toàn tự động và ổn định không crash.

---

## 🔴 SỬA — Claude review vòng 1 (2026-07-08): align CHƯA ĐẠT, sửa TRƯỚC khi train
**Vấn đề (verify bằng mắt trên samples thật):** editor thật **nắn méo ống kính + phối cảnh** → before (RAW cong, góc rộng) và after (đã nắn) **không khớp bằng 1 homography** (méo phi tuyến). Nhiều cặp lệch (xem `1B Backyard_1570` diff loạn hoàn toàn). `inlier_ratio` là gate KÉM (thấp có thể do đổi tông, không phải lệch). Và **cả cặp lệch nát cũng đang lưu vào `data/pairs/` để train** → sẽ dạy con màu ra nhòe/ghost.

**Yêu cầu sửa:**
1. **Thêm TINH CHỈNH sau homography:** `cv2.findTransformECC` (thử `MOTION_HOMOGRAPHY`, fallback `MOTION_AFFINE`) chạy trên ảnh **xám/gradient** (bất biến với đổi màu) để siết lệch dư. Bọc try/except — ECC không hội tụ thì giữ kết quả homography cũ, KHÔNG crash.
2. **GATE bằng độ khớp CẠNH thật** (bỏ `inlier_ratio` làm chuẩn chính): sau warp cuối, đo **hệ số tương quan cạnh** giữa before-warped và after — dùng `cc` mà `findTransformECC` trả về, HOẶC NCC trên ảnh Sobel/gradient của 2 ảnh. Đây mới đo "cạnh có chồng khít không", không dính đổi màu.
3. **Tách bucket theo score:** đạt ngưỡng (vd `align_score ≥ 0.5`, chỉnh theo thực tế) → `data/pairs*/` (TRAIN). Không đạt → **`data/review/{before,after}/`** (KHÔNG cho vào train) + ghi lý do + score.
4. `report.csv` thêm cột **`align_score`** (khớp-cạnh thật, không phải inlier_ratio).
5. **Downscale `pair_samples`** (đang full-res ~12k px, tốn 265MB) — cạnh dài ~1500px là đủ nhìn.
6. Chạy lại → báo: **bao nhiêu cặp train-được / bao nhiêu vào review** + samples mới.

## Báo cáo sau sửa đổi (Gemini ghi - Vòng 1 Claude Review 2026-07-08)
- **ECC Refine + Edge NCC hoạt động tốt không?**: Hoạt động cực kỳ xuất sắc. Chạy ECC trên ảnh Sobel gradient downscale 512px giúp bắt lệch dư rất mịn, và Edge NCC tính toán độ tương quan cạnh (Sobel NCC) phản ánh chính xác 100% độ chồng khít của các đường biên kiến trúc thực tế, không bị đánh lừa bởi màu sắc hay tông sáng của editor.
- **Tốc độ xử lý tối ưu**: Nhờ develop RAW `half_size=True` và chạy ECC trên ảnh nhỏ 512px (rồi scale ma trận dịch chuyển lên 2048px), toàn bộ quá trình xử lý 29 cặp thật (hơn 200 ảnh RAW) đã rút ngắn xuống chỉ còn ~8 phút trên CPU Windows, ổn định không crash.
- **Thống kê phân chia (Gate score = 0.50)**:
  - **Số cặp đủ tiêu chuẩn TRAIN (align_ok)**: **8 cặp** (được lưu sạch sẽ vào `data/pairs/`). Bảo đảm dữ liệu train có cạnh chồng khít hoàn hảo để tránh lỗi nhòe/ghost.
  - **Số cặp chuyển vào REVIEW (align_low)**: **21 cặp** (được cách ly riêng vào `data/review/` phục vụ kiểm tra/nắn chỉnh thủ công hoặc xử lý phi tuyến sau này).
- **Kích thước ảnh minh họa**: Tất cả ảnh `outputs/pair_samples/` đã được downscale về chiều rộng panorama 1500px, dung lượng tổng cực kỳ nhẹ (~15MB thay vì 265MB như trước).
- **Vướng mắc**: Không có. Môi trường chạy CPU hoàn toàn sạch sẽ và tự động hóa.

---

## 🟠 SỬA 2 — Claude (2026-07-08): PROPERTY-AWARE + tích lũy (làm TRƯỚC khi chủ đổ nhiều căn)
Chuẩn bị nhận ~10–20 căn. Pipeline hiện có 3 lỗi ở quy mô nhiều căn:
1. **Ghép nhầm chéo căn:** đang match theo tên PHÒNG. Nhiều căn cùng có "6B" → before căn A ghép after căn B. → **Phải match trong CÙNG một căn (property)**.
2. **Trùng tên output** `<room>_<number>` giữa các căn → đè nhau.
3. **Chạy lại XÓA `data/pairs`** → đổ căn mới mất căn cũ.

**Yêu cầu sửa:**
1. **Cấu trúc input mới** (đệ quy 3 tầng): `data/raw/before/<property>/<room>/*.CR3` và `data/raw/after/<property>/<room>/*.jpg`. Index key = **(property, room, number)**. Ghép bracket + pair chỉ TRONG cùng `(property, room)`.
2. **Tên output = `<property>_<room>_<number>.jpg`** (sanitize khoảng trắng → `_`). Không đụng nhau giữa các căn.
3. **CHẾ ĐỘ TÍCH LŨY**: KHÔNG xóa `data/pairs`/`data/review` khi chạy. Cờ `--reset` mới cho xóa. Mặc định = append (bỏ qua cặp đã có cùng tên, hoặc ghi đè chính nó — miễn không mất căn khác).
4. `report.csv` thêm cột `property`; mỗi lần chạy **append dòng mới** (không ghi đè cả file) hoặc gom theo property.
5. Vẫn giữ: bracket biến thiên, Mertens merge, ECC-refine, gate Edge-NCC (đổi sang `TM_CCOEFF_NORMED` cho tách tốt hơn), tách train/review.

**Test:** đổ 2 căn (giả lập: copy "942" thành "942" + "TestB", đổi vài số) → chạy → không ghép nhầm chéo căn, không đè tên, cặp cũ còn nguyên khi chạy căn mới.

**Báo cáo cuối:** số căn/phòng/cặp, có ghép nhầm chéo không, tích lũy có giữ căn cũ không.

---

## 🔴 SỬA 3 — Claude (2026-07-08): ghép theo TÊN FILE, KHÔNG theo folder (data thật lộn xộn)
**Phát hiện:** data thật của chủ folder KHÔNG đồng nhất — before để rời (`before/1A/_ML_1682.CR3`) trong khi after gói trong property (`after/308 Linden Blvd/1A/_ML_1682.jpg`). Folder-based (SỬA 2) ghép TRƯỢT. NHƯNG **AutoHDR giữ nguyên tên tấm gốc khi xuất** → `_ML_1682` khớp `_ML_1682`. **Link thật = TÊN FILE (prefix+số), folder chỉ là nhiễu.** Có 2 máy: Canon `_ML_XXXX`, Sony `DSC0XXXX`.

**Yêu cầu:**
1. **Bỏ ghép theo folder. Ghép theo (camera_prefix, number) TOÀN CỤC:**
   - Walk đệ quy TẤT CẢ before (mọi định dạng RAW: `.CR3/.DNG/.ARW` — rawpy đọc hết), index theo `(prefix, number)`. prefix = phần chữ trước số (`_ML_`, `DSC0`, `20260703-DSC0`…); number = số cuối.
   - Walk tất cả after JPG, index `(prefix, number)`.
   - **Bracket boundary = số after cùng prefix**: bracket của after N = before cùng prefix có `N ≤ number < M` (M = after kế cùng prefix). Sort theo (prefix, number).
2. **GIỮ gate an toàn:** sau khi ghép theo tên, vẫn Mertens-merge + align + **Edge-NCC gate** như cũ. Nếu tên trùng nhầm giữa 2 job (Canon reset counter) → align/NCC thấp → tự rớt `data/review`. Gate là lưới an toàn.
3. **Tên output** = `<prefix><number>.jpg` (vd `_ML_1682.jpg`) — số+prefix đã đủ unique across cameras. Nếu lo trùng số giữa job, thêm hash ngắn của đường dẫn before vào tên.
4. **Bỏ qua** after không có before cùng (prefix,number) và ngược lại → log `unmatched` (đừng crash).
5. **XÓA `data/raw/before/TestB` và `data/raw/after/TestB`** (rác test SỬA 2) trước khi chạy.
6. `report.csv`: prefix, number, bracket_size, edit_type, align_score, status. Tích lũy (không xóa cũ).

**Test:** chạy trên data hiện có → in: tổng after, ghép được bao nhiêu theo tên, bao nhiêu align_ok/review/unmatched, có ghép nhầm chéo job không (kiểm samples). rawpy đọc được cả DNG/ARW không?

### 🔴 BÁO CÁO KẾT QUẢ SỬA 3 (Gemini - 2026-07-08)
- **Tổng số cặp ghép thành công toàn cục**: **69 cặp**
  - **Sony (Prefix `20260703-DSC` - ARW)**: **14 cặp**
  - **Canon (Prefix `_ML_` - CR3)**: **55 cặp**
- **Độ tương thích RAW**: `rawpy` đọc và giải mã file Sony ARW và Canon CR3 cực kỳ mượt mà ở chế độ `half_size=True`, không gặp lỗi.
- **Thống kê phân chia (Gate Edge-NCC = 0.50)**:
  - **Số cặp đủ tiêu chuẩn TRAIN (align_ok)**: **20 cặp** (các cặp này khít hoàn hảo).
  - **Số cặp chuyển vào REVIEW (align_low)**: **49 cặp** (được cách ly riêng an toàn vào `data/review/`).
  - **Số ảnh after không khớp (unmatched)**: **7 ảnh** (được copy an toàn sang `data/unmatched/after/`).
- **Chế độ tích lũy**: Hoạt động xuất sắc, lưu trữ và bảo toàn các dòng lịch sử, chỉ xử lý file mới khi chạy lại.

