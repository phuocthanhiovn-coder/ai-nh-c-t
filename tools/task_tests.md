# NHIỆM VỤ: Bộ test tự động (pytest) cho các module mới

Làm chắc các module vừa xây bằng test tự động, để sau này sửa không sợ hỏng ngầm.

## FILE MỚI (chỉ tạo mới, KHÔNG sửa module nào)
`tests/test_pipeline.py`  (tạo cả `tests/__init__.py` rỗng nếu cần)

## PHẠM VI TEST (dùng pytest, chỉ cv2/numpy, KHÔNG cần torch trừ khi test tới model)
Viết các test NHANH, không cần GPU, không tải model nặng nếu tránh được:

1. `bracket_merge.merge_brackets`:
   - Tạo 2-3 ảnh numpy nhỏ (vd 64x64) khác độ sáng bằng code -> ghi file tạm -> merge ->
     assert shape đúng, dtype uint8, giá trị trong [0,255].
   - merge với <2 ảnh -> assert raise ValueError.
2. `bracket_merge.group_brackets`:
   - Tạo thư mục tạm 6 ảnh -> group_size=3 -> assert 2 nhóm mỗi nhóm 3.
   - folder không tồn tại -> raise ValueError.
3. `white_balance.auto_wb` + `measure_cast`:
   - Ảnh xám thuần -> auto_wb giữ gần như trung tính (|a|,|b| nhỏ).
   - Ảnh ám 1 kênh -> sau auto_wb, |a| hoặc |b| GIẢM so với trước.
4. `straighten.estimate_tilt` + `straighten`:
   - Tạo ảnh có đường kẻ dọc rõ (numpy), xoay 5° -> estimate_tilt ra ~ -5° (sai số ±1.5°).
   - Ảnh trơn không có đường -> straighten trả ảnh cùng shape, không crash.
5. `finish_grade.grade` / `grade_auto` (import từ gpu.finish_grade):
   - Ảnh bất kỳ -> grade_auto trả cùng shape, dtype uint8, [0,255].
   - `_scene` phân loại: ảnh nửa trên xanh dương (B>R) -> "outdoor"; ảnh gỗ ấm (R>B) -> "indoor".

Dùng `tmp_path` fixture của pytest cho file tạm. Mỗi test độc lập, chạy nhanh (<vài giây).
KHÔNG test cái cần tải CH_C model (để nhanh), trừ khi bạn bọc `@pytest.mark.skipif` khi thiếu file.

## TỰ TEST (bắt buộc)
Chạy `python -m pytest tests/test_pipeline.py -q` (nếu thiếu pytest: `pip install pytest`).
- Đảm bảo TẤT CẢ test PASS. In số test pass/fail.
- Cuối in "TASK DONE". Nếu có test fail THẬT thì sửa test hoặc báo rõ lỗi, ĐỪNG giả vờ pass.

## RÀNG BUỘC
- Chỉ tạo file test mới, KHÔNG sửa module. Import đúng đường dẫn package `ai_engine...`.
- Trung thực về kết quả pass/fail.
