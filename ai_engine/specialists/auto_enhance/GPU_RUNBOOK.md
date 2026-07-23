# GPU_RUNBOOK.md — thuê GPU, train auto_enhance, tải checkpoint về

Mục tiêu: khi thuê máy vast.ai/RunPod xong, mọi giờ GPU đi vào **train**, không vào setup/debug.
**LƯU Ý TRUNG THỰC:** toàn bộ runbook này viết dựa trên đọc code + test CPU thật. Đường CUDA
(`--device cuda`, driver, `torch+cu121`) **CHƯA được chạy thử trên máy nào** — máy dev hiện tại
không có GPU (`torch.cuda.is_available() == False`). Chạy xong bước 1 trên box thật rồi báo lại
nếu có gì khác so với runbook.

## 0. Trước khi thuê máy
- Chạy `python -m ai_engine.specialists.auto_enhance.pack_dataset` trên máy dev → được
  `outputs/dataset_v<N>.zip` (chứa `before/`, `after/`, `manifest.json`).
- Kiểm tra size in ra < 2.5GB (với ~50-300 cặp ảnh RAW->JPEG hiện tại, thực tế ~60-400MB).

## 1. Chọn máy & khởi tạo
- vast.ai hoặc RunPod, image Ubuntu 22.04 + CUDA 12.1, GPU RTX 3090 (24GB) hoặc 4090 (24GB).
- Mở terminal (web hoặc SSH), chạy trong `tmux` NGAY TỪ ĐẦU để sống sót qua mất kết nối:
  ```bash
  tmux new -s train
  ```

## 2. Cài đặt (copy-paste)
```bash
apt-get update && apt-get install -y python3-pip unzip
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
pip install opencv-python-headless numpy
```
- `torch cu121` là build phù hợp CUDA 12.1 (driver mới trên vast.ai/RunPod thường >= 12.1, kiểm tra
  bằng `nvidia-smi` trước khi cài nếu driver cũ hơn thì đổi sang cu118).
- KHÔNG cần `matplotlib`/`tensorboard` — logger chỉ ghi CSV.

## 3. Upload dataset
- Từ máy dev (scp) hoặc kéo trực tiếp link tải sẵn có (Dropbox/Drive) vào box:
  ```bash
  scp outputs/dataset_v1.zip root@<ip>:-p <port> ~/dataset_v1.zip
  ```
  (thay `<ip>`/`<port>` theo thông tin vast.ai/RunPod cấp; hoặc dùng nút "Upload" trên web UI của
  RunPod nếu scp không tiện.)
- Trên box:
  ```bash
  mkdir -p ~/autohdr_data
  unzip ~/dataset_v1.zip -d ~/autohdr_data
  cat ~/autohdr_data/manifest.json   # đối chiếu pair_count / names_hash_sha256 với bản gốc
  ```
- Copy code project lên box (git clone repo riêng hoặc scp toàn bộ `ai_engine/` — dự án hiện KHÔNG
  phải git repo, nên dùng `scp -r ai_engine requirements.txt root@<ip>:~/autohdr/`).

## 4. Chạy training
Từ thư mục gốc project trên box (thư mục cha của `ai_engine/`):

**RTX 3090 (24GB) — khuyến nghị:**
```bash
python -m ai_engine.specialists.auto_enhance.train_gpu \
  --data-dir ~/autohdr_data \
  --epochs 300 --batch-size 8 --crop 1024 --lr 3e-4 --val-frac 0.12 \
  --out checkpoints/auto_enhance_v2.pt --device cuda
```

**RTX 4090 (24GB, nhanh hơn ~1.5-2x) — có thể tăng batch:**
```bash
python -m ai_engine.specialists.auto_enhance.train_gpu \
  --data-dir ~/autohdr_data \
  --epochs 300 --batch-size 12 --crop 1024 --lr 3e-4 --val-frac 0.12 \
  --out checkpoints/auto_enhance_v2.pt --device cuda
```
- `--device auto` cũng tự chọn cuda nếu có, nhưng ghi rõ `--device cuda` cho chắc (nếu driver lỗi,
  script sẽ báo lỗi CUDA ngay thay vì âm thầm rơi về CPU).
- Muốn giảm nguy cơ mất checkpoint khi rớt kết nối: thêm `--save-every 1` (mặc định đã là 1).

### Ước tính thời gian (NGOẠI SUY từ số đo CPU thật, CHƯA đo GPU thật)
- Đo thật trên máy dev (CPU, 2 luồng, `cv2.setNumThreads(2)`/`torch.set_num_threads(2)`):
  ~40s/epoch với crop 512 batch 1 (53 ảnh train); ~11-12s/epoch với crop 256 batch 4 (53 ảnh train).
- Ngoại suy thô cho crop 1024 batch 8 trên RTX 3090/4090 (dữ liệu ~150-300 cặp): forward/backward
  trên GPU với conv nhỏ (kiến trúc HDRNet predictor cực nhẹ, phần nặng nhất là I/O ảnh full-res +
  guidance map full-res) — kỳ vọng **vài giây/epoch** với dataset 150-300 cặp, tức 300 epoch
  **~15-40 phút** trên 3090, nhanh hơn trên 4090. **ĐÂY LÀ ƯỚC LƯỢNG, KHÔNG PHẢI SỐ ĐO THẬT** — CPU
  bottleneck ở decode JPEG/cv2.resize full-res có thể chiếm tỷ trọng lớn hơn dự kiến vì mô hình rất
  nhỏ; nếu chậm hơn ước tính này nhiều, nghi ngờ đầu tiên là I/O ảnh chứ không phải GPU.
- Việc đầu tiên khi lên box thật: chạy `--epochs 2` trước để đo giây/epoch thật, rồi nhân ra cho
  300 epoch trước khi yên tâm rời máy.

## 5. Theo dõi & phục hồi sau mất kết nối
- `tmux attach -t train` để xem lại tiến trình nếu SSH rớt (đây là lý do bước 1 bắt buộc mở tmux).
- Theo dõi loss: `tail -f outputs/train_gpu_log.csv` (cột: epoch, train_l1, val_l1, lr, seconds).
- Nếu máy bị kill/restart giữa chừng, resume bằng đúng checkpoint cuối (không phải best):
  ```bash
  python -m ai_engine.specialists.auto_enhance.train_gpu \
    --data-dir ~/autohdr_data --resume checkpoints/auto_enhance_v2.pt \
    --epochs 300 --batch-size 8 --crop 1024 --out checkpoints/auto_enhance_v2.pt --device cuda
  ```
  - `--resume` đọc trọng số từ file chỉ định + file sidecar `<file>.meta` (chứa optimizer/epoch/
    best_val) nằm cùng thư mục — nhớ tải/copy CẢ HAI file nếu di chuyển checkpoint thủ công.
  - Nếu thiếu file `.meta`, script vẫn resume được trọng số nhưng optimizer/epoch reset về 0 (sẽ in
    cảnh báo rõ ràng).

## 6. Tải kết quả về
- File cần tải về máy dev: **checkpoint best-val**, tên có hậu tố `_best`, ví dụ
  `checkpoints/auto_enhance_v2_best.pt` (không phải file "cuối" `auto_enhance_v2.pt`, vì file cuối
  có thể là epoch overfit muộn hơn best-val).
  ```bash
  scp -P <port> root@<ip>:~/autohdr/checkpoints/auto_enhance_v2_best.pt ./checkpoints/
  ```
- Cũng nên tải `outputs/train_gpu_log.csv` về để xem lại đường cong loss.

## 7. Kiểm chứng lại trên máy dev (BẮT BUỘC trước khi coi là "checkpoint thật")
```bash
python -m ai_engine.specialists.auto_enhance.infer --verify
```
- Sửa tạm biến `checkpoint_path` trong lệnh trên trỏ đúng file mới TẢI VỀ nếu khác
  `checkpoints/auto_enhance.pt` (infer.py hiện hardcode path này — xem ghi chú dưới), hoặc đơn giản
  nhất: copy đè `checkpoints/auto_enhance_v2_best.pt` vào đúng `checkpoints/auto_enhance.pt`
  **SAU KHI đã backup bản pilot cũ** rồi mới chạy `--verify`.
- Sau đó chạy so sánh mắt: dùng `infer.py --input <1 ảnh trong data/pairs/before> --output
  outputs/gpu_ckpt_compare.jpg` rồi mở ảnh so sánh bằng mắt với `after` tương ứng — theo bài học
  "worker bịa report" trong `CLAUDE.md`, KHÔNG báo xong chỉ dựa vào số loss, phải NHÌN ảnh thật.
- Vì `train_gpu.py` lưu checkpoint dạng plain `state_dict` giống hệt format `train.py` hiện dùng,
  bước load này chắc chắn tương thích — đã kiểm chứng ngay trên máy dev (xem báo cáo cuối
  `tasks/20-gpu-training-kit.md`).
