# Task 20 — GPU TRAINING KIT (prepare everything BEFORE the GPU is rented — zero wasted GPU-hours)

**Assigned to:** Worker G2 (Sonnet on Max) · **Review:** Claude architect · **Read `CLAUDE.md` first.**

## Context
The owner will rent a GPU (vast.ai/RunPod, RTX 3090/4090, a few hours) as soon as the dataset reaches ~150-300 clean pairs. Everything must be ready so the rented hours go 100% into training, not setup/debugging. Current trainer (`ai_engine/specialists/auto_enhance/train.py`) is a CPU pilot: no val split, no augmentation, no LR schedule, no resume — NOT good enough for the real run.

## Files (`ai_engine/specialists/auto_enhance/`) — extend, don't break existing CLI
1. `train_gpu.py` (new, standalone — do NOT modify train.py):
   - Args: `--data-dir data/pairs`, `--epochs 300`, `--batch-size 4`, `--lr 3e-4`, `--val-frac 0.12`, `--resume <ckpt>`, `--out checkpoints/auto_enhance_v2.pt`, `--device auto` (cuda if available else cpu), `--crop 1024` (train on random crops of the full-res pair for batching — proxy is built FROM the crop).
   - Deterministic train/val split by filename hash (stable across runs). Report both losses every epoch.
   - Augmentation: random horizontal flip + random crop (crop is enough; no color aug — color IS the label).
   - AdamW + cosine LR schedule with warmup; grad clip 1.0.
   - Loss: L1 (as now) + optional `--charbonnier`. Keep it simple, NO perceptual nets (no VGG downloads on a rented box).
   - Checkpointing: save best-val + last every N epochs; safe atomic write (tmp then rename); `--resume` restores optimizer + epoch.
   - Logging: append CSV `outputs/train_gpu_log.csv` (epoch, train_l1, val_l1, lr, seconds) — no tensorboard dependency.
   - MUST run on CPU too (tiny run) — that is how you test it here.
2. `pack_dataset.py` (new): zip `data/pairs/{before,after}` into `outputs/dataset_v<N>.zip` with a manifest.json (pair count, names hash, created date). Print final size — must stay <2.5GB or warn.
3. `GPU_RUNBOOK.md` (new, in the same folder): exact copy-paste steps for a fresh vast.ai/RunPod Ubuntu box: pip installs (torch cu121, opencv-headless, numpy), scp/upload dataset zip, unzip, run command with recommended flags for a 3090 (batch 8, crop 1024) and 4090, expected wall-time estimate, how to resume after disconnect (tmux + --resume), what file to download back (best-val checkpoint), and how to verify it locally after download (`infer.py --verify` + smoke compare).

## Acceptance (run for real on THIS machine, CPU)
- [ ] `python -m ai_engine.specialists.auto_enhance.train_gpu --epochs 2 --batch-size 1 --crop 512 --val-frac 0.15 --out checkpoints/kit_smoke.pt` completes on CPU: prints train+val loss per epoch, writes checkpoint + CSV log, resume works (`--resume checkpoints/kit_smoke.pt --epochs 3` continues from epoch 3).
- [ ] Val split is stable: run twice, same files in val (print them).
- [ ] `pack_dataset.py` produces the zip + manifest; unzipping it into a temp dir and pointing `--data-dir` there trains fine (prove with 1 epoch).
- [ ] Model architecture is IDENTICAL to `model.py` HDRNet (the checkpoint must load in existing `infer.py`) — prove it: load `kit_smoke.pt` with infer.py and run one image.
- [ ] Honest report at end of this file: real numbers, wall-time/epoch on CPU, extrapolated GPU estimate, anything untested (e.g., actual CUDA path — be explicit that it's untested here).

## DO NOT
- No files outside `ai_engine/specialists/auto_enhance/` + `outputs/` + report here. Do NOT touch existing train.py/infer.py/model.py/dataset.py behavior. `data/` read-only. `cv2.setNumThreads(2)`, torch threads 2. Delete `checkpoints/kit_smoke.pt` when done (keep auto_enhance.pt intact).

---

## BÁO CÁO (Worker G2, 2026-07-14)

### Thiết kế / lựa chọn khi spec mơ hồ
- **Batching qua crop:** dataset gốc (`dataset.py`) trả ảnh full-res KHÁC kích thước nhau giữa các
  cặp → không stack được batch > 1. `train_gpu.py` viết dataset crop riêng (`CropHDRDataset`, không
  đụng `dataset.py`): lấy cửa sổ `--crop x --crop` (random khi train, center khi val) từ cặp
  before/after pixel-aligned, build proxy 256x256 TỪ crop đó — đúng như spec yêu cầu ("proxy is
  built FROM the crop"). Đã đo thực tế: batch 4 (crop 256) nhanh hơn hẳn batch 1 (crop 512) —
  11-12s/epoch so với ~40s/epoch trên 53 ảnh train, CPU.
- **Checkpoint tương thích infer.py mà vẫn resume được optimizer/epoch:** `--out` (và best-val cùng
  hậu tố `_best`) lưu **plain `model.state_dict()`** — giống hệt format `train.py` hiện dùng, nên
  `infer.py` load thẳng không cần sửa gì. Optimizer state + epoch counter + best_val nằm ở file
  sidecar riêng `<out>.meta` (torch.save dict). `--resume <ckpt>` đọc trọng số từ `<ckpt>` (plain
  state_dict) + đọc `<ckpt>.meta` để khôi phục optimizer/epoch/best_val; nếu thiếu `.meta` vẫn resume
  được trọng số (chỉ mất optimizer/epoch, có in cảnh báo). Lựa chọn này thỏa đồng thời 2 tiêu chí
  nghiệm thu tưởng như xung đột: "checkpoint load thẳng bằng infer.py" và "resume khôi phục đủ
  optimizer + epoch".
- **CSV log path cố định `outputs/train_gpu_log.csv`** đúng y hệt spec (không tham số hoá theo
  `--out`) — nghĩa là nhiều lần chạy khác nhau append chung 1 file; chấp nhận vì spec ghi rõ đường
  dẫn cụ thể, không phải placeholder.
- **LR schedule:** warmup_epochs = `min(5, max(1, epochs // 10))`, sau đó cosine decay theo
  `--epochs` hiện tại (kể cả khi resume mở rộng tổng epoch — recompute lại theo `--epochs` mới, đơn
  giản, chấp nhận được vì đây chỉ ảnh hưởng edge-case "đổi ý tổng epoch giữa chừng").
- **Manifest.json** ghi bên TRONG zip (không phải file rời cạnh zip) để không bị tách rời khi copy/
  upload — `pack_dataset.py` in ra console để xem nhanh không cần giải nén.

### Kết quả acceptance — chạy thật trên máy này (CPU, Windows, torch 2.13.0+cpu, KHÔNG có CUDA)
- ✅ `train_gpu --epochs 2 --batch-size 1 --crop 512 --val-frac 0.15 --out checkpoints/kit_smoke.pt`:
  chạy xong, in train+val loss mỗi epoch, ghi `checkpoints/kit_smoke.pt` + `.meta` +
  `outputs/train_gpu_log.csv`. Epoch1: train_l1=0.394648 val_l1=0.299516 (lr=3e-4). Epoch2:
  train_l1=0.299412 val_l1=0.299516 (lr→0, đúng vì epochs=2 quá nhỏ nên warmup=1 chiếm hết, epoch2
  rơi vào cuối cosine=0 — hành vi ĐÚNG theo công thức, chỉ là artifact của epoch count quá nhỏ dùng
  cho smoke test; với 300 epoch thật warmup ~30 epoch, không xảy ra tình huống này).
- ✅ Resume: `--resume checkpoints/kit_smoke.pt --epochs 3 ...` in
  `[*] Resumed ... start_epoch=2, best_val=0.299516` rồi CHỈ chạy epoch 3/3 (đúng "tiếp tục từ epoch
  3"), không chạy lại epoch 1-2.
- ✅ Val split ổn định: in ra CÙNG list 6 file
  `['_ML_1421.jpg','_ML_1422.jpg','_ML_1444.jpg','_ML_1507.jpg','_ML_1647.jpg','db01__ML_1633.jpg']`
  ở CẢ 3 lần chạy độc lập (run gốc, resume, và run trên dataset giải nén từ zip ở thư mục khác) —
  đúng tinh thần "hash theo tên file, không phụ thuộc thứ tự/vị trí dữ liệu".
- ✅ `pack_dataset.py`: tạo `outputs/dataset_v1.zip` (59 cặp, 60.86 MB, có `manifest.json` bên
  trong: `pair_count=59`, `names_hash_sha256=98e052e1...`, `created=2026-07-14T12:25:21`). Giải nén
  vào thư mục ngoài project, trỏ `--data-dir` vào đó, train 1 epoch: chạy được, loss/val-split GIỐNG
  HỆT run trên `data/pairs` gốc (train_l1=0.394648, val cùng 6 file) — chứng minh zip mang đủ dữ
  liệu, không sai lệch.
- ✅ Kiến trúc checkpoint giống hệt `model.py` HDRNet: đã load `checkpoints/kit_smoke.pt` bằng
  `model.load_state_dict(torch.load(...))` (đúng lệnh `infer.py` dùng) rồi gọi thẳng
  `infer.process_image()` (hàm thật trong infer.py, không sửa gì) trên 1 ảnh `data/pairs/before` —
  input (1365,2048,3) → output (1365,2048,3) khớp tuyệt đối, grid shape `[1,12,8,16,16]` đúng hợp
  đồng "operator không pixel". Ảnh kết quả lưu tại `outputs/kit_smoke_infer_check.jpg`.
- Đã dọn: xoá toàn bộ `checkpoints/kit_smoke*.pt` (+ các biến thể test batch4/charbonnier/packtest)
  sau khi verify xong; `checkpoints/auto_enhance.pt` (pilot cũ, 4,581,173 bytes) KHÔNG bị đụng tới
  (timestamp không đổi trong suốt phiên).

### THÀNH THẬT về phần chưa kiểm chứng được
- **`--device cuda` / toàn bộ đường CUDA CHƯA ĐƯỢC CHẠY THỬ trên bất kỳ máy nào.** Máy này không có
  GPU (`torch.cuda.is_available() == False`, bản torch cài là `2.13.0+cpu`). Code viết theo đúng API
  chuẩn PyTorch (`.to(device)`, `torch.cuda.is_available()`) nên về lý thuyết chạy được, nhưng CHƯA
  có bằng chứng thực nghiệm. `GPU_RUNBOOK.md` đã ghi rõ khuyến nghị: chạy `--epochs 2` đầu tiên trên
  box thật để đo giây/epoch thật trước khi tin ước tính.
- Số liệu ngoại suy sang GPU trong `GPU_RUNBOOK.md` (vài giây/epoch, 300 epoch trong 15-40 phút) LÀ
  ƯỚC LƯỢNG THÔ dựa trên việc kiến trúc HDRNet predictor rất nhẹ — CHƯA đo thật, có thể sai nếu I/O
  ảnh full-res (decode JPEG, cv2.resize guidance map) là bottleneck thay vì compute.
  `pip install torch cu121` trong runbook cũng chưa test cài thật trên box Ubuntu nào.
- `--num-workers` mặc định 0 (đơn luồng) để an toàn với ràng buộc "1 process, threads=2" trên máy
  production hiện tại; trên GPU box có thể tăng nếu cần, chưa đo tác động.

### Số liệu thật (CPU, máy dev này)
- **Giây/epoch (CPU, 53 ảnh train, threads=2):** crop 512 batch 1 ≈ 40s/epoch; crop 256 batch 4 ≈
  11.5s/epoch.
- **Checkpoint size:** 4,581,173 bytes (≈4.37 MB) — HDRNet state_dict, giống hệt size của
  `auto_enhance.pt` pilot hiện có (cùng kiến trúc).
- **Dataset zip size:** `outputs/dataset_v1.zip` = 60.86 MB cho 59 cặp hiện có trong `data/pairs/`.

TASK20=DONE — CPU: crop512/batch1 ≈40s/epoch, crop256/batch4 ≈11.5s/epoch; checkpoint 4,581,173 bytes (~4.37MB); dataset_v1.zip = 60.86MB / 59 cặp; CUDA path UNTESTED trên máy này (no GPU available).
