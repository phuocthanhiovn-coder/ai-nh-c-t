# REMOTE_GPU_RUNBOOK.md — V100-32GB rental, exact orchestration

Goal: the architect rents a V100-32GB box for **3 hours**, and every minute
goes into training — not setup, not debugging. This is a copy-paste runbook,
top to bottom.

> **HONESTY BANNER (read this).** Everything below is written from reading the
> code + real CPU smoke tests on the Windows dev box. **The CUDA path is
> UNVERIFIED here** — this machine is `torch 2.13.0+cpu`, `torch.cuda.is_available()
> == False`. AMP autocast/GradScaler in the trainer is guarded by `device=='cuda'`
> but has never executed on a GPU. **Step 5 (the 2-epoch timing probe) is
> mandatory** — do not trust any epoch/time estimate in this file until the probe
> gives you a real sec/epoch on the actual V100. Numbers here are extrapolations.
>
> **Dependency note.** The main run uses the sweep driver
> `ai_engine/specialists/auto_enhance/gpu/sweep.py` (companion deliverable of this
> session). Its CLI is documented in §6 as this runbook assumes it. If that file is
> not present yet, use the **fallback** in §6b (drive `train_gpu.py` directly).

Placeholders used throughout — fill in from the rental dashboard at runtime:
`HOST` = box IP/hostname · `PORT` = SSH port · `USER` = ssh user (usually `root`).

---

## 0. On the Windows dev box — pack the dataset (once, before renting)

From the project root `C:/Users/Administrator/Desktop/autohdr` in PowerShell or
Git Bash:

```bash
python -m ai_engine.specialists.auto_enhance.pack_dataset
```

This writes `outputs/dataset_v<N>.zip` (`before/`, `after/`, `manifest.json`).
This runbook assumes the file is **`outputs/dataset_v2.zip` (~143MB)**. If
pack_dataset emits a different version number, either rename it to
`dataset_v2.zip` or substitute the real name everywhere below.

---

## 1. Rent the box & open the SSH session

- Provider (vast.ai / RunPod / Lambda), image **`pytorch/pytorch:latest`**,
  GPU **V100-32GB**, Ubuntu. Grab `HOST`, `PORT`, `USER` from the dashboard.
- Open the SSH session and **enter tmux immediately** (survives disconnects —
  a dropped SSH must never kill a 3-hour train):

```bash
ssh -p PORT USER@HOST
# once inside:
tmux new -s train
```

Keep this tmux for the training run. Do the uploads (§2) from a **separate**
local terminal on Windows — you don't need to be inside tmux to scp.

---

## 2. Upload from Windows (scp) — code + dataset

Run these on the **Windows dev box**, from `C:/Users/Administrator/Desktop/autohdr`.
`scp` ships with Windows 10/11 (OpenSSH) and Git Bash. Use forward slashes.

First make the target dir on the box (one-liner over ssh):

```bash
ssh -p PORT USER@HOST "mkdir -p ~/autohdr"
```

Upload the **dataset zip** (~143MB) — put it at `~/autohdr/dataset_v2.zip`:

```bash
scp -P PORT outputs/dataset_v2.zip USER@HOST:~/autohdr/dataset_v2.zip
```

Upload the **whole code tree** `ai_engine/` (small — a few MB of .py):

```bash
scp -P PORT -r ai_engine USER@HOST:~/autohdr/ai_engine
```

> Notes:
> - `scp` uses **`-P PORT`** (capital P). `ssh` uses lowercase `-p`. Easy to mix up.
> - `data/pairs` is **inside the zip** — do NOT scp `data/` separately.
> - If `ai_engine/**/__pycache__` bloats the transfer, it's harmless; ignore it.
> - Slow link? `dataset_v2.zip` at ~143MB over ~10 MB/s ≈ 15–20s; over a poor
>   link it can be minutes — start it first, then do the code upload in parallel
>   from a second terminal.

---

## 3. On the box — run the one-shot setup script

Back in the tmux session on the box:

```bash
cd ~/autohdr
bash ai_engine/specialists/auto_enhance/gpu/remote_setup.sh
```

`remote_setup.sh` does all of: `unzip dataset_v2.zip -> ./data/pairs`,
`pip install opencv-python-headless numpy scikit-image` (torch/torchvision are
already in the image — it does **not** touch them), verifies CUDA, and prints
the GPU name + dataset pair count + manifest. It exits non-zero if CUDA is
missing or the dataset didn't unzip — stop and fix before spending GPU-hours.

Expected tail of its output (illustrative):

```
    torch 2.x.x | GPU: Tesla V100-SXM2-32GB | CUDA 12.x
[+] before=136  after=136 image files under data/pairs
[+] manifest.json: { ... "pair_count": 136 ... }
[+] SETUP DONE. ...
```

If you'd rather run the GPU check by hand (matches the script's core assertion):

```bash
python -c "import torch;print(torch.cuda.get_device_name(0))"
```

---

## 4. Budget map (3 hours = 180 min)

| Phase | Target | Notes |
|-------|--------|-------|
| Upload + setup (§2–§3) | ~5–10 min | not GPU-billed if you upload before the box spins, but on most providers the clock is already running — move fast |
| **Timing probe (§5)** | ~3–6 min | 2 epochs on ONE config; the single most important step |
| **Sweep (§6)** | **~150 min** | `--time-budget-min 150` |
| **Final long train (§6)** | **~24 min (0.4h)** | auto-triggered by the sweep after the budget elapses |
| Download + margin (§7) | ~5–10 min | leave slack; don't get killed mid-scp |

The sweep's `--time-budget-min 150` + auto-final (~0.4h) ≈ **2.9h**, leaving
~10min for probe + download. **Do not** set the sweep budget so high that the
final train or the download get guillotined by the rental expiry.

---

## 5. MANDATORY: 2-epoch timing probe (measure real sec/epoch)

Before trusting any time budget, measure **one** config for **2 epochs** on the
real V100 using the existing, verified trainer `train_gpu.py`. This prints
`time=<sec>s` per epoch — that number is ground truth for planning.

```bash
cd ~/autohdr
python -m ai_engine.specialists.auto_enhance.train_gpu \
  --data-dir data/pairs \
  --epochs 2 --batch-size 8 --crop 1024 --lr 3e-4 --val-frac 0.12 \
  --out checkpoints/_probe.pt --device cuda
```

Read the `time=...s` on the `Epoch 0002/0002` line (epoch 1 includes warm
caches/JIT; **use epoch 2**). Then do the arithmetic:

- Let `S` = measured sec/epoch (crop 1024, batch 8, 136 pairs → ~120 train imgs).
- **Sweep budget**: 150 min = 9000s. If the sweep tries `K` configs, each config
  gets ~`9000/K` s → `epochs_per_config ≈ 9000 / (K · S)`.
- **Final train**: 0.4h = 1440s → `final_epochs ≈ 1440 / S`.
- Feed those into the sweep flags (§6): `--epochs-per-config` and `--final-epochs`
  so the whole thing fits **~2.5h sweep + ~0.4h final**.

Reality check on `S`: the HDRNet predictor is tiny; the real cost is JPEG decode +
full-res `cv2.resize`/guidance on the CPU side of the dataloader. If `S` is much
larger than "a few seconds", the bottleneck is **I/O, not the GPU** — bump
`--num-workers` (see §6) rather than assuming the GPU is slow. If `S` comes in
so low that 300 epochs fits comfortably, raise the epoch counts to actually
consume the budget (idle GPU = wasted rental).

Delete the probe checkpoint so it can't be confused with the real run:

```bash
rm -f checkpoints/_probe.pt checkpoints/_probe.pt.meta
```

---

## 6. Launch the sweep (auto-does the final long train)

Inside tmux. The sweep explores grid/proxy/loss configs under a wall-clock
budget, writes a leaderboard, then automatically runs one long train on the
winning config and saves the final checkpoints.

```bash
cd ~/autohdr
python -m ai_engine.specialists.auto_enhance.gpu.sweep \
  --data-dir data/pairs \
  --device cuda \
  --time-budget-min 150 \
  --epochs-per-config <from §5> \
  --final-epochs <from §5> \
  --out checkpoints/auto_enhance_v2 \
  2>&1 | tee outputs/sweep/run.log
```

Expected outputs when it finishes:
- `outputs/sweep/leaderboard.csv` — one row per config (val loss, config knobs, time).
- `checkpoints/auto_enhance_v2_final.pt` — long-trained winner (plain `state_dict`,
  loadable by `infer.py` unchanged).
- `checkpoints/auto_enhance_v2_best.pt` — best-val checkpoint from the final train.
- `.meta` sidecars next to each (optimizer/epoch/best_val) — needed only for resume.

> The `--out` here is a **prefix**: the sweep appends `_final.pt` / `_best.pt`.
> Confirm the actual flag names against `gpu/sweep.py --help` on the box; this
> runbook documents the intended contract, and the sweep is a sibling deliverable.

### 6b. Fallback if `gpu/sweep.py` is not present

Skip the sweep and drive the verified trainer directly. Pick epochs so
`epochs · S ≈ 9000s` (from §5), e.g.:

```bash
cd ~/autohdr
mkdir -p outputs/sweep
python -m ai_engine.specialists.auto_enhance.train_gpu \
  --data-dir data/pairs \
  --epochs <≈9000/S> --batch-size 8 --crop 1024 --lr 3e-4 --val-frac 0.12 \
  --num-workers 4 --charbonnier \
  --out checkpoints/auto_enhance_v2_final.pt --device cuda \
  2>&1 | tee outputs/sweep/run.log
```

This produces `checkpoints/auto_enhance_v2_final.pt` and
`checkpoints/auto_enhance_v2_final_best.pt` (note the `_best` suffix rule in
`train_gpu.py`). Adjust the §7 download names accordingly.

---

## 7. Monitoring (second terminal / second tmux pane)

Split a tmux pane (`Ctrl-b "`) or open a second `ssh -p PORT USER@HOST`, then:

```bash
# leaderboard as configs finish (sweep path):
tail -f outputs/sweep/leaderboard.csv

# full run log either path:
tail -f outputs/sweep/run.log

# per-epoch loss curve (train_gpu.py CSV — used by probe & fallback):
tail -f outputs/train_gpu_log.csv     # cols: epoch,train_l1,val_l1,lr,seconds

# GPU utilisation / memory / temp, refreshed every 5s:
watch -n 5 nvidia-smi
# or continuous:
nvidia-smi -l 5
```

Sanity while it runs: `nvidia-smi` should show the python process holding GPU
memory and **>0% GPU-Util**. If GPU-Util sits near 0% but a python proc is
pinned, you're **dataloader-bound** — the fix is more `--num-workers`, not a
faster GPU.

---

## 8. Download the results back to Windows

Run on the **Windows dev box**, from `C:/Users/Administrator/Desktop/autohdr`.
Pull the final + best checkpoints (and their `.meta` sidecars for resume), plus
the leaderboard/log for the record:

```bash
scp -P PORT USER@HOST:~/autohdr/checkpoints/auto_enhance_v2_final.pt      ./checkpoints/
scp -P PORT USER@HOST:~/autohdr/checkpoints/auto_enhance_v2_best.pt       ./checkpoints/
scp -P PORT USER@HOST:~/autohdr/checkpoints/auto_enhance_v2_final.pt.meta ./checkpoints/
scp -P PORT USER@HOST:~/autohdr/outputs/sweep/leaderboard.csv             ./outputs/sweep/
scp -P PORT USER@HOST:~/autohdr/outputs/train_gpu_log.csv                 ./outputs/
```

(Fallback naming from §6b: the best file is `auto_enhance_v2_final_best.pt`.)
Verify the files arrived and are non-trivial in size before you terminate the
rental — a 0-byte checkpoint means the scp raced the training write.

---

## 9. Sanity check after download (MANDATORY — on Windows)

`infer.py` hardcodes `checkpoint_path = "checkpoints/auto_enhance.pt"`, and this
runbook must **not** modify infer.py. So back up the old pilot, copy the new
final over that path, then run the built-in verifier:

```bash
# from C:/Users/Administrator/Desktop/autohdr
cp checkpoints/auto_enhance.pt checkpoints/auto_enhance_pilot_backup.pt   # if it exists
cp checkpoints/auto_enhance_v2_final.pt checkpoints/auto_enhance.pt

python -m ai_engine.specialists.auto_enhance.infer --verify
```

`--verify` runs the operator-not-pixel smoke test (multi-resolution, asserts
input size == output size). It must end with `[+] TẤT CẢ SMOKE TEST ĐÃ ĐẠT!`.

> **Note (honest):** model architecture. `infer.py --verify` loads the v1
> `HDRNet` from `..model`. If the sweep trained the **v2** model
> (`gpu/model_v2.py`, different grid/tensor shapes — checkpoints are NOT
> interchangeable with v1), `--verify` will fail on `load_state_dict`. In that
> case verify with a v2-aware loader instead (construct `HDRNetV2(**cfg)` from the
> winning config in `leaderboard.csv`, `load_state_dict`, run one image). Confirm
> which architecture the sweep used before assuming `--verify` applies.

Then the **eye test** (the golden rule from `CLAUDE.md` — never trust the loss
number, look at the actual pixels; a worker once faked a report):

```bash
python -m ai_engine.specialists.auto_enhance.infer \
  --input data/pairs/before/<somefile.jpg> \
  --output outputs/gpu_ckpt_compare.jpg
```

Open `outputs/gpu_ckpt_compare.jpg` next to `data/pairs/after/<somefile.jpg>`
and judge by eye. Watch specifically for the **washout/silvering** failure that
sank the pilot checkpoint (per `CLAUDE.md` §8). If it looks washed out, the run
did not succeed regardless of what `val_l1` says — keep the pilot backup and do
not promote the new checkpoint.

---

## 10. Teardown

Only after §9 passes and files are safely on Windows: terminate the rental to
stop billing. Keep `outputs/sweep/leaderboard.csv` + `run.log` — they justify the
config choice for the next run and record real V100 sec/epoch for future budgets.
