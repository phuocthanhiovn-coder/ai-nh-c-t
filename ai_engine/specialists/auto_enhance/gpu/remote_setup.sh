#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# remote_setup.sh — one-shot ON-BOX setup for the rented V100-32GB trainer.
#
# Runs on the GPU box (Ubuntu, pytorch/pytorch:latest image) AFTER the
# architect has scp'd two things into $AUTOHDR_ROOT (default ~/autohdr):
#   - ai_engine/           (the whole code tree, unchanged)
#   - dataset_v2.zip       (packed by pack_dataset.py; ~143MB; contains
#                           before/  after/  manifest.json  at top level)
#
# What it does (idempotent, safe to re-run):
#   1. unzip dataset_v2.zip -> ./data/pairs/{before,after}
#   2. pip install the 3 libs the pytorch image is missing
#   3. verify CUDA + print the GPU name
#   4. echo the pair count / manifest so you can eyeball it before burning
#      GPU-hours on the wrong dataset.
#
# Usage on the box:
#   cd ~/autohdr && bash ai_engine/specialists/auto_enhance/gpu/remote_setup.sh
#
# Optional overrides:
#   AUTOHDR_ROOT=/workspace/autohdr  ZIP=dataset_v2.zip  bash .../remote_setup.sh
# ---------------------------------------------------------------------------
set -euo pipefail

AUTOHDR_ROOT="${AUTOHDR_ROOT:-$HOME/autohdr}"
ZIP="${ZIP:-dataset_v2.zip}"
DATA_DIR="${DATA_DIR:-data/pairs}"

cd "$AUTOHDR_ROOT"
echo "[*] AUTOHDR_ROOT = $AUTOHDR_ROOT"
echo "[*] pwd          = $(pwd)"

# --- 1. system deps (unzip may be absent in the slim pytorch image) --------
if ! command -v unzip >/dev/null 2>&1; then
  echo "[*] installing unzip via apt-get ..."
  apt-get update -y && apt-get install -y unzip
fi

# --- 2. unzip dataset into ./data/pairs ------------------------------------
if [ ! -f "$ZIP" ]; then
  echo "[!] Khong tim thay $ZIP trong $(pwd). Da scp dataset chua?" >&2
  exit 1
fi
mkdir -p "$DATA_DIR"
echo "[*] unzip $ZIP -> $DATA_DIR"
unzip -o -q "$ZIP" -d "$DATA_DIR"

if [ ! -d "$DATA_DIR/before" ] || [ ! -d "$DATA_DIR/after" ]; then
  echo "[!] $DATA_DIR/{before,after} khong ton tai sau khi giai nen — sai zip?" >&2
  exit 1
fi

# --- 3. python deps the pytorch image is missing ---------------------------
# torch / torchvision ship WITH the pytorch/pytorch image — do NOT reinstall
# (a fresh `pip install torch` could pull a CPU wheel and silently kill CUDA).
echo "[*] pip install opencv-python-headless numpy scikit-image ..."
pip install --no-input opencv-python-headless numpy scikit-image

# --- 4. GPU / CUDA verification --------------------------------------------
echo "[*] CUDA check:"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT available inside container!'; print('    torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0), '| CUDA', torch.version.cuda)"

# --- 5. dataset sanity -----------------------------------------------------
BEFORE_N=$(find "$DATA_DIR/before" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)
AFTER_N=$(find "$DATA_DIR/after"  -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)
echo "[+] before=$BEFORE_N  after=$AFTER_N image files under $DATA_DIR"
if [ -f "$DATA_DIR/manifest.json" ]; then
  echo "[+] manifest.json:"
  cat "$DATA_DIR/manifest.json"
  echo
fi

echo "[+] SETUP DONE. Next: run the 2-epoch timing probe (see REMOTE_GPU_RUNBOOK.md §5)."
