"""Deploy code + dataset len box GPU roi giai nen + verify. Chay 1 lan truoc khi train."""
import os
import time
from tools.gpu_ssh import run, put

BASE = "C:/Users/Administrator/Desktop/autohdr"
REMOTE = "/workspace/autohdr"
PKG = "ai_engine/specialists/auto_enhance/gpu"

# 1) Tao cay thu muc + __init__.py rong (tranh __init__ that keo theo import nang)
INIT_DIRS = [
    "ai_engine",
    "ai_engine/specialists",
    "ai_engine/specialists/auto_enhance",
    "ai_engine/specialists/auto_enhance/gpu",
]
mk = "set -e\n"
for d in INIT_DIRS:
    mk += f'mkdir -p {REMOTE}/{d}\n: > {REMOTE}/{d}/__init__.py\n'
mk += f'mkdir -p {REMOTE}/checkpoints/sweep {REMOTE}/outputs/sweep {REMOTE}/data\n'
mk += 'echo TREE_OK'
rc, out, err = run(mk, timeout=60)
print("[tree]", out.strip(), err.strip()[:200])

# 2) Upload 3 file code
for f in ("model_v2.py", "losses.py", "train_sweep.py"):
    lp = f"{BASE}/{PKG}/{f}"
    rp = f"{REMOTE}/{PKG}/{f}"
    put(lp, rp)
    print(f"[code] {f} -> {rp}")

# 3) Upload dataset zip (1.9GB) - do thoi gian
zip_local = f"{BASE}/outputs/dataset_v4.zip"
zip_remote = f"{REMOTE}/dataset_v4.zip"
sz = os.path.getsize(zip_local) / 1e6
print(f"[data] uploading dataset_v4.zip ({sz:.0f} MB) ... co the vai phut")
t0 = time.time()
put(zip_local, zip_remote)
print(f"[data] upload xong trong {time.time()-t0:.0f}s")

# 4) Giai nen + dem cap
unz = f'''
cd {REMOTE}
/opt/conda/bin/python - <<'PY'
import zipfile, os
z = zipfile.ZipFile("dataset_v4.zip")
z.extractall("data")
z.close()
bd = "data/pairs/before"; ad = "data/pairs/after"
import os
b = set(os.listdir(bd)) if os.path.isdir(bd) else set()
a = set(os.listdir(ad)) if os.path.isdir(ad) else set()
common = b & a
print("before", len(b), "after", len(a), "matched-pairs", len(common))
PY
rm -f dataset_v4.zip
df -h {REMOTE} | tail -1
'''
rc, out, err = run(unz, timeout=600)
print("[unzip]", out.strip())
if err.strip():
    print("[unzip STDERR]", err.strip()[:400])
