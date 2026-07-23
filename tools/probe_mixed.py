import os
import sys
sys.path.insert(0, "C:/Users/Administrator/Desktop/autohdr")
from ai_engine.data_pairing.fetch_job import download_dropbox_folder

URL = "https://www.dropbox.com/scl/fo/xneu87zm8hz0jquv9ztuo/AKKd_YsN-TlxvAVpa3OyCYE?rlkey=1uirczrnc33jdhvzow5d8l8jk&dl=0"
DEST = "data/newbatch/mixed_probe"

import shutil
free = shutil.disk_usage(".").free / (1024**3)
print(f"disk free: {free:.1f} GB")
if free < 40:
    print("STOP: disk under 40GB"); sys.exit(1)

ok = download_dropbox_folder(URL, DEST)
print("download ok:", ok)

# inspect structure
from collections import Counter
ext_count = Counter()
dir_count = Counter()
samples = {}
for root, dirs, files in os.walk(DEST):
    rel = os.path.relpath(root, DEST)
    for f in files:
        e = os.path.splitext(f)[1].lower()
        ext_count[e] += 1
        dir_count[rel] += 1
        if e not in samples:
            samples[e] = f

print("\n=== EXTENSIONS ===")
for e, c in ext_count.most_common():
    print(f"  {e or '(none)'}: {c}   e.g. {samples.get(e,'')}")
print("\n=== TOP DIRS (rel) ===")
for d, c in dir_count.most_common(15):
    print(f"  {d}: {c} files")
print("\n=== total files:", sum(ext_count.values()))
