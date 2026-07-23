"""Keo anh GOC (before) tuong ung voi cac anh AI da giao, dat CANH nhau de khach so sanh.
Chay: python -m tools.gpu_pull_originals C_bigcrop
"""
import os
import sys
from tools.gpu_ssh import run, get

REMOTE = "/workspace/autohdr"
LOCAL_ROOT = "C:/Users/Administrator/Desktop/autohdr/delivery"


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "C_bigcrop"
    folder = f"{LOCAL_ROOT}/{ckpt}"
    if not os.path.isdir(folder):
        print(f"[!] khong thay {folder}")
        return

    ai_files = [f for f in os.listdir(folder) if f.endswith("_AI.jpg")]
    stems = [f[:-len("_AI.jpg")] for f in ai_files]
    print(f"[*] {len(stems)} anh AI -> keo {len(stems)} anh goc tuong ung")

    tot = 0; ok = 0
    for stem in stems:
        remote = f"{REMOTE}/data/before/{stem}.jpg"
        local = f"{folder}/{stem}_GOC.jpg"
        if os.path.exists(local):
            ok += 1; continue
        try:
            get(remote, local)
            tot += os.path.getsize(local) / 1024
            ok += 1
        except Exception as e:
            print(f"  FAIL {stem}: {e}")
    print(f"[+] {ok}/{len(stems)} anh goc -> {folder}  (moi cap: <ten>_GOC.jpg canh <ten>_AI.jpg)")
    # dem tong folder
    allf = [f for f in os.listdir(folder) if f.endswith(".jpg")]
    total_mb = sum(os.path.getsize(os.path.join(folder, f)) for f in allf) / (1024*1024)
    print(f"[+] Folder hien co {len(allf)} anh, {total_mb:.0f} MB")


if __name__ == "__main__":
    main()
