"""
Task 20 - pack data/pairs/{before,after} into a single zip for uploading to
a rented GPU box. Includes manifest.json (pair count, content hash, created
date) so the box can sanity-check it received the right data.
"""
import os
import json
import zipfile
import hashlib
import datetime
import argparse

IMG_EXTS = (".jpg", ".jpeg", ".png")
WARN_BYTES = int(2.5 * 1024 ** 3)


def compute_names_hash(names):
    h = hashlib.sha256()
    for n in sorted(names):
        h.update(n.encode("utf-8"))
    return h.hexdigest()


def next_version(out_dir):
    if not os.path.isdir(out_dir):
        return 1
    versions = []
    for f in os.listdir(out_dir):
        if f.startswith("dataset_v") and f.endswith(".zip"):
            try:
                versions.append(int(f[len("dataset_v"):-len(".zip")]))
            except ValueError:
                pass
    return max(versions, default=0) + 1


def main():
    parser = argparse.ArgumentParser(description="Pack data/pairs into outputs/dataset_vN.zip for GPU upload")
    parser.add_argument("--data-dir", default="data/pairs")
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    before_dir = os.path.join(args.data_dir, "before")
    after_dir = os.path.join(args.data_dir, "after")

    names = sorted(f for f in os.listdir(before_dir) if f.lower().endswith(IMG_EXTS))
    names = [n for n in names if os.path.exists(os.path.join(after_dir, n))]
    if not names:
        print(f"[!] Khong tim thay cap anh nao trong {args.data_dir}")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    version = next_version(args.out_dir)
    zip_path = os.path.join(args.out_dir, f"dataset_v{version}.zip")
    tmp_path = zip_path + ".tmp"

    manifest = {
        "version": version,
        "pair_count": len(names),
        "names_hash_sha256": compute_names_hash(names),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }

    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for n in names:
            zf.write(os.path.join(before_dir, n), arcname=f"before/{n}")
            zf.write(os.path.join(after_dir, n), arcname=f"after/{n}")
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

    os.replace(tmp_path, zip_path)

    size_bytes = os.path.getsize(zip_path)
    size_mb = size_bytes / (1024 ** 2)
    print(f"[+] Packed {len(names)} pairs -> {zip_path}")
    print(f"    - manifest: {manifest}")
    print(f"    - size: {size_mb:.2f} MB")
    if size_bytes > WARN_BYTES:
        print(f"[!] CANH BAO: dataset zip vuot 2.5GB ({size_mb:.2f} MB)!")


if __name__ == "__main__":
    main()
