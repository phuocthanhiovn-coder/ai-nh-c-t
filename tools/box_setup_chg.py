"""Setup box CH_G (24/07): kiem tra may, cai deps (nen), upload kit + CH_F,
keo dataset tu Drive. Chay: python -m tools.box_setup_chg"""
import os
import zipfile

from tools.gpu_ssh import run, put

KIT = os.path.join(os.environ.get("TEMP", "."), "code_kit_chg.zip")


def build_kit():
    with zipfile.ZipFile(KIT, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk("ai_engine"):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for f in files:
                p = os.path.join(root, f)
                z.write(p, p.replace(os.sep, "/"))
        z.write("tools/launch_chg.py", "tools/launch_chg.py")
        z.write("tools/render_compare.py", "tools/render_compare.py")
        z.writestr("tools/__init__.py", "")
    print("kit:", os.path.getsize(KIT) // 1024, "KB")


def main():
    rc, out, err = run(
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; "
        "df -h / | tail -1; free -g | head -2 | tail -1; nproc", timeout=60)
    print("== BOX ==\n" + out)

    run("mkdir -p /root/autohdr/checkpoints/gpu /root/autohdr/checkpoints/sweep "
        "/root/.config/rclone", timeout=30)
    rc, out, err = run(
        "nohup sh -c 'pip install torch torchvision --index-url "
        "https://download.pytorch.org/whl/cu121 && pip install "
        "opencv-python-headless numpy pillow' > /root/pip.log 2>&1 & echo pip_started",
        timeout=30)
    print(out.strip())

    build_kit()
    put(KIT, "/root/autohdr/code_kit.zip")
    put(r"checkpoints\gpu\CH_F.pt", "/root/autohdr/checkpoints/gpu/CH_F.pt")
    put(r"C:\Users\Administrator\AppData\Roaming\rclone\rclone.conf",
        "/root/.config/rclone/rclone.conf")
    rc, out, err = run(
        "cd /root/autohdr && apt-get install -y unzip > /dev/null 2>&1; "
        "unzip -oq code_kit.zip && ls ai_engine | head -3 && "
        "(curl -s https://rclone.org/install.sh | bash > /dev/null 2>&1); "
        "rclone copy gdrive:autohdr_kit/dataset_v5.zip . --transfers 8 && "
        "rm -rf data && mkdir data && cd data && unzip -q ../dataset_v5.zip && "
        "echo pairs: $(ls before | wc -l)", timeout=1200)
    print(out[-600:])
    if err.strip():
        print("STDERR:", err[-300:])


if __name__ == "__main__":
    main()
