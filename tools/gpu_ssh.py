"""Helper SSH/SFTP toi box GPU thue. Dung: python -m tools.gpu_ssh <cmd|put|get> ..."""
import sys
import paramiko

# Thong tin box GPU thue doc tu tools/box_creds.json (KHONG commit — nam trong
# .gitignore; moi lan thue box moi chi sua file do, khong sua code).
import json as _json
import os as _os

_CREDS_FILE = _os.path.join(_os.path.dirname(__file__), "box_creds.json")
try:
    with open(_CREDS_FILE, encoding="utf-8") as _f:
        _c = _json.load(_f)
    HOST, PORT, USER, PWD = _c["host"], int(_c["port"]), _c["user"], _c["password"]
except FileNotFoundError:
    HOST = _os.environ.get("GPU_BOX_HOST", "")
    PORT = int(_os.environ.get("GPU_BOX_PORT", "22"))
    USER = _os.environ.get("GPU_BOX_USER", "root")
    PWD = _os.environ.get("GPU_BOX_PWD", "")


def client():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30, banner_timeout=30)
    return c


def run(cmd, timeout=None):
    c = client()
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout, get_pty=False)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    rc = stdout.channel.recv_exit_status()
    c.close()
    return rc, out, err


def put(local, remote):
    import os, time
    c = client()
    sf = c.open_sftp()
    size = os.path.getsize(local)
    sent = 0
    t0 = time.time()
    with open(local, "rb") as f, sf.open(remote, "wb") as r:
        r.set_pipelined(True)          # async writes -> fast over latency
        while True:
            data = f.read(1 << 20)     # 1 MB chunks
            if not data:
                break
            r.write(data)
            sent += len(data)
            if sent % (64 << 20) < (1 << 20):   # log ~every 64MB
                mb = sent / 1e6
                spd = mb / max(time.time() - t0, 1e-6)
                print(f"  {mb:.0f}/{size/1e6:.0f} MB  {spd:.1f} MB/s", flush=True)
    sf.close(); c.close()


def get(remote, local):
    c = client()
    sf = c.open_sftp()
    sf.get(remote, local)
    sf.close(); c.close()


if __name__ == "__main__":
    op = sys.argv[1]
    if op == "cmd":
        rc, out, err = run(sys.argv[2], timeout=None)
        print(out)
        if err.strip():
            print("STDERR:", err)
        print("EXIT", rc)
    elif op == "put":
        put(sys.argv[2], sys.argv[3]); print("PUT OK")
    elif op == "get":
        get(sys.argv[2], sys.argv[3]); print("GET OK")
