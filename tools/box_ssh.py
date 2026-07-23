"""Generic SSH/SFTP helper for an arbitrary box.
Usage: python -m tools.box_ssh <host> <port> <user> <pwd> <cmd|put|get> [args]
"""
import sys
import paramiko


def client(host, port, user, pwd):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, port=int(port), username=user, password=pwd,
              timeout=30, banner_timeout=30)
    return c


def main():
    host, port, user, pwd, op = sys.argv[1:6]
    c = client(host, port, user, pwd)
    if op == "cmd":
        _, o, e = c.exec_command(sys.argv[6], timeout=None)
        print(o.read().decode("utf-8", "replace"))
        er = e.read().decode("utf-8", "replace")
        if er.strip():
            print("STDERR:", er[:2000])
    elif op == "put":
        sf = c.open_sftp()
        with open(sys.argv[6], "rb") as f, sf.open(sys.argv[7], "wb") as r:
            r.set_pipelined(True)
            while True:
                d = f.read(1 << 20)
                if not d:
                    break
                r.write(d)
        sf.close()
        print("PUT OK")
    elif op == "get":
        sf = c.open_sftp()
        sf.get(sys.argv[6], sys.argv[7])
        sf.close()
        print("GET OK")
    c.close()


if __name__ == "__main__":
    main()
