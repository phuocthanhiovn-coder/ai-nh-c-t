from tools.gpu_ssh import run, get
import os
os.makedirs(r"outputs\compare_chh", exist_ok=True)
rc,out,err = run("ls /root/autohdr/outputs/compare_chh/", timeout=30)
names = [n for n in out.split() if n.endswith(".jpg")]
for n in names:
    get("/root/autohdr/outputs/compare_chh/"+n, r"outputs\compare_chh" + "\\" + n)
get("/root/autohdr/checkpoints/sweep/CH_H.pt", r"checkpoints\gpu\CH_H.pt")
print("pulled", len(names), "+ CH_H.pt")
