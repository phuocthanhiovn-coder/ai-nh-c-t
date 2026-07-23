from tools.gpu_ssh import run, get
import os
os.makedirs(r"outputs\compare_chg", exist_ok=True)
rc,out,err = run("ls /root/autohdr/outputs/compare_chg/", timeout=30)
names = [n for n in out.split() if n.endswith(".jpg")]
for n in names:
    get("/root/autohdr/outputs/compare_chg/"+n, r"outputs\compare_chg" + "\\" + n)
get("/root/autohdr/checkpoints/sweep/CH_G.pt", r"checkpoints\gpu\CH_G.pt")
print("pulled", len(names), "panels + CH_G.pt")
