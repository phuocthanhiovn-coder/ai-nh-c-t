from tools.gpu_ssh import run, get
import os
os.makedirs(r"outputs\compare_chf", exist_ok=True)
rc,out,err = run("ls /root/autohdr/outputs/compare_chf/", timeout=30)
names = [n for n in out.split() if n.endswith(".jpg")]
for n in names:
    get("/root/autohdr/outputs/compare_chf/"+n, r"outputs\compare_chf" + "\\" + n)
    print("pulled", n)
get("/root/autohdr/checkpoints/sweep/CH_F.pt", r"checkpoints\gpu\CH_F.pt")
get("/root/autohdr/outputs/sweep/CH_F.csv", r"outputs\compare_chf\CH_F_history.csv")
print("ALL_PULLED")
