from tools.gpu_ssh import run, get
import os
os.makedirs(r"outputs\compare_chj", exist_ok=True)
rc,out,err = run("ls /root/autohdr/outputs/compare_chj/", timeout=30)
for n in [x for x in out.split() if x.endswith(".jpg")]:
    get("/root/autohdr/outputs/compare_chj/"+n, r"outputs\compare_chj" + "\\" + n)
get("/root/autohdr/checkpoints/sweep/CH_J.pt", r"checkpoints\gpu\CH_J.pt")
print("pulled + CH_J.pt")
