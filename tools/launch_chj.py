"""Train CH_J (24/07 dem): CH_I + DATA SACH (912 cap — loai 28 cap doc:
23+5 cap before TIM HONG do loi develop RAW loat may 5O2A cac job
j002/003/004/005/026/056 + 2 cap chup xuyen kinh phan chieu). 28 cap nay da
day model phep bien doi rac "tim hong -> sach" — nghi pham gop phan
lam mau nhieu canh co do vat bi lech (chu du an che "phong co do vat khong
tien bo"). Warm-start CH_I, chay ngan (100 epoch) vi chi doi data.

Chay tren box: cd /root/autohdr && nohup python3 -u -m tools.launch_chj > train_chj.log 2>&1 &
"""
from ai_engine.specialists.auto_enhance.gpu.train_sweep import train_one

CFG = {
    "init_ckpt": "checkpoints/sweep/CH_I.pt",
    "data_dir": "data",
    "grid_bins": 10, "grid_size": 24, "proxy_res": 448, "width": 32,
    "crop": 512, "batch_size": 4, "lr": 5e-5, "epochs": 100,
    "loss": {
        "w_l1": 1.0,
        "w_lab": 0.6,
        "lab_weights": [0.6, 1.5, 1.5],
        "w_perc": 0.08,
        "w_hi": 0.2, "hi_gamma": 2.0,
        "w_dark": 0.6, "dark_thresh": 0.28,
    },
    "amp": True, "device": "cuda", "val_frac": 0.12,
    "num_workers": 6, "cache_ram": True, "cache_cap": 700,
    "out": "checkpoints/sweep/CH_J.pt",
    "save_every": 10,
}

if __name__ == "__main__":
    print("[launch] CH_J: CH_I + data sach 912 cap, cfg:", CFG, flush=True)
    result = train_one(CFG)
    print("RESULT", result, flush=True)
