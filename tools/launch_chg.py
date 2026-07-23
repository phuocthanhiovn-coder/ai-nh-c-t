"""Train CH_G (24/07): MO TROI KENH SANG — day la phien sua dung goc moi loi
chu du an cham 6 vong (goc khuat toi/o vang, den sau, "sang deu moi goc").

Khac CH_F duy nhat o LOSS (kien truc + data giu nguyen, warm-start CH_F):
  - lab_weights L: 0.25 -> 0.55  (CH_E/F bi troi kenh sang -> khong hoc duoc
    phan bo sang flambient cua AutoHDR; day la goc "cho sang cho toi")
  - w_hi: 0.5 -> 0.2             (noi phat day-sang de tuong trang duoc len)
Muc tieu nghiem thu (CLAUDE.md 23/07): p5 luma khop target ±5 · sat vung toi
(luma<80) <= target+10 · tuong trang >= target-5 — dat so roi moi NHIN ANH chot.

Chay tren box: cd /root/autohdr && nohup python3 -u -m tools.launch_chg > train_chg.log 2>&1 &
"""
from ai_engine.specialists.auto_enhance.gpu.train_sweep import train_one

CFG = {
    "init_ckpt": "checkpoints/gpu/CH_F.pt",
    "data_dir": "data",
    "grid_bins": 8, "grid_size": 16, "proxy_res": 384, "width": 24,
    "crop": 512, "batch_size": 4, "lr": 8e-5, "epochs": 200,
    "loss": {
        "w_l1": 1.0,
        "w_lab": 0.6,
        "lab_weights": [0.55, 1.5, 1.5],   # MO TROI L (truoc 0.25)
        "w_perc": 0.08,
        "w_hi": 0.2, "hi_gamma": 2.0,      # noi highlight-protection (truoc 0.5)
    },
    "amp": True, "device": "cuda", "val_frac": 0.12,
    "num_workers": 6, "cache_ram": True, "cache_cap": 700,
    "out": "checkpoints/sweep/CH_G.pt",
    "save_every": 10,
}

if __name__ == "__main__":
    print("[launch] CH_G: mo troi kenh sang, warm-start CH_F, cfg:", CFG, flush=True)
    result = train_one(CFG)
    print("RESULT", result, flush=True)
