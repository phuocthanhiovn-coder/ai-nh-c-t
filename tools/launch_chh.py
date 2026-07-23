"""Train CH_H (24/07 dem): NHAY DUNG LUONG KIEN TRUC — thi nghiem tu quyet
(chu du an giao toan quyen, chi quan tam ket qua).

VI SAO: val_l1 dung im ~0.053 qua 3 doi loss (E/F/G) => tran KIEN TRUC, khong
phai loss. Luoi 16x16x8 qua tho de "thap sang tung goc theo ngu canh" (nhan xet
6 vong cua chu: sang deu, goc ro, den sau — can dieu khien khong gian min hon).

Doi moi: grid 16->24 (2.25x o), bins 8->10, width 24->32, proxy 384->448.
Train TU DAU (khac shape, khong warm-start duoc). Loss giu cua CH_G (da chung
minh keo trang tot) + L nhinh 0.6.

Chay tren box: cd /root/autohdr && nohup python3 -u -m tools.launch_chh > train_chh.log 2>&1 &
"""
from ai_engine.specialists.auto_enhance.gpu.train_sweep import train_one

CFG = {
    # KHONG init_ckpt — train tu dau (kien truc moi)
    "data_dir": "data",
    "grid_bins": 10, "grid_size": 24, "proxy_res": 448, "width": 32,
    "crop": 512, "batch_size": 4, "lr": 2e-4, "epochs": 240,
    "loss": {
        "w_l1": 1.0,
        "w_lab": 0.6,
        "lab_weights": [0.6, 1.5, 1.5],
        "w_perc": 0.08,
        "w_hi": 0.2, "hi_gamma": 2.0,
    },
    "amp": True, "device": "cuda", "val_frac": 0.12,
    "num_workers": 6, "cache_ram": True, "cache_cap": 700,
    "out": "checkpoints/sweep/CH_H.pt",
    "save_every": 10,
}

if __name__ == "__main__":
    print("[launch] CH_H: kien truc lon (24x24x10, w32, proxy448), tu dau, cfg:", CFG, flush=True)
    result = train_one(CFG)
    print("RESULT", result, flush=True)
