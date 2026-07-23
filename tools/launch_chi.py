"""Train CH_I (24/07 dem): kien truc lon CH_H + LOSS VUNG TOI (dark_fidelity).

Warm-start CH_H (da thang G o den sau + bong toi sach). w_dark=0.6 phat sai lech
L + a/b CHI o vung target toi (thresh 0.28) — tri not: p5 lech 6.67 (muc tieu <=5),
sat toi vuot 4.30 (muc tieu 0), va "o vang do den" chu che 6 vong.

Chay tren box: cd /root/autohdr && nohup python3 -u -m tools.launch_chi > train_chi.log 2>&1 &
"""
from ai_engine.specialists.auto_enhance.gpu.train_sweep import train_one

CFG = {
    "init_ckpt": "checkpoints/sweep/CH_H.pt",
    "data_dir": "data",
    "grid_bins": 10, "grid_size": 24, "proxy_res": 448, "width": 32,
    "crop": 512, "batch_size": 4, "lr": 6e-5, "epochs": 150,
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
    "out": "checkpoints/sweep/CH_I.pt",
    "save_every": 10,
}

if __name__ == "__main__":
    print("[launch] CH_I: CH_H + dark_fidelity loss, cfg:", CFG, flush=True)
    result = train_one(CFG)
    print("RESULT", result, flush=True)
