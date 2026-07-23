"""Train CH_F tren box GPU 23/07: warm-start tu CH_E_antiwash, data 940 cap (v5).
Cung arch + loss anti-washout da thang o CH_E; nhieu data hon 30%, epochs dai hon.
Chay tren box: cd /root/autohdr && nohup python3 -m tools.launch_chf > train_chf.log 2>&1 &
"""
from ai_engine.specialists.auto_enhance.gpu.train_sweep import train_one

CFG = {
    "init_ckpt": "checkpoints/gpu/CH_E_antiwash.pt",  # warm-start tu champion hien tai
    "data_dir": "data",
    "grid_bins": 8, "grid_size": 16, "proxy_res": 384, "width": 24,  # arch CH_E
    "crop": 512, "batch_size": 4, "lr": 8e-5, "epochs": 200,
    "loss": {
        "w_l1": 1.0,
        "w_lab": 0.6,
        "lab_weights": [0.25, 1.5, 1.5],
        "w_perc": 0.08,
        "w_hi": 0.5, "hi_gamma": 2.0,
    },
    "amp": True, "device": "cuda", "val_frac": 0.12,
    "num_workers": 6, "cache_ram": True, "cache_cap": 700,  # RAM box 14GB — de tran la crash
    "out": "checkpoints/sweep/CH_F.pt",
    "save_every": 10,
}

if __name__ == "__main__":
    print("[launch] CH_F: warm-start CH_E, 940 cap, cfg:", CFG, flush=True)
    result = train_one(CFG)
    print("RESULT", result, flush=True)
