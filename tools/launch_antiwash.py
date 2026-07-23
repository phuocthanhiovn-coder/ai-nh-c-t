"""Launch the anti-washout fine-tune of CH_C on the GPU box.
Warm-start from CH_C + chroma-heavy Lab + highlight-protection.
Run on box: cd /root/autohdr && python3 -m tools.launch_antiwash
"""
from ai_engine.specialists.auto_enhance.gpu.train_sweep import train_one

CFG = {
    "init_ckpt": "checkpoints/gpu/CH_C.pt",   # warm-start from champion
    "data_dir": "data",
    "grid_bins": 8, "grid_size": 16, "proxy_res": 384, "width": 24,  # CH_C arch
    "crop": 512, "batch_size": 4, "lr": 1e-4, "epochs": 120,
    "loss": {
        "w_l1": 1.0,
        "w_lab": 0.6,
        "lab_weights": [0.25, 1.5, 1.5],   # chroma-heavy: preserve color, don't chase lightness
        "w_perc": 0.08,
        "w_hi": 0.5, "hi_gamma": 2.0,      # highlight-protection: stop over-brightening
    },
    "amp": True, "device": "cuda", "val_frac": 0.12,
    "num_workers": 6, "cache_ram": True, "cache_cap": 1024,
    "out": "checkpoints/sweep/CH_E_antiwash.pt",
}

if __name__ == "__main__":
    print("[launch] anti-washout fine-tune, cfg:", CFG, flush=True)
    result = train_one(CFG)
    print("RESULT", result, flush=True)
