"""All settings in one place."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent
RAW_FILE, INTERIM, PROCESSED, CKPT = ROOT / "data/raw/electricity.txt.gz", ROOT / "data/interim", ROOT / "data/processed", ROOT / "checkpoints"
RES, FIG = ROOT / "results", ROOT / "figures"
for p in (INTERIM, PROCESSED, CKPT, RES, FIG): p.mkdir(parents=True, exist_ok=True)

# ---- forecasting task: hourly electricity, look-back 1 week, forecast next 24 h ----
L, H, PATCH, STRIDE = 168, 24, 16, 8
TRAIN_CLIENTS, FORGET, CONTROL = list(range(10)), 9, list(range(10, 20))   # client 10 (index 9) is the one to forget; clients 11-20 never trained on
TRAIN_END, VAL_END, WINDOW_STRIDE = 0.70, 0.80, 4
EPOCHS, BATCH, LR = 20, 128, 1e-3
ARCHS = ["dlinear", "mlp", "patchtst"]
SEEDS = {"dlinear": [0, 1, 2], "mlp": [0, 1], "patchtst": [0]}        # MLP and PatchTST are slower to train: fewer seeds

# ---- quantization: post-training, weights of every Linear layer, signed integers ----
BITS = [8, 6, 4, 3, 2]
QUANT = {"channel": ("per-channel, abs-max", False, None),                    # standard
         "tensor": ("per-tensor, abs-max", True, None),                       # one scale per layer, common on small hardware
         "clip": ("per-channel, 99th-percentile clipping", False, 0.99)}      # outlier-clipping calibration used for time-series PTQ

# ---- unlearning: time-series methods only, ONE fixed setting each (no search, no tuning), judged after the last step ----
STEPS, BATCH_U = 400, 64
METHODS = {"RandLabel": 1e-3, "ZeroLabel": 1e-3, "NoisyLabel": 1e-3, "MaxLoss": 1e-4, "TSU": 1e-3, "GaussAmnesiac": 0.1, "Blindspot": 1e-3}
CALIB_WINDOWS = 512

# ---- rules fixed BEFORE running (docs/DESIGN.md) ----
STUCK_BELOW = 0.01            # fewer than 1% of the integer weights changed -> the method did nothing
SAME_SCORE, SAME_RETAIN = 0.25, 0.10   # "same effect as full precision": forget score within 0.25, retain error at most 0.10 worse
FORGOTTEN_RANGE, MAX_RETAIN_DAMAGE = (0.8, 1.2), 1.10
