# batch_highdim_test.py
import time
import math
import numpy as np
import torch
import gc
import pandas

# ------------------------------ Device ------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------------ Your imports ------------------------------
from ProblemSpecHD_0925 import (EX7HighDim, EX5HighDim)
from v2_highdim_0924 import (
    ProblemSpecD, train_vanilla_with_norm_d, recover_u_at_point_d
)

# ------------------------------ Config ------------------------------
N_RUNS = 1

# Choose spec & dimension
SPEC_CTOR = EX7HighDim     # switch to EX7HighDim if you prefer
D = 100

# Hyperparameters (kept consistent with your example)
T = 0.1
TIME_STEPS = 5
TRAIN_PATHS = 5000
BATCH_SIZE = 5000
EPOCHS = 200
NEURONS_V = 128
NEURONS_W = 128
LR = 6e-4
WEIGHT_DECAY = 0.0
NUM_A_POINTS = 96
INT_X_STEPS = 20
NORM_TYPE = "softmax"      # or "lp"
LP_P = 8.0
SOFT_TAU = "auto"
EARLY_STOP = True
ES_PATIENCE = 17
ES_MIN_DELTA = 1e-3

# Probe point
t0 = 0.0
x0 = 0.1 * np.ones(D, dtype=float)

def one_run(run_idx: int):
    """Runs one full train+recover; returns (u_hat, rel_err, elapsed_sec)."""
    # # distinct seeds for variability
    # torch.manual_seed(run_idx + 42)
    # np.random.seed(run_idx + 4242)

    spec = SPEC_CTOR(d=D, T=T, A_min=0.0, A_max=1.0, lambda_reg=5.0)

    start = time.time()
    v_Model, w_Model, meta = train_vanilla_with_norm_d(
        spec,
        T=spec.T, time_steps=TIME_STEPS,
        training_path_size=TRAIN_PATHS, nn_batch_size=BATCH_SIZE, num_epochs=EPOCHS,
        neuron_number_v=NEURONS_V, neuron_number_w=NEURONS_W,
        learning_rate=LR, weight_decay=WEIGHT_DECAY,
        A_min=spec.A_min, A_max=spec.A_max, num_a_points=NUM_A_POINTS, lambda_reg=spec.lambda_reg,
        int_x_steps=INT_X_STEPS,
        norm_type=NORM_TYPE, lp_p=LP_P, softmax_tau=SOFT_TAU,
        x0_init=x0,
        early_stopping=EARLY_STOP, es_patience=ES_PATIENCE, es_min_delta=ES_MIN_DELTA,
        verbose=False
    )

    u_hat, u_se = recover_u_at_point_d(
        spec, w_Model, t0=t0, x0=x0,
        T=meta['T'], time_steps=meta['time_steps'],
        num_a_points=meta['num_a_points'], lambda_reg=meta['lambda_reg'],
        A_min=meta['A_min'], A_max=meta['A_max'],
        n_paths=8000
    )
    elapsed = time.time() - start

    u_true = spec.u_true_np(t0, x0)
    denom = max(1e-12, abs(u_true)) if not isinstance(u_true, float) else max(1e-12, abs(u_true))
    rel_err = abs(u_hat - u_true) / denom

    # cleanup between runs
    del v_Model, w_Model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return u_hat, rel_err, elapsed

def main():
    u_hats = []
    rel_errs = []
    times = []

    print(f"Running {N_RUNS} trials on device={device} (spec={SPEC_CTOR.__name__}, d={D})...")
    for i in range(N_RUNS):
        u_hat, rel_err, elapsed = one_run(i)
        u_hats.append(u_hat)
        rel_errs.append(rel_err)
        times.append(elapsed)
        print(f"[{i+1:02d}/{N_RUNS}] u_hat={u_hat:.8f}, rel_err={rel_err:.6e}, time={elapsed:.2f}s")

    u_hats = np.asarray(u_hats, dtype=float)
    rel_errs = np.asarray(rel_errs, dtype=float)
    times = np.asarray(times, dtype=float)

    mean_u_hat = float(np.mean(u_hats))
    mean_rel_err = float(np.mean(rel_errs))
    #std_rel_err = float(np.std(rel_errs, ddof=1))  # sample variance
    mean_time = float(np.mean(times))

    print("\n===== Summary over {N} runs =====".format(N=N_RUNS))
    print(f"mean( u_hat )          = {mean_u_hat:.10f}")
    print(f"mean( rel error )      = {mean_rel_err:.6e}")
    #print(f"STD ( rel error )      = {std_rel_err:.6e} (sample STD)")
    print(f"mean runtime per run   = {mean_time:.2f} s")

    # Optional: save CSV
    try:
        import pandas as pd
        import datetime as dt
        ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        df = pd.DataFrame({
            "u_hat": u_hats,
            "rel_error": rel_errs,
            "runtime_sec": times,
        })
        out = f"highdim_batch_{SPEC_CTOR.__name__}_d{D}_{ts}.csv"
        df.to_csv(out, index=False)
        print(f"Saved per-run metrics to {out}")
    except Exception as e:
        print(f"(CSV save skipped: {e})")

if __name__ == "__main__":
    main()
