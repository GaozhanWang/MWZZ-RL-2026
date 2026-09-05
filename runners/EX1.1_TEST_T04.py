# batch_single_period_test.py
import time
import math
import numpy as np
import torch
import pandas

# ------------------------------ Device ------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------------ Your imports ------------------------------
from v2_single_0928 import (
    train_vanilla_with_norm, recover_u_at_point
)
# from v2_5_single_0928_beta import (
#     train_vanilla_with_norm, recover_u_at_point
# )
from ProblemSpec_260416 import (ProblemSpec, EX7, EX5)

# ------------------------------ Config ------------------------------
N_RUNS = 5

# Problem + training hyperparams (same as your example)
SPEC_CTOR = EX7                      # pick EX7/EX5/EX4
T = 0.4
TIME_STEPS = 11
TRAIN_PATHS = 5000
BATCH_SIZE = 5000
EPOCHS = 500
NEURONS_V = 64
NEURONS_W = 32
LR = 5e-4
WEIGHT_DECAY = 0.0
NUM_A_POINTS = 96
INT_X_STEPS = 40
NORM_TYPE = "softmax"                # "lp" or "softmax"
LP_P = 8.0
SOFT_TAU = "auto"
EARLY_STOP = True
ES_PATIENCE = 20
ES_MIN_DELTA = 1e-3

# Probe point
t0, x0 = 0.0, 0.1

def one_run(run_idx: int):
    """Runs one full train+recover, returns (u_hat, rel_err, elapsed_sec)."""
    # distinct seeds for variability
    # torch.manual_seed(run_idx + 1234)
    # np.random.seed(run_idx + 5678)

    spec = SPEC_CTOR()
    start = time.time()

    v_Model, w_Model, meta = train_vanilla_with_norm(
        spec,
        T=T, time_steps=TIME_STEPS,
        training_path_size=TRAIN_PATHS, nn_batch_size=BATCH_SIZE, num_epochs=EPOCHS,
        neuron_number_v=NEURONS_V, neuron_number_w=NEURONS_W,
        learning_rate=LR, weight_decay=WEIGHT_DECAY,
        A_min=spec.A_min, A_max=spec.A_max, num_a_points=NUM_A_POINTS,
        lambda_reg=spec.lambda_reg,
        int_x_steps=INT_X_STEPS,
        norm_type=NORM_TYPE, lp_p=LP_P, softmax_tau=SOFT_TAU,
        x0_init=x0,
        x0_mode= ("fixed", {"x_min": None, "x_max": None}),
        verbose=False,  # keep logs quiet for batch runs
        early_stopping=EARLY_STOP, es_patience=ES_PATIENCE, es_min_delta=ES_MIN_DELTA
    )

    u_hat, u_se = recover_u_at_point(
        spec, w_Model, t0=t0, x0=x0,
        T=meta["T"], time_steps=meta["time_steps"],
        num_a_points=meta["num_a_points"], lambda_reg=meta["lambda_reg"],
        A_min=meta["A_min"], A_max=meta["A_max"],
        n_paths=10000
    )

    elapsed = time.time() - start

    u_true = spec.u_true_np(t0, x0)
    denom = max(1e-12, abs(u_true))
    rel_err = abs(u_hat - u_true) / denom

    # Free GPU memory between runs
    del v_Model, w_Model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return u_hat, rel_err, elapsed

def main():
    u_hats = []
    rel_errs = []
    times = []

    print(f"Running {N_RUNS} trials on device={device} (spec={SPEC_CTOR.__name__})...")
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
    var_rel_err = float(np.var(rel_errs, ddof=1))  # sample variance
    mean_time = float(np.mean(times))

    print("\n===== Summary over {N} runs =====".format(N=N_RUNS))
    print(f"mean( u_hat )          = {mean_u_hat:.10f}")
    print(f"mean( rel error )      = {mean_rel_err:.6e}")
    print(f"var ( rel error )      = {var_rel_err:.6e} (sample variance)")
    print(f"mean runtime per run   = {mean_time:.2f} s")

    # Optional: save to CSV
    try:
        import pandas as pd
        import datetime as dt
        ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        df = pd.DataFrame({
            "u_hat": u_hats,
            "rel_error": rel_errs,
            "runtime_sec": times,
        })
        df.to_csv(f"single_period_batch_EX7_T04{SPEC_CTOR.__name__}_{ts}.csv", index=False)
        print(f"Saved per-run metrics to single_period_batch_{SPEC_CTOR.__name__}_{ts}.csv")
    except Exception as e:
        print(f"(CSV save skipped: {e})")

if __name__ == "__main__":
    main()
