# mp_repeat_test.py
import time
import math
import numpy as np
import torch
import torch.nn as nn
import gc

from ProblemSpec_0928 import EX5   # or EX7 if you want to switch
from v2_multi_0925 import (
    ProblemSpec, train_multi_period, recover_u_multi, probe_v_at
)

# def set_run_seed(run_idx: int):
#     """
#     Per-run reproducibility (while varying between runs).
#     """
#     seed = 42 + run_idx
#     np.random.seed(4242 + run_idx)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     # Determinism knobs (optional; comment out if you prefer speed)
#     torch.backends.cudnn.deterministic = False
#     torch.backends.cudnn.benchmark = True

def run_once(device: torch.device):
    """
    Single training/eval run. Returns (u_hat, rel_err, runtime_sec).
    """
    spec = EX5()  # choose your problem (EX5 in user snippet)

    # --- configuration (copied from your example) ---
    T = 0.3
    n_segments = 3
    time_steps = 6
    num_epochs = 250
    x_domain = (-0.90, 0.90)
    x0_mode  = ("uniform", {"x_min": -0.9, "x_max": 0.9})  # sampling for each segment's initial x
    initial_x = 0.0                                        # earliest segment starts here
    training_path_size = 5000
    nn_batch_size = 5000
    neuron_number_v = 64
    neuron_number_w = 64
    learning_rate = 5e-4
    weight_decay = 0.0
    num_a_points = 96
    lambda_reg = spec.lambda_reg
    int_x_steps = 100
    norm_type = "softmax"
    softmax_tau = "auto"
    early_stopping = True
    es_patience = 20
    es_min_delta = 5e-4
    verbose = False
    exp_clip = None

    t0 = 0.0
    x0 = initial_x

    # --- train + evaluate ---
    start = time.time()

    v_models, w_models, breaks, stats = train_multi_period(
        spec, T=T,
        n_segments=n_segments, time_steps=time_steps, num_epochs=num_epochs,
        x_domain=x_domain, x0_mode=x0_mode,
        # initial_x is used by your modified trainer to pin earliest segment start
        initial_x=initial_x,
        training_path_size=training_path_size, nn_batch_size=nn_batch_size,
        neuron_number_v=neuron_number_v, neuron_number_w=neuron_number_w,
        learning_rate=learning_rate, weight_decay=weight_decay,
        num_a_points=num_a_points, lambda_reg=lambda_reg,
        int_x_steps=int_x_steps,
        norm_type=norm_type, softmax_tau=softmax_tau,
        early_stopping=early_stopping, es_patience=es_patience, es_min_delta=es_min_delta,
        device=device, verbose=verbose,
        exp_clip=exp_clip,
    )

    # (Optional) probe derivative accuracy at (t0,x0)
    # v_hat = probe_v_at(v_models, breaks, t0=t0, x0=x0, device=device)

    # FK recovery of u(t0, x0)
    u_hat, u_se = recover_u_multi(
        spec, w_models, breaks, t0=t0, x0=x0,
        num_a_points=100, time_steps_per_segment=21, n_paths=10000, device=device
    )
    u_true = spec.u_true_np(t0, x0)
    rel_err = abs(u_hat - u_true) / max(1e-12, abs(u_true))

    runtime = time.time() - start

    # Clean up memory between runs (important on GPU)
    del v_models, w_models, breaks, stats
    torch.cuda.empty_cache()
    gc.collect()

    return u_hat, rel_err, runtime

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N_RUNS = 50

    u_hats = []
    rel_errs = []
    runtimes = []

    for run_idx in range(N_RUNS):
        # set_run_seed(run_idx)
        u_hat, rel_err, runtime = run_once(device)
        u_hats.append(u_hat)
        rel_errs.append(rel_err)
        runtimes.append(runtime)
        print(f"[{run_idx+1:02d}/{N_RUNS}] u_hat={u_hat:.6f} | rel_err={rel_err:.6%} | time={runtime:.2f}s")

    u_hats = np.asarray(u_hats, dtype=float)
    rel_errs = np.asarray(rel_errs, dtype=float)
    runtimes = np.asarray(runtimes, dtype=float)

    print("\n================ Summary over runs ================")
    print(f"Mean u_hat           : {u_hats.mean():.6f}")
    print(f"Mean relative error  : {rel_errs.mean():.6%}")
    print(f"STD(relative error)  : {rel_errs.std(ddof=0):.6e}")
    print(f"Mean runtime (sec)   : {runtimes.mean():.2f}")
    print("===================================================")

            # Optional: save to CSV
    try:
        import pandas as pd
        import datetime as dt
        ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        df = pd.DataFrame({
            "u_hat": u_hats,
            "rel_error": rel_errs,
            "runtime_sec": runtimes,
        })
        df.to_csv(f"multi_period_batch_EX5T03_{ts}.csv", index=False)
        print(f"Saved per-run metrics to single_period_batch_{ts}.csv")
    except Exception as e:
        print(f"(CSV save skipped: {e})")


if __name__ == "__main__":
    main()
