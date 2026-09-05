# batch_single_period_test_savefigs.py
import time
import math
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# ------------------------------ Device ------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------------ Your imports (from your module) ------------------------------
from v2_single_0928 import (
    train_vanilla_with_norm,
    recover_u_at_point,
    # the plotting helpers we added earlier; they accept show=False
    plot_policy_slice,
    plot_policy_with_truth,
    plot_u_and_relerror_surface,
    plot_u_and_abserror_surface,
    plot_u_curve_vs_true
)
from ProblemSpec_260416 import (ProblemSpec, EX7, EX5)

# ------------------------------ Config ------------------------------
N_RUNS = 1  # keep 1 as requested

# Problem + training hyperparams (same as your example)
SPEC_CTOR = EX7                      # choose EX7/EX5/EX4
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

# Probe point and interval
t0, x0 = 0.0, 0.1
x_min, x_max, M = -1.0, 1.0, 21

# Output files
OUT_U_INTERVAL = "u_interval.png"
OUT_PI_EST = "pi_est_surface.png"
OUT_PI_EST_VS_TRUE = "pi_est_vs_true.png"
OUT_U_CURVE_VS_TRUE = "u_curve_vs_true.png"
FIG_DPI = 200

def one_run(run_idx: int):
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
        x0_mode=("uniform", {"x_min": x_min, "x_max": x_max, "n_points": M}),
        verbose=False,
        early_stopping=EARLY_STOP, es_patience=ES_PATIENCE, es_min_delta=ES_MIN_DELTA
    )

    # Recover single point estimate (for reporting)
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

    return v_Model, w_Model, meta, spec, u_hat, u_se, u_true, rel_err, elapsed

def save_interval_u_plot(spec, w_model, meta, t0, x_min, x_max, M, outpath):
    # Recover interval
    x_grid, u_mean, u_se = recover_u_at_point(
        spec, w_model, t0=t0, x0=0.0,
        T=meta["T"], time_steps=meta["time_steps"],
        num_a_points=meta["num_a_points"], lambda_reg=meta["lambda_reg"],
        A_min=meta["A_min"], A_max=meta["A_max"],
        n_paths=4000,
        x0_mode=("interval", {"x_min": x_min, "x_max": x_max, "n_points": M})
    )

    plt.figure(figsize=(6,4))
    plt.plot(x_grid, u_mean, label=r"$\hat u^\lambda(t_0,x)$")
    plt.fill_between(x_grid, u_mean - 2*u_se, u_mean + 2*u_se, alpha=0.25, label="±2 SE")

    # overlay truth when available
    u_true_curve = np.array([spec.u_true_np(t0, float(x)) for x in x_grid], dtype=float)
    if not np.all(np.isnan(u_true_curve)):
        plt.plot(x_grid, u_true_curve, "--", label=r"$u(t_0,x)$ (true)")

    plt.xlabel("x"); plt.ylabel("u")
    plt.title(fr"$u(t_0={t0}, x)$ over [{x_min}, {x_max}]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=FIG_DPI)
    plt.close()
    print(f"Saved interval u plot to: {outpath}")

def save_pi_est_surface(spec, w_model, t0, x_min, x_max, n_x, a_min, a_max, n_a, outpath):
    # plot_policy_slice returns (X_plot, A_plot, pi) and accepts show=False
    # ensure the function doesn't block (show=False)
    _X, _A, pi = plot_policy_slice(
        spec, w_model, t0=t0,
        x_min=x_min, x_max=x_max, n_x=n_x,
        A_min=a_min, A_max=a_max, n_a=n_a,
        lambda_reg=spec.lambda_reg,
        show=False
    )
    # Save current figure
    try:
        plt.gcf().savefig(outpath, dpi=FIG_DPI)
        plt.close()
        print(f"Saved estimated pi surface to: {outpath}")
    except Exception as e:
        print(f"Failed to save estimated pi surface: {e}")

def save_pi_est_vs_true(spec, w_model, t0, x_min, x_max, n_x, a_min, a_max, n_a, outpath):
    """
    Saves figure with t0 embedded in filename.
    Example:
        outpath="pi_est_vs_true.png"
        -> saved as "pi_est_vs_true_t0_0.10.png"
    """
    # -------- create t0 string safely for filenames --------
    t0_str = f"{t0:.3f}".replace('.', 'p')   # e.g. 0.100 -> "0p100"

    # split filename and extension
    import os
    base, ext = os.path.splitext(outpath)
    new_outpath = f"{base}_t0_{t0_str}{ext}"

    # -------- generate figure --------
    X_plot, A_plot, pi_est, pi_true = plot_policy_with_truth(
        spec, w_model, t0=t0,
        x_min=x_min, x_max=x_max, n_x=n_x,
        A_min=a_min, A_max=a_max, n_a=n_a,
        lambda_reg=spec.lambda_reg,
        show=False
    )

    try:
        plt.gcf().savefig(new_outpath, dpi=FIG_DPI)
        plt.close()
        print(f"Saved pi est vs true surface to: {new_outpath}")
    except Exception as e:
        print(f"Failed to save pi est vs true surface: {e}")

def save_u_curve_vs_true(spec, w_model, meta, t0, x_min, x_max, n_x, n_paths, outpath):
    """
    Saves the 1D u curve vs true figure with t0 embedded in the filename.
    """
    t0_str = f"{t0:.3f}".replace('.', 'p')
    base, ext = os.path.splitext(outpath)
    new_outpath = f"{base}_t0_{t0_str}{ext}"

    plot_u_curve_vs_true(
        spec=spec,
        w_model=w_model,
        t0=t0,
        x_min=x_min,
        x_max=x_max,
        n_x=n_x,
        n_paths=n_paths,
        time_steps=meta["time_steps"],
        num_a_points=meta["num_a_points"],
        lambda_reg=meta["lambda_reg"],
        show=False,
        savepath=new_outpath
    )

def main():
    # single run only
    v_Model = w_Model = meta = spec = None

    print(f"Running single trial on device={device} (spec={SPEC_CTOR.__name__})...")
    v_Model, w_Model, meta, spec, u_hat, u_se, u_true, rel_err, elapsed = one_run(0)

    print(f"u_hat={u_hat:.8f}, u_se={u_se:.6e}, u_true={u_true:.8f}, rel_err={rel_err:.6e}, time={elapsed:.2f}s")

    # # compute & save u surface over t in [0, T] and x in [x_min,x_max]
    # t_grid, x_grid, U_mean, U_se = plot_u_surface(
    #     spec, w_Model,
    #     t_min=0.0, t_max=0.1, n_t=10,
    #     x_min=0.00, x_max=0.2, n_x=10,
    #     n_paths=10000,             # tune for speed/variance tradeoff
    #     time_steps=meta["time_steps"],
    #     num_a_points=meta["num_a_points"],
    #     lambda_reg=meta["lambda_reg"],
    #     show_true=True,
    #     cmap='viridis',
    #     savepath="u_surface.png",
    #     show=False
    # )

#     t_grid, x_grid, U_mean, U_err, fig = plot_u_and_relerror_surface(
#     spec, w_Model,
#     t_min=0.0, t_max=0.25, n_t=25,
#     x_min=-1, x_max=1, n_x=20,
#     n_paths=3000,
#     time_steps=meta["time_steps"],
#     num_a_points=meta["num_a_points"],
#     lambda_reg=meta["lambda_reg"],
#     show_true=True,
#     savepath="u_and_error_surface_t0toT.png",
#     show=False
#    )
    # Save interval u plot (with truth overlay)
    #save_interval_u_plot(spec, w_Model, meta, 0.0, x_min, x_max, M, OUT_U_INTERVAL)

    # Save estimated pi surface (over the same x-interval)
    #save_pi_est_surface(spec, w_Model, 0.0, x_min, x_max, n_x=81, a_min=spec.A_min, a_max=spec.A_max, n_a=101, outpath=OUT_PI_EST)

    # Save combined pi_est vs pi_true

    # save_pi_est_vs_true(spec, w_Model, t0=0.0, x_min=-1.0, x_max=1.0, n_x=81, a_min=spec.A_min, a_max=spec.A_max, n_a=101, outpath=OUT_PI_EST_VS_TRUE)
    # save_pi_est_vs_true(spec, w_Model, t0=0.2, x_min=-1.0, x_max=1.0, n_x=81, a_min=spec.A_min, a_max=spec.A_max, n_a=101, outpath=OUT_PI_EST_VS_TRUE)
    # save_pi_est_vs_true(spec, w_Model, t0=0.4, x_min=-1.0, x_max=1.0, n_x=81, a_min=spec.A_min, a_max=spec.A_max, n_a=101, outpath=OUT_PI_EST_VS_TRUE)


    
    save_u_curve_vs_true(spec, w_Model, meta, t0=0.0, x_min=-1.0, x_max=1.0, n_x=21, n_paths=3000, outpath=OUT_U_CURVE_VS_TRUE)

    # cleanup GPU memory
    del v_Model, w_Model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nAll done.")

if __name__ == "__main__":
    main()