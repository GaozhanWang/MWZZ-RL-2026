# ----- multi_run_50.py -----
import time
import numpy as np
import torch
import pandas as pd

from v1_volctrl_1129 import get_device, train_vol1d, Vol1DScaledSpec

def rel_err(pred: float, true: float, eps: float = 1e-6) -> float:
    return abs(pred - true) / max(abs(true), eps)

def quick_diagnostics(spec: Vol1DScaledSpec, u_model, v_model, theta_model,
                      grid: torch.Tensor):
    u_pred = u_model(grid.unsqueeze(-1)).squeeze(-1).cpu().numpy()
    v_pred = v_model(grid.unsqueeze(-1)).squeeze(-1).cpu().numpy()
    th_pred = theta_model(grid.unsqueeze(-1)).squeeze(-1).cpu().numpy()

    g = grid.cpu().numpy()
    u_true = np.exp(-(g / spec.c)**2)
    v_true = -(2.0 / spec.c**2) * g * u_true
    th_true = ((4.0 * g**2) / spec.c**4 - 2.0 / spec.c**2) * u_true

    err_u = np.mean(np.abs(u_pred - u_true))
    err_v = np.mean(np.abs(v_pred - v_true))
    err_th = np.mean(np.abs(th_pred - th_true))
    return dict(L1_u=err_u, L1_v=err_v, L1_theta=err_th)

# ------------------------------ Single trial runner ---------------------------
def run_single_trial(device):
    #set_seed(seed)
    spec = Vol1DScaledSpec(A_min=0.0, A_max=1.0, lambda_reg=1.0, rho=50.0, c=1.0)

    x0 = 0.0

    t0 = time.perf_counter()
    u_model, v_model, w_model, theta_model, meta = train_vol1d(
        spec,
        x_domain=(-1.5, 1.5),
        x0_mode=("fixed", {"x0": x0}),
        # x0_mode=("uniform", {"x_min": -0.2, "x_max": 0.2}),
        T_cut=0.2, time_steps=6,
        dt_psi=1e-2, trap_steps=32,
        training_path_size=8000, nn_batch_size=2000, num_epochs=400,
        width_u=32, width_v=32, width_w=32, width_theta=32,
        learning_rate=0.01, weight_decay=0.0,
        num_a_points=64,
        w_u=1.0, w_v=1.0, w_w=1.0, w_theta=1.0,
        early_stopping=True, es_patience=15, es_min_delta=1e-4,
        device=device, seed=None, verbose=False
    )
    elapsed = time.perf_counter() - t0

    with torch.no_grad():
        x0_torch = torch.tensor([x0], device=device)
        u_hat = u_model(x0_torch.unsqueeze(-1)).item()
        v_hat = v_model(x0_torch.unsqueeze(-1)).item()
        th_hat = theta_model(x0_torch.unsqueeze(-1)).item()

    u_true = spec.u_true_np(x0)
    v_true = spec.v_true_np(x0)
    th_true = spec.theta_true_np(x0)

    def rel_err(true, hat):
        denom = max(abs(true), 1e-12)
        return abs(hat - true) / denom

    out = {
        "u_true": u_true, "u_hat": u_hat, "rel_u": rel_err(u_true, u_hat),
        "v_true": v_true, "v_hat": v_hat, "rel_v": rel_err(v_true, v_hat),
        "th_true": th_true, "th_hat": th_hat, "rel_th": rel_err(th_true, th_hat),
        "time_sec": elapsed,
        "meta_epochs_run": meta["epochs_run"],
        "meta_best_J": float(meta["best_J"]),
    }
    return out

# ------------------------------ Multi-run script ------------------------------
if __name__ == "__main__":
    device = get_device(None)
    n_trials = 50
    #base_seed = 1337

    rows = []
    rel_u_list, rel_v_list, rel_th_list, time_list = [], [], [], []

    print(f"Running {n_trials} trials...\n")
    for i in range(n_trials):
        out = run_single_trial(device=device)

        rel_u_list.append(out["rel_u"])
        rel_v_list.append(out["rel_v"])
        rel_th_list.append(out["rel_th"])
        time_list.append(out["time_sec"])

        print(f"[Trial {i+1:02d}] "
              f"u_true={out['u_true']:.6f} u_hat={out['u_hat']:.6f} REL_u={out['rel_u']:.3e} | "
              f"v_true={out['v_true']:.6f} v_hat={out['v_hat']:.6f} REL_v={out['rel_v']:.3e} | "
              f"th_true={out['th_true']:.6f} th_hat={out['th_hat']:.6f} REL_th={out['rel_th']:.3e} | "
              f"time={out['time_sec']:.2f}s | epochs={out['meta_epochs_run']} J*={out['meta_best_J']:.4e}")

        rows.append({
            "trial": i + 1,
            "u_true": out["u_true"], "u_hat": out["u_hat"], "REL_u": out["rel_u"],
            "v_true": out["v_true"], "v_hat": out["v_hat"], "REL_v": out["rel_v"],
            "th_true": out["th_true"], "th_hat": out["th_hat"], "REL_th": out["rel_th"],
            "time_sec": out["time_sec"],
            "epochs_run": out["meta_epochs_run"],
            "best_J": out["meta_best_J"],
        })

    # Summary stats
    rel_u_arr = np.array(rel_u_list, dtype=float)
    rel_v_arr = np.array(rel_v_list, dtype=float)
    rel_th_arr = np.array(rel_th_list, dtype=float)
    time_arr = np.array(time_list, dtype=float)

    print("\n=== Summary over 50 trials ===")
    print(f"REL_u:  mean={rel_u_arr.mean():.3e}  std={rel_u_arr.std(ddof=1):.3e}")
    print(f"REL_v:  mean={rel_v_arr.mean():.3e}  std={rel_v_arr.std(ddof=1):.3e}")
    print(f"REL_th: mean={rel_th_arr.mean():.3e}  std={rel_th_arr.std(ddof=1):.3e}")
    print(f"time:   mean={time_arr.mean():.2f}s  std={time_arr.std(ddof=1):.2f}s")

    # Save "sheet"
    df = pd.DataFrame(rows)
    df.to_csv("multi_run_50_results.csv", index=False)
    try:
        df.to_excel("vol-ctril-multi_run_50_results.xlsx", index=False)
    except Exception as e:
        print(f"(Excel write skipped: {e})")
    print("\nSaved results to multi_run_50_results.csv (and .xlsx if available).")