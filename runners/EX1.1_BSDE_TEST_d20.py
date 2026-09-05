import torch
import numpy as np
import time
import json
from v1_deep_bsde_1005 import TrainConfig, train_deep_bsde_for_problemspec
from ProblemSpecHD_0925 import EX7HighDim

# ------------------ Problem setup ------------------
D = 20
spec = EX7HighDim(d=D, T=0.1, A_min=0.0, A_max=1.0, lambda_reg=5.0)

t0 = 0.0
x0 = 0.1 * np.ones(D, dtype=float)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------ Training configuration template ------------------
base_cfg = TrainConfig(
    n_steps=50,
    batch_size=5000,
    num_iters=300,
    lr=3e-4,
    y0_from_x0=True,
    z_hidden=(128, 128),
    action_integration="legendre",
    n_action=64,
    seed=None,  # to be overwritten per run
    log_every=200,
    verbose=False,
    early_stopping=True,
    patience=12,
    min_delta=1e-4,
)

# ------------------ Run multiple independent trials ------------------
n_runs = 50
results = []

for run_idx in range(n_runs):
    # torch.manual_seed(run_idx + 42)
    # np.random.seed(run_idx + 4242)

    #cfg = base_cfg._replace(seed=run_idx + 42)

    print(f"\n========== Run {run_idx + 1}/{n_runs} ==========")
    start_time = time.time()

    model, metrics = train_deep_bsde_for_problemspec(spec, base_cfg, device=device, x0_probe=x0)

    elapsed = time.time() - start_time

    u_hat = metrics.get("u_hat", float('nan'))
    u_true = metrics.get("u_true", float('nan'))
    abs_err = abs(u_hat - u_true) if not np.isnan(u_true) else float('nan')
    rel_err = abs_err / max(1e-12, abs(u_true)) if not np.isnan(u_true) else float('nan')

    run_result = {
        "run_idx": run_idx,
        "u_hat": u_hat,
        "u_true": u_true,
        "abs_err": abs_err,
        "rel_err": rel_err,
        "runtime_sec": elapsed,
        "final_loss": metrics.get("final_loss", float('nan')),
        "best_loss": metrics.get("best_loss", float('nan')),
        "iters": metrics.get("iters", base_cfg.num_iters),
    }

    results.append(run_result)
    print(f"Runtime {elapsed:.2f}s | u_hat={u_hat:.4f}, u_true={u_true:.4f}, rel_err={rel_err:.2e}")

# ------------------ Summary statistics ------------------
abs_errs = np.array([r["abs_err"] for r in results])
rel_errs = np.array([r["rel_err"] for r in results])
runtimes = np.array([r["runtime_sec"] for r in results])

summary = {
    "n_runs": n_runs,
    "abs_err_mean": float(np.nanmean(abs_errs)),
    "abs_err_std": float(np.nanstd(abs_errs)),
    "rel_err_mean": float(np.nanmean(rel_errs)),
    "rel_err_std": float(np.nanstd(rel_errs)),
    "runtime_mean": float(np.nanmean(runtimes)),
    "runtime_std": float(np.nanstd(runtimes)),
}

print("\n========== Summary ==========")
for k, v in summary.items():
    print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

# ------------------ Optional: Save to file ------------------
timestamp = time.strftime("%Y%m%d_%H%M%S")
save_path = f"deepbsde_ex7_d20_{n_runs}_{timestamp}.json"
with open(save_path, "w") as f:
    json.dump({"results": results, "summary": summary}, f, indent=2)
print(f"\nResults saved to {save_path}")
