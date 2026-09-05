# pia_batch_50runs.py
import time
import numpy as np
import torch
import pandas as pd
from datetime import datetime

from v1_pia_0924 import train_model_based_PIA, probe_u_and_v
from ProblemSpec_0928 import EX5

# ------------------------------------------------------------------------------
#  Configurations
# ------------------------------------------------------------------------------
N_RUNS = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results = []
spec = EX5()

# ------------------------------------------------------------------------------
#  Run independent experiments
# ------------------------------------------------------------------------------
for run_idx in range(N_RUNS):
    print(f"\n=== Run {run_idx+1}/{N_RUNS} ===")
    start_time = time.time()

    # Optional: use distinct seeds for reproducibility while keeping randomness
    # torch.manual_seed(run_idx + 123)
    # np.random.seed(run_idx + 4567)

    u_model, meta = train_model_based_PIA(
        spec,
        T=0.1, time_steps=6,
        num_policy_iters=50,
        policy_diff_samples=5000, policy_min_delta=5e-4,
        training_path_size=10000, nn_batch_size=5000, num_epochs=200,
        patience=10, min_delta=1e-3,
        neuron_number_u=64, learning_rate=1e-3, weight_decay=1e-4,
        A_min=spec.A_min, A_max=spec.A_max, num_a_points=96, lambda_reg=spec.lambda_reg,
        x0_init=0.0, x_floor=1e-8,
        device=device,
        verbose=False,
    )

    # Evaluate results at probe (t0,x0)
    res = probe_u_and_v(spec, u_model, t0=0.0, x0=0.0)
    u_hat = res["u"]
    u_true = res["u_true"]
    v_hat = res["v"]
    v_true = res["v_true"]

    u_rel = abs(u_hat - u_true) / max(1e-12, abs(u_true))
    v_rel = abs(v_hat - v_true) / max(1e-12, abs(v_true))
    runtime = time.time() - start_time

    print(f"u_hat={u_hat:.6f} | u_true={u_true:.6f} | rel_err={u_rel:.3e}")
    print(f"v_hat={v_hat:.6f} | v_true={v_true:.6f} | rel_err={v_rel:.3e}")
    print(f"Runtime = {runtime:.2f} sec")

    results.append([run_idx+1, u_hat, u_true, u_rel, v_hat, v_true, v_rel, runtime])

# ------------------------------------------------------------------------------
#  Save results
# ------------------------------------------------------------------------------
results = np.array(results)
df = pd.DataFrame(results, columns=[
    "run", "u_hat", "u_true", "u_rel", "v_hat", "v_true", "v_rel", "runtime"
])

ts = datetime.now().strftime("%Y%m%d-%H%M%S")
csv_name = f"PIA_50runsEX5_{ts}.csv"
npy_name = f"PIA_50runsEX5_{ts}.npy"

df.to_csv(csv_name, index=False)
np.save(npy_name, results)

# ------------------------------------------------------------------------------
#  Summary statistics
# ------------------------------------------------------------------------------
mean_u_hat = np.nanmean(df["u_hat"])
mean_u_rel = np.nanmean(df["u_rel"])
std_u_rel = np.nanstd(df["u_rel"])
mean_v_rel = np.nanmean(df["v_rel"])
std_v_rel = np.nanstd(df["v_rel"])
mean_time = np.mean(df["runtime"])

print("\n========== SUMMARY ==========")
print(f"Mean relative u error: {mean_u_rel:.3e}")
print(f"Mean u hat: {mean_u_hat:.3e}")
print(f"Mean runtime: {mean_time:.2f} sec")
print(f"Saved results to {csv_name} and {npy_name}")
