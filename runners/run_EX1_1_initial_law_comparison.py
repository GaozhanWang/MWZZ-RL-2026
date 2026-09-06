#!/usr/bin/env python3
"""Compare Dirac and uniform initial-state training for EX1.1 (EX7, d=1).

Place this file beside v2_single_0928.py and ProblemSpec_260416.py, or in
the repository's runner/ folder with those modules in algorithms/.
Requires Python >= 3.10, PyTorch, NumPy, Matplotlib and SciPy.

    python run_EX1_1_initial_law_comparison.py
    python run_EX1_1_initial_law_comparison.py --smoke-test --device cpu
    python run_EX1_1_initial_law_comparison.py --no-early-stopping

The defaults reproduce the TRAINING SETTINGS of EX1.1_TEST_d1_plot.py.
Two new models are trained: X_0 = 0.1 and X_0 ~ Uniform[-1, 1]. The
training algorithm is imported unchanged. The same seed gives the two
runs the same initial network weights; it does not give them identical
training trajectories, since uniform sampling consumes extra randomness.
Early stopping uses the same rule but can stop at different epochs. Use
--no-early-stopping for the same number of optimization steps in both runs.

Both learned curves are recovered at t=0 using the SAME evaluation noise,
grid, horizon and Monte Carlo budget. Recovery runs one x at a time in
small batches to limit memory. Estimates are plotted without smoothing.
The plotted models are the final returned models, as in the original
runner; the algorithm's separately saved 'best' models are not reloaded.

Each execution creates a new output directory containing PNG/PDF figures,
CSV curve data (including Monte Carlo SEs), JSON settings/metrics and
separate checkpoints for both training laws. No existing run is replaced.
Monte Carlo SEs describe recovery noise conditional on each trained model;
they do not measure variability across independent training runs. Change
TRAIN_SEED to repeat the experiment; a single pair does not establish a
general statistical advantage for either initialization law.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------- Experiment configuration ----------------------
# EX7 from the EXTERNAL problem module is EX1.1 in the supplied runner.
ALGORITHM_MODULE = "v2_single_0928"
PROBLEM_MODULE = "ProblemSpec_260416"
SPEC_NAME = "EX7"

TRAINING = dict(
    T=0.4,
    time_steps=11,                 # grid points; Delta t = T/(time_steps-1)
    training_path_size=5000,
    nn_batch_size=5000,
    num_epochs=500,
    neuron_number_v=64,
    neuron_number_w=32,
    learning_rate=5e-4,
    weight_decay=0.0,
    num_a_points=96,
    int_x_steps=40,
    norm_type="softmax",
    lp_p=8.0,
    softmax_tau="auto",
    early_stopping=True,
    es_patience=20,
    es_min_delta=1e-3,
    verbose=False,                # True prints the original training log
)

T0 = 0.0                          # fixed initial/evaluation time
FIXED_X0 = 0.1                    # Dirac mass; matches the supplied runner
X_MIN, X_MAX = -1.0, 1.0           # uniform training law AND evaluation interval
N_X = 21                         # shared evaluation grid, including endpoints
EVAL_PATHS = 5000                 # per evaluation x, per trained model
EVAL_CHUNK_SIZE = 512             # smaller values reduce recovery memory
TRAIN_SEED = None
EVAL_SEED = None                # independent of the training seed
FIG_DPI = 300
SHOW_MC_BANDS = False             # optional +/-2 recovery MC SE, not training CI


@contextmanager
def working_directory(directory):
    """Isolate the original algorithm's fixed checkpoint filenames."""
    previous = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)


# def seed_everything(seed):
#     import torch

#     random.seed(seed)
#     np.random.seed(seed % (2**32))
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)


def load_project_modules(module_dir=None):
    """Accept the original flat layout and runner/ + algorithms/ layouts."""
    here = Path(__file__).resolve().parent
    candidates = [here, here / "algorithms", here.parent / "algorithms"]
    if module_dir is not None:
        candidates.insert(0, Path(module_dir).resolve())
    for directory in reversed(candidates):
        if directory.is_dir():
            sys.path.insert(0, str(directory))
    return (importlib.import_module(ALGORITHM_MODULE),
            importlib.import_module(PROBLEM_MODULE))


def mc_batch_sizes(total, chunk_size):
    """Partition paths with no singleton batch (unbiased SE needs n >= 2)."""
    if total < 2 or chunk_size < 3:
        raise ValueError("EVAL_PATHS must be >= 2 and EVAL_CHUNK_SIZE >= 3.")
    count = math.ceil(total / chunk_size)
    quotient, remainder = divmod(total, count)
    return [quotient + (i < remainder) for i in range(count)]


def pool_mc_estimates(batches):
    """Combine (n, mean, SE) summaries into the mean and SE of all samples.

    recover_u_at_point reports unbiased sample SD / sqrt(n). Pool both
    within-batch and between-batch variation; averaging SEs is incorrect.
    """
    if not batches or any(n < 2 for n, _, _ in batches):
        raise ValueError("Each Monte Carlo batch must contain at least two paths.")
    if any(not math.isfinite(mu) or not math.isfinite(se) or se < 0
           for _, mu, se in batches):
        raise FloatingPointError("Recovery returned a non-finite estimate or invalid SE.")
    total = sum(n for n, _, _ in batches)
    mean = sum(n * mu for n, mu, _ in batches) / total
    m2 = sum((n - 1) * n * se**2 + n * (mu - mean)**2
             for n, mu, se in batches)
    return mean, math.sqrt(max(m2, 0.0) / (total * (total - 1)))


def recover_curve(algorithm, spec, w_model, meta, x_grid, n_paths, chunk_size):
    """Same per-x seed in each arm gives paired evaluation Brownian samples."""
    means, standard_errors = [], []
    sizes = mc_batch_sizes(n_paths, chunk_size)
    for index, x in enumerate(x_grid):
        #seed_everything(EVAL_SEED + index)
        batches = []
        for count in sizes:
            mean, se = algorithm.recover_u_at_point(
                spec, w_model, t0=T0, x0=float(x),
                T=meta["T"], time_steps=meta["time_steps"],
                A_min=meta["A_min"], A_max=meta["A_max"],
                num_a_points=meta["num_a_points"], lambda_reg=meta["lambda_reg"],
                n_paths=count,
            )
            batches.append((count, mean, se))
        mean, se = pool_mc_estimates(batches)
        means.append(mean)
        standard_errors.append(se)
        print(f"  recovery x={x:+.3f}: u_hat={mean:.8f}, MC SE={se:.3e}", flush=True)
    return np.asarray(means), np.asarray(standard_errors)


def train_and_evaluate(algorithm, spec, law, training, x_grid,
                       n_paths, chunk_size, output_dir):
    import torch

    arm_dir = output_dir / law
    arm_dir.mkdir()
    initial_law = (("fixed", {"x0": FIXED_X0}) if law == "dirac" else
                   ("uniform", {"x_min": X_MIN, "x_max": X_MAX}))
    # Seeding immediately before model construction pairs initial weights.
    #seed_everything(TRAIN_SEED)
    print(f"\nTraining {law}: x0_mode={initial_law}", flush=True)
    start = time.perf_counter()
    with working_directory(arm_dir):
        v_model, w_model, meta = algorithm.train_vanilla_with_norm(
            spec, **training,
            A_min=spec.A_min, A_max=spec.A_max, lambda_reg=spec.lambda_reg,
            x0_init=FIXED_X0, x0_mode=initial_law,
        )
    if algorithm.device.type == "cuda":
        torch.cuda.synchronize(algorithm.device)
    training_seconds = time.perf_counter() - start
    if meta["T"] != training["T"] or meta["time_steps"] != training["time_steps"]:
        raise ValueError("Training metadata does not match the requested time grid.")

    # Save the exact final networks used below, separately from 'best' files.
    for name, model in (("v", v_model), ("w", w_model)):
        model.eval()
        torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()},
                   arm_dir / f"final_{name}_model.pth")
    record = dict(initial_law=initial_law, training_meta=meta,
                  training_seconds=training_seconds, checkpoint_used="final")
    (arm_dir / "training.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"{law}: epochs={meta['epochs_run']}, training={training_seconds:.2f}s",
          flush=True)
    start = time.perf_counter()
    mean, se = recover_curve(algorithm, spec, w_model, meta, x_grid, n_paths, chunk_size)
    record["recovery_seconds"] = time.perf_counter() - start
    del v_model, w_model, model
    if algorithm.device.type == "cuda":
        torch.cuda.empty_cache()
    return dict(mean=mean, se=se, record=record)


def save_comparison(output_dir, x_grid, truth, results, smoke_test=False):
    if not np.all(np.isfinite(truth)):
        raise ValueError("spec.u_true_np must provide a finite exact value at every grid point.")
    denominator = np.maximum(np.abs(truth), 1e-12)
    metrics = {}
    columns = [x_grid, truth]
    names = ["x", "u_true"]
    for law in ("dirac", "uniform"):
        mean, se = results[law]["mean"], results[law]["se"]
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(se)):
            raise FloatingPointError(f"Invalid recovery values for {law}.")
        absolute_error = np.abs(mean - truth)
        relative_error = absolute_error / denominator
        metrics[law] = dict(
            mean_grid_relative_error=float(relative_error.mean()),
            max_grid_relative_error=float(relative_error.max()),
            grid_rmse=float(np.sqrt(np.mean(absolute_error**2))),
            max_recovery_mc_se=float(np.max(se)),
        )
        columns.extend([mean, se, absolute_error, relative_error])
        names.extend([f"u_{law}", f"mc_se_{law}", f"abs_error_{law}", f"rel_error_{law}"])
    np.savetxt(output_dir / "u_curve_initial_law_comparison.csv", np.column_stack(columns),
               delimiter=",", header=",".join(names), comments="", fmt="%.12e")

    styles = {
        "dirac": dict(color="#D55E00", linestyle="-.",
                      label=rf"$\widehat u_{{\mathrm{{Dirac}}}}$: $X_0={FIXED_X0:g}$"),
        "uniform": dict(color="#0072B2", linestyle="-",
                        label=rf"$\widehat u_{{\mathrm{{Unif}}}}$: $X_0\sim\mathrm{{Unif}}[{X_MIN:g},{X_MAX:g}]$"),
    }
    with plt.rc_context({"font.size": 12, "axes.titlesize": 14, "legend.fontsize": 10.5}):
        fig, (value_ax, error_ax) = plt.subplots(1, 2, figsize=(12.6, 4.8),
                                                constrained_layout=True)
        value_ax.plot(x_grid, truth, "k--", linewidth=2.1, label=r"$u^*(t_0,x)$", zorder=4)
        for law in ("dirac", "uniform"):
            mean, se = results[law]["mean"], results[law]["se"]
            value_ax.plot(x_grid, mean, linewidth=2.0, **styles[law])
            error_ax.plot(x_grid, np.abs(mean - truth) / denominator,
                          linewidth=2.0, **styles[law])
            if SHOW_MC_BANDS:
                value_ax.fill_between(x_grid, mean - 2 * se, mean + 2 * se,
                                      color=styles[law]["color"], alpha=0.13)
        value_ax.set(title=rf"Value functions at $t_0={T0:g}$", ylabel=r"$u(t_0,x)$")
        error_ax.set(title="Relative error",
                     ylabel=r"$|\widehat u-u^*|/|u^*|$")
        error_ax.set_ylim(bottom=0)
        for ax in (value_ax, error_ax):
            ax.set_xlabel(r"$x$")
            ax.set_xlim(X_MIN, X_MAX)
            ax.grid(True, alpha=0.22)
            ax.legend(loc="best", framealpha=0.95)

            # Mark the fixed training initial state on the horizontal axis.
            ax.plot(
                [FIXED_X0], [0],
                marker="^",
                markersize=7,
                color="black",
                linestyle="none",
                transform=ax.get_xaxis_transform(),
                clip_on=False,
                zorder=6,
            )
            ax.annotate(
                rf"$X_0={FIXED_X0:g}$",
                xy=(FIXED_X0, 0),
                xycoords=ax.get_xaxis_transform(),
                xytext=(6, -30),
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=10,
                annotation_clip=False,
            )
        if smoke_test:
            fig.suptitle("Smoke test only: reduced training and Monte Carlo budgets", fontsize=12)
        elif SHOW_MC_BANDS:
            fig.suptitle("Shading: +/-2 recovery Monte Carlo SE (conditional on trained models)",
                         fontsize=10)
        for extension in ("png", "pdf"):
            fig.savefig(output_dir / f"u_curve_initial_law_comparison.{extension}", dpi=FIG_DPI)
        plt.close(fig)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--module-dir", type=Path, help="Folder containing the algorithm/spec modules.")
    parser.add_argument("--output-root", type=Path, default=Path("initial_law_comparison_results"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-early-stopping", action="store_true",
                        help="Run both trainings for exactly num_epochs epochs.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Use tiny budgets to check execution; not a paper experiment.")
    args = parser.parse_args()

    import torch

    algorithm, problem_module = load_project_modules(args.module_dir)
    chosen_device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if chosen_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu or --device auto.")
    # The supplied algorithm uses a module-global device, not a function argument.
    algorithm.device = torch.device(chosen_device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    training = TRAINING.copy()
    n_x, n_paths, chunk_size = N_X, EVAL_PATHS, EVAL_CHUNK_SIZE
    if args.no_early_stopping:
        training["early_stopping"] = False
    if args.smoke_test:
        training.update(time_steps=4, training_path_size=16, nn_batch_size=8,
                        num_epochs=2, neuron_number_v=8, neuron_number_w=8,
                        num_a_points=8, int_x_steps=4, early_stopping=False, verbose=True)
        n_x, n_paths, chunk_size = 5, 24, 8
    if T0 != 0.0:
        raise ValueError("This comparison trains from fixed t0=0; keep T0=0.0.")
    if not X_MIN < X_MAX or not X_MIN <= FIXED_X0 <= X_MAX:
        raise ValueError("Require X_MIN < X_MAX and FIXED_X0 in the comparison interval.")
    if training["T"] <= 0 or training["time_steps"] < 2 or n_x < 2:
        raise ValueError("Require T > 0, at least two time points and at least two evaluation x values.")
    if training["training_path_size"] % training["nn_batch_size"] != 0:
        raise ValueError("Use train paths divisible by batch size for the supplied trainer's loss averaging.")
    mc_batch_sizes(n_paths, chunk_size)

    # Pass T to the constructor as well as training/recovery. The original
    # plotting helper used spec.T=0.1 even when the runner trained with T=0.4.
    spec_constructor = getattr(problem_module, SPEC_NAME)
    specs = {law: spec_constructor(T=training["T"]) for law in ("dirac", "uniform")}
    x_grid = np.linspace(X_MIN, X_MAX, n_x)
    truth = np.array([specs["dirac"].u_true_np(T0, float(x)) for x in x_grid], dtype=float)
    if not np.all(np.isfinite(truth)):
        raise ValueError("This runner requires an exact u_true_np for the chosen example.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    prefix = "smoke" if args.smoke_test else "comparison"
    output_dir = args.output_root.resolve() / f"{prefix}_seed{TRAIN_SEED}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    sources = {}
    for name, module in (("algorithm", algorithm), ("problem", problem_module)):
        path = Path(module.__file__).resolve()
        sources[name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    configuration = dict(
        example=SPEC_NAME, device=str(algorithm.device), training=training,
        A_min=specs["dirac"].A_min, A_max=specs["dirac"].A_max,
        lambda_reg=specs["dirac"].lambda_reg,
        t0=T0, fixed_x0=FIXED_X0, uniform_interval=[X_MIN, X_MAX],
        evaluation_grid_points=n_x, evaluation_paths_per_x=n_paths,
        evaluation_chunk_size=chunk_size, train_seed=TRAIN_SEED, evaluation_seed=EVAL_SEED,
        paired_evaluation_noise=True, checkpoint_used="final", show_mc_bands=SHOW_MC_BANDS,
        smoke_test=args.smoke_test, sources=sources,
        versions=dict(python=sys.version.split()[0], torch=str(torch.__version__),
                      numpy=np.__version__, matplotlib=matplotlib.__version__),
    )
    (output_dir / "configuration.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    print(f"EX1.1 initial-law comparison on {algorithm.device}; output: {output_dir}", flush=True)
    print(f"T={training['T']}, Delta t={training['T'] / (training['time_steps'] - 1):g}, "
          f"evaluation t0={T0}; {n_paths} recovery paths per x per law.", flush=True)
    results = {}
    for law in ("dirac", "uniform"):
        results[law] = train_and_evaluate(algorithm, specs[law], law, training, x_grid,
                                         n_paths, chunk_size, output_dir)
        # Preserve the first curve even if the second training is interrupted.
        np.savetxt(output_dir / law / "recovered_curve.csv",
                   np.column_stack([x_grid, results[law]["mean"], results[law]["se"]]),
                   delimiter=",", header="x,u_hat,mc_se", comments="", fmt="%.12e")
    metrics = save_comparison(output_dir, x_grid, truth, results, args.smoke_test)
    summary = {law: dict(**results[law]["record"], **metrics[law]) for law in results}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    for law in ("dirac", "uniform"):
        print(f"{law:7s}: mean grid relative error={metrics[law]['mean_grid_relative_error']:.4%}, "
              f"max={metrics[law]['max_grid_relative_error']:.4%}, "
              f"RMSE={metrics[law]['grid_rmse']:.6g}")
    print(f"\nSaved PNG, PDF, CSV, configuration, metrics and checkpoints in:\n{output_dir}")


if __name__ == "__main__":
    main()
