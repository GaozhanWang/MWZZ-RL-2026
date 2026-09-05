# MWZZ-RL-2025

## Learning to Solve Stochastic Controls when Drifts and Running Rewards are Unknown: Theory, Algorithms and Convergence

This repository contains the numerical algorithms and experiment runners accompanying our research on reinforcement learning for continuous-time stochastic control with model uncertainty.

We study continuous-time, possibly high-dimensional stochastic control problems in which the drift coefficients and running reward functions are unknown. Following the exploratory reinforcement-learning framework of Wang, Zariphopoulou, and Zhou, controls are relaxed into probability distributions and exploration is encouraged through entropy regularization. Our objective is to develop theoretically grounded, efficient, and scalable algorithms that learn both:

- the optimal value function, which solves the exploratory Hamilton–Jacobi–Bellman equation; 
- the optimal exploratory feedback-control policy.

The repository includes one-dimensional and high-dimensional manufactured examples with analytical reference solutions, multi-period implementations, a model-based policy-iteration benchmark, and runner scripts configured to reproduce the numerical experiments reported in the paper.

> **Research-code status.** This repository is intended to support the accompanying paper and reproduce its numerical results. It is not currently distributed as an installable Python package.

## Repository structure

```text
MWZZ-RL-2025/
├── algorithms/
│   ├── Algorithm1_single.py
│   ├── Algorithm2_multi.py
│   ├── Algorithm3_volcontrol.py
│   ├── Algorithm_model_based_pia.py
│   ├── ProblemSpec.py
│   └── ProblemSpecHD.py
├── runners/
│   ├── EX1.1_*.py
│   ├── EX1.2_*.py
│   └── EX1_vol_TEST.py
└── README.md
```

### Algorithms

| File | Description | Main entry points |
| --- | --- | --- |
| `algorithms/Algorithm1_single.py` | Single-period model-free algorithm for uncontrolled diffusion. It jointly trains the value-gradient network $v$ and local Hamiltonian network $w$, then recovers $u$ by a Feynman–Kac representation. | `train_vanilla_with_norm`, `recover_u_at_point` |
| `algorithms/Algorithm2_multi.py` | Multi-period extension using backward training and network stitching across time segments. Initial states for later segments are sampled from empirical reference-process laws. | `train_single_segment`, `train_multi_period`, `recover_u_multi` |
| `algorithms/Algorithm3_volcontrol.py` | One-dimensional, infinite-horizon controlled-diffusion algorithm. It jointly learns $u$, $u_x$, $w$, and $u_{xx}$. | `train_vol1d`, `quick_diagnostics` |
| `algorithms/Algorithm_model_based_pia.py` | Model-based policy-iteration benchmark used for comparison with the proposed model-free method. | `train_model_based_PIA`, `probe_u_and_v` |

Both single- and multi-period implementations support $L^p$ and softmax/log-sum-exp surrogates for the temporal supremum norm, neural-network and optimization hyperparameters, early stopping, action-grid integration, and Monte Carlo value recovery.

### Problem specifications

`algorithms/ProblemSpec.py` contains the one-dimensional problem interface and numerical examples. Active specifications include manufactured examples with analytical solutions and finite-difference benchmarks, including `EX5`, `EX7`, `EX7_FDM`, `EXTrue`, `EXTrue_compare`, `EXTrue2`, and `EXFDM1`–`EXFDM3`.

`algorithms/ProblemSpecHD.py` contains the multidimensional interface and the examples:

- `EX4HighDim`: diagonal square-root diffusion;
- `EX5HighDim`: a nonconstant, uniformly nondegenerate diffusion with one active state direction; and
- `EX7HighDim`: an identity-diffusion manufactured example.

The controlled-diffusion specifications `Vol1DSpec`, `Vol1DScaledSpec`, and `Vol1DScaledSpec2` are defined in `Algorithm3_volcontrol.py`.

To add a new one-dimensional benchmark, subclass `ProblemSpec` and implement the terminal reward and derivative, controlled drift, diffusion and its derivative, and running reward. The analytical value and gradient methods are optional unless the example is used for error evaluation. High-dimensional specifications similarly implement the diffusion matrix, inverse, and Jacobian through the `ProblemSpecD` interface.

## Numerical experiments

The runner filenames encode the paper example, algorithm or comparator, horizon, and dimension. Hyperparameters used for the reported tables and figures are declared near the top of each runner.

### Example 1.1 (`EX7`)

| Experiment | Runner files |
| --- | --- |
| Single-period horizon study | `EX1.1_TEST_T04.py`, `EX1.1_TEST_T05.py`, `EX1.1_TEST_T06.py`, `EX1.1_TEST_T08.py` |
| Dimension study | `EX1.1_TEST_d1.py`, `EX1.1_TEST_d5.py`, `EX1.1_TEST_d10.py`, `EX1.1_TEST_d20.py`, `EX1.1_TEST_d50.py`, `EX1.1_TEST_d100.py` |
| Figures and policy/value comparisons | `EX1.1_TEST_d1_plot.py` |
| Multi-period study | `EX1.1_MULTI_T06.py`, `EX1.1_MULTI_T08.py` |
| Model-based PIA benchmark | `EX1.1_PIA_TEST.py` |
| Deep-BSDE benchmark | `EX1.1_BSDE_TEST_d5.py`, `EX1.1_BSDE_TEST_d10.py`, `EX1.1_BSDE_TEST_d20.py`, `EX1.1_BSDE_TEST_d50.py`, `EX1.1_BSDE_TEST_d100.py` |

### Example 1.2 (`EX5`)

| Experiment | Runner files |
| --- | --- |
| Single-period horizon study | `EX1.2_TEST_T01.py`, `EX1.2_TEST_T02.py`, `EX1.2_TEST_T03.py` |
| Multi-period study | `EX1.2_MULTI_T02.py`, `EX1.2_MULTI_T03.py` |
| Model-based PIA benchmark | `EX1.2_PIA_TEST.py` |

### Controlled diffusion

`runners/EX1_vol_TEST.py` runs the controlled-volatility experiment, evaluates the learned value, gradient, and Hessian networks, and reports errors across repeated trials.

## Installation

Python 3.10 or newer is recommended. A CUDA-capable GPU is strongly recommended for the full paper configurations, especially the high-dimensional experiments and repeated-trial studies.

```bash
git clone https://github.com/GaozhanWang/MWZZ-RL-2025.git
cd MWZZ-RL-2025

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy pandas matplotlib torch openpyxl
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Install the PyTorch build appropriate for your operating system and CUDA version when GPU acceleration is required. `openpyxl` is optional and is used only for Excel output in the controlled-diffusion runner.

## Runner/module compatibility

Several runners retain historical development filenames in their import statements. The corresponding published repository filenames are:

| Historical import | Repository module |
| --- | --- |
| `v2_single_0928` | `algorithms.Algorithm1_single` |
| `v2_multi_0925` | `algorithms.Algorithm2_multi` |
| `v1_volctrl_1129` | `algorithms.Algorithm3_volcontrol` |
| `v1_pia_0924` | `algorithms.Algorithm_model_based_pia` |
| `ProblemSpec_0928` or `ProblemSpec_260416` | `algorithms.ProblemSpec` |
| `ProblemSpecHD_0925` | `algorithms.ProblemSpecHD` |

For example, a clean-clone runner should import the single-period implementation as

```python
from algorithms.Algorithm1_single import (
    train_vanilla_with_norm,
    recover_u_at_point,
)
from algorithms.ProblemSpec import EX7
```

The high-dimensional runners additionally reference `v2_highdim_0924`, and the Deep-BSDE runners reference `v1_deep_bsde_1005`. Those two trainer modules are not included in the current repository snapshot and must be added before those runner families can be executed.

## Running experiments

After aligning the imports described above, run scripts from the repository root. For example:

```bash
# One-dimensional single-period experiment
python runners/EX1.1_TEST_d1.py

# Multi-period experiment
python runners/EX1.1_MULTI_T06.py

# Model-based policy-iteration benchmark
python runners/EX1.1_PIA_TEST.py

# Controlled-diffusion experiment
python runners/EX1_vol_TEST.py
```

The full paper settings can be computationally expensive: several runners execute 30 or 50 independent trials with batch sizes of 5,000 or more. For a smoke test, temporarily reduce `N_RUNS` or `n_trials`, the training-path count, and the number of epochs in the selected runner. Restore the committed settings when reproducing the paper tables.

Each runner automatically selects CUDA when available and otherwise falls back to CPU.

## Outputs

Depending on the selected runner, execution prints per-trial estimates and summary statistics and may write:

- CSV files containing value estimates, relative errors, and runtimes;
- NumPy `.npy` files for PIA benchmark results;
- JSON files for Deep-BSDE benchmark results;
- PNG figures for value-function and policy comparisons;
- PyTorch `.pth` checkpoints for the best single-period networks; and
- an optional Excel workbook for the controlled-diffusion experiment.

Outputs are written relative to the current working directory. Running from the repository root therefore keeps generated files at the repository root unless a runner specifies another path.

## Reproducibility notes

- The committed runner configurations are the configurations used for the numerical results in the paper.
- Most experiments intentionally perform independent stochastic trials. Seed-setting blocks are included but commented out in several runners. Uncomment and record these seeds when exact run-level repeatability is required.
- Report the Python, PyTorch, CUDA, and GPU versions with reproduced results, since numerical behavior and runtime can vary across hardware and software environments.
- The analytical functions in the problem specifications are evaluation references; they are not intended to provide extra information to the learning algorithm.
- Full runs may create large checkpoints and result files. Consider directing generated artifacts to a separate results directory for systematic experiment management.



