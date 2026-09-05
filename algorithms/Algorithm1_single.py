# %%
# J-minimization with selectable norm: L^p or Softmax (log-sum-exp)
# J(v,w) = ||v - Φ(w)|| + ||w - Ψ(v)||  (both norms chosen via norm_type argument)
# - Includes Feynman–Kac recovery of u(t,x) under the uncontrolled diffusion 𝓧
# - Problem-specification class to keep training code generic

import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (used implicitly by mpl)

# # ------------------------------ Repro & device ------------------------------
# torch.manual_seed(0)
# np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==============================================================================
#                                PROBLEM SPEC
# ==============================================================================
class ProblemSpec:
    """
    All model-dependent definitions. Training code calls only into 'spec'.

    IMPORTANT:
      - For Φ and Feynman–Kac, the reference process is driftless:
            d𝓧 = σ(t,𝓧) dW,   d(∇X) = σ_x(t,𝓧) ∇X dW
        regardless of any b_unc/b_unc_x defined here (kept for extensibility).

      - Required to implement:
            g_value_torch(x,T), g_x_torch(x,T),
            b_ctrl_torch(t,x,a),
            sigma_torch(t,x), sigma_x_torch(t,x),
            running_reward_torch(t,x,a)
    """
    T: float = 1.0
    A_min: float = 0.0
    A_max: float = 1.0
    lambda_reg: float = 5.0
    x_floor: float = 1e-3

    # NEW: whether to clamp the state (e.g., EX4 needs X ≥ 0 for sqrt(x))
    clamp_state: bool = False

    # Optional truths (diagnostics)
    def u_true_np(self, t: float, x: float) -> float:
        return float('nan')

    def u_x_true_np(self, t: float, x: float) -> float:
        return float('nan')

    # Terminal value g(x) and derivative g_x(x)
    def g_value_torch(self, x: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def g_x_torch(self, x: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # Controlled drift b(t,x,a) used only in Ψ
    def b_ctrl_torch(self, t, x, a):
        raise NotImplementedError

    # (Not used by Φ/FK; kept for extensibility)
    def b_unc_torch(self, t, x):
        return torch.zeros_like(x)

    def b_unc_x_torch(self, t, x):
        return torch.zeros_like(x)

    # Diffusion σ and its derivative σ_x (needed for ∇X SDE)
    def sigma_torch(self, t, x):
        raise NotImplementedError

    def sigma_x_torch(self, t, x):
        raise NotImplementedError

    # Running reward r(t,x,a)
    def running_reward_torch(self, t, x, a):
        raise NotImplementedError


class EX4(ProblemSpec):
    """
      u(t,x)  = exp( -(t^2 + x^2 + 1) ),  g(x)=u(T,x)
      g_x     = -2x u
      b(t,x,a)= x^2 + a - 0.5
      σ(t,x)  = sqrt(x),  σ_x = 1/(2 sqrt(x))
      r(t,x,a)= (2t + 2ax) u
    """
    def __init__(self):
        self.T = 0.1
        self.A_min, self.A_max = 0.0, 1.0
        self.lambda_reg = 5.0
        self.x_floor = 1e-2
        # sqrt(x) requires X≥0 in simulation; enable clamping
        self.clamp_state = True

    def u_true_np(self, t, x):
        return float(np.exp(-(t**2 + x**2 + 1.0)))

    def u_x_true_np(self, t, x):
        return float(-2.0 * x * np.exp(-(t**2 + x**2 + 1.0)))

    @torch.no_grad()
    def g_value_torch(self, x, T):
        return torch.exp(-(x**2 + T**2 + 1.0))

    def g_x_torch(self, x, T):
        return -2.0 * x * torch.exp(-(x**2 + T**2 + 1.0))

    def b_ctrl_torch(self, t, x, a):
        return x**2 + a - 0.5

    def sigma_torch(self, t, x):
        return torch.sqrt(x.clamp(min=self.x_floor))

    def sigma_x_torch(self, t, x):
        sig = self.sigma_torch(t, x).clamp(min=1e-6)
        return 0.5 / sig  # d/dx sqrt(x) = 1/(2 sqrt(x))

    def running_reward_torch(self, t, x, a):
        u = torch.exp(-(t**2 + x**2 + 1.0))
        return (2.0 * t + 2.0 * a * x) * u



class EX5(ProblemSpec):
    """
      u(t,x)  = cos(t + x),  g(x)=u(T,x)=cos(T + x)
      g_x     = -sin(T + x)
      b(t,x,a)= -sin(t + x)cos(t + x) + a  = -0.5*sin(2t + 2x) + a
      σ(t,x)  = sin(t + x),  (implemented with sign-preserving clamp)
      σ_x     = cos(t + x) * sign(sin(t + x))   (a.e.; ignores clamp kinks)
      r(t,x,a)= (a + 1) * sin(t + x)
    """
    def __init__(self, T=0.1, A_min=0.0, A_max=1.0, lambda_reg=5.0, x_floor=1e-2):
        self.T = float(T)
        self.A_min, self.A_max = float(A_min), float(A_max)
        self.lambda_reg = float(lambda_reg)
        # Numerical floor for |σ| to avoid division-by-zero in N_s^r and explosions in ∇X.
        self.x_floor = float(x_floor)
        # No need to clamp the state for sin-diffusion
        self.clamp_state = False

    # ----- ground-truth (for diagnostics only) -----
    def u_true_np(self, t, x):
        return float(np.cos(t + x))

    def u_x_true_np(self, t, x):
        return float(-np.sin(t + x))

    # ----- terminal value g and its x-derivative -----
    @torch.no_grad()
    def g_value_torch(self, x: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        return torch.cos(T + x)

    def g_x_torch(self, x: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        return -torch.sin(T + x)

    # ----- controlled drift b(t,x,a) used in Ψ one-step move -----
    def b_ctrl_torch(self, t, x, a):
        s = t + x
        return -0.5*(torch.sin(s) * torch.cos(s)) + a  # = -0.25 * sin(2s) + a

    # ----- diffusion and its x-derivative for the Ma–Zhang kernel -----
    def sigma_torch(self, t, x):
        """
        σ(t,x) = sin(t+x), but we use a sign-preserving clamp:
            σ = sign(sin) * max(|sin|, x_floor)
        This stabilizes σ^{-1} in N_s^r while preserving the sign of σ.
        """
        s = torch.sin(t + x)
        return s.sign() * torch.clamp(s.abs(), min=self.x_floor)

    def sigma_x_torch(self, t, x):
        """
        d/dx (sign-preserving clamp of sin) ≈ cos(t+x) * sign(sin(t+x)).
        This ignores the (measure-zero) kinks introduced by clamping near 0.
        """
        return torch.cos(t + x) * torch.sign(torch.sin(t + x))

    # ----- running reward r(t,x,a) -----
    def running_reward_torch(self, t, x, a):
        return (a + 1.0) * torch.sin(t + x)


class EX7(ProblemSpec):
    """
    Uniformly non-degenerate example (σ ≡ 1):
      b(t,x,a) = x + a
      σ(t,x)   = 1
      r(t,x,a) = (2t + 2ax + 1) * u(t,x)
      g(x)     = exp(-(T^2 + x^2 + 1))

    Manufactured solution (exact for λ=1, A=[0,1]):
      u(t,x)   = exp(-(t^2 + x^2 + 1))
      u_x      = -2x u
      u_xx     = (-2 + 4x^2) u
      u_t      = -2t u
    """
    def __init__(self, T=0.1, A_min=0.0, A_max=1.0, lambda_reg=1.0, x_floor=0.0):
        self.T = float(T)
        self.A_min, self.A_max = float(A_min), float(A_max)  # keep [0,1] for exactness
        self.lambda_reg = float(lambda_reg)                  # set 1.0 for exact PDE
        self.x_floor = float(x_floor)                        # unused here; kept for API symmetry

    # ----- ground truth (for diagnostics) -----
    def u_true_np(self, t, x):
        return float(np.exp(-(t**2 + x**2 + 1.0)))

    def u_x_true_np(self, t, x):
        return float(-2.0 * x * np.exp(-(t**2 + x**2 + 1.0)))

    # ----- terminal value g and its x-derivative -----
    @torch.no_grad()
    def g_value_torch(self, x: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        return torch.exp(-(T**2 + x**2 + 1.0))

    def g_x_torch(self, x: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        return -2.0 * x * torch.exp(-(T**2 + x**2 + 1.0))

    # ----- controlled drift b(t,x,a) used in Ψ one-step move -----
    def b_ctrl_torch(self, t, x, a):
        return x + a

    # ----- diffusion and its x-derivative (constant σ) -----
    def sigma_torch(self, t, x):
        # shape-safe: returns a tensor matching x
        return torch.ones_like(x)

    def sigma_x_torch(self, t, x):
        return torch.zeros_like(x)

    # ----- running reward r(t,x,a) -----
    def running_reward_torch(self, t, x, a):
        u = torch.exp(-(t**2 + x**2 + 1.0))
        return (2.0 * t + 2.0 * a * x + 1.0) * u




# ==============================================================================
#                                   MODELS
# ==============================================================================
def build_v_model(width, input_dim=2, output_dim=1):
    model = nn.Sequential(
        nn.Linear(input_dim, width), nn.Tanh(),
        nn.Linear(width, width),     nn.Tanh(),
        nn.Linear(width, width),     nn.Tanh(),
        nn.Linear(width, output_dim),
    )
    def init_(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight); nn.init.constant_(m.bias, 0.0)
    model.apply(init_)
    return model

def build_w_model(width, input_dim=3, output_dim=1):
    model = nn.Sequential(
        nn.Linear(input_dim, width), nn.Tanh(),
        nn.Linear(width, width),     nn.Tanh(),
        nn.Linear(width, width),     nn.Tanh(),
        nn.Linear(width, width),     nn.Tanh(),
        nn.Linear(width, output_dim),
    )
    def init_(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight); nn.init.constant_(m.bias, 0.0)
    model.apply(init_)
    return model


# ==============================================================================
#                                NORM REDUCERS
# ==============================================================================
def lp_reduce_over_time(residual_matrix, p=8.0, time_dim=1):
    """
    residual_matrix: tensor [B, K] where K is the number of time points.
    Returns: scalar = mean over batch of L^p over time.
    """
    per_sample = residual_matrix.pow(p).mean(dim=time_dim).pow(1.0 / p)  # [B]
    return per_sample.mean()

def soft_sup(r, tau, dim):
    """
    Smooth max: tau * logsumexp(r / tau, dim), stabilized by subtracting max.
    """
    m = torch.amax(r, dim=dim, keepdim=True)
    return tau * torch.logsumexp((r - m) / tau, dim=dim) + m.squeeze(dim)

def softmax_reduce_over_time(residual_matrix, tau="auto", time_dim=1):
    """
    residual_matrix: [B, K]
    tau: float or "auto"
      - "auto": tau ≈ 0.5 * median(residual) / log(K), clamped to [1e-4, 1.0]
    Returns: scalar = mean over batch of soft-sup over time.
    """
    B, K = residual_matrix.size(0), residual_matrix.size(time_dim)
    if tau == "auto":
        med = residual_matrix.detach().median().item()
        denom = max(1.0, math.log(max(2, K)))
        tau_val = max(1e-4, min(1.0, 0.5 * med / denom))
    else:
        tau_val = float(tau)
    per_sample = soft_sup(residual_matrix, tau=tau_val, dim=time_dim)  # [B]
    return per_sample.mean()

def reduce_residuals(residual_matrix, norm_type="lp", lp_p=8.0, softmax_tau="auto", time_dim=1):
    """
    Unified reducer used for both Φ and Ψ residuals.
    - residual_matrix: [B, K], K = number of time points contributing to the batch loss
    """
    if norm_type == "lp":
        return lp_reduce_over_time(residual_matrix, p=lp_p, time_dim=time_dim)
    elif norm_type == "softmax":
        return softmax_reduce_over_time(residual_matrix, tau=softmax_tau, time_dim=time_dim)
    else:
        raise ValueError(f"Unknown norm_type '{norm_type}'. Use 'lp' or 'softmax'.")


# ==============================================================================
#                             TRAINING (VANILLA)
# ==============================================================================
def train_vanilla_with_norm(
    spec: ProblemSpec,
    # grids / horizon (T default comes from spec if None)
    T=None, time_steps=21,
    # MC & epochs
    training_path_size=4000, nn_batch_size=1000, num_epochs=40,
    # nets & opt
    neuron_number_v=128, neuron_number_w=128, learning_rate=5e-4, weight_decay=0.0,
    # action & entropy (defaults come from spec if None)
    A_min=None, A_max=None, num_a_points=160, lambda_reg=None,
    # Ψ integral resolution
    int_x_steps=50,
    # norm selection
    norm_type="lp", lp_p=8.0, softmax_tau="auto",
    # probe
    x0_init=1.0,
    # NEW: choose initial x per path: fixed at x0 or uniform over [x_min, x_max]
    x0_mode: tuple[str, dict] | None = None,   # ("fixed", {"x0": ...}) or ("uniform", {"x_min": ..., "x_max": ...})
    # --- early stopping ---
    early_stopping=True, es_patience=20, es_min_delta=5e-4,
    verbose: bool = True,
):

    """
    Vanilla training of (v,w) minimizing J(v,w) with selectable norm surrogate.
    Early stopping monitors epoch-averaged J with patience/min_delta.
    Returns trained models and a small dict of metadata.
    """
    # --- resolve defaults from spec ---
    if T is None: T = spec.T
    if A_min is None: A_min = spec.A_min
    if A_max is None: A_max = spec.A_max
    if lambda_reg is None: lambda_reg = spec.lambda_reg

    vprint = print if verbose else (lambda *a, **k: None)

    # --- set up time/action grids ---
    delta_t = T / (time_steps - 1)
    t_seq  = torch.linspace(0.0, T, time_steps, device=device)
    T_t    = torch.tensor(T, dtype=torch.float32, device=device)
    dt_t   = torch.tensor(delta_t, dtype=torch.float32, device=device)
    x0_t   = torch.tensor(x0_init, dtype=torch.float32, device=device)
    a_grid = torch.linspace(A_min, A_max, num_a_points, device=device)

    # helper: clamp state only if the spec asks for it (e.g., EX4)
    def maybe_clamp_state(x):
        return x.clamp(min=spec.x_floor) if getattr(spec, "clamp_state", False) else x

    # --- helper: ∫ v(t, y) dy via trapezoid ---
    def integral_v(model, t, low, up, steps=50):
        """
        Compute ∫_{low}^{up} v(t, y) dy by trapezoid, for batch inputs.
        t: scalar or [B], low, up: [B]  -> returns [B]
        """
        B = low.size(0)
        delta = (up - low) / steps
        ar = torch.arange(0, steps + 1, device=low.device) / steps
        # y-grid
        x_grid = low.unsqueeze(1) + delta.unsqueeze(1) * ar.unsqueeze(0)  # [B, steps+1]
        if getattr(spec, "clamp_state", False):
            x_grid = x_grid.clamp(min=spec.x_floor)
        # t-grid to match shape
        if isinstance(t, torch.Tensor):
            if t.dim() == 0:
                t_grid = t * torch.ones_like(x_grid)
            elif t.dim() == 1:
                t_grid = t.unsqueeze(1).expand_as(x_grid)
            else:
                t_grid = t
        else:
            t_grid = torch.tensor(t, dtype=x_grid.dtype, device=x_grid.device) * torch.ones_like(x_grid)
        # evaluate v
        inputs = torch.stack([t_grid, x_grid], dim=2).view(-1, 2)
        v_vals = model(inputs).view(B, steps + 1)
        return delta / 2 * (v_vals[:, 0] + v_vals[:, -1] + 2 * v_vals[:, 1:-1].sum(dim=1))

    # --- build models & optimizer ---
    v_Model = build_v_model(neuron_number_v).to(device)  # approximates u_x
    w_Model = build_w_model(neuron_number_w).to(device)  # approximates w = b u_x + r

    optim = torch.optim.Adam(
        list(v_Model.parameters()) + list(w_Model.parameters()),
        lr=learning_rate, weight_decay=weight_decay
    )

    best_loss = float('inf')
    patience_ctr = 0
    stopped_early = False
    epochs_run = 0

    # --- training loop ---
    for epoch in range(num_epochs):
        epoch_loss   = 0.0
        epoch_loss_v = 0.0
        epoch_loss_w = 0.0

        for start in range(0, training_path_size, nn_batch_size):
        
            # --- initial x sampler (fixed or uniform) ---
            if x0_mode is None or x0_mode[0] == "fixed":
                x_fixed = x0_init if (x0_mode is None or x0_mode[1].get("x0") is None) else float(x0_mode[1]["x0"])
                def sample_x0(B: int) -> torch.Tensor:
                    return torch.full((B,), x_fixed, device=device, dtype=torch.float32)
            elif x0_mode[0] == "uniform":
                x_min = x0_mode[1].get("x_min", None)
                x_max = x0_mode[1].get("x_max", None)
                if x_min is None or x_max is None:
                    raise ValueError("x0_mode=('uniform', {'x_min':..., 'x_max':...}) requires both x_min and x_max.")
                x_min = float(x_min); x_max = float(x_max)
                if not (x_max > x_min):
                    raise ValueError("Require x_max > x_min for uniform x0.")
                def sample_x0(B: int) -> torch.Tensor:
                    return (x_min + (x_max - x_min) * torch.rand(B, device=device, dtype=torch.float32))
            else:
                raise ValueError("x0_mode must be None, ('fixed', {...}), or ('uniform', {...}).")

            
            B = min(nn_batch_size, training_path_size - start)
            
            # 1) Brownian increments
            dW  = torch.sqrt(dt_t) * torch.randn(B, time_steps - 1, device=device)  # [M, T-1]
            dWX = torch.sqrt(dt_t) * torch.randn(B, time_steps - 1, device=device) # Independent BM for controlled process

            # 2) Uncontrolled 𝓧
            X_unc = torch.zeros(B, time_steps, device=device)
            X_unc[:, 0] = sample_x0(B) 
            for i in range(1, time_steps):
                t_cur = t_seq[i - 1]
                sig_prev = spec.sigma_torch(t_cur, X_unc[:, i - 1])
                X_unc[:, i] = maybe_clamp_state(X_unc[:, i - 1] + sig_prev * dW[:, i - 1])  

            # 3) Controlled X^a (one action per path for Ψ)
            a_batch = A_min + (A_max - A_min) * torch.rand(B, device=device)  # [M]
            X_ctrl = torch.zeros(B, time_steps, device=device)
            X_ctrl[:, 0] = sample_x0(B)
            for i in range(1, time_steps):
                t_cur = t_seq[i - 1]
                drift_inc = spec.b_ctrl_torch(t_cur, X_ctrl[:, i - 1], a_batch) * dt_t
                sig_prev  = spec.sigma_torch(t_cur, X_ctrl[:, i - 1])
                X_ctrl[:, i] = maybe_clamp_state(X_ctrl[:, i - 1] + drift_inc + sig_prev * dWX[:, i - 1]) 

            # 4) Jacobian ∇X on 𝓧 and M (for Φ kernel)
            nabla = torch.zeros(B, time_steps, device=device); nabla[:, 0] = 1.0
            for i in range(1, time_steps):
                t_cur  = t_seq[i - 1]
                X_prev = X_unc[:, i - 1]
                sig_prev = spec.sigma_torch(t_cur, X_prev).clamp(min=1e-6)
                sigma_x  = spec.sigma_x_torch(t_cur, X_prev)      # general σ_x
                expo     = sigma_x * dW[:, i - 1] - 0.5 * (sigma_x**2) * dt_t
                nabla[:, i] = (nabla[:, i - 1] * torch.exp(expo)).clamp(min=1e-12)

            sigma_all = spec.sigma_torch(t_seq.unsqueeze(0).repeat(B, 1), X_unc).clamp(min=1e-6)
            eta = (1.0 / sigma_all) * nabla                    # [B, T]
            M   = torch.cumsum(eta[:, :-1] * dW, dim=1)        # [B, T-1]

            # 5) Φ: λ ln ∫ exp(w/λ) da along 𝓧
            x_vals = X_unc[:, 1:]                              # [B, T-1]
            t_vals = t_seq[1:].unsqueeze(0).repeat(B, 1)       # [B, T-1]
            t_rep = t_vals.unsqueeze(2).repeat(1, 1, num_a_points)
            x_rep = x_vals.unsqueeze(2).repeat(1, 1, num_a_points)
            a_rep = torch.linspace(A_min, A_max, num_a_points, device=device).view(1, 1, -1).repeat(B, t_vals.shape[1], 1)

            inp_w_grid = torch.stack([t_rep, x_rep, a_rep], dim=3).view(-1, 3)
            w_out_grid = w_Model(inp_w_grid).view(B, t_vals.shape[1], num_a_points)  # [B, T-1, A]
            ln_integral = lambda_reg * torch.log(
                torch.trapz(torch.exp(w_out_grid / lambda_reg),
                            torch.linspace(A_min, A_max, num_a_points, device=device),
                            dim=2) + 1e-12
            )  # [B, T-1]

            terminal_full = spec.g_x_torch(X_unc[:, -1], T_t) * nabla[:, -1]  # [B]

            # 6) Build Φ-residual matrix R_v: [B, K] over time k=0..T-2
            Rv_list = []
            num_points = time_steps - 1
            for k in range(num_points):
                num_future = time_steps - 1 - k
                if num_future <= 0:
                    continue

                t_k = t_seq[k]
                dt_future = (t_seq[k+1 : k+1+num_future] - t_k)          # [num_future]
                M_tk = torch.zeros(B, device=device) if k == 0 else M[:, k-1]
                M_future = M[:, k : k+num_future]                         # [B, num_future]
                nabla_k = nabla[:, k].clamp(min=1e-12)                    # [B]

                kernel = (M_future - M_tk.unsqueeze(1)) / (dt_future.unsqueeze(0) * nabla_k.unsqueeze(1))  # [B, num_future]
                integral_w_k = (ln_integral[:, k : k+num_future] * kernel * dt_t).sum(dim=1)               # [B]
                phi_k = integral_w_k + (terminal_full / nabla_k)                                           # [B]

                x_k = X_unc[:, k]
                v_in = torch.stack((t_k * torch.ones(B, device=device), x_k), dim=1)
                v_out = v_Model(v_in).squeeze()                                                             # [B]

                Rv_list.append((v_out - phi_k).abs())  # [B]

            Rv = torch.stack(Rv_list, dim=0).transpose(0, 1) if len(Rv_list) > 0 else torch.zeros(B, 1, device=device)
            loss_v_batch = reduce_residuals(Rv, norm_type=norm_type, lp_p=lp_p, softmax_tau=softmax_tau, time_dim=1)

            # 7) Build Ψ-residual matrix R_w: [B, K] over time k=0..T-2 (local one-step)
            Rw_list = []
            for k in range(num_points):
                t_k  = t_seq[k]
                cur_x = X_unc[:, k]                                  # [B]
                sig_k = spec.sigma_torch(t_k, cur_x)                 # [B]
                dW_k  = dW[:, k]                                     # [B]

                next_c = maybe_clamp_state(cur_x + sig_k * dW_k)  # uncontrolled one-step
                drift_k = spec.b_ctrl_torch(t_k, cur_x, a_batch) * dt_t
                next_X  = maybe_clamp_state(cur_x + drift_k + sig_k * dW_k)   # controlled one-step

                lo_c = torch.minimum(cur_x, next_c); up_c = torch.maximum(cur_x, next_c)
                lo_X = torch.minimum(cur_x, next_X); up_X = torch.maximum(cur_x, next_X)

                int_c = integral_v(v_Model, t_k, lo_c, up_c, steps=int_x_steps) * torch.sign(next_c - cur_x)
                int_X = integral_v(v_Model, t_k, lo_X, up_X, steps=int_x_steps) * torch.sign(next_X - cur_x)

                reward_k = spec.running_reward_torch(t_k, next_X, a_batch) * dt_t
                psi_k = (int_X - int_c + reward_k) / dt_t

                w_in = torch.stack((t_k * torch.ones(B, device=device), cur_x, a_batch), dim=1)
                w_out = w_Model(w_in).squeeze()

                Rw_list.append((w_out - psi_k).abs())  # [B]

            Rw = torch.stack(Rw_list, dim=0).transpose(0, 1) if len(Rw_list) > 0 else torch.zeros(B, 1, device=device)
            loss_w_batch = reduce_residuals(Rw, norm_type=norm_type, lp_p=lp_p, softmax_tau=softmax_tau, time_dim=1)

            # 8) Joint loss and update
            J_batch = loss_v_batch + loss_w_batch

            optim.zero_grad()
            J_batch.backward()
            optim.step()

            epoch_loss   += J_batch.item()
            epoch_loss_v += loss_v_batch.item()
            epoch_loss_w += loss_w_batch.item()

        # epoch averages
        denom = training_path_size / nn_batch_size
        avg_epoch_loss   = epoch_loss   / denom
        avg_epoch_loss_v = epoch_loss_v / denom
        avg_epoch_loss_w = epoch_loss_w / denom
        epochs_run = epoch + 1

        vprint(f"Epoch {epoch+1}/{num_epochs} | J: {avg_epoch_loss:.6f} | Lv: {avg_epoch_loss_v:.6f} | Lw: {avg_epoch_loss_w:.6f} | norm={norm_type}")

        # save best + early stopping check
        if avg_epoch_loss < best_loss - es_min_delta:
            best_loss = avg_epoch_loss
            patience_ctr = 0
            torch.save(v_Model.state_dict(), f'best_v_model_{norm_type}.pth')
            torch.save(w_Model.state_dict(), f'best_w_model_{norm_type}.pth')
        else:
            patience_ctr += 1
            if early_stopping and patience_ctr >= es_patience:
                vprint(f"[Early stop] No improvement ≥ {es_min_delta} for {es_patience} consecutive epochs.")
                stopped_early = True
                break

    meta = {
        "T": T, "time_steps": time_steps,
        "norm_type": norm_type, "lp_p": lp_p, "softmax_tau": softmax_tau,
        "lambda_reg": lambda_reg, "num_a_points": num_a_points, "int_x_steps": int_x_steps,
        "A_min": A_min, "A_max": A_max,
        # early stopping meta
        "early_stopping": early_stopping, "es_patience": es_patience, "es_min_delta": es_min_delta,
        "stopped_early": stopped_early, "epochs_run": epochs_run,
    }
    return v_Model, w_Model, meta



# ==============================================================================
#                    FEYNMAN–KAC RECOVERY OF u(t,x)  (generic)
# ==============================================================================

@torch.no_grad()
def recover_u_at_point(
    spec: ProblemSpec, w_model,
    t0: float, x0,                                 # x0 can be float, or ignored when x0_mode is interval
    T=None, time_steps=21,
    A_min=None, A_max=None, num_a_points=160, lambda_reg=None,
    n_paths: int = 10000,
    # NEW: optional interval mode to evaluate u(t0, x) for many x at once
    x0_mode: tuple[str, dict] | None = None,       # ("interval", {"x_min":..., "x_max":..., "n_points": M})
):
    """
    Estimate u under the uncontrolled diffusion 𝓧.

    Scalar mode (backward compatible):
      Input:  x0 is a float
      Return: (u_hat: float, u_se: float)

    Interval mode:
      Input:  x0_mode=("interval", {"x_min":..., "x_max":..., "n_points": M})
      Return: (x_grid: [M], u_mean: [M], u_se: [M])
    """
    def maybe_clamp_state(x):
        return x.clamp(min=spec.x_floor) if getattr(spec, "clamp_state", False) else x

    # defaults
    if T is None: T = spec.T
    if A_min is None: A_min = spec.A_min
    if A_max is None: A_max = spec.A_max
    if lambda_reg is None: lambda_reg = spec.lambda_reg

    # time & action grids
    delta_t = T / (time_steps - 1)
    t_seq  = torch.linspace(0.0, T, time_steps, device=device)
    T_t    = torch.tensor(T, dtype=torch.float32, device=device)
    dt_t   = torch.tensor(delta_t, dtype=torch.float32, device=device)
    a_grid = torch.linspace(A_min, A_max, num_a_points, device=device)

    # identify interval vs scalar
    interval = (x0_mode is not None and x0_mode[0] == "interval")
    if interval:
        x_min = float(x0_mode[1]["x_min"]); x_max = float(x0_mode[1]["x_max"])
        M = int(x0_mode[1].get("n_points", 50))
        if not (x_max > x_min): raise ValueError("x_max must be > x_min in x0_mode interval")
        x_grid = torch.linspace(x_min, x_max, M, device=device)
    else:
        # scalar mode
        x0 = float(x0)
        x_grid = torch.tensor([x0], dtype=torch.float32, device=device)
        M = 1

    # nearest grid index to t0
    k0 = int(round(t0 / float(delta_t)))
    k0 = max(0, min(k0, time_steps - 1))
    steps_left = time_steps - 1 - k0

    if steps_left <= 0:
        # already at terminal
        if interval:
            g_vals = spec.g_value_torch(x_grid, T_t)             # [M]
            u_mean = g_vals.detach().cpu().numpy()
            u_se   = np.zeros_like(u_mean)
            return x_grid.detach().cpu().numpy(), u_mean, u_se
        else:
            X_T = torch.tensor([x0], device=device, dtype=torch.float32)
            u_hat = spec.g_value_torch(X_T, T_t).mean().item()
            return u_hat, 0.0

    # Vectorized simulation over all x in the grid at once:
    # create M * n_paths paths by repeating each x_j, n_paths times
    N = M * n_paths
    # dW: [N, steps_left]
    dW = torch.sqrt(dt_t) * torch.randn(N, steps_left, device=device)
    # X: [N, steps_left+1]
    X  = torch.zeros(N, steps_left + 1, device=device)
    # seed initial conditions
    X0 = x_grid.repeat_interleave(n_paths)  # [N]
    X[:, 0] = X0

    # simulate uncontrolled 𝓧
    for i in range(1, steps_left + 1):
        t_prev = t_seq[k0 + i - 1]
        sig_prev = spec.sigma_torch(t_prev, X[:, i - 1])
        X[:, i] = maybe_clamp_state(X[:, i - 1] + sig_prev * dW[:, i - 1])

    # Compute H via w
    x_vals = X[:, 1:]                                        # [N, steps_left]
    t_vals = t_seq[k0 + 1:].unsqueeze(0).repeat(N, 1)        # [N, steps_left]

    t_rep = t_vals.unsqueeze(2).repeat(1, 1, num_a_points)   # [N, L, A]
    x_rep = x_vals.unsqueeze(2).repeat(1, 1, num_a_points)   # [N, L, A]
    a_rep = a_grid.view(1, 1, -1).repeat(N, t_vals.shape[1], 1)

    inp   = torch.stack([t_rep, x_rep, a_rep], dim=3).view(-1, 3)
    w_out = w_model(inp).view(N, t_vals.shape[1], num_a_points)

    # stabilized log ∫ exp(w/λ) da
    S = w_out / lambda_reg
    Smax, _ = torch.max(S, dim=2, keepdim=True)
    int_exp = torch.trapz(torch.exp(S - Smax), a_grid, dim=2).clamp_min(1e-40)  # [N, L]
    log_int = torch.log(int_exp) + Smax.squeeze(2)                               # [N, L]
    H_vals  = lambda_reg * log_int                                               # [N, L]
    integral_term = (H_vals * dt_t).sum(dim=1)                                   # [N]

    g_term = spec.g_value_torch(X[:, -1], T_t)                                   # [N]
    u_samples = g_term + integral_term                                           # [N]

    # Aggregate per x_j (block of size n_paths)
    u_samples = u_samples.view(M, n_paths)
    u_mean = u_samples.mean(dim=1).detach().cpu().numpy()                        # [M]
    u_se   = (u_samples.std(unbiased=True, dim=1) / math.sqrt(n_paths)).detach().cpu().numpy()

    if interval:
        return x_grid.detach().cpu().numpy(), u_mean, u_se
    else:
        return float(u_mean[0]), float(u_se[0])


def plot_u_over_interval(
    spec: ProblemSpec, w_model,
    t0: float, x_min: float, x_max: float,
    *, n_points: int = 101,
    T=None, time_steps=21,
    A_min=None, A_max=None, num_a_points=160, lambda_reg=None,
    n_paths: int = 5000,
    show_true: bool = True
):
    import matplotlib.pyplot as plt
    x_grid, u_mean, u_se = recover_u_at_point(
        spec, w_model, t0, x0=0.0,  # ignored in interval mode
        T=T, time_steps=time_steps,
        A_min=A_min, A_max=A_max, num_a_points=num_a_points, lambda_reg=lambda_reg,
        n_paths=n_paths,
        x0_mode=("interval", {"x_min": x_min, "x_max": x_max, "n_points": n_points})
    )

    plt.figure(figsize=(6,4))
    plt.plot(x_grid, u_mean, label=r"$\hat u^\lambda(t_0,x)$")
    plt.fill_between(x_grid, u_mean - u_se, u_mean + u_se, alpha=0.2, label="±1 SE")

    if show_true:
        # draw true curve when available
        u_true_vals = np.array([spec.u_true_np(t0, float(x)) for x in x_grid], dtype=float)
        if not np.all(np.isnan(u_true_vals)):
            plt.plot(x_grid, u_true_vals, linestyle="--", label=r"$u(t_0,x)$ (true)")

    plt.xlabel("x")
    plt.ylabel("u")
    plt.title(fr"$u(t_0={t0}, x)$ over [{x_min}, {x_max}]")
    plt.legend()
    plt.tight_layout()
    plt.show()

@torch.no_grad()
def plot_u_and_relerror_surface(
    spec,
    w_model,
    t_min: float,
    t_max: float,
    n_t: int = 11,
    x_min: float = 0.0,
    x_max: float = 1.0,
    n_x: int = 41,
    # per-x MC paths (we create N = n_x * n_paths samples per t)
    n_paths: int = 2000,
    time_steps: int | None = None,
    A_min: float | None = None,
    A_max: float | None = None,
    num_a_points: int | None = None,
    lambda_reg: float | None = None,
    show_true: bool = True,
    cmap_u: str = 'viridis',
    cmap_err: str = 'inferno',
    wireframe_color: str = 'k',
    figsize: tuple = (14,6),
    elev: float = 30, azim: float = -60,
    show: bool = True,
    savepath: str | None = None,
):
    """
    Compute & plot both \hat u^\lambda(t,x) and relative error |hat - true| using
    the same per-(t) simulated paths across x.

    Returns:
      t_grid, x_grid, U_mean, U_err, fig

    Notes:
      - For each t in t_grid, we vectorize over x by repeating each x `n_paths` times,
        simulate uncontrolled X paths forward to T, evaluate w_model to form H and
        compute MC estimate u_sample = g(X_T) + \int H dt. Then aggregate per-x.
      - This mirrors logic from `recover_u_at_point` but kept here so we reuse the
        same simulated samples for both plots.
    """
    device = next(w_model.parameters()).device if any(True for _ in w_model.parameters()) else torch.device('cpu')

    # defaults from spec
    if time_steps is None:
        time_steps = spec.T if hasattr(spec, "T") else 21
    if A_min is None: A_min = spec.A_min
    if A_max is None: A_max = spec.A_max
    if num_a_points is None: num_a_points = getattr(spec, "num_a_points", 160) if hasattr(spec, "num_a_points") else 160
    if lambda_reg is None: lambda_reg = spec.lambda_reg

    # time discretization used for simulation (match recover_u_at_point)
    T = spec.T if hasattr(spec, "T") else float(t_max)
    delta_t = T / (time_steps - 1)
    t_seq = torch.linspace(0.0, T, time_steps, device=device)
    dt_t = float(delta_t)
    T_t = torch.tensor(T, dtype=torch.float32, device=device)
    a_grid = torch.linspace(A_min, A_max, num_a_points, device=device)

    # t / x grids for plotting
    t_grid = np.linspace(t_min, t_max, n_t, dtype=float)
    x_grid = np.linspace(x_min, x_max, n_x, dtype=float)

    # containers: rows=time, cols=x
    U_mean_rows = []
    U_se_rows = []
    U_true_rows = []

    for t_val in t_grid:
        # determine discrete starting index k0 and steps_left (same logic as recover_u_at_point)
        k0 = int(round(float(t_val) / delta_t))
        k0 = max(0, min(k0, time_steps - 1))
        steps_left = time_steps - 1 - k0

        if steps_left <= 0:
            # already at terminal time: u_hat = g(x), zero variance
            x_tensor = torch.tensor(x_grid, dtype=torch.float32, device=device)
            with torch.no_grad():
                g_vals = spec.g_value_torch(x_tensor, T_t).detach().cpu().numpy()  # [n_x]
            U_mean_rows.append(g_vals)
            U_se_rows.append(np.zeros_like(g_vals))
            # true u if available
            U_true_rows.append(np.array([spec.u_true_np(t_val, float(xv)) for xv in x_grid], dtype=float))
            continue

        # Build MC sample grid: M = n_x, N = M * n_paths
        M = n_x
        N = M * n_paths

        # generate dW: [N, steps_left]
        dW = (math.sqrt(dt_t) * torch.randn(N, steps_left, device=device))

        # initialize X: [N, steps_left+1], first column is repeated x_grid
        X = torch.zeros(N, steps_left + 1, device=device)
        X0 = torch.tensor(x_grid, dtype=torch.float32, device=device).repeat_interleave(n_paths)  # [N]
        X[:, 0] = X0

        # simulate uncontrolled X from k0 to T
        for i in range(1, steps_left + 1):
            t_prev = t_seq[k0 + i - 1]
            sig_prev = spec.sigma_torch(t_prev, X[:, i-1]).clamp(min=spec.x_floor)
            X[:, i] = torch.clamp(X[:, i-1] + sig_prev * dW[:, i-1], min=spec.x_floor) if getattr(spec, "clamp_state", False) else (X[:, i-1] + sig_prev * dW[:, i-1])

        # Evaluate w on the interior points x_vals = X[:,1:]
        x_vals = X[:, 1:]                                         # [N, steps_left]
        t_vals = t_seq[k0+1:].unsqueeze(0).repeat(N, 1)           # [N, steps_left]

        # prepare inputs for w_model: expand to (N * steps_left * num_a_points, 3)
        t_rep = t_vals.unsqueeze(2).repeat(1, 1, num_a_points)    # [N, L, A]
        x_rep = x_vals.unsqueeze(2).repeat(1, 1, num_a_points)    # [N, L, A]
        a_rep = a_grid.view(1, 1, -1).repeat(N, t_vals.shape[1], 1) # [N, L, A]

        inp = torch.stack([t_rep, x_rep, a_rep], dim=3).view(-1, 3)  # [(N*L*A), 3]
        with torch.no_grad():
            w_out = w_model(inp).view(N, t_vals.shape[1], num_a_points)  # [N, L, A]

        # stabilized log-int over a (same as recover_u_at_point)
        S = w_out / lambda_reg
        Smax, _ = torch.max(S, dim=2, keepdim=True)
        int_exp = torch.trapz(torch.exp(S - Smax), a_grid, dim=2).clamp_min(1e-40).to(device)  # [N, L]
        log_int = torch.log(int_exp) + Smax.squeeze(2)  # [N, L]
        H_vals = (lambda_reg * log_int)  # [N, L]

        # integral term (sum over L and multiply by dt)
        integral_term = (H_vals * dt_t).sum(dim=1)  # [N]

        # terminal g at X[:, -1]
        g_term = spec.g_value_torch(X[:, -1], T_t)  # [N]

        # u_samples per MC path
        u_samples = (g_term + integral_term).detach().cpu().numpy()  # [N]

        # aggregate per x_j: reshape to (M, n_paths)
        u_samples = u_samples.reshape(M, n_paths)
        u_mean = u_samples.mean(axis=1)
        u_se = u_samples.std(axis=1, ddof=1) / math.sqrt(n_paths)

        U_mean_rows.append(u_mean)
        U_se_rows.append(u_se)

        # true u on this x_grid
        u_true_row = np.array([spec.u_true_np(float(t_val), float(xv)) for xv in x_grid], dtype=float)
        U_true_rows.append(u_true_row)

    # stack into arrays (n_t, n_x)
    U_mean = np.vstack(U_mean_rows)
    U_se = np.vstack(U_se_rows)
    U_true = np.vstack(U_true_rows)

    # relative error (avoid division by zero)
    eps = 1e-12
    denom = np.maximum(np.abs(U_true), eps)
    U_err = np.abs(U_mean - U_true) / denom

    # Build mesh for plotting: X_plot, T_plot shapes (n_t, n_x)
    T_plot, X_plot = np.meshgrid(t_grid, x_grid, indexing='xy')
    T_plot = T_plot.T
    X_plot = X_plot.T

    # Create figure with 2 subplots side-by-side
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=figsize)

    # Left: estimated u surface with true wireframe
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X_plot, T_plot, U_mean, cmap=cmap_u, linewidth=0, antialiased=True, rstride=1, cstride=1)
    ax1.set_xlabel("x"); ax1.set_ylabel("t"); ax1.set_zlabel(r"$\hat u$", labelpad=15)
    ax1.set_title(fr"$\hat{{u}}$ and $u^*$ over $t\in[{t_min},{t_max}], x\in[{x_min},{x_max}]$")
    ax1.view_init(elev=elev, azim=azim)
    fig.colorbar(surf1, ax=ax1, pad=0.10, shrink=0.6)

    if show_true:
        ax1.plot_wireframe(X_plot, T_plot, U_true, color=wireframe_color, linewidth=0.8, alpha=0.9,
                           rstride=max(1, n_t//10), cstride=max(1, n_x//10))

    # create a legend using proxy artists
    est_patch = mpatches.Patch(color='green', label='$\hat{u}$ ')
    true_patch = mpatches.Patch(color='black', label='$u^*$')
    ax1.legend([est_patch, true_patch], ['$\hat{u}$ ', '$u^*$'], loc='upper right')

    # Right: relative error surface
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(X_plot, T_plot, U_err, cmap=cmap_err, linewidth=0, antialiased=True, rstride=1, cstride=1)
    ax2.set_xlabel("x"); ax2.set_ylabel("t"); ax2.set_zlabel(r"Relative Error", labelpad=15)
    ax2.set_title("Relative Error")
    ax2.view_init(elev=elev, azim=azim)
    fig.colorbar(surf2, ax=ax2, pad=0.10, shrink=0.6)

    # optional: set same zscale for comparability (uncomment to fix)
    # max_err = float(np.nanmax(U_err))
    # ax2.set_zlim(0.0, max_err * 1.05)

    plt.tight_layout()

    if savepath is not None:
        try:
            fig.savefig(savepath, dpi=200)
            print(f"Saved combined u & error figure to: {savepath}")
        except Exception as e:
            print(f"[plot_u_and_error_surface] failed to save figure: {e}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return t_grid, x_grid, U_mean, U_err, fig

@torch.no_grad()
def plot_u_and_abserror_surface(
    spec,
    w_model,
    t_min: float,
    t_max: float,
    n_t: int = 11,
    x_min: float = 0.0,
    x_max: float = 1.0,
    n_x: int = 41,
    # per-x MC paths (we create N = n_x * n_paths samples per t)
    n_paths: int = 2000,
    time_steps: int | None = None,
    A_min: float | None = None,
    A_max: float | None = None,
    num_a_points: int | None = None,
    lambda_reg: float | None = None,
    show_true: bool = True,
    cmap_u: str = 'viridis',
    cmap_err: str = 'inferno',
    wireframe_color: str = 'k',
    figsize: tuple = (14,6),
    elev: float = 30, azim: float = -60,
    show: bool = True,
    savepath: str | None = None,
):
    """
    Compute & plot both \hat u^\lambda(t,x) and absolute error |hat - true| using
    the same per-(t) simulated paths across x.

    Returns:
      t_grid, x_grid, U_mean, U_err, fig

    Notes:
      - For each t in t_grid, we vectorize over x by repeating each x `n_paths` times,
        simulate uncontrolled X paths forward to T, evaluate w_model to form H and
        compute MC estimate u_sample = g(X_T) + \int H dt. Then aggregate per-x.
      - This mirrors logic from `recover_u_at_point` but kept here so we reuse the
        same simulated samples for both plots.
    """
    device = next(w_model.parameters()).device if any(True for _ in w_model.parameters()) else torch.device('cpu')

    # defaults from spec
    if time_steps is None:
        time_steps = spec.T if hasattr(spec, "T") else 21
    if A_min is None: A_min = spec.A_min
    if A_max is None: A_max = spec.A_max
    if num_a_points is None: num_a_points = getattr(spec, "num_a_points", 160) if hasattr(spec, "num_a_points") else 160
    if lambda_reg is None: lambda_reg = spec.lambda_reg

    # time discretization used for simulation (match recover_u_at_point)
    T = spec.T if hasattr(spec, "T") else float(t_max)
    delta_t = T / (time_steps - 1)
    t_seq = torch.linspace(0.0, T, time_steps, device=device)
    dt_t = float(delta_t)
    T_t = torch.tensor(T, dtype=torch.float32, device=device)
    a_grid = torch.linspace(A_min, A_max, num_a_points, device=device)

    # t / x grids for plotting
    t_grid = np.linspace(t_min, t_max, n_t, dtype=float)
    x_grid = np.linspace(x_min, x_max, n_x, dtype=float)

    # containers: rows=time, cols=x
    U_mean_rows = []
    U_se_rows = []
    U_true_rows = []

    for t_val in t_grid:
        # determine discrete starting index k0 and steps_left (same logic as recover_u_at_point)
        k0 = int(round(float(t_val) / delta_t))
        k0 = max(0, min(k0, time_steps - 1))
        steps_left = time_steps - 1 - k0

        if steps_left <= 0:
            # already at terminal time: u_hat = g(x), zero variance
            x_tensor = torch.tensor(x_grid, dtype=torch.float32, device=device)
            with torch.no_grad():
                g_vals = spec.g_value_torch(x_tensor, T_t).detach().cpu().numpy()  # [n_x]
            U_mean_rows.append(g_vals)
            U_se_rows.append(np.zeros_like(g_vals))
            # true u if available
            U_true_rows.append(np.array([spec.u_true_np(t_val, float(xv)) for xv in x_grid], dtype=float))
            continue

        # Build MC sample grid: M = n_x, N = M * n_paths
        M = n_x
        N = M * n_paths

        # generate dW: [N, steps_left]
        dW = (math.sqrt(dt_t) * torch.randn(N, steps_left, device=device))

        # initialize X: [N, steps_left+1], first column is repeated x_grid
        X = torch.zeros(N, steps_left + 1, device=device)
        X0 = torch.tensor(x_grid, dtype=torch.float32, device=device).repeat_interleave(n_paths)  # [N]
        X[:, 0] = X0

        # simulate uncontrolled X from k0 to T
        for i in range(1, steps_left + 1):
            t_prev = t_seq[k0 + i - 1]
            sig_prev = spec.sigma_torch(t_prev, X[:, i-1]).clamp(min=spec.x_floor)
            X[:, i] = torch.clamp(X[:, i-1] + sig_prev * dW[:, i-1], min=spec.x_floor) if getattr(spec, "clamp_state", False) else (X[:, i-1] + sig_prev * dW[:, i-1])

        # Evaluate w on the interior points x_vals = X[:,1:]
        x_vals = X[:, 1:]                                         # [N, steps_left]
        t_vals = t_seq[k0+1:].unsqueeze(0).repeat(N, 1)           # [N, steps_left]

        # prepare inputs for w_model: expand to (N * steps_left * num_a_points, 3)
        t_rep = t_vals.unsqueeze(2).repeat(1, 1, num_a_points)    # [N, L, A]
        x_rep = x_vals.unsqueeze(2).repeat(1, 1, num_a_points)    # [N, L, A]
        a_rep = a_grid.view(1, 1, -1).repeat(N, t_vals.shape[1], 1) # [N, L, A]

        inp = torch.stack([t_rep, x_rep, a_rep], dim=3).view(-1, 3)  # [(N*L*A), 3]
        with torch.no_grad():
            w_out = w_model(inp).view(N, t_vals.shape[1], num_a_points)  # [N, L, A]

        # stabilized log-int over a (same as recover_u_at_point)
        S = w_out / lambda_reg
        Smax, _ = torch.max(S, dim=2, keepdim=True)
        int_exp = torch.trapz(torch.exp(S - Smax), a_grid, dim=2).clamp_min(1e-40).to(device)  # [N, L]
        log_int = torch.log(int_exp) + Smax.squeeze(2)  # [N, L]
        H_vals = (lambda_reg * log_int)  # [N, L]

        # integral term (sum over L and multiply by dt)
        integral_term = (H_vals * dt_t).sum(dim=1)  # [N]

        # terminal g at X[:, -1]
        g_term = spec.g_value_torch(X[:, -1], T_t)  # [N]

        # u_samples per MC path
        u_samples = (g_term + integral_term).detach().cpu().numpy()  # [N]

        # aggregate per x_j: reshape to (M, n_paths)
        u_samples = u_samples.reshape(M, n_paths)
        u_mean = u_samples.mean(axis=1)
        u_se = u_samples.std(axis=1, ddof=1) / math.sqrt(n_paths)

        U_mean_rows.append(u_mean)
        U_se_rows.append(u_se)

        # true u on this x_grid
        u_true_row = np.array([spec.u_true_np(float(t_val), float(xv)) for xv in x_grid], dtype=float)
        U_true_rows.append(u_true_row)

    # stack into arrays (n_t, n_x)
    U_mean = np.vstack(U_mean_rows)
    U_se = np.vstack(U_se_rows)
    U_true = np.vstack(U_true_rows)

    # error
    U_err = np.abs(U_mean - U_true)

    # Build mesh for plotting: X_plot, T_plot shapes (n_t, n_x)
    T_plot, X_plot = np.meshgrid(t_grid, x_grid, indexing='xy')
    T_plot = T_plot.T
    X_plot = X_plot.T

    # Create figure with 2 subplots side-by-side
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=figsize)

    # Left: estimated u surface with true wireframe
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X_plot, T_plot, U_mean, cmap=cmap_u, linewidth=0, antialiased=True, rstride=1, cstride=1)
    ax1.set_xlabel("x"); ax1.set_ylabel("t"); ax1.set_zlabel(r"$\hat u$", labelpad=15)
    ax1.set_title(fr"$\hat{{u}}$ and $u^*$ over $t\in[{t_min},{t_max}], x\in[{x_min},{x_max}]$")
    ax1.view_init(elev=elev, azim=azim)
    fig.colorbar(surf1, ax=ax1, pad=0.10, shrink=0.6)

    if show_true:
        ax1.plot_wireframe(X_plot, T_plot, U_true, color=wireframe_color, linewidth=0.8, alpha=0.9,
                           rstride=max(1, n_t//10), cstride=max(1, n_x//10))

    # create a legend using proxy artists
    est_patch = mpatches.Patch(color='green', label='$\hat{u}$ ')
    true_patch = mpatches.Patch(color='black', label='$u^*$')
    ax1.legend([est_patch, true_patch], ['$\hat{u}$ ', '$u^*$'], loc='upper right')

    # Right: absolute error surface
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(X_plot, T_plot, U_err, cmap=cmap_err, linewidth=0, antialiased=True, rstride=1, cstride=1)
    ax2.set_xlabel("x"); ax2.set_ylabel("t"); ax2.set_zlabel(r"$|\hat u - u^*|$", labelpad=15)
    ax2.set_title("Absolute error")
    ax2.view_init(elev=elev, azim=azim)
    fig.colorbar(surf2, ax=ax2, pad=0.10, shrink=0.6)

    # optional: set same zscale for comparability (uncomment to fix)
    # max_err = float(np.nanmax(U_err))
    # ax2.set_zlim(0.0, max_err * 1.05)

    plt.tight_layout()

    if savepath is not None:
        try:
            fig.savefig(savepath, dpi=200)
            print(f"Saved combined u & error figure to: {savepath}")
        except Exception as e:
            print(f"[plot_u_and_error_surface] failed to save figure: {e}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return t_grid, x_grid, U_mean, U_err, fig

@torch.no_grad()
def plot_policy_slice(
    spec: ProblemSpec,
    w_model,
    t0: float,
    x_min: float, x_max: float, n_x: int = 81,
    A_min: float | None = None, A_max: float | None = None, n_a: int = 101,
    lambda_reg: float | None = None,
    show=True,
    figsize=(8,6),
    elev=30, azim=220,
    surface_kwargs=None
):
    """
    Plot π(t0, x, a) as a 3D surface over (x,a) at fixed time t0.
    Returns (X_plot, A_plot, pi) as numpy arrays of shape [n_x, n_a].
    """
    if A_min is None: A_min = spec.A_min
    if A_max is None: A_max = spec.A_max
    if lambda_reg is None: lambda_reg = spec.lambda_reg
    if surface_kwargs is None: surface_kwargs = {}

    # safe device detection
    try:
        device = next(w_model.parameters()).device
    except StopIteration:
        device = torch.device('cpu')

    # ensure eval mode & no_grad
    try:
        w_model.eval()
    except Exception:
        pass

    # grids
    x_grid = np.linspace(x_min, x_max, n_x, dtype=float)
    a_grid = np.linspace(A_min, A_max, n_a, dtype=float)

    # prepare (t,x,a) flattened input
    X_mesh, A_mesh = np.meshgrid(x_grid, a_grid, indexing='xy')  # (n_a, n_x)
    X_flat = X_mesh.T.reshape(-1)   # length n_x * n_a
    A_flat = A_mesh.T.reshape(-1)

    t_col = torch.full((X_flat.shape[0], 1), float(t0), dtype=torch.float32, device=device)
    x_col = torch.tensor(X_flat, dtype=torch.float32, device=device).unsqueeze(1)
    a_col = torch.tensor(A_flat, dtype=torch.float32, device=device).unsqueeze(1)

    inp = torch.cat([t_col, x_col, a_col], dim=1)
    w_out = w_model(inp).squeeze().cpu().numpy()  # length n_x * n_a

    # reshape to [n_x, n_a] : rows = x index, cols = a index
    W = w_out.reshape(n_x, n_a)

    # Gibbs policy (stable): pi(x,a) = exp(W/λ - max) / ∫_a exp(...)
    L = float(lambda_reg)
    W_div = W / L
    Wmax = np.max(W_div, axis=1, keepdims=True)            # [n_x, 1]
    exp_shift = np.exp(W_div - Wmax)                        # [n_x, n_a]

    # np.trapz returns shape (n_x,), reshape to (n_x,1)
    denom_1d = np.trapz(exp_shift, a_grid, axis=1).clip(min=1e-30)  # [n_x]
    denom = denom_1d.reshape(-1, 1)                                # [n_x,1]

    pi = exp_shift / denom                                        # [n_x, n_a]

    # mesh for plotting with shape (n_x, n_a)
    X_plot = X_mesh.T
    A_plot = A_mesh.T

    # plotting
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X_plot, A_plot, pi, rstride=1, cstride=1, linewidth=0, antialiased=True, **surface_kwargs)
    ax.set_xlabel("x")
    ax.set_ylabel("a")
    ax.set_zlabel(r"$\pi(t_0,x,a)$")
    ax.set_title(fr"$\pi(t_0={t0}, x, a)$ (Gibbs, $\lambda$={L:.3g})")
    ax.view_init(elev=elev, azim=azim)
    fig.colorbar(surf, pad=0.1, shrink=0.8)
    plt.tight_layout()
    if show:
        plt.show()

    return X_plot, A_plot, pi


@torch.no_grad()
def plot_policy_with_truth(
    spec: ProblemSpec,
    w_model,
    t0: float,
    x_min: float, x_max: float, n_x: int = 81,
    A_min: float | None = None, A_max: float | None = None, n_a: int = 101,
    lambda_reg: float | None = None,
    show=True,
    figsize=(10,7),
    elev=30, azim=220,
    est_cmap='viridis', true_cmap='plasma',
    est_alpha=0.85, true_alpha=0.6,
):
    """
    Plot estimated pi(t0,x,a) and true pi(t0,x,a) together (3D surface).

    Returns:
      (X_plot, A_plot, pi_est, pi_true) as numpy arrays with shape (n_x, n_a).
    """
    if A_min is None: A_min = spec.A_min
    if A_max is None: A_max = spec.A_max
    if lambda_reg is None: lambda_reg = spec.lambda_reg

    # safe device detection
    try:
        device = next(w_model.parameters()).device
    except Exception:
        device = torch.device('cpu')

    # ensure eval mode
    try:
        w_model.eval()
    except Exception:
        pass

    # grids
    x_grid = np.linspace(x_min, x_max, n_x, dtype=float)
    a_grid = np.linspace(A_min, A_max, n_a, dtype=float)

    # mesh flattened for model eval
    X_mesh, A_mesh = np.meshgrid(x_grid, a_grid, indexing='xy')  # shape (n_a, n_x)
    X_flat = X_mesh.T.reshape(-1)   # length n_x * n_a
    A_flat = A_mesh.T.reshape(-1)

    # -----------------------
    # 1) Estimated w -> pi_est
    # -----------------------
    t_col = torch.full((X_flat.shape[0], 1), float(t0), dtype=torch.float32, device=device)
    x_col = torch.tensor(X_flat, dtype=torch.float32, device=device).unsqueeze(1)
    a_col = torch.tensor(A_flat, dtype=torch.float32, device=device).unsqueeze(1)
    inp = torch.cat([t_col, x_col, a_col], dim=1)
    with torch.no_grad():
        w_out = w_model(inp).squeeze().cpu().numpy()
    W_est = w_out.reshape(n_x, n_a)  # rows=x, cols=a

    # stable Gibbs per x
    L = float(lambda_reg)
    Wdiv_est = W_est / L
    Wmax_est = np.max(Wdiv_est, axis=1, keepdims=True)         # [n_x,1]
    exp_shift_est = np.exp(Wdiv_est - Wmax_est)                # [n_x,n_a]
    denom_est = np.trapz(exp_shift_est, a_grid, axis=1).clip(min=1e-30).reshape(-1, 1)  # [n_x,1]
    pi_est = exp_shift_est / denom_est                         # [n_x, n_a]

    # -----------------------
    # 2) True w -> pi_true
    # -----------------------
    # compute u_x_true on x_grid using spec.u_x_true_np (returns float/nan)
    u_x_list = []
    for xv in x_grid:
        try:
            u_xv = spec.u_x_true_np(t0, float(xv))
        except Exception:
            u_xv = float('nan')
        u_x_list.append(u_xv)
    u_x_arr = np.array(u_x_list, dtype=float)  # shape (n_x,)

    # If no analytic u_x available, warn and fill with nan (will produce NaN pi_true)
    if np.all(np.isnan(u_x_arr)):
        print("[plot_policy_with_truth] Warning: spec.u_x_true_np returned all NaNs; cannot compute true pi.")
        W_true = np.full((n_x, n_a), np.nan)
        pi_true = np.full((n_x, n_a), np.nan)
    else:
        # To compute b(t,x,a) and r(t,x,a) robustly, evaluate them in torch on grid.
        # Build tensors shaped (n_x * n_a, 1) but we will broadcast u_x per x later.
        t_col = torch.full((X_flat.shape[0], 1), float(t0), dtype=torch.float32, device=device)
        x_col = torch.tensor(X_flat, dtype=torch.float32, device=device).unsqueeze(1)
        a_col = torch.tensor(A_flat, dtype=torch.float32, device=device).unsqueeze(1)

        with torch.no_grad():
            # b_ctrl_torch and running_reward_torch accept (t, x, a)
            b_flat = spec.b_ctrl_torch(t_col.squeeze(1), x_col.squeeze(1), a_col.squeeze(1)).detach().cpu().numpy().reshape(n_x, n_a)
            r_flat = spec.running_reward_torch(t_col.squeeze(1), x_col.squeeze(1), a_col.squeeze(1)).detach().cpu().numpy().reshape(n_x, n_a)

        # broadcast u_x_arr over action dimension: shape (n_x, 1) * (1, n_a) -> (n_x, n_a)
        u_x_mat = u_x_arr.reshape(n_x, 1)
        W_true = b_flat * u_x_mat + r_flat   # (n_x, n_a)

        # stable Gibbs for true W
        Wdiv_true = W_true / L
        Wmax_true = np.max(Wdiv_true, axis=1, keepdims=True)
        exp_shift_true = np.exp(Wdiv_true - Wmax_true)
        denom_true = np.trapz(exp_shift_true, a_grid, axis=1).clip(min=1e-30).reshape(-1, 1)
        pi_true = exp_shift_true / denom_true

    # -----------------------
    # 3) Prepare plot meshes
    # -----------------------
    X_plot = X_mesh.T   # (n_x, n_a)
    A_plot = A_mesh.T   # (n_x, n_a)

    # -----------------------
    # 4) Plot both surfaces
    # -----------------------
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # estimated surface
    surf_est = ax.plot_surface(
        X_plot, A_plot, pi_est,
        rstride=1, cstride=1, linewidth=0, antialiased=True,
        cmap=est_cmap, alpha=est_alpha
    )

    # true surface (slightly offset in z to avoid z-fighting if necessary)
    # overlay with different cmap and transparency
    surf_true = ax.plot_surface(
        X_plot, A_plot, pi_true,
        rstride=1, cstride=1, linewidth=0, antialiased=True,
        cmap=true_cmap, alpha=true_alpha
    )
    ax.set_zlim(0.7, 1.2)
    ax.set_xlabel("x")
    ax.set_ylabel("a")
    ax.set_zlabel(r"$\pi(t_0,x,a)$")
    ax.set_title(fr"$\hat{{\pi}}$ vs $\pi^*$ at $t={t0}$")
    ax.view_init(elev=elev, azim=azim)

    # colorbars (one for estimated, one for true) — place them side by side
    fig.colorbar(surf_est, ax=ax, pad=0.08, shrink=0.6, fraction=0.05)
    #fig.colorbar(surf_true, ax=ax, pad=0.02, shrink=0.6, fraction=0.05, label='pi_true')
    ax.text2D(
    1.05, 0.85,
    r" $\pi^*=1.00$",
    transform=ax.transAxes,
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5')
    )
    # create a legend using proxy artists
    import matplotlib.cm as cm
    est_color = cm.get_cmap('viridis')(0.7)   # sample blue-green tone
    true_color = cm.get_cmap('plasma')(0.7)   # sample red-yellow tone

    est_patch = mpatches.Patch(color=est_color, label=r'$\hat{\pi}$')
    true_patch = mpatches.Patch(color=true_color, label=r'$\pi^*$')
    ax.legend([est_patch, true_patch], ['$\hat{\pi}$', '$\pi^*$'], loc='upper right')

    plt.tight_layout()
    if show:
        plt.show()

    return X_plot, A_plot, pi_est, pi_true

@torch.no_grad()
def plot_u_curve_vs_true(
    spec: ProblemSpec,
    w_model,
    t0: float,
    x_min: float,
    x_max: float,
    n_x: int = 51,
    n_paths: int = 4000,
    time_steps: int | None = None,
    A_min: float | None = None,
    A_max: float | None = None,
    num_a_points: int | None = None,
    lambda_reg: float | None = None,
    show: bool = True,
    savepath: str | None = None,
    figsize: tuple = (12, 5)
):
    """
    Plots a 1D slice of \hat{u}^\lambda(t_0, x) against u^*(t_0, x) over x,
    alongside a subplot for the relative error.
    """
    import matplotlib.pyplot as plt

    # Resolve defaults from spec if not provided
    if time_steps is None: time_steps = spec.T if hasattr(spec, "T") else 21
    if A_min is None: A_min = spec.A_min
    if A_max is None: A_max = spec.A_max
    if num_a_points is None: num_a_points = getattr(spec, "num_a_points", 160) if hasattr(spec, "num_a_points") else 160
    if lambda_reg is None: lambda_reg = spec.lambda_reg
    T = spec.T if hasattr(spec, "T") else 1.0

    # 1. Recover u over the interval
    x_grid, u_mean, u_se = recover_u_at_point(
        spec, w_model, t0=t0, x0=0.0,  # x0 is ignored in interval mode
        T=T, time_steps=time_steps,
        A_min=A_min, A_max=A_max, num_a_points=num_a_points, lambda_reg=lambda_reg,
        n_paths=n_paths,
        x0_mode=("interval", {"x_min": x_min, "x_max": x_max, "n_points": n_x})
    )

    # 2. Compute true u and relative error
    u_true = np.array([spec.u_true_np(t0, float(x)) for x in x_grid], dtype=float)
    
    eps = 1e-12
    denom = np.maximum(np.abs(u_true), eps)
    rel_err = np.abs(u_mean - u_true) / denom

    # 3. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left Plot: Curves
    ax1.plot(x_grid, u_mean, label=r"$\hat u(t_0,x)$", color='blue', linewidth=2)
    #ax1.fill_between(x_grid, u_mean - 2*u_se, u_mean + 2*u_se, alpha=0.2, color='blue', label="±2 SE")
    
    if not np.all(np.isnan(u_true)):
        ax1.plot(x_grid, u_true, "--", label=r"$u^*(t_0,x)$", color='black', linewidth=2)
        
    ax1.set_xlabel("x")
    ax1.set_ylabel("u")
    ax1.set_title(fr"$\hat u$ and $u^*$ at $t_0={t0}$")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right Plot: Relative Error
    if not np.all(np.isnan(u_true)):
        ax2.plot(x_grid, rel_err, color='red', linewidth=2)
        ax2.set_xlabel("x")
        ax2.set_ylabel("Relative Error")
        ax2.set_title("Relative Error")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "True function not available\nCannot compute error", 
                 ha='center', va='center', transform=ax2.transAxes)

    plt.tight_layout()

    if savepath is not None:
        try:
            fig.savefig(savepath, dpi=200)
            print(f"Saved 1D curve figure to: {savepath}")
        except Exception as e:
            print(f"[plot_u_curve_vs_true] failed to save figure: {e}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return x_grid, u_mean, u_true, rel_err, fig
# ==============================================================================
#                                EXAMPLE RUN I fixed
# ==============================================================================
# if __name__ == "__main__":
#     spec = EX4()  # choose your problem

#     # Choose norm surrogate:
#     #   norm_type="lp",  lp_p=8.0
#     #   norm_type="softmax", softmax_tau="auto"  (or a fixed float)
#     norm_type = "softmax"      # "lp" or "softmax"
#     lp_p      = 8.0
#     soft_tau  = "auto"         # or a float e.g., 0.005

#     v_Model, w_Model, meta = train_vanilla_with_norm(
#         spec,
#         T=spec.T, time_steps=6,
#         training_path_size=5000, nn_batch_size=2000, num_epochs=50,
#         neuron_number_v=64, neuron_number_w=64, learning_rate=5e-4, weight_decay=0.0,
#         A_min=spec.A_min, A_max=spec.A_max, num_a_points=100, lambda_reg=spec.lambda_reg,
#         int_x_steps=50,
#         norm_type="softmax", lp_p=8.0, softmax_tau="auto",
#         x0_init=1.0,  # ignored when x0_mode is given
#         x0_mode=("fixed", {"x_min": None, "x_max": None})
#     )


#     # Quick probes at (t0=0,x0=1)
#     t0, x0 = 0.0, 1.0
#     with torch.no_grad():
#         v_hat = v_Model(torch.tensor([[t0, x0]], dtype=torch.float32, device=device)).item()
#     v_true = spec.u_x_true_np(t0, x0)
#     if not np.isnan(v_true):
#         print(f"\nV_x(t0={t0}, x0={x0}) ≈ {v_hat:.6f} | True {v_true:.6f}")
#     else:
#         print(f"\nV_x(t0={t0}, x0={x0}) ≈ {v_hat:.6f} | (no closed-form provided)")

#     u_hat, u_se = recover_u_at_point(
#         spec, w_Model, t0=t0, x0=x0,
#         T=meta["T"], time_steps=meta["time_steps"],
#         num_a_points=meta["num_a_points"], lambda_reg=meta["lambda_reg"],
#         A_min=meta["A_min"], A_max=meta["A_max"],
#         n_paths=10000
#     )
#     u_true = spec.u_true_np(t0, x0)
#     if not np.isnan(u_true):
#         print(f"u(t0={t0}, x0={x0})  ≈ {u_hat:.6f}  (MC SE≈{u_se:.6f}) | True {u_true:.6f}")
#     else:
#         print(f"u(t0={t0}, x0={x0})  ≈ {u_hat:.6f}  (MC SE≈{u_se:.6f}) | (no closed-form provided)")


# ==============================================================================
#                                EXAMPLE RUN II interval
# ==============================================================================

if __name__ == "__main__":
    spec = EX7()  # choose your problem

    # Choose norm surrogate:
    #   norm_type="lp",  lp_p=8.0
    #   norm_type="softmax", softmax_tau="auto"  (or a fixed float)
    norm_type = "softmax"      # "lp" or "softmax"
    lp_p      = 8.0
    soft_tau  = "auto"         # or a float e.g., 0.005

    # Train with randomized initial states in an interval (better coverage / less noise)
    v_Model, w_Model, meta = train_vanilla_with_norm(
        spec,
        T=0.2, time_steps=6,
        training_path_size=5000, nn_batch_size=5000, num_epochs=200,
        neuron_number_v=64, neuron_number_w=32, learning_rate=5e-4, weight_decay=0.0,
        A_min=spec.A_min, A_max=spec.A_max, num_a_points=96, lambda_reg=5.0,
        int_x_steps=40,
        norm_type=norm_type, lp_p=lp_p, softmax_tau=soft_tau,
        x0_init=0.1,  # ignored when x0_mode is given
        x0_mode=("uniform", {"x_min": 0.0, "x_max": 0.2}),
        early_stopping=True, es_patience=10, es_min_delta=1e-4
    )

    # ---------------------- Single-point probe ----------------------
    t0, x0 = 0.0, 0.1
    with torch.no_grad():
        v_hat = v_Model(torch.tensor([[t0, x0]], dtype=torch.float32, device=device)).item()
    v_true = spec.u_x_true_np(t0, x0)
    if not np.isnan(v_true):
        print(f"\nV_x(t0={t0}, x0={x0}) ≈ {v_hat:.6f} | True {v_true:.6f}")
    else:
        print(f"\nV_x(t0={t0}, x0={x0}) ≈ {v_hat:.6f} | (no closed-form provided)")

    u_hat, u_se = recover_u_at_point(
        spec, w_Model, t0=t0, x0=x0,
        T=meta["T"], time_steps=meta["time_steps"],
        num_a_points=meta["num_a_points"], lambda_reg=meta["lambda_reg"],
        A_min=meta["A_min"], A_max=meta["A_max"],
        n_paths=10000
    )
    u_true = spec.u_true_np(t0, x0)
    if not np.isnan(u_true):
        print(f"u(t0={t0}, x0={x0})  ≈ {u_hat:.6f}  (MC SE≈{u_se:.6f}) | True {u_true:.6f}")
    else:
        print(f"u(t0={t0}, x0={x0})  ≈ {u_hat:.6f}  (MC SE≈{u_se:.6f}) | (no closed-form provided)")

    # ---------------------- Interval recovery (vectorized) ----------------------
    # Evaluate \hat{u}^\lambda(t0, x) across an interval using one simulation pass.
    x_min, x_max, M = 0.0, 0.2, 21
    x_grid, u_mean, u_se = recover_u_at_point(
        spec, w_Model, t0=t0, x0=0.1,  # x0 ignored in interval mode
        T=meta["T"], time_steps=meta["time_steps"],
        num_a_points=meta["num_a_points"], lambda_reg=meta["lambda_reg"],
        A_min=meta["A_min"], A_max=meta["A_max"],
        n_paths=4000,
        x0_mode=("interval", {"x_min": x_min, "x_max": x_max, "n_points": M})
    )

    # Print a brief summary and (optionally) plot if matplotlib is available
    print(f"\nInterval recovery over [{x_min}, {x_max}] with {M} points:")
    print(f"  First 3 (x, u_hat, SE): {list(zip(x_grid[:3], u_mean[:3], u_se[:3]))}")
    print(f"  Last 3  (x, u_hat, SE): {list(zip(x_grid[-3:], u_mean[-3:], u_se[-3:]))}")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,4))
        plt.plot(x_grid, u_mean, label=r"$\hat u^\lambda(t_0,x)$")
        plt.fill_between(x_grid, u_mean - 2*u_se, u_mean + 2*u_se, alpha=0.2, label="±2 SE")
        # overlay truth when available
        u_true_curve = np.array([spec.u_true_np(t0, float(x)) for x in x_grid], dtype=float)
        if not np.all(np.isnan(u_true_curve)):
            plt.plot(x_grid, u_true_curve, "--", label=r"$u(t_0,x)$ (true)")
        plt.xlabel("x"); plt.ylabel("u")
        plt.title(fr"$u(t_0={t0}, x)$ over [{x_min}, {x_max}]")
        plt.legend(); plt.tight_layout(); plt.show()
    except Exception as e:
        print(f"(Skipping plot: {e})")

    # surface over x∈[0.0,0.1], a∈[A_min,A_max] at t0=0
    X_plot, A_plot, pi = plot_policy_slice(spec, w_Model, t0=0.0, x_min=0.0, x_max=0.1, n_x=81, n_a=101)

    # smaller quick view
    plot_policy_slice(spec, w_Model, t0=0.1, x_min=0.0, x_max=0.1, n_x=41, n_a=61, lambda_reg=spec.lambda_reg)
    # after training and with w_Model available
    X_plot, A_plot, pi_est, pi_true = plot_policy_with_truth(
        spec, w_Model, t0=0.1, x_min=0.0, x_max=0.1, n_x=81, A_min=spec.A_min, A_max=spec.A_max, n_a=101)