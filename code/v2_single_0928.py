# %%
# J-minimization with selectable norm: L^p or Softmax (log-sum-exp)
# J(v,w) = ||v - Φ(w)|| + ||w - Ψ(v)||  (both norms chosen via norm_type argument)
# - Includes Feynman–Kac recovery of u(t,x) under the uncontrolled diffusion 𝓧
# - Problem-specification class to keep training code generic

import math
import numpy as np
import torch
import torch.nn as nn

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
            dW = torch.sqrt(dt_t) * torch.randn(B, time_steps - 1, device=device)  # [B, T-1]

            # 2) Uncontrolled 𝓧
            X_unc = torch.zeros(B, time_steps, device=device)
            X_unc[:, 0] = sample_x0(B) 
            for i in range(1, time_steps):
                t_cur = t_seq[i - 1]
                sig_prev = spec.sigma_torch(t_cur, X_unc[:, i - 1])
                X_unc[:, i] = maybe_clamp_state(X_unc[:, i - 1] + sig_prev * dW[:, i - 1])  

            # 3) Controlled X^a (one action per path for Ψ)
            a_batch = A_min + (A_max - A_min) * torch.rand(B, device=device)  # [B]
            X_ctrl = torch.zeros(B, time_steps, device=device)
            X_ctrl[:, 0] = sample_x0(B)
            for i in range(1, time_steps):
                t_cur = t_seq[i - 1]
                drift_inc = spec.b_ctrl_torch(t_cur, X_ctrl[:, i - 1], a_batch) * dt_t
                sig_prev  = spec.sigma_torch(t_cur, X_ctrl[:, i - 1])
                X_ctrl[:, i] = maybe_clamp_state(X_ctrl[:, i - 1] + drift_inc + sig_prev * dW[:, i - 1]) 

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



# ==============================================================================
#                                EXAMPLE RUN I fixed
# ==============================================================================
if __name__ == "__main__":
    spec = EX4()  # choose your problem

    # Choose norm surrogate:
    #   norm_type="lp",  lp_p=8.0
    #   norm_type="softmax", softmax_tau="auto"  (or a fixed float)
    norm_type = "softmax"      # "lp" or "softmax"
    lp_p      = 8.0
    soft_tau  = "auto"         # or a float e.g., 0.005

    v_Model, w_Model, meta = train_vanilla_with_norm(
        spec,
        T=spec.T, time_steps=11,
        training_path_size=20000, nn_batch_size=10000, num_epochs=50,
        neuron_number_v=64, neuron_number_w=64, learning_rate=5e-4, weight_decay=0.0,
        A_min=spec.A_min, A_max=spec.A_max, num_a_points=100, lambda_reg=spec.lambda_reg,
        int_x_steps=50,
        norm_type="softmax", lp_p=8.0, softmax_tau="auto",
        x0_init=1.0,  # ignored when x0_mode is given
        x0_mode=("fixed", {"x_min": None, "x_max": None})
    )


    # Quick probes at (t0=0,x0=1)
    t0, x0 = 0.0, 1.0
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


# ==============================================================================
#                                EXAMPLE RUN II interval
# ==============================================================================

if __name__ == "__main__":
    spec = EX4()  # choose your problem

    # Choose norm surrogate:
    #   norm_type="lp",  lp_p=8.0
    #   norm_type="softmax", softmax_tau="auto"  (or a fixed float)
    norm_type = "softmax"      # "lp" or "softmax"
    lp_p      = 8.0
    soft_tau  = "auto"         # or a float e.g., 0.005

    # Train with randomized initial states in an interval (better coverage / less noise)
    v_Model, w_Model, meta = train_vanilla_with_norm(
        spec,
        T=spec.T, time_steps=3,
        training_path_size=500, nn_batch_size=500, num_epochs=10,
        neuron_number_v=8, neuron_number_w=8, learning_rate=5e-4, weight_decay=0.0,
        A_min=spec.A_min, A_max=spec.A_max, num_a_points=100, lambda_reg=spec.lambda_reg,
        int_x_steps=20,
        norm_type=norm_type, lp_p=lp_p, softmax_tau=soft_tau,
        x0_init=1.0,  # ignored when x0_mode is given
        x0_mode=("uniform", {"x_min": 0.4, "x_max": 1.6})
    )

    # ---------------------- Single-point probe ----------------------
    t0, x0 = 0.0, 1.0
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
    x_min, x_max, M = 0.4, 1.6, 81
    x_grid, u_mean, u_se = recover_u_at_point(
        spec, w_Model, t0=t0, x0=0.0,  # x0 ignored in interval mode
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
        plt.fill_between(x_grid, u_mean - u_se, u_mean + u_se, alpha=0.2, label="±1 SE")
        # overlay truth when available
        u_true_curve = np.array([spec.u_true_np(t0, float(x)) for x in x_grid], dtype=float)
        if not np.all(np.isnan(u_true_curve)):
            plt.plot(x_grid, u_true_curve, "--", label=r"$u(t_0,x)$ (true)")
        plt.xlabel("x"); plt.ylabel("u")
        plt.title(fr"$u(t_0={t0}, x)$ over [{x_min}, {x_max}]")
        plt.legend(); plt.tight_layout(); plt.show()
    except Exception as e:
        print(f"(Skipping plot: {e})")

