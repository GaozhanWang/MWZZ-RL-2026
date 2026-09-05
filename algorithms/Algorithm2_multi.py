# ==============================================================================
#   MULTI-PERIOD J-MINIMIZATION WITH BACKWARD STITCHING (Φ/Ψ + MASKS) — v4.3
# ==============================================================================
# - Same as v4.2, but the x0 "bank" for each segment is now REFRESHED **every
#   epoch** inside train_single_segment by simulating the driftless reference 𝓧
#   from time 0 to the segment start t0 (using the global initial law).
# - No need to pass precomputed bank[j] anymore; train_multi_period does NOT
#   call build_reference_banks and instead lets each segment self-refresh.
# ==============================================================================

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Callable, List, Tuple, Optional

# ------------------------------ Device utils ----------------------------------
def normalize_device(device: Optional[torch.device | str]) -> torch.device:
    if device is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device) if isinstance(device, str) else device


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

    # whether to clamp the state (e.g., EX4 needs X ≥ 0 for sqrt(x))
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


class EX5(ProblemSpec):
    """
      u(t,x)  = cos(t + x),  g(x)=u(T,x)=cos(T + x)
      u_x     = -sin(t + x),  u_xx = -cos(t + x),  u_t = -sin(t + x)
      σ(t,x)  = 2 + sin(t + x)   (≥1 -> no clamping needed)
      b(t,x,a)= a - [1 + 2 cos(t+x) + 0.5 sin(t+x) cos(t+x)]
      r(t,x,a)= 2 cos(t + x) + a sin(t + x)
      HJB with λ=1 holds exactly (Gibbs integral independent of a).
    """
    def __init__(self, T=0.1, A_min=0.0, A_max=1.0, lambda_reg=1.0):
        self.T = float(T)
        self.A_min, self.A_max = float(A_min), float(A_max)
        self.lambda_reg = float(lambda_reg)
        self.x_floor = 0.0
        self.clamp_state = False

    # truths
    def u_true_np(self, t, x): return float(np.cos(t + x))
    def u_x_true_np(self, t, x): return float(-np.sin(t + x))

    # terminal
    @torch.no_grad()
    def g_value_torch(self, x, T): return torch.cos(T + x)
    def g_x_torch(self, x, T):     return -torch.sin(T + x)

    # drift / diffusion / reward
    def b_ctrl_torch(self, t, x, a):
        s = t + x
        return a - (1.0 + 2.0*torch.cos(s) + 0.5*torch.sin(s)*torch.cos(s))

    def sigma_torch(self, t, x):
        return 2.0 + torch.sin(t + x)

    def sigma_x_torch(self, t, x):
        return torch.cos(t + x)

    def running_reward_torch(self, t, x, a):
        s = t + x
        return 2.0*torch.cos(s) + a*torch.sin(s)


class EX7(ProblemSpec):
    """
    σ ≡ 1, b(t,x,a)=x+a; manufactured solution (exact for λ=1, A=[0,1]).
    """
    def __init__(self, T=0.1, A_min=0.0, A_max=1.0, lambda_reg=1.0, x_floor=0.0):
        self.T = float(T)
        self.A_min, self.A_max = float(A_min), float(A_max)
        self.lambda_reg = float(lambda_reg)
        self.x_floor = float(x_floor)
        self.clamp_state = False

    def u_true_np(self, t, x):   return float(np.exp(-(t**2 + x**2 + 1.0)))
    def u_x_true_np(self, t, x): return float(-2.0 * x * np.exp(-(t**2 + x**2 + 1.0)))

    @torch.no_grad()
    def g_value_torch(self, x, T): return torch.exp(-(T**2 + x**2 + 1.0))
    def g_x_torch(self, x, T):     return -2.0 * x * torch.exp(-(T**2 + x**2 + 1.0))

    def b_ctrl_torch(self, t, x, a): return x + a
    def sigma_torch(self, t, x):     return torch.ones_like(x)
    def sigma_x_torch(self, t, x):   return torch.zeros_like(x)
    def running_reward_torch(self, t, x, a):
        u = torch.exp(-(t**2 + x**2 + 1.0))
        return (2.0 * t + 2.0 * a * x + 1.0) * u


# ==============================================================================
#                                   MODELS
# ==============================================================================
def build_v_model(width: int, input_dim=2, output_dim=1) -> nn.Module:
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

def build_w_model(width: int, input_dim=3, output_dim=1) -> nn.Module:
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
#                                NORM REDUCERS (MASKED)
# ==============================================================================
def reduce_lp_masked(R: torch.Tensor, M: torch.Tensor, p: float = 8.0, eps=1e-12) -> torch.Tensor:
    M = M.to(R.dtype)
    num = (M * (R ** p)).sum(dim=1)                   # [B]
    den = M.sum(dim=1).clamp_min(eps)                 # [B]
    per_sample = (num / den).pow(1.0 / p)             # [B]
    valid_sample = (den > eps).to(R.dtype)
    denom = valid_sample.sum().clamp_min(1.0)
    return (per_sample * valid_sample).sum() / denom


def reduce_softmax_masked(R: torch.Tensor, M: torch.Tensor, tau="auto") -> torch.Tensor:
    B, K = R.shape
    M_bool = M.bool()

    if tau == "auto":
        if M_bool.any():
            med = R.detach()[M_bool].median().item()
            K_est = max(2.0, float(M_bool.sum().item()) / max(1.0, float(B)))
            tau_val = max(1e-4, min(1.0, 0.5 * med / math.log(K_est)))
        else:
            tau_val = 1e-3
    else:
        tau_val = float(tau)

    soft = torch.zeros(B, dtype=R.dtype, device=R.device)
    has_valid = (M_bool.sum(dim=1) > 0)

    if has_valid.any():
        neg_inf = torch.full_like(R, -float('inf'))
        Rm = torch.where(M_bool, R, neg_inf)
        m = torch.amax(Rm, dim=1, keepdim=True)
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        Z = torch.logsumexp((Rm - m) / tau_val, dim=1)
        soft_all = tau_val * Z + m.squeeze(1)
        soft = torch.where(has_valid, soft_all, torch.zeros_like(soft_all))

    denom = has_valid.float().sum().clamp_min(1.0)
    return (soft.sum() / denom)


def reduce_residuals_masked(R: torch.Tensor, M: torch.Tensor,
                            norm_type="lp", lp_p=8.0, softmax_tau="auto") -> torch.Tensor:
    R = torch.nan_to_num(R, nan=0.0, posinf=1e6, neginf=0.0)
    if norm_type == "lp":
        return reduce_lp_masked(R, M, p=lp_p)
    elif norm_type == "softmax":
        return reduce_softmax_masked(R, M, tau=softmax_tau)
    else:
        raise ValueError(f"Unknown norm_type '{norm_type}' (use 'lp' or 'softmax').")


# ==============================================================================
#                               HELPER: ∫ v dy
# ==============================================================================
def integral_v_trap(model: nn.Module, t, low: torch.Tensor, up: torch.Tensor,
                    steps: int, x_floor: float, clamp_state: bool = False) -> torch.Tensor:
    B = low.size(0)
    delta = (up - low) / steps
    ar = torch.arange(0, steps + 1, device=low.device) / steps
    x_grid = (low.unsqueeze(1) + delta.unsqueeze(1) * ar.unsqueeze(0))
    if clamp_state:
        x_grid = x_grid.clamp(min=x_floor)

    if isinstance(t, torch.Tensor):
        if t.dim() == 0:
            t_grid = t * torch.ones_like(x_grid)
        elif t.dim() == 1:
            t_grid = t.unsqueeze(1).expand_as(x_grid)
        else:
            t_grid = t
    else:
        t_grid = torch.tensor(t, dtype=x_grid.dtype, device=x_grid.device) * torch.ones_like(x_grid)
    inputs = torch.stack([t_grid, x_grid], dim=2).view(-1, 2)
    v_vals = model(inputs).view(B, steps + 1)
    return delta / 2 * (v_vals[:, 0] + v_vals[:, -1] + 2 * v_vals[:, 1:-1].sum(dim=1))


# ==============================================================================
#                               HELPER: Law(𝓧_t) BANKS
# ==============================================================================
@torch.no_grad()
def build_reference_banks(
    spec: ProblemSpec,
    breaks: List[float],
    time_steps_per_segment: int,
    n_paths: int,
    initial_x: Optional[float] = None,
    x0_mode: Tuple[str, dict] = ("fixed", {"x0": 0.0}),
    device: Optional[torch.device | str] = None,
    seed: Optional[int] = None,
):
    """
    Simulates driftless reference 𝓧 over the absolute time grid defined by 'breaks'
    and returns the per-segment initial draws X_ref at each τ_j (segment start).
    """
    device = normalize_device(device)
    if seed is not None:
        torch.manual_seed(seed); np.random.seed(seed)

    # Assemble absolute grid and record start_idx on the fly
    t_grid_abs: List[float] = []
    start_idx: List[int] = []
    for j in range(len(breaks) - 1):
        start, end = breaks[j], breaks[j + 1]
        K = time_steps_per_segment
        tg = np.linspace(start, end, K, endpoint=True)
        if j == 0:
            start_idx.append(len(t_grid_abs))       # = 0
            t_grid_abs.extend(list(tg))             # include both start & end
        else:
            start_idx.append(len(t_grid_abs) - 1)   # τ_j already present
            t_grid_abs.extend(list(tg[1:]))

    t_seq_abs = torch.tensor(t_grid_abs, dtype=torch.float32, device=device)
    M = t_seq_abs.numel()

    # Initialize reference X at time 0
    X_ref = torch.zeros(n_paths, M, device=device)
    def maybe_clamp_state(x: torch.Tensor) -> torch.Tensor:
        return x.clamp_min(spec.x_floor) if getattr(spec, "clamp_state", False) else x

    if initial_x is not None:
        X_ref[:, 0] = float(initial_x)
    else:
        mode, opts = x0_mode
        if mode == "uniform":
            lo = float(opts.get("x_min", -1.0)); hi = float(opts.get("x_max", 1.0))
            X_ref[:, 0] = lo + (hi - lo) * torch.rand(n_paths, device=device)
        elif mode == "fixed":
            X_ref[:, 0] = float(opts.get("x0", 0.0))
        else:
            raise ValueError("x0_mode must be ('uniform', ...) or ('fixed', ...).")

    # Simulate driftless reference with sign-preserving σ clamp
    def clamp_sigma(sig: torch.Tensor, eps=1e-6):
        return sig.sign() * torch.clamp(sig.abs(), min=eps)

    dt = t_seq_abs[1:] - t_seq_abs[:-1]
    for k in range(1, M):
        t_prev = t_seq_abs[k - 1]
        sig = clamp_sigma(spec.sigma_torch(t_prev, X_ref[:, k - 1]))
        dW  = torch.sqrt(dt[k - 1]) * torch.randn(n_paths, device=device)
        X_ref[:, k] = maybe_clamp_state(X_ref[:, k - 1] + sig * dW)

    banks = [X_ref[:, idx].detach() for idx in start_idx]  # X at τ_j
    return t_seq_abs, X_ref, banks, start_idx


# ==============================================================================
#                         TRAIN SINGLE SEGMENT (t0 -> t1)  (REFRESH BANK/EPOCH)
# ==============================================================================
def train_single_segment(
    spec: ProblemSpec,
    t0: float, t1: float,
    # data domain & initial sampling (fallback if bank is not used)
    x_domain: Optional[Tuple[float, float]] = None,   # None -> no masks/gates
    x0_mode: Tuple[str, dict] = ("uniform", {"x_min": None, "x_max": None}),
    mask_mode: str = "timepoint",
    # grid
    time_steps: int = 21,
    # MC
    training_path_size: int = 4000, nn_batch_size: int = 1000,
    # nets/opt
    neuron_number_v: int = 128, neuron_number_w: int = 128,
    learning_rate: float = 5e-4, weight_decay: float = 0.0,
    # actions/entropy
    A_min: Optional[float] = None, A_max: Optional[float] = None,
    num_a_points: int = 160, lambda_reg: Optional[float] = None,
    # Ψ integral resolution
    int_x_steps: int = 50,
    # norms
    norm_type: str = "lp", lp_p: float = 8.0, softmax_tau: str | float = "auto",
    # boundary derivative provider at t1 (for Φ terminal term)
    g_x_boundary: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    # early stop
    num_epochs: int = 100, early_stopping: bool = True, es_patience: int = 20, es_min_delta: float = 5e-4,
    # misc
    device: Optional[torch.device | str] = None, verbose: bool = True,
    # numerics
    exp_clip: Optional[float] = None,
    # ===== NEW: bank controls =====
    ref_bank_size: Optional[int] = None,             # if None -> fallback to x0_mode sampling
    refresh_bank_each_epoch: bool = True,            # True -> rebuild bank every epoch
    initial_x_global: Optional[float] = None,        # global law at t=0: fixed x0 if provided
    x0_mode_global: Optional[Tuple[str, dict]] = None,  # else use this as global initial law
):
    """
    Train (v,w) on a single segment [t0,t1], refreshing the x0-bank each epoch by
    simulating 𝓧 from time 0 to t0 using the global initial law.
    """
    device = normalize_device(device)
    vprint = print if verbose else (lambda *a, **k: None)

    if A_min is None: A_min = spec.A_min
    if A_max is None: A_max = spec.A_max
    if lambda_reg is None: lambda_reg = spec.lambda_reg
    if x0_mode_global is None: x0_mode_global = x0_mode  # sensible default

    Δ = float(t1 - t0)
    assert Δ > 0.0, "t1 must be greater than t0"
    delta_t = Δ / (time_steps - 1)
    t_seq = torch.linspace(t0, t1, time_steps, device=device)
    dt_t  = torch.tensor(delta_t, dtype=torch.float32, device=device)
    T1_t  = torch.tensor(t1, dtype=torch.float32, device=device)

    a_grid = torch.linspace(A_min, A_max, num_a_points, device=device)

    def maybe_clamp_state(x: torch.Tensor) -> torch.Tensor:
        return x.clamp_min(spec.x_floor) if getattr(spec, "clamp_state", False) else x

    use_masks = (x_domain is not None) and (mask_mode != "none")
    if use_masks:
        x_min, x_max = x_domain

    # boundary derivative g_x at t1
    if g_x_boundary is None:
        def g_x_boundary_fn(x: torch.Tensor) -> torch.Tensor:
            return spec.g_x_torch(x, T1_t)
    else:
        g_x_boundary_fn = g_x_boundary

    # models & optimizer
    v_Model = build_v_model(neuron_number_v).to(device)
    w_Model = build_w_model(neuron_number_w).to(device)
    optim = torch.optim.Adam(list(v_Model.parameters()) + list(w_Model.parameters()),
                             lr=learning_rate, weight_decay=weight_decay)

    def clamp_sigma(sig: torch.Tensor, eps=1e-6):
        return sig.sign() * torch.clamp(sig.abs(), min=eps)

    best_loss = float('inf')
    patience_ctr = 0
    stopped_early = False
    epochs_run = 0

    # -------------- helper: create (or refresh) per-epoch x0-bank --------------
    def make_epoch_x0_bank() -> Optional[torch.Tensor]:
        if ref_bank_size is None:
            return None
        # Construct a tiny 'breaks' just to land exactly at t0:
        # If t0 == 0, use [0, t1] and take banks[0]; else [0, t0, t1] and take banks[1].
        eps = 1e-12
        if t0 <= eps:
            bank_breaks = [0.0, t1]
            bank_index = 0  # τ_0 = 0
        else:
            bank_breaks = [0.0, t0, t1]
            bank_index = 1  # τ_1 = t0

        _, _, banks_epoch, _ = build_reference_banks(
            spec,
            breaks=bank_breaks,
            time_steps_per_segment=time_steps,   # align with segment discretization
            n_paths=ref_bank_size,
            initial_x=initial_x_global,
            x0_mode=x0_mode_global,
            device=device,
            seed=None,   # don't fix -> fresh randomness each epoch
        )
        return banks_epoch[bank_index].to(device)

    # Fallback x0 sampler if bank is disabled
    if x_domain is None:
        fallback_xmin, fallback_xmax = -1.0, 1.0
    else:
        fallback_xmin, fallback_xmax = x_domain

    mode_fallback, opts_fb = x0_mode
    if mode_fallback == "uniform":
        fb_lo = fallback_xmin if opts_fb.get("x_min") is None else opts_fb["x_min"]
        fb_hi = fallback_xmax if opts_fb.get("x_max") is None else opts_fb["x_max"]
        def sample_x0_fallback(B):
            return maybe_clamp_state(fb_lo + (fb_hi - fb_lo) * torch.rand(B, device=device))
    elif mode_fallback == "fixed":
        x_fixed = float(opts_fb.get("x0", (fallback_xmin + fallback_xmax)/2))
        def sample_x0_fallback(B):
            return maybe_clamp_state(torch.full((B,), x_fixed, device=device, dtype=torch.float32))
    else:
        raise ValueError("x0_mode must be ('uniform', {...}) or ('fixed', {...}).")

    for epoch in range(num_epochs):
        # ---- refresh epoch bank (or None to use fallback) ----
        epoch_bank = make_epoch_x0_bank() if refresh_bank_each_epoch else None
        if (epoch_bank is None) and (ref_bank_size is not None) and (not refresh_bank_each_epoch):
            # build once (lazy) if requested but not per-epoch
            epoch_bank = make_epoch_x0_bank()

        def sample_x0(B: int) -> torch.Tensor:
            if epoch_bank is None:
                return sample_x0_fallback(B)
            idx = torch.randint(0, epoch_bank.shape[0], (B,), device=device)
            return maybe_clamp_state(epoch_bank[idx])

        epoch_loss = epoch_loss_v = epoch_loss_w = 0.0
        phi_hits_total = 0.0
        phi_possible   = 0.0
        psi_valid_steps = 0.0
        psi_total_steps = 0.0

        for start in range(0, training_path_size, nn_batch_size):
            B = min(nn_batch_size, training_path_size - start)
            # 1) sample initial x and Brownian increments
            X_unc = torch.zeros(B, time_steps, device=device)
            X_ctrl = torch.zeros(B, time_steps, device=device)
            X_unc[:, 0] = sample_x0(B)
            X_ctrl[:, 0] = X_unc[:, 0].clone()

            dW  = torch.sqrt(dt_t) * torch.randn(B, time_steps - 1, device=device)
            dWX = torch.sqrt(dt_t) * torch.randn(B, time_steps - 1, device=device)
            # 2) simulate 𝓧 (driftless) & controlled X^a (one a per path for Ψ)
            a_batch = A_min + (A_max - A_min) * torch.rand(B, device=device)
            for i in range(1, time_steps):
                t_prev = t_seq[i - 1]
                # uncontrolled
                sig_u = clamp_sigma(spec.sigma_torch(t_prev, X_unc[:, i - 1]))
                X_unc[:, i] = maybe_clamp_state(X_unc[:, i - 1] + sig_u * dW[:, i - 1])

                # controlled
                sig_c = clamp_sigma(spec.sigma_torch(t_prev, X_ctrl[:, i - 1]))
                drift_inc = spec.b_ctrl_torch(t_prev, X_ctrl[:, i - 1], a_batch) * dt_t
                X_ctrl[:, i] = maybe_clamp_state(X_ctrl[:, i - 1] + drift_inc + sig_c * dWX[:, i - 1])

            # 3) jacobian ∇ and M for Φ
            nabla = torch.zeros(B, time_steps, device=device); nabla[:, 0] = 1.0
            for i in range(1, time_steps):
                t_prev = t_seq[i - 1]
                x_prev = X_unc[:, i - 1]
                sigma_x = spec.sigma_x_torch(t_prev, x_prev)
                expo = sigma_x * dW[:, i - 1] - 0.5 * (sigma_x**2) * dt_t
                if exp_clip is not None:
                    expo = expo.clamp(min=-abs(exp_clip), max=abs(exp_clip))
                nabla[:, i] = (nabla[:, i - 1] * torch.exp(expo)).clamp_min(1e-12)

            sigma_all = clamp_sigma(spec.sigma_torch(t_seq.unsqueeze(0).repeat(B, 1), X_unc))
            eta = (1.0 / sigma_all) * nabla
            M = torch.cumsum(eta[:, :-1] * dW, dim=1)               # [B, T-1]

            # 4) Φ term: λ ln ∫ exp(w/λ) da along 𝓧  (stabilized per timepoint)
            t_vals = t_seq[1:].unsqueeze(0).repeat(B, 1)            # [B, T-1]
            x_vals = X_unc[:, 1:]                                   # [B, T-1]
            t_rep = t_vals.unsqueeze(2).repeat(1, 1, num_a_points)
            x_rep = x_vals.unsqueeze(2).repeat(1, 1, num_a_points)
            a_rep = a_grid.view(1, 1, -1).repeat(B, t_vals.shape[1], 1)

            inp_w_grid = torch.stack([t_rep, x_rep, a_rep], dim=3).view(-1, 3)
            w_out_grid = w_Model(inp_w_grid).view(B, t_vals.shape[1], num_a_points)  # [B, T-1, A]

            S = w_out_grid / lambda_reg
            S_max, _ = torch.max(S, dim=2, keepdim=True)                               # [B,T-1,1]
            int_exp = torch.trapz(torch.exp(S - S_max), a_grid, dim=2).clamp_min(1e-40)# [B,T-1]
            ln_integral = lambda_reg * (torch.log(int_exp) + S_max.squeeze(2))         # [B,T-1]

            # 5) terminal derivative at t1
            X_T = X_unc[:, -1]
            terminal_full = g_x_boundary_fn(X_T) * nabla[:, -1]   # [B]

            if use_masks:
                term_ok = (X_T >= x_min) & (X_T <= x_max)
            else:
                term_ok = torch.ones(B, dtype=torch.bool, device=device)

            phi_possible += float(B)
            phi_hits_total += float(term_ok.sum().item())

            # 6) Φ residuals (masked)
            num_points = time_steps - 1
            Rv_list, Mv_list = [], []
            for k in range(num_points):
                x_k = X_unc[:, k]
                if use_masks:
                    mask_k = (x_k >= x_min) & (x_k <= x_max) & term_ok
                else:
                    mask_k = torch.ones(B, dtype=torch.bool, device=device)

                num_future = time_steps - 1 - k
                t_k = t_seq[k]
                dt_future = (t_seq[k+1:k+1+num_future] - t_k)                 # [nf]
                M_tk = torch.zeros(B, device=device) if k == 0 else M[:, k-1]
                M_future = M[:, k:k+num_future]                               # [B,nf]
                nabla_k = nabla[:, k].clamp_min(1e-12)

                kernel = (M_future - M_tk.unsqueeze(1)) / (dt_future.unsqueeze(0) * nabla_k.unsqueeze(1))
                integral_w_k = (ln_integral[:, k:k+num_future] * kernel * dt_t).sum(dim=1)   # [B]
                phi_k = integral_w_k + (terminal_full / nabla_k)

                v_out = v_Model(torch.stack((t_k * torch.ones(B, device=device), x_k), dim=1)).squeeze()
                Rv_list.append((v_out - phi_k).abs())
                Mv_list.append(mask_k)

            Rv = torch.stack(Rv_list, dim=1) if Rv_list else torch.zeros(B, 1, device=device)
            Mv = torch.stack(Mv_list, dim=1) if Mv_list else torch.zeros(B, 1, device=device)
            loss_v = reduce_residuals_masked(Rv, Mv, norm_type=norm_type, lp_p=lp_p, softmax_tau=softmax_tau)

            # 7) Ψ residuals (masked on both cur_x and next_X)
            Rw_list, Mw_list = [], []
            for k in range(num_points):
                t_k = t_seq[k]
                cur_x = X_unc[:, k]
                sig_k = clamp_sigma(spec.sigma_torch(t_k, cur_x))
                dW_k = dW[:, k]

                next_c = maybe_clamp_state(cur_x + sig_k * dW_k)
                drift_k = spec.b_ctrl_torch(t_k, cur_x, a_batch) * dt_t
                next_X = maybe_clamp_state(cur_x + drift_k + sig_k * dW_k)

                if use_masks:
                    mask_k = (cur_x >= x_min) & (cur_x <= x_max) & (next_X >= x_min) & (next_X <= x_max)
                else:
                    mask_k = torch.ones(B, dtype=torch.bool, device=device)

                psi_total_steps += float(B)
                psi_valid_steps += float(mask_k.sum().item())

                lo_c = torch.minimum(cur_x, next_c); up_c = torch.maximum(cur_x, next_c)
                lo_X = torch.minimum(cur_x, next_X); up_X = torch.maximum(cur_x, next_X)

                int_c = integral_v_trap(v_Model, t_k, lo_c, up_c, steps=int_x_steps,
                                        x_floor=spec.x_floor, clamp_state=spec.clamp_state) * torch.sign(next_c - cur_x)
                int_X = integral_v_trap(v_Model, t_k, lo_X, up_X, steps=int_x_steps,
                                        x_floor=spec.x_floor, clamp_state=spec.clamp_state) * torch.sign(next_X - cur_x)

                reward_k = spec.running_reward_torch(t_k, next_X, a_batch) * dt_t
                psi_k = (int_X - int_c + reward_k) / dt_t

                w_out = w_Model(torch.stack((t_k * torch.ones(B, device=device), cur_x, a_batch), dim=1)).squeeze()

                Rw_list.append((w_out - psi_k).abs())
                Mw_list.append(mask_k)

            Rw = torch.stack(Rw_list, dim=1) if Rw_list else torch.zeros(B, 1, device=device)
            Mw = torch.stack(Mw_list, dim=1) if Mw_list else torch.zeros(B, 1, device=device)
            loss_w = reduce_residuals_masked(Rw, Mw, norm_type=norm_type, lp_p=lp_p, softmax_tau=softmax_tau)

            # 8) update
            J_batch = loss_v + loss_w
            optim.zero_grad()
            J_batch.backward()
            optim.step()

            epoch_loss += J_batch.item()
            epoch_loss_v += loss_v.item()
            epoch_loss_w += loss_w.item()

        denom = max(1, math.ceil(training_path_size / nn_batch_size))
        avg_epoch_loss = epoch_loss / denom
        avg_lv = epoch_loss_v / denom
        avg_lw = epoch_loss_w / denom
        epochs_run += 1

        phi_hit = (phi_hits_total / max(1.0, phi_possible))
        psi_cov = (psi_valid_steps / max(1.0, psi_total_steps))
        vprint(f"[{t0:.3f}→{t1:.3f}] Epoch {epochs_run} | J: {avg_epoch_loss:.6f} | "
               f"Lv: {avg_lv:.6f} | Lw: {avg_lw:.6f} | Φ-hit: {phi_hit:.2f} | Ψ-cov: {psi_cov:.2f}")

        if avg_epoch_loss < best_loss - es_min_delta:
            best_loss = avg_epoch_loss
            patience_ctr = 0
            best_v = {k: v.clone().detach().cpu() for k, v in v_Model.state_dict().items()}
            best_w = {k: v.clone().detach().cpu() for k, v in w_Model.state_dict().items()}
        else:
            patience_ctr += 1
            if early_stopping and patience_ctr >= es_patience:
                vprint(f"[{t0:.3f}→{t1:.3f}] Early stop (patience={es_patience}, Δ={es_min_delta}).")
                stopped_early = True
                break

    if 'best_v' in locals():
        v_Model.load_state_dict(best_v)
        w_Model.load_state_dict(best_w)

    stats = dict(
        t0=t0, t1=t1, time_steps=time_steps, epochs_run=epochs_run,
        best_loss=best_loss, stopped_early=stopped_early,
        x_domain=x_domain, norm_type=norm_type, lp_p=lp_p, softmax_tau=softmax_tau
    )
    return v_Model, w_Model, stats


# ==============================================================================
#                      MULTI-PERIOD WRAPPER (BACKWARD STITCH)
# ==============================================================================
def train_multi_period(
    spec: ProblemSpec,
    # segmentation
    T: Optional[float] = None, n_segments: Optional[int] = None,
    time_breaks: Optional[List[float]] = None,   # if provided, overrides n_segments
    # domains & sampling
    x_domain: Optional[Tuple[float, float]] = None,
    x0_mode: Tuple[str, dict] = ("uniform", {"x_min": None, "x_max": None}),
    initial_x: Optional[float] = None,           # earliest segment fixed x0 if provided
    mask_mode: str = "none",
    # per-segment grid/MC
    time_steps: int = 21, training_path_size: int = 20000, nn_batch_size: int = 10000,
    num_epochs: int = 100,
    ref_bank_size: Optional[int] = None,         # per-epoch bank size (if None -> fallback)
    seed: Optional[int] = None,                  # kept for API symmetry
    # models/opt
    neuron_number_v: int = 128, neuron_number_w: int = 128,
    learning_rate: float = 5e-4, weight_decay: float = 0.0,
    # actions/entropy
    A_min: Optional[float] = None, A_max: Optional[float] = None,
    num_a_points: int = 160, lambda_reg: Optional[float] = None,
    # Ψ integration
    int_x_steps: int = 50,
    # norms
    norm_type: str = "lp", lp_p: float = 8.0, softmax_tau: str | float = "auto",
    # early stop
    early_stopping: bool = True, es_patience: int = 20, es_min_delta: float = 5e-4,
    # misc
    device: Optional[torch.device | str] = None, verbose: bool = True,
    # numerics
    exp_clip: Optional[float] = None,
    # bank refresh toggle
    refresh_bank_each_epoch: bool = True,
):
    """
    Train across multiple segments backward in time.
    Now each segment self-refreshes its x0-bank every epoch using the GLOBAL
    initial law (initial_x or x0_mode) propagated under 𝓧 from time 0 to τ_j.
    """
    device = normalize_device(device)
    vprint = print if verbose else (lambda *a, **k: None)

    if T is None: T = spec.T
    if A_min is None: A_min = spec.A_min
    if A_max is None: A_max = spec.A_max
    if lambda_reg is None: lambda_reg = spec.lambda_reg

    # build breaks
    if time_breaks is None:
        assert n_segments is not None and n_segments >= 1
        breaks = list(np.linspace(0.0, T, n_segments + 1))
    else:
        breaks = list(sorted(time_breaks))
        assert abs(breaks[0]) < 1e-12 and abs(breaks[-1] - T) < 1e-12, "breaks must start at 0 and end at T"
    m = len(breaks) - 1

    v_models: List[nn.Module] = [None] * m
    w_models: List[nn.Module] = [None] * m
    stats_list: List[dict]    = []

    # boundary g_x provider for last segment is true g_x at T
    def gx_true_T(x):
        T_t = torch.tensor(T, dtype=torch.float32, device=x.device)
        return spec.g_x_torch(x, T_t)

    next_v_model = None
    for j in reversed(range(m)):
        t0, t1 = float(breaks[j]), float(breaks[j+1])

        # sampling fallback for this segment if bank disabled
        if j == 0 and (initial_x is not None):
            seg_x0_mode = ("fixed", {"x0": float(initial_x)})
        else:
            seg_x0_mode = x0_mode

        if next_v_model is None:
            g_x_boundary = gx_true_T
        else:
            t1_t = torch.tensor(t1, dtype=torch.float32, device=device)
            def g_x_boundary(x, _t1=t1_t, _v=next_v_model):
                with torch.no_grad():
                    inp = torch.stack([_t1.expand_as(x), x], dim=1)
                    return _v(inp).squeeze(-1)

        vprint(f"\n=== Train segment [{t0:.4f} → {t1:.4f}] "
               f"{'(fixed x0 at earliest)' if (j==0 and initial_x is not None) else ''} ===")

        v_mod, w_mod, stats = train_single_segment(
            spec, t0, t1,
            x_domain=x_domain, x0_mode=seg_x0_mode, mask_mode=mask_mode,
            time_steps=time_steps,
            training_path_size=training_path_size, nn_batch_size=nn_batch_size,
            neuron_number_v=neuron_number_v, neuron_number_w=neuron_number_w,
            learning_rate=learning_rate, weight_decay=weight_decay,
            A_min=A_min, A_max=A_max, num_a_points=num_a_points, lambda_reg=lambda_reg,
            int_x_steps=int_x_steps,
            norm_type=norm_type, lp_p=lp_p, softmax_tau=softmax_tau,
            g_x_boundary=g_x_boundary,
            num_epochs=num_epochs, early_stopping=early_stopping,
            es_patience=es_patience, es_min_delta=es_min_delta,
            device=device, verbose=verbose,
            exp_clip=exp_clip,
            # NEW: epoch-bank settings
            ref_bank_size=ref_bank_size,
            refresh_bank_each_epoch=refresh_bank_each_epoch,
            initial_x_global=initial_x,
            x0_mode_global=x0_mode,
        )

        v_models[j] = v_mod
        w_models[j] = w_mod
        stats_list.append(stats)
        next_v_model = v_mod  # for the previous (earlier) segment

    return v_models, w_models, breaks, list(reversed(stats_list))


# ==============================================================================
#                    MULTI-PERIOD FEYNMAN–KAC RECOVERY OF u
# ==============================================================================
@torch.no_grad()
def recover_u_multi(spec: ProblemSpec, w_models: List[nn.Module], breaks: List[float],
                    t0: float, x0: float,
                    num_a_points: int = 160, lambda_reg: Optional[float] = None,
                    time_steps_per_segment: int = 21, n_paths: int = 10000,
                    device: Optional[torch.device | str] = None):
    device = normalize_device(device)
    if lambda_reg is None: lambda_reg = spec.lambda_reg

    Ts = breaks[-1]
    def maybe_clamp_state(x: torch.Tensor) -> torch.Tensor:
        return x.clamp_min(spec.x_floor) if getattr(spec, "clamp_state", False) else x
    def seg_index(t):
        if abs(t - Ts) < 1e-12: return len(breaks) - 2
        for j in range(len(breaks) - 1):
            if breaks[j] <= t < breaks[j+1]:
                return j
        return len(breaks) - 2

    j0 = seg_index(t0)

    t_grid_abs = [t0]
    for j in range(j0, len(breaks) - 1):
        start = max(t0, breaks[j])
        end   = breaks[j+1]
        K = time_steps_per_segment
        if end - start <= 0: continue
        tg = np.linspace(start, end, K, endpoint=True)
        if len(t_grid_abs) > 0 and abs(t_grid_abs[-1] - tg[0]) < 1e-12:
            t_grid_abs.extend(list(tg[1:]))
        else:
            t_grid_abs.extend(list(tg))

    t_seq = torch.tensor(t_grid_abs, dtype=torch.float32, device=device)
    dt_seq = t_seq[1:] - t_seq[:-1]
    M = t_seq.numel()
    if M <= 1:
        X_T = torch.tensor([x0], dtype=torch.float32, device=device)
        u_hat = spec.g_value_torch(X_T, torch.tensor(Ts, device=device)).mean().item()
        return u_hat, 0.0

    def clamp_sigma(sig: torch.Tensor, eps=1e-6):
        return sig.sign() * torch.clamp(sig.abs(), min=eps)

    X = torch.zeros(n_paths, M, device=device); X[:, 0] = torch.tensor(x0, device=device)
    for k in range(1, M):
        t_prev = t_seq[k - 1]
        sig = clamp_sigma(spec.sigma_torch(t_prev, X[:, k - 1]))
        dW = torch.sqrt(dt_seq[k - 1]) * torch.randn(n_paths, device=device)
        X[:, k] = maybe_clamp_state(X[:, k - 1] + sig * dW)

    a_grid = torch.linspace(spec.A_min, spec.A_max, num_a_points, device=device)
    H_sum = torch.zeros(n_paths, device=device)

    for k in range(1, M):
        t_k = t_seq[k]
        x_k = X[:, k]
        j = 0
        for jj in range(len(breaks) - 1):
            if (breaks[jj] <= t_k < breaks[jj+1]) or (k == M - 1 and abs(t_k - breaks[-1]) < 1e-12):
                j = jj; break
        wj = w_models[j].to(device).eval()

        t_rep = t_k.expand(n_paths, num_a_points)
        x_rep = x_k.unsqueeze(1).expand(n_paths, num_a_points)
        a_rep = a_grid.view(1, -1).expand(n_paths, num_a_points)

        inp = torch.stack([t_rep, x_rep, a_rep], dim=2).reshape(-1, 3)
        w_out = wj(inp).view(n_paths, num_a_points)

        S = w_out / lambda_reg
        S_max, _ = torch.max(S, dim=1, keepdim=True)
        int_exp = torch.trapz(torch.exp(S - S_max), a_grid, dim=1).clamp_min(1e-40)
        log_int = torch.log(int_exp) + S_max.squeeze(1)
        H_vals = lambda_reg * log_int
        H_sum += H_vals * dt_seq[k - 1]

    g_term = spec.g_value_torch(X[:, -1], torch.tensor(Ts, device=device))
    u_samples = g_term + H_sum
    u_hat = u_samples.mean().item()
    u_se  = u_samples.std(unbiased=True).item() / math.sqrt(n_paths)
    return u_hat, u_se


# ==============================================================================
#                                 PROBING HELPERS
# ==============================================================================
@torch.no_grad()
def probe_v_at(v_models: List[nn.Module], breaks: List[float],
               t0: float, x0: float, device: Optional[torch.device | str] = None):
    device = normalize_device(device)
    if abs(t0 - breaks[-1]) < 1e-12:
        j = len(breaks) - 2
    else:
        j = max(0, next(i for i in range(len(breaks) - 1) if breaks[i] <= t0 < breaks[i+1]))
    vj = v_models[j].to(device).eval()
    t = torch.tensor([t0], dtype=torch.float32, device=device)
    x = torch.tensor([x0], dtype=torch.float32, device=device)
    return vj(torch.stack([t, x], dim=1)).item()


# ==============================================================================
#                                   EXAMPLE RUN
# ==============================================================================
if __name__ == "__main__":
    device = normalize_device(None)

    spec = EX7(T=0.4, A_min=0.0, A_max=1.0, lambda_reg=5.0)
    initial_x = 0.0  # earliest segment starts at a fixed point

    v_models, w_models, breaks, stats = train_multi_period(
        spec,
        n_segments=2, time_steps=6, num_epochs=200,
        x_domain=None, mask_mode="none",
        x0_mode=("uniform", {"x_min": 0.4, "x_max": 1.6}),
        initial_x=initial_x,
        training_path_size=5000, nn_batch_size=5000,
        neuron_number_v=64, neuron_number_w=64,
        learning_rate=5e-4, weight_decay=0.0,
        num_a_points=96, lambda_reg=spec.lambda_reg,
        int_x_steps=96,
        norm_type="softmax", softmax_tau="auto",
        early_stopping=True, es_patience=12, es_min_delta=5e-4,
        ref_bank_size=5000,                     # per-epoch bank size
        seed=None,
        device=device, verbose=False,
        exp_clip=None,
        refresh_bank_each_epoch=True,           # <-- NEW: refresh each epoch
    )

    t0, x0 = 0.0, initial_x
    v_hat = probe_v_at(v_models, breaks, t0=t0, x0=x0, device=device)
    print(f"\nProbe: v(t0={t0}, x0={x0}) ≈ {v_hat:.6f} | True {spec.u_x_true_np(t0, x0):.6f}")

    u_hat, u_se = recover_u_multi(spec, w_models, breaks, t0=t0, x0=x0,
                                  num_a_points=100, time_steps_per_segment=11, n_paths=10000, device=device)
    print(f"FK u(t0={t0}, x0={x0}) ≈ {u_hat:.6f}  (SE≈{u_se:.6f}) | True {spec.u_true_np(t0, x0):.6f}")
