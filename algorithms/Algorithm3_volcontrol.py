# ==============================================================================
#  1-D Infinite-Horizon Volatility Control (J-minimization with 4 nets)
# ==============================================================================
# - Time-homogeneous problem; discount factor rho > 0
# - Reference processes:
#     𝓧^x_t     = x + B_t                     (driftless Brownian reference)
#     𝓧^{x,a}_t  ≈ x + σ(x,a) ΔW               (one-step σ-only reference)
# - True controlled:
#     X^{x,a}_t ≈ x + b(x,a) Δt + σ(x,a) ΔW   (Euler one-step for \tilde w)
# - Four maps (discretized):
#     \tilde u(x)  ≈ ∑ e^{-ρ t_k} H(t_k, 𝓧^x_{t_k}) Δt
#     \tilde v(x)  ≈ ∑ e^{-ρ t_k} (B_{t_k}/t_k) H(t_k, 𝓧^x_{t_k}; θ(𝓧^x_{t_k})) Δt
#     \tilde w(x,a) ≈ [ (∫_x^{X^{x,a}_{Δt}} v - ∫_x^{𝓧^{x,a}_{Δt}} v) / Δt ] + r(x,a)
#     \tilde θ(x)  = 2ρ u(x) - 2 * H(x; θ(x), w)
#
#   where H(t, ξ) = λ log ∫_A exp{ [ (1/2) σ1^2(ξ,a) θ(ξ) + w(ξ,a) ] / λ } da
#         Ĥ uses θ(x) (fixed at the *initial* x) per your definition for \tilde v.
#
# - J(u,v,w,θ) = ||\tilde u - u|| + ||\tilde v - v|| + ||\tilde w - w|| + ||\tilde θ - θ||
#   (we use mean absolute error over the training batch; you can change to Lp if desired)
#
# - Includes diagnostics against the manufactured ground truth you provided.
# ==============================================================================

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple

# ------------------------------ Device & seeds --------------------------------
def get_device(dev: Optional[str | torch.device] = None):
    if dev is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(dev) if isinstance(dev, str) else dev

def set_seed(seed: Optional[int] = None):
    if seed is not None:
        torch.manual_seed(seed); np.random.seed(seed)

# ------------------------------ Models ----------------------------------------
def mlp_1in(width=128, out_dim=1):  # for u(x), v(x), θ(x)
    net = nn.Sequential(
        nn.Linear(1, width), nn.Tanh(),
        nn.Linear(width, width), nn.Tanh(),
        nn.Linear(width, width), nn.Tanh(),
        #nn.Linear(width, width), nn.Tanh(),
        nn.Linear(width, out_dim)
    )
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    return net

def mlp_2in(width=128, out_dim=1):  # for w(x,a)
    net = nn.Sequential(
        nn.Linear(2, width), nn.Tanh(),
        nn.Linear(width, width), nn.Tanh(),
        nn.Linear(width, width), nn.Tanh(),
        nn.Linear(width, width), nn.Tanh(),
        nn.Linear(width, out_dim)
    )
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    return net

def soft_sup(r, tau, dim=-1):
    m = torch.amax(r, dim=dim, keepdim=True)
    return tau * torch.logsumexp((r - m) / tau, dim=dim) + m.squeeze(dim)

def softmax_reduce_over_any(residuals: torch.Tensor, tau="auto", reduce_dims=(-1,)):
    """
    Soft sup over the given reduce_dims.
    Uses the same τ rule you had: tau ≈ 0.5 * median / log(K), clamped to [1e-4, 1.0],
    with K = number of elements being reduced.
    Returns a scalar.
    """
    # flatten the reduce dims into the last axis
    if isinstance(reduce_dims, int):
        reduce_dims = (reduce_dims,)
    keep_dims = [i for i in range(residuals.dim()) if i not in reduce_dims]
    perm = keep_dims + list(reduce_dims)
    R = residuals.permute(*perm)
    K = int(np.prod([residuals.size(d) for d in reduce_dims])) or 1
    R_flat_last = R.reshape(*R.shape[:len(keep_dims)], K)

    if tau == "auto":
        med = R_flat_last.detach().median().item() if R_flat_last.numel() else 0.0
        denom = max(1.0, math.log(max(2, K)))
        tau_val = max(1e-4, min(1.0, 0.5 * med / denom))
    else:
        tau_val = float(tau)

    # soft sup over the last axis; then average over remaining axes to get a scalar
    out = soft_sup(R_flat_last, tau=tau_val, dim=-1)
    return out.mean()

# ------------------------------ Problem Spec ----------------------------------
class Vol1DSpec:
    """
    1-D volatility control, time-homogeneous (infinite horizon).
    Example provided by user (manufactured solution).
    """
    def __init__(self,
                 A_min: float = 0.0,
                 A_max: float = 1.0,
                 lambda_reg: float = 1.0,
                 rho: float = 1.0,
                 eps_sigma: float = 1e-6):
        self.A_min = float(A_min)
        self.A_max = float(A_max)
        self.lambda_reg = float(lambda_reg)
        self.rho = float(rho)
        self.eps_sigma = float(eps_sigma)

    # ----- Manufactured truth (diagnostics) -----
    @staticmethod
    def u_true_np(x: float) -> float:
        return float(np.exp(-x**2))

    @staticmethod
    def v_true_np(x: float) -> float:
        return float(-2.0 * x * np.exp(-x**2))

    @staticmethod
    def theta_true_np(x: float) -> float:
        # u_xx = (4x^2 - 2) u
        u = np.exp(-x**2)
        return float((4.0 * x**2 - 2.0) * u)

    # ----- Building blocks: b, sigma_1^2, sigma, r -----
    def b_torch(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # b(x,a) = x * exp(-2(x^2 + a))
        #return 5*x+a+7
        return x * torch.exp(-2.0 * (x**2 + a))

    def sigma1_sq_torch(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # σ1^2(x,a) = exp(-(x^2 + a))   ∈ (0,1]
        #return a**2+6*x+8
        return torch.exp(-(x**2 + a))

    def sigma_torch(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # σ^2 = 1 + σ1^2
        s2 = 1.0 + self.sigma1_sq_torch(x, a)
        # clamp to keep away from 0 (though s2>=1 so this is just defensive)
        return torch.sqrt(torch.clamp(s2, min=self.eps_sigma))

    def r_torch(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # r(x,a) = (e^{-2(x^2+a)} - 2x^2 + ρ + 1) u*(x)
        u = torch.exp(-x**2)
        return (torch.exp(-2.0 * (x**2 + a)) - 2.0 * x**2 + self.rho + 1.0) * u


class Vol1DScaledSpec:
    """
    1-D volatility control, time-homogeneous (infinite horizon), scaled by c>0.

    Manufactured solution (user-specified):
        u*(x)      = exp(-(x/c)^2)
        v*(x)      = -(2/c^2) x u*(x)
        theta*(x)  = ((4/c^4) x^2 - 2/c^2) u*(x)
        b(x,a)     = (x/c^2) * exp(-((x/c)^2 + a/c))
        sigma_0^2  = 1
        sigma_1^2  = exp(-((x/c)^2 + a/c))
        r(x,a)     = ( (1/c^2) exp(-((x/c)^2 + a/c)) - 2 x^2 / c^4 + rho + 1/c^2 ) * u*(x)

    HJB exactness note:
        With this r, the HJB holds exactly when |A| = A_max - A_min = 1 (e.g., A=[0,1]).
        If you use another |A|, to preserve exactness adjust r by subtracting lambda_reg * log(|A|).

    API matches your training code:
        - b_torch(x,a)
        - sigma1_sq_torch(x,a)
        - sigma_torch(x,a)  = sqrt(1 + sigma1_sq)
        - r_torch(x,a)
        - u_true_np, v_true_np, theta_true_np (for diagnostics)
    """

    def __init__(self,
                 c: float = 1.0,
                 A_min: float = 0.0,
                 A_max: float = 1.0,
                 lambda_reg: float = 1.0,
                 rho: float = 1.0,
                 eps_sigma: float = 1e-6):
        if c <= 0:
            raise ValueError("c must be positive.")
        self.c = float(c)
        self.A_min = float(A_min)
        self.A_max = float(A_max)
        self.lambda_reg = float(lambda_reg)
        self.rho = float(rho)
        self.eps_sigma = float(eps_sigma)

    # ---------- Manufactured truths (diagnostics) ----------
    def u_true_np(self, x: float) -> float:
        c = self.c
        return float(np.exp(-(x / c) ** 2))

    def v_true_np(self, x: float) -> float:
        c = self.c
        u = np.exp(-(x / c) ** 2)
        return float(-(2.0 / c**2) * x * u)

    def theta_true_np(self, x: float) -> float:
        c = self.c
        u = np.exp(-(x / c) ** 2)
        return float(((4.0 / c**4) * x**2 - (2.0 / c**2)) * u)

    # ---------- Building blocks for training ----------
    def b_torch(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        b(x,a) = (x/c^2) * exp(-((x/c)^2 + a/c))
        Shapes: x[...,], a[...]
        """
        c = self.c
        return (x / (c**2)) * torch.exp(-((x / c) ** 2 + a / c))

    def sigma1_sq_torch(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        sigma_1^2(x,a) = exp(-((x/c)^2 + a/c))
        """
        c = self.c
        return torch.exp(-((x / c) ** 2 + a / c))

    def sigma_torch(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        sigma(x,a) = sqrt( 1 + sigma_1^2(x,a) )
        """
        s2 = 1.0 + self.sigma1_sq_torch(x, a)
        return torch.sqrt(torch.clamp(s2, min=self.eps_sigma))

    def r_torch(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        r(x,a) = ( (1/c^2) * exp(-((x/c)^2 + a/c)) - 2 x^2 / c^4 + rho + 1/c^2 ) * u*(x),
        where u*(x) = exp(-(x/c)^2).
        """
        c = self.c
        u_star = torch.exp(- (x / c) ** 2)
        term1 = (1.0 / c**2) * torch.exp(-((x / c) ** 2 + a / c))
        term2 = - (2.0 / c**4) * (x ** 2)
        const = self.rho + (1.0 / c**2)
        return (term1 + term2 + const) * u_star

class Vol1DScaledSpec2:
    """
    1-D volatility control, time-homogeneous (infinite horizon), scaled by c>0.

    Manufactured solution (user-specified):
        u*(x)      = exp(-(x/c)^2)
        v*(x)      = -(2/c^2) x u*(x)
        theta*(x)  = ((4/c^4) x^2 - 2/c^2) u*(x)
        b(x,a)     = (x/c^2) * exp(-((x/c)^2 + a))
        sigma_0^2  = 1
        sigma_1^2  = exp(-((x/c)^2 + a))
        r(x,a)     = ( (1/c^2) exp(-((x/c)^2 + a)) - 2 x^2 / c^4 + rho + 1/c^2 ) * u*(x)

    HJB exactness note:
        With this r, the HJB holds exactly when |A| = A_max - A_min = 1 (e.g., A=[0,1]).
        If you use another |A|, to preserve exactness adjust r by subtracting lambda_reg * log(|A|).

    API matches your training code:
        - b_torch(x,a)
        - sigma1_sq_torch(x,a)
        - sigma_torch(x,a)  = sqrt(1 + sigma1_sq)
        - r_torch(x,a)
        - u_true_np, v_true_np, theta_true_np (for diagnostics)
    """

    def __init__(self,
                 c: float = 1.0,
                 A_min: float = 0.0,
                 A_max: float = 1.0,
                 lambda_reg: float = 1.0,
                 rho: float = 1.0,
                 eps_sigma: float = 1e-6):
        if c <= 0:
            raise ValueError("c must be positive.")
        self.c = float(c)
        self.A_min = float(A_min)
        self.A_max = float(A_max)
        self.lambda_reg = float(lambda_reg)
        self.rho = float(rho)
        self.eps_sigma = float(eps_sigma)

    # ---------- Manufactured truths (diagnostics) ----------
    def u_true_np(self, x: float) -> float:
        c = self.c
        return float(np.exp(-(x / c) ** 2))

    def v_true_np(self, x: float) -> float:
        c = self.c
        u = np.exp(-(x / c) ** 2)
        return float(-(2.0 / c**2) * x * u)

    def theta_true_np(self, x: float) -> float:
        c = self.c
        u = np.exp(-(x / c) ** 2)
        return float(((4.0 / c**4) * x**2 - (2.0 / c**2)) * u)

    # ---------- Building blocks for training ----------
    def b_torch(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        b(x,a) = (x/c^2) * exp(-((x/c)^2 + a))
        Shapes: x[...,], a[...]
        """
        c = self.c
        return (x / (c**2)) * torch.exp(-((x / c) ** 2 + a / c))

    def sigma1_sq_torch(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        sigma_1^2(x,a) = exp(-((x/c)^2 + a))
        """
        c = self.c
        return torch.exp(-((x / c) ** 2 + a / c))

    def sigma_torch(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        sigma(x,a) = sqrt( 1 + sigma_1^2(x,a) )
        """
        s2 = 1.0 + self.sigma1_sq_torch(x, a)
        return torch.sqrt(torch.clamp(s2, min=self.eps_sigma))

    def r_torch(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        r(x,a) = ( (1/c^2) * exp(-((x/c)^2 + a)) - 2 x^2 / c^4 + rho + 1/c^2 ) * u*(x),
        where u*(x) = exp(-(x/c)^2).
        """
        c = self.c
        u_star = torch.exp(- (x / c) ** 2)
        term1 = (1.0 / c**2) * torch.exp(-((x / c) ** 2 + a ))
        term2 = - (2.0 / c**4) * (x ** 2)
        const = self.rho + (1.0 / c**2)
        return (term1 + term2 + const) * u_star


# ------------------------------ Helpers ---------------------------------------
#@torch.no_grad()
def integral_v_trap(v_model: nn.Module, x0: torch.Tensor, x1: torch.Tensor, steps: int = 50) -> torch.Tensor:
    """
    ∫_{x0}^{x1} v(y) dy, vectorized over batch (same steps for all).
    Returns [B] tensor. Handles sign automatically via orientation.
    """
    B = x0.shape[0]
    y0 = x0; y1 = x1
    delta = (y1 - y0) / steps  # [B]
    ar = torch.linspace(0.0, 1.0, steps + 1, device=x0.device)  # [S+1]
    grid = y0.unsqueeze(1) + delta.unsqueeze(1) * ar.unsqueeze(0)  # [B,S+1]
    v_vals = v_model(grid.unsqueeze(-1)).squeeze(-1)  # [B,S+1]
    trap = (delta / 2.0) * (v_vals[:, 0] + v_vals[:, -1] + 2.0 * v_vals[:, 1:-1].sum(dim=1))
    return trap

def logint_exp_over_a(stabiland: torch.Tensor, a_grid: torch.Tensor, lam: float) -> torch.Tensor:
    """
    Given 'stabiland' = (1/λ)[ ... ] with shape [*, A], compute
      λ * log ∫_A exp(stabiland) da
    using trapezoidal rule and row-wise stabilization.
    Returns [*].
    """
    # stabiland: [N, A]
    m, _ = torch.max(stabiland, dim=-1, keepdim=True)  # [N,1]
    int_exp = torch.trapz(torch.exp(stabiland - m), a_grid, dim=-1).clamp_min(1e-40)  # [N]
    return lam * (torch.log(int_exp) + m.squeeze(-1))

# -------------------------- Core batch estimators ------------------------------
def tilde_u_batch(spec: Vol1DSpec,
                  theta_model: nn.Module, w_model: nn.Module,
                  x0: torch.Tensor,
                  T_cut: float, time_steps: int,
                  num_a_points: int,
                  device: torch.device) -> torch.Tensor:
    """
    Compute \tilde u(x) for a batch x0: R^B → R^B.
    Discretize ∫_0^∞ e^{-ρ t} H(t, 𝓧^x_t) dt by [0, T_cut] grid with discount weights.
    """
    B = x0.shape[0]
    # time grid (exclude t=0 to avoid division issues elsewhere)
    t = torch.linspace(0.0, T_cut, time_steps, device=device)
    dt = t[1:] - t[:-1]                     # [K-1]
    t_mid = t[1:]                           # right-end discretization
    disc = torch.exp(-spec.rho * t_mid)     # [K-1]

    # simulate reference Brownian 𝓧^x_t = x + B_t
    dW = torch.sqrt(dt).unsqueeze(0) * torch.randn(B, dt.numel(), device=device)  # [B, K-1]
    B_t = torch.cumsum(dW, dim=1)                                               # [B, K-1]
    X_ref = x0.unsqueeze(1) + B_t                                               # [B, K-1]

    # Build action grid
    a_grid = torch.linspace(spec.A_min, spec.A_max, num_a_points, device=device)  # [A]

    # Evaluate H at each (x_ref, a)
    # theta at x_ref
    theta_vals = theta_model(X_ref.unsqueeze(-1)).squeeze(-1)                    # [B, K-1]
    # w(x_ref, a) over grid
    N = B * (time_steps - 1)
    xr_flat = X_ref.reshape(N)                                                   # [N]
    a_rep = a_grid.unsqueeze(0).expand(N, -1)                                    # [N, A]
    x_rep = xr_flat.unsqueeze(1).expand_as(a_rep)                                # [N, A]
    w_flat = w_model(torch.stack([x_rep, a_rep], dim=-1)).squeeze(-1)            # [N, A]
    sigma1_sq = spec.sigma1_sq_torch(x_rep, a_rep)                               # [N, A]

    lam = spec.lambda_reg
    theta_flat = theta_vals.reshape(N, 1)                                        # [N,1]
    stabiland = (0.5 * sigma1_sq * theta_flat + w_flat) / lam                   # [N, A]
    H_flat = logint_exp_over_a(stabiland, a_grid, lam)                           # [N]
    H = H_flat.reshape(B, -1)                                                    # [B, K-1]

    # Riemann sum with discount
    integrand = disc.unsqueeze(0) * H                                            # [B, K-1]
    tilde_u = torch.sum(integrand * dt.unsqueeze(0), dim=1)                      # [B]
    return tilde_u

def tilde_v_batch(spec: Vol1DSpec,
                  theta_model: nn.Module, w_model: nn.Module,
                  x0: torch.Tensor,
                  T_cut: float, time_steps: int,
                  num_a_points: int,
                  device: torch.device) -> torch.Tensor:
    """
    Compute \tilde v(x) for a batch x0 using the (B_t / t) kernel,
    with θ evaluated along the reference path: θ(𝓧_t^x).
    """
    B = x0.shape[0]
    t = torch.linspace(0.0, T_cut, time_steps, device=device)
    dt = t[1:] - t[:-1]
    t_mid = t[1:]
    disc = torch.exp(-spec.rho * t_mid)

    # Brownian reference
    dW = torch.sqrt(dt).unsqueeze(0) * torch.randn(B, dt.numel(), device=device)  # [B, K-1]
    B_t = torch.cumsum(dW, dim=1)                                                # [B, K-1]
    X_ref = x0.unsqueeze(1) + B_t                                                # [B, K-1]

    # Kernel B_t / t (t>0 by construction)
    kernel = B_t / t_mid.unsqueeze(0)                                            # [B, K-1]

    # Action grid
    a_grid = torch.linspace(spec.A_min, spec.A_max, num_a_points, device=device) # [A]

    # --- θ along the path instead of θ(x0) ----
    theta_vals = theta_model(X_ref.unsqueeze(-1)).squeeze(-1)                    # [B, K-1]

    # w(X_ref, a) on the grid
    N = B * (time_steps - 1)
    xr_flat = X_ref.reshape(N)                                                   # [N]
    a_rep   = a_grid.unsqueeze(0).expand(N, -1)                                  # [N, A]
    x_rep   = xr_flat.unsqueeze(1).expand_as(a_rep)                              # [N, A]
    w_flat  = w_model(torch.stack([x_rep, a_rep], dim=-1)).squeeze(-1)           # [N, A]
    sigma1_sq = spec.sigma1_sq_torch(x_rep, a_rep)                               # [N, A]

    lam = spec.lambda_reg
    theta_flat = theta_vals.reshape(N, 1)                                        # [N, 1]
    stabiland = (0.5 * sigma1_sq * theta_flat + w_flat) / lam                    # [N, A]
    H_flat = logint_exp_over_a(stabiland, a_grid, lam)                            # [N]
    H = H_flat.reshape(B, -1)                                                    # [B, K-1]

    integrand = disc.unsqueeze(0) * kernel * H                                   # [B, K-1]
    tilde_v = torch.sum(integrand * dt.unsqueeze(0), dim=1)                      # [B]
    return tilde_v


def tilde_w_batch(spec: Vol1DSpec,
                  v_model: nn.Module,
                  x0: torch.Tensor,
                  num_a_samples: int,
                  dt_psi: float,
                  trap_steps: int,
                  device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute \tilde w(x,a) for a batch x0 and a batch of sampled a's.
    Returns:
        tilde_w_vals: [B] for the sampled actions,
        a_samples:    [B] actions (for pairing with w(x,a))
    """
    B = x0.shape[0]
    # sample actions per path (Uniform over A)
    a_lo, a_hi = spec.A_min, spec.A_max
    # Either draw one action per x, or average over a few and return mean — we’ll draw ONE per sample:
    a = a_lo + (a_hi - a_lo) * torch.rand(B, device=device)

    # One-step Euler with same ΔW for controlled vs σ-only reference
    dW = math.sqrt(dt_psi) * torch.randn(B, device=device)
    dWX = math.sqrt(dt_psi) * torch.randn(B, device=device)

    sig = spec.sigma_torch(x0, a)             # [B]
    x_refa = x0 + sig * dW                    # 𝓧^{x,a}_{Δt} ≈ x + σ(x,a) ΔW
    x_ctrl = x0 + spec.b_torch(x0, a) * dt_psi + sig * dWX

    # line integrals of v between x and the two endpoints
    int_ctrl = integral_v_trap(v_model, x0, x_ctrl, steps=trap_steps)
    int_refa = integral_v_trap(v_model, x0, x_refa, steps=trap_steps)

    # r(x,a) at *x0*
    reward = spec.r_torch(x0, a) * dt_psi

    tilde_w = (int_ctrl - int_refa + reward) / dt_psi  # [B]
    return tilde_w, a

def tilde_w_grid(spec: Vol1DSpec,
                 v_model: nn.Module,
                 x0: torch.Tensor,
                 a_grid: torch.Tensor,
                 dt_psi: float,
                 trap_steps: int) -> torch.Tensor:
    """
    Returns \tilde w(x,a) on a grid: shape [B, A].
    """
    B, A = x0.shape[0], a_grid.numel()
    x_rep = x0.unsqueeze(1).expand(B, A)                 # [B,A]
    a_rep = a_grid.unsqueeze(0).expand(B, A)             # [B,A]

    dW  = math.sqrt(dt_psi) * torch.randn(B, A, device=x0.device)
    dWX = math.sqrt(dt_psi) * torch.randn(B, A, device=x0.device)

    sig    = spec.sigma_torch(x_rep, a_rep)              # [B,A]
    x_refa = x_rep + sig * dW
    x_ctrl = x_rep + spec.b_torch(x_rep, a_rep) * dt_psi + sig * dWX

    int_ctrl = integral_v_trap(v_model, x_rep.reshape(-1),  x_ctrl.reshape(-1),  steps=trap_steps).reshape(B, A)
    int_refa = integral_v_trap(v_model, x_rep.reshape(-1),  x_refa.reshape(-1),  steps=trap_steps).reshape(B, A)

    reward = spec.r_torch(x_rep, a_rep) * dt_psi         # [B,A]
    return (int_ctrl - int_refa + reward) / dt_psi       # [B,A]

def tilde_theta_batch(spec: Vol1DSpec,
                      u_model: nn.Module, theta_model: nn.Module, w_model: nn.Module,
                      x0: torch.Tensor,
                      num_a_points: int,
                      device: torch.device) -> torch.Tensor:
    """
    Compute \tilde θ(x) = 2ρ u(x) - 2 * H(x; θ(x), w(·)).
    Note: H here depends on θ(x) itself (fixed-point form), per your definition.
    """
    B = x0.shape[0]
    u_vals = u_model(x0.unsqueeze(-1)).squeeze(-1)        # [B]
    theta_x = theta_model(x0.unsqueeze(-1)).squeeze(-1)   # [B]

    a_grid = torch.linspace(spec.A_min, spec.A_max, num_a_points, device=device)  # [A]
    x_rep = x0.unsqueeze(1).expand(B, a_grid.numel())     # [B, A]
    a_rep = a_grid.unsqueeze(0).expand_as(x_rep)          # [B, A]

    w_vals = w_model(torch.stack([x_rep, a_rep], dim=-1)).squeeze(-1)   # [B, A]
    sigma1_sq = spec.sigma1_sq_torch(x_rep, a_rep)                      # [B, A]

    lam = spec.lambda_reg
    theta_rep = theta_x.unsqueeze(1)                                     # [B,1]
    stabiland = (0.5 * sigma1_sq * theta_rep + w_vals) / lam            # [B, A]
    H_vals = logint_exp_over_a(stabiland, a_grid, lam)                  # [B]

    tilde_theta = 2.0 * spec.rho * u_vals - 2.0 * H_vals
    return tilde_theta

# ------------------------------ Training loop ---------------------------------
def train_vol1d(
    spec: Vol1DSpec,
    # domain & sampling of x
    x_domain: Tuple[float, float] = (-1.5, 1.5),
    x0_mode: Tuple[str, dict] = ("uniform", {"x_min": None, "x_max": None}),
    # time discretization for the ∫ e^{-ρ t} dt parts
    T_cut: float = 2.0, time_steps: int = 41,
    # Ψ one-step settings
    dt_psi: float = 1e-2, trap_steps: int = 50,
    # MC & epochs
    training_path_size: int = 4000, nn_batch_size: int = 1000, num_epochs: int = 200,
    # nets & opt
    width_u: int = 128, width_v: int = 128, width_w: int = 128, width_theta: int = 128,
    learning_rate: float = 5e-4, weight_decay: float = 0.0,
    # action quadrature/sampling
    num_a_points: int = 160,
    # loss weights
    w_u: float = 1.0, w_v: float = 1.0, w_w: float = 1.0, w_theta: float = 1.0,
    # early stopping
    early_stopping: bool = True, es_patience: int = 25, es_min_delta: float = 5e-4,
    # misc
    device: Optional[str | torch.device] = None, seed: Optional[int] = None, verbose: bool = True,
):
    """
    Train four models (u, v, w, θ) jointly by minimizing J on the batch.
    """
    device = get_device(device)
    set_seed(seed)

    # Build models
    u_model     = mlp_1in(width_u).to(device)
    v_model     = mlp_1in(width_v).to(device)
    w_model     = mlp_2in(width_w).to(device)
    theta_model = mlp_1in(width_theta).to(device)

    params = list(u_model.parameters()) + list(v_model.parameters()) + \
             list(w_model.parameters()) + list(theta_model.parameters())
    optim = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)

    # x sampler
    x_min, x_max = x_domain
    mode, opts = x0_mode
    if opts.get("x_min") is None: opts["x_min"] = x_min
    if opts.get("x_max") is None: opts["x_max"] = x_max

    def sample_x0(B: int) -> torch.Tensor:
        if mode == "uniform":
            lo, hi = float(opts["x_min"]), float(opts["x_max"])
            return lo + (hi - lo) * torch.rand(B, device=device)
        elif mode == "fixed":
            return torch.full((B,), float(opts["x0"]), device=device)
        else:
            raise ValueError("x0_mode must be ('uniform', {...}) or ('fixed', {...})")

    # train
    best = float('inf'); patience = 0; epochs_run = 0
    lam = spec.lambda_reg

    for ep in range(num_epochs):
        epoch_losses = []
        for start in range(0, training_path_size, nn_batch_size):
            B = min(nn_batch_size, training_path_size - start)
            x0 = sample_x0(B)

            # Forward: compute the four tilde maps
            #with torch.no_grad():
            tu = tilde_u_batch(spec, theta_model, w_model, x0,
                            T_cut=T_cut, time_steps=time_steps,
                            num_a_points=num_a_points, device=device)
            tv = tilde_v_batch(spec, theta_model, w_model, x0,
                                   T_cut=T_cut, time_steps=time_steps,
                                   num_a_points=num_a_points, device=device)
                # tw, a_samp = tilde_w_batch(spec, v_model, x0,
                #                            num_a_samples=1, dt_psi=dt_psi,
                #                            trap_steps=trap_steps, device=device)
            tt = tilde_theta_batch(spec, u_model, theta_model, w_model,
                                    x0, num_a_points=num_a_points, device=device)

            # Networks' outputs at needed inputs
            u_pred = u_model(x0.unsqueeze(-1)).squeeze(-1)                # [B]
            v_pred = v_model(x0.unsqueeze(-1)).squeeze(-1)                # [B]
            #w_pred = w_model(torch.stack([x0, a_samp], dim=-1)).squeeze(-1)  # [B]
            th_pred = theta_model(x0.unsqueeze(-1)).squeeze(-1)           # [B]

            # residuals
            Ru  = torch.abs(u_pred  - tu)             # [B]
            Rv  = torch.abs(v_pred  - tv)             # [B]
            Rth = torch.abs(th_pred - tt)             # [B]

            # build w on an action grid to approximate sup_a
            a_grid = torch.linspace(spec.A_min, spec.A_max, num_a_points, device=device)
            x_rep = x0.unsqueeze(1).expand(x0.shape[0], a_grid.numel())                 # [B,A]
            a_rep = a_grid.unsqueeze(0).expand_as(x_rep)                                # [B,A]
            w_pred_grid = w_model(torch.stack([x_rep, a_rep], dim=-1)).squeeze(-1)      # [B,A]
            tw_grid = tilde_w_grid(spec, v_model, x0, a_grid, dt_psi, trap_steps)       # [B,A]
            Rw = torch.abs(w_pred_grid - tw_grid)                                       # [B,A]

            # the same tau="auto" rule you used before
            soft_tau = "auto"

            # sup over x  -> reduce_dims=(0,)
            loss_u  = softmax_reduce_over_any(Ru,  tau=soft_tau, reduce_dims=(0,))
            loss_v  = softmax_reduce_over_any(Rv,  tau=soft_tau, reduce_dims=(0,))
            loss_th = softmax_reduce_over_any(Rth, tau=soft_tau, reduce_dims=(0,))

            # sup over x and a  -> reduce_dims=(0,1)
            loss_w  = softmax_reduce_over_any(Rw,  tau=soft_tau, reduce_dims=(0,1))

            J = w_u * loss_u + w_v * loss_v + w_w * loss_w + w_theta * loss_th


            optim.zero_grad()
            J.backward()
            optim.step()

            epoch_losses.append((loss_u.item(), loss_v.item(), loss_w.item(), loss_th.item(), J.item()))

        epochs_run += 1
        mu = np.mean(epoch_losses, axis=0)
        if verbose:
            print(f"Epoch {epochs_run:4d} | "
                  f"Lu={mu[0]:.6f} Lv={mu[1]:.6f} Lw={mu[2]:.6f} Lθ={mu[3]:.6f} | J={mu[4]:.6f}")

        # early stopping on total J
        cur = mu[4]
        if cur < best - es_min_delta:
            best = cur; patience = 0
            best_state = dict(
                u=u_model.state_dict(), v=v_model.state_dict(),
                w=w_model.state_dict(), th=theta_model.state_dict()
            )
        else:
            patience += 1
            if early_stopping and patience >= es_patience:
                if verbose:
                    print(f"[Early stop] No J improvement ≥ {es_min_delta} for {es_patience} epochs.")
                break

    # restore best
    if 'best_state' in locals():
        u_model.load_state_dict(best_state['u'])
        v_model.load_state_dict(best_state['v'])
        w_model.load_state_dict(best_state['w'])
        theta_model.load_state_dict(best_state['th'])

    meta = dict(
        epochs_run=epochs_run, best_J=best, T_cut=T_cut, time_steps=time_steps,
        dt_psi=dt_psi, trap_steps=trap_steps, num_a_points=num_a_points,
        x_domain=x_domain, rho=spec.rho, lambda_reg=lam
    )
    return u_model, v_model, w_model, theta_model, meta

# ------------------------------ Diagnostics -----------------------------------
@torch.no_grad()
def quick_diagnostics(spec: Vol1DSpec, u_model, v_model, theta_model,
                      grid: torch.Tensor):
    u_pred = u_model(grid.unsqueeze(-1)).squeeze(-1).cpu().numpy()
    v_pred = v_model(grid.unsqueeze(-1)).squeeze(-1).cpu().numpy()
    th_pred = theta_model(grid.unsqueeze(-1)).squeeze(-1).cpu().numpy()

    u_true = np.exp(-grid.cpu().numpy()**2)
    v_true = -2.0 * grid.cpu().numpy() * u_true
    th_true = (4.0 * grid.cpu().numpy()**2 - 2.0) * u_true

    err_u = np.mean(np.abs(u_pred - u_true))
    err_v = np.mean(np.abs(v_pred - v_true))
    err_th = np.mean(np.abs(th_pred - th_true))
    return dict(L1_u=err_u, L1_v=err_v, L1_theta=err_th)

# ------------------------------ Example run -----------------------------------
if __name__ == "__main__":
    device = get_device(None)
    spec = Vol1DScaledSpec(A_min=0.0, A_max=1.0, lambda_reg=1.0, rho=40.0,c=1.0)

    x0 = 1.0

    u_model, v_model, w_model, theta_model, meta = train_vol1d(
        spec,
        x_domain=(-1.5, 1.5),
        x0_mode=("fixed", {"x0": x0}),
        #x0_mode=("uniform", {"x_min": -0.2, "x_max": 0.2}),
        T_cut=4.0, time_steps=11,
        dt_psi=1e-2, trap_steps=32,
        training_path_size=8000, nn_batch_size=2000, num_epochs=400,
        width_u=32, width_v=32, width_w=32, width_theta=32,
        learning_rate=0.01, weight_decay=0.0,
        num_a_points=64,
        w_u=1.0, w_v=1.0, w_w=1.0, w_theta=1.0,
        early_stopping=True, es_patience=15, es_min_delta=1e-4,
        device=device, seed=None, verbose=True
    )

    # Quick probe at x0
    
    with torch.no_grad():
        x0_torch = torch.tensor([x0], device=device)
        u0 = u_model(x0_torch.unsqueeze(-1)).item()
        v0 = v_model(x0_torch.unsqueeze(-1)).item()
        th0 = theta_model(x0_torch.unsqueeze(-1)).item()
    print(f"\nProbe at x={x0}:  "
          f"u≈{u0:.6f} (true {spec.u_true_np(x0):.6f}),  "
          f"v≈{v0:.6f} (true {spec.v_true_np(x0):.6f}),  "
          f"θ≈{th0:.6f} (true {spec.theta_true_np(x0):.6f})")

    # Grid diagnostics
    xs = torch.linspace(-1.0, 1.0, 81, device=device)
    diag = quick_diagnostics(spec, u_model, v_model, theta_model, xs)
    print(f"Grid L1 errors | u: {diag['L1_u']:.4e} | v: {diag['L1_v']:.4e} | θ: {diag['L1_theta']:.4e}")

    print("\nMeta:", meta)
