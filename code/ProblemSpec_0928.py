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
        # no state clamp needed; sigma never vanishes
        self.x_floor = 0.0

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
        # exact: no clamp needed since 2+sin >= 1
        return 2.0 + torch.sin(t + x)

    def sigma_x_torch(self, t, x):
        # d/dx (2+sin(t+x)) = cos(t+x)
        return torch.cos(t + x)

    def running_reward_torch(self, t, x, a):
        s = t + x
        return 2.0*torch.cos(s) + a*torch.sin(s)





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
    
class EXTrue(ProblemSpec):
    """
    Unregularized ground-truth example:
        dX = (x + a) dt + 1 dW,   a in [0,1]
        u(t,x) = exp( -(t^2 + x^2 + 1) )
      -> r(t,x,a) = (2t + 2 a x + 1) * u(t,x)
    This class plugs into the single-period training API (which still uses λ).
    """
    def __init__(self, T=0.1, A_min=0.0, A_max=1.0, lambda_reg=1.0, x_floor=-1e9):
        # NOTE: lambda_reg is only used by the *regularized* trainer.
        self.T = float(T)
        self.A_min, self.A_max = float(A_min), float(A_max)
        self.lambda_reg = float(lambda_reg)
        # We set a very negative x_floor so any "clamp_min(x_floor)" in legacy code
        # does not artificially force X>=0 (since σ=1 doesn’t need it).
        self.x_floor = float(x_floor)

    # ---- (optional) closed forms ----
    def u_true_np(self, t, x):
        return float(np.exp(-(t**2 + x**2 + 1.0)))

    def u_x_true_np(self, t, x):
        u = np.exp(-(t**2 + x**2 + 1.0))
        return float(-2.0 * x * u)

    # ---- terminal g and its derivative ----
    @torch.no_grad()
    def g_value_torch(self, x, T):
        # x: [B]
        return torch.exp(-(x**2 + T**2 + 1.0))

    def g_x_torch(self, x, T):
        # ∂_x g = -2x * g
        g = self.g_value_torch(x, T)
        return -2.0 * x * g

    # ---- controlled drift b(t,x,a) ----
    def b_ctrl_torch(self, t, x, a):
        # works with broadcasting; returns [B]
        return x + a

    # ---- diffusion and its x-derivative ----
    def sigma_torch(self, t, x):
        # constant σ=1 (no need for floors)
        return torch.ones_like(x)

    def sigma_x_torch(self, t, x):
        # d/dx σ = 0
        return torch.zeros_like(x)

    # ---- running reward r(t,x,a) ----
    def running_reward_torch(self, t, x, a):
        u = torch.exp(-(t**2 + x**2 + 1.0))
        return (2.0 * t + 2.0 * a * x + 1.0) * u
    
class EXTrue2(ProblemSpec):
    """
    Unregularized ground-truth example:
      u(t,x)  = cos(t + x),  g(x)=u(T,x)=cos(T + x)
      u_x     = -sin(t + x),  u_xx = -cos(t + x),  u_t = -sin(t + x)
      σ(t,x)  = 2 + sin(t + x)   (≥1 -> no clamping needed)
      b(t,x,a)= a - [1 + 2 cos(t+x) + 0.5 sin(t+x) cos(t+x)]
      -> r(t,x,a) = sin(t+x)-b*u_x-0.5*sigma^2*u_xx
    This class plugs into the single-period training API (which still uses λ).
    """
    def __init__(self, T=0.1, A_min=0.0, A_max=1.0, lambda_reg=1.0, x_floor=-1e9):
        # NOTE: lambda_reg is only used by the *regularized* trainer.
        self.T = float(T)
        self.A_min, self.A_max = float(A_min), float(A_max)
        self.lambda_reg = float(lambda_reg)

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
        # exact: no clamp needed since 2+sin >= 1
        return 2.0 + torch.sin(t + x)

    def sigma_x_torch(self, t, x):
        # d/dx (2+sin(t+x)) = cos(t+x)
        return torch.cos(t + x)

    # ---- running reward r(t,x,a) ----
    def running_reward_torch(self, t, x, a):
        return torch.sin(t+x)+self.b_ctrl_torch(t, x, a)*torch.sin(t+x)+0.5*self.sigma_torch( t, x)*self.sigma_torch( t, x)*torch.cos(t+x)
