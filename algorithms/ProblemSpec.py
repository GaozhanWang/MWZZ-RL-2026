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


# class EX4(ProblemSpec):
#     """
#       u(t,x)  = exp( -(t^2 + x^2 + 1) ),  g(x)=u(T,x)
#       g_x     = -2x u
#       b(t,x,a)= x^2 + a - 0.5
#       σ(t,x)  = sqrt(x),  σ_x = 1/(2 sqrt(x))
#       r(t,x,a)= (2t + 2ax) u
#     """
#     def __init__(self):
#         self.T = 0.1
#         self.A_min, self.A_max = 0.0, 1.0
#         self.lambda_reg = 1.0
#         self.x_floor = 1e-2
#         # sqrt(x) requires X≥0 in simulation; enable clamping
#         self.clamp_state = True

#     def u_true_np(self, t, x):
#         return float(np.exp(-(t**2 + x**2 + 1.0)))

#     def u_x_true_np(self, t, x):
#         return float(-2.0 * x * np.exp(-(t**2 + x**2 + 1.0)))

#     @torch.no_grad()
#     def g_value_torch(self, x, T):
#         return torch.exp(-(x**2 + T**2 + 1.0))

#     def g_x_torch(self, x, T):
#         return -2.0 * x * torch.exp(-(x**2 + T**2 + 1.0))

#     def b_ctrl_torch(self, t, x, a):
#         return x**2 + a - 0.5

#     def sigma_torch(self, t, x):
#         return torch.sqrt(x.clamp(min=self.x_floor))

#     def sigma_x_torch(self, t, x):
#         sig = self.sigma_torch(t, x).clamp(min=1e-6)
#         return 0.5 / sig  # d/dx sqrt(x) = 1/(2 sqrt(x))

#     def running_reward_torch(self, t, x, a):
#         u = torch.exp(-(t**2 + x**2 + 1.0))
#         return (2.0 * t + 2.0 * a * x) * u


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


# import torch
# import numpy as np
# from scipy.interpolate import RegularGridInterpolator

# class EX4_FDM(ProblemSpec):
#     """
#     FDM-approximated ground-truth example based on EX4.
#     - u_lam (Regularized) is provided analytically.
#     - u_0 (Unregularized) is solved via explicit FDM.
#     """
#     def __init__(self, T=0.1, A_min=0.0, A_max=1.0, lambda_reg=1.0, x_floor=1e-2,
#                  Nx=400, x_lim=6.0, Nt=2000, Na=50):
#         self.T = float(T)
#         self.A_min, self.A_max = float(A_min), float(A_max)
#         self.lambda_reg = float(lambda_reg)
#         self.x_floor = float(x_floor)
#         self.clamp_state = True
        
#         # FDM Grid Parameters
#         self.Nx = Nx
#         self.x_lim = float(x_lim) 
#         self.Nt = Nt
#         self.Na = Na 
        
#         # Run the FDM solver ONCE to cache the unregularized u_0 interpolators
#         self._interp_u_0, self._interp_u_0_x = self._solve_unregularized_hjb_fdm()

#     # ---- Analytical Regularized Truth (u_lam) ----
#     def u_lam_true_np(self, t, x):
#         """ Analytical true solution for the entropy-regularized problem. """
#         return float(np.exp(-(t**2 + x**2 + 1.0)))

#     def u_lam_x_true_np(self, t, x):
#         return float(-2.0 * x * np.exp(-(t**2 + x**2 + 1.0)))

#     # ---- Interpolated Unregularized Truth (u_0) ----
#     def u_true_np(self, t, x):
#         """ Evaluates the FDM approximated u_0^* at (t, x). """
#         t_arr, x_arr = np.atleast_1d(t), np.atleast_1d(x)
#         pts = np.stack(np.broadcast_arrays(t_arr, x_arr), axis=-1)
#         res = self._interp_u_0(pts)
#         return float(res[0]) if res.size == 1 else res

#     def u_x_true_np(self, t, x):
#         """ Evaluates the FDM approximated spatial derivative of u_0^* at (t, x). """
#         t_arr, x_arr = np.atleast_1d(t), np.atleast_1d(x)
#         pts = np.stack(np.broadcast_arrays(t_arr, x_arr), axis=-1)
#         res = self._interp_u_0_x(pts)
#         return float(res[0]) if res.size == 1 else res

#     def update_lambda(self, new_lambda):
#         """ 
#         For EX4_FDM, the unregularized FDM doesn't depend on lambda, 
#         and the analytical u_lam is invariant to lambda. We just update the attribute.
#         """
#         self.lambda_reg = float(new_lambda)

#     # ---- Terminal g ----
#     def g_value_np(self, x, T):
#         return np.exp(-(x**2 + T**2 + 1.0))

#     @torch.no_grad()
#     def g_value_torch(self, x, T):
#         return torch.exp(-(x**2 + T**2 + 1.0))

#     def g_x_torch(self, x, T):
#         return -2.0 * x * torch.exp(-(x**2 + T**2 + 1.0))

#     # ---- Controlled drift & Diffusion ----
#     def b_ctrl_torch(self, t, x, a):
#         return x**2 + a - 0.5

#     def sigma_torch(self, t, x):
#         return torch.sqrt(x.clamp(min=self.x_floor))

#     def sigma_x_torch(self, t, x):
#         sig = self.sigma_torch(t, x).clamp(min=1e-6)
#         return 0.5 / sig  

#     # ---- Running Reward ----
#     def running_reward_np(self, t, x, a):
#         u_base = np.exp(-(t**2 + x**2 + 1.0))
#         return (2.0 * t + 2.0 * a * x) * u_base

#     def running_reward_torch(self, t, x, a):
#         u_base = torch.exp(-(t**2 + x**2 + 1.0))
#         return (2.0 * t + 2.0 * a * x) * u_base

#     # =========================================================================
#     # FDM SOLVER (UNREGULARIZED)
#     # =========================================================================
#     def _solve_unregularized_hjb_fdm(self):
#         """ Solves the standard unregularized HJB equation (u_0). """
#         x_grid = np.linspace(self.x_floor, self.x_lim, self.Nx)
#         t_grid = np.linspace(0, self.T, self.Nt + 1)
#         a_grid = np.linspace(self.A_min, self.A_max, self.Na)
        
#         dx = x_grid[1] - x_grid[0]
#         dt = t_grid[1] - t_grid[0]
        
#         # CFL condition for explicit scheme: dt <= dx^2 / max(sigma^2)
#         # Here sigma(x) = sqrt(x), so max(sigma^2) = x_lim
#         max_sigma_sq = self.x_lim
#         if dt > 0.5 * dx**2 / max_sigma_sq:
#             print(f"Warning: FDM might be unstable. dt={dt:.5f}, required<={0.5 * dx**2 / max_sigma_sq:.5f}")

#         U = np.zeros((self.Nt + 1, self.Nx))
#         U[-1, :] = self.g_value_np(x_grid, self.T)
        
#         for n in range(self.Nt - 1, -1, -1):
#             t = t_grid[n]
#             u_next = U[n + 1, :]
            
#             u_xx = (u_next[2:] - 2*u_next[1:-1] + u_next[:-2]) / (dx**2)
#             u_x = (u_next[2:] - u_next[:-2]) / (2*dx)
            
#             H_max = np.full(self.Nx - 2, -np.inf)
#             for a in a_grid:
#                 drift = x_grid[1:-1]**2 + a - 0.5
#                 reward = self.running_reward_np(t, x_grid[1:-1], a)
#                 H_max = np.maximum(H_max, drift * u_x + reward)
            
#             # Diffusion term: 0.5 * sigma^2 * u_xx = 0.5 * x * u_xx
#             diffusion = 0.5 * x_grid[1:-1] * u_xx
            
#             U[n, 1:-1] = u_next[1:-1] + dt * (diffusion + H_max)
            
#             # Flat Neumann boundary conditions
#             U[n, 0] = U[n, 1] 
#             U[n, -1] = U[n, -2]

#         U_x = np.gradient(U, x_grid, axis=1)
#         interp_u_0 = RegularGridInterpolator((t_grid, x_grid), U, bounds_error=False, fill_value=None)
#         interp_u_0_x = RegularGridInterpolator((t_grid, x_grid), U_x, bounds_error=False, fill_value=None)
#         return interp_u_0, interp_u_0_x


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
    
import torch
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Assuming ProblemSpec is defined elsewhere
# class ProblemSpec: pass

class EX7_FDM(ProblemSpec):
    """
    FDM-approximated ground-truth example based on EX7.
    - Uniformly non-degenerate (sigma = 1).
    - u_lam (Regularized) is provided analytically.
    - u_0 (Unregularized) is solved via explicit FDM.
    """
    def __init__(self, T=0.1, A_min=0.0, A_max=1.0, lambda_reg=1.0, x_floor=0.0,
                 Nx=800, x_lim=6.0, Nt=8000, Na=100):
        self.T = float(T)
        self.A_min, self.A_max = float(A_min), float(A_max)
        self.lambda_reg = float(lambda_reg)
        self.x_floor = float(x_floor) 
        self.clamp_state = False      
        
        # High-Precision FDM Grid Parameters
        self.Nx = Nx
        self.x_lim = float(x_lim)     
        self.Nt = Nt
        self.Na = Na 
        
        self._interp_u_0, self._interp_u_0_x = self._solve_unregularized_hjb_fdm()

    # ---- Analytical Regularized Truth (u_lam) ----
    def u_lam_true_np(self, t, x):
        """ Analytical true solution for the entropy-regularized problem. """
        return float(np.exp(-(t**2 + x**2 + 1.0)))

    def u_lam_x_true_np(self, t, x):
        return float(-2.0 * x * np.exp(-(t**2 + x**2 + 1.0)))

    # ---- Interpolated Unregularized Truth (u_0) ----
    def u_true_np(self, t, x):
        """ Evaluates the FDM approximated u_0^* at (t, x). """
        t_arr, x_arr = np.atleast_1d(t), np.atleast_1d(x)
        pts = np.stack(np.broadcast_arrays(t_arr, x_arr), axis=-1)
        res = self._interp_u_0(pts)
        return float(res[0]) if res.size == 1 else res

    def u_x_true_np(self, t, x):
        """ Evaluates the FDM approximated spatial derivative of u_0^* at (t, x). """
        t_arr, x_arr = np.atleast_1d(t), np.atleast_1d(x)
        pts = np.stack(np.broadcast_arrays(t_arr, x_arr), axis=-1)
        res = self._interp_u_0_x(pts)
        return float(res[0]) if res.size == 1 else res

    def update_lambda(self, new_lambda):
        """ 
        For EX7_FDM, the unregularized FDM doesn't depend on lambda, 
        and the analytical u_lam is invariant to lambda. 
        """
        self.lambda_reg = float(new_lambda)

    # ---- Terminal g ----
    def g_value_np(self, x, T):
        return np.exp(-(x**2 + T**2 + 1.0))

    @torch.no_grad()
    def g_value_torch(self, x, T):
        return torch.exp(-(T**2 + x**2 + 1.0))

    def g_x_torch(self, x, T):
        return -2.0 * x * torch.exp(-(T**2 + x**2 + 1.0))

    # ---- Controlled drift & Diffusion ----
    def b_ctrl_torch(self, t, x, a):
        return x + a

    def sigma_torch(self, t, x):
        return torch.ones_like(x)

    def sigma_x_torch(self, t, x):
        return torch.zeros_like(x)

    # ---- Running Reward ----
    def running_reward_np(self, t, x, a):
        u_base = np.exp(-(t**2 + x**2 + 1.0))
        return (2.0 * t + 2.0 * a * x + 1.0) * u_base

    def running_reward_torch(self, t, x, a):
        u_base = torch.exp(-(t**2 + x**2 + 1.0))
        return (2.0 * t + 2.0 * a * x + 1.0) * u_base

    # =========================================================================
    # FDM SOLVER (UNREGULARIZED)
    # =========================================================================
    def _solve_unregularized_hjb_fdm(self):
        """ Solves the standard unregularized HJB equation (u_0) with high precision. """
        x_grid = np.linspace(-self.x_lim, self.x_lim, self.Nx)
        t_grid = np.linspace(0, self.T, self.Nt + 1)
        a_grid = np.linspace(self.A_min, self.A_max, self.Na)
        
        dx = x_grid[1] - x_grid[0]
        dt = t_grid[1] - t_grid[0]
        
        # Strict CFL Check
        if dt > 0.5 * dx**2:
            raise ValueError(f"FDM is unstable! dt={dt:.7f}, required<={0.5 * dx**2:.7f}. Increase Nt.")

        U = np.zeros((self.Nt + 1, self.Nx))
        U[-1, :] = self.g_value_np(x_grid, self.T)
        
        for n in range(self.Nt - 1, -1, -1):
            t = t_grid[n]
            u_next = U[n + 1, :]
            
            # Central differences for interior spatial derivatives
            u_xx = (u_next[2:] - 2*u_next[1:-1] + u_next[:-2]) / (dx**2)
            u_x = (u_next[2:] - u_next[:-2]) / (2*dx)
            
            # Sup over actions
            H_max = np.full(self.Nx - 2, -np.inf)
            for a in a_grid:
                drift = x_grid[1:-1] + a
                reward = self.running_reward_np(t, x_grid[1:-1], a)
                H_max = np.maximum(H_max, drift * u_x + reward)
            
            # Diffusion term: 0.5 * sigma^2 * u_xx = 0.5 * 1.0 * u_xx
            diffusion = 0.5 * u_xx
            
            # Update interior points
            U[n, 1:-1] = u_next[1:-1] + dt * (diffusion + H_max)
            
            # DIRICHLET BOUNDARY CONDITIONS: 
            # Since u(t,x) = exp(-(t^2 + x^2 + 1)), it decays to ~0 at x = +/- 6.
            # Forcing it to 0 is highly accurate here and prevents boundary drift.
            U[n, 0] = 0.0
            U[n, -1] = 0.0

        U_x = np.gradient(U, x_grid, axis=1)
        interp_u_0 = RegularGridInterpolator((t_grid, x_grid), U, bounds_error=False, fill_value=None)
        interp_u_0_x = RegularGridInterpolator((t_grid, x_grid), U_x, bounds_error=False, fill_value=None)
        
        print(f"High-Precision FDM Initialized (Nx={self.Nx}, Nt={self.Nt}, Na={self.Na})")
        return interp_u_0, interp_u_0_x
    
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
    
class EXTrue_compare(ProblemSpec):
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
        return float(np.exp(-(t**2 + x**2 + 2.0)))

    def u_x_true_np(self, t, x):
        u = np.exp(-(t**2 + x**2 + 1.0))
        return float(-2.0 * x * u)

    # ---- terminal g and its derivative ----
    @torch.no_grad()
    def g_value_torch(self, x, T):
        # x: [B]
        return torch.exp(-(x**2 + T**2 + 2.0))

    def g_x_torch(self, x, T):
        # ∂_x g = -2x * g
        g = self.g_value_torch(x, T)
        return -2.0 * x * g

    # ---- controlled drift b(t,x,a) ----
    def b_ctrl_torch(self, t, x, a):
        # works with broadcasting; returns [B]
        return x - a

    # ---- diffusion and its x-derivative ----
    def sigma_torch(self, t, x):
        # constant σ=1 (no need for floors)
        return torch.ones_like(x)

    def sigma_x_torch(self, t, x):
        # d/dx σ = 0
        return torch.zeros_like(x)

    # ---- running reward r(t,x,a) ----
    def running_reward_torch(self, t, x, a):
        u = torch.exp(-(t**2 + x**2 + 2.0))
        return (2.0 * t - 2.0 * a * x + 1.0) * u
    
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
    



import torch
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import torch
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Assuming ProblemSpec is defined elsewhere in your codebase
# class ProblemSpec: pass

class EXFDM1(ProblemSpec):
    """
    High-Precision FDM-approximated ground-truth example for perturbed problems.
    Solves BOTH:
      1. The unregularized HJB (u_0)
      2. The entropy-regularized HJB (u_lambda)
    """
    # Increased default resolutions: Nx=800, Nt=8000, Na=100
    def __init__(self, T=0.1, A_min=0.0, A_max=1.0, lambda_reg=1.0, x_floor=-1e9,
                 Nx=800, x_lim=6.0, Nt=8000, Na=100):
        self.T = float(T)
        self.A_min, self.A_max = float(A_min), float(A_max)
        self.lambda_reg = float(lambda_reg)
        self.x_floor = float(x_floor)
        
        # High-Precision FDM Grid Parameters
        self.Nx = Nx
        self.x_lim = float(x_lim) 
        self.Nt = Nt
        self.Na = Na 
        
        # 1. Cache the UNREGULARIZED ground truth (u_0) - Only needs to run once
        self._interp_u, self._interp_u_x = self._solve_hjb_fdm()

        # 2. Cache the REGULARIZED ground truth (u_lambda)
        self._interp_u_lam, self._interp_u_lam_x = None, None
        self.update_lambda(self.lambda_reg)

    def update_lambda(self, new_lambda):
        """ 
        Updates lambda and re-runs the regularized FDM solver. 
        MUST be called in your sweep loop when lambda changes!
        """
        self.lambda_reg = float(new_lambda)
        self._interp_u_lam, self._interp_u_lam_x = self._solve_regularized_hjb_fdm()

    # ---- Perturbed Running Reward ----
    def running_reward_np(self, t, x, a):
        u_base = np.exp(-(t**2 + x**2 + 1.0))
        base_reward = (2.0 * t + 2.0 * a * x + 1.0) * u_base
        perturbation = 0.5 * np.cos(x) 
        return base_reward + perturbation

    def running_reward_torch(self, t, x, a):
        u_base = torch.exp(-(t**2 + x**2 + 1.0))
        base_reward = (2.0 * t + 2.0 * a * x + 1.0) * u_base
        perturbation = 0.5 * torch.cos(x)
        return base_reward + perturbation

    # ---- Terminal g ----
    def g_value_np(self, x, T):
        return np.exp(-(x**2 + T**2 + 1.0))

    @torch.no_grad()
    def g_value_torch(self, x, T):
        return torch.exp(-(x**2 + T**2 + 1.0))

    def g_x_torch(self, x, T):
        g = self.g_value_torch(x, T)
        return -2.0 * x * g

    # ---- Controlled drift & Diffusion ----
    def b_ctrl_torch(self, t, x, a):
        return x + a

    def sigma_torch(self, t, x):
        return torch.ones_like(x)

    def sigma_x_torch(self, t, x):
        return torch.zeros_like(x)

    # =========================================================================
    # FDM SOLVERS
    # =========================================================================
    def _solve_hjb_fdm(self):
        """ Solves the UNREGULARIZED HJB equation (u_0). """
        x_grid = np.linspace(-self.x_lim, self.x_lim, self.Nx)
        t_grid = np.linspace(0, self.T, self.Nt + 1)
        a_grid = np.linspace(self.A_min, self.A_max, self.Na)
        
        dx = x_grid[1] - x_grid[0]
        dt = t_grid[1] - t_grid[0]
        
        # Strict CFL Check
        if dt > 0.5 * dx**2:
            raise ValueError(f"FDM is unstable! dt={dt:.7f}, required<={0.5 * dx**2:.7f}. Increase Nt.")

        U = np.zeros((self.Nt + 1, self.Nx))
        U[-1, :] = self.g_value_np(x_grid, self.T)
        
        for n in range(self.Nt - 1, -1, -1):
            t = t_grid[n]
            u_next = U[n + 1, :]
            
            u_xx = (u_next[2:] - 2*u_next[1:-1] + u_next[:-2]) / (dx**2)
            u_x = (u_next[2:] - u_next[:-2]) / (2*dx)
            
            H_max = np.full(self.Nx - 2, -np.inf)
            for a in a_grid:
                drift = (x_grid[1:-1] + a) * u_x
                reward = self.running_reward_np(t, x_grid[1:-1], a)
                H_max = np.maximum(H_max, drift + reward)
            
            U[n, 1:-1] = u_next[1:-1] + dt * (0.5 * u_xx + H_max)
            
            # DIRICHLET BOUNDARY CONDITIONS
            U[n, 0] = 0.0
            U[n, -1] = 0.0

        U_x = np.gradient(U, x_grid, axis=1)
        interp_u = RegularGridInterpolator((t_grid, x_grid), U, bounds_error=False, fill_value=None)
        interp_u_x = RegularGridInterpolator((t_grid, x_grid), U_x, bounds_error=False, fill_value=None)
        
        print(f"EXFDM1: Unregularized High-Precision FDM Initialized (Nx={self.Nx}, Nt={self.Nt}, Na={self.Na})")
        return interp_u, interp_u_x

    def _solve_regularized_hjb_fdm(self):
        """ Solves the ENTROPY-REGULARIZED HJB equation (u_lambda). """
        x_grid = np.linspace(-self.x_lim, self.x_lim, self.Nx)
        t_grid = np.linspace(0, self.T, self.Nt + 1)
        a_grid = np.linspace(self.A_min, self.A_max, self.Na)
        
        dx = x_grid[1] - x_grid[0]
        dt = t_grid[1] - t_grid[0]
        
        # Strict CFL Check
        if dt > 0.5 * dx**2:
            raise ValueError(f"FDM is unstable! dt={dt:.7f}, required<={0.5 * dx**2:.7f}. Increase Nt.")

        U = np.zeros((self.Nt + 1, self.Nx))
        U[-1, :] = self.g_value_np(x_grid, self.T)
        
        for n in range(self.Nt - 1, -1, -1):
            t = t_grid[n]
            u_next = U[n + 1, :]
            
            u_xx = (u_next[2:] - 2*u_next[1:-1] + u_next[:-2]) / (dx**2)
            u_x = (u_next[2:] - u_next[:-2]) / (2*dx)
            
            # W array: shape (Nx-2, Na)
            W = np.zeros((self.Nx - 2, self.Na))
            for i, a in enumerate(a_grid):
                drift = (x_grid[1:-1] + a) * u_x
                reward = self.running_reward_np(t, x_grid[1:-1], a)
                W[:, i] = drift + reward
            
            # Stabilized Log-Sum-Exp trick for the integral
            W_max = np.max(W, axis=1) 
            
            # Trapz integral of exp((W - W_max)/lambda)
            exp_shifted = np.exp((W - W_max[:, None]) / self.lambda_reg)
            integral_exp = np.trapz(exp_shifted, a_grid, axis=1)
            
            # H_reg = lambda * ln( \int exp ) + W_max
            # clip integral_exp to avoid log(0)
            H_reg = self.lambda_reg * np.log(np.maximum(integral_exp, 1e-40)) + W_max
            
            U[n, 1:-1] = u_next[1:-1] + dt * (0.5 * u_xx + H_reg)
            
            # DIRICHLET BOUNDARY CONDITIONS
            U[n, 0] = 0.0
            U[n, -1] = 0.0

        U_x = np.gradient(U, x_grid, axis=1)
        interp_u_lam = RegularGridInterpolator((t_grid, x_grid), U, bounds_error=False, fill_value=None)
        interp_u_lam_x = RegularGridInterpolator((t_grid, x_grid), U_x, bounds_error=False, fill_value=None)
        return interp_u_lam, interp_u_lam_x

    # =========================================================================
    # INTERPOLATED EVALUATORS
    # =========================================================================
    def u_true_np(self, t, x):
        """ Evaluates the UNREGULARIZED u_0^* at (t, x). """
        t_arr, x_arr = np.atleast_1d(t), np.atleast_1d(x)
        pts = np.stack(np.broadcast_arrays(t_arr, x_arr), axis=-1)
        res = self._interp_u(pts)
        return float(res[0]) if res.size == 1 else res

    def u_lam_true_np(self, t, x):
        """ Evaluates the REGULARIZED u_lambda^* at (t, x). """
        t_arr, x_arr = np.atleast_1d(t), np.atleast_1d(x)
        pts = np.stack(np.broadcast_arrays(t_arr, x_arr), axis=-1)
        res = self._interp_u_lam(pts)
        return float(res[0]) if res.size == 1 else res
    

class EXFDM2(ProblemSpec):
    """
    High-Precision FDM-approximated ground-truth example for perturbed problems.
    Solves BOTH:
      1. The unregularized HJB (u_0)
      2. The entropy-regularized HJB (u_lambda)
    """
    # Increased default resolutions: Nx=800, Nt=8000, Na=100
    def __init__(self, T=0.1, A_min=0.0, A_max=1.0, lambda_reg=1.0, x_floor=-1e9,
                 Nx=800, x_lim=6.0, Nt=8000, Na=100):
        self.T = float(T)
        self.A_min, self.A_max = float(A_min), float(A_max)
        self.lambda_reg = float(lambda_reg)
        self.x_floor = float(x_floor)
        
        # High-Precision FDM Grid Parameters
        self.Nx = Nx
        self.x_lim = float(x_lim) 
        self.Nt = Nt
        self.Na = Na 
        
        # 1. Cache the UNREGULARIZED ground truth (u_0) - Only needs to run once
        self._interp_u, self._interp_u_x = self._solve_hjb_fdm()

        # 2. Cache the REGULARIZED ground truth (u_lambda)
        self._interp_u_lam, self._interp_u_lam_x = None, None
        self.update_lambda(self.lambda_reg)

    def update_lambda(self, new_lambda):
        """ 
        Updates lambda and re-runs the regularized FDM solver. 
        MUST be called in your sweep loop when lambda changes!
        """
        self.lambda_reg = float(new_lambda)
        self._interp_u_lam, self._interp_u_lam_x = self._solve_regularized_hjb_fdm()

    # ---- Perturbed Running Reward ----
    def running_reward_np(self, t, x, a):
        """ Numpy version of the perturbed running reward for the FDM solver. """
        u_base = np.exp(-(t**2 + x**2 + 1.0))
        base_reward = (2.0 * t + 2.0 * a * x + 1.0) * u_base
        # Add a perturbation that breaks the analytical solution but maintains regularity
        perturbation = u_base
        return base_reward + perturbation

    def running_reward_torch(self, t, x, a):
        """ Torch version for the RL training loop. """
        u_base = torch.exp(-(t**2 + x**2 + 1.0))
        base_reward = (2.0 * t + 2.0 * a * x + 1.0) * u_base
        perturbation = u_base
        return base_reward + perturbation


    # ---- Terminal g ----
    def g_value_np(self, x, T):
        return np.exp(-(x**2 + T**2 + 1.0))

    @torch.no_grad()
    def g_value_torch(self, x, T):
        return torch.exp(-(x**2 + T**2 + 1.0))

    def g_x_torch(self, x, T):
        g = self.g_value_torch(x, T)
        return -2.0 * x * g

    # ---- Controlled drift & Diffusion ----
    def b_ctrl_torch(self, t, x, a):
        return x + a

    def sigma_torch(self, t, x):
        return torch.ones_like(x)

    def sigma_x_torch(self, t, x):
        return torch.zeros_like(x)

    # =========================================================================
    # FDM SOLVERS
    # =========================================================================
    def _solve_hjb_fdm(self):
        """ Solves the UNREGULARIZED HJB equation (u_0). """
        x_grid = np.linspace(-self.x_lim, self.x_lim, self.Nx)
        t_grid = np.linspace(0, self.T, self.Nt + 1)
        a_grid = np.linspace(self.A_min, self.A_max, self.Na)
        
        dx = x_grid[1] - x_grid[0]
        dt = t_grid[1] - t_grid[0]
        
        # Strict CFL Check
        if dt > 0.5 * dx**2:
            raise ValueError(f"FDM is unstable! dt={dt:.7f}, required<={0.5 * dx**2:.7f}. Increase Nt.")

        U = np.zeros((self.Nt + 1, self.Nx))
        U[-1, :] = self.g_value_np(x_grid, self.T)
        
        for n in range(self.Nt - 1, -1, -1):
            t = t_grid[n]
            u_next = U[n + 1, :]
            
            u_xx = (u_next[2:] - 2*u_next[1:-1] + u_next[:-2]) / (dx**2)
            u_x = (u_next[2:] - u_next[:-2]) / (2*dx)
            
            H_max = np.full(self.Nx - 2, -np.inf)
            for a in a_grid:
                drift = (x_grid[1:-1] + a) * u_x
                reward = self.running_reward_np(t, x_grid[1:-1], a)
                H_max = np.maximum(H_max, drift + reward)
            
            U[n, 1:-1] = u_next[1:-1] + dt * (0.5 * u_xx + H_max)
            
            # DIRICHLET BOUNDARY CONDITIONS
            U[n, 0] = 0.0
            U[n, -1] = 0.0

        U_x = np.gradient(U, x_grid, axis=1)
        interp_u = RegularGridInterpolator((t_grid, x_grid), U, bounds_error=False, fill_value=None)
        interp_u_x = RegularGridInterpolator((t_grid, x_grid), U_x, bounds_error=False, fill_value=None)
        
        print(f"EXFDM1: Unregularized High-Precision FDM Initialized (Nx={self.Nx}, Nt={self.Nt}, Na={self.Na})")
        return interp_u, interp_u_x

    def _solve_regularized_hjb_fdm(self):
        """ Solves the ENTROPY-REGULARIZED HJB equation (u_lambda). """
        x_grid = np.linspace(-self.x_lim, self.x_lim, self.Nx)
        t_grid = np.linspace(0, self.T, self.Nt + 1)
        a_grid = np.linspace(self.A_min, self.A_max, self.Na)
        
        dx = x_grid[1] - x_grid[0]
        dt = t_grid[1] - t_grid[0]
        
        # Strict CFL Check
        if dt > 0.5 * dx**2:
            raise ValueError(f"FDM is unstable! dt={dt:.7f}, required<={0.5 * dx**2:.7f}. Increase Nt.")

        U = np.zeros((self.Nt + 1, self.Nx))
        U[-1, :] = self.g_value_np(x_grid, self.T)
        
        for n in range(self.Nt - 1, -1, -1):
            t = t_grid[n]
            u_next = U[n + 1, :]
            
            u_xx = (u_next[2:] - 2*u_next[1:-1] + u_next[:-2]) / (dx**2)
            u_x = (u_next[2:] - u_next[:-2]) / (2*dx)
            
            # W array: shape (Nx-2, Na)
            W = np.zeros((self.Nx - 2, self.Na))
            for i, a in enumerate(a_grid):
                drift = (x_grid[1:-1] + a) * u_x
                reward = self.running_reward_np(t, x_grid[1:-1], a)
                W[:, i] = drift + reward
            
            # Stabilized Log-Sum-Exp trick for the integral
            W_max = np.max(W, axis=1) 
            
            # Trapz integral of exp((W - W_max)/lambda)
            exp_shifted = np.exp((W - W_max[:, None]) / self.lambda_reg)
            integral_exp = np.trapz(exp_shifted, a_grid, axis=1)
            
            # H_reg = lambda * ln( \int exp ) + W_max
            # clip integral_exp to avoid log(0)
            H_reg = self.lambda_reg * np.log(np.maximum(integral_exp, 1e-40)) + W_max
            
            U[n, 1:-1] = u_next[1:-1] + dt * (0.5 * u_xx + H_reg)
            
            # DIRICHLET BOUNDARY CONDITIONS
            U[n, 0] = 0.0
            U[n, -1] = 0.0

        U_x = np.gradient(U, x_grid, axis=1)
        interp_u_lam = RegularGridInterpolator((t_grid, x_grid), U, bounds_error=False, fill_value=None)
        interp_u_lam_x = RegularGridInterpolator((t_grid, x_grid), U_x, bounds_error=False, fill_value=None)
        return interp_u_lam, interp_u_lam_x

    # =========================================================================
    # INTERPOLATED EVALUATORS
    # =========================================================================
    def u_true_np(self, t, x):
        """ Evaluates the UNREGULARIZED u_0^* at (t, x). """
        t_arr, x_arr = np.atleast_1d(t), np.atleast_1d(x)
        pts = np.stack(np.broadcast_arrays(t_arr, x_arr), axis=-1)
        res = self._interp_u(pts)
        return float(res[0]) if res.size == 1 else res

    def u_lam_true_np(self, t, x):
        """ Evaluates the REGULARIZED u_lambda^* at (t, x). """
        t_arr, x_arr = np.atleast_1d(t), np.atleast_1d(x)
        pts = np.stack(np.broadcast_arrays(t_arr, x_arr), axis=-1)
        res = self._interp_u_lam(pts)
        return float(res[0]) if res.size == 1 else res


class EXFDM3(ProblemSpec):
    """
    High-Precision FDM-approximated ground-truth example for perturbed problems.
    Solves BOTH:
      1. The unregularized HJB (u_0)
      2. The entropy-regularized HJB (u_lambda)
      3. Scaled version 5 times the control influence in the drift, i.e. b(t,x,a)=x+5a, which makes the problem more sensitive to control errors and thus a better stress test for algorithms.
    """
    # Increased default resolutions: Nx=800, Nt=8000, Na=100
    def __init__(self, T=0.1, A_min=0.0, A_max=1.0, lambda_reg=1.0, x_floor=-1e9,
                 Nx=800, x_lim=6.0, Nt=8000, Na=100):
        self.T = float(T)
        self.A_min, self.A_max = float(A_min), float(A_max)
        self.lambda_reg = float(lambda_reg)
        self.x_floor = float(x_floor)
        
        # High-Precision FDM Grid Parameters
        self.Nx = Nx
        self.x_lim = float(x_lim) 
        self.Nt = Nt
        self.Na = Na 
        
        # 1. Cache the UNREGULARIZED ground truth (u_0) - Only needs to run once
        self._interp_u, self._interp_u_x = self._solve_hjb_fdm()

        # 2. Cache the REGULARIZED ground truth (u_lambda)
        self._interp_u_lam, self._interp_u_lam_x = None, None
        self.update_lambda(self.lambda_reg)

    def update_lambda(self, new_lambda):
        """ 
        Updates lambda and re-runs the regularized FDM solver. 
        MUST be called in your sweep loop when lambda changes!
        """
        self.lambda_reg = float(new_lambda)
        self._interp_u_lam, self._interp_u_lam_x = self._solve_regularized_hjb_fdm()

    # ---- Perturbed Running Reward ----
    def running_reward_np(self, t, x, a):
        """ Numpy version of the perturbed running reward for the FDM solver. """
        u_base = np.exp(-(t**2 + x**2 + 1.0))
        base_reward = (2.0 * t + 2.0 * 5.0*a * x + 1.0) * u_base
        # Add a perturbation that breaks the analytical solution but maintains regularity
        perturbation = u_base
        return base_reward + perturbation

    def running_reward_torch(self, t, x, a):
        """ Torch version for the RL training loop. """
        u_base = torch.exp(-(t**2 + x**2 + 1.0))
        base_reward = (2.0 * t + 2.0 * 5.0*a * x + 1.0) * u_base
        perturbation = u_base
        return base_reward + perturbation


    # ---- Terminal g ----
    def g_value_np(self, x, T):
        return np.exp(-(x**2 + T**2 + 1.0))

    @torch.no_grad()
    def g_value_torch(self, x, T):
        return torch.exp(-(x**2 + T**2 + 1.0))

    def g_x_torch(self, x, T):
        g = self.g_value_torch(x, T)
        return -2.0 * x * g

    # ---- Controlled drift & Diffusion ----
    def b_ctrl_torch(self, t, x, a):
        return x + 5.0*a

    def sigma_torch(self, t, x):
        return torch.ones_like(x)

    def sigma_x_torch(self, t, x):
        return torch.zeros_like(x)

    # =========================================================================
    # FDM SOLVERS
    # =========================================================================
    def _solve_hjb_fdm(self):
        """ Solves the UNREGULARIZED HJB equation (u_0). """
        x_grid = np.linspace(-self.x_lim, self.x_lim, self.Nx)
        t_grid = np.linspace(0, self.T, self.Nt + 1)
        a_grid = np.linspace(self.A_min, self.A_max, self.Na)
        
        dx = x_grid[1] - x_grid[0]
        dt = t_grid[1] - t_grid[0]
        
        # Strict CFL Check
        if dt > 0.5 * dx**2:
            raise ValueError(f"FDM is unstable! dt={dt:.7f}, required<={0.5 * dx**2:.7f}. Increase Nt.")

        U = np.zeros((self.Nt + 1, self.Nx))
        U[-1, :] = self.g_value_np(x_grid, self.T)
        
        for n in range(self.Nt - 1, -1, -1):
            t = t_grid[n]
            u_next = U[n + 1, :]
            
            u_xx = (u_next[2:] - 2*u_next[1:-1] + u_next[:-2]) / (dx**2)
            u_x = (u_next[2:] - u_next[:-2]) / (2*dx)
            
            H_max = np.full(self.Nx - 2, -np.inf)
            for a in a_grid:
                drift = (x_grid[1:-1] + a) * u_x
                reward = self.running_reward_np(t, x_grid[1:-1], a)
                H_max = np.maximum(H_max, drift + reward)
            
            U[n, 1:-1] = u_next[1:-1] + dt * (0.5 * u_xx + H_max)
            
            # DIRICHLET BOUNDARY CONDITIONS
            U[n, 0] = 0.0
            U[n, -1] = 0.0

        U_x = np.gradient(U, x_grid, axis=1)
        interp_u = RegularGridInterpolator((t_grid, x_grid), U, bounds_error=False, fill_value=None)
        interp_u_x = RegularGridInterpolator((t_grid, x_grid), U_x, bounds_error=False, fill_value=None)
        
        print(f"EXFDM1: Unregularized High-Precision FDM Initialized (Nx={self.Nx}, Nt={self.Nt}, Na={self.Na})")
        return interp_u, interp_u_x

    def _solve_regularized_hjb_fdm(self):
        """ Solves the ENTROPY-REGULARIZED HJB equation (u_lambda). """
        x_grid = np.linspace(-self.x_lim, self.x_lim, self.Nx)
        t_grid = np.linspace(0, self.T, self.Nt + 1)
        a_grid = np.linspace(self.A_min, self.A_max, self.Na)
        
        dx = x_grid[1] - x_grid[0]
        dt = t_grid[1] - t_grid[0]
        
        # Strict CFL Check
        if dt > 0.5 * dx**2:
            raise ValueError(f"FDM is unstable! dt={dt:.7f}, required<={0.5 * dx**2:.7f}. Increase Nt.")

        U = np.zeros((self.Nt + 1, self.Nx))
        U[-1, :] = self.g_value_np(x_grid, self.T)
        
        for n in range(self.Nt - 1, -1, -1):
            t = t_grid[n]
            u_next = U[n + 1, :]
            
            u_xx = (u_next[2:] - 2*u_next[1:-1] + u_next[:-2]) / (dx**2)
            u_x = (u_next[2:] - u_next[:-2]) / (2*dx)
            
            # W array: shape (Nx-2, Na)
            W = np.zeros((self.Nx - 2, self.Na))
            for i, a in enumerate(a_grid):
                drift = (x_grid[1:-1] + a) * u_x
                reward = self.running_reward_np(t, x_grid[1:-1], a)
                W[:, i] = drift + reward
            
            # Stabilized Log-Sum-Exp trick for the integral
            W_max = np.max(W, axis=1) 
            
            # Trapz integral of exp((W - W_max)/lambda)
            exp_shifted = np.exp((W - W_max[:, None]) / self.lambda_reg)
            integral_exp = np.trapz(exp_shifted, a_grid, axis=1)
            
            # H_reg = lambda * ln( \int exp ) + W_max
            # clip integral_exp to avoid log(0)
            H_reg = self.lambda_reg * np.log(np.maximum(integral_exp, 1e-40)) + W_max
            
            U[n, 1:-1] = u_next[1:-1] + dt * (0.5 * u_xx + H_reg)
            
            # DIRICHLET BOUNDARY CONDITIONS
            U[n, 0] = 0.0
            U[n, -1] = 0.0

        U_x = np.gradient(U, x_grid, axis=1)
        interp_u_lam = RegularGridInterpolator((t_grid, x_grid), U, bounds_error=False, fill_value=None)
        interp_u_lam_x = RegularGridInterpolator((t_grid, x_grid), U_x, bounds_error=False, fill_value=None)
        return interp_u_lam, interp_u_lam_x

    # =========================================================================
    # INTERPOLATED EVALUATORS
    # =========================================================================
    def u_true_np(self, t, x):
        """ Evaluates the UNREGULARIZED u_0^* at (t, x). """
        t_arr, x_arr = np.atleast_1d(t), np.atleast_1d(x)
        pts = np.stack(np.broadcast_arrays(t_arr, x_arr), axis=-1)
        res = self._interp_u(pts)
        return float(res[0]) if res.size == 1 else res

    def u_lam_true_np(self, t, x):
        """ Evaluates the REGULARIZED u_lambda^* at (t, x). """
        t_arr, x_arr = np.atleast_1d(t), np.atleast_1d(x)
        pts = np.stack(np.broadcast_arrays(t_arr, x_arr), axis=-1)
        res = self._interp_u_lam(pts)
        return float(res[0]) if res.size == 1 else res

