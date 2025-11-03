import math
import numpy as np
import torch
import torch.nn as nn

# # ------------------------------ Repro & device ------------------------------
# torch.manual_seed(0)
# np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---- module-level helpers used by specs ----
def _diag_batch(vec):                # vec: [B,d] -> diag matrices [B,d,d]
    return torch.diag_embed(vec)


# ------------------------------ Base spec (d-dimensional) ------------------------------
class ProblemSpecD:
    """
    d-dimensional entropic-control problem spec.

    Conventions used by your multi-d code:
      - State x is R^d (PyTorch tensors shape [..., d])
      - Control a is scalar in [A_min, A_max]
      - Brownian W has dimension d
      - sigma(t,x) returns a [*, d, d] matrix (typically diagonal)
      - sigma_x_apply(t,x,J,dW) returns the Stratonovich/Itô Jacobian increment
        ∑_r ∂_x σ(:, r) @ J * dW_r  (shape [*, d, d])
      - g_x_torch returns gradient in R^d
      - b_ctrl_torch returns drift vector in R^d
      - running_reward_torch returns scalar

    We provide diagonal Σ so sigma_x_apply is simple and stable.
    """
    def __init__(self, d:int, T:float, A_min:float, A_max:float, lambda_reg:float=5.0, x_floor:float=1e-3):
        self.d          = int(d)
        self.T          = float(T)
        self.A_min      = float(A_min)
        self.A_max      = float(A_max)
        self.lambda_reg = float(lambda_reg)
        self.x_floor    = float(x_floor)
        self.clamp_state = False

    # ------- (optional) truths, can return NaN if not provided -------
    def u_true_np(self, t: float, x: np.ndarray) -> float:
        return float('nan')

    def u_x_true_np(self, t: float, x: np.ndarray) -> np.ndarray:
        return np.full(self.d, np.nan, dtype=float)

    # ------- required APIs for training/inference -------
    def g_value_torch(self, x: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def g_x_torch(self, x: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError  # shape [..., d]

    def b_ctrl_torch(self, t: torch.Tensor, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError  # shape [..., d]

    # --- d-D diffusion API (matrix form + jacobian of entries) ---
    def sigma_mat_torch(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError  # shape [..., d, d]

    def sigma_inv_torch(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError  # shape [..., d, d]

    def sigma_jac_torch(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError  # shape [..., d, d, d] with entries ∂Σ_ij/∂x_k

    # --- backward-compatibility shim for existing call sites ---
    def sigma_torch(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Alias: in d>1 we treat 'sigma_torch' as the diffusion matrix Σ."""
        return self.sigma_mat_torch(t, x)

    # kept for docstring completeness; the trainer uses sigma_jac_torch instead
    def sigma_x_apply(self, t: torch.Tensor, x: torch.Tensor, J: torch.Tensor, dW: torch.Tensor) -> torch.Tensor:
        """
        Given current (t, x), the state Jacobian J (shape [..., d, d]), and a Brownian increment dW (shape [..., d]),
        return the Jacobian increment   sum_r (∂_x σ(:, r)) @ J * dW_r   with shape [..., d, d].
        For diagonal σ, this is cheap and stable; we implement that in the subclasses.
        """
        raise NotImplementedError

    def running_reward_torch(self, t: torch.Tensor, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError  # shape [...]
    # ---- Helpers for safe floors (sign-preserving where needed) ----
    def _floor_abs(x, eps):              # scalar or tensor
        return torch.clamp(x.abs(), min=eps) * x.sign()
    
    def _diag_batch(vec):                # vec: [B,d] -> diag matrices [B,d,d]
        return torch.diag_embed(vec)

# ======================= EX4HighDim (√x diagonal diffusion) =======================
class EX4HighDim(ProblemSpecD):
    """
    Manufactured solution:
      u(t,x) = exp( - ( t^2 + ||x||^2/d + 1 ) )

    Gradient / Hessian (componentwise):
      ∇u = -(2/d) x u,   ∂^2_{x_i x_i} u = (-2/d + 4 x_i^2/d^2) u

    Drift (per-coordinate), diffusion (diagonal), reward:
      b_i(t,x,a) = x_i^2/d + a - 1/2
      σ(t,x) = diag( sqrt( clamp(x_i, x_floor) ) )
      r(t,x,a) = ( 2 t + (2 a / d) * sum_i x_i ) * u(t,x)

    With these choices, the entropic HJB
      u_t + (1/2) Tr[ ΣΣᵀ ∇^2 u ] + λ log ∫_A exp( (b·∇u + r)/λ ) da = 0
    holds exactly for any d (λ cancels as a constant factor).
    """
    def __init__(self, d, T=0.05, A_min=0.0, A_max=1.0, lambda_reg=5.0, x_floor=1e-3):
        super().__init__(d=d, T=T, A_min=A_min, A_max=A_max, lambda_reg=lambda_reg, x_floor=x_floor)
        self.clamp_state = True
    # ----- Truths (optional) -----
    def u_true_np(self, t, x):
        x = np.asarray(x, dtype=float)
        return float(np.exp(-(t**2 + (np.dot(x, x)/self.d) + 1.0)))

    def u_x_true_np(self, t, x):
        x = np.asarray(x, dtype=float)
        u = np.exp(-(t**2 + (np.dot(x, x)/self.d) + 1.0))
        return (-2.0/self.d) * x * u

    # ----- Terminal -----
    @torch.no_grad()
    def g_value_torch(self, x, T):
        # x: [B,d]
        q = (x * x).sum(dim=1) / float(self.d)     # [B]
        return torch.exp(-(q + T**2 + 1.0))        # [B]

    def g_x_torch(self, x, T):
        # ∇_x g = (-2/d) x * g
        gv = self.g_value_torch(x, T).unsqueeze(-1)       # [B,1]
        return (-2.0/float(self.d)) * x * gv              # [B,d]

    # ----- Drift for Ψ (componentwise) -----
    def b_ctrl_torch(self, t, x, a):
        # x: [B,d], a: [B] or [B,1] -> broadcast to [B,1]
        a = a.view(-1, 1)
        return (x**2)/float(self.d) + a - 0.5  # [B,d]

    # ----- Diffusion matrix & Jacobian -----
    def sigma_mat_torch(self, t, x):
        # Σ = diag( sqrt(x_i_clamped) )
        s = torch.sqrt(torch.clamp(x, min=self.x_floor))     # [B,d]
        return _diag_batch(s)                                 # [B,d,d]

    def sigma_inv_torch(self, t, x):
        # Σ^{-1} = diag( 1/sqrt(x_i_clamped) )
        s = torch.sqrt(torch.clamp(x, min=self.x_floor))
        inv = 1.0 / torch.clamp(s, min=1e-6)
        return _diag_batch(inv)

    def sigma_jac_torch(self, t, x):
        """
        Tensor J with J[:, i, j, k] = ∂Σ_ij/∂x_k.
        For diagonal Σ_ii = sqrt(x_i), J nonzero only when j=i=k:
            ∂Σ_ii/∂x_i = 1/(2 sqrt(x_i)).
        """
        B, d = x.shape
        s = torch.sqrt(torch.clamp(x, min=self.x_floor))      # [B,d]
        dSigma_dxi = 0.5 / torch.clamp(s, min=1e-6)           # [B,d]
        J = torch.zeros(B, d, d, d, device=x.device, dtype=x.dtype)
        idx = torch.arange(d, device=x.device)
        J[:, idx, idx, idx] = dSigma_dxi  # only i=j=k entries
        return J  # [B,d,d,d]

    # ----- Running reward (manufactured) -----
    def running_reward_torch(self, t, x, a):
        # r = (2t + 2 a * (x⋅1)/d) * u
        u = torch.exp(-(t**2 + (x*x).sum(dim=1)/float(self.d) + 1.0))  # [B]
        ax = (a.view(-1) * x.sum(dim=1)) / float(self.d)               # [B]
        return (2.0*t + 2.0*ax) * u                                     # [B]


# =================== EX5HighDim (cosine with sin-based isotropic Σ) ===================
class EX5HighDim(ProblemSpecD):
    """
    High-d EX5 with 1 active axis:
      s = t + x_1,   u(t,x) = cos(s).
      ∇u = (-sin(s), 0, ..., 0),   ∂_{11}u = -cos(s), others 0.
      Σ = diag( 2+sin(s), 1, ..., 1 )  (invertible; no floor needed)
      b_1 = a - [1 + 2 cos(s) + 0.5 sin(s) cos(s)],   b_i = 0 (i≥2)
      r(t,x,a) = 2 cos(s) + a sin(s).

    With λ=1, the HJB holds exactly (integrand independent of a).
    """
    def __init__(self, d, T=0.1, A_min=0.0, A_max=1.0, lambda_reg=1.0):
        super().__init__(d=d, T=T, A_min=A_min, A_max=A_max, lambda_reg=lambda_reg, x_floor=0.0)

    # truths
    def u_true_np(self, t, x):
        x = np.asarray(x, dtype=float)
        return float(np.cos(t + x[0]))

    def u_x_true_np(self, t, x):
        x = np.asarray(x, dtype=float)
        s = t + x[0]
        g = np.zeros(self.d, dtype=float)
        g[0] = -np.sin(s)
        return g

    # terminal
    @torch.no_grad()
    def g_value_torch(self, x, T):
        # x: [B,d]
        s = T + x[:, 0:1]              # [B,1]
        return torch.cos(s).squeeze(-1)

    def g_x_torch(self, x, T):
        s = T + x[:, 0:1]              # [B,1]
        B, d = x.shape
        gx = torch.zeros(B, d, device=x.device, dtype=x.dtype)
        gx[:, 0] = -torch.sin(s).squeeze(-1)
        return gx

    # drift b ∈ R^d
    def b_ctrl_torch(self, t, x, a):
        # t: [B] or [B,1], x: [B,d], a: [B] or [B,1]
        s = t.view(-1, 1) + x[:, 0:1]  # [B,1]
        b1 = a.view(-1, 1) - (1.0 + 2.0*torch.cos(s) + 0.5*torch.sin(s)*torch.cos(s))
        B, d = x.shape
        b = torch.zeros(B, d, device=x.device, dtype=x.dtype)
        b[:, 0] = b1.squeeze(-1)
        return b

    # diffusion Σ, Σ^{-1}, and Jacobian ∂Σ/∂x
    def sigma_mat_torch(self, t, x):
        s = t.view(-1, 1) + x[:, 0:1]          # [B,1]
        diag = torch.ones_like(x)
        diag[:, 0] = 2.0 + torch.sin(s).squeeze(-1)  # ≥1
        return torch.diag_embed(diag)

    def sigma_inv_torch(self, t, x):
        s = t.view(-1, 1) + x[:, 0:1]
        inv_diag = torch.ones_like(x)
        inv_diag[:, 0] = 1.0 / (2.0 + torch.sin(s).squeeze(-1))
        return torch.diag_embed(inv_diag)

    def sigma_jac_torch(self, t, x):
        """
        Only Σ_11 depends on x_1 via s = t + x_1: ∂Σ_11/∂x_1 = cos(s).
        All other partials are zero.
        """
        B, d = x.shape
        J = torch.zeros(B, d, d, d, device=x.device, dtype=x.dtype)
        s = t.view(-1, 1) + x[:, 0:1]
        J[:, 0, 0, 0] = torch.cos(s).squeeze(-1)  # ∂Σ_11/∂x_1
        return J

    # running reward r (scalar)
    def running_reward_torch(self, t, x, a):
        s = t.view(-1, 1) + x[:, 0:1]
        return (2.0*torch.cos(s) + a.view(-1, 1)*torch.sin(s)).squeeze(-1)


    # ======================= EX7HighDim (linear-in-x diagonal diffusion) =======================


class EX7HighDim(ProblemSpecD):
    """
    d-D uniformly non-degenerate example (Σ ≡ I_d):
      b(t,x,a) = x/d + a * 1_d
      Σ(t,x)   = I_d
      r(t,x,a) = ( 2t + (2 a 1ᵀx + 1)/d ) * u(t,x)
      g(x)     = exp(-(T^2 + ||x||^2/d + 1))

    Manufactured solution (exact for λ=1, A=[0,1]):
      u(t,x)   = exp(-(t^2 + ||x||^2/d + 1))
      ∇u       = -(2/d) x u
      ∂^2_{x_i x_i} u = (-2/d + 4 x_i^2/d^2) u
      u_t      = -2t u
    """
    def __init__(self, d, T=0.1, A_min=0.0, A_max=1.0, lambda_reg=1.0, x_floor=0.0):
        super().__init__(d=d, T=T, A_min=A_min, A_max=A_max,
                         lambda_reg=lambda_reg, x_floor=x_floor)

    # ----- truths -----
    def u_true_np(self, t, x):
        x = np.asarray(x, dtype=float)
        q = (x @ x) / float(self.d)
        return float(np.exp(-(t**2 + q + 1.0)))

    def u_x_true_np(self, t, x):
        x = np.asarray(x, dtype=float)
        u = np.exp(-(t**2 + (x @ x)/float(self.d) + 1.0))
        return (-2.0/float(self.d)) * x * u

    # ----- terminal -----
    @torch.no_grad()
    def g_value_torch(self, x: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        q = (x * x).sum(dim=1) / float(self.d)
        return torch.exp(-(T**2 + q + 1.0))

    def g_x_torch(self, x: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        g = self.g_value_torch(x, T).unsqueeze(-1)
        return (-2.0/float(self.d)) * x * g

    # ----- drift: b(t,x,a) ∈ R^d -----
    def b_ctrl_torch(self, t: torch.Tensor, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return x / float(self.d) + a.view(-1, 1)

    # ----- diffusion matrix & helpers: Σ ≡ I_d -----
    def sigma_mat_torch(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        I = torch.eye(self.d, dtype=x.dtype, device=x.device)
        return I.unsqueeze(0).expand(B, -1, -1)  # [B,d,d]

    def sigma_inv_torch(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # inverse of I_d is I_d
        return self.sigma_mat_torch(t, x)

    def sigma_jac_torch(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # derivative of a constant matrix is zero
        B = x.shape[0]
        return torch.zeros(B, self.d, self.d, self.d, dtype=x.dtype, device=x.device)

    # ----- running reward r(t,x,a) -----
    def running_reward_torch(self, t: torch.Tensor, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        q = (x * x).sum(dim=1) / float(self.d)                # [B]
        u = torch.exp(-(t**2 + q + 1.0))                      # [B]
        ax = (2.0 * a.view(-1) * x.sum(dim=1) + 1.0) / float(self.d)  # [B]
        return (2.0 * t + ax) * u

