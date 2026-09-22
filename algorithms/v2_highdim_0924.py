# ==============================================================================
#     J(v,w) MINIMIZATION IN R^d WITH L^p / SOFTMAX TIME REDUCERS (d PARAM)
# ==============================================================================
# - State dim 'd' is defined once on the ProblemSpec: spec.d
# - All components (models, SDE, Jacobian flow, kernels, Ψ line integral, FK)
#   derive their shapes from spec.d
# ==============================================================================

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
    Manufactured solution:
      Let s(t,x) = t + (1/d) * sum_i x_i.
      u(t,x) = cos( s(t,x) ).

    Choose
      Σ(t,x) = (1/√d) * sin(s) * I_d      (with sign-preserving floor),
      b(t,x,a) = - (1/(2 d^2)) sin(s) cos(s) * 1_d          (independent of a),
      r(t,x,a) = sin(s)                                        (independent of a),

    so that
      u_t = - sin(s),     (1/2) Tr[ΣΣᵀ ∇^2 u] = - (1/(2 d^2)) sin^2(s) cos(s),
      b·∇u + r = + (1/(2 d^2)) sin^2(s) cos(s) + sin(s)    (no a-dependence),
      ⇒   u_t + (1/2)Tr + log ∫ exp(b·∇u + r) da = -sin + ( -A ) + ( +A + sin ) = 0.

    Notes:
      • Keeping r, b independent of a makes the A-integral trivial (it is still a valid
        entropy-regularized problem; the control is non-active in closed form but the
        learning pipeline remains exercised).
      • This spec is algebraically clean and stable in high-dim, and avoids the scaling
        inconsistencies that appear if one mixes s = t + x vs t + (1/d) ∑ x_i.
    """
    def __init__(self, d, T=0.05, A_min=0.0, A_max=1.0, lambda_reg=5.0, x_floor=1e-3):

      
        super().__init__(d=d, T=T, A_min=A_min, A_max=A_max, lambda_reg=lambda_reg, x_floor=x_floor)
        self.x_floor = float(x_floor)

    def _s(self, t, x):
        # x: [B,d], returns [B,1]
        return t + x.mean(dim=1, keepdim=True)

    # truths
    def u_true_np(self, t, x):
        x = np.asarray(x, dtype=float)
        s = t + float(np.mean(x))
        return float(np.cos(s))

    def u_x_true_np(self, t, x):
        x = np.asarray(x, dtype=float)
        s = t + float(np.mean(x))
        grad = -(1.0/float(self.d)) * np.sin(s) * np.ones(self.d, dtype=float)
        return grad

    # terminal
    @torch.no_grad()
    def g_value_torch(self, x, T):
        s = T + x.mean(dim=1, keepdim=True)
        return torch.cos(s).squeeze(-1)  # [B]

    def g_x_torch(self, x, T):
        s = T + x.mean(dim=1, keepdim=True)
        return (-(1.0/float(self.d)) * torch.sin(s) * torch.ones_like(x))  # [B,d]

    # drift (no a-dependence here; OK for training loop)
    def b_ctrl_torch(self, t, x, a):
        s = self._s(t, x)
        return -(1.0/(2.0*float(self.d)**2)) * torch.sin(s) * torch.cos(s) * torch.ones_like(x)

    # diffusion matrix, inverse, and Jacobian
    def sigma_mat_torch(self, t, x):
        s = self._s(t, x)                      # [B,1]
        mag = torch.sin(s)
        # sign-preserving clamp on magnitude, then scale 1/√d
        mag = mag.sign() * torch.clamp(mag.abs(), min=self.x_floor)
        mag = mag / math.sqrt(self.d)
        return _diag_batch(mag.expand_as(x))   # [B,d,d]

    def sigma_inv_torch(self, t, x):
        s = self._s(t, x)
        mag = torch.sin(s)
        mag = mag.sign() * torch.clamp(mag.abs(), min=self.x_floor)
        inv = (1.0 / mag) * math.sqrt(self.d)  # inverse of (mag/√d)
        return _diag_batch(inv.expand_as(x))

    def sigma_jac_torch(self, t, x):
        """
        Σ = (1/√d) sin(s) I_d,  s = t + mean(x).
        ∂Σ_ii/∂x_k = (1/√d) cos(s) * ∂s/∂x_k * δ_{ik} = (1/(d√d)) cos(s) δ_{ik}.
        """
        B, d = x.shape
        s = self._s(t, x)                                     # [B,1]
        c = torch.cos(s) / (float(d) * math.sqrt(d))          # [B,1]
        J = torch.zeros(B, d, d, d, device=x.device, dtype=x.dtype)
        idx = torch.arange(d, device=x.device)
        J[:, idx, idx, idx] = c                               # broadcast [B,1] -> [B]
        return J

    def running_reward_torch(self, t, x, a):
        s = self._s(t, x)
        return torch.sin(s).squeeze(-1)  # [B]


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





# ==============================================================================
#                                   MODELS
# ==============================================================================
def build_v_model(width, d, input_dim=None, output_dim=None):
    if input_dim is None:  input_dim = 1 + d   # (t, x∈R^d)
    if output_dim is None: output_dim = d      # v ∈ R^d
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

def build_w_model(width, d, input_dim=None, output_dim=1):
    if input_dim is None: input_dim = 1 + d + 1   # (t, x∈R^d, a)
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
def lp_reduce_over_time(R, p=8.0, time_dim=1):
    per_sample = R.pow(p).mean(dim=time_dim).pow(1.0 / p)
    return per_sample.mean()

def soft_sup(r, tau, dim):
    m = torch.amax(r, dim=dim, keepdim=True)
    return tau * torch.logsumexp((r - m) / tau, dim=dim) + m.squeeze(dim)

def softmax_reduce_over_time(R, tau="auto", time_dim=1):
    B, K = R.size(0), R.size(time_dim)
    if tau == "auto":
        med = torch.nan_to_num(R.detach(), nan=0.0, posinf=0.0, neginf=0.0).median().item()
        denom = max(1.0, math.log(max(2, K)))
        tau_val = max(1e-4, min(1.0, 0.5 * med / denom))
    else:
        tau_val = float(tau)
    per_sample = soft_sup(R, tau=tau_val, dim=time_dim)
    return per_sample.mean()

def reduce_residuals(R, norm_type="lp", lp_p=8.0, softmax_tau="auto", time_dim=1):
    R = torch.nan_to_num(R, nan=0.0, posinf=1e6, neginf=0.0)
    if norm_type == "lp":
        return lp_reduce_over_time(R, p=lp_p, time_dim=time_dim)
    elif norm_type == "softmax":
        return softmax_reduce_over_time(R, tau=softmax_tau, time_dim=time_dim)
    else:
        raise ValueError("norm_type must be 'lp' or 'softmax'.")


# ==============================================================================
#                         GEOMETRY: JACOBIAN & KERNEL (d-D)
# ==============================================================================
def advance_jacobian(J_prev, A_cols, dW):  # J_prev [B,d,d], A_cols [B,d,d,d], dW [B,d]
    """
    Euler step for J:  J_{k+1} = J_k + sum_r A_r J_k ΔW^{(r)}.
    A_cols[..., r] = A_r ∈ R^{d×d}.
    """
    B, d = dW.size()
    J = J_prev
    for r in range(d):
        Ar = A_cols[:, :, :, r]                   # [B,d,d]
        incr = torch.bmm(Ar, J) * dW[:, r].view(B, 1, 1)
        J = J + incr
    return J

def eta_from_sigma_J(S, J):
    """
    η = σ^{-1} J using a batched linear solve: S @ η = J.
    """
    return torch.linalg.solve(S, J)  # [B,d,d]


# ==============================================================================
#                         LINE INTEGRAL OF v ALONG SEGMENT
# ==============================================================================
def integral_v_line_trap(v_model, t, x0, x1, steps=50):
    """
    ∫_0^1 v(t, x0 + sΔx) · Δx ds  via trapezoid; x0,x1 ∈ R^{B×d}.
    Returns [B].
    """
    B, d = x0.shape
    Δx = x1 - x0
    s = torch.linspace(0.0, 1.0, steps + 1, device=x0.device)
    Y = x0.unsqueeze(1) + Δx.unsqueeze(1) * s.view(1, -1, 1)   # [B,K+1,d]

    if isinstance(t, torch.Tensor):
        if t.dim() == 0:  t_grid = t.expand(B, steps + 1)
        elif t.dim() == 1: t_grid = t.unsqueeze(1).expand(B, steps + 1)
        else:              t_grid = t
    else:
        t_grid = torch.tensor(t, device=x0.device).expand(B, steps + 1)

    inp = torch.cat([t_grid.reshape(-1, 1), Y.reshape(-1, d)], dim=1)  # [(B(K+1)), 1+d]
    v_vals = v_model(inp).reshape(B, steps + 1, d)                      # [B,K+1,d]

    trap = (v_vals[:, 0, :] + v_vals[:, -1, :] + 2.0 * v_vals[:, 1:-1, :].sum(dim=1)) / steps  # [B,d]
    return (trap * Δx).sum(dim=1)  # [B]


# ==============================================================================
#                             TRAINING (VANILLA, d-D)
# ==============================================================================
def train_vanilla_with_norm_d(
    spec: ProblemSpecD,
    T=None, time_steps=21,
    training_path_size=4000, nn_batch_size=1000, num_epochs=40,
    neuron_number_v=128, neuron_number_w=128, learning_rate=5e-4, weight_decay=0.0,
    A_min=None, A_max=None, num_a_points=160, lambda_reg=None,
    int_x_steps=50,
    norm_type="lp", lp_p=8.0, softmax_tau="auto",
    x0_init=None,                     # length-d vector; if None -> 1.0 * 1_d
    early_stopping=True, es_patience=20, es_min_delta=5e-4,
    verbose: bool = True,
):
    """
    d-dimensional training of (v,w). State dim = spec.d.
    """
    # ---- defaults ----
    if T is None: T = spec.T
    if A_min is None: A_min = spec.A_min
    if A_max is None: A_max = spec.A_max
    if lambda_reg is None: lambda_reg = spec.lambda_reg

    d = spec.d
    if x0_init is None:
        x0_init = torch.full((d,), 1.0, dtype=torch.float32, device=device)
    else:
        x0_init = torch.as_tensor(x0_init, dtype=torch.float32, device=device)
        assert x0_init.numel() == d, f"x0_init must have length {d}"

    vprint = print if verbose else (lambda *a, **k: None)

    # ---- grids ----
    delta_t = T / (time_steps - 1)
    t_seq  = torch.linspace(0.0, T, time_steps, device=device)
    T_t    = torch.tensor(T, dtype=torch.float32, device=device)
    dt_t   = torch.tensor(delta_t, dtype=torch.float32, device=device)
    a_grid = torch.linspace(A_min, A_max, num_a_points, device=device)

    # ---- models & opt ----
    v_Model = build_v_model(neuron_number_v, d).to(device)     # outputs R^d
    w_Model = build_w_model(neuron_number_w, d).to(device)     # outputs R
    optim = torch.optim.Adam(list(v_Model.parameters())+list(w_Model.parameters()),
                             lr=learning_rate, weight_decay=weight_decay)

    best_loss = float('inf'); patience_ctr = 0
    stopped_early = False; epochs_run = 0

    # NEW: helper to clamp states only if required by the spec
    def maybe_clamp_state(X: torch.Tensor) -> torch.Tensor:
        return X.clamp_min(spec.x_floor) if getattr(spec, "clamp_state", False) else X


    # ---- training loop ----
    for epoch in range(num_epochs):
        epoch_loss = epoch_loss_v = epoch_loss_w = 0.0

        for start in range(0, training_path_size, nn_batch_size):
            B = min(nn_batch_size, training_path_size - start)

            # 1) Brownian increments in R^d
            dW = torch.sqrt(dt_t) * torch.randn(B, time_steps - 1, d, device=device)  # [B,T-1,d]

            # 2) Uncontrolled 𝓧 in R^d
            X_unc = torch.zeros(B, time_steps, d, device=device)
            X_unc[:, 0, :] = maybe_clamp_state(x0_init.unsqueeze(0).expand(B, -1))

            for i in range(1, time_steps):
                t_prev = t_seq[i - 1]
                x_prev = X_unc[:, i - 1, :]  # [B,d]
                S = spec.sigma_torch(t_prev, x_prev)  # [B,d,d]
                step = torch.bmm(S, dW[:, i - 1, :].unsqueeze(-1)).squeeze(-1)  # [B,d]
                X_unc[:, i, :] = maybe_clamp_state(x_prev + step)


            # 3) Controlled X^a for Ψ (single a per path)
            a_batch = A_min + (A_max - A_min) * torch.rand(B, device=device)  # [B]
            X_ctrl = torch.zeros_like(X_unc); X_ctrl[:, 0, :] = X_unc[:, 0, :]

            for i in range(1, time_steps):
                t_prev = t_seq[i - 1]
                x_prev = X_ctrl[:, i - 1, :]
                S = spec.sigma_torch(t_prev, x_prev)                          # [B,d,d]
                drift = spec.b_ctrl_torch(t_prev, x_prev, a_batch) * dt_t     # [B,d]
                noise = torch.bmm(S, dW[:, i - 1, :].unsqueeze(-1)).squeeze(-1)
                X_ctrl[:, i, :] = maybe_clamp_state(x_prev + drift + noise)


            # 4) Flow Jacobian J and martingale M for Φ
            J = torch.zeros(B, time_steps, d, d, device=device)
            J[:, 0, :, :] = torch.eye(d, device=device).unsqueeze(0).expand(B, -1, -1)

            for i in range(1, time_steps):
                t_prev = t_seq[i - 1]
                x_prev = X_unc[:, i - 1, :]
                Acols  = spec.sigma_jac_torch(t_prev, x_prev)   # [B,d,d,d]
                J[:, i, :, :] = advance_jacobian(J[:, i-1, :, :], Acols, dW[:, i-1, :])

            # η = σ^{-1} J; M_k = Σ_{m<=k-1} η_m ΔW_m  (each M_k ∈ R^d)
            M = torch.zeros(B, time_steps - 1, d, device=device)
            for i in range(time_steps - 1):
                t_prev = t_seq[i]
                x_prev = X_unc[:, i, :]
                S = spec.sigma_torch(t_prev, x_prev)          # [B,d,d]
                Jk = J[:, i, :, :]                            # [B,d,d]
                eta = eta_from_sigma_J(S, Jk)                 # [B,d,d]
                dW_i = dW[:, i, :]                            # [B,d]
                inc = torch.bmm(eta, dW_i.unsqueeze(-1)).squeeze(-1)  # [B,d]
                M[:, i, :] = inc if i == 0 else M[:, i-1, :] + inc

            # 5) Φ: H = λ log ∫_A exp(w/λ) da  along 𝓧
            t_vals = t_seq[1:].unsqueeze(0).repeat(B, 1)     # [B,T-1]
            x_vals = X_unc[:, 1:, :]                         # [B,T-1,d]

            t_rep = t_vals.unsqueeze(2).repeat(1, 1, num_a_points)             # [B,T-1,A]
            x_rep = x_vals.unsqueeze(2).repeat(1, 1, num_a_points, 1)          # [B,T-1,A,d]
            a_rep = torch.linspace(A_min, A_max, num_a_points, device=device).view(1,1,-1).repeat(B, t_vals.shape[1], 1)

            inp = torch.cat([
                t_rep.reshape(-1, 1),
                x_rep.reshape(-1, d),
                a_rep.reshape(-1, 1)
            ], dim=1)  # [B(T-1)A, 1+d+1]
            w_out = w_Model(inp).view(B, t_vals.shape[1], num_a_points)  # [B,T-1,A]

            Sscaled = w_out / lambda_reg
            Smax, _ = torch.max(Sscaled, dim=2, keepdim=True)
            a_axis = torch.linspace(A_min, A_max, num_a_points, device=device)
            int_exp = torch.trapz(torch.exp(Sscaled - Smax), a_axis, dim=2).clamp_min(1e-40)
            ln_integral = lambda_reg * (torch.log(int_exp) + Smax.squeeze(2))  # [B,T-1]

            # Terminal pullback: J_k^{-T} (J_T^T g_x(X_T))
            X_T = X_unc[:, -1, :]                       # [B,d]
            gT  = spec.g_x_torch(X_T, T_t)              # [B,d]
            JT  = J[:, -1, :, :]                        # [B,d,d]

            # 6) Φ-residuals: || v(t_k,x_k) - φ_k ||_2
            num_points = time_steps - 1
            Rv_list = []
            for k in range(num_points):
                t_k = t_seq[k]
                x_k = X_unc[:, k, :]                   # [B,d]
                Jk  = J[:, k, :, :]                    # [B,d,d]

                # K_{k→j} = J_k^{-T} * (M_j - M_{k-1}) / (t_j - t_k)
                M_base = torch.zeros(B, d, device=device) if k == 0 else M[:, k-1, :]
                future = M[:, k:, :]                   # [B,num_future,d]
                dt_future = (t_seq[k+1:] - t_k)        # [num_future]
                raw = (future - M_base.unsqueeze(1)) / dt_future.view(1, -1, 1)  # [B,num_future,d]

                JkT = Jk.transpose(1, 2)
                rhs = raw.transpose(1, 2)              # [B,d,num_future]
                pulled = torch.linalg.solve(JkT, rhs).transpose(1, 2)  # [B,num_future,d]

                H_slice = ln_integral[:, k:]           # [B,num_future]
                integ_vec = (H_slice.unsqueeze(2) * pulled).sum(dim=1) * dt_t  # [B,d]

                term_vec = torch.linalg.solve(JkT, torch.bmm(JT.transpose(1,2), gT.unsqueeze(-1))).squeeze(-1)  # [B,d]
                phi_k = integ_vec + term_vec           # [B,d]

                v_out = v_Model(torch.cat([t_k.expand(B,1), x_k], dim=1))  # [B,d]
                res_k = torch.norm(v_out - phi_k, dim=1)                   # [B]
                Rv_list.append(res_k)

            Rv = torch.stack(Rv_list, dim=1) if Rv_list else torch.zeros(B, 1, device=device)  # [B,K]
            loss_v_batch = reduce_residuals(Rv, norm_type=norm_type, lp_p=lp_p, softmax_tau=softmax_tau, time_dim=1)

            # 7) Ψ: local one-step residual via line integrals of v
            Rw_list = []
            for k in range(num_points):
                t_k  = t_seq[k]
                cur_x = X_unc[:, k, :]                                 # [B,d]
                S = spec.sigma_torch(t_k, cur_x)                        # [B,d,d]
                dW_k = dW[:, k, :]                                      # [B,d]

                next_c = maybe_clamp_state(cur_x + torch.bmm(S, dW_k.unsqueeze(-1)).squeeze(-1))           # [B,d]
                drift_k = spec.b_ctrl_torch(t_k, cur_x, a_batch) * dt_t                                   # [B,d]
                next_X = maybe_clamp_state(cur_x + drift_k + torch.bmm(S, dW_k.unsqueeze(-1)).squeeze(-1)) 

                int_c = integral_v_line_trap(v_Model, t_k, cur_x, next_c, steps=int_x_steps)  # [B]
                int_X = integral_v_line_trap(v_Model, t_k, cur_x, next_X, steps=int_x_steps)  # [B]

                reward_k = spec.running_reward_torch(t_k, next_X, a_batch) * dt_t             # [B]
                psi_k = (int_X - int_c + reward_k) / dt_t                                     # [B]

                w_in = torch.cat([t_k.expand(B,1), cur_x, a_batch.view(B,1)], dim=1)          # [B,1+d+1]
                w_out = w_Model(w_in).squeeze(-1)                                             # [B]

                Rw_list.append((w_out - psi_k).abs())

            Rw = torch.stack(Rw_list, dim=1) if Rw_list else torch.zeros(B, 1, device=device)  # [B,K]
            loss_w_batch = reduce_residuals(Rw, norm_type=norm_type, lp_p=lp_p, softmax_tau=softmax_tau, time_dim=1)

            # 8) Update
            J_batch = loss_v_batch + loss_w_batch
            optim.zero_grad(); J_batch.backward(); optim.step()

            epoch_loss   += J_batch.item()
            epoch_loss_v += loss_v_batch.item()
            epoch_loss_w += loss_w_batch.item()

        # ---- epoch summary ----
        denom = max(1, math.ceil(training_path_size / nn_batch_size))
        avgJ  = epoch_loss   / denom
        avgLv = epoch_loss_v / denom
        avgLw = epoch_loss_w / denom
        epochs_run = epoch + 1
        vprint(f"Epoch {epochs_run}/{num_epochs} | J: {avgJ:.6f} | Lv: {avgLv:.6f} | Lw: {avgLw:.6f} | norm={norm_type}")

        # early stopping
        if avgJ < best_loss - es_min_delta:
            best_loss = avgJ; patience_ctr = 0
            torch.save(v_Model.state_dict(), f'best_v_d_{norm_type}.pth')
            torch.save(w_Model.state_dict(), f'best_w_d_{norm_type}.pth')
        else:
            patience_ctr += 1
            if early_stopping and patience_ctr >= es_patience:
                vprint(f"[Early stop] No improvement ≥ {es_min_delta} for {es_patience} epochs.")
                stopped_early = True; break

    meta = dict(T=T, time_steps=time_steps, norm_type=norm_type, lp_p=lp_p, softmax_tau=softmax_tau,
                lambda_reg=lambda_reg, num_a_points=num_a_points, int_x_steps=int_x_steps,
                A_min=A_min, A_max=A_max, stopped_early=stopped_early, epochs_run=epochs_run,
                state_dim=d)
    
    return v_Model, w_Model, meta


# ==============================================================================
#                    FEYNMAN–KAC RECOVERY OF u(t0,x0) (d-D)
# ==============================================================================
@torch.no_grad()
def recover_u_at_point_d(spec: ProblemSpecD, w_model,
                         t0: float, x0,
                         T=None, time_steps=21,
                         A_min=None, A_max=None, num_a_points=160, lambda_reg=None,
                         n_paths: int = 10000):
    """
    u(t0,x0) ≈ E[ g(𝓧_T) + ∑_{i: t_i>t0} H(t_i, 𝓧_{t_i}) Δt ], with H(t,x) = λ log ∫_A exp(w/λ) da.
    """
    def maybe_clamp_state(X: torch.Tensor) -> torch.Tensor:
        return X.clamp_min(spec.x_floor) if getattr(spec, "clamp_state", False) else X
        
    if T is None: T = spec.T
    if A_min is None: A_min = spec.A_min
    if A_max is None: A_max = spec.A_max
    if lambda_reg is None: lambda_reg = spec.lambda_reg

    d = spec.d
    x0 = torch.as_tensor(x0, dtype=torch.float32, device=device)
    assert x0.numel() == d, f"x0 must have length {d}"

    delta_t = T / (time_steps - 1)
    t_seq = torch.linspace(0.0, T, time_steps, device=device)
    T_t   = torch.tensor(T, dtype=torch.float32, device=device)
    dt_t  = torch.tensor(delta_t, dtype=torch.float32, device=device)

    a_grid = torch.linspace(A_min, A_max, num_a_points, device=device)

    k0 = int(round(t0 / float(delta_t)))
    k0 = max(0, min(k0, time_steps - 1))
    steps_left = time_steps - 1 - k0
    if steps_left <= 0:
        X_T = x0.view(1, d)
        u_hat = spec.g_value_torch(X_T, T_t).mean().item()
        return u_hat, 0.0

    # simulate uncontrolled
    dW = torch.sqrt(dt_t) * torch.randn(n_paths, steps_left, d, device=device)
    X  = torch.zeros(n_paths, steps_left + 1, d, device=device)
    X[:, 0, :] = x0.view(1, d)

    for i in range(1, steps_left + 1):
        t_prev = t_seq[k0 + i - 1]
        x_prev = X[:, i - 1, :]
        S = spec.sigma_torch(t_prev, x_prev)                         # [N,d,d]
        step = torch.bmm(S, dW[:, i - 1, :].unsqueeze(-1)).squeeze(-1)
        X[:, i, :] = maybe_clamp_state(x_prev + step) 

    # H via w
    x_vals = X[:, 1:, :]                                   # [N,steps_left,d]
    t_vals = t_seq[k0 + 1:].unsqueeze(0).repeat(n_paths, 1)  # [N,steps_left]

    t_rep = t_vals.unsqueeze(2).repeat(1, 1, num_a_points)        # [N,L,A]
    x_rep = x_vals.unsqueeze(2).repeat(1, 1, num_a_points, 1)     # [N,L,A,d]
    a_rep = a_grid.view(1, 1, -1).repeat(n_paths, t_vals.shape[1], 1)

    inp = torch.cat([
        t_rep.reshape(-1, 1),
        x_rep.reshape(-1, d),
        a_rep.reshape(-1, 1)
    ], dim=1)  # [NLA, 1+d+1]
    w_out = w_model(inp).view(n_paths, t_vals.shape[1], num_a_points)

    Sscaled = w_out / lambda_reg
    Smax, _ = torch.max(Sscaled, dim=2, keepdim=True)
    int_exp = torch.trapz(torch.exp(Sscaled - Smax), a_grid, dim=2).clamp_min(1e-40)
    log_mean_exp = torch.log(int_exp) + Smax.squeeze(2)
    H_vals = lambda_reg * log_mean_exp                        # [N,L]
    integral_term = (H_vals * dt_t).sum(dim=1)                # [N]

    g_term = spec.g_value_torch(X[:, -1, :], T_t)             # [N]
    u_samples = g_term + integral_term
    u_hat = u_samples.mean().item()
    u_se  = u_samples.std(unbiased=True).item() / math.sqrt(n_paths)
    return u_hat, u_se


# ==============================================================================
#                                EXAMPLE RUN
# ==============================================================================
if __name__ == "__main__":
    
    torch.manual_seed(0); np.random_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Choose dimension once:
    d = 2
    spec = EX4HighDim(d=d, T=0.10, A_min=0.0, A_max=1.0, lambda_reg=5.0, x_floor=1e-3)

    norm_type = "softmax"   # or "lp"
    lp_p      = 8.0
    soft_tau  = "auto"

    v_Model, w_Model, meta = train_vanilla_with_norm_d(
        spec,
        T=spec.T, time_steps=11,
        training_path_size=12000, nn_batch_size=6000, num_epochs=40,
        neuron_number_v=96, neuron_number_w=96, learning_rate=6e-4, weight_decay=0.0,
        A_min=spec.A_min, A_max=spec.A_max, num_a_points=96, lambda_reg=spec.lambda_reg,
        int_x_steps=40,
        norm_type=norm_type, lp_p=lp_p, softmax_tau=soft_tau,
        x0_init=np.ones(d, dtype=np.float32),
        early_stopping=True, es_patience=12, es_min_delta=1e-3,
        verbose=True
    )


    # ---- Probe ∇u and u at (t0, x0) with ground truth if available ----
    t0 = 0.0
    x0 = np.ones(d, dtype=float)

    with torch.no_grad():
        v_hat = v_Model(torch.tensor([[t0, *x0]], dtype=torch.float32, device=device)).squeeze(0).cpu().numpy()
    print(f"∇u_model(t0={t0:.3f}, x0={x0.tolist()}) = {v_hat}")

    v_true = spec.u_x_true_np(t0, x0)
    if not (isinstance(v_true, float) and np.isnan(v_true)) and not np.any(np.isnan(v_true)):
        v_true = np.asarray(v_true, dtype=float)
        g_err  = np.linalg.norm(v_hat - v_true)
        g_rel  = g_err / max(1e-12, np.linalg.norm(v_true))
        print(f"∇u_true (ground truth)           = {v_true}")
        print(f"∇u error: ||pred-true||2 = {g_err:.6e}  (rel {g_rel:.6%})")
    else:
        print("(No closed-form ∇u_true provided.)")

    # Recover u and compare
    u_hat, u_se = recover_u_at_point_d(
        spec, w_Model, t0=t0, x0=x0,
        T=meta['T'], time_steps=meta['time_steps'],
        num_a_points=meta['num_a_points'], lambda_reg=meta['lambda_reg'],
        A_min=meta['A_min'], A_max=meta['A_max'],
        n_paths=8000
    )
    print(f"u_model(t0={t0:.3f}, x0={x0.tolist()}) = {u_hat:.6f}  (MC SE≈{u_se:.6f})")

    u_true = spec.u_true_np(t0, x0)
    if not (isinstance(u_true, float) and np.isnan(u_true)):
        u_err = abs(u_hat - u_true)
        u_rel = u_err / max(1e-12, abs(u_true))
        print(f"u_true (ground truth)             = {u_true:.6f}")
        print(f"u error: |pred-true|   = {u_err:.6e}  (rel {u_rel:.6%})")
    else:
        print("(No closed-form u_true provided.)")



