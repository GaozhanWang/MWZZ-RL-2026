# ==============================================================================
#             MODEL-BASED POLICY ITERATION (PIA) BASELINE: GENERIC, CLEAN
# ==============================================================================
# - Fits u(t,x) with MC targets under a Gibbs policy that uses u_x from a frozen
#   snapshot of the previous iteration:  pi(a|t,x) ∝ exp{ (b(t,x,a)*u_x + r(t,x,a)) / λ }.
# - Does NOT directly train on u_x. Hence u_x can be inaccurate even when u is fit,
#   which is the whole point of this baseline.
#
# Requirements:
#   - ProblemSpec (same interface as your main algorithm): g, g_x, b_ctrl, sigma, sigma_x, r
#   - EX4/EX5 (or your own problems) can be passed as `spec`.
#
# Exposed API:
#   - build_u_model(...)
#   - get_u_x(u_model, t, x, create_graph=False)
#   - train_model_based_PIA(spec, ..., device=None)
#   - probe_u_and_w(spec, u_model, t0, x0, a0, ...)
#
# Notes:
#   - Uses log-sum-exp for the Gibbs normalization over an action grid.
#   - Freezes policy_net each policy-iteration (deepcopy).
#   - Early stopping per *inner* u-fit, and a policy-diff stopping across PIA loops.
# ==============================================================================

import math
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
    

class EX7(ProblemSpec):
    """
      u(t,x)  = exp( -(t^2 + x^2 - 2) ),  g(x)=u(T,x)
      g_x     = -2x u
      b(t,x,a)= x^3 + a - 0.5*x
      σ(t,x)  = x   (implemented with a sign-preserving clamp for stability)
      r(t,x,a)= (2t + 2ax) u
    """
    def __init__(self, T=0.1, A_min=0.0, A_max=1.0, lambda_reg=5.0, x_floor=1e-2):
        self.T = float(T)
        self.A_min, self.A_max = float(A_min), float(A_max)
        self.lambda_reg = float(lambda_reg)
        # Numerical floor for |σ| to avoid degeneration near x≈0 in kernel/FK.
        self.x_floor = float(x_floor)
        # Linear diffusion: state may be negative; do NOT clamp the state
        self.clamp_state = False

    # ----- ground truth (for diagnostics) -----
    def u_true_np(self, t, x):
        return float(np.exp(-(t**2 + x**2 - 2.0)))

    def u_x_true_np(self, t, x):
        u = np.exp(-(t**2 + x**2 - 2.0))
        return float(-2.0 * x * u)

    # ----- terminal value g and its x-derivative -----
    @torch.no_grad()
    def g_value_torch(self, x: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        return torch.exp(-(T**2 + x**2 - 2.0))

    def g_x_torch(self, x: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        return -2.0 * x * torch.exp(-(T**2 + x**2 - 2.0))

    # ----- controlled drift b(t,x,a) used in Ψ one-step move -----
    def b_ctrl_torch(self, t, x, a):
        # Works with scalar or batched tensors; broadcasts naturally.
        return x**3 + a - 0.5 * x

    # ----- diffusion and its x-derivative for the Ma–Zhang kernel -----
    def sigma_torch(self, t, x):
        """
        σ(t,x) = x, stabilized via sign-preserving clamp:
            σ = sign(x) * max(|x|, x_floor)
        This keeps σ^{-1} and the ∇X update well-behaved when |x| is small.
        """
        return x.sign() * torch.clamp(x.abs(), min=self.x_floor)

    def sigma_x_torch(self, t, x):
        """
        d/dx (sign-preserving clamp of x):
          ≈ 1 for |x| > x_floor, and 0 inside the flat region (|x| ≤ x_floor).
        This ignores measure-zero kinks at the transition.
        """
        return torch.where(x.abs() > self.x_floor, torch.ones_like(x), torch.zeros_like(x))

    # ----- running reward r(t,x,a) -----
    def running_reward_torch(self, t, x, a):
        u = torch.exp(-(t**2 + x**2 - 2.0))
        return (2.0 * t + 2.0 * a * x) * u




# ------------------------------ Models ----------------------------------------
def build_u_model(neuron_number: int, input_dim: int = 2, output_dim: int = 1) -> nn.Module:
    """Simple MLP for u(t,x)."""
    width = int(neuron_number)
    model = nn.Sequential(
        nn.Linear(input_dim, width), nn.Tanh(),
        nn.Linear(width, width),     nn.Tanh(),
        nn.Linear(width, output_dim),
    )
    def init_(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight); nn.init.constant_(m.bias, 1e-2)
    model.apply(init_)
    return model


def get_u_x(model: nn.Module, t: torch.Tensor, x: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
    """
    Compute ∂_x u(t,x) via autograd.
    t, x: [B] tensors on the right device.
    Returns: [B]
    """
    inputs = torch.stack([t, x], dim=1).requires_grad_(True)  # [B,2]
    u = model(inputs).squeeze(-1)                              # [B]
    grad_inputs = torch.autograd.grad(
        outputs=u.sum(), inputs=inputs,
        create_graph=create_graph, retain_graph=False, allow_unused=False
    )[0]                                                      # [B,2]
    return grad_inputs[:, 1]                                  # ∂/∂x


# ------------------------------ Gibbs helpers ---------------------------------
def gibbs_expectations(spec, t: torch.Tensor, x: torch.Tensor, z: torch.Tensor,
                       a_grid: torch.Tensor, lambda_reg: float):
    """
    Compute, on a batch (t,x,z), the Gibbs density over `a_grid` and the expectations:
      E_b = E_pi[b(t,x,a)], E_r = E_pi[r(t,x,a)], H = -∫ pi log pi da
    Using a uniform grid and log-sum-exp stabilization.
    Inputs:
      t, x, z: [B]
      a_grid: [A]
    Returns:
      E_b, E_r, H   (all [B])
    """
    B, A = x.shape[0], a_grid.shape[0]
    # Broadcast to [B,A]
    t_rep = t.view(-1, 1).expand(B, A)
    x_rep = x.view(-1, 1).expand(B, A)
    a_rep = a_grid.view(1, -1).expand(B, A)

    b_val = spec.b_ctrl_torch(t_rep, x_rep, a_rep)            # [B,A]
    r_val = spec.running_reward_torch(t_rep, x_rep, a_rep)    # [B,A]

    S = (b_val * z.view(-1, 1) + r_val) / float(lambda_reg)   # [B,A]
    S_max, _ = S.max(dim=1, keepdim=True)                     # [B,1]
    W = torch.exp(S - S_max)                                  # stabilized weights
    Z = W.sum(dim=1, keepdim=True).clamp_min(1e-12)           # [B,1]
    pi = W / Z                                                # [B,A], uniform Riemann weight

    # Riemann sums (uniform grid) for expectations
    E_b = (pi * b_val).sum(dim=1)                             # [B]
    E_r = (pi * r_val).sum(dim=1)                             # [B]

    # Discrete entropy of pi over the grid (proxy for continuous)
    log_pi = torch.log(pi.clamp_min(1e-12))
    H = -(pi * log_pi).sum(dim=1)                             # [B]
    return E_b, E_r, H


# ------------------------------ Training --------------------------------------
def train_model_based_PIA(
    spec,
    # horizon/grid
    T=None, time_steps: int = 21,
    # outer PIA
    num_policy_iters: int = 100,
    policy_diff_samples: int = 10000, policy_min_delta: float = 1e-3,
    # inner u-fit
    training_path_size: int = 10000, nn_batch_size: int = 5000, num_epochs: int = 200,
    patience: int = 10, min_delta: float = 1e-3,
    # models/opt
    neuron_number_u: int = 64, learning_rate: float = 1e-3, weight_decay: float = 1e-4,
    # actions/entropy
    A_min=None, A_max=None, num_a_points: int = 256, lambda_reg: float = None,
    # sim
    x0_init: float = 1.0, x_floor: float = 1e-8,
    # device
    device: torch.device | str | None = None,
    # checkpoint
    u_ckpt_path: str = "best_u_model_pia.pth",

    verbose: bool = True,
):
    """
    Model-based PIA that fits u(t,x) to MC returns under the current Gibbs policy.
    Returns: u_model, meta dict
    """
    # Clamp helper: only clamp if the spec says so (e.g., EX4 with sqrt)
    def maybe_clamp_state(x):
        return x.clamp_min(x_floor) if getattr(spec, "clamp_state", False) else x

    # defaults
    if T is None: T = spec.T
    if A_min is None: A_min = spec.A_min
    if A_max is None: A_max = spec.A_max
    if lambda_reg is None: lambda_reg = spec.lambda_reg
        
    vprint = print if verbose else (lambda *a, **k: None)

    device = torch.device(device) if isinstance(device, str) else (device or torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # time & action grids
    delta_t = T / (time_steps - 1)
    t_seq   = torch.linspace(0.0, T, time_steps, device=device)
    T_t     = torch.tensor(T, dtype=torch.float32, device=device)
    dt_t    = torch.tensor(delta_t, dtype=torch.float32, device=device)
    x0_t    = torch.tensor(x0_init, dtype=torch.float32, device=device)

    a_grid  = torch.linspace(A_min, A_max, num_a_points, device=device)  # uniform
    # (Note: we use a probability mass over grid points; no explicit delta_a factor needed
    # because pi is normalized discretely.)

    # model & optimizer templates
    u_model = build_u_model(neuron_number_u, input_dim=2, output_dim=1).to(device)

    # PIA loop
    history = []
    for it in range(num_policy_iters):
        vprint(f"\n[PIA] Iteration {it+1}/{num_policy_iters}")
        policy_u = copy.deepcopy(u_model).to(device).eval()   # freeze policy snapshot

        optim = torch.optim.Adam(u_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        sched = ReduceLROnPlateau(optim, mode='min', factor=0.9, patience=5)

        best_loss = float('inf')
        stop_ctr  = 0

        # ======= train u under frozen policy =======
        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for start in range(0, training_path_size, nn_batch_size):
                B = min(nn_batch_size, training_path_size - start)

                # simulate under current policy (expected drift)
                dW = torch.sqrt(dt_t) * torch.randn(B, time_steps - 1, device=device)   # [B,T-1]
                X  = torch.zeros(B, time_steps, device=device); X[:, 0] = x0_t

                for k in range(1, time_steps):
                    t_km1  = t_seq[k-1]
                    x_prev = X[:, k-1]

                    # u_x from frozen policy
                    z = get_u_x(policy_u, t_km1.expand_as(x_prev), x_prev, create_graph=False).detach()

                    # Gibbs expectations for drift & reward
                    E_b, E_r, H = gibbs_expectations(spec, t_km1, x_prev, z, a_grid, lambda_reg)

                    drift_inc = E_b * dt_t
                    sig_prev  = spec.sigma_torch(t_km1, x_prev)   # let the spec handle any σ stabilizing
                    diff_inc  = sig_prev * dW[:, k-1]
                    X[:, k]   = maybe_clamp_state(x_prev + drift_inc + diff_inc)


                # build effective rewards r+λH at each step
                rewards = torch.zeros(B, time_steps - 1, device=device)
                for k in range(time_steps - 1):
                    t_k = t_seq[k]
                    x_k = X[:, k]
                    z_k = get_u_x(policy_u, t_k.expand_as(x_k), x_k, create_graph=False).detach()
                    E_bk, E_rk, Hk = gibbs_expectations(spec, t_k, x_k, z_k, a_grid, lambda_reg)
                    eff_r = E_rk + lambda_reg * Hk
                    rewards[:, k] = eff_r * dt_t

                # terminal payoff
                g_T = spec.g_value_torch(X[:, -1], T_t)  # [B]

                # backward cum-sum: targets[i] = sum_{j=i}^{T-2} rewards_j + g_T
                targets = torch.zeros(B, time_steps - 1, device=device)
                targets[:, -1] = rewards[:, -1] + g_T
                for k in range(time_steps - 3, -1, -1):
                    targets[:, k] = rewards[:, k] + targets[:, k + 1]

                # u predictions at (t_k, X_k) for k = 0..T-2
                u_preds = torch.zeros(B, time_steps - 1, device=device)
                for k in range(time_steps - 1):
                    t_k = t_seq[k]
                    x_k = X[:, k]
                    u_preds[:, k] = u_model(torch.stack([t_k.expand_as(x_k), x_k], dim=1)).squeeze(-1)

                # squared residuals & average
                loss = ((u_preds - targets) ** 2).mean()
                optim.zero_grad()
                loss.backward()
                optim.step()
                sched.step(loss)

                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(1, math.ceil(training_path_size / nn_batch_size))
            vprint(f"  [fit u] Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.6f}")

            # early stopping per inner fit
            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                stop_ctr  = 0
                torch.save(u_model.state_dict(), u_ckpt_path)
            else:
                stop_ctr += 1
                if stop_ctr >= patience:
                    vprint(f"  [fit u] early stop (patience={patience})")
                    break

        # restore best for this policy iteration
        u_model.load_state_dict(torch.load(u_ckpt_path, map_location=device))

        # ======= policy-diff stopping across PIA iterations =======
        with torch.no_grad():
            
            t_s = torch.rand(policy_diff_samples, device=device) * T_t
            
            if getattr(spec, "clamp_state", False):
            # keep positive support when the model requires X ≥ 0 (e.g., sqrt)
                x_s = torch.rand(policy_diff_samples, device=device) * 5.0 + x_floor
            else:
            # symmetric range is safer when negatives are allowed
                x_s = (torch.rand(policy_diff_samples, device=device) * 10.0) - 5.0

            u_new = u_model(torch.stack([t_s, x_s], dim=1)).squeeze(-1)
            u_old = policy_u(torch.stack([t_s, x_s], dim=1)).squeeze(-1)
            diff  = (u_new - u_old).pow(2).mean().item()

        history.append({"iter": it + 1, "best_u_loss": best_loss, "policy_diff": diff})
        vprint(f"  [PIA] policy L2 diff: {diff:.6f}")

        if diff < policy_min_delta:
            vprint(f"[PIA] stopping at iter {it+1} (diff<{policy_min_delta})")
            break

    meta = dict(
        T=T, time_steps=time_steps,
        num_policy_iters=num_policy_iters, history=history,
        num_a_points=num_a_points, lambda_reg=lambda_reg,
        A_min=A_min, A_max=A_max,
    )
    return u_model, meta


# ---------- Safe device utilities ----------
import torch

def normalize_device(device):
    """
    Return a torch.device that is guaranteed to be usable.
    - If device requests CUDA but CUDA is unavailable, fall back to CPU.
    - Accepts str, torch.device, or None.
    """
    if device is None:
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if isinstance(device, str):
        d = device.lower()
        if 'cuda' in d:
            if torch.cuda.is_available():
                return torch.device(d)
            else:
                print("[warn] CUDA requested but not available; falling back to CPU.")
                return torch.device('cpu')
        return torch.device(d)

    if isinstance(device, torch.device):
        if device.type == 'cuda' and not torch.cuda.is_available():
            print("[warn] CUDA device provided but not available; falling back to CPU.")
            return torch.device('cpu')
        return device

    # Unknown type → CPU
    return torch.device('cpu')


def infer_model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device('cpu')
        
# ------------------ testing API ------------------


def probe_u_and_v(spec, u_model, t0: float, x0: float, device=None):
    """
    Report only:
      - estimated u(t0, x0)
      - true u(t0, x0)    (if ProblemSpec provides it)
      - estimated v(t0, x0) = ∂_x u(t0, x0)
      - true v(t0, x0)    (if ProblemSpec provides it)
    """
    base_dev = infer_model_device(u_model)
    device = normalize_device(device if device is not None else base_dev)

    # --- u(t0,x0) can be done without grad
    with torch.no_grad():
        t_val = torch.tensor([t0], dtype=torch.float32, device=device)
        x_val = torch.tensor([x0], dtype=torch.float32, device=device)
        u_hat = u_model.to(device)(torch.stack([t_val, x_val], dim=1)).item()

    # --- v(t0,x0) = ∂_x u needs grad enabled
    u_model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        t = torch.tensor([t0], dtype=torch.float32, device=device, requires_grad=True)
        x = torch.tensor([x0], dtype=torch.float32, device=device, requires_grad=True)
        v_hat = get_u_x(u_model, t, x, create_graph=False).item()

    # Optional truths (gracefully handle missing)
    import math as _math
    def _safe(v):
        try:
            f = float(v)
            return None if _math.isnan(f) else f
        except Exception:
            return None

    u_true = _safe(getattr(spec, "u_true_np", lambda *_: float("nan"))(t0, x0))
    v_true = _safe(getattr(spec, "u_x_true_np", lambda *_: float("nan"))(t0, x0))

    return {"u": u_hat, "u_true": u_true, "v": v_hat, "v_true": v_true}
    
# --- Optional: backward-compatible shim ---

def probe_u_and_w(spec, u_model, t0: float, x0: float, a0: float,
                  lambda_reg: float | None = None, device=None):
    """
    Deprecated shim: kept for compatibility. Ignores a0/lambda_reg and returns only u/v info.
    """
    return probe_u_and_v(spec, u_model, t0=t0, x0=x0, device=device)




####################################################
# ==============================================================================
#                                EXAMPLE RUN
# ==============================================================================
if __name__ == "__main__":
    
    spec = EX7()  # choose your problem

    u_model, meta = train_model_based_PIA(
        spec,
        T=0.4, time_steps=11,
        num_policy_iters=50,
        policy_diff_samples=5000, policy_min_delta=5e-4,
        training_path_size=10000, nn_batch_size=5000, num_epochs=200,
        patience=10, min_delta=1e-3,
        neuron_number_u=64, learning_rate=1e-3, weight_decay=1e-4,
        A_min=spec.A_min, A_max=spec.A_max, num_a_points=96, lambda_reg=spec.lambda_reg,
        x0_init=0.1, x_floor=1e-8,
        device = device, # or "cpu",
        verbose = False,
    )
    
    res = probe_u_and_v(spec, u_model, t0=0.0, x0=0.0)
    rel = res['u']-res['u_true']/res['u_true']
    print(f"Estimated u: {res['u']:.6f} | True u: {res['u_true']}")
    print(f"Estimated v: {res['v']:.6f} | True v: {res['v_true']}")
    print(f"REL:{rel}")
    
    
