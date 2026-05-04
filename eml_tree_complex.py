"""CEMLTree: continuous-affine EML tree with COMPLEX parameters.

Storage: each slot's affine vector is stored as twin real Parameters
(p_re, p_im); on forward we form `p = p_re + i*p_im` and use complex128
arithmetic throughout. Adam steps in real parameter space, which is
equivalent to Wirtinger-style complex Adam when the loss depends only on
Re(.) and Im(.) of the output.

Slot semantics (matches EMLTree but in C):
  internal slot:    slot = sum_i a_i * x_i + b * child + c   (a, b, c in C)
  leaf-parent slot: slot = sum_i a_i * x_i + c               (a, c in C)

eml(z, w) = exp(z) - log(w), with principal-branch log on C.

Output: Re(T_theta(x)) for real-valued target fitting.

Hardening:
- Re(z) of the exp argument clamped to [-_EXP_CLAMP, _EXP_CLAMP].
- log argument offset by _LOG_EPS to avoid singularity at 0.
- imag-part penalty hook (lambda_im * mean Im^2) optional in fit.
"""
from __future__ import annotations
import torch
import torch.nn as nn

_EXP_CLAMP: float = 60.0
_LOG_EPS:   float = 1e-6


def _safe_exp_c(z: torch.Tensor) -> torch.Tensor:
    """exp(z) with Re clamp; Im is unbounded but |exp(i Im)| = 1."""
    re = torch.clamp(z.real, -_EXP_CLAMP, _EXP_CLAMP)
    im = z.imag
    return torch.exp(torch.complex(re, im))


class CEMLTree(nn.Module):
    def __init__(self, depth: int, in_dim: int = 1, init_scale: float = 0.3):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = depth
        self.in_dim = in_dim
        self.init_scale = init_scale

        params: dict[str, nn.Parameter] = {}
        for level in range(depth):
            n_nodes = 2 ** level
            is_leaf_parent = (level == depth - 1)
            slot_dim = in_dim + (1 if not is_leaf_parent else 0) + 1
            for pos in range(n_nodes):
                key = f"l{level}p{pos}"
                for side in ("L", "R"):
                    params[f"{key}{side}_re"] = nn.Parameter(
                        init_scale * torch.randn(slot_dim, dtype=torch.float64))
                    params[f"{key}{side}_im"] = nn.Parameter(
                        init_scale * torch.randn(slot_dim, dtype=torch.float64))
        self.params = nn.ParameterDict(params)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.params.values())

    def _slot_param(self, key: str) -> torch.Tensor:
        return torch.complex(self.params[f"{key}_re"], self.params[f"{key}_im"])

    def _slot(self, key: str, x_c: torch.Tensor,
              child: torch.Tensor | None) -> torch.Tensor:
        p = self._slot_param(key)
        D = self.in_dim
        a = p[:D]
        s = x_c @ a    # (N,) complex
        if child is not None:
            b = p[D]
            c = p[D + 1]
            s = s + b * child + c
        else:
            c = p[D]
            s = s + c
        return s

    def _eval(self, level: int, pos: int, x_c: torch.Tensor) -> torch.Tensor:
        key = f"l{level}p{pos}"
        is_leaf_parent = (level == self.depth - 1)
        if is_leaf_parent:
            l_in = self._slot(key + "L", x_c, None)
            r_in = self._slot(key + "R", x_c, None)
        else:
            lc = self._eval(level + 1, 2 * pos,     x_c)
            rc = self._eval(level + 1, 2 * pos + 1, x_c)
            l_in = self._slot(key + "L", x_c, lc)
            r_in = self._slot(key + "R", x_c, rc)
        return _safe_exp_c(l_in) - torch.log(r_in + _LOG_EPS)

    def forward_complex(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        x_c = x.to(torch.complex128)
        return self._eval(0, 0, x_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Real-valued output: Re(complex tree)."""
        return self.forward_complex(x).real


def fit_complex(model: CEMLTree, x: torch.Tensor, y: torch.Tensor,
                steps: int = 3000, lr: float = 5e-3,
                lambda_im: float = 0.0,
                lambda_im_end: float | None = None,
                anneal: str = "cosine") -> float:
    """Adam fit on Re(T(x)) - y with optional Im-penalty schedule.

    If lambda_im_end is None: constant penalty = lambda_im.
    Else: ramp from lambda_im (start) to lambda_im_end (end) over `steps`
    iterations with shape `anneal` in {"linear", "cosine", "exp"}.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    best = float("inf")
    use_anneal = lambda_im_end is not None
    for it in range(steps):
        if use_anneal:
            t = it / max(1, steps - 1)
            if anneal == "linear":
                w = t
            elif anneal == "cosine":
                w = 0.5 * (1 - torch.cos(torch.tensor(t * 3.141592653589793)).item())
            elif anneal == "exp":
                # Exponential ramp: small early, large late
                import math
                w = (math.exp(t) - 1) / (math.e - 1)
            else:
                raise ValueError(f"unknown anneal: {anneal}")
            lam = (1 - w) * lambda_im + w * lambda_im_end
        else:
            lam = lambda_im
        opt.zero_grad()
        z = model.forward_complex(x)
        loss = ((z.real - y) ** 2).mean()
        if lam > 0:
            loss = loss + lam * (z.imag ** 2).mean()
        if not torch.isfinite(loss):
            return float("inf")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            z2 = model.forward_complex(x)
            mse = float(((z2.real - y) ** 2).mean())
        if mse < best:
            best = mse
    return best


def lbfgs_refine_complex(model: CEMLTree, x: torch.Tensor, y: torch.Tensor,
                         max_iter: int = 200, lambda_im: float = 0.0) -> float:
    opt = torch.optim.LBFGS(model.parameters(), max_iter=max_iter,
                            line_search_fn="strong_wolfe", history_size=20)
    def closure():
        opt.zero_grad()
        z = model.forward_complex(x)
        loss = ((z.real - y) ** 2).mean()
        if lambda_im > 0:
            loss = loss + lambda_im * (z.imag ** 2).mean()
        if not torch.isfinite(loss):
            loss = torch.tensor(1e10, dtype=loss.dtype)
        loss.backward()
        return loss
    try:
        opt.step(closure)
    except Exception:
        pass
    with torch.no_grad():
        z = model.forward_complex(x)
        return float(((z.real - y) ** 2).mean())
