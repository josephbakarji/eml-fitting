"""
Continuous EMLTree: a binary tree of eml(x,y) = exp(x) - log(y) with affine
leaves and learnable affine input-mixing at every internal node.

Differences from Odrzywolek/Ninhache repo:
- They snap to a discrete symbolic grammar (softmax over {1, x, child}).
  We instead allow CONTINUOUS coefficients on every input slot:
      input = a * x + b * (child output if any) + c
  i.e. the user's "eml(ax, by)" idea, generalized.
- Goal here is data-fitting flexibility, not exact symbolic recovery.

Complex handling:
- log of negative input gives complex output. We support two modes:
    'real_softplus' : map right-arg through softplus + epsilon, return real.
    'complex_real'  : evaluate fully in complex128, return Re().
- For symbolic-completeness in principle we need 'complex_real'; for
  data-fitting on R, both work.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


_EXP_CLAMP = 60.0          # avoid exp overflow
_LOG_EPS   = 1e-6          # safe-log offset


def _safe_exp(x: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(x):
        return torch.exp(torch.complex(torch.clamp(x.real, -_EXP_CLAMP, _EXP_CLAMP),
                                       x.imag))
    return torch.exp(torch.clamp(x, -_EXP_CLAMP, _EXP_CLAMP))


def _eml_real_softplus(x_l: torch.Tensor, x_r: torch.Tensor) -> torch.Tensor:
    """eml on R with right-arg passed through softplus + eps so log is finite."""
    pos_r = F.softplus(x_r) + _LOG_EPS
    return _safe_exp(x_l) - torch.log(pos_r)


def _eml_complex_real(x_l: torch.Tensor, x_r: torch.Tensor) -> torch.Tensor:
    """eml in C, returning Re(.)."""
    xl = x_l if torch.is_complex(x_l) else x_l.to(torch.complex128)
    xr = x_r if torch.is_complex(x_r) else x_r.to(torch.complex128)
    # Avoid log(0) singularity
    xr = xr + _LOG_EPS
    out = _safe_exp(xl) - torch.log(xr)
    return out.real


_BACKENDS = {
    "real_softplus": _eml_real_softplus,
    "complex_real":  _eml_complex_real,
}


class EMLTree(nn.Module):
    """Full binary EML tree with continuous affine input mixing.

    Each internal node has two input slots (left, right). Each slot is:
        slot = a*x + b*child + c        (internal level: child = subtree value)
        slot = a*x + c                  (leaf-parent level: no child)

    For multivariate input of dim D:
        slot = sum_i a_i * x_i + b * child + c

    At depth d:
        - leaf-parent nodes contribute (D+1) params per slot
        - internal nodes contribute   (D+2) params per slot
    Total params at depth d, input dim D:
        2 * sum_{level=0..d-1} 2^level * ((D+2) if level<d-1 else (D+1))
    """

    def __init__(self, depth: int, in_dim: int = 1,
                 backend: str = "real_softplus", init_scale: float = 0.5):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if backend not in _BACKENDS:
            raise ValueError(f"backend must be in {list(_BACKENDS)}")
        self.depth = depth
        self.in_dim = in_dim
        self.backend = backend
        self._eml = _BACKENDS[backend]
        self.init_scale = init_scale

        params: dict[str, nn.Parameter] = {}
        for level in range(depth):
            n_nodes = 2 ** level
            is_leaf_parent = (level == depth - 1)
            slot_dim = in_dim + (1 if not is_leaf_parent else 0) + 1
            for pos in range(n_nodes):
                key = f"l{level}p{pos}"
                # Init small random; bias init to ~0
                params[f"{key}L"] = nn.Parameter(init_scale * torch.randn(slot_dim))
                params[f"{key}R"] = nn.Parameter(init_scale * torch.randn(slot_dim))
        self.params = nn.ParameterDict(params)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.params.values())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, D) or (N,) for D=1. Returns (N,)."""
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self._eval(0, 0, x)

    def _slot(self, p: torch.Tensor, x: torch.Tensor,
              child: torch.Tensor | None) -> torch.Tensor:
        # p layout: [a_0..a_{D-1}, (b if child), c]
        D = self.in_dim
        a = p[:D]
        # x: (N, D)
        s = x @ a            # (N,)
        if child is not None:
            b = p[D]
            c = p[D + 1]
            s = s + b * child + c
        else:
            c = p[D]
            s = s + c
        return s

    def _eval(self, level: int, pos: int, x: torch.Tensor) -> torch.Tensor:
        key = f"l{level}p{pos}"
        is_leaf_parent = (level == self.depth - 1)
        if is_leaf_parent:
            l_in = self._slot(self.params[f"{key}L"], x, None)
            r_in = self._slot(self.params[f"{key}R"], x, None)
        else:
            lc = self._eval(level + 1, 2 * pos,     x)
            rc = self._eval(level + 1, 2 * pos + 1, x)
            l_in = self._slot(self.params[f"{key}L"], x, lc)
            r_in = self._slot(self.params[f"{key}R"], x, rc)
        return self._eml(l_in, r_in)


def fit(model: EMLTree, x: torch.Tensor, y: torch.Tensor,
        steps: int = 3000, lr: float = 5e-3, verbose: bool = False) -> float:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    best = float("inf")
    for it in range(steps):
        opt.zero_grad()
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        if not torch.isfinite(loss):
            return float("inf")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if loss.item() < best:
            best = loss.item()
        if verbose and it % 500 == 0:
            print(f"  step {it:5d}  loss={loss.item():.4e}")
    return best


def fit_multistart(target_fn, x: torch.Tensor, depth: int, in_dim: int = 1,
                   n_restarts: int = 8, steps: int = 3000, lr: float = 5e-3,
                   backend: str = "real_softplus", seed_base: int = 0):
    y = target_fn(x)
    if y.dim() > 1:
        y = y.squeeze(-1)
    best_loss, best_model = float("inf"), None
    losses = []
    for s in range(n_restarts):
        torch.manual_seed(seed_base + s)
        m = EMLTree(depth=depth, in_dim=in_dim, backend=backend)
        L = fit(m, x, y, steps=steps, lr=lr)
        losses.append(L)
        if L < best_loss:
            best_loss, best_model = L, m
    return best_model, best_loss, losses
