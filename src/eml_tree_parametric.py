"""ParametricEMLTree: per-node 6-parameter eml formulation.

Each node i computes:

    EML_θ_i(x, y) = a_i · exp(b_i · x + c_i) + d_i · ln(e_i · y + f_i)

with theta_i = (a_i, b_i, c_i, d_i, e_i, f_i). The paper's parameter-free
eml(x, y) = exp(x) - ln(y) is the special case
(a, b, c, d, e, f) = (1, 1, 0, -1, 1, 0).

Critically: x and y at internal nodes are the RAW eml outputs of the two
children — no affine wrapper between nodes. This is the "eml all the way
down" architecture.

At the deepest level (leaf-parents), x and y refer to the input variable
itself. For multivariate input of dim D, b_i and e_i become vectors at the
leaf-parent level only (so b · x_input is a dot product); at all higher
levels b_i and e_i remain scalars (children outputs are scalars).

Numerical safeguards:
- exp argument clamp to [-_EXP_CLAMP, _EXP_CLAMP]
- log argument routed through softplus + eps (real_softplus backend) so that
  the operator stays in R. We keep the 'complex_real' option for parity.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


_EXP_CLAMP: float = 60.0
_LOG_EPS:   float = 1e-6


def _safe_exp(z):
    if torch.is_complex(z):
        return torch.exp(torch.complex(torch.clamp(z.real, -_EXP_CLAMP, _EXP_CLAMP),
                                       z.imag))
    return torch.exp(torch.clamp(z, -_EXP_CLAMP, _EXP_CLAMP))


def _eml_real(arg_exp, arg_log, a, d):
    """a*exp(arg_exp) + d*log(softplus(arg_log) + eps)."""
    pos = F.softplus(arg_log) + _LOG_EPS
    return a * _safe_exp(arg_exp) + d * torch.log(pos)


def _eml_complex(arg_exp, arg_log, a, d):
    """a*exp(arg_exp) + d*log(arg_log + eps), in C^128, returning Re()."""
    e = _safe_exp(arg_exp.to(torch.complex128))
    l = torch.log(arg_log.to(torch.complex128) + _LOG_EPS)
    return (a.to(torch.complex128) * e + d.to(torch.complex128) * l).real


_BACKENDS = {"real_softplus": _eml_real, "complex_real": _eml_complex}


class ParametricEMLTree(nn.Module):
    """Per-node 6-parameter eml tree, eml-all-the-way-down.

    Tree structure: full binary, total internal nodes = 2^depth - 1.
    Levels 0 .. depth-1. Level 0 is the root, level depth-1 is the
    leaf-parent (deepest internal node).

    At a leaf-parent node, the "children" are the input variable itself —
    so x and y in the formula become the input vector x_input.
    Internal-node b and e are scalars (children are scalar eml outputs).
    Leaf-parent b and e are D-dim vectors (input is D-dim).
    """

    def __init__(self, depth: int, in_dim: int = 1,
                 backend: str = "real_softplus", init_scale: float = 0.3):
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
            b_dim = in_dim if is_leaf_parent else 1
            e_dim = in_dim if is_leaf_parent else 1
            for pos in range(n_nodes):
                key = f"l{level}p{pos}"
                # Initialize a_i ~ 1, d_i ~ -1 (paper's eml as starting point), else small
                params[f"{key}_a"] = nn.Parameter(
                    torch.tensor(1.0, dtype=torch.float64) +
                    init_scale * torch.randn(1, dtype=torch.float64).squeeze())
                params[f"{key}_b"] = nn.Parameter(
                    init_scale * torch.randn(b_dim, dtype=torch.float64))
                params[f"{key}_c"] = nn.Parameter(
                    init_scale * torch.randn(1, dtype=torch.float64).squeeze())
                params[f"{key}_d"] = nn.Parameter(
                    torch.tensor(-1.0, dtype=torch.float64) +
                    init_scale * torch.randn(1, dtype=torch.float64).squeeze())
                params[f"{key}_e"] = nn.Parameter(
                    init_scale * torch.randn(e_dim, dtype=torch.float64))
                params[f"{key}_f"] = nn.Parameter(
                    init_scale * torch.randn(1, dtype=torch.float64).squeeze())
        self.params = nn.ParameterDict(params)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.params.values())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self._eval(0, 0, x)

    def _node(self, key: str, lc, rc, x_input):
        """Compute EML_theta_i(lc, rc) at this node."""
        a = self.params[f"{key}_a"]
        b = self.params[f"{key}_b"]
        c = self.params[f"{key}_c"]
        d = self.params[f"{key}_d"]
        e = self.params[f"{key}_e"]
        f = self.params[f"{key}_f"]
        # arg_exp = b · lc + c
        # arg_log = e · rc + f
        # If lc/rc are scalars (interior), b/e are scalars too.
        # If lc/rc are vectors (input), b/e are vectors of matching length.
        if lc.dim() == 1:           # (N,) scalar children outputs
            arg_exp = b.squeeze() * lc + c
            arg_log = e.squeeze() * rc + f
        else:                       # (N, D) — leaf-parent with raw input
            arg_exp = lc @ b + c    # (N,)
            arg_log = rc @ e + f
        return self._eml(arg_exp, arg_log, a, d)

    def _eval(self, level: int, pos: int, x_input: torch.Tensor) -> torch.Tensor:
        key = f"l{level}p{pos}"
        is_leaf_parent = (level == self.depth - 1)
        if is_leaf_parent:
            # children are the raw input
            return self._node(key, x_input, x_input, x_input)
        else:
            lc = self._eval(level + 1, 2 * pos,     x_input)
            rc = self._eval(level + 1, 2 * pos + 1, x_input)
            return self._node(key, lc, rc, x_input)


def fit_parametric(model: ParametricEMLTree, x: torch.Tensor, y: torch.Tensor,
                   steps: int = 3000, lr: float = 5e-3) -> float:
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
    return best


def lbfgs_refine_parametric(model, x, y, max_iter=200):
    opt = torch.optim.LBFGS(model.parameters(), max_iter=max_iter,
                            line_search_fn="strong_wolfe", history_size=20)
    def closure():
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        if not torch.isfinite(loss):
            loss = torch.tensor(1e10, dtype=loss.dtype)
        loss.backward()
        return loss
    try: opt.step(closure)
    except Exception: pass
    with torch.no_grad():
        return float(((model(x) - y) ** 2).mean())
