"""CParametricEMLTree: complex-parameter PEM, eml-all-the-way-down.

Per-node 6-parameter EML formulation with COMPLEX parameters:

    EML_θ_i(x, y) = a_i · exp(b_i · x + c_i) + d_i · ln(e_i · y + f_i)

where θ_i = (a_i, b_i, c_i, d_i, e_i, f_i) ∈ ℂ^6 (vector for b_i, e_i at
leaf-parent level when input is multi-dimensional).

Storage: each parameter is a twin pair (p_re, p_im) of real Parameters.
On forward, p = p_re + i · p_im is formed; arithmetic and EML run in
complex128. Output is Re(T_θ(x)) for real-valued target fitting.

This is the PEM analogue of CEMLTree (which is the affine-glue version).
The motivation is that the paper's universality construction routes
trigonometric functions through the principal-branch ln(-1) = iπ; with
real parameters only, the imaginary degrees of freedom required for
constructive trig interference are unreachable in slot-space.
"""
from __future__ import annotations
import torch
import torch.nn as nn


_EXP_CLAMP: float = 60.0
_LOG_EPS:   float = 1e-6


def _safe_exp_c(z: torch.Tensor) -> torch.Tensor:
    """exp(z) for complex z, with Re(z) clamped; |exp(i·Im(z))| = 1 so Im is safe."""
    re = torch.clamp(z.real, -_EXP_CLAMP, _EXP_CLAMP)
    im = z.imag
    return torch.exp(torch.complex(re, im))


class CParametricEMLTree(nn.Module):
    """Per-node 6-parameter PEM with complex parameters."""

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
            b_dim = in_dim if is_leaf_parent else 1
            e_dim = in_dim if is_leaf_parent else 1
            for pos in range(n_nodes):
                key = f"l{level}p{pos}"
                # init bias toward paper-eml: a≈1, d≈-1, others ≈0
                # init imag parts small around 0
                a_re_init = torch.tensor(1.0) + init_scale * torch.randn(1).squeeze()
                d_re_init = torch.tensor(-1.0) + init_scale * torch.randn(1).squeeze()
                params[f"{key}_a_re"] = nn.Parameter(a_re_init.to(torch.float64))
                params[f"{key}_a_im"] = nn.Parameter(
                    init_scale * torch.randn(1, dtype=torch.float64).squeeze())
                params[f"{key}_b_re"] = nn.Parameter(
                    init_scale * torch.randn(b_dim, dtype=torch.float64))
                params[f"{key}_b_im"] = nn.Parameter(
                    init_scale * torch.randn(b_dim, dtype=torch.float64))
                params[f"{key}_c_re"] = nn.Parameter(
                    init_scale * torch.randn(1, dtype=torch.float64).squeeze())
                params[f"{key}_c_im"] = nn.Parameter(
                    init_scale * torch.randn(1, dtype=torch.float64).squeeze())
                params[f"{key}_d_re"] = nn.Parameter(d_re_init.to(torch.float64))
                params[f"{key}_d_im"] = nn.Parameter(
                    init_scale * torch.randn(1, dtype=torch.float64).squeeze())
                params[f"{key}_e_re"] = nn.Parameter(
                    init_scale * torch.randn(e_dim, dtype=torch.float64))
                params[f"{key}_e_im"] = nn.Parameter(
                    init_scale * torch.randn(e_dim, dtype=torch.float64))
                params[f"{key}_f_re"] = nn.Parameter(
                    init_scale * torch.randn(1, dtype=torch.float64).squeeze())
                params[f"{key}_f_im"] = nn.Parameter(
                    init_scale * torch.randn(1, dtype=torch.float64).squeeze())
        self.params = nn.ParameterDict(params)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.params.values())

    def _cp(self, key: str, letter: str) -> torch.Tensor:
        return torch.complex(self.params[f"{key}_{letter}_re"],
                             self.params[f"{key}_{letter}_im"])

    def _node(self, key: str, lc: torch.Tensor, rc: torch.Tensor) -> torch.Tensor:
        a = self._cp(key, "a"); c = self._cp(key, "c")
        d = self._cp(key, "d"); f = self._cp(key, "f")
        b = self._cp(key, "b"); e = self._cp(key, "e")
        # If lc/rc come from raw (N,D) input, b and e are vectors of length D.
        # If interior (N,) scalars, b and e are length-1 (squeeze to scalar).
        if lc.dim() == 1:
            arg_exp = b.squeeze() * lc + c
            arg_log = e.squeeze() * rc + f
        else:
            arg_exp = lc.to(torch.complex128) @ b + c
            arg_log = rc.to(torch.complex128) @ e + f
        # log of complex value uses principal branch automatically
        return a * _safe_exp_c(arg_exp) + d * torch.log(arg_log + _LOG_EPS)

    def _eval(self, level: int, pos: int, x_input: torch.Tensor) -> torch.Tensor:
        key = f"l{level}p{pos}"
        is_leaf_parent = (level == self.depth - 1)
        if is_leaf_parent:
            return self._node(key, x_input, x_input)
        else:
            lc = self._eval(level + 1, 2 * pos, x_input)
            rc = self._eval(level + 1, 2 * pos + 1, x_input)
            return self._node(key, lc, rc)

    def forward_complex(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        x_c = x.to(torch.complex128)
        return self._eval(0, 0, x_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Real-valued output: Re(complex tree)."""
        return self.forward_complex(x).real


def fit_cpem(model: CParametricEMLTree, x: torch.Tensor, y: torch.Tensor,
             steps: int = 3000, lr: float = 5e-3,
             lambda_im: float = 1e-3) -> float:
    """Adam fit on Re(T(x)) - y with light Im penalty (keeps tree honest)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    best = float("inf")
    for it in range(steps):
        opt.zero_grad()
        z = model.forward_complex(x)
        loss = ((z.real - y) ** 2).mean()
        if lambda_im > 0:
            loss = loss + lambda_im * (z.imag ** 2).mean()
        if not torch.isfinite(loss):
            return float("inf")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            mse = float(((model.forward_complex(x).real - y) ** 2).mean())
        if mse < best:
            best = mse
    return best


def lbfgs_refine_cpem(model: CParametricEMLTree, x: torch.Tensor, y: torch.Tensor,
                      max_iter: int = 200, lambda_im: float = 1e-3) -> float:
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
    try: opt.step(closure)
    except Exception: pass
    with torch.no_grad():
        return float(((model.forward_complex(x).real - y) ** 2).mean())
