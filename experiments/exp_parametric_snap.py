"""Snap-to-symbol fidelity audit on ParametricEMLTree at depth 3.

Hypothesis: per-parameter discrete candidate sets give cleaner snap recovery
than affine-glue's slot-vertex snap, because each parameter has interpretable
discrete values from the operator's algebra.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _setup  # noqa: F401  (adds project root to sys.path and chdirs)

import json, time
import torch, numpy as np
from src.eml_tree_parametric import ParametricEMLTree, fit_parametric, lbfgs_refine_parametric
from snap.snap_parametric import snap_greedy_parametric, decode_pem

torch.set_default_dtype(torch.float64)

TARGETS = {
    "sin":       (lambda x: torch.sin(x),                          (-3.14, 3.14)),
    "x2":        (lambda x: x**2,                                  (-2.0, 2.0)),
    "abs":       (lambda x: torch.abs(x),                          (-2.0, 2.0)),
    "gauss":     (lambda x: torch.exp(-x**2),                      (-2.5, 2.5)),
    "tanh":      (lambda x: torch.tanh(2*x),                       (-2.0, 2.0)),
    "exp_decay": (lambda x: torch.exp(-x),                         (0.0, 3.0)),
    "rip":       (lambda x: torch.sin(3*x)*torch.exp(-x**2/2),     (-3.0, 3.0)),
}
DEPTH = 3
N_RESTARTS = 12
STEPS = 3000


def fit_PEM_multistart(fn, x, y, depth, n_restarts, steps, lr=5e-3):
    best, best_model = float("inf"), None
    for s in range(n_restarts):
        torch.manual_seed(s)
        m = ParametricEMLTree(depth, in_dim=1)
        L = fit_parametric(m, x, y, steps=steps, lr=lr)
        if L < best:
            best, best_model = L, m
    if best_model is not None:
        L_ref = lbfgs_refine_parametric(best_model, x, y, max_iter=200)
        if np.isfinite(L_ref) and L_ref < best:
            best = L_ref
    return best, best_model


def main():
    out = []
    t0 = time.time()
    for name, (fn, (a, b)) in TARGETS.items():
        x = torch.linspace(a, b, 200)
        y = fn(x)
        ynrm = float((y**2).mean().sqrt())
        # Fit
        best, model = fit_PEM_multistart(fn, x, y, DEPTH, N_RESTARTS, STEPS)
        rel_pre = float(np.sqrt(best)/max(ynrm, 1e-8))
        # Greedy snap
        pre, post, n_snap, snapped_model, snapped_set, total = \
            snap_greedy_parametric(model, x, y, tol_rel=0.20, refit_steps=600)
        rel_post = float(np.sqrt(post)/max(ynrm, 1e-8))
        expr = decode_pem(snapped_model, snapped_set)
        expr_short = expr if len(expr) < 350 else expr[:340] + "...]"
        row = {
            "target": name, "depth": DEPTH, "n_total_params": total,
            "rel_pre": rel_pre, "rel_post": rel_post,
            "n_snapped": n_snap, "frac_snapped": n_snap / total,
            "expression": expr,
        }
        out.append(row)
        print(f"{name:10s}  pre={rel_pre:.3e}  post={rel_post:.3e}  "
              f"snap={n_snap}/{total} ({100*n_snap/total:.0f}%)")
        print(f"           {expr_short}")
    print(f"elapsed: {time.time()-t0:.1f}s")
    with open("results/results_parametric_snap.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
