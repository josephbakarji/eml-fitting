"""Push depth 5-6 with the best strategy. Headline figure: error vs depth.

Run multi-restart Adam with curriculum warm-start (strategy C) at fixed
wall-clock per (target, depth). 11 canonical 1D targets, depths 2..6.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _setup  # noqa: F401  (adds project root to sys.path and chdirs)

import json, time
import torch, numpy as np
from src.eml_tree import EMLTree, fit
from experiments.exp_strategy import strategy_B  # LBFGS-refined (winner)

torch.set_default_dtype(torch.float64)

TARGETS = {
    "sin":       (lambda x: torch.sin(x),                          (-3.14, 3.14)),
    "cos":       (lambda x: torch.cos(x),                          (-3.14, 3.14)),
    "x2":        (lambda x: x**2,                                  (-2.0, 2.0)),
    "abs":       (lambda x: torch.abs(x),                          (-2.0, 2.0)),
    "gauss":     (lambda x: torch.exp(-x**2),                      (-2.5, 2.5)),
    "sigmoid":   (lambda x: torch.sigmoid(3*x),                    (-2.0, 2.0)),
    "polyhi":    (lambda x: x**3 - x,                              (-1.5, 1.5)),
    "tanh":      (lambda x: torch.tanh(2*x),                       (-2.0, 2.0)),
    "log_safe":  (lambda x: torch.log(1 + x**2),                   (-2.0, 2.0)),
    "exp_decay": (lambda x: torch.exp(-x),                         (0.0, 3.0)),
    "rip":       (lambda x: torch.sin(3*x)*torch.exp(-x**2/2),     (-3.0, 3.0)),
}
DEPTHS = [2, 3, 4, 5, 6]
BUDGET_S = 25.0  # per (target, depth)


def main():
    out = []
    grand_t0 = time.time()
    for name, (fn, (a, b)) in TARGETS.items():
        x = torch.linspace(a, b, 200)
        y = fn(x)
        ynrm = float((y**2).mean().sqrt())
        for d in DEPTHS:
            t0 = time.time()
            try:
                best, runs = strategy_B(fn, x, y, d, BUDGET_S)
            except Exception as e:
                best, runs = float("inf"), -1
                print(f"  ERROR {name}/d={d}: {e}")
            elapsed = time.time() - t0
            rel = float(np.sqrt(best) / max(ynrm, 1e-8)) if np.isfinite(best) else float("inf")
            n_params = EMLTree(d, in_dim=1).n_params
            row = {
                "target": name, "depth": d, "n_params": n_params,
                "budget_s": BUDGET_S, "elapsed_s": elapsed, "n_runs": runs,
                "best_loss": best, "best_relRMSE": rel,
            }
            out.append(row)
            print(f"{name:10s} d={d}  P={n_params:4d}  rel={rel:.3e}  "
                  f"runs={runs:3d}  ({elapsed:.1f}s)")
    print(f"total: {time.time()-grand_t0:.1f}s")
    with open("results/results_universality.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
