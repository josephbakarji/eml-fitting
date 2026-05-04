"""1D fitting stress test for EMLTree."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _setup  # noqa: F401  (adds project root to sys.path and chdirs)

import json, time
import torch
import numpy as np
from src.eml_tree import EMLTree, fit_multistart

torch.set_default_dtype(torch.float64)

TARGETS = {
    "sin":          (lambda x: torch.sin(x),                      (-3.14, 3.14)),
    "cos":          (lambda x: torch.cos(x),                      (-3.14, 3.14)),
    "x2":           (lambda x: x**2,                              (-2.0, 2.0)),
    "abs":          (lambda x: torch.abs(x),                      (-2.0, 2.0)),
    "gauss":        (lambda x: torch.exp(-x**2),                  (-2.5, 2.5)),
    "sigmoid":      (lambda x: torch.sigmoid(3 * x),              (-2.0, 2.0)),
    "polyhi":       (lambda x: x**3 - x,                          (-1.5, 1.5)),
    "tanh":         (lambda x: torch.tanh(2 * x),                 (-2.0, 2.0)),
    "log_safe":     (lambda x: torch.log(1 + x**2),               (-2.0, 2.0)),
    "exp_decay":    (lambda x: torch.exp(-x),                     (0.0, 3.0)),
    "rip":          (lambda x: torch.sin(3*x) * torch.exp(-x**2/2),(-3.0, 3.0)),
}

DEPTHS = [1, 2, 3, 4]
N_RESTARTS = 12
N_TRAIN = 200
STEPS = 3000
LR = 5e-3


def y_norm(y: torch.Tensor) -> float:
    return float((y**2).mean().sqrt())


def main():
    x = torch.linspace(-1, 1, N_TRAIN)  # rescaled per target
    results = []
    t0 = time.time()
    for name, (fn, (lo, hi)) in TARGETS.items():
        x_t = torch.linspace(lo, hi, N_TRAIN)
        y_t = fn(x_t)
        ynrm = y_norm(y_t)
        for depth in DEPTHS:
            torch.manual_seed(0)
            best_model, best_loss, losses = fit_multistart(
                fn, x_t, depth=depth, n_restarts=N_RESTARTS,
                steps=STEPS, lr=LR, backend="real_softplus"
            )
            rmse = float(np.sqrt(best_loss))
            relrmse = rmse / max(ynrm, 1e-8)
            success = sum(1 for L in losses if np.sqrt(L) / max(ynrm,1e-8) < 0.05)
            row = {
                "target": name, "depth": depth,
                "n_params": best_model.n_params,
                "best_rmse": rmse, "best_relrmse": relrmse,
                "success_rate": success / N_RESTARTS,
                "losses": losses,
            }
            results.append(row)
            print(f"{name:10s} d={depth} params={best_model.n_params:3d}  "
                  f"relRMSE={relrmse:.3e}  succ={success}/{N_RESTARTS}")
    print(f"elapsed: {time.time()-t0:.1f}s")
    with open("results/results_1d.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
