"""2D fitting test: bivariate targets with input dim = 2."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _setup  # noqa: F401  (adds project root to sys.path and chdirs)

import json, time
import torch
import numpy as np
from src.eml_tree import EMLTree, fit_multistart

torch.set_default_dtype(torch.float64)


def franke(xy):
    x, y = xy[..., 0], xy[..., 1]
    a = 0.75 * torch.exp(-((9*x-2)**2 + (9*y-2)**2) / 4)
    b = 0.75 * torch.exp(-((9*x+1)**2)/49 - (9*y+1)/10)
    c = 0.5  * torch.exp(-((9*x-7)**2 + (9*y-3)**2) / 4)
    d = -0.2 * torch.exp(-((9*x-4)**2 + (9*y-7)**2))
    return a + b + c + d

def sincos(xy):
    return torch.sin(xy[..., 0]) * torch.cos(xy[..., 1])

def saddle(xy):
    return xy[..., 0]**2 - xy[..., 1]**2

def rosenbrock(xy):
    x, y = xy[..., 0], xy[..., 1]
    return (1 - x)**2 + 100 * (y - x**2)**2


TARGETS = {
    "franke":     (franke,      (0, 1, 0, 1)),
    "sincos":     (sincos,      (-3, 3, -3, 3)),
    "saddle":     (saddle,      (-2, 2, -2, 2)),
    "rosenbrock": (rosenbrock,  (-1.5, 1.5, -1.0, 2.0)),
}

DEPTHS = [2, 3, 4]
N_RESTARTS = 8
STEPS = 3000
GRID = 25  # 25x25 = 625 points


def main():
    out = []
    t0 = time.time()
    for name, (fn, (xa,xb,ya,yb)) in TARGETS.items():
        xs = torch.linspace(xa, xb, GRID)
        ys = torch.linspace(ya, yb, GRID)
        X, Y = torch.meshgrid(xs, ys, indexing='ij')
        XY = torch.stack([X.flatten(), Y.flatten()], dim=1)
        Z  = fn(XY)
        znrm = float((Z**2).mean().sqrt())
        for d in DEPTHS:
            t1 = time.time()
            _, best, losses = fit_multistart(fn, XY, depth=d, in_dim=2,
                                              n_restarts=N_RESTARTS, steps=STEPS,
                                              lr=5e-3)
            elapsed = time.time() - t1
            relrmse = np.sqrt(best) / max(znrm, 1e-8)
            succ = sum(1 for L in losses if np.sqrt(L)/max(znrm,1e-8) < 0.05)
            row = {"target": name, "depth": d,
                   "n_params": EMLTree(d, in_dim=2).n_params,
                   "relRMSE": relrmse, "success": succ, "n_restarts": N_RESTARTS,
                   "time_s": elapsed}
            out.append(row)
            print(f"{name:11s} d={d}  P={row['n_params']:3d}  "
                  f"relRMSE={relrmse:.3e}  succ={succ}/{N_RESTARTS}  "
                  f"({elapsed:.1f}s)")
    print(f"total: {time.time()-t0:.1f}s")
    with open("results/results_2d.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
