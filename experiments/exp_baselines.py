"""Compare EMLTree to polynomial / MLP / RBF baselines at matched param count."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _setup  # noqa: F401  (adds project root to sys.path and chdirs)

import json, time
import torch
import numpy as np
from src.eml_tree import EMLTree, fit, fit_multistart

torch.set_default_dtype(torch.float64)

TARGETS = {
    "sin":       (lambda x: torch.sin(x),                          (-3.14, 3.14)),
    "x2":        (lambda x: x**2,                                  (-2.0, 2.0)),
    "abs":       (lambda x: torch.abs(x),                          (-2.0, 2.0)),
    "gauss":     (lambda x: torch.exp(-x**2),                      (-2.5, 2.5)),
    "tanh":      (lambda x: torch.tanh(2 * x),                     (-2.0, 2.0)),
    "rip":       (lambda x: torch.sin(3*x) * torch.exp(-x**2/2),   (-3.0, 3.0)),
}

# Param-count budgets to compare
BUDGETS = {
    "tiny":   14,   # eml depth 2
    "small":  34,   # eml depth 3
    "medium": 74,   # eml depth 4
}


def fit_polynomial(x, y, n_params):
    """Least-squares polynomial of degree n_params - 1."""
    deg = n_params - 1
    X = torch.stack([x**k for k in range(deg + 1)], dim=1)
    coef = torch.linalg.lstsq(X, y).solution
    pred = X @ coef
    return float(((pred - y)**2).mean())


class TinyMLP(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, hidden, dtype=torch.float64),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, 1, dtype=torch.float64),
        )
    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(-1)
        return self.net(x).squeeze(-1)


def fit_mlp(x, y, n_params, n_restarts=8, steps=3000):
    # MLP params = 1*h + h + h*1 + 1 = 3h+1. So h=(n-1)/3
    h = max(2, (n_params - 1) // 3)
    best = float("inf")
    for s in range(n_restarts):
        torch.manual_seed(s)
        m = TinyMLP(h)
        opt = torch.optim.Adam(m.parameters(), lr=5e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        for it in range(steps):
            opt.zero_grad()
            pred = m(x)
            loss = ((pred - y)**2).mean()
            loss.backward(); opt.step(); sched.step()
            if loss.item() < best: best = loss.item()
    return best


class RBFNet(torch.nn.Module):
    def __init__(self, n_centers, x_range):
        super().__init__()
        self.centers = torch.nn.Parameter(
            torch.linspace(x_range[0], x_range[1], n_centers).to(torch.float64))
        self.log_widths = torch.nn.Parameter(torch.zeros(n_centers, dtype=torch.float64))
        self.weights = torch.nn.Parameter(torch.zeros(n_centers, dtype=torch.float64))
        self.bias = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    def forward(self, x):
        d = x.unsqueeze(-1) - self.centers.unsqueeze(0)
        w = torch.exp(self.log_widths)
        phi = torch.exp(-(d**2) / (2 * w**2 + 1e-6))
        return phi @ self.weights + self.bias


def fit_rbf(x, y, n_params, x_range, n_restarts=8, steps=3000):
    # RBF params = 3*K + 1. K = (n-1)/3
    K = max(2, (n_params - 1) // 3)
    best = float("inf")
    for s in range(n_restarts):
        torch.manual_seed(s)
        m = RBFNet(K, x_range)
        opt = torch.optim.Adam(m.parameters(), lr=5e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        for it in range(steps):
            opt.zero_grad()
            pred = m(x)
            loss = ((pred - y)**2).mean()
            loss.backward(); opt.step(); sched.step()
            if loss.item() < best: best = loss.item()
    return best


def main():
    out = []
    t0 = time.time()
    for name, (fn, rng) in TARGETS.items():
        x = torch.linspace(rng[0], rng[1], 200)
        y = fn(x)
        ynrm = float((y**2).mean().sqrt())
        for budget_name, P in BUDGETS.items():
            # EMLTree
            depth = {14: 2, 34: 3, 74: 4}[P]
            t1 = time.time()
            _, eml_loss, _ = fit_multistart(fn, x, depth=depth, n_restarts=8,
                                             steps=3000, lr=5e-3)
            eml_t = time.time() - t1
            # poly
            poly_loss = fit_polynomial(x, y, P)
            # mlp
            t1 = time.time()
            mlp_loss = fit_mlp(x, y, P, n_restarts=8, steps=3000)
            mlp_t = time.time() - t1
            # rbf
            t1 = time.time()
            rbf_loss = fit_rbf(x, y, P, rng, n_restarts=8, steps=3000)
            rbf_t = time.time() - t1

            row = {
                "target": name, "budget": budget_name, "n_params": P,
                "eml_relRMSE":  np.sqrt(eml_loss)/max(ynrm,1e-8),
                "poly_relRMSE": np.sqrt(poly_loss)/max(ynrm,1e-8),
                "mlp_relRMSE":  np.sqrt(mlp_loss)/max(ynrm,1e-8),
                "rbf_relRMSE":  np.sqrt(rbf_loss)/max(ynrm,1e-8),
                "eml_time": eml_t, "mlp_time": mlp_t, "rbf_time": rbf_t,
            }
            out.append(row)
            print(f"{name:8s} {budget_name:6s} P={P:3d}  "
                  f"eml={row['eml_relRMSE']:.2e}  "
                  f"poly={row['poly_relRMSE']:.2e}  "
                  f"mlp={row['mlp_relRMSE']:.2e}  "
                  f"rbf={row['rbf_relRMSE']:.2e}")
    print(f"elapsed: {time.time()-t0:.1f}s")
    with open("results/results_baselines.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
