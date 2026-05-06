"""Focused universality experiment for the NeurIPS paper §4.

Targets chosen to align with the paper's narrative:
  - x^3 - x          : polynomial (Bramble-Hilbert proof's main route)
  - tanh(2x)         : used in the partition-of-unity construction (§6)
  - sin(x)           : trigonometric case (Remark on trig in §3.4)
  - exp(-x^2)        : smooth analytic, standard benchmark
  - sin(3x)*exp(-x^2/2) : composite (oscillation × envelope), hardest

Architecture: PEM (per-node parametric eml-all-the-way-down,
EML_θ(x,y) = a*exp(b*x+c) + d*ln(e*y+f)), exactly as in the paper.

Two parameterisations:
  R = real parameters, log argument routed through softplus + eps
  C = complex parameters (twin-real (re,im) Parameters), forward in
      complex128, return Re(.). Required to access the trig path through
      ln(-1) = iπ.

Strategy: Adam multi-restart (1500 steps each, lr=5e-3, cosine schedule)
followed by a single LBFGS refinement of the best-of-restarts model
(strong-Wolfe line search, 200 max iter). Wall-clock budget = 30 s/cell.

Note: the existing PEM-real results in `results/results_parametric.json`
already cover this depth-vs-target sweep. We re-run the real half here
with matched protocol (BUDGET_S=30, depths {3,4,5}) so both columns of
the headline table are produced under identical conditions.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _setup  # noqa: F401

import json, time
import torch, numpy as np

from src.eml_tree_parametric import (
    ParametricEMLTree, fit_parametric, lbfgs_refine_parametric)
from src.eml_tree_parametric_complex import (
    CParametricEMLTree, fit_cpem, lbfgs_refine_cpem)

torch.set_default_dtype(torch.float64)


TARGETS = {
    "x3_minus_x":    (lambda x: x**3 - x,                          (-1.5, 1.5)),
    "tanh_2x":       (lambda x: torch.tanh(2 * x),                 (-2.0, 2.0)),
    "sin_x":         (lambda x: torch.sin(x),                      (-3.14, 3.14)),
    "exp_neg_x2":    (lambda x: torch.exp(-x**2),                  (-2.5, 2.5)),
    "sin3x_envelope":(lambda x: torch.sin(3*x)*torch.exp(-x**2/2), (-3.0, 3.0)),
}
DEPTHS    = [3, 4, 5]
BUDGET_S  = 30.0
N_SAMPLES = 200


def fit_real_PEM(fn, x, y, depth, t_budget):
    t0 = time.time()
    best, runs = float("inf"), 0
    best_model = None
    s = 0
    while time.time() - t0 < t_budget * 0.7:
        torch.manual_seed(s); s += 1
        m = ParametricEMLTree(depth, in_dim=1)
        L = fit_parametric(m, x, y, steps=1500, lr=5e-3)
        if L < best:
            best, best_model = L, m
        runs += 1
    if best_model is not None and np.isfinite(best):
        L_ref = lbfgs_refine_parametric(best_model, x, y, max_iter=200)
        if np.isfinite(L_ref):
            best = min(best, L_ref)
    return best, runs


def fit_complex_PEM(fn, x, y, depth, t_budget):
    t0 = time.time()
    best, runs = float("inf"), 0
    best_model = None
    s = 0
    while time.time() - t0 < t_budget * 0.7:
        torch.manual_seed(s); s += 1
        m = CParametricEMLTree(depth, in_dim=1)
        L = fit_cpem(m, x, y, steps=1500, lr=5e-3, lambda_im=1e-3)
        if L < best:
            best, best_model = L, m
        runs += 1
    if best_model is not None and np.isfinite(best):
        L_ref = lbfgs_refine_cpem(best_model, x, y, max_iter=200, lambda_im=1e-3)
        if np.isfinite(L_ref):
            best = min(best, L_ref)
    return best, runs


def main():
    out = []
    grand_t0 = time.time()
    for name, (fn, (a, b)) in TARGETS.items():
        x = torch.linspace(a, b, N_SAMPLES)
        y = fn(x)
        ynrm = float((y**2).mean().sqrt())
        for d in DEPTHS:
            # Real-PEM
            t0 = time.time()
            best, runs = fit_real_PEM(fn, x, y, d, BUDGET_S)
            elapsed = time.time() - t0
            rel = float(np.sqrt(best)/max(ynrm,1e-8)) if np.isfinite(best) else float("inf")
            n_p = ParametricEMLTree(d, in_dim=1).n_params
            out.append({"target": name, "param": "real", "depth": d,
                        "n_params": n_p, "n_runs": runs,
                        "best_relRMSE": rel, "elapsed_s": elapsed})
            print(f"{name:18s} R d={d}  P={n_p:3d}  rel={rel:.3e}  runs={runs}  ({elapsed:.1f}s)")
            # Complex-PEM
            t0 = time.time()
            best, runs = fit_complex_PEM(fn, x, y, d, BUDGET_S)
            elapsed = time.time() - t0
            rel = float(np.sqrt(best)/max(ynrm,1e-8)) if np.isfinite(best) else float("inf")
            n_p = CParametricEMLTree(d, in_dim=1).n_params
            out.append({"target": name, "param": "complex", "depth": d,
                        "n_params": n_p, "n_runs": runs,
                        "best_relRMSE": rel, "elapsed_s": elapsed})
            print(f"{name:18s} C d={d}  P={n_p:3d}  rel={rel:.3e}  runs={runs}  ({elapsed:.1f}s)")
    print(f"total: {time.time()-grand_t0:.1f}s")
    with open("results/results_universality_pem.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
