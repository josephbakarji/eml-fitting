"""Test complex_real backend on oscillatory + sanity targets.

The complex_real backend evaluates eml(x,y) = exp(x) - log(y) fully in
complex128 (no softplus on y) and returns Re(.). Parameters remain real.
This admits the paper's complex-mediated trig constructions.

Targets:
  - rip: sin(3x)*exp(-x^2/2)   -- the persistent failure
  - osc: sin(3x)               -- pure oscillation
  - cos2: cos(2x)              -- pure oscillation
  - x2, exp_decay              -- sanity (should not regress)

Strategy: B (Adam + LBFGS refine), 25s budget per (target, depth).
Compare side-by-side to real_softplus.
"""
import json, time
import torch, numpy as np
from eml_tree import EMLTree, fit
from exp_strategy import strategy_B  # works regardless of backend if model created right

torch.set_default_dtype(torch.float64)

TARGETS = {
    "rip":       (lambda x: torch.sin(3*x)*torch.exp(-x**2/2),     (-3.0, 3.0)),
    "osc":       (lambda x: torch.sin(3*x),                        (-3.14, 3.14)),
    "cos2":      (lambda x: torch.cos(2*x),                        (-3.14, 3.14)),
    "x2":        (lambda x: x**2,                                  (-2.0, 2.0)),
    "exp_decay": (lambda x: torch.exp(-x),                         (0.0, 3.0)),
}
DEPTHS = [3, 4, 5]
BUDGET_S = 25.0


def strategy_B_backend(target_fn, x, y, depth, t_budget, backend):
    """Strategy B but with explicit backend choice."""
    t0 = time.time()
    best, runs = float("inf"), 0
    best_model = None
    s = 0
    while time.time() - t0 < t_budget * 0.7:
        torch.manual_seed(s); s += 1
        m = EMLTree(depth, in_dim=1, backend=backend)
        L = fit(m, x, y, steps=1500, lr=5e-3)
        if L < best:
            best, best_model = L, m
        runs += 1
    # LBFGS refine
    if best_model is not None and np.isfinite(best):
        opt = torch.optim.LBFGS(best_model.parameters(), max_iter=200,
                                line_search_fn="strong_wolfe", history_size=20)
        def closure():
            opt.zero_grad()
            pred = best_model(x)
            loss = ((pred - y)**2).mean()
            if not torch.isfinite(loss):
                loss = torch.tensor(1e10, dtype=loss.dtype)
            loss.backward()
            return loss
        try:
            opt.step(closure)
        except Exception:
            pass
        with torch.no_grad():
            L_ref = float(((best_model(x) - y)**2).mean())
        best = min(best, L_ref) if np.isfinite(L_ref) else best
    return best, runs


def main():
    out = []
    grand_t0 = time.time()
    for name, (fn, (a, b)) in TARGETS.items():
        x = torch.linspace(a, b, 200)
        y = fn(x)
        ynrm = float((y**2).mean().sqrt())
        for d in DEPTHS:
            for backend in ("real_softplus", "complex_real"):
                t0 = time.time()
                try:
                    best, runs = strategy_B_backend(fn, x, y, d, BUDGET_S, backend)
                except Exception as e:
                    best, runs = float("inf"), -1
                    print(f"  ERROR {name}/{backend}/d={d}: {e}")
                elapsed = time.time() - t0
                rel = (float(np.sqrt(best) / max(ynrm, 1e-8))
                       if np.isfinite(best) else float("inf"))
                row = {
                    "target": name, "backend": backend, "depth": d,
                    "n_params": EMLTree(d, in_dim=1, backend=backend).n_params,
                    "budget_s": BUDGET_S, "elapsed_s": elapsed, "n_runs": runs,
                    "best_loss": best, "best_relRMSE": rel,
                }
                out.append(row)
                print(f"{name:10s} {backend:14s} d={d}  rel={rel:.3e}  "
                      f"runs={runs:3d}  ({elapsed:.1f}s)")
    print(f"total: {time.time()-grand_t0:.1f}s")
    with open("results_complex.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
