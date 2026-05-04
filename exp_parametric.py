"""ParametricEMLTree universality sweep.

Per-node 6-param formulation:
    EML_θ_i(x, y) = a·exp(b·x+c) + d·ln(e·y+f)
eml-all-the-way-down — no affine wrapper between nodes.

Compare side-by-side with EMLTree (affine-glue) on canonical 1D targets,
strategy B (Adam multi-restart + LBFGS), 25s/cell.
"""
import json, time
import torch, numpy as np
from eml_tree import EMLTree, fit
from eml_tree_parametric import ParametricEMLTree, fit_parametric, lbfgs_refine_parametric

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
DEPTHS = [2, 3, 4, 5]
BUDGET_S = 25.0


def fit_PEM_strategy_B(fn, x, y, depth, t_budget):
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
    return best, runs, best_model


def fit_EM_strategy_B(fn, x, y, depth, t_budget):
    t0 = time.time()
    best, runs = float("inf"), 0
    best_model = None
    s = 0
    while time.time() - t0 < t_budget * 0.7:
        torch.manual_seed(s); s += 1
        m = EMLTree(depth, in_dim=1, backend="real_softplus")
        L = fit(m, x, y, steps=1500, lr=5e-3)
        if L < best:
            best, best_model = L, m
        runs += 1
    if best_model is not None and np.isfinite(best):
        opt = torch.optim.LBFGS(best_model.parameters(), max_iter=200,
                                line_search_fn="strong_wolfe", history_size=20)
        def closure():
            opt.zero_grad()
            loss = ((best_model(x) - y)**2).mean()
            if not torch.isfinite(loss):
                loss = torch.tensor(1e10, dtype=loss.dtype)
            loss.backward()
            return loss
        try: opt.step(closure)
        except Exception: pass
        with torch.no_grad():
            best = min(best, float(((best_model(x) - y)**2).mean()))
    return best, runs, best_model


def main():
    out = []
    grand_t0 = time.time()
    for name, (fn, (a, b)) in TARGETS.items():
        x = torch.linspace(a, b, 200)
        y = fn(x)
        ynrm = float((y**2).mean().sqrt())
        for d in DEPTHS:
            # PEM (parametric eml-all-the-way-down)
            t0 = time.time()
            best, runs, _ = fit_PEM_strategy_B(fn, x, y, d, BUDGET_S)
            elapsed = time.time() - t0
            rel = float(np.sqrt(best)/max(ynrm,1e-8)) if np.isfinite(best) else float("inf")
            n_p = ParametricEMLTree(d, in_dim=1).n_params
            out.append({"target": name, "model": "PEM", "depth": d,
                        "n_params": n_p, "best_relRMSE": rel,
                        "n_runs": runs, "elapsed_s": elapsed})
            print(f"{name:10s} PEM d={d}  P={n_p:3d}  rel={rel:.3e}  runs={runs:3d}  ({elapsed:.1f}s)")
            # EM (affine-glue eml, the §5 headline)
            t0 = time.time()
            best, runs, _ = fit_EM_strategy_B(fn, x, y, d, BUDGET_S)
            elapsed = time.time() - t0
            rel = float(np.sqrt(best)/max(ynrm,1e-8)) if np.isfinite(best) else float("inf")
            n_p = EMLTree(d, in_dim=1).n_params
            out.append({"target": name, "model": "EM", "depth": d,
                        "n_params": n_p, "best_relRMSE": rel,
                        "n_runs": runs, "elapsed_s": elapsed})
            print(f"{name:10s} EM  d={d}  P={n_p:3d}  rel={rel:.3e}  runs={runs:3d}  ({elapsed:.1f}s)")
    print(f"total: {time.time()-grand_t0:.1f}s")
    with open("results_parametric.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
