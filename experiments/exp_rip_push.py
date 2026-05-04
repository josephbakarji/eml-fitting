"""Push the rip target with extended budget to test if the wall breaks.

Compare CC (complex params) at depths 3..6 with 60s budget per depth, plus
R (real-soft, real params) for paired control. Best schedule from
exp_complex_anneal.py is used (anneal 0 -> 1e-1).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _setup  # noqa: F401  (adds project root to sys.path and chdirs)

import json, time
import torch, numpy as np
from src.eml_tree import EMLTree, fit
from src.eml_tree_complex import CEMLTree, fit_complex, lbfgs_refine_complex

torch.set_default_dtype(torch.float64)


def fn_rip(x):
    return torch.sin(3*x) * torch.exp(-x**2/2)


def fit_R(x, y, depth, t_budget):
    t0 = time.time()
    best, runs = float("inf"), 0
    best_model = None
    s = 0
    while time.time() - t0 < t_budget * 0.8:
        torch.manual_seed(s); s += 1
        m = EMLTree(depth, in_dim=1, backend="real_softplus")
        L = fit(m, x, y, steps=2000, lr=5e-3)
        if L < best:
            best, best_model = L, m
        runs += 1
    if best_model is not None and np.isfinite(best):
        opt = torch.optim.LBFGS(best_model.parameters(), max_iter=300,
                                line_search_fn="strong_wolfe", history_size=30)
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
    return best, runs


def fit_CC(x, y, depth, t_budget):
    t0 = time.time()
    best, runs = float("inf"), 0
    best_model = None
    s = 0
    while time.time() - t0 < t_budget * 0.8:
        torch.manual_seed(s); s += 1
        m = CEMLTree(depth, in_dim=1)
        L = fit_complex(m, x, y, steps=2000, lr=5e-3,
                        lambda_im=0.0, lambda_im_end=1e-1, anneal="cosine")
        if L < best:
            best, best_model = L, m
        runs += 1
    if best_model is not None and np.isfinite(best):
        L_ref = lbfgs_refine_complex(best_model, x, y, max_iter=300,
                                      lambda_im=1e-1)
        best = min(best, L_ref) if np.isfinite(L_ref) else best
    return best, runs


def main():
    x = torch.linspace(-3, 3, 200)
    y = fn_rip(x)
    ynrm = float((y**2).mean().sqrt())
    out = []
    grand_t0 = time.time()
    BUDGET = 60.0
    for d in [3, 4, 5, 6]:
        # R
        t0 = time.time()
        best, runs = fit_R(x, y, d, BUDGET)
        elapsed = time.time() - t0
        rel = float(np.sqrt(best)/max(ynrm,1e-8)) if np.isfinite(best) else float("inf")
        n_p = EMLTree(d, in_dim=1).n_params
        out.append({"setting":"R","depth":d,"n_params":n_p,
                    "budget_s":BUDGET,"elapsed_s":elapsed,"runs":runs,
                    "best_relRMSE":rel})
        print(f"rip R   d={d}  P={n_p:3d}  rel={rel:.3e}  runs={runs}  ({elapsed:.1f}s)")
        # CC
        t0 = time.time()
        best, runs = fit_CC(x, y, d, BUDGET)
        elapsed = time.time() - t0
        rel = float(np.sqrt(best)/max(ynrm,1e-8)) if np.isfinite(best) else float("inf")
        n_p = CEMLTree(d, in_dim=1).n_params
        out.append({"setting":"CC","depth":d,"n_params":n_p,
                    "budget_s":BUDGET,"elapsed_s":elapsed,"runs":runs,
                    "best_relRMSE":rel})
        print(f"rip CC  d={d}  P={n_p:3d}  rel={rel:.3e}  runs={runs}  ({elapsed:.1f}s)")
    print(f"total: {time.time()-grand_t0:.1f}s")
    with open("results/results_rip_push.json","w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
