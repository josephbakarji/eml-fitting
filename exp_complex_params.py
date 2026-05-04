"""Test CEMLTree (complex parameters) on oscillatory + sanity targets.

Compare three settings at the same wall-clock budget:
  R   - real_softplus, real params (the headline)
  CR  - complex_real, real params (Section 5.6)
  CC  - CEMLTree, complex params (this experiment)

We hold strategy = B (Adam multi-restart + LBFGS refine), 25s/cell.

Also: report the imag-part magnitude of the trained CEMLTree, since loss
penalises only Re(.). If Im is huge, that's a regularization signal.
"""
import json, time
import torch, numpy as np
from eml_tree import EMLTree, fit
from eml_tree_complex import CEMLTree, fit_complex, lbfgs_refine_complex

torch.set_default_dtype(torch.float64)

TARGETS = {
    "rip":       (lambda x: torch.sin(3*x)*torch.exp(-x**2/2),     (-3.0, 3.0)),
    "osc":       (lambda x: torch.sin(3*x),                        (-3.14, 3.14)),
    "cos2":      (lambda x: torch.cos(2*x),                        (-3.14, 3.14)),
    "sin":       (lambda x: torch.sin(x),                          (-3.14, 3.14)),
    "cos":       (lambda x: torch.cos(x),                          (-3.14, 3.14)),
    "x2":        (lambda x: x**2,                                  (-2.0, 2.0)),
    "exp_decay": (lambda x: torch.exp(-x),                         (0.0, 3.0)),
}
DEPTHS = [3, 4, 5]
BUDGET_S = 25.0
LAMBDA_IM = 1e-3   # small penalty on Im(T(x)) to prevent runaway


def real_strategy_B(target_fn, x, y, depth, t_budget, backend):
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
        if np.isfinite(L_ref):
            best = min(best, L_ref)
    return best, runs, best_model


def complex_strategy_B(target_fn, x, y, depth, t_budget):
    t0 = time.time()
    best, runs = float("inf"), 0
    best_model = None
    s = 0
    while time.time() - t0 < t_budget * 0.7:
        torch.manual_seed(s); s += 1
        m = CEMLTree(depth, in_dim=1)
        L = fit_complex(m, x, y, steps=1500, lr=5e-3, lambda_im=LAMBDA_IM)
        if L < best:
            best, best_model = L, m
        runs += 1
    if best_model is not None and np.isfinite(best):
        L_ref = lbfgs_refine_complex(best_model, x, y, max_iter=200,
                                      lambda_im=LAMBDA_IM)
        if np.isfinite(L_ref):
            best = min(best, L_ref)
    return best, runs, best_model


def im_magnitude(model: CEMLTree, x: torch.Tensor) -> float:
    with torch.no_grad():
        z = model.forward_complex(x)
        return float(z.imag.abs().mean())


def main():
    out = []
    grand_t0 = time.time()
    for name, (fn, (a, b)) in TARGETS.items():
        x = torch.linspace(a, b, 200)
        y = fn(x)
        ynrm = float((y**2).mean().sqrt())
        for d in DEPTHS:
            # R: real_softplus, real params
            t0 = time.time()
            best, runs, _ = real_strategy_B(fn, x, y, d, BUDGET_S, "real_softplus")
            elapsed = time.time() - t0
            rel = float(np.sqrt(best)/max(ynrm, 1e-8)) if np.isfinite(best) else float("inf")
            n_p = EMLTree(d, in_dim=1, backend="real_softplus").n_params
            out.append({"target": name, "setting": "R_realsoft_realparam",
                        "depth": d, "n_params": n_p,
                        "best_relRMSE": rel, "n_runs": runs,
                        "elapsed_s": elapsed, "im_mag": None})
            print(f"{name:10s} R   d={d}  P={n_p:3d}  rel={rel:.3e}  runs={runs:3d}  ({elapsed:.1f}s)")

            # CR: complex_real, real params
            t0 = time.time()
            best, runs, _ = real_strategy_B(fn, x, y, d, BUDGET_S, "complex_real")
            elapsed = time.time() - t0
            rel = float(np.sqrt(best)/max(ynrm, 1e-8)) if np.isfinite(best) else float("inf")
            n_p = EMLTree(d, in_dim=1, backend="complex_real").n_params
            out.append({"target": name, "setting": "CR_realparam",
                        "depth": d, "n_params": n_p,
                        "best_relRMSE": rel, "n_runs": runs,
                        "elapsed_s": elapsed, "im_mag": None})
            print(f"{name:10s} CR  d={d}  P={n_p:3d}  rel={rel:.3e}  runs={runs:3d}  ({elapsed:.1f}s)")

            # CC: complex params
            t0 = time.time()
            best, runs, model = complex_strategy_B(fn, x, y, d, BUDGET_S)
            elapsed = time.time() - t0
            rel = float(np.sqrt(best)/max(ynrm, 1e-8)) if np.isfinite(best) else float("inf")
            n_p = CEMLTree(d, in_dim=1).n_params
            im_mag = im_magnitude(model, x) if model is not None else None
            out.append({"target": name, "setting": "CC_complexparam",
                        "depth": d, "n_params": n_p,
                        "best_relRMSE": rel, "n_runs": runs,
                        "elapsed_s": elapsed, "im_mag": im_mag})
            print(f"{name:10s} CC  d={d}  P={n_p:3d}  rel={rel:.3e}  runs={runs:3d}  "
                  f"im_mag={im_mag:.2e}  ({elapsed:.1f}s)")
    print(f"total: {time.time()-grand_t0:.1f}s")
    with open("results_complex_params.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
