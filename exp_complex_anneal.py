"""Test λ_im annealing schedule for CEMLTree.

Hypothesis: starting with weak (or zero) Im-penalty lets the optimizer use
imaginary degrees of freedom to construct oscillation, then annealing to
strong penalty cleans up the final tree's Im output. This should:
  - retain CC's oscillatory advantage
  - reduce CC's penalty on non-oscillatory targets

Compare three schedules at strategy B, 25 s/cell:
  S0  : λ = 0 (no penalty, baseline)
  S1  : λ = 1e-3 constant (current default)
  S2  : λ: 0 -> 1e-1 cosine ramp (the suggested anneal)
  S3  : λ: 0 -> 1.0  cosine ramp (aggressive)
"""
import json, time
import torch, numpy as np
from eml_tree_complex import CEMLTree, fit_complex, lbfgs_refine_complex

torch.set_default_dtype(torch.float64)

TARGETS = {
    "rip":       (lambda x: torch.sin(3*x)*torch.exp(-x**2/2),     (-3.0, 3.0)),
    "osc":       (lambda x: torch.sin(3*x),                        (-3.14, 3.14)),
    "sin":       (lambda x: torch.sin(x),                          (-3.14, 3.14)),
    "x2":        (lambda x: x**2,                                  (-2.0, 2.0)),
    "exp_decay": (lambda x: torch.exp(-x),                         (0.0, 3.0)),
}
DEPTHS = [3, 4, 5]
BUDGET_S = 25.0

SCHEDULES = {
    "S0_lambda0":   dict(lambda_im=0.0,  lambda_im_end=None),
    "S1_const1e-3": dict(lambda_im=1e-3, lambda_im_end=None),
    "S2_anneal0_1e-1": dict(lambda_im=0.0, lambda_im_end=1e-1, anneal="cosine"),
    "S3_anneal0_1":    dict(lambda_im=0.0, lambda_im_end=1.0,  anneal="cosine"),
}


def fit_strategy_B(target_fn, x, y, depth, t_budget, sched_kwargs):
    t0 = time.time()
    best, runs = float("inf"), 0
    best_model = None
    s = 0
    while time.time() - t0 < t_budget * 0.7:
        torch.manual_seed(s); s += 1
        m = CEMLTree(depth, in_dim=1)
        L = fit_complex(m, x, y, steps=1500, lr=5e-3, **sched_kwargs)
        if L < best:
            best, best_model = L, m
        runs += 1
    if best_model is not None and np.isfinite(best):
        # LBFGS refine with same final lambda
        final_lam = sched_kwargs.get("lambda_im_end") or sched_kwargs["lambda_im"]
        L_ref = lbfgs_refine_complex(best_model, x, y, max_iter=200,
                                      lambda_im=final_lam)
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
            for sname, kw in SCHEDULES.items():
                t0 = time.time()
                try:
                    best, runs, model = fit_strategy_B(fn, x, y, d, BUDGET_S, kw)
                except Exception as e:
                    best, runs, model = float("inf"), -1, None
                    print(f"  ERROR {name}/{sname}/d={d}: {e}")
                elapsed = time.time() - t0
                rel = (float(np.sqrt(best) / max(ynrm, 1e-8))
                       if np.isfinite(best) else float("inf"))
                im_mag = im_magnitude(model, x) if model is not None else None
                row = {
                    "target": name, "schedule": sname, "depth": d,
                    "n_params": CEMLTree(d, in_dim=1).n_params,
                    "best_relRMSE": rel, "n_runs": runs,
                    "elapsed_s": elapsed, "im_mag": im_mag,
                }
                out.append(row)
                im_str = f"{im_mag:.2e}" if im_mag is not None else "n/a"
                print(f"{name:10s} {sname:18s} d={d}  rel={rel:.3e}  "
                      f"runs={runs:3d}  im={im_str}  ({elapsed:.1f}s)")
    print(f"total: {time.time()-grand_t0:.1f}s")
    with open("results_complex_anneal.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
