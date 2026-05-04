"""Apply 3 snap modes to depth-4 fits on canonical 1D targets.

Reports:
  loss_pre, loss_after_pure, loss_after_coef, loss_after_greedy,
  n_slots_total, n_slots_snapped_greedy, expression_post_greedy.
"""
import json, time
import torch, numpy as np
from eml_tree import EMLTree, fit_multistart
from snap import (snap_pure_symbol, snap_coef, snap_greedy,
                  decode_expression, _slot_keys)

torch.set_default_dtype(torch.float64)

TARGETS = {
    "sin":       (lambda x: torch.sin(x),                          (-3.14, 3.14)),
    "x2":        (lambda x: x**2,                                  (-2.0, 2.0)),
    "abs":       (lambda x: torch.abs(x),                          (-2.0, 2.0)),
    "gauss":     (lambda x: torch.exp(-x**2),                      (-2.5, 2.5)),
    "tanh":      (lambda x: torch.tanh(2*x),                       (-2.0, 2.0)),
    "exp_decay": (lambda x: torch.exp(-x),                         (0.0, 3.0)),
    "rip":       (lambda x: torch.sin(3*x)*torch.exp(-x**2/2),     (-3.0, 3.0)),
}

DEPTH = 4
N_RESTARTS = 12
STEPS = 3000


def main():
    out = []
    t0 = time.time()
    for name, (fn, (a, b)) in TARGETS.items():
        x = torch.linspace(a, b, 200)
        y = fn(x)
        ynrm = float((y**2).mean().sqrt())
        # Fit
        model, best, _ = fit_multistart(fn, x, depth=DEPTH, n_restarts=N_RESTARTS,
                                         steps=STEPS, lr=5e-3)
        rel_pre = float(np.sqrt(best) / max(ynrm, 1e-8))
        n_slots = len(_slot_keys(model))
        # Pure symbolic
        _, post_pure, _, _ = snap_pure_symbol(model, x, y)
        rel_pure = float(np.sqrt(post_pure) / max(ynrm, 1e-8))
        # Coefficient (a,b binary; c continuous)
        _, post_coef, _, _ = snap_coef(model, x, y, refit_steps=500)
        rel_coef = float(np.sqrt(post_coef) / max(ynrm, 1e-8))
        # Greedy
        _, post_greedy, n_snap, snapped_model, snapped_keys = snap_greedy(
            model, x, y, tol_rel=0.20, refit_steps=600
        )
        rel_greedy = float(np.sqrt(post_greedy) / max(ynrm, 1e-8))
        expr = decode_expression(snapped_model, snapped_keys)
        # Truncate giant expressions for log
        expr_short = expr if len(expr) < 400 else expr[:380] + "...]"
        row = {
            "target": name,
            "rel_pre":    rel_pre,
            "rel_pure":   rel_pure,
            "rel_coef":   rel_coef,
            "rel_greedy": rel_greedy,
            "n_slots":    n_slots,
            "n_snapped":  n_snap,
            "frac_snapped": n_snap / n_slots,
            "expression": expr,
        }
        out.append(row)
        print(f"{name:10s}  pre={rel_pre:.2e} pure={rel_pure:.2e} "
              f"coef={rel_coef:.2e} greedy={rel_greedy:.2e}  "
              f"snap={n_snap}/{n_slots}")
        print(f"           expr: {expr_short}")
    print(f"elapsed: {time.time()-t0:.1f}s")
    with open("results_snap.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
