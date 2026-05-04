"""Diagnose the optimization landscape for EMLTree fitting.

Questions:
- What fraction of random inits get within 1% relRMSE?
- Are there pathological seeds where loss explodes / NaNs?
- Does multi-restart ~converge to a plateau?
- What's the 'best of N' as N grows?
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _setup  # noqa: F401  (adds project root to sys.path and chdirs)

import json, time
import torch, numpy as np
from src.eml_tree import EMLTree, fit

torch.set_default_dtype(torch.float64)

TARGETS = {
    "sin":   (lambda x: torch.sin(x),                       (-3.14, 3.14)),
    "gauss": (lambda x: torch.exp(-x**2),                   (-2.5, 2.5)),
    "rip":   (lambda x: torch.sin(3*x)*torch.exp(-x**2/2),  (-3.0, 3.0)),
    "x2":    (lambda x: x**2,                               (-2.0, 2.0)),
}
DEPTHS = [2, 3, 4]
N_SEEDS = 50
STEPS = 2000
LR = 5e-3


def main():
    out = []
    t0 = time.time()
    for name, (fn, (a, b)) in TARGETS.items():
        x = torch.linspace(a, b, 200)
        y = fn(x)
        ynrm = float((y**2).mean().sqrt())
        for d in DEPTHS:
            losses = []
            t1 = time.time()
            for s in range(N_SEEDS):
                torch.manual_seed(s)
                m = EMLTree(d, in_dim=1)
                L = fit(m, x, y, steps=STEPS, lr=LR)
                losses.append(L)
            losses = np.array(losses)
            rel = np.sqrt(losses) / max(ynrm, 1e-8)
            row = {
                "target": name, "depth": d, "n_seeds": N_SEEDS,
                "n_finite": int(np.isfinite(losses).sum()),
                "frac_lt_5pct":  float((rel < 0.05).mean()),
                "frac_lt_1pct":  float((rel < 0.01).mean()),
                "best_relRMSE":  float(rel.min()),
                "median_relRMSE": float(np.median(rel)),
                "p90_relRMSE":   float(np.percentile(rel, 90)),
                "best_of_5":     float(np.sqrt(np.sort(losses)[:5].min()) / max(ynrm,1e-8)),
                "best_of_20":    float(np.sqrt(np.sort(losses)[:20].min()) / max(ynrm,1e-8)),
                "elapsed":       time.time() - t1,
            }
            out.append(row)
            print(f"{name:6s} d={d}  best={row['best_relRMSE']:.2e}  "
                  f"med={row['median_relRMSE']:.2e}  "
                  f"<5%={row['frac_lt_5pct']:.0%}  <1%={row['frac_lt_1pct']:.0%}")
    print(f"total: {time.time()-t0:.1f}s")
    with open("results/results_landscape.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
