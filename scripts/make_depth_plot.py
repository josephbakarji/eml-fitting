"""Plot relRMSE vs depth for each target — the universality figure."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _setup  # noqa: F401  (adds project root to sys.path and chdirs)

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("results/results_universality.json") as f:
    data = json.load(f)

# Group by target
by_target: dict[str, list[tuple[int, float, int]]] = {}
for r in data:
    by_target.setdefault(r["target"], []).append(
        (r["depth"], r["best_relRMSE"], r["n_params"])
    )

fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
markers = ["o", "s", "^", "D", "v", "p", "h", "*", "X", "P", ">"]
for (name, pts), m in zip(sorted(by_target.items()), markers):
    pts.sort()
    depths = [p[0] for p in pts]
    rels   = [p[1] for p in pts]
    ax.semilogy(depths, rels, marker=m, label=name, lw=1.4, ms=6)

ax.set_xlabel("tree depth $d$")
ax.set_ylabel("best relRMSE (log scale)")
ax.set_title("EMLTree practical universality:\n"
             "best-of-runs relRMSE vs depth at fixed wall-clock budget\n"
             "(strategy B = Adam multi-restart + LBFGS refinement, ~25 s/target)")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="upper right", fontsize=8, ncol=2)
ax.set_xticks([2, 3, 4, 5, 6])
ax.axhline(0.01, color="k", ls="--", lw=0.8, alpha=0.5)
ax.text(2, 0.011, "1% relRMSE", fontsize=8, va="bottom")

plt.tight_layout()
plt.savefig("figures/fits_depth.png", dpi=130, bbox_inches="tight")
print("wrote fits_depth.png")
