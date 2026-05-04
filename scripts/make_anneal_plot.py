"""Plot λ schedule comparison: best relRMSE per (target, depth, schedule)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _setup  # noqa: F401  (adds project root to sys.path and chdirs)

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("results/results_complex_anneal.json") as f:
    data = json.load(f)

targets   = sorted({r["target"] for r in data})
schedules = ["S0_lambda0", "S1_const1e-3", "S2_anneal0_1e-1", "S3_anneal0_1"]
labels    = ["λ=0", "λ=1e-3 const", "λ: 0→0.1 cos", "λ: 0→1.0 cos"]
colors    = ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"]
markers   = ["x", "o", "^", "s"]

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
axes = list(axes.flat)
for ax, t in zip(axes, targets):
    for s, lbl, c, m in zip(schedules, labels, colors, markers):
        rows = [r for r in data if r["target"] == t and r["schedule"] == s]
        rows.sort(key=lambda r: r["depth"])
        ax.semilogy([r["depth"] for r in rows],
                    [r["best_relRMSE"] for r in rows],
                    color=c, marker=m, label=lbl, lw=1.5, ms=7)
    ax.set_title(t)
    ax.set_xlabel("depth")
    ax.set_xticks([3, 4, 5])
    ax.grid(True, which="both", alpha=0.3)
    ax.axhline(0.01, color="k", ls="--", lw=0.8, alpha=0.5)

# Hide unused
for ax in axes[len(targets):]: ax.axis("off")

handles = [plt.Line2D([], [], color=c, marker=m, lw=1.5, ms=7, label=l)
           for c, m, l in zip(colors, markers, labels)]
fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.99, 0.05),
           fontsize=10, ncol=1)
fig.suptitle("CEMLTree λ_im schedule comparison (strategy B, 25 s/cell)")
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig("figures/fits_anneal.png", dpi=130, bbox_inches="tight")
print("wrote fits_anneal.png")
