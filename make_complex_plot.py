"""Plot R vs CR vs CC across targets and depths."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("results_complex_params.json") as f:
    data = json.load(f)

targets = sorted({r["target"] for r in data})
settings = ["R_realsoft_realparam", "CR_realparam", "CC_complexparam"]
labels   = ["R: real_softplus, real params",
            "CR: complex_real, real params",
            "CC: complex_real, complex params"]
colors   = ["#1f77b4", "#ff7f0e", "#d62728"]
markers  = ["o", "s", "^"]

fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=False)
axes = axes.flat
for ax, t in zip(axes, targets):
    for s, lbl, c, m in zip(settings, labels, colors, markers):
        rows = [r for r in data if r["target"] == t and r["setting"] == s]
        rows.sort(key=lambda r: r["depth"])
        ax.semilogy([r["depth"] for r in rows],
                    [r["best_relRMSE"] for r in rows],
                    color=c, marker=m, label=lbl, lw=1.6, ms=7)
    ax.set_title(t)
    ax.set_xlabel("depth")
    ax.set_xticks([3, 4, 5])
    ax.grid(True, which="both", alpha=0.3)
    ax.axhline(0.01, color="k", ls="--", lw=0.8, alpha=0.5)

# Hide unused axes
for ax in list(axes)[len(targets):]:
    ax.axis("off")

# Single legend
handles = [plt.Line2D([], [], color=c, marker=m, lw=1.6, ms=7, label=l)
           for c, m, l in zip(colors, markers, labels)]
fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.99, 0.05),
           fontsize=9)
fig.suptitle("Backend × parameterisation comparison (best relRMSE, strategy B, 25 s/cell)")
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig("fits_complex_params.png", dpi=130, bbox_inches="tight")
print("wrote fits_complex_params.png")
