"""Headline plot for the NeurIPS paper §4: relRMSE vs depth, PEM real and complex."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _setup  # noqa: F401

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("results/results_universality_pem.json") as f:
    data = json.load(f)

# Display names matching the paper
DISPLAY = {
    "x3_minus_x":     r"$x^3 - x$",
    "tanh_2x":        r"$\tanh(2x)$",
    "sin_x":          r"$\sin(x)$",
    "exp_neg_x2":     r"$e^{-x^2}$",
    "sin3x_envelope": r"$\sin(3x)\,e^{-x^2/2}$",
}
COLORS = {
    "x3_minus_x":     "#1f77b4",
    "tanh_2x":        "#2ca02c",
    "sin_x":          "#d62728",
    "exp_neg_x2":     "#ff7f0e",
    "sin3x_envelope": "#9467bd",
}

fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.0))

for tkey, label in DISPLAY.items():
    color = COLORS[tkey]
    real = sorted([r for r in data if r["target"]==tkey and r["param"]=="real"],
                  key=lambda r: r["depth"])
    cplx = sorted([r for r in data if r["target"]==tkey and r["param"]=="complex"],
                  key=lambda r: r["depth"])
    if real:
        ax.semilogy([r["depth"] for r in real],
                    [r["best_relRMSE"] for r in real],
                    color=color, marker="o", linestyle="-", lw=1.6, ms=6,
                    label=f"{label} (real)")
    if cplx:
        ax.semilogy([r["depth"] for r in cplx],
                    [r["best_relRMSE"] for r in cplx],
                    color=color, marker="s", linestyle="--", lw=1.4, ms=6,
                    markerfacecolor="white",
                    label=f"{label} (complex)")

ax.axhline(0.01, color="k", ls=":", lw=0.8, alpha=0.6)
ax.text(2.95, 0.011, "1% relRMSE", fontsize=8, va="bottom", color="0.3")

ax.set_xlabel("PEM tree depth $d$")
ax.set_ylabel("best relative RMSE (log scale)")
ax.set_xticks([3, 4, 5])
ax.grid(True, which="both", alpha=0.25)
ax.legend(loc="upper right", fontsize=8.5, ncol=2, framealpha=0.95)
ax.set_title("Practical universality of PEM trees: best relRMSE vs depth\n"
             "Adam multi-restart + LBFGS, 30 s/cell, single CPU",
             fontsize=10)

plt.tight_layout()
out = "paper/NeurIPS_2026___EML/Figures/pem_universality.pdf"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, bbox_inches="tight")
plt.savefig("figures/pem_universality.png", dpi=150, bbox_inches="tight")
print(f"wrote {out} and figures/pem_universality.png")
