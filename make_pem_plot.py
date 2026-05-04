"""Plot PEM vs EM across targets and depths."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("results_parametric.json") as f:
    data = json.load(f)

targets = sorted({r["target"] for r in data})
fig, axes = plt.subplots(3, 4, figsize=(15, 9), sharex=True)
axes = list(axes.flat)
for ax, t in zip(axes, targets):
    for setting, color, marker, label in [
        ("PEM", "#d62728", "o", "PEM (per-node 6-param)"),
        ("EM",  "#1f77b4", "s", "EM (affine-glue, headline)")]:
        rows = [r for r in data if r["target"] == t and r["model"] == setting]
        rows.sort(key=lambda r: r["depth"])
        ax.semilogy([r["depth"] for r in rows],
                    [r["best_relRMSE"] for r in rows],
                    color=color, marker=marker, label=label, lw=1.6, ms=6)
    ax.set_title(t)
    ax.set_xticks([2, 3, 4, 5])
    ax.grid(True, which="both", alpha=0.3)
    ax.axhline(0.01, color="k", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("depth")

for ax in axes[len(targets):]: ax.axis("off")

handles = [plt.Line2D([], [], color="#d62728", marker="o", lw=1.6, ms=6,
                       label="PEM (per-node 6-param)"),
           plt.Line2D([], [], color="#1f77b4", marker="s", lw=1.6, ms=6,
                       label="EM (affine-glue, headline)")]
fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.99, 0.05),
           fontsize=10, ncol=1)
fig.suptitle("Parametric per-node EML vs affine-glue EML "
             "(strategy B, 25 s/cell)")
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig("fits_pem.png", dpi=130, bbox_inches="tight")
print("wrote fits_pem.png")
