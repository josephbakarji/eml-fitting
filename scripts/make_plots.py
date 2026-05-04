"""Visualize EMLTree fits for representative 1D targets and write PNGs."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _setup  # noqa: F401  (adds project root to sys.path and chdirs)

import torch, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.eml_tree import EMLTree, fit_multistart

torch.set_default_dtype(torch.float64)

TARGETS = {
    "sin (smooth)":          (lambda x: torch.sin(x),                          (-3.14, 3.14)),
    "x² (poly)":             (lambda x: x**2,                                  (-2.0, 2.0)),
    "|x| (kink)":            (lambda x: torch.abs(x),                          (-2.0, 2.0)),
    "exp(-x²) (gauss)":      (lambda x: torch.exp(-x**2),                      (-2.5, 2.5)),
    "tanh(2x)":              (lambda x: torch.tanh(2*x),                       (-2.0, 2.0)),
    "sin(3x)·exp(-x²/2)":    (lambda x: torch.sin(3*x)*torch.exp(-x**2/2),     (-3.0, 3.0)),
}


def main():
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (name, (fn, (a,b))) in zip(axes.flat, TARGETS.items()):
        x = torch.linspace(a, b, 300)
        y_true = fn(x)
        ynrm = float((y_true**2).mean().sqrt())
        m, best, _ = fit_multistart(fn, x, depth=4, n_restarts=12, steps=3000)
        y_pred = m(x).detach()
        rel = float(np.sqrt(best) / max(ynrm, 1e-8))
        ax.plot(x.numpy(), y_true.numpy(), 'k-', lw=2, label='target')
        ax.plot(x.numpy(), y_pred.numpy(), 'r--', lw=1.5,
                label=f'eml d=4 (relRMSE={rel:.2e})')
        ax.set_title(name); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.suptitle("EMLTree depth=4 fits (best of 12 restarts)")
    plt.tight_layout()
    plt.savefig("figures/fits_1d.png", dpi=130, bbox_inches='tight')
    print("wrote fits_1d.png")


if __name__ == "__main__":
    main()
