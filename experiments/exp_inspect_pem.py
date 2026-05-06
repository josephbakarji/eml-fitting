"""Inspect what fitted PEM trees actually look like.

For each target, refit a depth-3 PEM tree (7 atoms, 42 params), then:
  1. Print the full parameter table (per-atom (a, b, c, d, e, f)).
  2. Render the tree as a nested EML expression with numerical coefficients.
  3. Try greedy snap of each parameter to nearest of {-2, -1, 0, 1, 2}; accept
     a snap if it doesn't worsen loss by more than tol_rel; refit unsnapped.
     Report fraction snapped and final expression.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _setup  # noqa: F401

import json, time, copy
import torch, numpy as np
from src.eml_tree_parametric import (
    ParametricEMLTree, fit_parametric, lbfgs_refine_parametric)

torch.set_default_dtype(torch.float64)


TARGETS = {
    "x3_minus_x":    (lambda x: x**3 - x,                          (-1.5, 1.5)),
    "tanh_2x":       (lambda x: torch.tanh(2 * x),                 (-2.0, 2.0)),
    "sin_x":         (lambda x: torch.sin(x),                      (-3.14, 3.14)),
    "exp_neg_x2":    (lambda x: torch.exp(-x**2),                  (-2.5, 2.5)),
    "sin3x_envelope":(lambda x: torch.sin(3*x)*torch.exp(-x**2/2), (-3.0, 3.0)),
}
DEPTH = 3
N_RESTARTS = 24       # spend more on getting a good fit since we'll inspect it
N_STEPS    = 3000
N_SAMPLES  = 200
SNAP_CANDIDATES = [-2.0, -1.0, 0.0, 1.0, 2.0]
SNAP_TOL_REL = 0.30   # relative loss tolerance per snap


def best_fit(fn, x, y, depth, n_restarts, n_steps):
    best, best_model = float("inf"), None
    for s in range(n_restarts):
        torch.manual_seed(s)
        m = ParametricEMLTree(depth, in_dim=1)
        L = fit_parametric(m, x, y, steps=n_steps, lr=5e-3)
        if L < best:
            best, best_model = L, m
    if best_model is not None:
        L_ref = lbfgs_refine_parametric(best_model, x, y, max_iter=300)
        if np.isfinite(L_ref) and L_ref < best:
            best = L_ref
    return best, best_model


def all_param_handles(model: ParametricEMLTree):
    """Yield (atom_key, letter, comp_idx_or_None, tensor) for every scalar."""
    for level in range(model.depth):
        for pos in range(2 ** level):
            base = f"l{level}p{pos}"
            for letter in "abcdef":
                key = f"{base}_{letter}"
                p = model.params[key]
                if p.dim() == 0 or (p.dim() == 1 and p.numel() == 1):
                    yield base, letter, None, p
                else:
                    for i in range(p.numel()):
                        yield base, letter, i, p


def parameter_table(model: ParametricEMLTree) -> str:
    """Print one row per atom: (a, b, c, d, e, f)."""
    lines = []
    lines.append(f"{'atom':<10s}  {'a':>9s} {'b':>9s} {'c':>9s} {'d':>9s} {'e':>9s} {'f':>9s}")
    for level in range(model.depth):
        for pos in range(2 ** level):
            base = f"l{level}p{pos}"
            vals = []
            for letter in "abcdef":
                p = model.params[f"{base}_{letter}"]
                # Show first scalar if vector
                v = p.item() if p.dim() == 0 or p.numel() == 1 else p[0].item()
                vals.append(v)
            row = f"{base:<10s}  " + " ".join(f"{v:>9.3f}" for v in vals)
            lines.append(row)
    return "\n".join(lines)


def render_atom(model, base: str, lc_str: str, rc_str: str, decimals: int = 3) -> str:
    p = lambda lt: model.params[f"{base}_{lt}"]
    a = p("a").item(); c = p("c").item()
    d = p("d").item(); f = p("f").item()
    b_t = p("b"); e_t = p("e")
    b = b_t.item() if b_t.dim() == 0 or b_t.numel() == 1 else b_t[0].item()
    e = e_t.item() if e_t.dim() == 0 or e_t.numel() == 1 else e_t[0].item()
    fmt = f"{{:.{decimals}g}}"
    # exp arm: a*exp(b*lc + c)
    if abs(a) < 1e-9:
        exp_arm = "0"
    else:
        if abs(b) < 1e-9:
            exp_arm = f"{fmt.format(a*np.exp(c))}"
        else:
            inner_e = f"{fmt.format(b)}*({lc_str})"
            if abs(c) > 1e-9: inner_e += f" + {fmt.format(c)}"
            exp_arm = (f"{fmt.format(a)}*exp({inner_e})" if abs(a-1) > 1e-6
                       else f"exp({inner_e})") if abs(a+1) > 1e-6 else f"-exp({inner_e})"
    # log arm: d*ln(e*rc + f) [softplus stabilised — note in display]
    if abs(d) < 1e-9:
        log_arm = "0"
    else:
        if abs(e) < 1e-9:
            log_arm = f"{fmt.format(d*np.log(np.maximum(np.log(1+np.exp(f)), 1e-6)))}"
        else:
            inner_l = f"{fmt.format(e)}*({rc_str})"
            if abs(f) > 1e-9: inner_l += f" + {fmt.format(f)}"
            log_arm = f"{fmt.format(d)}*ln+({inner_l})"
    if exp_arm == "0" and log_arm == "0":
        return "0"
    if exp_arm == "0":
        return log_arm
    if log_arm == "0":
        return exp_arm
    return f"({exp_arm} + {log_arm})"


def render_tree(model: ParametricEMLTree, level=0, pos=0) -> str:
    base = f"l{level}p{pos}"
    is_leaf = (level == model.depth - 1)
    if is_leaf:
        return render_atom(model, base, "x", "x")
    lc = render_tree(model, level+1, 2*pos)
    rc = render_tree(model, level+1, 2*pos+1)
    return render_atom(model, base, lc, rc)


def loss(model, x, y):
    with torch.no_grad():
        return float(((model(x) - y) ** 2).mean())


def greedy_snap_to_set(model, x, y, candidates, tol_rel,
                       refit_steps=600, refit_lr=5e-3):
    """Per-parameter greedy snap to nearest member of `candidates`."""
    pre = loss(model, x, y)
    out = copy.deepcopy(model)
    handles = list(all_param_handles(out))
    snapped = set()
    cur = pre
    for base, letter, comp, p in handles:
        original_val = p.detach().clone()
        # find closest candidate
        if comp is None:
            v = p.item()
        else:
            v = p.view(-1)[comp].item()
        best = None
        for cv in candidates:
            with torch.no_grad():
                if comp is None: p.fill_(cv)
                else: p.view(-1)[comp] = cv
                trial = loss(out, x, y)
            if best is None or trial < best[1]:
                best = (cv, trial)
        cv, trial_loss = best
        if trial_loss <= cur * (1.0 + tol_rel) + 1e-12:
            with torch.no_grad():
                if comp is None: p.fill_(cv)
                else: p.view(-1)[comp] = cv
            snapped.add((base, letter, comp))
            cur = trial_loss
        else:
            with torch.no_grad():
                p.copy_(original_val)
    # Refit unsnapped
    if any((b, l, c) not in snapped for (b, l, c, _) in handles):
        opt = torch.optim.Adam(out.parameters(), lr=refit_lr)
        for it in range(refit_steps):
            opt.zero_grad()
            L = ((out(x) - y) ** 2).mean()
            if not torch.isfinite(L): break
            L.backward()
            with torch.no_grad():
                for base, letter, comp, p in handles:
                    if (base, letter, comp) in snapped:
                        if p.grad is None: continue
                        if comp is None: p.grad.zero_()
                        else: p.grad.view(-1)[comp] = 0.0
            opt.step()
    post = loss(out, x, y)
    return pre, post, snapped, out, len(handles)


def main():
    out_records = []
    for name, (fn, (a, b)) in TARGETS.items():
        x = torch.linspace(a, b, N_SAMPLES)
        y = fn(x)
        ynrm = float((y**2).mean().sqrt())
        print(f"\n{'='*78}\nTARGET: {name}\n{'='*78}")
        t0 = time.time()
        L, model = best_fit(fn, x, y, DEPTH, N_RESTARTS, N_STEPS)
        rel_pre = float(np.sqrt(L) / max(ynrm, 1e-8))
        print(f"\n[fit]  depth={DEPTH}, restarts={N_RESTARTS}, steps={N_STEPS}, "
              f"elapsed={time.time()-t0:.1f}s")
        print(f"       MSE = {L:.4e}, relRMSE = {rel_pre:.3e}")
        print(f"\n[parameter table]")
        print(parameter_table(model))
        print(f"\n[expression] (ln+ = log of softplus-stabilised arg)")
        expr = render_tree(model)
        # Pretty: split on top-level '+' to flow lines
        print(f"  T(x) = {expr}")
        # Snap to {-2,-1,0,1,2}
        print(f"\n[greedy snap] candidates {SNAP_CANDIDATES}, tol_rel={SNAP_TOL_REL}")
        pre, post, snapped, snap_model, total = greedy_snap_to_set(
            model, x, y, SNAP_CANDIDATES, SNAP_TOL_REL)
        rel_post = float(np.sqrt(post) / max(ynrm, 1e-8))
        print(f"  pre  relRMSE = {rel_pre:.3e}")
        print(f"  post relRMSE = {rel_post:.3e}")
        print(f"  snapped = {len(snapped)}/{total} params ({100*len(snapped)/total:.0f}%)")
        # Show which atoms got fully snapped (all 6 params discrete)
        atoms_snapped = {}
        for (base, letter, comp) in snapped:
            atoms_snapped.setdefault(base, set()).add(letter)
        all_discrete_atoms = [b for b, ls in atoms_snapped.items() if ls >= set("abcdef")]
        print(f"  fully-discrete atoms: {all_discrete_atoms or '(none)'}")
        print(f"\n[snapped expression]")
        snap_expr = render_tree(snap_model)
        print(f"  T_snap(x) = {snap_expr}")
        out_records.append({
            "target": name, "depth": DEPTH,
            "relRMSE_pre": rel_pre,
            "relRMSE_post_snap": rel_post,
            "n_snapped": len(snapped),
            "n_total": total,
            "fully_discrete_atoms": all_discrete_atoms,
            "expression": expr,
            "expression_snapped": snap_expr,
            "param_table": parameter_table(model),
        })
    # Save
    with open("results/results_inspect_pem.json", "w") as f:
        json.dump(out_records, f, indent=2)
    print(f"\nsaved to results/results_inspect_pem.json")


if __name__ == "__main__":
    main()
