"""Snap-to-symbol post-training for continuous-affine EMLTree.

Three modes:
  pure_symbol  : every slot snapped to nearest of {1, x_i, child} (one-hot)
  coef_snap    : a, b ∈ {0, 1}; c kept continuous (preserve symbolic struct)
  greedy       : try snapping each slot in isolation; keep snaps with
                 Δloss < tol; refit remaining continuous slots.

Returns: (loss_pre, loss_post, n_slots_snapped, snapped_model, expression_string)
"""
from __future__ import annotations
import copy
import torch
import torch.nn as nn
from eml_tree import EMLTree, fit


def _slot_keys(model: EMLTree):
    keys = []
    for level in range(model.depth):
        for pos in range(2**level):
            for side in ("L", "R"):
                keys.append(f"l{level}p{pos}{side}")
    return keys


def _slot_is_leaf(key: str, depth: int) -> bool:
    level = int(key[1:].split("p")[0])
    return level == depth - 1


def _slot_dim(model: EMLTree, key: str) -> int:
    return model.params[key].shape[0]


def _eval_loss(model: EMLTree, x, y) -> float:
    with torch.no_grad():
        p = model(x)
        return float(((p - y) ** 2).mean())


def _candidate_vertices(model: EMLTree, key: str) -> list[torch.Tensor]:
    """Symbolic-vertex candidates for a slot.

    Returns list of param-vectors corresponding to each symbol:
      - constant `1`     -> a=0, (b=0), c=1
      - input   `x_i`    -> a_i=1, (b=0), c=0
      - child            -> a=0, b=1, c=0   (only if not leaf-parent)
    """
    D = model.in_dim
    is_leaf = _slot_is_leaf(key, model.depth)
    dim = _slot_dim(model, key)
    out: list[tuple[str, torch.Tensor]] = []
    # constant 1
    v = torch.zeros(dim, dtype=model.params[key].dtype)
    v[-1] = 1.0
    out.append(("1", v))
    # each input
    for i in range(D):
        v = torch.zeros(dim, dtype=model.params[key].dtype)
        v[i] = 1.0
        out.append((f"x{i}" if D > 1 else "x", v.clone()))
    # child
    if not is_leaf:
        v = torch.zeros(dim, dtype=model.params[key].dtype)
        v[D] = 1.0
        out.append(("c", v.clone()))
    return out


def snap_pure_symbol(model: EMLTree, x, y) -> tuple[float, float, int, EMLTree]:
    """Snap every slot to nearest symbolic vertex (no refit)."""
    pre = _eval_loss(model, x, y)
    out = copy.deepcopy(model)
    snapped = 0
    for key in _slot_keys(out):
        cands = _candidate_vertices(out, key)
        with torch.no_grad():
            cur = out.params[key].detach().clone()
            best_v, best_d = None, float("inf")
            for _, v in cands:
                d = (cur - v).pow(2).sum().item()
                if d < best_d:
                    best_d, best_v = d, v
            out.params[key].copy_(best_v)
            snapped += 1
    post = _eval_loss(out, x, y)
    return pre, post, snapped, out


def snap_coef(model: EMLTree, x, y, refit_steps: int = 500, lr: float = 1e-2):
    """Snap a, b to {0,1}; keep c continuous; refit c-only."""
    pre = _eval_loss(model, x, y)
    out = copy.deepcopy(model)
    D = out.in_dim
    snapped = 0
    for key in _slot_keys(out):
        is_leaf = _slot_is_leaf(key, out.depth)
        with torch.no_grad():
            v = out.params[key].detach().clone()
            for i in range(D):
                v[i] = 1.0 if v[i].abs() > 0.5 else 0.0
            if not is_leaf:
                v[D] = 1.0 if v[D].abs() > 0.5 else 0.0
            out.params[key].copy_(v)
            snapped += 1
    # Refit only the bias slot c per slot via gradient
    # Mark non-c as no-grad by zeroing their grads after backward
    opt = torch.optim.Adam(out.parameters(), lr=lr)
    keys = _slot_keys(out)
    for it in range(refit_steps):
        opt.zero_grad()
        loss = ((out(x) - y) ** 2).mean()
        if not torch.isfinite(loss):
            break
        loss.backward()
        # zero grads on a,b — keep only the last (c) entry
        with torch.no_grad():
            for k in keys:
                g = out.params[k].grad
                if g is None: continue
                D_ = out.in_dim
                is_leaf = _slot_is_leaf(k, out.depth)
                cut = D_ + (0 if is_leaf else 1)
                g[:cut] = 0.0
        opt.step()
    post = _eval_loss(out, x, y)
    return pre, post, snapped, out


def snap_greedy(model: EMLTree, x, y, tol_rel: float = 0.10,
                refit_steps: int = 500, lr: float = 5e-3):
    """For each slot, try snapping to nearest vertex; accept if loss within tol_rel.

    After all attempts, refit remaining unsnapped slots.
    Returns (pre, post, n_snapped, model, snapped_keys).
    """
    pre = _eval_loss(model, x, y)
    out = copy.deepcopy(model)
    snapped_keys = set()
    keys = _slot_keys(out)
    # Try slots in random-ish order: shallow first to lock structure
    keys_sorted = sorted(keys, key=lambda k: int(k[1:].split("p")[0]))
    cur_loss = pre
    for key in keys_sorted:
        cands = _candidate_vertices(out, key)
        original = out.params[key].detach().clone()
        best = None
        for sym, v in cands:
            with torch.no_grad():
                out.params[key].copy_(v)
                trial = _eval_loss(out, x, y)
            if best is None or trial < best[1]:
                best = (sym, trial, v.clone())
        sym, trial_loss, v = best
        # Accept snap if loss doesn't grow more than tol_rel above best-so-far
        if trial_loss <= cur_loss * (1.0 + tol_rel) + 1e-12:
            with torch.no_grad():
                out.params[key].copy_(v)
            snapped_keys.add(key)
            cur_loss = trial_loss
        else:
            with torch.no_grad():
                out.params[key].copy_(original)
    # Refit non-snapped slots
    opt = torch.optim.Adam(
        [p for k, p in out.params.items() if k not in snapped_keys],
        lr=lr,
    )
    if any(True for k in keys if k not in snapped_keys):
        for it in range(refit_steps):
            opt.zero_grad()
            loss = ((out(x) - y) ** 2).mean()
            if not torch.isfinite(loss):
                break
            loss.backward()
            opt.step()
    post = _eval_loss(out, x, y)
    return pre, post, len(snapped_keys), out, snapped_keys


# -------- expression decoding --------

def decode_expression(model: EMLTree, snapped_keys: set | None = None) -> str:
    """Render a (possibly partially) snapped tree as a string.

    Snapped slots render as their symbol. Unsnapped slots render as
    affine combos a*x + b*child + c.
    """
    return _expr(model, 0, 0, snapped_keys)


def _slot_str(model: EMLTree, key: str, child_expr: str | None,
              snapped: set | None) -> str:
    p = model.params[key].detach().tolist()
    D = model.in_dim
    is_leaf = _slot_is_leaf(key, model.depth)
    if snapped is not None and key in snapped:
        # Decode as symbol
        # Find which vertex it matches
        cands = _candidate_vertices(model, key)
        v = model.params[key].detach()
        for sym, vv in cands:
            if torch.allclose(v, vv, atol=1e-6):
                if sym == "c": return child_expr or "f"
                return sym
        return "?"
    # Unsnapped: write as affine combo
    parts = []
    for i in range(D):
        a = p[i]
        if abs(a) < 1e-6: continue
        var = "x" if D == 1 else f"x{i}"
        parts.append(f"{a:.3g}*{var}")
    if not is_leaf:
        b = p[D]
        if abs(b) > 1e-6:
            parts.append(f"{b:.3g}*({child_expr})")
    c = p[-1]
    if abs(c) > 1e-6 or not parts:
        parts.append(f"{c:.3g}")
    return "(" + " + ".join(parts) + ")"


def _expr(model: EMLTree, level: int, pos: int, snapped: set | None) -> str:
    key = f"l{level}p{pos}"
    is_leaf = _slot_is_leaf(key + "L", model.depth)
    if is_leaf:
        l = _slot_str(model, key + "L", None, snapped)
        r = _slot_str(model, key + "R", None, snapped)
    else:
        lc = _expr(model, level + 1, 2 * pos, snapped)
        rc = _expr(model, level + 1, 2 * pos + 1, snapped)
        l = _slot_str(model, key + "L", lc, snapped)
        r = _slot_str(model, key + "R", rc, snapped)
    return f"eml({l}, {r})"
