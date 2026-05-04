"""Greedy snap-to-symbol for ParametricEMLTree.

Per-parameter discrete candidate sets:
    a_i ∈ {-1, 0, 1}   (paper's eml has a=1; 0 disables exp arm; -1 negates)
    b_i ∈ {0, 1}       (scalar interior; per-component for leaf-parent vector)
    c_i ∈ {0}          (paper has c=0)
    d_i ∈ {-1, 0, 1}   (paper's eml has d=-1; 0 disables log arm; 1 flips sign)
    e_i ∈ {0, 1}       (scalar interior; per-component for leaf-parent vector)
    f_i ∈ {0, 1}       (paper has f=0; 1 enables softplus(x+1) shape)

Greedy strategy: shallow-first per parameter, accept snap if loss within
tol_rel of current. After all attempts, refit unsnapped params by Adam.
"""
from __future__ import annotations
import copy
import torch
from src.eml_tree_parametric import ParametricEMLTree, fit_parametric


CANDIDATES = {
    "a": [-1.0, 0.0, 1.0],
    "b": [0.0, 1.0],
    "c": [0.0],
    "d": [-1.0, 0.0, 1.0],
    "e": [0.0, 1.0],
    "f": [0.0, 1.0],
}


def _eval_loss(model, x, y) -> float:
    with torch.no_grad():
        return float(((model(x) - y) ** 2).mean())


def _all_param_keys(model: ParametricEMLTree):
    """Yield (slot_key, param_letter, component_index_or_None)."""
    for level in range(model.depth):
        for pos in range(2 ** level):
            base = f"l{level}p{pos}"
            for letter in "abcdef":
                key = f"{base}_{letter}"
                p = model.params[key]
                if p.dim() == 0 or (p.dim() == 1 and p.numel() == 1):
                    yield key, letter, None
                else:
                    for i in range(p.numel()):
                        yield key, letter, i


def snap_greedy_parametric(model: ParametricEMLTree, x, y,
                            tol_rel: float = 0.20,
                            refit_steps: int = 600, lr: float = 5e-3):
    """Greedy slot-by-slot snap. Returns (loss_pre, loss_post, n_snap, model, snapped_set)."""
    pre = _eval_loss(model, x, y)
    out = copy.deepcopy(model)
    snapped: set[tuple[str, int | None]] = set()

    keys_ordered = sorted(
        _all_param_keys(out),
        key=lambda kli: int(kli[0][1:].split("p")[0])
    )
    cur_loss = pre
    for key, letter, comp in keys_ordered:
        cands = CANDIDATES[letter]
        p = out.params[key]
        original = p.detach().clone()
        # Try each candidate
        best = None
        for cv in cands:
            with torch.no_grad():
                if comp is None:
                    p.fill_(cv)
                else:
                    p.view(-1)[comp] = cv
                trial = _eval_loss(out, x, y)
            if best is None or trial < best[1]:
                best = (cv, trial)
        cv, trial_loss = best
        if trial_loss <= cur_loss * (1.0 + tol_rel) + 1e-12:
            with torch.no_grad():
                if comp is None:
                    p.fill_(cv)
                else:
                    p.view(-1)[comp] = cv
            snapped.add((key, comp))
            cur_loss = trial_loss
        else:
            with torch.no_grad():
                p.copy_(original)

    # Refit unsnapped: zero gradients on snapped components after backward
    if any((k, c) not in snapped for (k, _, c) in _all_param_keys(out)):
        opt = torch.optim.Adam(out.parameters(), lr=lr)
        for it in range(refit_steps):
            opt.zero_grad()
            loss = ((out(x) - y) ** 2).mean()
            if not torch.isfinite(loss):
                break
            loss.backward()
            with torch.no_grad():
                for key, letter, comp in _all_param_keys(out):
                    if (key, comp) in snapped:
                        g = out.params[key].grad
                        if g is None: continue
                        if comp is None:
                            g.zero_()
                        else:
                            g.view(-1)[comp] = 0.0
            opt.step()

    post = _eval_loss(out, x, y)
    total = sum(1 for _ in _all_param_keys(out))
    return pre, post, len(snapped), out, snapped, total


def decode_pem_node(model: ParametricEMLTree, key_base: str,
                     snapped: set, lc_str: str, rc_str: str) -> str:
    """Render one PEM node as a string."""
    p = lambda lt: model.params[f"{key_base}_{lt}"].detach()
    a, b, c = p("a").item(), p("b"), p("c").item()
    d, e, f = p("d").item(), p("e"), p("f").item()
    bs = b.tolist() if b.numel() > 1 else [b.item()]
    es = e.tolist() if e.numel() > 1 else [e.item()]
    bs_str = "+".join(f"{bv:.3g}*x{i}" for i, bv in enumerate(bs)
                      if abs(bv) > 1e-8) if len(bs) > 1 else f"{bs[0]:.3g}*x"
    es_str = "+".join(f"{ev:.3g}*x{i}" for i, ev in enumerate(es)
                      if abs(ev) > 1e-8) if len(es) > 1 else f"{es[0]:.3g}*x"
    arg_e = f"{bs_str}+{c:.3g}" if abs(c) > 1e-8 else bs_str
    arg_l = f"{es_str}+{f:.3g}" if abs(f) > 1e-8 else es_str
    if not arg_e: arg_e = "0"
    if not arg_l: arg_l = "0"
    # Substitute child-input strings (only for interior nodes; leaf-parent uses x)
    arg_e_full = arg_e.replace("x", lc_str)
    arg_l_full = arg_l.replace("x", rc_str)
    return f"({a:.3g}*exp({arg_e_full}) + {d:.3g}*ln({arg_l_full}))"


def decode_pem(model: ParametricEMLTree, snapped: set | None = None) -> str:
    return _decode(model, 0, 0, snapped or set())


def _decode(model, level, pos, snapped):
    key = f"l{level}p{pos}"
    is_leaf = (level == model.depth - 1)
    if is_leaf:
        return decode_pem_node(model, key, snapped, "x", "x")
    lc = _decode(model, level + 1, 2*pos, snapped)
    rc = _decode(model, level + 1, 2*pos + 1, snapped)
    return decode_pem_node(model, key, snapped, lc, rc)
