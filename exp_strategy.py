"""Compare optimisation strategies at fixed wall-clock budget per target.

Strategies:
  A) random-restart Adam (baseline; what we've been doing)
  B) Adam followed by LBFGS refinement of best-of-N
  C) curriculum: fit depth 2 -> warm-start depth 3 -> warm-start depth 4
  D) symbolic warm-start: init slots near {1, x, child} then Adam-refine

Budget: 30 seconds per (target, strategy).
"""
import json, time
import torch, numpy as np
import torch.nn as nn
from eml_tree import EMLTree, fit
from snap import _slot_keys, _candidate_vertices

torch.set_default_dtype(torch.float64)

TARGETS = {
    "sin":   (lambda x: torch.sin(x),                       (-3.14, 3.14)),
    "x2":    (lambda x: x**2,                               (-2.0, 2.0)),
    "gauss": (lambda x: torch.exp(-x**2),                   (-2.5, 2.5)),
    "tanh":  (lambda x: torch.tanh(2*x),                    (-2.0, 2.0)),
    "rip":   (lambda x: torch.sin(3*x)*torch.exp(-x**2/2),  (-3.0, 3.0)),
}
TARGET_DEPTH = 4
BUDGET_S = 30.0


def fit_one(model, x, y, steps, lr=5e-3):
    return fit(model, x, y, steps=steps, lr=lr)


def lbfgs_refine(model, x, y, max_iter=200):
    opt = torch.optim.LBFGS(model.parameters(), max_iter=max_iter,
                            line_search_fn="strong_wolfe", history_size=20)
    def closure():
        opt.zero_grad()
        pred = model(x)
        loss = ((pred - y)**2).mean()
        if not torch.isfinite(loss):
            loss = torch.tensor(1e10, dtype=loss.dtype)
        loss.backward()
        return loss
    try:
        opt.step(closure)
    except Exception:
        pass
    with torch.no_grad():
        return float(((model(x) - y)**2).mean())


def warm_init_from(small_model: EMLTree, big_depth: int) -> EMLTree:
    """Initialise depth=big_depth tree from a smaller fitted model.

    We graft the small model into the LEFT subtree of root and zero-bias
    everything else. Affine slots in unused parts left at default random.
    """
    big = EMLTree(big_depth, in_dim=small_model.in_dim)
    # Map small.l{lvl}p{pos} -> big.l{lvl + (big_depth - small.depth)}p{pos}
    offset = big_depth - small_model.depth
    with torch.no_grad():
        for k, p in small_model.params.items():
            # k like "l0p0L"
            lvl = int(k[1:].split("p")[0])
            posside = k.split("p")[1]
            new_lvl = lvl + offset
            new_key = f"l{new_lvl}p{posside}"
            if new_key in big.params and big.params[new_key].shape == p.shape:
                big.params[new_key].copy_(p.detach())
    return big


def symbolic_warm_init(depth: int, in_dim: int = 1, p_symbol: float = 0.5,
                        seed: int = 0) -> EMLTree:
    """Init each slot at a random symbolic vertex with prob p_symbol,
    else random affine."""
    torch.manual_seed(seed)
    m = EMLTree(depth, in_dim=in_dim, init_scale=0.3)
    with torch.no_grad():
        for key in _slot_keys(m):
            if torch.rand(1).item() < p_symbol:
                cands = _candidate_vertices(m, key)
                idx = int(torch.randint(0, len(cands), (1,)).item())
                m.params[key].copy_(cands[idx][1])
    return m


def strategy_A(target_fn, x, y, depth, t_budget):
    """Pure random-restart Adam."""
    t0 = time.time()
    best, runs = float("inf"), 0
    s = 0
    while time.time() - t0 < t_budget:
        torch.manual_seed(s); s += 1
        m = EMLTree(depth, in_dim=1)
        # Cheap fit per restart
        L = fit_one(m, x, y, steps=2000, lr=5e-3)
        best = min(best, L); runs += 1
    return best, runs


def strategy_B(target_fn, x, y, depth, t_budget):
    """Adam + LBFGS refinement of best."""
    t0 = time.time()
    best, runs = float("inf"), 0
    best_model = None
    s = 0
    while time.time() - t0 < t_budget * 0.7:
        torch.manual_seed(s); s += 1
        m = EMLTree(depth, in_dim=1)
        L = fit_one(m, x, y, steps=1500, lr=5e-3)
        if L < best:
            best, best_model = L, m
        runs += 1
    if best_model is not None:
        L_ref = lbfgs_refine(best_model, x, y, max_iter=200)
        best = min(best, L_ref)
    return best, runs


def strategy_C(target_fn, x, y, depth, t_budget):
    """Curriculum: depth 2 -> 3 -> 4. Each tier warm-starts the next."""
    t0 = time.time()
    per_tier = t_budget / max(1, depth - 1)
    best_at_tier = None
    runs = 0
    cur_depth = 2
    cur_best_model, cur_best = None, float("inf")
    s = 0
    while time.time() - t0 < per_tier:
        torch.manual_seed(s); s += 1
        m = EMLTree(cur_depth, in_dim=1)
        L = fit_one(m, x, y, steps=1500, lr=5e-3)
        if L < cur_best: cur_best, cur_best_model = L, m
        runs += 1
    while cur_depth < depth:
        cur_depth += 1
        # Warm-start depth = cur_depth from cur_best_model
        tier_t0 = time.time()
        warmed = warm_init_from(cur_best_model, cur_depth)
        L = fit_one(warmed, x, y, steps=1500, lr=5e-3)
        cur_best, cur_best_model = L, warmed
        # Spend remaining tier time on a few fresh restarts to escape
        s2 = 0
        while time.time() - tier_t0 < per_tier:
            torch.manual_seed(1000 + s2); s2 += 1
            m = EMLTree(cur_depth, in_dim=1)
            L = fit_one(m, x, y, steps=1500, lr=5e-3)
            if L < cur_best: cur_best, cur_best_model = L, m
            runs += 1
    return cur_best, runs


def strategy_D(target_fn, x, y, depth, t_budget):
    """Symbolic warm-start: init slots near vertices, then Adam-refine."""
    t0 = time.time()
    best, runs = float("inf"), 0
    s = 0
    while time.time() - t0 < t_budget:
        m = symbolic_warm_init(depth, in_dim=1, p_symbol=0.5, seed=s); s += 1
        L = fit_one(m, x, y, steps=2000, lr=5e-3)
        if L < best: best = L
        runs += 1
    return best, runs


STRATEGIES = {
    "A_restart": strategy_A,
    "B_lbfgs":   strategy_B,
    "C_curric":  strategy_C,
    "D_symwarm": strategy_D,
}


def main():
    out = []
    grand_t0 = time.time()
    for name, (fn, (a, b)) in TARGETS.items():
        x = torch.linspace(a, b, 200)
        y = fn(x)
        ynrm = float((y**2).mean().sqrt())
        for sname, sfn in STRATEGIES.items():
            t0 = time.time()
            try:
                best, runs = sfn(fn, x, y, TARGET_DEPTH, BUDGET_S)
            except Exception as e:
                best, runs = float("inf"), -1
                print(f"  ERROR {name}/{sname}: {e}")
            elapsed = time.time() - t0
            rel = float(np.sqrt(best) / max(ynrm, 1e-8)) if np.isfinite(best) else float("inf")
            row = {
                "target": name, "strategy": sname, "depth": TARGET_DEPTH,
                "budget_s": BUDGET_S, "elapsed_s": elapsed, "n_runs": runs,
                "best_loss": best, "best_relRMSE": rel,
            }
            out.append(row)
            print(f"{name:6s} {sname:10s} runs={runs:3d}  rel={rel:.3e}  "
                  f"({elapsed:.1f}s)")
    print(f"total: {time.time()-grand_t0:.1f}s")
    with open("results_strategy.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
