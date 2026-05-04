# Weaknesses and threats to the headline claim

A self-audit of the experimental setup. Listed in rough order of impact on the
"practical universality at ≤1% relRMSE" claim.

## A. Architectural restrictions that make it not-quite the paper's grammar

### A1. Real-only backend used for all results so far
- **What**: All 7 experiments used `real_softplus`. The complex-mediated trig
  constructions in the paper (`sin`, `cos`, `π`, `i`) are foreclosed.
- **Impact on the claim**: We're testing universality of a real-restricted
  grammar, not the operator from the paper. The headline figure is empirically
  honest, but the grammar is strictly weaker than the one the universality
  theorem covers.
- **Status**: Being tested now (`exp_complex.py`).

### A2. Softplus distortion of the right argument
- **What**: `real_softplus` computes `eml(x, y) ≈ exp(x) - log(softplus(y) + ε)`,
  not `exp(x) - log(y)`. For `y > 0`, `softplus(y) ≈ y` only in the limit; near
  zero it deviates by `log 2 ≈ 0.693`. So even in the real-positive regime, the
  operator is a smooth surrogate, not the paper's operator.
- **Impact**: Trees fit with this backend are not strictly `eml`-trees. The
  snap-to-symbol results inherit the distortion: a slot snapped to "child"
  passes through softplus, so the post-snap expression is not literally the
  paper's grammar.
- **Mitigation**: `complex_real` does not soften the right arg. We should
  state results with both backends in the paper.

### A3. Affine slots vs. discrete grammar
- **What**: The paper's grammar is `S → 1 | x | eml(S,S)` — three discrete
  vertices per slot. Our slot is `a·x + b·child + c` with continuous `(a,b,c)`,
  which contains the paper's grammar as a measure-zero subset and is **strictly
  more expressive at every depth**.
- **Impact**: Depth-vs-error comparisons against the paper's depth claims are
  not apples-to-apples. Our depth-3 tree has 34 free parameters; the paper's
  depth-3 has 0 free parameters (just structural choice over a 3^14-sized
  space). Saying "depth 3 reaches 1%" understates the cost; the paper's depth
  3 reaches 0% (exact) on a small set of recoverable functions and ∞ on the
  rest.

### A4. Bias term `c` trivialises constant construction
- **What**: With `c` continuous, any constant — including `π`, `e` — is one
  parameter away from any slot. The paper requires `eml`-trees to construct
  `π` to leaf-count ≥ 53.
- **Impact**: Significant fraction of the apparent capacity per depth is
  spent on free constants that the paper's grammar would have to construct
  symbolically. Greedy snap reveals this — slots that snap to "1" or "0"
  without harming loss were probably degenerate.

## B. Optimisation-side weaknesses

### B1. No held-out / extrapolation test
- **What**: Train and "test" share the same 200-point grid. We report
  in-sample relRMSE.
- **Impact**: Continuous parameters can find numerical-artefact solutions
  that interpolate the grid but oscillate wildly between samples. None of
  our claims prove that the recovered tree is a global function approximator
  — only that it matches values at the grid.
- **Fix**: Re-evaluate fitted trees on a held-out grid (uniform and
  random); report extrapolation relRMSE on a domain extension.

### B2. Wall-clock fairness across strategies
- **What**: Strategy comparison gave each method 30 s. LBFGS does fewer but
  more expensive iterations than Adam; "30 s" is not equal compute.
- **Impact**: Strategy B's win is partly because LBFGS converts a fixed
  wall-clock window into more progress per iteration. A fair compute
  comparison would also report iteration counts and FLOP estimates.

### B3. Init scheme is ad-hoc
- **What**: `init_scale=0.5` Gaussian. No principled scaling for tree depth.
- **Impact**: For deeper trees, accumulating affine combos saturate the
  exp-clamp at init, biasing where Adam can move. Smarter init (e.g.\ scale
  inversely with depth, or warm-start from depth-1 closed forms) likely
  unlocks more.

### B4. Gradient pathologies in the operator
- `exp(x)` clamped at `x ∈ [-60, 60]`: hard wall, gradient zero outside.
- `log(softplus(y) + ε)`: gradient `1/(softplus(y) + ε)` blows up near
  `y → -∞`.
- In `complex_real`: branch cut of `log` at `arg = π` causes gradient
  discontinuity when slot output crosses the negative real axis. Adam
  steps near the cut effectively bounce.
- **Impact**: Optimiser is biased toward "safe interior" regions where the
  gradient is well-conditioned. Some basins on the cut are unreachable.

### B5. Curriculum strategy may be implemented wrong
- **What**: `warm_init_from(small, big_depth)` grafts the small tree into a
  level-offset position; the rest of the big tree is random. The random
  surroundings can blow up the small tree's signal before refinement settles.
- **Impact**: Strategy C underperformed and we attributed this to "curriculum
  doesn't help"; equally consistent with "the graft scheme is broken." A
  cleaner test: replicate the small tree at the new root with all surrounding
  slots zeroed (so they are no-ops).

## C. Statistical / claims-level weaknesses

### C1. Single-seed wall-clock tables
- **What**: Strategy and universality tables report best-of-runs at each
  cell, but not variance across full repeats of the experiment.
- **Impact**: A "best 1.9e-3" might be 1e-4 to 5e-3 across full reruns. We
  don't know.
- **Fix**: Repeat the universality sweep 3× and report mean ± std.

### C2. Loss-landscape claims are hand-wavy
- **What**: Section 5.2 talks about "wide basins" and "rough landscape" based
  on success rates. We never plotted a loss surface or measured a basin
  volume.
- **Impact**: Cosmetic. The data supports "many restarts help"; it does not
  support specific topology claims.

### C3. No theoretical lower bound on depth-vs-error
- **What**: We have empirical curves but no theorem on minimum depth needed
  to express each target to ε accuracy in `eml`-trees.
- **Impact**: We can't say "depth 3 is enough for `x²`"; we only know
  "depth 3 was enough at 25 s of LBFGS-refined Adam."

### C4. No comparison to the paper's discrete-grammar version
- **What**: We never re-ran `EMLMasterFormula` (Ninhache repo) on our targets
  to confirm baseline behavior in our environment.
- **Impact**: The "we beat the paper's <1% recovery rate" claim is grounded
  in their reported numbers, not a controlled head-to-head.
- **Fix**: One-shot benchmark of `EMLMasterFormula` under the same Adam
  settings as our affine model on a few targets. (This gets us A vs B for
  the discrete-vs-continuous parameterisation question.)

## D. Generalisation / scope weaknesses

### D1. Bivariate is bad and the paper says nothing about it
- **What**: At depth 4 (104 params), 5–10% relRMSE; success ≤25%/restart.
- **Impact**: Cross-terms `xy` cannot appear at the leaf — must be deep in
  the tree. This is a structural cost the paper does not address; their
  universality theorem is for one-variable functions.

### D2. No noise robustness / data-efficiency study
- **What**: All targets noiseless, 200 dense samples per 1D target.
- **Impact**: Real-world data is sparse and noisy. We have no evidence that
  the optimisation strategy is stable under either condition. Continuous
  parameters give the optimiser room to overfit noise; symbolic snapping
  may help, but we haven't tested.

### D3. The "rip wall" could be many things
- 2% plateau on `sin(3x)·exp(-x²/2)` may be:
  - Fundamental: real-restricted `eml`-trees cannot express oscillation×envelope
    on ℝ to better than 2% at depth ≤6.
  - Optimisation: the basin exists but is too narrow for our restart budget.
  - Backend: `real_softplus` distorts in the regions where the function has
    its zero crossings (where `y` flips sign, `softplus(y)` flattens).
- **Impact**: The single most cited limitation of the approach is uncharacterised.
  We need (A1) results to disambiguate.

## E. Interpretability claims that cut both ways

### E1. Greedy snap "improving" loss is partially over-parameterisation
- **What**: Greedy snap with refit reduces loss because the snap acts as a
  regulariser, freeing slots that were stuck at degenerate continuous values.
- **Impact**: The snap-improves-loss result is real but its interpretation
  needs care: it does not prove the underlying function is "almost symbolic
  in eml". It shows that 30 slots is more than enough capacity for these
  targets.

### E2. The decoded expressions are still huge
- **What**: Even after snapping 13/30 slots on `gauss`, the rendered expression
  string is 400+ characters with deeply nested mixed-affine inner slots.
- **Impact**: "Interpretable" is generous. A more honest claim: snapping
  recovers a *symbolic skeleton* but the leaves and coefficients remain
  numerical. Not a closed-form readout.

---

## Priority for follow-ups

1. **Run `complex_real` on rip and trig targets** (in progress) — directly
   addresses A1, A2, D3.
2. **Held-out test set evaluation** (B1) — single afternoon, materially
   strengthens or weakens the headline.
3. **3× repeat of universality sweep** (C1) — adds error bars to the figure.
4. **Run `EMLMasterFormula` on our targets** (C4) — closes the
   discrete-vs-continuous question with a paired control.
5. **Try complex parameters** — biggest leap in expressivity, likely needs
   complex Adam variants or treating `(real, imag)` as twin real params.
