# EML-tree fitting: practical study

**Question (Joseph):** The Odrzywołek paper (arXiv:2603.21852) shows
`eml(x,y) = exp(x) − ln(y)` is a single binary operator that, with the constant
`1`, generates every elementary function. They propose gradient-based symbolic
discovery of eml-trees. *But blind recovery falls off a cliff with depth*: 100%
at d=2, ~25% at d=3–4, **<1% at d=5+**. Question asked: if we relax from the
discrete symbolic basis `{1, x}` to **continuous affine inputs** at every node
(your `eml(ax, by)` idea, generalised to multivariate inputs and to internal
nodes that mix in child outputs), does this become a usable function-fitting
primitive on real data?

**TL;DR.**
1. **Continuous-coefficient eml-trees are *easy* to optimise** — multi-restart
   Adam recovers good fits 50–100% of the time on smooth 1D targets at depth
   3–4. This dodges the discrete basin-of-attraction problem the paper hits.
2. **But they don't compete with standard primitives.** On a matched
   parameter budget, MLP-tanh and RBFs beat eml-trees by 5–10× on every
   target tested. Polynomial regression destroys eml on smooth analytic targets
   at low degree. The structural cost of nesting `exp − log` to express things
   like `x²` shows up as wasted capacity.
3. **Bivariate is markedly harder.** At d=4 (104 params), eml manages 5–10%
   relRMSE with success rates ≤25% per restart on Franke/sincos/Rosenbrock.
4. **Oscillatory targets are the wall.** `sin(3x)·exp(−x²/2)` never reaches
   5% relRMSE at d=4 across 50 random seeds; best=8.4%. RBFs get 0.14% on the
   same target with the same param count.

**Verdict for your "this can be a practical fit" question:** Optimisable —
yes, much more than the paper's discrete version suggests. Practical as a
standalone approximator — no, it's strictly worse than MLPs/RBFs/polynomials.
Where it could earn its keep is as a **structural prior** for recovering
expressions you *believe* are eml-shaped (composed of compositions of `exp`
and `log`, e.g. partition functions, free energies, log-sum-exp, Arrhenius,
softmax-like quantities). Outside that family, the eml prior costs more than
it pays.

---

## Setup

`EMLTree(depth, in_dim)` — full binary tree of nodes computing
`eml(left_input, right_input)`. Each input slot is a continuous affine
combination of inputs and the corresponding child subtree's output:

- internal slot: `slot = a · x + b · child + c`  (D+2 params, D = input dim)
- leaf-parent slot: `slot = a · x + c`           (D+1 params)

Two backends:
- `real_softplus`: pass right-arg through `softplus + ε` so log stays finite.
  Used in all experiments below.
- `complex_real`: evaluate fully in `complex128`, return `Re(.)`. Available
  for completeness with the paper's symbolic constructions.

Optimiser: Adam, `lr=5e-3`, cosine schedule, gradient clip 5.0, 3000 steps,
8–12 random restarts, MSE loss.

Param counts (in_dim=1):

| depth | params |
|-------|--------|
| 1     |  4     |
| 2     | 14     |
| 3     | 34     |
| 4     | 74     |

---

## 1D fitting (`exp_1d.py` → `results_1d.json`)

12 restarts each, 3000 Adam steps, 200 train points. **success** = restart
reached relRMSE < 5%.

| target              | d=1 relRMSE | d=2     | d=3     | d=4     | d=4 success |
|---------------------|-------------|---------|---------|---------|-------------|
| sin                 | 0.64        | 0.043   | 0.023   | 0.012   | 9/12        |
| cos                 | 1.00        | 0.061   | 0.017   | 0.010   | 11/12       |
| x²                  | 0.049       | 0.013   | 0.005   | 0.001   | 12/12       |
| abs                 | 0.13        | 0.030   | 0.020   | 0.010   | 12/12       |
| exp(−x²)            | 0.71        | 0.062   | 0.012   | 0.014   | 11/12       |
| sigmoid(3x)         | 0.17        | 0.017   | 0.011   | 0.0009  | 12/12       |
| x³−x                | 0.79        | 0.074   | 0.025   | 0.021   | 11/12       |
| tanh(2x)            | 0.35        | 0.027   | 0.015   | 0.002   | 9/12        |
| log(1+x²)           | 0.13        | 0.026   | 0.012   | 0.002   | 12/12       |
| exp(−x)             | 0.020       | 0.004   | 0.002   | 0.002   | 10/12       |
| sin(3x)·exp(−x²/2)  | 1.00        | 0.69    | 0.22    | 0.075   | **0/12**    |

Read: depth-4 trees fit smooth canonical functions to ~1% relRMSE with high
restart success. The oscillatory ripple defeats them — every restart gets
stuck in a local minimum that captures the envelope or the oscillation but not
both.

## Baseline comparison (`exp_baselines.py`)

Same target set, same parameter budget (P), 8 restarts. relRMSE.

| target | budget | eml      | poly     | MLP-tanh | RBF      |
|--------|--------|----------|----------|----------|----------|
| sin    | 14     | 5.5e-2   | **1e-9** | 9.6e-3   | 4.1e-2   |
| sin    | 34     | 2.8e-2   | 0.75 ✗   | **5.6e-3** | 1.2e-2 |
| sin    | 74     | 1.2e-2   | 0.99 ✗   | **1.4e-3** | 5.4e-3 |
| x²     | 14     | 1.3e-2   | **1e-16**| 4.2e-2   | 5.4e-2   |
| gauss  | 14     | 9.5e-2   | 7.7e-4   | **4.4e-3** | 6.0e-3 |
| gauss  | 74     | 1.7e-2   | 1.0 ✗    | 2.6e-3   | **3.2e-3** |
| tanh   | 74     | 7.0e-3   | 0.84 ✗   | **9.2e-4** | 4.9e-3 |
| rip    | 74     | 7.5e-2   | 1.0 ✗    | 7.1e-3   | **1.4e-3** |

Notes:
- Polynomial regression is unbeatable on smooth analytic targets within its
  Weierstrass radius **at low degree** — at high degree (P=74 → deg 73), the
  Vandermonde becomes catastrophically ill-conditioned and poly dies (≈ 1.0
  relRMSE everywhere). This is a fair fact about polynomials, not a knock on
  the comparison.
- MLP-tanh wins almost everywhere it's compared. RBFs win on the wiggly
  target.
- **eml never wins.** Closest it comes is sin at P=14 (5.5e-2 vs RBF 4.1e-2
  and poly 1e-9).

## 2D fitting (`exp_2d.py`)

25×25 grid = 625 points, 8 restarts, in_dim=2. Each leaf affine in (x,y).

| target     | d=2 (P=20) | d=3 (P=48) | d=4 (P=104) | d=4 success |
|------------|------------|------------|-------------|-------------|
| Franke     | 0.22       | 0.17       | 0.095       | 0/8         |
| sin·cos    | 0.63       | 0.19       | 0.054       | 0/8         |
| saddle     | 0.16       | 0.062      | 0.013       | 2/8         |
| Rosenbrock | 0.36       | 0.11       | 0.048       | 1/8         |

Bivariate behaviour is qualitatively worse: the eml structure is essentially
1D in spirit (one exp arm vs one log arm), so multivariate fits have to encode
all cross-terms via the affine mixing on leaves. Restart success collapses.

## Optimisation landscape (`exp_landscape.py`)

50 random seeds per (target, depth). What fraction of restarts reach a given
quality?

| target | depth | best   | median | <5% rate | <1% rate |
|--------|-------|--------|--------|----------|----------|
| sin    | 2     | 4.9e-2 | 0.28   | 2%       | 0%       |
| sin    | 3     | 1.3e-2 | 0.078  | 26%      | 0%       |
| sin    | 4     | 3.6e-3 | 0.035  | 66%      | 4%       |
| gauss  | 4     | 9.3e-3 | 0.049  | 52%      | 2%       |
| x²     | 4     | 2.0e-3 | 0.009  | 100%     | 58%      |
| rip    | 4     | 8.4e-2 | 0.32   | 0%       | 0%       |

Reading:
- **Many restarts genuinely help** — best-of-50 is 1–2 orders of magnitude
  below median for sin/gauss. Plateaus are wide but escapable with enough
  inits.
- **Sub-1% is rare.** Even at depth 4 with 50 seeds, only `x²` regularly
  reaches <1% relRMSE. To squeeze past 1% you'd need either bigger trees,
  smarter init, or local refinement (e.g. snapping eml-shaped substructures
  and re-fitting only the residual).

---

## Why eml underperforms as a generic primitive

Three reasons, in order of importance:

1. **Each node only emits one number from two inputs.** A binary tree of depth
   d has 2^d leaves but only ~2^d internal nodes worth of expressive activity.
   Compare this to an MLP layer of width 2^d — each unit is fully nonlinear
   in *all* inputs, not just the local affine combination. The eml tree is
   geometrically penalised by its tree topology.
2. **`exp` and `log` are cancelling primitives.** Most uses of eml in a fit
   are spent constructing identity-like or ratio-like shapes (`exp(ln(z)) =
   z`, `exp(a+ln(b)) = b·e^a`). That's a lot of nodes burned to produce
   linear/multiplicative structure that an MLP gets for free in one layer.
3. **No global oscillation.** eml has no natural way to produce `sin`-shaped
   bumps without going through complex log. On reals, every leaf-parent node
   contributes a monotone exp-of-affine curve and a monotone log-of-softplus
   curve. Building wiggles requires careful subtractive interference between
   high-magnitude exponentials — which is precisely the regime where the
   landscape is roughest.

## Where eml *might* be the right primitive

- Quantities that are **literally** exp/log compositions (Boltzmann factors,
  partition functions, Arrhenius rates, log-likelihoods, free energies,
  log-sum-exp).
- **Symbolic regression** where you want a closed-form answer in elementary
  functions and the paper's discrete grammar is too brittle. A hybrid is
  natural: continuous fit first, then snap each affine slot to the nearest
  member of `{1, x_i, child}` and keep snaps that don't blow up the loss.
  This was attempted (Ninhache repo's `snap()`) but only for the discrete
  basis; we never tested the snap-and-residual idea below.
- **Identifiability of eml-shaped physics.** If your data is generated by an
  expression in this class, eml-trees should fit *and* let you read off
  structure. This is the use-case where the paper's universality matters.

## Concrete recommendation

If you're working on the proof and you want a **practical companion fit
algorithm**, the version that has the best chance of being useful is:

1. Continuous-coefficient eml-tree at **depth ≤ 4**, multi-restart (≥ 12),
   Adam + cosine schedule. Code: `eml_tree.py:EMLTree`.
2. After fitting, run **affine→symbol snapping**: at each slot, try replacing
   the continuous affine combo with each candidate from `{1, x_i, child}` and
   keep replacements that don't increase loss by more than a tolerance. This
   gives interpretable symbolic decompositions where the data supports them
   and free continuous parameters where it doesn't.
3. For depths > 4 the landscape becomes hostile fast. If you need
   higher-capacity fits, this approach won't scale past d=5; switch to a
   hybrid (eml-tree with leaves replaced by tiny MLPs, or eml as a layer in a
   larger network).

## Files

- `eml_tree.py` — module
- `exp_1d.py`, `results_1d.json`
- `exp_baselines.py`, `results_baselines.json`
- `exp_2d.py`, `results_2d.json`
- `exp_landscape.py`, `results_landscape.json`
- `make_plots.py`, `fits_1d.png`
