# What do the fitted PEM trees actually look like?

Joseph asked: the whole point of EML trees is interpretability — can we
look at what we got, and tell whether the optimizer recovered something
symbolic ("a = 1, d = -1, b = 1, c = 0, ...") or just smooth continuous
curves that happen to have the right shape?

## TL;DR

**Across all five targets, the fitted depth-3 PEM trees are NOT recoveries
of symbolic constructions.** They are continuous-coefficient
approximations that exploit the 6-parameter freedom of each atom to
interpolate smoothly. Even with a generous snap tolerance (allow loss to
grow by 30%) and candidate set $\{-2, -1, 0, 1, 2\}$, **at most 17% of the
42 parameters snap to discrete values, and zero atoms become fully
discrete on any target**. On the two cleanest targets ($\tanh$ and
$e^{-x^2}$), nothing snaps at all.

This is an honest finding worth flagging in the paper.

## Procedure

For each of the five targets in §4 of the NeurIPS paper, we
- Refit a depth-3 PEM (7 atoms, 42 params) with 24 Adam restarts × 3000
  steps + LBFGS refinement (≈ 72 s/target instead of the headline 30 s,
  to make sure the fit is well-converged)
- Print the full parameter table $\theta_i = (a_i, b_i, c_i, d_i, e_i, f_i)$
  per atom $i$
- Greedy-snap each parameter to the nearest of $\{-2, -1, 0, 1, 2\}$,
  accepting a snap if the loss does not grow by more than 30%
- Refit the un-snapped parameters with the snapped ones frozen
- Render the resulting tree as a nested EML expression

The candidate set $\{-2, -1, 0, 1, 2\}$ is generous — it includes both the
"original-EML defaults" $(a, d) = (1, -1)$ and `0` (which kills an atom's
arm).

Code: `experiments/exp_inspect_pem.py`. Raw output: `logs/inspect_pem.log`.
JSON: `results/results_inspect_pem.json`.

## Snap statistics

| Target | relRMSE pre-snap | relRMSE post-snap | params snapped | fully-discrete atoms |
|---|---|---|---|---|
| $x^3 - x$              | $7.3\!\cdot\!10^{-3}$ | $6.9\!\cdot\!10^{-3}$ | **7/42 (17%)** | 0 |
| $\tanh(2x)$            | $3.8\!\cdot\!10^{-3}$ | $3.5\!\cdot\!10^{-3}$ | **0/42 (0%)**  | 0 |
| $\sin(x)$              | $1.9\!\cdot\!10^{-3}$ | $1.7\!\cdot\!10^{-3}$ | **3/42 (7%)**  | 0 |
| $e^{-x^2}$             | $3.9\!\cdot\!10^{-3}$ | $3.2\!\cdot\!10^{-3}$ | **0/42 (0%)**  | 0 |
| $\sin(3x)e^{-x^2/2}$   | $1.18\!\cdot\!10^{-1}$ | $7.7\!\cdot\!10^{-2}$ | **3/42 (7%)**  | 0 |

A few things to notice:

1. **Loss decreases on every target after greedy snap.** This is the same
   "snap-as-regularizer" effect we saw in earlier (affine-glue) snap
   experiments: snapping a near-zero parameter to exactly zero, plus
   refitting the rest, gives a slightly cleaner local minimum.
2. **No atom becomes fully discrete on any target.** Every atom in every
   tree has at least one continuous-valued parameter that resists
   snapping at the 30% tolerance.
3. **The cleanest fits are the LEAST snappable.** $\tanh$ at relRMSE
   $3.8\!\cdot\!10^{-3}$ and $e^{-x^2}$ at $3.9\!\cdot\!10^{-3}$ each have
   exactly zero parameters that admit a snap. The optimizer found
   genuinely continuous-valued solutions.

## A representative parameter table

For $x^3 - x$ (the polynomial — the proof's headline construction), here
is the fitted tree's parameter table:

```
atom            a        b        c        d        e        f
l0p0        1.516   -0.654    0.796   -2.080    0.652    0.029
l1p0        1.218   -0.436    0.147   -0.755   -0.034    1.501
l1p1        1.803    0.473    0.138   -2.348    0.115   -0.785
l2p0        1.299   -1.359    0.847   -0.143    0.127    0.449
l2p1        1.272   -2.646    0.116   -0.065    0.357    0.841
l2p2        1.267   -0.072    0.504   -1.190    0.114    0.471
l2p3        0.913    2.388    0.475   -0.672   -0.219    1.080
```

Read the columns and look for "interpretable" values:

- `a` column (outer scale of the exp arm): 1.52, 1.22, 1.80, 1.30, 1.27,
  1.27, 0.91 — *clustered around 1–2 but no exact integers*. The
  constructive proof (Fig. ExponentiationWithConstant) uses $a = 1$
  exactly; the fit drifts.
- `d` column (outer scale of the log arm): −2.08, −0.76, −2.35, −0.14,
  −0.07, −1.19, −0.67 — *no values near the original EML's $d = -1$*.
  Several are near zero (−0.14, −0.07), meaning those atoms' log arms
  are nearly off — the atom is reduced to "exp arm only."
- `b` column (slope inside exp): −0.654, −0.436, 0.473, −1.359, −2.646,
  −0.072, 2.388 — *fully continuous*; the constructive proof uses $b
  \in \{1\}$ at most points, with integer powers built by composition.

Compare to the proof's polynomial construction (Lemma in §5): a degree-3
polynomial is built from 3 monomial blocks (each $mx^n$, 2 atoms) and 2
addition blocks (3 atoms each), totaling 12 atoms with $a = 1$ exactly,
$d = 0$ in the monomial blocks (no log arm used), $b = 1$, $c = \log m$,
etc. The fitted tree uses 7 atoms but spreads the work across 42
continuous parameters with no resemblance to the constructive scheme.

## What the trees are doing instead

A depth-3 PEM has 42 free parameters. A degree-3 polynomial on $[-1.5,
1.5]$ has 4 coefficients. The fit is enormously over-parameterized, and
the optimizer uses the slack to wander into a smooth region of parameter
space where the loss is low. The resulting expression is, structurally:

$$
T(x) = a_0\, \exp\bigl( b_0\, [\text{stuff}_L(x)] + c_0 \bigr)
       + d_0\, \ln^+\!\bigl( e_0\, [\text{stuff}_R(x)] + f_0 \bigr)
$$

where the $[\text{stuff}]$ branches are themselves nested
$\exp + \ln$-of-softplus expressions in $x$. The function value on the
$[-1.5, 1.5]$ grid happens to match $x^3 - x$ to $0.7\%$, but the
mechanism is *additive interference of 7 stacked exp/log curves with
arbitrary continuous parameters*, not a recognizable polynomial expansion.

This is also why **complex parameters didn't help on $\sin x$** in the §4
table: at depth 3 the optimizer can already approximate $\sin x$ to
$3 \!\cdot\! 10^{-3}$ via continuous exp/log interference on $\R$,
without needing the imaginary axis. The trig path through $\ln(-1) = i\pi$
that the construction uses is a different (and, for the optimizer,
non-obvious) basin.

## Why this happens

Three reasons, in order of importance.

1. **The PEM parameter space is much larger than the symbolic basis.**
   Each atom has $\R^6$ of freedom. A symbolic atom is one of finitely
   many points in $\R^6$ (e.g., $(1, 1, \log m, 0, *, *)$ for a $mx^n$
   block, with the unused $e, f$ parameters arbitrary because $d = 0$).
   The symbolic basin has measure zero in $\R^6$; gradient descent from
   Gaussian initialization almost surely lands somewhere else.
2. **The loss landscape is shape-determined, not parameter-determined.**
   What the optimizer minimizes is $\|T_\theta - f\|_2^2$ over the
   training grid. Many parameter settings produce the same function shape
   to within MSE tolerance. The optimizer settles on whichever one its
   trajectory reaches first; that one is overwhelmingly likely to be
   "non-symbolic."
3. **Snap tolerance fights with optimization tolerance.** At relRMSE $=
   3 \!\cdot\! 10^{-3}$, every parameter is fine-tuned to ~3 significant
   digits to maintain that loss. Snapping a parameter from $-1.190$ to
   $-1.0$ is an 19% perturbation; the loss surface is sharp enough that
   such a perturbation usually lifts loss by more than the 30% tolerance.

## What this means for the paper

The paper's framing ("EML trees as a theoretically grounded framework for
function approximation") is supported by the §4 results: trees do
approximate to small error in practice. The connected claim sometimes
made about EML trees — that the recovered tree is *interpretable*, with
each atom's parameters carrying symbolic meaning — is **not** supported
by the fitted trees. Without an interpretability-promoting objective
(e.g., $\ell_0$/$\ell_1$ on deviation from $\{0, \pm 1\}$, or warm-start
from the proof's construction), the trees are smooth curve-fits.

There are two clean ways to handle this in the paper:

- **Option A (honest, current direction):** Add one paragraph to §4.4
  Limitations noting that the trees are *function-level* approximators,
  not symbolic recoveries, and that interpretable recovery would require
  additional structural regularization. This is the simplest and most
  defensible position given the data we have.
- **Option B (broader claim):** Run an additional experiment that
  *forces* discreteness — sparsity penalty pulling each parameter
  toward $\{0, \pm 1\}$, or a two-stage train+snap+refit with tighter
  tolerance. Report what's recoverable when the optimizer is told what to
  prefer. This is a follow-up paper's worth of work.

Option A keeps the §4 honest and tight. I recommend going with it for
this NeurIPS submission and flagging Option B as future work.

## Snapped expressions (illustrative)

For $x^3 - x$ (the most snappable case, 7/42 params snapped):

```
T_snap(x) = (1.52*exp(-0.649*((1.22*exp(-0.435*(1.3*exp(-1.37*x + 0.845))
            + 0.147) + -0.749*ln+(-0.0354*(1.27*exp(-2.63*x + 0.110))
            + 1.5))) + 0.795)
            + -2.08*ln+(0.652*((1.8*exp(0.471*((1.26*exp(-0.0722*x + 0.502)
            + -1.17*ln+(0.114*x + 0.473))) + 0.137)
            + -2.36*ln+(0.114*((0.91*exp(2.38*x + 0.471)
            + -0.672*ln+(-0.216*x + 1))) + -0.783))) + 0.0301))
```

Note `ln+` denotes $\ln(\mathrm{softplus}(\cdot) + \eps)$ (the
real-softplus surrogate from the §4 setup). The structure is recognisably
an EML tree, but reading it as a closed-form polynomial expansion is
hopeless: it is a nested chain of seven $\exp/\ln$ curves with continuous
slopes and intercepts.

For comparison, the constructive proof would give something like (for
$x^3 - x$ rendered in the same notation):

```
T_construct(x) = (1*exp(0*x + 0) + -1*ln+(1*x^3 - 1*x + 0))   -- not real, conceptual
              \cong ADD(MX^N_BLOCK(1, 3), MX^N_BLOCK(-1, 1))    -- using §5 blocks
```

i.e., a much more parsimonious tree built from named blocks with $a, d
\in \{0, \pm 1\}$, $b, e \in \{1\}$, and the coefficients $m \in \{1,
-1\}$ absorbed into the $c$ via $c = \log |m|$ and the sign carried by the
sign of $a$. The fit doesn't find that.
