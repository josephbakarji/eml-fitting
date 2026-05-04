# EML-tree fitting: a practical study

Empirical study of whether the binary operator
`eml(x, y) = exp(x) - log(y)` from
[Odrzywołek 2026 (arXiv:2603.21852)](https://arxiv.org/abs/2603.21852)
can be used as a practical function-approximation primitive via
gradient-based optimisation.

The paper proves that `eml` together with the constant `1` generates every
elementary function as a finite tree. It also shows that *blind symbolic
recovery* of those trees by gradient methods works at depth ≤ 4 but
collapses to <1% success at depth ≥ 5. We ask the empirical follow-up
question: with a continuous-parameter relaxation of the grammar and a
properly designed optimisation strategy, can `eml`-trees fit arbitrary
target functions to small error in practice?

## Headline findings

1. **Practical universality realised on the canonical 1D test suite.**
   With Adam multi-restart + LBFGS refinement, continuous-parameter
   `eml`-trees reach **≤ 1% relRMSE on all 11 canonical 1D targets**,
   frequently below 0.3%. The hardest target (`sin(3x)·exp(-x²/2)`)
   reaches 0.58% at depth 6 with 60 s of compute on a single CPU.
2. **LBFGS refinement** after Adam reduces error 3–5× versus pure
   multi-restart Adam, and nearly 10× on the hardest target.
3. **Greedy slot-by-slot symbol snapping with refit** *improves* loss on
   every target, while making 10–43% of slots literal symbols `{1, x,
   child}`. Naive (pure) snapping is catastrophic — the continuous fit
   lives nowhere near the symbolic vertices.
4. **Complex parameters** trade depth for capacity on oscillatory targets:
   3.7× improvement on rip and pure-sin at depth 3, but they hurt
   non-oscillatory targets where the imaginary degrees of freedom have no
   useful work.
5. **Two architectures, two regimes.**
   - **EM** (affine-glue): each input slot is a continuous affine combo
     `a·x + b·child + c` around the parameter-free `eml` operator.
   - **PEM** (eml-all-the-way-down): each node carries its own parameters,
     `EML_θ(x, y) = a·exp(b·x + c) + d·ln(e·y + f)`, with raw children
     fed directly between nodes.

   PEM wins shape-rich targets at shallow depth (rip d=3: 5.6× better
   than EM d=3; matches EM d=4 with half the parameters). EM wins simple
   monotone shapes and scales better with depth.

## Repository layout

### Modules
- `eml_tree.py` — `EMLTree`, the affine-glue architecture (EM)
- `eml_tree_complex.py` — `CEMLTree`, complex-parameter variant of EM
  (twin (real, imag) Parameters, complex128 forward, λ_im annealing
  schedule)
- `eml_tree_parametric.py` — `ParametricEMLTree` (PEM), per-node 6-param
  formulation, eml-all-the-way-down

### Snap-to-symbol post-processing
- `snap.py` — three snap modes for EM (pure, coefficient, greedy + refit)
- `snap_parametric.py` — per-parameter greedy snap for PEM with discrete
  candidate sets `{0, ±1}` per parameter; expression decoder

### Experiments
- `exp_1d.py` — 1D depth scaling, 11 targets × depths 1–4
- `exp_2d.py` — bivariate fits (Franke, sincos, saddle, Rosenbrock)
- `exp_baselines.py` — vs. polynomial / MLP-tanh / RBF baselines
- `exp_landscape.py` — 50-seed landscape diagnostic
- `exp_snap.py` — three EM snap modes on 7 targets
- `exp_strategy.py` — A: random restart vs. B: Adam+LBFGS vs.
  C: curriculum vs. D: symbolic warm-start
- `exp_universality.py` — depth 2–6 sweep with strategy B (headline)
- `exp_complex.py` — `real_softplus` vs `complex_real` backend, real
  parameters
- `exp_complex_params.py` — R / CR / CC: real-soft + real, complex-real
  + real, complex-real + complex
- `exp_complex_anneal.py` — λ_im schedule sweep on CC
- `exp_rip_push.py` — rip target at d=3..6 with 60 s budget
- `exp_parametric.py` — PEM vs EM, 11 targets × 4 depths
- `exp_parametric_snap.py` — PEM snap at d=3

### Plots
- `make_plots.py`, `make_depth_plot.py`, `make_complex_plot.py`,
  `make_anneal_plot.py`, `make_pem_plot.py`

### Outputs
- `results_*.json` — per-experiment full breakdowns (per-run, per-cell)
- `*.png` — generated figures
- `EXPERIMENTS.md` — chronological experiment log with timing and
  headline result per run

### Documents
- `paper/main.tex` — full experimental write-up (22 pages)
- `paper/main.pdf` — compiled PDF
- `REPORT.md` — short prose summary of the early findings
- `WEAKNESSES.md` — self-audit of architectural, optimisation, and
  generalisation weaknesses

## Running the experiments

All experiments are pure Python. Dependencies:
```
torch >= 2.0
numpy
matplotlib
```

To reproduce a single experiment:
```
python3 exp_universality.py     # ~18 minutes single CPU
python3 make_depth_plot.py
```

Total compute across all experiments is ~3 hours on a single Apple-M
CPU, no GPU.

## Compute summary

| Experiment | Coverage | Wall (s) |
|---|---|---|
| `exp_1d` | 11 targets × 4 depths × 12 restarts | 974 |
| `exp_baselines` | 6 targets × 3 budgets × 4 methods | 515 |
| `exp_2d` | 4 targets × 3 depths × 8 restarts | 262 |
| `exp_landscape` | 4 targets × 3 depths × 50 seeds | 915 |
| `exp_snap` | 7 targets, 3 snap modes (depth 4) | 336 |
| `exp_strategy` | 5 targets × 4 strategies | 579 |
| `exp_universality` | 11 targets × 5 depths, strategy B | 1080 |
| `exp_complex` | 5 targets × 3 depths × 2 backends | 575 |
| `exp_complex_params` | 7 targets × 3 depths × 3 settings | 1269 |
| `exp_complex_anneal` | 5 targets × 3 depths × 4 schedules | 1245 |
| `exp_rip_push` | rip × 4 depths, 60 s budget | 423 |
| `exp_parametric` | 11 targets × 4 depths × {PEM, EM} | 1675 |
| `exp_parametric_snap` | 7 targets, depth 3 PEM + greedy snap | 234 |
| **Total** | | **~10 080 s ≈ 2.8 h** |

## Status

Experimental study; no theorem yet. The companion-theorem questions
(approximation-rate `d*(ε, f)`, identifiability of minimum-depth `eml`-trees,
branch-cut handling for the complex grammar, minimal sufficient grammar)
are open and listed in §7 of the paper.

## License

MIT.

## References

- Odrzywołek, A. *All elementary functions from a single binary operator*.
  arXiv:2603.21852, 2026.
- Reference PyTorch implementation of softmax-discrete symbolic recovery:
  [Ninhache/EML-Operator](https://github.com/Ninhache/EML-Operator).
