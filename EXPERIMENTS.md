# Experiment log

Every run logged here: timestamp, script, config, wall-clock, output, headline result.

| # | Date       | Script              | Config                                  | Wall (s) | Output                  | Headline                                                                 |
|---|------------|---------------------|-----------------------------------------|----------|-------------------------|--------------------------------------------------------------------------|
| 1 | 2026-05-04 | `exp_1d.py`         | 11 targets, d∈[1,4], 12 restart, 3000 it| 974      | `results_1d.json`       | d=4 hits ≤2% relRMSE on 9/11 targets; rip stays at 7.5%                  |
| 2 | 2026-05-04 | `exp_baselines.py`  | 6 targets, P∈{14,34,74}, 8 restart      | 515      | `results_baselines.json`| MLP/RBF beat eml 5–10× at every budget; poly wins low-degree, dies high  |
| 3 | 2026-05-04 | `exp_2d.py`         | 4 targets, d∈[2,4], 25×25 grid, 8 restart| 262     | `results_2d.json`       | d=4 best 5–10% relRMSE, success ≤25%                                     |
| 4 | 2026-05-04 | `exp_landscape.py`  | 4 targets, d∈[2,4], 50 seeds, 2000 it   | 915      | `results_landscape.json`| sin/gauss <5%: 52–66%; <1%: 2–4%. rip never <5%                          |
| 5 | 2026-05-04 | `exp_snap.py`       | 7 targets, d=4, pure/coef/greedy snap   | 336      | `results_snap.json`     | Greedy snap *improves* loss every target; gauss 43% slots snapped         |
| 6 | 2026-05-04 | `exp_strategy.py`   | 5 targets, 4 strategies, 30s budget     | 579      | `results_strategy.json` | LBFGS refinement (B) wins universally; rip 9.9e-2→1.9e-2                  |
| 7 | 2026-05-04 | `exp_universality.py`| 11 targets, d∈[2,6], 25s budget, strat B| 1080    | `results_universality.json`| 10/11 targets ≤1% relRMSE; rip plateaus at 1.9%; optimal depth often d=3 |
| 8 | 2026-05-04 | `exp_complex.py`    | 5 targets × 3 depths × {real,complex}   | 575     | `results_complex.json`  | complex_real is WORSE on every target/depth — rip wall is not lack of complex |
| 9 | 2026-05-04 | `exp_complex_params.py`| 7 targets × 3 depths × 3 settings (R/CR/CC)| 1269 | `results_complex_params.json` | CC breaks rip wall at d=3 (3.7× better than R); hurts non-osc targets   |
| 10| 2026-05-04 | `exp_complex_anneal.py`| 5 targets × 3 depths × 4 λ schedules    | 1245    | `results_complex_anneal.json`| λ=0 best for osc; anneal gives clean Im AND helps non-osc (x²/exp_decay) |
| 11| 2026-05-04 | `exp_rip_push.py`   | rip @ d∈[3,6], 60s budget, R vs CC      | 423     | `results_rip_push.json` | **Rip wall breaks**: R d=6 → 0.58% relRMSE; CC plateaus at 4–5%          |
| 12| 2026-05-04 | `exp_parametric.py` | 11 targets × 4 depths × {PEM,EM}, 25s/cell| 1675   | `results_parametric.json`| PEM beats EM at d=3 on rip (5.6×), sin (3×), abs (5×), tanh, log_safe; EM wins on x², exp_decay |
| 13| 2026-05-04 | `exp_parametric_snap.py`| 7 targets, d=3 PEM + greedy snap     | 234     | `results_parametric_snap.json`| Snap rate modest (5–21%) but loss decreases on every target |
| 14| 2026-05-06 | `exp_universality_pem.py`| 5 targets × 3 depths × {real, complex} PEM, 30s/cell | 750 | `results_universality_pem.json`| Real PEM hits sub-1% on 4/5 by d=4; complex helps only on rip; basis for NeurIPS §4 |
| 15| 2026-05-06 | `exp_inspect_pem.py`| 5 targets × d=3 PEM, refit + greedy snap to {-2,-1,0,1,2} | 360 | `results_inspect_pem.json`| At most 17% of params snap; 0 atoms become fully discrete; trees are continuous fits, not symbolic recoveries |
| 10| 2026-05-04 | `exp_complex_anneal.py`| 5 targets × 3 depths × 4 λ schedules     | 1245    | `results_complex_anneal.json` | λ=0 is best for oscillatory; anneal→0.1 helps non-osc at d=4 (exp_decay 5.7e-4 beats R) |
| 11| 2026-05-04 | `exp_rip_push.py`     | rip target, d=3..6, 60s budget, R vs CC  | 423     | `results_rip_push.json`  | **rip wall breaks**: R d=6 60s → 5.8e-3 (0.58%). CC plateaus ~4-5%       |
