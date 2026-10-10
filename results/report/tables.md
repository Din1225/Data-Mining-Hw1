## main_cv

5-fold CV on the training portion (mean ± std over folds). Bold: best mean per column. Decision threshold: ±20% relative error. Within: share of players with value > 0 whose prediction is within the threshold. †: RMSLE ≤ ln(1.2) = 0.182, i.e. the typical error is within the threshold.

| Model | RMSLE | RMSLE (value > 0) | Within ±20% (%) | MAE (€K) | MedAE (€K) | R² |
|---|---|---|---|---|---|---|
| Trivial (training mean) | 2.002 ± 0.068 | 1.621 ± 0.024 | 7.4 ± 0.3 | 3,248 ± 63 | 2,169 ± 27 | 0.000 ± 0.000 |
| Simple (OLS on overall) | 1.293 ± 0.129 | 0.596 ± 0.007 | 26.3 ± 0.9 | 1,533 ± 54 | 292 ± 10 | 0.468 ± 0.013 |
| Strong (Random Forest) | 0.269 ± 0.114 | **0.098 ± 0.031 †** | **97.8 ± 0.4** | **174 ± 14** | **17 ± 1** | 0.969 ± 0.012 |
| Ours (OLS, Set C) | **0.171 ± 0.009 †** | 0.156 ± 0.006 † | 86.2 ± 0.4 | 362 ± 17 | 71 ± 2 | **0.976 ± 0.005** |

## linear_variants

Linear variants on Set C, tuned by 5-fold CV. Every penalized variant selects the smallest penalty in its grid, i.e. it reduces to OLS.

| Variant | Search space | Best setting | CV RMSLE |
|---|---|---|---|
| ols (selected) | — | — | 0.1711 ± 0.0091 |
| ridge | alpha ∈ [0.001, 10000] (15 values) | alpha = 0.001 | 0.1711 ± 0.0091 |
| lasso | alpha ∈ [1e-06, 0.1] (11 values) | alpha = 1e-06 | 0.1711 ± 0.0091 |
| elastic_net | alpha ∈ [1e-06, 0.1] (11 values); l1_ratio ∈ [0.2, 0.8] (3 values) | alpha = 1e-06, l1_ratio = 0.8 | 0.1711 ± 0.0091 |

## ablation

Feature-set ablation with the final OLS (same folds and tuning). Δ and paired t-test are per-fold differences from Set C. Decision threshold: ±20% relative error. Within: share of players with value > 0 whose prediction is within the threshold. †: RMSLE ≤ ln(1.2) = 0.182, i.e. the typical error is within the threshold.

| Feature set | #Features | RMSLE | Within ±20% (%) | ΔRMSLE vs C | Folds worse than C | Paired t p | MAE (€K) | R² |
|---|---|---|---|---|---|---|---|---|
| A: processed raw columns | 1,170–1,179 | 0.507 ± 0.083 | 61.6 ± 1.1 | 0.336 ± 0.079 | 5/5 | < 0.001 | 762 ± 63 | 0.763 ± 0.095 |
| B: top-31 by \|Pearson r\| with value | 31 | 1.143 ± 0.111 | 54.3 ± 0.4 | 0.972 ± 0.110 | 5/5 | < 0.001 | 1,404 ± 147 | 0.321 ± 0.181 |
| B': top-31 by \|Pearson r\| with log1p(value) | 31 | 0.613 ± 0.097 | 56.9 ± 0.6 | 0.442 ± 0.092 | 5/5 | < 0.001 | 930 ± 48 | 0.802 ± 0.024 |
| **C: ours (final)** | 31 | 0.171 ± 0.009 † | 86.2 ± 0.4 | — | — | — | 362 ± 17 | 0.976 ± 0.005 |
| C − nonlinear (overall², age²) | 29 | 0.228 ± 0.007 | 72.2 ± 1.2 | 0.057 ± 0.003 | 5/5 | < 0.001 | 528 ± 13 | 0.946 ± 0.006 |
| C − interaction (overall × reputation) | 30 | 0.174 ± 0.009 † | 85.5 ± 0.4 | 0.003 ± 0.001 | 5/5 | 0.004 | 471 ± 24 | 0.906 ± 0.033 |
| C − domain rules (free agent, age ≥ 40, top-5) | 28 | 1.041 ± 0.105 | 60.7 ± 2.6 | 0.870 ± 0.105 | 5/5 | < 0.001 | 1,239 ± 87 | 0.373 ± 0.103 |
| C − overall and derived (H1) | 28 | 0.283 ± 0.005 | 62.2 ± 1.0 | 0.112 ± 0.006 | 5/5 | < 0.001 | 667 ± 33 | 0.902 ± 0.008 |
| C − reputation and derived (H2) | 29 | 0.175 ± 0.009 † | 85.2 ± 0.4 | 0.003 ± 0.001 | 5/5 | 0.002 | 452 ± 25 | 0.922 ± 0.029 |
| C + club/league mean overall | 33 | 0.171 ± 0.009 † | 86.2 ± 0.4 | 0.000 ± 0.000 | 2/5 | 0.658 | 362 ± 17 | 0.976 ± 0.005 |

## slices

CV out-of-fold error by slice. Within ±20%: share of players with value > 0 whose prediction is within the decision threshold (undefined for €0 players). Bold: the better model.

| Dimension | Slice | n | RMSLE RF | RMSLE Ours | Within ±20% (%) RF | Within ±20% (%) Ours |
|---|---|---|---|---|---|---|
| value band | €0 | 93 | 3.382 | **0.898** | — | — |
| value band | <€1M | 7,201 | **0.129** | 0.167 | **97.0** | 87.9 |
| value band | €1M–10M | 6,543 | **0.058** | 0.136 | **99.0** | 86.0 |
| value band | ≥€10M | 887 | **0.100** | 0.199 | **95.4** | 74.2 |
| league level | level 1 | 10,799 | 0.334 | **0.170** | **97.4** | 85.0 |
| league level | level 2 | 2,317 | **0.048** | 0.144 | **99.1** | 88.6 |
| league level | level 3–4 | 1,534 | **0.071** | 0.136 | **98.8** | 91.1 |
| league level | no club | 74 | **0.006** | 0.769 | — | — |
| position | GK | 1,672 | **0.241** | 0.275 | **94.0** | 74.1 |
| position | DEF | 4,864 | 0.268 | **0.139** | **98.1** | 88.8 |
| position | MID | 5,514 | 0.208 | **0.151** | **98.3** | 86.9 |
| position | FWD | 2,674 | 0.445 | **0.180** | **98.6** | 87.7 |
| age band | ≤21 | 3,749 | **0.037** | 0.117 | **99.8** | 92.1 |
| age band | 22–29 | 8,057 | **0.048** | 0.140 | **99.5** | 87.9 |
| age band | ≥30 | 2,918 | 0.639 | **0.276** | **90.5** | 73.7 |

## threshold_sensitivity

Share of players with value > 0 whose CV prediction is within each candidate relative-error threshold (%, mean over 5 folds). The model ranking is the same at every threshold. Bold: best per column.

| Model | ±10% | ±20% (decided) | ±30% | ±50% |
|---|---|---|---|---|
| Trivial (training mean) | 3.8 | 7.4 | 10.6 | 16.8 |
| Simple (OLS on overall) | 13.1 | 26.3 | 40.2 | 77.4 |
| Strong (Random Forest) | **92.4** | **97.8** | **99.0** | **99.7** |
| Ours (OLS, Set C) | 54.6 | 86.2 | 95.8 | 98.7 |

## test

Held-out test set (evaluated once, after all model and feature decisions were fixed). Bold: best per column. Decision threshold: ±20% relative error. Within: share of players with value > 0 whose prediction is within the threshold. †: RMSLE ≤ ln(1.2) = 0.182, i.e. the typical error is within the threshold.

| Model | RMSLE | RMSLE (value > 0) | Within ±20% (%) | MAE (€K) | MedAE (€K) | R² |
|---|---|---|---|---|---|---|
| Trivial (training mean) | 1.867 | 1.592 | 8.2 | 3,265 | 2,139 | 0.000 |
| Simple (OLS on overall) | 1.109 | 0.593 | 25.3 | 1,572 | 304 | 0.478 |
| Strong (Random Forest) | **0.078 †** | **0.070 †** | **98.2** | **155** | **16** | **0.979** |
| Ours (OLS, Set C) | 0.160 † | 0.155 † | 85.8 | 365 | 74 | 0.977 |
