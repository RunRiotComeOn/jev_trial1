# Pilot results: Jev vs lightweight models on spam filtering

> **Jev has not been run yet** (no complete `jev_cache.jsonl`). Baseline curves only; n*, E3, E4 Jev columns, stability and wording checks are pending.

## Jev zero-shot (test set, main rubric)

| Dataset | F1 | Brier | ECE | Coverage @5% | Realized risk | Model string |
|---|---|---|---|---|---|---|
| youtube | pending | | | | | |
| sms | pending | | | | | |

## Baselines at full pool (true labels, mean ± sd over 10 seeds)

| Dataset | Model | Variant | n | F1 | Brier | ECE | Coverage @5% | Realized risk |
|---|---|---|---|---|---|---|---|---|
| youtube | Naive Bayes | raw | 1369 | 0.913 ± 0.000 | 0.068 | 0.057 | 1.000 | 0.092 |
| youtube | Naive Bayes | calibrated | 1369 | 0.925 ± 0.004 | 0.059 | 0.036 | 0.963 | 0.063 |
| youtube | TF-IDF + LR | raw | 1369 | 0.934 ± 0.000 | 0.078 | 0.144 | 0.244 | 0.000 |
| youtube | TF-IDF + LR | calibrated | 1369 | 0.924 ± 0.007 | 0.055 | 0.030 | 0.957 | 0.059 |
| sms | Naive Bayes | raw | 3901 | 0.932 ± 0.000 | 0.015 | 0.009 | 1.000 | 0.017 |
| sms | Naive Bayes | calibrated | 3901 | 0.922 ± 0.005 | 0.017 | 0.008 | 1.000 | 0.019 |
| sms | TF-IDF + LR | raw | 3901 | 0.818 ± 0.000 | 0.035 | 0.087 | 0.504 | 0.000 |
| sms | TF-IDF + LR | calibrated | 3901 | 0.916 ± 0.003 | 0.016 | 0.007 | 1.000 | 0.021 |

## n*: labels needed to match Jev (E1, E2)

Smallest budget whose mean reaches Jev's bootstrap interval; `>pool` = never reached.

| Dataset | Model | Variant | n* F1 | n* Brier | n* ECE | n* coverage |
|---|---|---|---|---|---|---|
| youtube | – | – | pending | pending | pending | pending |
| sms | – | – | pending | pending | pending | pending |

## E4: latency and cost per 1,000 decisions

| Dataset | Model | Median ms | p95 ms | US$ / 1,000 |
|---|---|---|---|---|
| youtube | Jev (client-side) | pending |  |  |
| youtube | Naive Bayes (CPU) | 0.36 | 0.49 | ~0 |
| youtube | TF-IDF + LR (CPU) | 0.66 | 4.66 | ~0 |
| sms | Jev (client-side) | pending |  |  |
| sms | Naive Bayes (CPU) | 0.39 | 0.52 | ~0 |
| sms | TF-IDF + LR (CPU) | 0.71 | 1.17 | ~0 |

## Checks

- **Wording sensitivity (youtube):** pending.
- **Stability (youtube):** pending (run `jev_run.py --stability --run-tag day2` on a second day).
- **Wording sensitivity (sms):** pending.
- **Stability (sms):** pending (run `jev_run.py --stability --run-tag day2` on a second day).

## Figures

![Figure 1](figures/fig1_f1.png)

![Figure 2](figures/fig2_calibration.png)

## Notes

- Coverage @5% risk: each model automates messages whose confidence max(p, 1−p) clears a threshold chosen on its own (unlabeled) training-pool predictions so that its expected error is ≤ 5%. Realized risk is the actual error rate among automated test messages.
- Calibrated variants fit on 80% of the sample and fit Platt scaling on the other 20%; when that 20% lacks a class (small n) Platt scaling is skipped.
- Baselines: mean over 10 stratified draws per budget. Jev: bootstrap 95% intervals over test messages.
