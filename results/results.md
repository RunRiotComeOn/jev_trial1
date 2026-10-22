# Pilot results: Jev vs lightweight models on spam filtering

## Jev zero-shot (test set, main rubric)

| Dataset | F1 | Brier | ECE | Coverage @5% | Realized risk | Model string |
|---|---|---|---|---|---|---|
| youtube | 0.938 [0.917, 0.956] | 0.051 [0.040, 0.065] | 0.073 | 0.876 | 0.039 | jev-1.13.0 |
| sms | 0.927 [0.903, 0.949] | 0.014 [0.011, 0.018] | 0.032 | 1.000 | 0.020 | jev-1.13.0 |

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
| youtube | Naive Bayes | calibrated | 1369 | 1369 | 200 | 500 |
| youtube | Naive Bayes | raw | >pool | >pool | 500 | 200 |
| youtube | TF-IDF + LR | calibrated | 1369 | 1369 | 500 | 1369 |
| youtube | TF-IDF + LR | raw | 500 | >pool | >pool | >pool |
| sms | Naive Bayes | calibrated | 3901 | 3901 | 500 | >pool |
| sms | Naive Bayes | raw | 3901 | 3901 | 200 | 3901 |
| sms | TF-IDF + LR | calibrated | 3901 | 3901 | 3901 | >pool |
| sms | TF-IDF + LR | raw | >pool | >pool | >pool | >pool |

## E3: Jev as teacher

| Dataset | Model | Label source | n* F1 (vs Jev) | Best mean F1 | Exceeds Jev? |
|---|---|---|---|---|---|
| youtube | Naive Bayes | true | >pool | 0.913 | no |
| youtube | TF-IDF + LR | true | 500 | 0.934 | no |
| youtube | Naive Bayes | jev_hard | >pool | 0.913 | no |
| youtube | TF-IDF + LR | jev_hard | >pool | 0.916 | no |
| youtube | Naive Bayes | jev_soft | >pool | 0.900 | no |
| youtube | TF-IDF + LR | jev_soft | >pool | 0.902 | no |
| sms | Naive Bayes | true | 3901 | 0.932 | yes |
| sms | TF-IDF + LR | true | >pool | 0.818 | no |
| sms | Naive Bayes | jev_hard | 3901 | 0.924 | no |
| sms | TF-IDF + LR | jev_hard | >pool | 0.807 | no |
| sms | Naive Bayes | jev_soft | 3901 | 0.932 | yes |
| sms | TF-IDF + LR | jev_soft | >pool | 0.826 | no |

## E4: latency and cost per 1,000 decisions

| Dataset | Model | Median ms | p95 ms | US$ / 1,000 |
|---|---|---|---|---|
| youtube | Jev (client-side) | 95.1 | 169.8 | 0.016296 |
| youtube | Naive Bayes (CPU) | 0.20 | 0.25 | ~0 |
| youtube | TF-IDF + LR (CPU) | 0.51 | 4.58 | ~0 |
| sms | Jev (client-side) | 97.8 | 169.0 | 0.015861 |
| sms | Naive Bayes (CPU) | 0.21 | 0.26 | ~0 |
| sms | TF-IDF + LR (CPU) | 0.50 | 4.57 | ~0 |

## Checks

- **Wording sensitivity (youtube):** alt rubric F1 0.935 vs main 0.938; Brier 0.056 vs 0.051.
- **Stability (youtube):** 0/100 decisions changed on day 2 (mean |Δp| 0.006; day-2 model jev-1.13.0).
- **Wording sensitivity (sms):** alt rubric F1 0.928 vs main 0.927; Brier 0.014 vs 0.014.
- **Stability (sms):** 0/100 decisions changed on day 2 (mean |Δp| 0.003; day-2 model jev-1.13.0).
- **Contamination:** Jev F1 minus best full-pool baseline F1: youtube +0.005, sms -0.005. A much larger gap on SMS than on YouTube suggests exposure, not skill.

## Figures

![Figure 1](figures/fig1_f1.png)

![Figure 2](figures/fig2_calibration.png)

![Figure 3](figures/fig3_student.png)

## Notes

- Coverage @5% risk: each model automates messages whose confidence max(p, 1−p) clears a threshold chosen on its own (unlabeled) training-pool predictions so that its expected error is ≤ 5%. Realized risk is the actual error rate among automated test messages.
- Calibrated variants fit on 80% of the sample and fit Platt scaling on the other 20%; when that 20% lacks a class (small n) Platt scaling is skipped.
- Baselines: mean over 10 stratified draws per budget. Jev: bootstrap 95% intervals over test messages.
