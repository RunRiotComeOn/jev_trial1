# E5: cross-dataset generalization of the lightweight models

Rows are training configurations, columns are test sets. `only:A` = full training pool of A; `loo:B` = every pool except B; `mix:k` = k labels from each dataset (`full` = all pools). `*` = the test dataset's own training data was used (in-domain). Baselines: mean over seeds; Jev: zero-shot, never trained, with bootstrap 95% interval.

## F1

**raw**

| Train | Model | youtube | sms |
|---|---|---|---|
| **Jev (zero-shot)** | – | 0.940 [0.918, 0.958] | 0.931 [0.908, 0.954] |
| only:youtube | Naive Bayes | 0.913* | 0.319 |
| only:youtube | TF-IDF + LR | 0.934* | 0.344 |
| only:sms | Naive Bayes | 0.378 | 0.932* |
| only:sms | TF-IDF + LR | 0.020 | 0.818* |
| mix:100 | Naive Bayes | 0.862* | 0.658* |
| mix:100 | TF-IDF + LR | 0.565* | 0.005* |
| mix:500 | Naive Bayes | 0.913* | 0.837* |
| mix:500 | TF-IDF + LR | 0.847* | 0.412* |
| mix:1000 | Naive Bayes | 0.926* | 0.862* |
| mix:1000 | TF-IDF + LR | 0.889* | 0.687* |
| mix:full | Naive Bayes | 0.916* | 0.922* |
| mix:full | TF-IDF + LR | 0.869* | 0.892* |

**calibrated**

| Train | Model | youtube | sms |
|---|---|---|---|
| **Jev (zero-shot)** | – | 0.940 [0.918, 0.958] | 0.931 [0.908, 0.954] |
| only:youtube | Naive Bayes | 0.927* | 0.381 |
| only:youtube | TF-IDF + LR | 0.928* | 0.342 |
| only:sms | Naive Bayes | 0.180 | 0.921* |
| only:sms | TF-IDF + LR | 0.165 | 0.917* |
| mix:100 | Naive Bayes | 0.831* | 0.593* |
| mix:100 | TF-IDF + LR | 0.767* | 0.134* |
| mix:500 | Naive Bayes | 0.898* | 0.848* |
| mix:500 | TF-IDF + LR | 0.911* | 0.721* |
| mix:1000 | Naive Bayes | 0.913* | 0.882* |
| mix:1000 | TF-IDF + LR | 0.928* | 0.824* |
| mix:full | Naive Bayes | 0.893* | 0.925* |
| mix:full | TF-IDF + LR | 0.925* | 0.906* |

## Brier (lower is better)

**raw**

| Train | Model | youtube | sms |
|---|---|---|---|
| **Jev (zero-shot)** | – | 0.052 [0.040, 0.065] | 0.015 [0.011, 0.018] |
| only:youtube | Naive Bayes | 0.068* | 0.403 |
| only:youtube | TF-IDF + LR | 0.078* | 0.213 |
| only:sms | Naive Bayes | 0.393 | 0.015* |
| only:sms | TF-IDF + LR | 0.383 | 0.035* |
| mix:100 | Naive Bayes | 0.112* | 0.076* |
| mix:100 | TF-IDF + LR | 0.181* | 0.120* |
| mix:500 | Naive Bayes | 0.073* | 0.038* |
| mix:500 | TF-IDF + LR | 0.108* | 0.087* |
| mix:1000 | Naive Bayes | 0.064* | 0.032* |
| mix:1000 | TF-IDF + LR | 0.087* | 0.070* |
| mix:full | Naive Bayes | 0.074* | 0.018* |
| mix:full | TF-IDF + LR | 0.087* | 0.036* |

**calibrated**

| Train | Model | youtube | sms |
|---|---|---|---|
| **Jev (zero-shot)** | – | 0.052 [0.040, 0.065] | 0.015 [0.011, 0.018] |
| only:youtube | Naive Bayes | 0.059* | 0.227 |
| only:youtube | TF-IDF + LR | 0.054* | 0.204 |
| only:sms | Naive Bayes | 0.404 | 0.017* |
| only:sms | TF-IDF + LR | 0.409 | 0.016* |
| mix:100 | Naive Bayes | 0.127* | 0.077* |
| mix:100 | TF-IDF + LR | 0.139* | 0.100* |
| mix:500 | Naive Bayes | 0.081* | 0.033* |
| mix:500 | TF-IDF + LR | 0.065* | 0.048* |
| mix:1000 | Naive Bayes | 0.070* | 0.027* |
| mix:1000 | TF-IDF + LR | 0.053* | 0.038* |
| mix:full | Naive Bayes | 0.085* | 0.017* |
| mix:full | TF-IDF + LR | 0.056* | 0.019* |

## Coverage @5% risk

**raw**

| Train | Model | youtube | sms |
|---|---|---|---|
| **Jev (zero-shot)** | – | 0.871 | 0.999 |
| only:youtube | Naive Bayes | 1.000* | 0.648 |
| only:youtube | TF-IDF + LR | 0.244* | 0.000 |
| only:sms | Naive Bayes | 0.985 | 1.000* |
| only:sms | TF-IDF + LR | 0.245 | 0.504* |
| mix:100 | Naive Bayes | 0.861* | 0.872* |
| mix:100 | TF-IDF + LR | 0.000* | 0.000* |
| mix:500 | Naive Bayes | 0.981* | 0.999* |
| mix:500 | TF-IDF + LR | 0.067* | 0.000* |
| mix:1000 | Naive Bayes | 0.999* | 0.999* |
| mix:1000 | TF-IDF + LR | 0.198* | 0.000* |
| mix:full | Naive Bayes | 1.000* | 0.999* |
| mix:full | TF-IDF + LR | 0.295* | 0.189* |

**calibrated**

| Train | Model | youtube | sms |
|---|---|---|---|
| **Jev (zero-shot)** | – | 0.871 | 0.999 |
| only:youtube | Naive Bayes | 0.952* | 0.369 |
| only:youtube | TF-IDF + LR | 0.950* | 0.145 |
| only:sms | Naive Bayes | 0.892 | 1.000* |
| only:sms | TF-IDF + LR | 0.981 | 1.000* |
| mix:100 | Naive Bayes | 0.593* | 0.642* |
| mix:100 | TF-IDF + LR | 0.078* | 0.000* |
| mix:500 | Naive Bayes | 0.869* | 0.966* |
| mix:500 | TF-IDF + LR | 0.911* | 0.775* |
| mix:1000 | Naive Bayes | 0.930* | 0.989* |
| mix:1000 | TF-IDF + LR | 0.995* | 0.915* |
| mix:full | Naive Bayes | 0.883* | 1.000* |
| mix:full | TF-IDF + LR | 0.995* | 1.000* |

## Realized risk among automated messages (target 0.05)

**raw**

| Train | Model | youtube | sms |
|---|---|---|---|
| **Jev (zero-shot)** | – | 0.035 | 0.018 |
| only:youtube | Naive Bayes | 0.092* | 0.519 |
| only:youtube | TF-IDF + LR | 0.000* | 0.000 |
| only:sms | Naive Bayes | 0.400 | 0.017* |
| only:sms | TF-IDF + LR | 0.299 | 0.000* |
| mix:100 | Naive Bayes | 0.105* | 0.060* |
| mix:100 | TF-IDF + LR | 0.000* | 0.000* |
| mix:500 | Naive Bayes | 0.078* | 0.045* |
| mix:500 | TF-IDF + LR | 0.000* | 0.000* |
| mix:1000 | Naive Bayes | 0.073* | 0.038* |
| mix:1000 | TF-IDF + LR | 0.000* | 0.000* |
| mix:full | Naive Bayes | 0.082* | 0.020* |
| mix:full | TF-IDF + LR | 0.000* | 0.000* |

**calibrated**

| Train | Model | youtube | sms |
|---|---|---|---|
| **Jev (zero-shot)** | – | 0.035 | 0.018 |
| only:youtube | Naive Bayes | 0.060* | 0.257 |
| only:youtube | TF-IDF + LR | 0.056* | 0.218 |
| only:sms | Naive Bayes | 0.444 | 0.020* |
| only:sms | TF-IDF + LR | 0.468 | 0.021* |
| mix:100 | Naive Bayes | 0.132* | 0.046* |
| mix:100 | TF-IDF + LR | 0.001* | 0.000* |
| mix:500 | Naive Bayes | 0.070* | 0.029* |
| mix:500 | TF-IDF + LR | 0.056* | 0.013* |
| mix:1000 | Naive Bayes | 0.066* | 0.027* |
| mix:1000 | TF-IDF + LR | 0.068* | 0.023* |
| mix:full | Naive Bayes | 0.075* | 0.019* |
| mix:full | TF-IDF + LR | 0.071* | 0.024* |

