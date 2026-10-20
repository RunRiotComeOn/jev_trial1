# E5: cross-dataset generalization of the lightweight models

Rows are training configurations, columns are test sets. `only:A` = full training pool of A; `loo:B` = every pool except B; `mix:k` = k labels from each dataset (`full` = all pools). `*` = the test dataset's own training data was used (in-domain). Baselines: mean over seeds; Jev: zero-shot, never trained, with bootstrap 95% interval.

## F1

**raw**

| Train | Model | youtube | sms | enron |
|---|---|---|---|---|
| **Jev (zero-shot)** | – | 0.940 [0.918, 0.958] | 0.931 [0.908, 0.954] | 0.957 [0.952, 0.961] |
| only:youtube | Naive Bayes | 0.913* | 0.319 | 0.607 |
| only:youtube | TF-IDF + LR | 0.934* | 0.344 | 0.588 |
| only:sms | Naive Bayes | 0.378 | 0.932* | 0.331 |
| only:sms | TF-IDF + LR | 0.020 | 0.818* | 0.078 |
| only:enron | Naive Bayes | 0.654 | 0.298 | 0.988* |
| only:enron | TF-IDF + LR | 0.643 | 0.245 | 0.986* |
| loo:youtube | Naive Bayes | 0.630 | 0.475* | 0.988* |
| loo:youtube | TF-IDF + LR | 0.582 | 0.804* | 0.981* |
| loo:sms | Naive Bayes | 0.733* | 0.300 | 0.988* |
| loo:sms | TF-IDF + LR | 0.847* | 0.266 | 0.987* |
| loo:enron | Naive Bayes | 0.916* | 0.922* | 0.483 |
| loo:enron | TF-IDF + LR | 0.869* | 0.892* | 0.402 |
| mix:100 | Naive Bayes | 0.823* | 0.495* | 0.873* |
| mix:100 | TF-IDF + LR | 0.719* | 0.051* | 0.220* |
| mix:500 | Naive Bayes | 0.884* | 0.640* | 0.941* |
| mix:500 | TF-IDF + LR | 0.884* | 0.663* | 0.754* |
| mix:1000 | Naive Bayes | 0.889* | 0.661* | 0.962* |
| mix:1000 | TF-IDF + LR | 0.907* | 0.810* | 0.838* |
| mix:full | Naive Bayes | 0.773* | 0.464* | 0.988* |
| mix:full | TF-IDF + LR | 0.887* | 0.833* | 0.979* |

**calibrated**

| Train | Model | youtube | sms | enron |
|---|---|---|---|---|
| **Jev (zero-shot)** | – | 0.940 [0.918, 0.958] | 0.931 [0.908, 0.954] | 0.957 [0.952, 0.961] |
| only:youtube | Naive Bayes | 0.927* | 0.381 | 0.591 |
| only:youtube | TF-IDF + LR | 0.928* | 0.342 | 0.581 |
| only:sms | Naive Bayes | 0.180 | 0.921* | 0.284 |
| only:sms | TF-IDF + LR | 0.165 | 0.917* | 0.318 |
| only:enron | Naive Bayes | 0.648 | 0.297 | 0.988* |
| only:enron | TF-IDF + LR | 0.624 | 0.252 | 0.986* |
| loo:youtube | Naive Bayes | 0.487 | 0.747* | 0.979* |
| loo:youtube | TF-IDF + LR | 0.585 | 0.798* | 0.980* |
| loo:sms | Naive Bayes | 0.772* | 0.356 | 0.986* |
| loo:sms | TF-IDF + LR | 0.857* | 0.275 | 0.986* |
| loo:enron | Naive Bayes | 0.893* | 0.925* | 0.445 |
| loo:enron | TF-IDF + LR | 0.925* | 0.906* | 0.538 |
| mix:100 | Naive Bayes | 0.786* | 0.549* | 0.812* |
| mix:100 | TF-IDF + LR | 0.846* | 0.377* | 0.632* |
| mix:500 | Naive Bayes | 0.873* | 0.762* | 0.916* |
| mix:500 | TF-IDF + LR | 0.920* | 0.795* | 0.872* |
| mix:1000 | Naive Bayes | 0.886* | 0.809* | 0.942* |
| mix:1000 | TF-IDF + LR | 0.929* | 0.835* | 0.894* |
| mix:full | Naive Bayes | 0.727* | 0.720* | 0.980* |
| mix:full | TF-IDF + LR | 0.881* | 0.820* | 0.978* |

## Brier (lower is better)

**raw**

| Train | Model | youtube | sms | enron |
|---|---|---|---|---|
| **Jev (zero-shot)** | – | 0.052 [0.040, 0.065] | 0.015 [0.011, 0.018] | 0.032 [0.030, 0.035] |
| only:youtube | Naive Bayes | 0.068* | 0.403 | 0.484 |
| only:youtube | TF-IDF + LR | 0.078* | 0.213 | 0.284 |
| only:sms | Naive Bayes | 0.393 | 0.015* | 0.462 |
| only:sms | TF-IDF + LR | 0.383 | 0.035* | 0.325 |
| only:enron | Naive Bayes | 0.370 | 0.466 | 0.009* |
| only:enron | TF-IDF + LR | 0.319 | 0.419 | 0.022* |
| loo:youtube | Naive Bayes | 0.333 | 0.215* | 0.010* |
| loo:youtube | TF-IDF + LR | 0.246 | 0.085* | 0.027* |
| loo:sms | Naive Bayes | 0.265* | 0.461 | 0.009* |
| loo:sms | TF-IDF + LR | 0.122* | 0.351 | 0.022* |
| loo:enron | Naive Bayes | 0.074* | 0.018* | 0.415 |
| loo:enron | TF-IDF + LR | 0.087* | 0.036* | 0.262 |
| mix:100 | Naive Bayes | 0.147* | 0.154* | 0.095* |
| mix:100 | TF-IDF + LR | 0.171* | 0.144* | 0.211* |
| mix:500 | Naive Bayes | 0.101* | 0.098* | 0.047* |
| mix:500 | TF-IDF + LR | 0.107* | 0.105* | 0.149* |
| mix:1000 | Naive Bayes | 0.096* | 0.096* | 0.030* |
| mix:1000 | TF-IDF + LR | 0.087* | 0.086* | 0.120* |
| mix:full | Naive Bayes | 0.221* | 0.223* | 0.010* |
| mix:full | TF-IDF + LR | 0.097* | 0.083* | 0.027* |

**calibrated**

| Train | Model | youtube | sms | enron |
|---|---|---|---|---|
| **Jev (zero-shot)** | – | 0.052 [0.040, 0.065] | 0.015 [0.011, 0.018] | 0.032 [0.030, 0.035] |
| only:youtube | Naive Bayes | 0.059* | 0.227 | 0.462 |
| only:youtube | TF-IDF + LR | 0.054* | 0.204 | 0.373 |
| only:sms | Naive Bayes | 0.404 | 0.017* | 0.450 |
| only:sms | TF-IDF + LR | 0.409 | 0.016* | 0.336 |
| only:enron | Naive Bayes | 0.298 | 0.388 | 0.009* |
| only:enron | TF-IDF + LR | 0.433 | 0.568 | 0.010* |
| loo:youtube | Naive Bayes | 0.280 | 0.068* | 0.015* |
| loo:youtube | TF-IDF + LR | 0.294 | 0.048* | 0.015* |
| loo:sms | Naive Bayes | 0.181* | 0.286 | 0.010* |
| loo:sms | TF-IDF + LR | 0.115* | 0.416 | 0.010* |
| loo:enron | Naive Bayes | 0.085* | 0.017* | 0.402 |
| loo:enron | TF-IDF + LR | 0.056* | 0.019* | 0.303 |
| mix:100 | Naive Bayes | 0.151* | 0.117* | 0.112* |
| mix:100 | TF-IDF + LR | 0.115* | 0.110* | 0.169* |
| mix:500 | Naive Bayes | 0.095* | 0.062* | 0.060* |
| mix:500 | TF-IDF + LR | 0.059* | 0.050* | 0.084* |
| mix:1000 | Naive Bayes | 0.081* | 0.048* | 0.042* |
| mix:1000 | TF-IDF + LR | 0.053* | 0.040* | 0.071* |
| mix:full | Naive Bayes | 0.170* | 0.076* | 0.014* |
| mix:full | TF-IDF + LR | 0.084* | 0.046* | 0.016* |

## Coverage @5% risk

**raw**

| Train | Model | youtube | sms | enron |
|---|---|---|---|---|
| **Jev (zero-shot)** | – | 0.871 | 0.999 | 0.973 |
| only:youtube | Naive Bayes | 1.000* | 0.648 | 1.000 |
| only:youtube | TF-IDF + LR | 0.244* | 0.000 | 0.001 |
| only:sms | Naive Bayes | 0.985 | 1.000* | 1.000 |
| only:sms | TF-IDF + LR | 0.245 | 0.504* | 0.041 |
| only:enron | Naive Bayes | 0.727 | 0.730 | 1.000* |
| only:enron | TF-IDF + LR | 0.037 | 0.002 | 0.699* |
| loo:youtube | Naive Bayes | 0.593 | 0.769* | 1.000* |
| loo:youtube | TF-IDF + LR | 0.032 | 0.082* | 0.691* |
| loo:sms | Naive Bayes | 0.775* | 0.711 | 1.000* |
| loo:sms | TF-IDF + LR | 0.271* | 0.001 | 0.694* |
| loo:enron | Naive Bayes | 1.000* | 0.999* | 1.000 |
| loo:enron | TF-IDF + LR | 0.295* | 0.189* | 0.005 |
| mix:100 | Naive Bayes | 0.734* | 0.628* | 1.000* |
| mix:100 | TF-IDF + LR | 0.000* | 0.000* | 0.000* |
| mix:500 | Naive Bayes | 0.806* | 0.832* | 1.000* |
| mix:500 | TF-IDF + LR | 0.070* | 0.000* | 0.000* |
| mix:1000 | Naive Bayes | 0.829* | 0.843* | 1.000* |
| mix:1000 | TF-IDF + LR | 0.179* | 0.000* | 0.002* |
| mix:full | Naive Bayes | 0.717* | 0.742* | 1.000* |
| mix:full | TF-IDF + LR | 0.261* | 0.074* | 0.701* |

**calibrated**

| Train | Model | youtube | sms | enron |
|---|---|---|---|---|
| **Jev (zero-shot)** | – | 0.871 | 0.999 | 0.973 |
| only:youtube | Naive Bayes | 0.952* | 0.369 | 0.930 |
| only:youtube | TF-IDF + LR | 0.950* | 0.145 | 0.371 |
| only:sms | Naive Bayes | 0.892 | 1.000* | 1.000 |
| only:sms | TF-IDF + LR | 0.981 | 1.000* | 0.670 |
| only:enron | Naive Bayes | 0.341 | 0.314 | 1.000* |
| only:enron | TF-IDF + LR | 0.817 | 0.588 | 1.000* |
| loo:youtube | Naive Bayes | 0.281 | 0.681* | 1.000* |
| loo:youtube | TF-IDF + LR | 0.528 | 0.947* | 1.000* |
| loo:sms | Naive Bayes | 0.319* | 0.245 | 1.000* |
| loo:sms | TF-IDF + LR | 0.850* | 0.328 | 1.000* |
| loo:enron | Naive Bayes | 0.883* | 1.000* | 0.971 |
| loo:enron | TF-IDF + LR | 0.995* | 1.000* | 0.483 |
| mix:100 | Naive Bayes | 0.123* | 0.113* | 0.871* |
| mix:100 | TF-IDF + LR | 0.140* | 0.000* | 0.002* |
| mix:500 | Naive Bayes | 0.486* | 0.617* | 0.999* |
| mix:500 | TF-IDF + LR | 0.925* | 0.630* | 0.680* |
| mix:1000 | Naive Bayes | 0.645* | 0.806* | 1.000* |
| mix:1000 | TF-IDF + LR | 0.998* | 0.839* | 0.858* |
| mix:full | Naive Bayes | 0.304* | 0.604* | 1.000* |
| mix:full | TF-IDF + LR | 0.902* | 0.940* | 1.000* |

## Realized risk among automated messages (target 0.05)

**raw**

| Train | Model | youtube | sms | enron |
|---|---|---|---|---|
| **Jev (zero-shot)** | – | 0.035 | 0.018 | 0.031 |
| only:youtube | Naive Bayes | 0.092* | 0.519 | 0.517 |
| only:youtube | TF-IDF + LR | 0.000* | 0.000 | 0.125 |
| only:sms | Naive Bayes | 0.400 | 0.017* | 0.480 |
| only:sms | TF-IDF + LR | 0.299 | 0.000* | 0.417 |
| only:enron | Naive Bayes | 0.480 | 0.588 | 0.011* |
| only:enron | TF-IDF + LR | 0.364 | 0.750 | 0.000* |
| loo:youtube | Naive Bayes | 0.431 | 0.232* | 0.012* |
| loo:youtube | TF-IDF + LR | 0.158 | 0.000* | 0.000* |
| loo:sms | Naive Bayes | 0.310* | 0.587 | 0.011* |
| loo:sms | TF-IDF + LR | 0.006* | 0.500 | 0.000* |
| loo:enron | Naive Bayes | 0.082* | 0.020* | 0.440 |
| loo:enron | TF-IDF + LR | 0.000* | 0.000* | 0.289 |
| mix:100 | Naive Bayes | 0.136* | 0.129* | 0.108* |
| mix:100 | TF-IDF + LR | 0.000* | 0.000* | 0.000* |
| mix:500 | Naive Bayes | 0.081* | 0.080* | 0.054* |
| mix:500 | TF-IDF + LR | 0.000* | 0.000* | 0.000* |
| mix:1000 | Naive Bayes | 0.078* | 0.082* | 0.035* |
| mix:1000 | TF-IDF + LR | 0.000* | 0.000* | 0.000* |
| mix:full | Naive Bayes | 0.242* | 0.242* | 0.011* |
| mix:full | TF-IDF + LR | 0.013* | 0.000* | 0.001* |

**calibrated**

| Train | Model | youtube | sms | enron |
|---|---|---|---|---|
| **Jev (zero-shot)** | – | 0.035 | 0.018 | 0.031 |
| only:youtube | Naive Bayes | 0.060* | 0.257 | 0.515 |
| only:youtube | TF-IDF + LR | 0.056* | 0.218 | 0.506 |
| only:sms | Naive Bayes | 0.444 | 0.020* | 0.485 |
| only:sms | TF-IDF + LR | 0.468 | 0.021* | 0.400 |
| only:enron | Naive Bayes | 0.302 | 0.587 | 0.011* |
| only:enron | TF-IDF + LR | 0.511 | 0.769 | 0.014* |
| loo:youtube | Naive Bayes | 0.328 | 0.032* | 0.020* |
| loo:youtube | TF-IDF + LR | 0.362 | 0.043* | 0.019* |
| loo:sms | Naive Bayes | 0.137* | 0.445 | 0.013* |
| loo:sms | TF-IDF + LR | 0.109* | 0.607 | 0.013* |
| loo:enron | Naive Bayes | 0.075* | 0.019* | 0.444 |
| loo:enron | TF-IDF + LR | 0.071* | 0.024* | 0.379 |
| mix:100 | Naive Bayes | 0.106* | 0.037* | 0.107* |
| mix:100 | TF-IDF + LR | 0.000* | 0.000* | 0.000* |
| mix:500 | Naive Bayes | 0.039* | 0.023* | 0.075* |
| mix:500 | TF-IDF + LR | 0.053* | 0.010* | 0.036* |
| mix:1000 | Naive Bayes | 0.034* | 0.026* | 0.053* |
| mix:1000 | TF-IDF + LR | 0.070* | 0.015* | 0.052* |
| mix:full | Naive Bayes | 0.098* | 0.035* | 0.019* |
| mix:full | TF-IDF + LR | 0.082* | 0.038* | 0.021* |

