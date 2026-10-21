# Jev-taught students vs Jev: paired bootstrap on the same test messages

Students trained on Jev hard labels for the full pool (no human labels), raw variant. 2000 paired resamples of the test set.

| Student | Trained on | Test | Student F1 | Jev F1 | Difference [95% CI] | P(student better) |
|---|---|---|---|---|---|---|
| Char n-gram LR | same dataset | youtube | 0.950 | 0.938 | +0.012 [-0.008, +0.033] | 0.87 |
| Char n-gram LR | same dataset | sms | 0.946 | 0.927 | +0.019 [-0.005, +0.044] | 0.94 |
| Char n-gram LR | both pools | youtube | 0.951 | 0.938 | +0.013 [-0.008, +0.034] | 0.87 |
| Char n-gram LR | both pools | sms | 0.934 | 0.927 | +0.007 [-0.018, +0.032] | 0.70 |
| Word+char+surface LR | same dataset | youtube | 0.943 | 0.938 | +0.005 [-0.015, +0.025] | 0.69 |
| Word+char+surface LR | same dataset | sms | 0.937 | 0.927 | +0.010 [-0.015, +0.035] | 0.78 |
| Word+char+surface LR | both pools | youtube | 0.956 | 0.938 | +0.018 [-0.002, +0.039] | 0.96 |
| Word+char+surface LR | both pools | sms | 0.935 | 0.927 | +0.008 [-0.015, +0.032] | 0.76 |
| Ensemble | same dataset | youtube | 0.948 | 0.938 | +0.011 [-0.009, +0.030] | 0.85 |
| Ensemble | same dataset | sms | 0.946 | 0.927 | +0.019 [-0.006, +0.043] | 0.93 |
| Ensemble | both pools | youtube | 0.945 | 0.938 | +0.007 [-0.014, +0.028] | 0.74 |
| Ensemble | both pools | sms | 0.939 | 0.927 | +0.012 [-0.010, +0.034] | 0.85 |
