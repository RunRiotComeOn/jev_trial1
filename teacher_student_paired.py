"""Paired bootstrap: Jev-taught student (Jev hard labels, full pool, raw) vs Jev on the same test messages.

Output: results/teacher_student_paired.md
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import data
from analyze import RESULTS, jev_records
from baselines import fit, jev_pool_labels, proba
from metrics import f1
from strong_models import STRONG_LABELS

STUDENTS = ["char_lr", "word_char_surface_lr", "ensemble"]
N_BOOT = 2000


def paired(y, a, b):
    rng = np.random.default_rng(0)
    d = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        d.append(f1(y[i], a[i]) - f1(y[i], b[i]))
    d = np.array(d)
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), (d > 0).mean()


def main():
    ds = {d: data.load(d) for d in data.DATASETS}
    pools = {d: df[df["split"] == "train"].reset_index(drop=True) for d, df in ds.items()}
    tests = {d: df[df["split"] == "test"] for d, df in ds.items()}
    labels = {d: (jev_pool_labels(d, pools[d]) >= 0.5).astype(float) for d in ds}
    L = ["# Jev-taught students vs Jev: paired bootstrap on the same test messages", "",
         "Students trained on Jev hard labels for the full pool (no human labels), raw variant. "
         f"{N_BOOT} paired resamples of the test set.", "",
         "| Student | Trained on | Test | Student F1 | Jev F1 | Difference [95% CI] | P(student better) |",
         "|---|---|---|---|---|---|---|"]
    for model in STUDENTS:
        both = fit(model, np.concatenate([pools[d]["text"].to_numpy(object) for d in ds]),
                   np.concatenate([labels[d] for d in ds]))
        for setting in ["same dataset", "both pools"]:
            for d in ds:
                m = both if setting == "both pools" else fit(model, pools[d]["text"].to_numpy(object), labels[d])
                y = tests[d]["label"].to_numpy()
                ps = proba(m, tests[d]["text"])
                pj = jev_records(d).set_index("id")["p_spam"].loc[tests[d]["id"]].to_numpy()
                mu, lo, hi, pr = paired(y, ps, pj)
                L.append(f"| {STRONG_LABELS[model]} | {setting} | {d} | {f1(y, ps):.3f} | {f1(y, pj):.3f} "
                         f"| {mu:+.3f} [{lo:+.3f}, {hi:+.3f}] | {pr:.2f} |")
    (RESULTS / "teacher_student_paired.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
