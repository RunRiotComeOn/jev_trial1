"""E5: do the lightweight models generalize across datasets? (Jev needs no training data at all.)

Training configurations, all with true labels:
  only:A     the full training pool of dataset A
  loo:B      the full training pools of every dataset except B (leave-one-out; B is never seen)
  mix:k      k labels from each dataset's pool (stratified within each dataset); k = "full" uses every pool

Every configuration is tested on every dataset's test set and compared with Jev zero-shot on that test set.
As in baselines.py there is a raw variant (fit on all sampled labels) and a calibrated variant (Platt scaling on
20% of the sample, so calibrated on the training data, not the target). The coverage threshold is chosen on the
target dataset's unlabeled training pool, as it would be in deployment.

Outputs: results/transfer.csv (one row per config, model, variant, target; mean and sd over seeds) and
results/transfer.md. Needs only the processed datasets and, for the Jev rows, jev_cache.jsonl.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import analyze
import baselines
import data
from metrics import all_metrics, coverage_threshold

RESULTS = Path(__file__).parent / "results"
MIX_BUDGETS = [100, 500, 1000, None]  # per dataset; None = full pool
METRIC_COLS = ["f1", "brier", "ece", "coverage", "auto_risk"]


def splits(dataset, _cache={}):
    if dataset in _cache:
        return _cache[dataset]
    df = data.load(dataset)
    pool, test = df[df["split"] == "train"].reset_index(drop=True), df[df["split"] == "test"].reset_index(drop=True)
    _cache[dataset] = (pool["text"].to_numpy(object), pool["label"].to_numpy(int),
                       test["text"].to_numpy(object), test["label"].to_numpy(int))
    return _cache[dataset]


def configs(datasets):
    out = [("only", d, {d: None}) for d in datasets]
    if len(datasets) > 2:
        out += [("loo", d, {s: None for s in datasets if s != d}) for d in datasets]
    out += [("mix", "full" if k is None else str(k), {s: k for s in datasets}) for k in MIX_BUDGETS]
    return out


def draw(kind, name, budgets, seed, datasets):
    """Train both models (raw and calibrated) on one draw of the configuration; evaluate on every test set."""
    texts, labels = [], []
    for i, (d, k) in enumerate(sorted(budgets.items())):
        pool_x, pool_y, _, _ = splits(d)
        idx = baselines.sample_pool(pool_y, k, np.random.default_rng([seed, 0 if k is None else k, i]))
        texts.append(pool_x[idx])
        labels.append(pool_y[idx])
    X, y = np.concatenate(texts), np.concatenate(labels).astype(float)
    rng = np.random.default_rng([seed, 1])
    order = rng.permutation(len(y))
    X, y = X[order], y[order]
    fit_pos, cal_pos = baselines.split_cal(y.astype(int), rng)

    rows = []
    for model_name in baselines.MODELS:
        raw = baselines.fit(model_name, X, y)
        m = baselines.fit(model_name, X[fit_pos], y[fit_pos])
        cal = baselines.platt(baselines.proba(m, X[cal_pos]), y[cal_pos]) if len(cal_pos) else None
        variants = {"raw": lambda t: baselines.proba(raw, t),
                    "calibrated": lambda t: (cal or (lambda p: p))(baselines.proba(m, t))}
        for variant, predict in variants.items():
            for target in datasets:
                pool_x, _, test_x, test_y = splits(target)
                t = coverage_threshold(predict(pool_x))
                rows.append({"kind": kind, "config": name, "n_train": len(y), "seed": seed, "model": model_name,
                             "variant": variant, "target": target, "in_domain": target in budgets,
                             **all_metrics(test_y, predict(test_x), t)})
    return rows


def jev_rows(datasets):
    rows = {}
    for d in datasets:
        test = data.load(d)
        j = analyze.jev_summary(d, list(test.loc[test["split"] == "test", "id"]))
        if j:
            rows[d] = j
    return rows


def report(summary, jev, datasets):
    def cell(g, metric):
        if g.empty:
            return "–"
        r = g.iloc[0]
        mark = "" if not r["in_domain"] else "*"
        return f"{r[f'{metric}_mean']:.3f}{mark}"

    L = ["# E5: cross-dataset generalization of the lightweight models", "",
         "Rows are training configurations, columns are test sets. `only:A` = full training pool of A; "
         "`loo:B` = every pool except B; `mix:k` = k labels from each dataset (`full` = all pools). "
         "`*` = the test dataset's own training data was used (in-domain). Baselines: mean over seeds; "
         "Jev: zero-shot, never trained, with bootstrap 95% interval.", ""]
    for metric, title in [("f1", "F1"), ("brier", "Brier (lower is better)"), ("coverage", "Coverage @5% risk"),
                          ("auto_risk", "Realized risk among automated messages (target 0.05)")]:
        L += [f"## {title}", ""]
        for variant in ["raw", "calibrated"]:
            L += [f"**{variant}**", "", "| Train | Model | " + " | ".join(datasets) + " |",
                  "|---|---|" + "---|" * len(datasets)]
            if jev:
                cells = []
                for d in datasets:
                    if d not in jev:
                        cells.append("–")
                    elif metric in ("f1", "brier"):
                        lo, hi = jev[d]["ci"][metric]
                        cells.append(f"{jev[d]['point'][metric]:.3f} [{lo:.3f}, {hi:.3f}]")
                    else:
                        cells.append(f"{jev[d]['point'][metric]:.3f}")
                L.append("| **Jev (zero-shot)** | – | " + " | ".join(cells) + " |")
            s = summary[summary["variant"] == variant]
            for (kind, name), g in s.groupby(["kind", "config"], sort=False):
                for model_name, gm in g.groupby("model", sort=False):
                    L.append(f"| {kind}:{name} | {analyze.MODEL_LABEL[model_name]} | "
                             + " | ".join(cell(gm[gm["target"] == d], metric) for d in datasets) + " |")
            L.append("")
    (RESULTS / "transfer.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=data.DATASETS, choices=data.DATASETS)
    args = ap.parse_args()

    tasks = [(kind, name, budgets, seed) for kind, name, budgets in configs(args.datasets)
             for seed in baselines.SEEDS]
    out = Parallel(n_jobs=-1)(delayed(draw)(*t, args.datasets) for t in tasks)
    runs = pd.DataFrame([r for rows in out for r in rows])
    summary = (runs.groupby(["kind", "config", "model", "variant", "target"], sort=False)
               .agg(n_train=("n_train", "mean"), in_domain=("in_domain", "first"),
                    **{f"{m}_mean": (m, "mean") for m in METRIC_COLS}, **{f"{m}_std": (m, "std") for m in METRIC_COLS})
               .reset_index())
    RESULTS.mkdir(exist_ok=True)
    summary.to_csv(RESULTS / "transfer.csv", index=False)
    report(summary, jev_rows(args.datasets), args.datasets)
    print((RESULTS / "transfer.md").read_text())


if __name__ == "__main__":
    main()
