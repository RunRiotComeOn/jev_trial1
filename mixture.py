"""Train the lightweight models on mixtures of both datasets and test on each dataset's test set.

Grid: YouTube share f of the training labels x total labels N, 10 draws each, raw and calibrated variants.
Each draw is stratified by label within each dataset. Infeasible cells (more YouTube labels than the pool
holds) are skipped. "full" = both training pools combined (YouTube share 26%).

Ablation: for each draw, the same model trained only on the target dataset's part of the mix, to show
whether the other dataset's labels help or hurt on that target.

Outputs: results/mixture.csv, results/mixture.md, results/figures/fig5_mixture.png
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
from analyze import FIGURES, MODEL_LABEL, RESULTS, jev_summary
from baselines import MODELS, SEEDS, fit, platt, proba, sample_pool, split_cal
from metrics import all_metrics, coverage_threshold

SHARES = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]  # YouTube share of training labels
TOTALS = [50, 100, 200, 500, 1000, 2000, "full"]


def draw(pools, n_yt, n_sms, rng):
    parts = []
    for d, k in [("youtube", n_yt), ("sms", n_sms)]:
        if k:
            p = pools[d]
            parts.append(p.iloc[sample_pool(p["label"].to_numpy(), k, rng)])
    return pd.concat(parts, ignore_index=True)


def train_variants(model_name, train, rng):
    """Raw model on all of `train`, and calibrated model (fit 80%, Platt on 20%) as prediction functions."""
    X, y = train["text"].to_numpy(object), train["label"].to_numpy(float)
    raw = fit(model_name, X, y)
    fit_pos, cal_pos = split_cal(y.astype(int), rng)
    m = fit(model_name, X[fit_pos], y[fit_pos])
    cal = (platt(proba(m, X[cal_pos]), y[cal_pos]) if len(cal_pos) else None) or (lambda p: p)
    return {"raw": lambda t: proba(raw, t), "calibrated": lambda t: cal(proba(m, t))}


def run():
    ds = {d: data.load(d) for d in data.DATASETS}
    pools = {d: df[df["split"] == "train"].reset_index(drop=True) for d, df in ds.items()}
    tests = {d: df[df["split"] == "test"] for d, df in ds.items()}
    rows = []
    for total in TOTALS:
        shares = [len(pools["youtube"]) / sum(map(len, pools.values()))] if total == "full" else SHARES
        for f in shares:
            if total == "full":
                n_yt, n_sms = len(pools["youtube"]), len(pools["sms"])
            else:
                n_yt = round(f * total)
                n_sms = total - n_yt
                if n_yt > len(pools["youtube"]) or n_sms > len(pools["sms"]):
                    continue
            for seed in SEEDS:
                rng = np.random.default_rng([seed, 7, n_yt, n_sms])
                train = draw(pools, n_yt, n_sms, rng)
                for model_name in MODELS:
                    mixed = train_variants(model_name, train, rng)
                    for tgt in data.DATASETS:
                        own = pd.Series([tgt == "youtube"] * n_yt + [tgt == "sms"] * n_sms)
                        single = train_variants(model_name, train[own.to_numpy()], rng) \
                            if 0 < own.sum() < len(train) and train[own.to_numpy()]["label"].nunique() == 2 else None
                        for setting, preds in [("mixed", mixed), ("target-only part", single)]:
                            if preds is None:
                                continue
                            for variant, predict in preds.items():
                                t = coverage_threshold(predict(pools[tgt]["text"]))
                                rows.append({"total": str(total), "n": n_yt + n_sms, "yt_share": round(f, 2),
                                             "n_youtube": n_yt, "n_sms": n_sms, "seed": seed, "model": model_name,
                                             "variant": variant, "setting": setting, "test": tgt,
                                             **all_metrics(tests[tgt]["label"], predict(tests[tgt]["text"]), t)})
            print(f"N={total} f={f:.2f} done")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "mixture.csv", index=False)
    return df


def summarize(df):
    keys = ["total", "n", "yt_share", "n_youtube", "n_sms", "model", "variant", "setting", "test"]
    return df.groupby(keys, sort=False).agg(**{f"{m}": (m, "mean") for m in ["f1", "brier", "ece", "coverage", "auto_risk"]},
                                            f1_sd=("f1", "std")).reset_index()


def grid(s, model, test, metric="f1", variant="raw", setting="mixed"):
    g = s[(s["model"] == model) & (s["test"] == test) & (s["variant"] == variant) & (s["setting"] == setting)]
    return g.pivot_table(index="yt_share", columns="total", values=metric, sort=False).reindex(
        columns=[str(t) for t in TOTALS]).sort_index()


def figure(s, jev):
    fig, axes = plt.subplots(len(MODELS), 3, figsize=(16, 4.2 * len(MODELS)), squeeze=False)
    for i, model in enumerate(MODELS):
        g = {t: grid(s, model, t) for t in data.DATASETS}
        panels = [("youtube", g["youtube"]), ("sms", g["sms"]), ("min of both", np.minimum(g["youtube"], g["sms"]))]
        for j, (name, mat) in enumerate(panels):
            ax = axes[i][j]
            im = ax.imshow(mat.to_numpy(float), vmin=0, vmax=1, cmap="viridis", aspect="auto")
            for (r, c), v in np.ndenumerate(mat.to_numpy(float)):
                if not np.isnan(v):
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7, color="w" if v < 0.6 else "k")
            ax.set_xticks(range(mat.shape[1]), mat.columns)
            ax.set_yticks(range(mat.shape[0]), [f"{x:.2f}" for x in mat.index])
            ax.set_xlabel("total labels N")
            ax.set_ylabel("YouTube share of labels")
            ref = (f"Jev {jev[name]['point']['f1']:.2f}" if name in jev
                   else f"Jev min {min(v['point']['f1'] for v in jev.values()):.2f}")
            ax.set_title(f"{MODEL_LABEL[model]}: F1 on {name} test ({ref})", fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.6, label="F1 (raw)")
    fig.savefig(FIGURES / "fig5_mixture.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def table(mat, digits=2):
    cols = list(mat.columns)
    L = ["| YouTube share \\ N | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for share, row in mat.iterrows():
        L.append(f"| {share:.2f} | " + " | ".join("–" if np.isnan(v) else f"{v:.{digits}f}" for v in row) + " |")
    return L


def report(s, jev):
    L = ["# Mixed-dataset training", "",
         "Lightweight models trained on a mix of YouTube and SMS labels (YouTube share × total N, mean of 10 draws), "
         "tested on each dataset's test set. `full` = both pools combined (YouTube share 0.26). "
         f"Jev zero-shot F1: YouTube {jev['youtube']['point']['f1']:.3f}, SMS {jev['sms']['point']['f1']:.3f}.", ""]
    for model in MODELS:
        for test in data.DATASETS:
            L += [f"## {MODEL_LABEL[model]}: F1 on {test} test (raw)", ""] + table(grid(s, model, test)) + [""]
        L += [f"## {MODEL_LABEL[model]}: Brier on each test (calibrated)", ""]
        for test in data.DATASETS:
            L += [f"{test}:", ""] + table(grid(s, model, test, "brier", "calibrated"), 3) + [""]

    L += ["## Does the other dataset help? Mixed minus target-only part (F1, raw)", "",
          "Positive = adding the other dataset's labels helped on this test set.", ""]
    for model in MODELS:
        for test in data.DATASETS:
            diff = grid(s, model, test) - grid(s, model, test, setting="target-only part")
            L += [f"{MODEL_LABEL[model]}, {test} test:", ""] + table(diff.dropna(how="all"), 3) + [""]

    L += ["## Best mix per total (by the lower of the two F1s, raw)", "",
          "| Model | N | Best YouTube share | F1 YouTube | F1 SMS | Within Jev's interval on both? |", "|---|---|---|---|---|---|"]
    for model in MODELS:
        yt, sms = grid(s, model, "youtube"), grid(s, model, "sms")
        worst = np.minimum(yt, sms)
        for total in worst.columns:
            col = worst[total].dropna()
            if col.empty:
                continue
            f = col.idxmax()
            ok = yt.loc[f, total] >= jev["youtube"]["ci"]["f1"][0] and sms.loc[f, total] >= jev["sms"]["ci"]["f1"][0]
            L.append(f"| {MODEL_LABEL[model]} | {total} | {f:.2f} | {yt.loc[f, total]:.3f} | {sms.loc[f, total]:.3f} "
                     f"| {'yes' if ok else 'no'} |")
    L += ["", "![Figure 5](figures/fig5_mixture.png)", ""]
    (RESULTS / "mixture.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    df = run()
    s = summarize(df)
    jev = {}
    for d in data.DATASETS:
        t = data.load(d)
        jev[d] = jev_summary(d, list(t[t["split"] == "test"]["id"]))
    figure(s, jev)
    report(s, jev)
    print((RESULTS / "mixture.md").read_text())
