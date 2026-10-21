"""Generalization of the lightweight models under distribution shift, against zero-shot Jev.

G1 cross-dataset: train on one dataset's full training pool, test on the other dataset's test set.
G2 leave-one-video-out (YouTube): train on all comments from 4 videos, test on the 5th video's test-split
   comments; the in-distribution reference trains on the usual pool (which includes that video).

Both settings score every model on the same test messages as Jev, using cached Jev predictions only.
Coverage thresholds are chosen on unlabeled target-domain predictions (the target's training-split messages).

Outputs: results/generalization.csv, results/generalization.md, results/figures/fig4_generalization.png
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
from analyze import FIGURES, MODEL_LABEL, N_BOOT, RESULTS, fmt, jev_records
from baselines import MODELS, SEEDS, fit, platt, proba, split_cal
from metrics import all_metrics, coverage_threshold

METRICS = ["f1", "brier", "ece", "coverage", "auto_risk"]


def baseline_scores(train: pd.DataFrame, test: pd.DataFrame, unlabeled: pd.DataFrame):
    """Raw (all labels) and calibrated (80/20 Platt, mean over seeds) metrics for each model."""
    X, y = train["text"].to_numpy(object), train["label"].to_numpy(float)
    rows = []
    for model_name in MODELS:
        m = fit(model_name, X, y)
        t = coverage_threshold(proba(m, unlabeled["text"]))
        rows.append({"model": model_name, "variant": "raw", **all_metrics(test["label"], proba(m, test["text"]), t)})
        cal_rows = []
        for seed in SEEDS:
            fit_pos, cal_pos = split_cal(y.astype(int), np.random.default_rng(seed))
            m = fit(model_name, X[fit_pos], y[fit_pos])
            cal = platt(proba(m, X[cal_pos]), y[cal_pos]) or (lambda p: p)
            t = coverage_threshold(cal(proba(m, unlabeled["text"])))
            cal_rows.append(all_metrics(test["label"], cal(proba(m, test["text"])), t))
        rows.append({"model": model_name, "variant": "calibrated", **pd.DataFrame(cal_rows).mean().to_dict()})
    return rows


def jev_scores(dataset: str, test: pd.DataFrame, unlabeled: pd.DataFrame):
    p = jev_records(dataset).set_index("id")["p_spam"]
    y, pt = test["label"].to_numpy(), p.loc[test["id"]].to_numpy()
    t = coverage_threshold(p.loc[unlabeled["id"]])
    point = all_metrics(y, pt, t)
    rng = np.random.default_rng(0)
    boots = pd.DataFrame([all_metrics(y[i], pt[i], t) for i in (rng.integers(0, len(y), len(y)) for _ in range(N_BOOT))])
    return {"model": "jev", "variant": "zero-shot", **point,
            "f1_lo": boots["f1"].quantile(0.025), "f1_hi": boots["f1"].quantile(0.975)}


def run():
    ds = {d: data.load(d) for d in data.DATASETS}
    pool = {d: df[df["split"] == "train"] for d, df in ds.items()}
    test = {d: df[df["split"] == "test"] for d, df in ds.items()}
    rows = []

    # G1: cross-dataset.
    for tgt in data.DATASETS:
        for src in data.DATASETS:
            setting = "in-domain" if src == tgt else "cross-dataset"
            for r in baseline_scores(pool[src], test[tgt], pool[tgt]):
                rows.append({"experiment": "G1", "target": tgt, "train_on": src, "setting": setting, **r})
        rows.append({"experiment": "G1", "target": tgt, "train_on": "-", "setting": "zero-shot",
                     **jev_scores(tgt, test[tgt], pool[tgt])})
        print(f"G1 target={tgt} done")

    # G2: leave-one-video-out, plus all videos pooled (each comment scored by the model that never saw its video).
    yt = ds["youtube"]
    videos = sorted(yt["group"].unique())
    lovo_preds = {}
    for v in videos + ["all"]:
        held = yt["group"] == v if v != "all" else yt["group"].notna()
        tst = test["youtube"][held[test["youtube"].index]]
        unl = pool["youtube"][held[pool["youtube"].index]]
        if v == "all":
            # Pooled unseen-video score: concatenate per-video predictions made without that video.
            for model_name, recs in lovo_preds.items():
                y = np.concatenate([r[0] for r in recs])
                p = np.concatenate([r[1] for r in recs])
                t = coverage_threshold(np.concatenate([r[2] for r in recs]))
                rows.append({"experiment": "G2", "target": "all videos", "train_on": "other videos",
                             "setting": "unseen video", "model": model_name, "variant": "raw", **all_metrics(y, p, t)})
        else:
            other = yt[yt["group"] != v]
            for r in baseline_scores(other, tst, unl):
                rows.append({"experiment": "G2", "target": v, "train_on": "other videos", "setting": "unseen video", **r})
            for model_name in MODELS:
                m = fit(model_name, other["text"].to_numpy(object), other["label"].to_numpy(float))
                lovo_preds.setdefault(model_name, []).append(
                    (tst["label"].to_numpy(), proba(m, tst["text"]), proba(m, unl["text"])))
        for r in baseline_scores(pool["youtube"], tst, unl):
            rows.append({"experiment": "G2", "target": "all videos" if v == "all" else v, "train_on": "pool",
                         "setting": "in-distribution", **r})
        rows.append({"experiment": "G2", "target": "all videos" if v == "all" else v, "train_on": "-",
                     "setting": "zero-shot", **jev_scores("youtube", tst, unl)})
        print(f"G2 video={v} done")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "generalization.csv", index=False)
    return df


def label(r):
    return "Jev (zero-shot)" if r.model == "jev" else f"{MODEL_LABEL[r.model]} ({r.variant})"


def figure(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    colors = {"in-domain": "#4c72b0", "in-distribution": "#4c72b0", "cross-dataset": "#dd8452",
              "unseen video": "#dd8452", "zero-shot": "#333333"}
    panels = [("G1", "youtube", "test on YouTube"), ("G1", "sms", "test on SMS"), ("G2", "all videos", "YouTube, unseen video")]
    for ax, (exp, tgt, title) in zip(axes, panels):
        g = df[(df["experiment"] == exp) & (df["target"] == tgt) & (df["variant"] != "calibrated")]
        names, vals, cols = [], [], []
        for r in g.itertuples():
            names.append("Jev" if r.model == "jev" else f"{MODEL_LABEL[r.model]}\n{r.setting}")
            vals.append(r.f1)
            cols.append(colors[r.setting])
        ax.barh(range(len(vals)), vals, color=cols)
        for i, v in enumerate(vals):
            ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)
        ax.set_yticks(range(len(vals)), names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.08)
        ax.set_xlabel("F1 (raw variants)")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_generalization.png", dpi=150)
    plt.close(fig)


def report(df):
    L = ["# Generalization under distribution shift", "",
         "Same test messages for every row. Calibrated rows: mean over 10 seeds. "
         "Coverage threshold set on unlabeled target-domain messages.", ""]
    head = "| Train on | Setting | Model | F1 | Brier | ECE | Coverage @5% | Realized risk |\n|---|---|---|---|---|---|---|---|"
    for tgt in data.DATASETS:
        L += [f"## G1 cross-dataset: test on {tgt}", "", head]
        for r in df[(df["experiment"] == "G1") & (df["target"] == tgt)].itertuples():
            f1 = f"{fmt(r.f1)} [{fmt(r.f1_lo)}, {fmt(r.f1_hi)}]" if r.model == "jev" else fmt(r.f1)
            L.append(f"| {r.train_on} | {r.setting} | {label(r)} | {f1} | {fmt(r.brier)} | {fmt(r.ece)} "
                     f"| {fmt(r.coverage)} | {fmt(r.auto_risk)} |")
        L.append("")
    g2 = df[df["experiment"] == "G2"]
    L += ["## G2 leave-one-video-out (YouTube), F1 of raw variants", "",
          "| Video | " + " | ".join(f"{MODEL_LABEL[m]} in-dist → unseen" for m in MODELS) + " | Jev |",
          "|---|" + "---|" * (len(MODELS) + 1)]
    for v in list(g2["target"].unique()):
        g = g2[(g2["target"] == v) & (g2["variant"] != "calibrated")]
        cells = []
        for m in MODELS:
            a = g[(g["model"] == m) & (g["setting"] == "in-distribution")]["f1"].iloc[0]
            b = g[(g["model"] == m) & (g["setting"] == "unseen video")]["f1"].iloc[0]
            cells.append(f"{a:.3f} → {b:.3f} ({b - a:+.3f})")
        j = g[g["model"] == "jev"].iloc[0]
        L.append(f"| {v} | " + " | ".join(cells) + f" | {j.f1:.3f} [{j.f1_lo:.3f}, {j.f1_hi:.3f}] |")
    L += ["", "## G2 all videos pooled", "", head]
    for r in g2[g2["target"] == "all videos"].itertuples():
        L.append(f"| {r.train_on} | {r.setting} | {label(r)} | {fmt(r.f1)} | {fmt(r.brier)} | {fmt(r.ece)} "
                 f"| {fmt(r.coverage)} | {fmt(r.auto_risk)} |")
    L += ["", "![Figure 4](figures/fig4_generalization.png)", ""]
    (RESULTS / "generalization.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    df = run()
    figure(df)
    report(df)
    print((RESULTS / "generalization.md").read_text())
