"""Compare stronger classic (non-pretrained) models with the original baselines and Jev.

A. Label-budget curve on true labels (same draws as baselines.py), raw and calibrated variants.
B. Distribution shift: cross-dataset, leave-one-video-out (pooled over the 5 YouTube videos), and both
   training pools combined. Coverage thresholds are set on unlabeled target-domain messages.

Outputs: results/strong_curve.csv, results/strong_shift.csv, results/strong.md,
         results/figures/fig6_strong_curve.png, results/figures/fig7_strong_shift.png
"""

import os
import time

# One thread per worker: LightGBM/BLAS threads x joblib workers otherwise oversubscribe the CPU.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import data
from analyze import FIGURES, MODEL_LABEL, RESULTS, jev_summary
from baselines import BUDGETS, SEEDS, fit, platt, proba, sample_pool, split_cal
from generalize import baseline_scores, jev_scores
from metrics import all_metrics, coverage_threshold
from strong_models import STRONG_LABELS, STRONG_MODELS

MODELS = ["naive_bayes", "tfidf_logreg", *STRONG_MODELS]
LABELS = {**MODEL_LABEL, **STRONG_LABELS}
N_JOBS = 4


# ---------- A. label-budget curve ----------

def curve_job(dataset, n, seed):
    t0 = time.perf_counter()
    df = data.load(dataset)
    pool, test = df[df["split"] == "train"].reset_index(drop=True), df[df["split"] == "test"]
    texts, y = pool["text"].to_numpy(object), pool["label"].to_numpy(float)
    rng = np.random.default_rng([seed, 0 if n is None else n])  # same draws as baselines.py
    idx = sample_pool(y.astype(int), n, rng)
    fit_pos, cal_pos = split_cal(y[idx].astype(int), rng)
    rows = []
    for model_name in MODELS:
        meta = {"dataset": dataset, "budget": "full" if n is None else n, "n": len(idx), "seed": seed, "model": model_name}
        m = fit(model_name, texts[idx], y[idx])
        rows.append({**meta, "variant": "raw", **all_metrics(
            test["label"], proba(m, test["text"]), coverage_threshold(proba(m, texts)))})
        m = fit(model_name, texts[idx[fit_pos]], y[idx[fit_pos]])
        cal = (platt(proba(m, texts[idx[cal_pos]]), y[idx[cal_pos]]) if len(cal_pos) else None) or (lambda p: p)
        rows.append({**meta, "variant": "calibrated", **all_metrics(
            test["label"], cal(proba(m, test["text"])), coverage_threshold(cal(proba(m, texts))))})
    print(f"curve {dataset} n={'full' if n is None else n} seed={seed} {time.perf_counter() - t0:.0f}s", flush=True)
    return rows


def run_curve():
    jobs = [(d, n, s) for d in data.DATASETS for n in BUDGETS for s in SEEDS]
    out = Parallel(n_jobs=N_JOBS, verbose=0)(delayed(curve_job)(*j) for j in jobs)
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_csv(RESULTS / "strong_curve.csv", index=False)
    return df


# ---------- B. distribution shift ----------

def shift_tasks():
    ds = {d: data.load(d) for d in data.DATASETS}
    pool = {d: df[df["split"] == "train"] for d, df in ds.items()}
    test = {d: df[df["split"] == "test"] for d, df in ds.items()}
    combined = pd.concat(pool.values(), ignore_index=True)
    tasks = []
    for tgt in data.DATASETS:
        src = "sms" if tgt == "youtube" else "youtube"
        tasks.append(("in-domain", tgt, pool[tgt], test[tgt], pool[tgt]))
        tasks.append(("cross-dataset", tgt, pool[src], test[tgt], pool[tgt]))
        tasks.append(("both pools", tgt, combined, test[tgt], pool[tgt]))
    yt = ds["youtube"]
    for v in sorted(yt["group"].unique()):
        held = yt["group"] == v
        tasks.append((f"unseen video:{v}", "youtube", yt[~held],
                      test["youtube"][held[test["youtube"].index]], pool["youtube"][held[pool["youtube"].index]]))
    return tasks


def shift_job(task, model_name):
    setting, tgt, train, tst, unl = task
    t0 = time.perf_counter()
    rows = [{"setting": setting, "target": tgt, "n_test": len(tst), **r}
            for r in baseline_scores(train, tst, unl, models=[model_name])]
    raw = fit(model_name, train["text"].to_numpy(object), train["label"].to_numpy(float))
    preds = (tst["label"].to_numpy(), proba(raw, tst["text"]), proba(raw, unl["text"]))
    print(f"shift {setting} -> {tgt} {model_name} {time.perf_counter() - t0:.0f}s", flush=True)
    return rows, (setting, model_name, preds)


def run_shift():
    tasks = shift_tasks()
    out = Parallel(n_jobs=N_JOBS, verbose=0)(delayed(shift_job)(t, m) for t in tasks for m in MODELS)
    rows = [r for rs, _ in out for r in rs]
    # Pool the five unseen-video runs per model (raw variant): every comment scored by a model that never saw its video.
    for model_name in MODELS:
        parts = [p for _, (s, m, p) in out if m == model_name and s.startswith("unseen video:")]
        y, p, u = (np.concatenate([x[i] for x in parts]) for i in range(3))
        rows.append({"setting": "unseen video", "target": "youtube", "n_test": len(y), "model": model_name,
                     "variant": "raw", **all_metrics(y, p, coverage_threshold(u))})
    yt = data.load("youtube")
    for setting, tgt, train, tst, unl in [t for t in tasks if t[0] in ("in-domain", "cross-dataset")] + [
            ("unseen video", "youtube", None, yt[yt["split"] == "test"], yt[yt["split"] == "train"])]:
        if setting == "cross-dataset":
            continue
        rows.append({"setting": "zero-shot" if setting == "in-domain" else "zero-shot (unseen video)",
                     "target": tgt, "n_test": len(tst), **jev_scores(tgt, tst, unl)})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "strong_shift.csv", index=False)
    return df


# ---------- report ----------

def curve_summary(curve):
    return (curve.groupby(["dataset", "model", "variant", "budget"], sort=False)
            .agg(n=("n", "mean"), f1=("f1", "mean"), f1_sd=("f1", "std"), brier=("brier", "mean"),
                 ece=("ece", "mean"), coverage=("coverage", "mean"), auto_risk=("auto_risk", "mean"))
            .reset_index())


def figures(cs, shift, jev):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    cmap = plt.get_cmap("tab10")
    for ax, d in zip(axes, data.DATASETS):
        for i, m in enumerate(MODELS):
            g = cs[(cs["dataset"] == d) & (cs["model"] == m) & (cs["variant"] == "raw")].sort_values("n")
            ax.plot(g["n"], g["f1"], "--" if m in MODEL_LABEL else "-", marker="o", ms=3, color=cmap(i), label=LABELS[m])
        ax.axhspan(*jev[d]["ci"]["f1"], color="k", alpha=0.12)
        ax.axhline(jev[d]["point"]["f1"], color="k", lw=1.5, label="Jev (zero-shot)")
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_xlabel("number of labels")
        ax.set_ylabel("F1 (test)")
        ax.set_title(f"{d}: F1 vs labels (dashed = original baselines)")
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig6_strong_curve.png", dpi=150)

    panels = [("in-domain", "youtube"), ("unseen video", "youtube"), ("cross-dataset", "youtube"),
              ("in-domain", "sms"), ("cross-dataset", "sms"), ("both pools", "youtube"), ("both pools", "sms")]
    fig, ax = plt.subplots(figsize=(13, 4.8))
    width = 0.8 / len(MODELS)
    for i, m in enumerate(MODELS):
        vals = [shift[(shift["setting"] == s) & (shift["target"] == t) & (shift["model"] == m)
                      & (shift["variant"] == "raw")]["f1"].iloc[0] for s, t in panels]
        ax.bar(np.arange(len(panels)) + (i - len(MODELS) / 2 + 0.5) * width, vals, width, color=cmap(i), label=LABELS[m])
    for k, (s, t) in enumerate(panels):
        ax.hlines(jev[t]["point"]["f1"], k - 0.45, k + 0.45, color="k", lw=2, label="Jev" if k == 0 else None)
    ax.set_xticks(range(len(panels)), [f"{s}\n(test: {t})" for s, t in panels], fontsize=8)
    ax.set_ylabel("F1 (raw)")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(FIGURES / "fig7_strong_shift.png", dpi=150)
    plt.close("all")


def report(cs, shift, jev):
    L = ["# Stronger classic models vs Jev", "",
         "Non-pretrained models with configurations fixed in advance (no tuning on test data). "
         "Original baselines: Naive Bayes, TF-IDF + LR. Curves: mean of 10 draws.", ""]
    for d in data.DATASETS:
        lo, hi = jev[d]["ci"]["f1"]
        L += [f"## {d}: F1 vs labels (raw); Jev {jev[d]['point']['f1']:.3f} [{lo:.3f}, {hi:.3f}]", "",
              "| Model | " + " | ".join(str(b) for b in ["10", "20", "50", "100", "200", "500", "full"])
              + " | n* to match Jev | n to beat Jev's upper bound |", "|---|" + "---|" * 9]
        for m in MODELS:
            g = cs[(cs["dataset"] == d) & (cs["model"] == m) & (cs["variant"] == "raw")].sort_values("n")
            match = g[g["f1"] >= lo]
            beat = g[g["f1"] > hi]
            L.append(f"| {LABELS[m]} | " + " | ".join(f"{v:.3f}" for v in g["f1"])
                     + f" | {int(match['n'].iloc[0]) if len(match) else '>pool'} | {int(beat['n'].iloc[0]) if len(beat) else '–'} |")
        L.append("")
    L += ["## Full pool, all metrics", "",
          "| Dataset | Model | Variant | F1 | Brier | ECE | Coverage @5% | Realized risk |", "|---|---|---|---|---|---|---|---|"]
    for d in data.DATASETS:
        j = jev[d]["point"]
        L.append(f"| {d} | Jev (zero-shot) | – | {j['f1']:.3f} | {j['brier']:.3f} | {j['ece']:.3f} | {j['coverage']:.3f} | {j['auto_risk']:.3f} |")
        for m in MODELS:
            for v in ["raw", "calibrated"]:
                r = cs[(cs["dataset"] == d) & (cs["model"] == m) & (cs["variant"] == v) & (cs["budget"].astype(str) == "full")].iloc[0]
                L.append(f"| {d} | {LABELS[m]} | {v} | {r.f1:.3f} | {r.brier:.3f} | {r.ece:.3f} | {r.coverage:.3f} | {r.auto_risk:.3f} |")
    L.append("")
    L += ["## Distribution shift (F1, raw; calibrated Brier and realized risk in brackets)", "",
          "| Model | YouTube in-domain | YouTube unseen video | YouTube ← SMS | SMS in-domain | SMS ← YouTube | Both pools → YouTube | Both pools → SMS |",
          "|---|---|---|---|---|---|---|---|"]
    cols = [("in-domain", "youtube"), ("unseen video", "youtube"), ("cross-dataset", "youtube"),
            ("in-domain", "sms"), ("cross-dataset", "sms"), ("both pools", "youtube"), ("both pools", "sms")]

    def cell(m, s, t):
        raw = shift[(shift["setting"] == s) & (shift["target"] == t) & (shift["model"] == m) & (shift["variant"] == "raw")]
        cal = shift[(shift["setting"] == s) & (shift["target"] == t) & (shift["model"] == m) & (shift["variant"] == "calibrated")]
        txt = f"{raw['f1'].iloc[0]:.3f}"
        if len(cal):
            txt += f" [{cal['brier'].iloc[0]:.3f}, {cal['auto_risk'].iloc[0]:.2f}]"
        return txt

    for m in MODELS:
        L.append(f"| {LABELS[m]} | " + " | ".join(cell(m, s, t) for s, t in cols) + " |")
    jy, js = jev["youtube"]["point"], jev["sms"]["point"]
    L.append(f"| Jev (zero-shot) | {jy['f1']:.3f} | {jy['f1']:.3f} | {jy['f1']:.3f} | {js['f1']:.3f} | {js['f1']:.3f} "
             f"| {jy['f1']:.3f} | {js['f1']:.3f} |")
    L += ["", "Jev needs no training labels, so its score is the same in every column of a dataset.", "",
          "![Figure 6](figures/fig6_strong_curve.png)", "", "![Figure 7](figures/fig7_strong_shift.png)", ""]
    (RESULTS / "strong.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    jev = {}
    for d in data.DATASETS:
        t = data.load(d)
        jev[d] = jev_summary(d, list(t[t["split"] == "test"]["id"]))
    cs = curve_summary(run_curve())
    shift = run_shift()
    figures(cs, shift, jev)
    report(cs, shift, jev)
    print((RESULTS / "strong.md").read_text())
