"""Metrics, n* estimates, figures, and results.md.

Reads results/runs_*.csv, results/preds_*.npz, results/latency_baselines.json and jev_cache.jsonl;
never calls the API. Works without the Jev cache (baseline-only report).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
import jev_run
from metrics import all_metrics, coverage_threshold

RESULTS = Path(__file__).parent / "results"
FIGURES = RESULTS / "figures"
PRICE_PER_M_INPUT = 0.042  # US$ per million input tokens, TypeSafe list price
N_BOOT = 1000
METRICS = ["f1", "brier", "ece", "coverage", "auto_risk"]
HIGHER_IS_BETTER = {"f1": True, "brier": False, "ece": False, "coverage": True}
MODEL_LABEL = {"naive_bayes": "Naive Bayes", "tfidf_logreg": "TF-IDF + LR"}
COLORS = {"naive_bayes": "#1f77b4", "tfidf_logreg": "#d62728"}


# ---------- Jev ----------

def jev_records(dataset, rubric="main", run_tag="day1", split=None):
    recs = [r for r in jev_run.load_cache()
            if r["dataset"] == dataset and r["rubric"] == rubric and r["run_tag"] == run_tag
            and (split is None or r["split"] == split)]
    return pd.DataFrame(recs).drop_duplicates("id", keep="last") if recs else pd.DataFrame()


def jev_summary(dataset, test_ids, rubric="main"):
    """Jev test metrics with bootstrap 95% intervals, or None if Jev has not been run on the full test set."""
    test = jev_records(dataset, rubric, split="test")
    if test.empty or not set(test_ids) <= set(test["id"]):
        return None
    test = test.set_index("id").loc[test_ids]
    pool = jev_records(dataset, rubric, split="train")
    # Coverage threshold from training-pool predictions (as for the baselines); fall back to test if absent.
    t = coverage_threshold(pool["p_spam"] if not pool.empty else test["p_spam"])
    y, p = test["label"].to_numpy(), test["p_spam"].to_numpy()
    point = all_metrics(y, p, t)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        boots.append(all_metrics(y[i], p[i], t))
    boots = pd.DataFrame(boots)
    ci = {m: (float(boots[m].quantile(0.025)), float(boots[m].quantile(0.975))) for m in METRICS}
    return {"point": point, "ci": ci, "threshold": t, "models": sorted(test["model"].unique())}


def jev_cost(dataset):
    df = jev_records(dataset)
    if df.empty:
        return None
    return {
        "calls": len(df),
        "median_ms": float(df["latency_ms"].median()),
        "p95_ms": float(df["latency_ms"].quantile(0.95)),
        "mean_input_tokens": float(df["input_tokens"].mean()),
        "usd_per_1000": float(df["input_tokens"].mean() * 1000 * PRICE_PER_M_INPUT / 1e6),
    }


def stability(dataset):
    day1, day2 = jev_records(dataset), jev_records(dataset, run_tag="day2")
    if day1.empty or day2.empty:
        return None
    m = day2.merge(day1, on="id", suffixes=("_2", "_1"))
    changed = int(((m["p_spam_1"] >= 0.5) != (m["p_spam_2"] >= 0.5)).sum())
    return {"n": len(m), "changed": changed, "mean_abs_dp": float((m["p_spam_1"] - m["p_spam_2"]).abs().mean()),
            "models_day2": sorted(m["model_2"].unique())}


# ---------- Baselines ----------

def baseline_metrics(dataset):
    # Read as strings: pandas would otherwise parse label_source "true" as a boolean.
    runs = pd.read_csv(RESULTS / f"runs_{dataset}.csv", dtype={"label_source": str, "budget": str})
    npz = np.load(RESULTS / f"preds_{dataset}.npz")
    y = npz["labels"]
    rows = [all_metrics(y, p, t) for p, t in zip(npz["probs"], runs["cov_threshold"])]
    per_run = pd.concat([runs, pd.DataFrame(rows)], axis=1)
    agg = (per_run.groupby(["label_source", "model", "variant", "budget"], sort=False)
           .agg(n=("n", "mean"), **{f"{m}_mean": (m, "mean") for m in METRICS},
                **{f"{m}_std": (m, "std") for m in METRICS}, calibrated_share=("calibrated", "mean"))
           .reset_index())
    return per_run, agg, npz["ids"]


def n_star(agg, jev, metric, source="true"):
    """Smallest label budget whose mean metric reaches Jev's (within Jev's bootstrap interval)."""
    out = {}
    for (model, variant), g in agg[agg["label_source"] == source].groupby(["model", "variant"]):
        g = g.sort_values("n")
        if HIGHER_IS_BETTER[metric]:
            hit = g[g[f"{metric}_mean"] >= jev["ci"][metric][0]]
        else:
            hit = g[g[f"{metric}_mean"] <= jev["ci"][metric][1]]
        out[(model, variant)] = int(hit["n"].iloc[0]) if len(hit) else None
    return out


# ---------- Figures ----------

def plot_curves(ax, agg, metric, source, variants, jev, styles=None):
    for model in MODEL_LABEL:
        for variant in variants:
            g = agg[(agg["label_source"] == source) & (agg["model"] == model) & (agg["variant"] == variant)]
            if g.empty:
                continue
            g = g.sort_values("n")
            ls = "-" if variant == "raw" else "--"
            label = MODEL_LABEL[model] + ("" if len(variants) == 1 else f" ({variant})")
            ax.plot(g["n"], g[f"{metric}_mean"], ls, marker="o", ms=3, color=COLORS[model], label=label)
            ax.fill_between(g["n"], g[f"{metric}_mean"] - g[f"{metric}_std"],
                            g[f"{metric}_mean"] + g[f"{metric}_std"], color=COLORS[model], alpha=0.12)
    if jev:
        lo, hi = jev["ci"][metric]
        ax.axhspan(lo, hi, color="k", alpha=0.12)
        ax.axhline(jev["point"][metric], color="k", lw=1.2, label="Jev (zero-shot)")
    ax.set_xscale("log")
    ax.set_xlabel("number of labels")
    ax.grid(alpha=0.3)


def figures(aggs, jevs):
    FIGURES.mkdir(parents=True, exist_ok=True)
    ds = list(aggs)

    fig, axes = plt.subplots(1, len(ds), figsize=(5.5 * len(ds), 4), squeeze=False)
    for ax, d in zip(axes[0], ds):
        plot_curves(ax, aggs[d], "f1", "true", ["raw"], jevs[d])
        ax.set_title(f"{d}: F1 vs labels")
        ax.set_ylabel("F1 (test)")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_f1.png", dpi=150)

    fig, axes = plt.subplots(2, len(ds), figsize=(5.5 * len(ds), 7.5), squeeze=False)
    for j, d in enumerate(ds):
        for i, (metric, lab) in enumerate([("brier", "Brier score (lower is better)"),
                                           ("coverage", "coverage at 5% risk")]):
            plot_curves(axes[i][j], aggs[d], metric, "true", ["raw", "calibrated"], jevs[d])
            axes[i][j].set_title(f"{d}: {metric}")
            axes[i][j].set_ylabel(lab)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_calibration.png", dpi=150)

    has_e3 = any((aggs[d]["label_source"] != "true").any() for d in ds)
    if has_e3:
        fig, axes = plt.subplots(len(MODEL_LABEL), len(ds), figsize=(5.5 * len(ds), 7.5), squeeze=False)
        for i, model in enumerate(MODEL_LABEL):
            for j, d in enumerate(ds):
                ax = axes[i][j]
                for source, ls in [("true", "-"), ("jev_hard", "--"), ("jev_soft", ":")]:
                    g = aggs[d][(aggs[d]["label_source"] == source) & (aggs[d]["model"] == model)
                                & (aggs[d]["variant"] == "raw")].sort_values("n")
                    if not g.empty:
                        ax.plot(g["n"], g["f1_mean"], ls, marker="o", ms=3, color=COLORS[model],
                                label=f"{source} labels")
                if jevs[d]:
                    ax.axhspan(*jevs[d]["ci"]["f1"], color="k", alpha=0.12)
                    ax.axhline(jevs[d]["point"]["f1"], color="k", lw=1.2, label="Jev (teacher)")
                ax.set_xscale("log")
                ax.set_title(f"{d}: {MODEL_LABEL[model]} student")
                ax.set_xlabel("number of labels")
                ax.set_ylabel("F1 vs true test labels")
                ax.grid(alpha=0.3)
        axes[0][0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / "fig3_student.png", dpi=150)
    plt.close("all")
    return has_e3


# ---------- Report ----------

def fmt(x, digits=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def report(aggs, jevs, alts, costs, stab, lat, has_e3):
    L = ["# Pilot results: Jev vs lightweight models on spam filtering", ""]
    if not any(jevs.values()):
        L += ["> **Jev has not been run yet** (no complete `jev_cache.jsonl`). Baseline curves only; "
              "n*, E3, E4 Jev columns, stability and wording checks are pending.", ""]

    L += ["## Jev zero-shot (test set, main rubric)", "",
          "| Dataset | F1 | Brier | ECE | Coverage @5% | Realized risk | Model string |", "|---|---|---|---|---|---|---|"]
    for d, j in jevs.items():
        if j:
            c = j["ci"]
            L.append(f"| {d} | {fmt(j['point']['f1'])} [{fmt(c['f1'][0])}, {fmt(c['f1'][1])}] "
                     f"| {fmt(j['point']['brier'])} [{fmt(c['brier'][0])}, {fmt(c['brier'][1])}] "
                     f"| {fmt(j['point']['ece'])} | {fmt(j['point']['coverage'])} | {fmt(j['point']['auto_risk'])} "
                     f"| {', '.join(j['models'])} |")
        else:
            L.append(f"| {d} | pending | | | | | |")
    L.append("")

    L += ["## Baselines at full pool (true labels, mean ± sd over 10 seeds)", "",
          "| Dataset | Model | Variant | n | F1 | Brier | ECE | Coverage @5% | Realized risk |",
          "|---|---|---|---|---|---|---|---|---|"]
    for d, agg in aggs.items():
        g = agg[(agg["label_source"] == "true") & (agg["budget"] == "full")]
        for r in g.itertuples():
            L.append(f"| {d} | {MODEL_LABEL[r.model]} | {r.variant} | {int(r.n)} | {fmt(r.f1_mean)} ± {fmt(r.f1_std)} "
                     f"| {fmt(r.brier_mean)} | {fmt(r.ece_mean)} | {fmt(r.coverage_mean)} | {fmt(r.auto_risk_mean)} |")
    L.append("")

    L += ["## n*: labels needed to match Jev (E1, E2)", "",
          "Smallest budget whose mean reaches Jev's bootstrap interval; `>pool` = never reached.", "",
          "| Dataset | Model | Variant | n* F1 | n* Brier | n* ECE | n* coverage |", "|---|---|---|---|---|---|---|"]
    for d, agg in aggs.items():
        if not jevs[d]:
            L.append(f"| {d} | – | – | pending | pending | pending | pending |")
            continue
        stars = {m: n_star(agg, jevs[d], m) for m in HIGHER_IS_BETTER}
        for key in stars["f1"]:
            cells = [">pool" if stars[m][key] is None else str(stars[m][key]) for m in HIGHER_IS_BETTER]
            L.append(f"| {d} | {MODEL_LABEL[key[0]]} | {key[1]} | " + " | ".join(cells) + " |")
    L.append("")

    if has_e3:
        L += ["## E3: Jev as teacher", "",
              "| Dataset | Model | Label source | n* F1 (vs Jev) | Best mean F1 | Exceeds Jev? |", "|---|---|---|---|---|---|"]
        for d, agg in aggs.items():
            if not jevs[d]:
                continue
            for source in ["true", "jev_hard", "jev_soft"]:
                stars = n_star(agg, jevs[d], "f1", source)
                for model in MODEL_LABEL:
                    g = agg[(agg["label_source"] == source) & (agg["model"] == model) & (agg["variant"] == "raw")]
                    if g.empty:
                        continue
                    best = g["f1_mean"].max()
                    s = stars.get((model, "raw"))
                    L.append(f"| {d} | {MODEL_LABEL[model]} | {source} | {'>pool' if s is None else s} "
                             f"| {fmt(best)} | {'yes' if best > jevs[d]['point']['f1'] else 'no'} |")
        L.append("")

    L += ["## E4: latency and cost per 1,000 decisions", "",
          "| Dataset | Model | Median ms | p95 ms | US$ / 1,000 |", "|---|---|---|---|---|"]
    for d in aggs:
        c = costs.get(d)
        L.append(f"| {d} | Jev (client-side) | {fmt(c['median_ms'], 1) if c else 'pending'} "
                 f"| {fmt(c['p95_ms'], 1) if c else ''} | {fmt(c['usd_per_1000'], 6) if c else ''} |")
        for model, v in lat.get(d, {}).items():
            L.append(f"| {d} | {MODEL_LABEL[model]} (CPU) | {fmt(v['median_ms'], 2)} | {fmt(v['p95_ms'], 2)} | ~0 |")
    L.append("")

    L += ["## Checks", ""]
    for d in aggs:
        if alts.get(d) and jevs[d]:
            a, m = alts[d]["point"], jevs[d]["point"]
            L.append(f"- **Wording sensitivity ({d}):** alt rubric F1 {fmt(a['f1'])} vs main {fmt(m['f1'])}; "
                     f"Brier {fmt(a['brier'])} vs {fmt(m['brier'])}.")
        else:
            L.append(f"- **Wording sensitivity ({d}):** pending.")
        if stab.get(d):
            s = stab[d]
            L.append(f"- **Stability ({d}):** {s['changed']}/{s['n']} decisions changed on day 2 "
                     f"(mean |Δp| {fmt(s['mean_abs_dp'])}; day-2 model {', '.join(s['models_day2'])}).")
        else:
            L.append(f"- **Stability ({d}):** pending (run `jev_run.py --stability --run-tag day2` on a second day).")
    if all(jevs.values()):
        gap = {d: jevs[d]["point"]["f1"] - aggs[d][(aggs[d]["label_source"] == "true") & (aggs[d]["budget"] == "full")
                                                   & (aggs[d]["variant"] == "raw")]["f1_mean"].max() for d in aggs}
        L.append("- **Contamination:** Jev F1 minus best full-pool baseline F1: "
                 + ", ".join(f"{d} {gap[d]:+.3f}" for d in gap)
                 + ". A much larger gap on SMS than on YouTube suggests exposure, not skill.")
    L += ["", "## Figures", "", "![Figure 1](figures/fig1_f1.png)", "", "![Figure 2](figures/fig2_calibration.png)", ""]
    if has_e3:
        L += ["![Figure 3](figures/fig3_student.png)", ""]
    L += ["## Notes", "",
          "- Coverage @5% risk: each model automates messages whose confidence max(p, 1−p) clears a threshold chosen on "
          "its own (unlabeled) training-pool predictions so that its expected error is ≤ 5%. "
          "Realized risk is the actual error rate among automated test messages.",
          "- Calibrated variants fit on 80% of the sample and fit Platt scaling on the other 20%; when that 20% "
          "lacks a class (small n) Platt scaling is skipped.",
          "- Baselines: mean over 10 stratified draws per budget. Jev: bootstrap 95% intervals over test messages."]
    (RESULTS / "results.md").write_text("\n".join(L) + "\n")


def main():
    aggs, jevs, alts, costs, stab = {}, {}, {}, {}, {}
    for d in data.DATASETS:
        if not (RESULTS / f"runs_{d}.csv").exists():
            continue
        per_run, agg, test_ids = baseline_metrics(d)
        agg.to_csv(RESULTS / f"summary_{d}.csv", index=False)
        aggs[d] = agg
        jevs[d] = jev_summary(d, list(test_ids))
        alts[d] = jev_summary(d, list(test_ids), rubric="alt")
        costs[d] = jev_cost(d)
        stab[d] = stability(d)
    lat_path = RESULTS / "latency_baselines.json"
    lat = json.loads(lat_path.read_text()) if lat_path.exists() else {}
    has_e3 = figures(aggs, jevs)
    report(aggs, jevs, alts, costs, stab, lat, has_e3)
    print((RESULTS / "results.md").read_text())


if __name__ == "__main__":
    main()
