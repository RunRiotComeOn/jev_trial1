# Pilot: Jev vs lightweight models on spam filtering

Implements the pilot proposal "Jev vs Lightweight Models on Spam Filtering" in code: zero-shot Jev
(`jev-1.13.0`, one Noul question) against naive Bayes and TF-IDF + logistic regression on the YouTube
and SMS spam datasets.

| File | Purpose |
|---|---|
| `data.py` | Load both datasets, normalize to `id, text, label`, stratified 70/30 split (seed 0) → `data/processed/` |
| `jev_run.py` | Call Jev on every message (8 concurrent requests, backoff), append to `jev_cache.jsonl` |
| `baselines.py` | Label-budget sweep (E1–E3) and single-message CPU latency (E4) → `results/` |
| `metrics.py` | F1, Brier, ECE (10 equal-mass bins), coverage at 5% risk |
| `analyze.py` | n* estimates, figures, `results/results.md`; never calls the API |
| `run_all.sh` | Day 1: everything; `./run_all.sh stability` on day 2 |

## Running

```bash
pip install -r requirements.txt
export TYPESAFE_API_KEY=...
./run_all.sh              # day 1
./run_all.sh stability    # day 2: rerun Jev on 100 fixed test messages
```

In the cloud: GitHub → Actions → **Jev spam pilot** → Run workflow (`full`, `stability`, or
`baselines`). It needs a repository secret `TYPESAFE_API_KEY`. The workflow uploads `results/` and the
Jev cache as an artifact and commits them back to the branch, so later runs reuse the cache.

## Data

The UCI site is not reachable from the Claude Code cloud session, so the raw files are committed under
`data/raw/`, taken from mirrors of the UCI originals:

- YouTube Spam Collection (5 CSVs, 1,956 comments, 51% spam): from
  `PacktPublishing/Python-Artificial-Intelligence-Projects-for-Beginners/Chapter03`.
- SMS Spam Collection (5,574 messages, 13% spam): `justmarkham/DAT8/data/sms.tsv`.

## Choices the proposal left open

- **Calibrated variant:** fit on 80% of the n sampled labels, then Platt scaling on the other 20%
  (with soft targets for Jev soft labels). If that 20% lacks a class (very small n), Platt scaling is
  skipped and the run is flagged `calibrated=False` in `results/runs_*.csv`. The raw variant uses all n.
- **Coverage at 5% risk:** a model automates messages whose confidence max(p, 1−p) clears a threshold
  chosen on its own unlabeled training-pool predictions, so that its expected error is at most 5%.
  This needs no extra labels and penalizes miscalibration. The report also gives the realized error
  rate among automated test messages.
- **Soft labels (E3):** each message enters training twice, as spam with weight p and as ham with
  weight 1−p.
- **n\*:** the smallest budget whose mean over 10 draws reaches Jev's bootstrap 95% interval
  (F1, coverage: ≥ lower bound; Brier, ECE: ≤ upper bound).
- **Cost:** mean input tokens × US$0.042 per million (the list price in the proposal).
