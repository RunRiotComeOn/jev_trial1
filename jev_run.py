"""Call Jev (zero-shot, one Noul question) on every message and cache responses to jev_cache.jsonl.

Runs:
  python jev_run.py                                  # main rubric, both splits, both datasets
  python jev_run.py --rubric alt --split test        # wording-sensitivity check
  python jev_run.py --stability --run-tag day2       # 100 test messages per dataset, second day

The cache is append-only and keyed by (dataset, id, rubric, run_tag); re-running skips what is cached,
so analysis never re-calls the API.
"""

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import data
from typesafe_sdk import (
    Noul,
    NoulCriteria,
    TypeSafeAPIConnectionError,
    TypeSafeAPITimeoutError,
    TypeSafeClient,
    TypeSafeInternalServerError,
    TypeSafeRateLimitError,
)

MODEL = "jev-1.13.0"  # pinned; jev-latest is moving to 1.14
CACHE = Path(__file__).parent / "jev_cache.jsonl"
CONCURRENCY = 8
STABILITY_N = 100

# Frozen before any test result is seen.
RUBRICS = {
    "main": Noul(
        instructions="Is `message` spam?",
        criteria=NoulCriteria(
            true="Promotes a channel, product, or website; asks readers to subscribe, "
            "follow, or visit something; or advertises a deal or prize",
            false="A genuine reaction, opinion, question, or conversation, "
            "even if it contains a link relevant to the content",
        ),
    ),
    # Independently worded; reported for wording sensitivity only.
    "alt": Noul(
        instructions="Was `message` sent to advertise or self-promote rather than to communicate?",
        criteria=NoulCriteria(
            true="Unsolicited advertising, self-promotion, scams, prize or offer notices, "
            "or requests to click, call, text, subscribe, or share",
            false="An ordinary personal or on-topic message written for the people reading it",
        ),
    ),
}

RETRYABLE = (TypeSafeRateLimitError, TypeSafeAPITimeoutError, TypeSafeAPIConnectionError, TypeSafeInternalServerError)


def ask_jev(client: TypeSafeClient, text: str, platform: str, rubric: str, model: str) -> dict:
    t0 = time.perf_counter()
    r = client.system_one(
        state={"platform": platform, "message": text},
        questions={"is_spam": RUBRICS[rubric]},
        model=model,
    )
    return {
        "p_spam": r.answers["is_spam"].noul,
        "model": r.model,
        "latency_ms": (time.perf_counter() - t0) * 1000,
        "input_tokens": r.usage.input_tokens,
    }


def ask_with_backoff(client, text, platform, rubric, model, attempts=6) -> dict:
    # The SDK retries a little on its own; this adds a longer backoff for sustained rate limiting.
    for attempt in range(attempts):
        try:
            return ask_jev(client, text, platform, rubric, model)
        except RETRYABLE:
            if attempt == attempts - 1:
                raise
            time.sleep(min(60.0, 2.0 * 2**attempt) + random.random())


def load_cache(path: Path = CACHE) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def cache_key(rec: dict) -> tuple:
    return (rec["dataset"], rec["id"], rec["rubric"], rec["run_tag"])


def jobs_for(dataset: str, args) -> list[dict]:
    df = data.load(dataset)
    if args.split != "both":
        df = df[df["split"] == args.split]
    if args.stability:
        # Fixed sample of test messages so the day-2 rerun lines up with the main run.
        df = df[df["split"] == "test"].sample(n=STABILITY_N, random_state=data.SEED)
    if args.limit:
        df = df.head(args.limit)
    return [
        {"dataset": dataset, "id": row.id, "split": row.split, "label": int(row.label), "text": row.text}
        for row in df.itertuples()
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=data.DATASETS, choices=data.DATASETS)
    ap.add_argument("--rubric", default="main", choices=list(RUBRICS))
    ap.add_argument("--split", default="both", choices=["train", "test", "both"])
    ap.add_argument("--run-tag", default="day1")
    ap.add_argument("--stability", action="store_true", help=f"only {STABILITY_N} fixed test messages per dataset")
    ap.add_argument("--limit", type=int, default=0, help="cap messages per dataset (smoke test)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = ap.parse_args()

    done = {cache_key(r) for r in load_cache()}
    jobs = [
        j
        for d in args.datasets
        for j in jobs_for(d, args)
        if (j["dataset"], j["id"], args.rubric, args.run_tag) not in done
    ]
    print(f"{len(jobs)} calls to make ({len(done)} cached), model={args.model}, rubric={args.rubric}")
    if not jobs:
        return

    client = TypeSafeClient(api_key=os.environ["TYPESAFE_API_KEY"], timeout=60.0)
    lock = threading.Lock()
    failures = 0

    def work(job):
        out = ask_with_backoff(client, job["text"], data.PLATFORM[job["dataset"]], args.rubric, args.model)
        return {
            "dataset": job["dataset"],
            "id": job["id"],
            "split": job["split"],
            "label": job["label"],
            "rubric": args.rubric,
            "run_tag": args.run_tag,
            "requested_model": args.model,
            **out,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    with CACHE.open("a") as f, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(work, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                rec = fut.result()
            except Exception as e:  # keep going; failed calls are retried on the next run
                failures += 1
                print(f"FAILED {futures[fut]['dataset']}/{futures[fut]['id']}: {type(e).__name__}: {e}")
                continue
            with lock:
                f.write(json.dumps(rec) + "\n")
                f.flush()
            if i % 250 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}")

    print(f"done; {failures} failures")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
