"""Load the spam datasets, normalize to id/text/label, make the fixed 70/30 split.

Output: data/processed/{youtube,sms,enron}.csv with columns id, text, label, split.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
SEED = 0
TEST_SIZE = 0.30
DATASETS = ["youtube", "sms", "enron"]
# Emails are truncated so every model (Jev and baselines) sees the same text; covers >90% of Enron emails whole.
MAX_CHARS = 4000

# State passed to Jev alongside the message text (see jev_run.py).
PLATFORM = {
    "youtube": "YouTube comment on a music video",
    "sms": "SMS text message",
    "enron": "Email (subject line, then body) received in a corporate mailbox",
}


def load_youtube() -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted((RAW / "youtube").glob("Youtube0*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    # Only the text and the binary label; author and date are dropped.
    df = df.rename(columns={"COMMENT_ID": "id", "CONTENT": "text", "CLASS": "label"})[["id", "text", "label"]]
    # A handful of comment ids repeat across files; make ids unique.
    dup = df["id"].duplicated(keep=False)
    df.loc[dup, "id"] = df.loc[dup, "id"] + "-" + df[dup].groupby("id").cumcount().astype(str)
    return df


def load_sms() -> pd.DataFrame:
    df = pd.read_csv(RAW / "sms" / "sms.tsv", sep="\t", header=None, names=["label", "text"], quoting=3)
    df["label"] = (df["label"] == "spam").astype(int)
    df["id"] = [f"sms-{i:05d}" for i in range(len(df))]
    return df[["id", "text", "label"]]


def load_enron() -> pd.DataFrame:
    # Enron-Spam (Metsis et al. 2006), all six folders, as one CSV from MWiechmann/enron_spam_data.
    df = pd.read_csv(RAW / "enron" / "enron_spam_data.zip")
    df["text"] = (df["Subject"].fillna("") + "\n\n" + df["Message"].fillna("")).str.strip().str[:MAX_CHARS]
    df["label"] = (df["Spam/Ham"] == "spam").astype(int)
    df["id"] = [f"enron-{i:05d}" for i in df["Message ID"]]
    # ~10% of emails are exact duplicates (same text, same label); keep one so train and test never share a text.
    df = df[df["text"] != ""].drop_duplicates("text")
    return df[["id", "text", "label"]].reset_index(drop=True)


LOADERS = {"youtube": load_youtube, "sms": load_sms, "enron": load_enron}


def make_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    _, test_idx = train_test_split(df.index, test_size=TEST_SIZE, stratify=df["label"], random_state=SEED)
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"
    return df


def load(dataset: str) -> pd.DataFrame:
    """Load the processed dataset, building it from the raw files if needed."""
    path = PROCESSED / f"{dataset}.csv"
    if not path.exists():
        build(dataset)
    return pd.read_csv(path, keep_default_na=False)


def build(dataset: str) -> pd.DataFrame:
    df = make_split(LOADERS[dataset]())
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED / f"{dataset}.csv", index=False)
    return df


if __name__ == "__main__":
    for name in DATASETS:
        df = build(name)
        for split, part in df.groupby("split"):
            print(f"{name:8s} {split:5s} n={len(part):5d} spam={part['label'].mean():.3f}")
