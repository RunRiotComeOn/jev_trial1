"""Load the YouTube and SMS spam datasets, normalize to id/text/label, make the fixed 70/30 split.

Output: data/processed/{youtube,sms}.csv with columns id, text, label, split.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
SEED = 0
TEST_SIZE = 0.30
DATASETS = ["youtube", "sms"]

# State passed to Jev alongside the message text (see jev_run.py).
PLATFORM = {
    "youtube": "YouTube comment on a music video",
    "sms": "SMS text message",
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


LOADERS = {"youtube": load_youtube, "sms": load_sms}


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
