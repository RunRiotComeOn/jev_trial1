"""Stronger classic (non-pretrained) text classifiers.

Configurations are fixed in advance from common defaults; nothing is tuned on test data.
Every model is a Pipeline whose last step accepts sample_weight, as baselines.fit expects.
"""

import re

import numpy as np
import scipy.sparse as sp
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FunctionTransformer, make_pipeline, make_union

URL = re.compile(r"https?://|www\.|\.com\b|\.co\.uk\b|\.net\b", re.I)
LONG_NUMBER = re.compile(r"\d{5,}")
CURRENCY = re.compile(r"[£$€]|\bp\b|\bpence\b|\bcash\b", re.I)


def surface_features(texts):
    """Platform-agnostic surface cues: links, numbers, money, shouting, length."""
    rows = []
    for t in texts:
        n = max(len(t), 1)
        letters = [c for c in t if c.isalpha()]
        rows.append([
            bool(URL.search(t)),
            bool(LONG_NUMBER.search(t)),
            bool(CURRENCY.search(t)),
            sum(c.isdigit() for c in t) / n,
            sum(c.isupper() for c in letters) / max(len(letters), 1),
            min(t.count("!"), 5) / 5,
            np.log1p(len(t)) / 6,
            np.log1p(len(t.split())) / 5,
        ])
    return sp.csr_matrix(np.asarray(rows, dtype=np.float64))


def surface():
    return FunctionTransformer(surface_features, accept_sparse=True)


def word_tfidf(**kw):
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, **kw)


def char_tfidf(**kw):
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2, **kw)


def balanced_lr(C=1.0):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=3000)


class NBLogReg(BaseEstimator, ClassifierMixin):
    """NB-SVM (Wang & Manning 2012) with a logistic loss: scale binary features by the NB log-count ratio."""

    def __init__(self, alpha=1.0, C=4.0):
        self.alpha = alpha
        self.C = C

    def fit(self, X, y, sample_weight=None):
        X = (X > 0).astype(np.float64)
        y = np.asarray(y)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        p = self.alpha + X.T @ (w * (y == 1))
        q = self.alpha + X.T @ (w * (y == 0))
        self.r_ = np.log((p / p.sum()) / (q / q.sum()))
        self.lr_ = balanced_lr(self.C).fit(X.multiply(self.r_).tocsr(), y, sample_weight=w)
        self.classes_ = self.lr_.classes_
        return self

    def predict_proba(self, X):
        return self.lr_.predict_proba((X > 0).astype(np.float64).multiply(self.r_).tocsr())

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class SoftEnsemble(BaseEstimator, ClassifierMixin):
    """Average the spam probabilities of several pipelines; takes raw texts."""

    def __init__(self, members):
        self.members = members

    def fit(self, X, y, sample_weight=None):
        self.fitted_ = []
        for m in self.members:
            m = clone(m)
            m.fit(X, y, **{f"{m.steps[-1][0]}__sample_weight": sample_weight})
            self.fitted_.append(m)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.fitted_], axis=0)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


def lr_balanced():
    return make_pipeline(word_tfidf(), balanced_lr())


def char_lr():
    return make_pipeline(char_tfidf(), balanced_lr(C=10.0))


def nbsvm():
    return make_pipeline(CountVectorizer(ngram_range=(1, 2), binary=True), NBLogReg())


def word_char_surface_lr():
    return make_pipeline(make_union(word_tfidf(), char_tfidf(), surface()), balanced_lr(C=10.0))


def lightgbm():
    return make_pipeline(
        make_union(word_tfidf(max_features=20000), char_tfidf(max_features=20000), surface()),
        LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=5,
                       colsample_bytree=0.3, class_weight="balanced", random_state=0, n_jobs=1, verbose=-1),
    )


def ensemble():
    return make_pipeline(SoftEnsemble([nbsvm(), char_lr(), word_char_surface_lr()]))


STRONG_MODELS = {
    "lr_balanced": lr_balanced,
    "char_lr": char_lr,
    "nbsvm": nbsvm,
    "word_char_surface_lr": word_char_surface_lr,
    "lightgbm": lightgbm,
    "ensemble": ensemble,
}

STRONG_LABELS = {
    "lr_balanced": "TF-IDF + LR (balanced)",
    "char_lr": "Char n-gram LR",
    "nbsvm": "NB-SVM",
    "word_char_surface_lr": "Word+char+surface LR",
    "lightgbm": "LightGBM",
    "ensemble": "Ensemble",
}
