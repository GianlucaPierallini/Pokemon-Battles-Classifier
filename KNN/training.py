"""5-fold stratified kNN model selection with strict fold-local preprocessing."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

from .features import BattleFeatureExtractor
from .model import NeighborEngine, SparseFeatureScaler

LOG = logging.getLogger(__name__)


@dataclass
class CVResults:
    folds: pd.DataFrame
    summary: pd.DataFrame
    best_k: int
    elapsed_seconds: float


class KNNCrossValidator:
    """Compute neighbor lists once per fold, evaluate all k on the same folds.

    The vectorizer and scaler are both fitted solely on that fold's train rows.
    Group CV can optionally keep identical six-species P1 teams together.
    """

    def __init__(self, feature_set: str = "summary", scaling: str = "numeric",
                 category_weight: float = 1.0, k_values: list[int] | None = None,
                 folds: int = 5, seed: int = 42, group_cv: bool = False,
                 engine: NeighborEngine | None = None):
        self.feature_set = feature_set
        self.scaling = scaling
        self.category_weight = category_weight
        self.k_values = k_values or list(range(1, 52, 2))
        self.folds = folds
        self.seed = seed
        self.group_cv = group_cv
        self.engine = engine or NeighborEngine("cpu")
        self.extractor = BattleFeatureExtractor(feature_set)
        if folds < 2:
            raise ValueError("At least 2 folds required")
        if not self.k_values or any(k < 1 or k % 2 == 0 for k in self.k_values):
            raise ValueError("k_values must contain positive odd integers")

    def run(self, records: list[dict], y: np.ndarray) -> CVResults:
        y = np.asarray(y, dtype=np.int8)
        if len(records) != len(y):
            raise ValueError("X and y size mismatch")
        cv = (
            StratifiedGroupKFold(n_splits=self.folds, shuffle=True, random_state=self.seed)
            if self.group_cv
            else StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=self.seed)
        )
        groups = None
        if self.group_cv:
            groups = [
                "|".join(sorted(str(mon.get("name", "unknown")) for mon in (r.get("p1_team_details") or [])))
                for r in records
            ]
        start = time.perf_counter()
        rows = []
        max_k = max(self.k_values)
        for fold_idx, (i_train, i_valid) in enumerate(cv.split(np.arange(len(records)), y, groups=groups), 1):
            if len(i_train) < max_k:
                raise ValueError(f"max_k={max_k} exceeds training size {len(i_train)} in fold {fold_idx}")
            LOG.info("Fold %d/%d: %d train, %d validation; vectorizing...",
                     fold_idx, self.folds, len(i_train), len(i_valid))
            # Strictly train-only vocabulary: this is more rigorous than the notebook.
            dv = DictVectorizer(sparse=True, sort=True)
            X_train = dv.fit_transform(self.extractor.transform(records[i] for i in i_train))
            X_valid = dv.transform(self.extractor.transform(records[i] for i in i_valid))
            X_train = X_train.tocsr().astype(np.float32)
            X_valid = X_valid.tocsr().astype(np.float32)
            scaler = SparseFeatureScaler(self.scaling, self.category_weight)
            scaler.fit(X_train, dv.get_feature_names_out().tolist())
            X_train = scaler.transform(X_train)
            X_valid = scaler.transform(X_valid)
            LOG.info("Fold %d: matrix shape=%s; finding %d neighbors...",
                     fold_idx, X_train.shape, max_k)
            neighbor_idx = self.engine.kneighbors(X_train, X_valid, max_k)
            sorted_neighbor_labels = y[i_train][neighbor_idx]
            cumulative_wins = np.cumsum(sorted_neighbor_labels, axis=1)
            y_true = y[i_valid]
            for k in self.k_values:
                positive_neighbors = cumulative_wins[:, k - 1]
                probs = positive_neighbors / k
                preds = (positive_neighbors > k / 2).astype(np.int8)
                # AUC may be undefined on tiny/grouped folds with a single class.
                roc_auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) == 2 else np.nan
                rows.append({
                    "fold": fold_idx,
                    "k": k,
                    "accuracy": float(np.mean(preds == y_true)),
                    "f1": float(f1_score(y_true, preds, zero_division=0)),
                    "roc_auc": roc_auc,
                    "n_train": len(i_train),
                    "n_valid": len(i_valid),
                    "backend": self.engine.last_backend,
                })
            fold_best = max((x for x in rows if x["fold"] == fold_idx), key=lambda x: x["accuracy"])
            LOG.info("Fold %d complete: best accuracy %.4f (k=%d), backend=%s",
                     fold_idx, fold_best["accuracy"], fold_best["k"], self.engine.last_backend)
        df = pd.DataFrame(rows)
        summary = (
            df.groupby("k", sort=True)
            .agg(mean_accuracy=("accuracy", "mean"), std_accuracy=("accuracy", "std"),
                 mean_f1=("f1", "mean"), mean_roc_auc=("roc_auc", "mean"))
            .reset_index()
        )
        # Deterministic tie rule: prefer smaller SD then lower k.
        rank = summary.sort_values(["mean_accuracy", "std_accuracy", "k"],
                                   ascending=[False, True, True])
        best_k = int(rank.iloc[0]["k"])
        return CVResults(df, summary, best_k, time.perf_counter() - start)
