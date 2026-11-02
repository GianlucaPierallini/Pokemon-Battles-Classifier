"""Reproducible development CV, group-safe variants, and holdout evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedGroupKFold,
    StratifiedKFold, train_test_split,
)


@dataclass
class ExperimentResult:
    search: Any
    dev_indices: np.ndarray
    holdout_indices: np.ndarray
    holdout_pred: np.ndarray
    holdout_prob: np.ndarray
    train_accuracy: float
    baseline_cv: dict | None


class PruningCrossValidator:
    """Tune only on development data; evaluate the selected tree once on holdout."""

    def __init__(self, *, folds=5, seed=42, group_cv=False,
                 metric="accuracy", search="random", trials=18, jobs=1):
        if folds < 2:
            raise ValueError("folds must be >=2")
        if trials < 1:
            raise ValueError("trials must be >=1")
        if metric not in ("accuracy", "roc_auc", "f1"):
            raise ValueError("Unsupported metric")
        self.folds = folds
        self.seed = seed
        self.group_cv = group_cv
        self.metric = metric
        self.search_mode = search
        self.trials = trials
        self.jobs = jobs

    def split(self, y, groups):
        indices = np.arange(len(y))
        if self.group_cv:
            # Disjoint team compositions across development and holdout.
            outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=self.seed)
            train_idx, holdout_idx = next(outer.split(indices, y, groups))
        else:
            # Exactly mirrors the previous DecisionTree experiment seed/split.
            train_idx, holdout_idx = train_test_split(
                indices, test_size=0.2, stratify=y, random_state=self.seed)
        return np.asarray(train_idx), np.asarray(holdout_idx)

    def make_cv(self):
        cls = StratifiedGroupKFold if self.group_cv else StratifiedKFold
        return cls(n_splits=self.folds, shuffle=True, random_state=self.seed)

    def run(self, features, y, groups, estimator, param_grid, baseline_estimator=None, baseline_params=None):
        dev, holdout = self.split(y, groups)
        X_dev = [features[i] for i in dev]
        X_hold = [features[i] for i in holdout]
        cv = self.make_cv()
        split_groups = groups[dev] if self.group_cv else None
        total = int(np.prod([len(v) for v in param_grid.values()]))
        kwargs = dict(estimator=estimator, scoring=self.metric, cv=cv,
                      n_jobs=self.jobs, refit=True, return_train_score=True,
                      verbose=1, error_score="raise")
        if self.search_mode == "grid":
            search = GridSearchCV(param_grid=param_grid, **kwargs)
            count = total
        elif self.search_mode == "random":
            count = min(self.trials, total)
            search = RandomizedSearchCV(param_distributions=param_grid,
                                        n_iter=count, random_state=self.seed, **kwargs)
        else:
            raise ValueError("search must be random or grid")
        print(f"\nSearch: {count} candidate configurations x {self.folds} folds = {count*self.folds} CV fits")
        search.fit(X_dev, y[dev], groups=split_groups)
        best = search.best_estimator_

        from sklearn.metrics import accuracy_score
        train_accuracy = accuracy_score(y[dev], best.predict(X_dev))
        holdout_pred = np.asarray(best.predict(X_hold), dtype=int)
        holdout_prob = np.asarray(best.predict_proba(X_hold)[:, 1], dtype=float)

        baseline_cv = None
        if baseline_estimator is not None and baseline_params is not None:
            from sklearn.model_selection import cross_validate
            baseline = clone(baseline_estimator).set_params(**baseline_params)
            sc = cross_validate(baseline, X_dev, y[dev], cv=cv, groups=split_groups,
                                scoring=self.metric, n_jobs=self.jobs,
                                error_score="raise")
            baseline_cv = {"fixed_params": baseline_params,
                           "fold_scores": list(map(float, sc["test_score"])),
                           "mean": float(np.mean(sc["test_score"])),
                           "std": float(np.std(sc["test_score"]))}

        return ExperimentResult(search, dev, holdout, holdout_pred,
                                holdout_prob, float(train_accuracy), baseline_cv)

    @staticmethod
    def refit_all(estimator, params, X, y):
        final = clone(estimator).set_params(**params)
        final.fit(X, y)
        return final
