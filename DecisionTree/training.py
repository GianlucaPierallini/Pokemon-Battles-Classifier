"""Holdout + stratified nested-in-development hyperparameter search."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, StratifiedGroupKFold,
    RandomizedSearchCV, GridSearchCV,
)
from sklearn.metrics import accuracy_score
from .model import DecisionTreeModelFactory


@dataclass
class TrainingResult:
    search: object
    best_estimator: object
    dev_indices: np.ndarray
    holdout_indices: np.ndarray
    pred_holdout: np.ndarray
    prob_holdout: np.ndarray
    cv_splitter: object


class DecisionTreeTrainer:
    def __init__(self, backend="cpu", folds=5, trials=24, seed=42,
                 metric="accuracy", group_cv=False, search_mode="random", jobs=1, select_k=512):
        if folds < 2 or trials < 1:
            raise ValueError("folds >= 2 and trials >= 1 are required")
        self.backend = backend
        self.folds = folds
        self.trials = trials
        self.seed = seed
        self.metric = metric
        self.group_cv = group_cv
        self.search_mode = search_mode
        self.jobs = jobs
        self.select_k = select_k

    def fit_validate(self, X: list[dict], y: np.ndarray, groups: np.ndarray) -> TrainingResult:
        indices = np.arange(len(y))
        if self.group_cv:
            # One fold out of five (~20% holdout, possibly not exact in sample counts).
            outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=self.seed)
            dev_idx, hold_idx = next(outer.split(indices, y, groups=groups))
        else:
            dev_idx, hold_idx = train_test_split(
                indices, test_size=0.20, random_state=self.seed, stratify=y
            )
        X_dev = [X[i] for i in dev_idx]
        y_dev = y[dev_idx]
        groups_dev = groups[dev_idx]
        if self.group_cv:
            inner = StratifiedGroupKFold(n_splits=self.folds, shuffle=True, random_state=self.seed)
            cv_iter = list(inner.split(X_dev, y_dev, groups_dev))
            for tr, va in cv_iter:
                if len(set(groups_dev[tr]) & set(groups_dev[va])):
                    raise RuntimeError("Group split contamination detected")
        else:
            inner = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=self.seed)
            cv_iter = list(inner.split(X_dev, y_dev))
        estimator = DecisionTreeModelFactory.create(self.backend, seed=self.seed, select_k=self.select_k)
        params = DecisionTreeModelFactory.distributions(self.backend)
        opts = dict(scoring=self.metric, cv=cv_iter, n_jobs=self.jobs,
                    refit=True, return_train_score=True, verbose=1, error_score="raise")
        if self.search_mode == "grid":
            search = GridSearchCV(estimator, param_grid=params, **opts)
        else:
            search = RandomizedSearchCV(estimator, param_distributions=params,
                                        n_iter=self.trials, random_state=self.seed, **opts)
        print(f"Training {self.backend.upper()} tree: {len(dev_idx)} development / {len(hold_idx)} holdout")
        print(f"CV folds={self.folds}, search={self.search_mode}, metric={self.metric}, group_cv={self.group_cv}")
        search.fit(X_dev, y_dev)
        best = search.best_estimator_
        X_hold = [X[i] for i in hold_idx]
        pred = np.asarray(best.predict(X_hold), dtype=np.int32)
        prob = np.asarray(best.predict_proba(X_hold)[:, 1], dtype=float)
        print(f"Best development CV {self.metric}: {search.best_score_:.4f}")
        print(f"Untouched holdout accuracy: {accuracy_score(y[hold_idx], pred):.4f}")
        print(f"Best parameters: {search.best_params_}")
        return TrainingResult(search, best, dev_idx, hold_idx, pred, prob, cv_iter)

    def refit_all(self, X, y, best_params):
        model = DecisionTreeModelFactory.create(self.backend, seed=self.seed, select_k=self.select_k)
        model.set_params(**best_params)
        model.fit(X, y)
        return model
