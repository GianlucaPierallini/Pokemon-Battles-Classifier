"""Development CV for model selection; holdout evaluated once afterward."""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedGroupKFold, StratifiedKFold,
    train_test_split,
)

from .model import RandomForestFactory


@dataclass
class ForestTrainingResult:
    search: object
    model_dev: object
    development_indices: np.ndarray
    holdout_indices: np.ndarray
    holdout_pred: np.ndarray
    holdout_proba: np.ndarray
    cv_splits: list


class RandomForestTrainer:
    def __init__(self, seed=42, folds=3, trials=8, jobs=1,
                 group_cv=False, holdout_fraction=0.20, distributions=None):
        if folds < 2 or trials < 1 or not (0 < holdout_fraction < 0.5):
            raise ValueError("Invalid folds, trials, or holdout_fraction")
        self.seed = seed
        self.folds = folds
        self.trials = trials
        self.jobs = jobs
        self.group_cv = group_cv
        self.holdout_fraction = holdout_fraction
        self.distributions = distributions or RandomForestFactory.distributions()

    @staticmethod
    def _rows(X, indices):
        return [X[int(i)] for i in indices]

    def run(self, X: list[dict], y: np.ndarray, groups: np.ndarray):
        indices = np.arange(len(y))
        if self.group_cv:
            n_splits = max(2, round(1.0 / self.holdout_fraction))
            outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                        random_state=self.seed)
            dev_idx, hold_idx = next(outer.split(indices, y, groups))
        else:
            dev_idx, hold_idx = train_test_split(
                indices, test_size=self.holdout_fraction,
                stratify=y, random_state=self.seed
            )
        if len(np.unique(y[dev_idx])) != 2 or len(np.unique(y[hold_idx])) != 2:
            raise ValueError("Both development and holdout must have both classes")
        X_dev = self._rows(X, dev_idx)
        y_dev = y[dev_idx]
        if self.group_cv:
            gdev = groups[dev_idx]
            splitter = StratifiedGroupKFold(n_splits=self.folds, shuffle=True,
                                            random_state=self.seed)
            cv_splits = list(splitter.split(X_dev, y_dev, gdev))
            if set(groups[dev_idx]).intersection(groups[hold_idx]):
                raise RuntimeError("Group leakage between development and holdout")
            for it, iv in cv_splits:
                if set(gdev[it]).intersection(gdev[iv]):
                    raise RuntimeError("Group leakage within CV")
        else:
            splitter = StratifiedKFold(n_splits=self.folds, shuffle=True,
                                      random_state=self.seed)
            cv_splits = list(splitter.split(X_dev, y_dev))
        if any(len(np.unique(y_dev[t])) < 2 or len(np.unique(y_dev[v])) < 2
               for t, v in cv_splits):
            raise ValueError("Each CV fold must contain both classes")

        # Forest threads are set via --jobs; the search itself is serial to
        # avoid multiplying RAM usage by the number of concurrent fits.
        pipeline = RandomForestFactory.create(seed=self.seed, jobs=self.jobs)
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=self.distributions,
            n_iter=self.trials,
            scoring="accuracy",
            cv=cv_splits,
            random_state=self.seed,
            n_jobs=1,
            refit=True,
            return_train_score=True,
            verbose=1,
            error_score="raise",
        )
        print(f"Development={len(dev_idx)}, untouched holdout={len(hold_idx)}")
        print(f"RandomizedSearchCV: {self.trials} candidates × {self.folds} folds")
        search.fit(X_dev, y_dev)
        best = search.best_estimator_
        X_holdout = self._rows(X, hold_idx)
        predictions = np.asarray(best.predict(X_holdout), dtype=int)
        probabilities = np.asarray(best.predict_proba(X_holdout)[:, 1], dtype=float)
        print("Selected CV accuracy:", round(float(search.best_score_), 5))
        print("Holdout accuracy:", round(float(accuracy_score(y[hold_idx], predictions)), 5))
        print("Selected params:", search.best_params_)
        return ForestTrainingResult(search, best, dev_idx, hold_idx,
                                    predictions, probabilities, cv_splits)

    def refit_all(self, X, y, params):
        final = RandomForestFactory.create(seed=self.seed, jobs=self.jobs)
        final.set_params(**params)
        final.fit(X, y)
        return final
