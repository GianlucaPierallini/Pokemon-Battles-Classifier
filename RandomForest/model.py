"""Sparse, fold-safe Random Forest pipeline.

The full vocabulary is fitted on the training portion of each CV split.
All engineered and turn-level features remain candidates for forest splits:
there is deliberately NO SelectKBest(512).
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class RandomForestFactory:
    @staticmethod
    def create(seed: int = 42, jobs: int = 1, n_estimators: int = 120) -> Pipeline:
        return Pipeline([
            ("vectorizer", DictVectorizer(sparse=True, sort=True)),
            # Replace explicit NaNs, preserving the sparse one-hot representation.
            ("imputer", SimpleImputer(
                strategy="constant", fill_value=0, keep_empty_features=True
            )),
            ("forest", RandomForestClassifier(
                n_estimators=n_estimators,
                criterion="gini",
                max_features="sqrt",
                bootstrap=True,
                random_state=seed,
                n_jobs=jobs,
                class_weight=None,
            )),
        ])

    @staticmethod
    def distributions(
        estimators=(200, 400),
        max_depths=(10, 20, 30, None),
        leaf_sizes=(1, 2, 4, 8),
        feature_choices=("sqrt", "log2", 0.02, 0.05, 0.1, 0.2),
    ) -> dict:
        return {
            "forest__n_estimators": list(estimators),
            "forest__max_depth": list(max_depths),
            "forest__min_samples_leaf": list(leaf_sizes),
            "forest__max_features": list(feature_choices),
            "forest__min_samples_split": [2, 6, 12],
            "forest__max_samples": [0.8, 1.0],
        }
