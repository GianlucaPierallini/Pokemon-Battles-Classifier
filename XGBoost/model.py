"""XGBoost v2 native Booster: fixed model as selected using 5-fold CV.

This module deliberately excludes Random Forest and hyperparameter searches.
Do not change parameters when reproducing the historic 67.45% local holdout run.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

class BoostedTreeClassifier(ClassifierMixin, BaseEstimator):
    """Cloneable native XGBoost boosted ensemble, not a single tree.

    CSR input keeps ALL encoded features sparse. XGBoost interprets implicit
    sparse zeros as missing in its tree booster: this is documented by XGBoost.
    Boostr.predict(DMatrix) deliberately avoids GPU/CPU inplace_predict mismatch.
    """
    def __init__(self, device='cpu', random_state=42, n_jobs=2,
                 n_estimators=300, learning_rate=0.05, max_depth=4,
                 min_child_weight=10, gamma=0., subsample=0.8,
                 colsample_bytree=0.8, reg_lambda=3., reg_alpha=0.1):
        self.device = device
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha

    @staticmethod
    def _dmatrix(X, label=None):
        import xgboost as xgb
        if sparse.issparse(X):
            X = X.tocsr().astype(np.float32)
        else:
            X = np.asarray(X, dtype=np.float32)
        return xgb.DMatrix(X, label=label, missing=np.nan)

    def fit(self, X, y):
        import xgboost as xgb
        labels = np.asarray(y, dtype=np.int32)
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError('XGBoost expects both classes 0 and 1')
        self.classes_ = np.array([0, 1], dtype=np.int32)
        self.n_features_in_ = X.shape[1]
        params = {
            'objective': 'binary:logistic', 'eval_metric': 'logloss',
            'tree_method': 'hist', 'device': self.device,
            'seed': self.random_state, 'nthread': self.n_jobs,
            'eta': self.learning_rate, 'max_depth': self.max_depth,
            'min_child_weight': self.min_child_weight, 'gamma': self.gamma,
            'subsample': self.subsample, 'colsample_bytree': self.colsample_bytree,
            'lambda': self.reg_lambda, 'alpha': self.reg_alpha,
            'max_bin': 256,
        }
        # Native DMatrix works on CPU-backed scipy CSR with GPU training; the
        # booster transfers input as needed and predicts via Booster.predict.
        self.booster_ = xgb.train(params, self._dmatrix(X, labels),
                                  num_boost_round=self.n_estimators,
                                  verbose_eval=False)
        return self

    def predict_proba(self, X):
        p1 = np.asarray(self.booster_.predict(self._dmatrix(X)), dtype=np.float64)
        return np.column_stack((1. - p1, p1))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int32)

    @property
    def feature_importances_(self):
        scores = self.booster_.get_score(importance_type='total_gain')
        importance = np.zeros(self.n_features_in_, dtype=float)
        for key, value in scores.items():
            index = int(key[1:]) if key.startswith('f') and key[1:].isdigit() else None
            if index is not None and index < len(importance):
                importance[index] = value
        if importance.sum() > 0:
            importance /= importance.sum()
        return importance


# Hyperparameters selected by CV in EnsembleV2, NOT fitted to Kaggle labels.
BEST_XGB_PARAMS = {
    "booster__n_estimators": 500,
    "booster__max_depth": 3,
    "booster__learning_rate": 0.03,
    "booster__min_child_weight": 5,
    "booster__gamma": 0.0,
    "booster__subsample": 0.85,
    "booster__colsample_bytree": 0.80,
    "booster__reg_lambda": 8.0,
    "booster__reg_alpha": 0.1,
}


def build_model(seed: int = 42, jobs: int = 4, device: str = "cpu") -> Pipeline:
    """Sparse DictVectorizer fitted on training only, then 500 boosted trees."""
    model = Pipeline([
        ("vectorizer", DictVectorizer(sparse=True, sort=True)),
        ("booster", BoostedTreeClassifier(device=device, random_state=seed, n_jobs=jobs)),
    ])
    return model.set_params(**BEST_XGB_PARAMS)
