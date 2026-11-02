"""Single-tree XGBoost factory. No neural networks, no external data."""
from __future__ import annotations

import shutil
import subprocess
from typing import Any

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class PrunedXGBoostFactory:
    """Build one XGBoost tree and CV parameter candidates.

    gamma is the minimum gain required to split. This is *not* the same as
    sklearn CART cost-complexity pruning (ccp_alpha).
    """

    @staticmethod
    def nvidia_present() -> bool:
        if shutil.which("nvidia-smi") is None:
            return False
        try:
            res = subprocess.run(["nvidia-smi", "-L"], text=True,
                                 capture_output=True, timeout=8, check=False)
            return res.returncode == 0 and "GPU" in res.stdout
        except (OSError, subprocess.TimeoutExpired):
            return False

    @classmethod
    def device(cls, requested: str) -> str:
        if requested == "cpu":
            return "cpu"
        present = cls.nvidia_present()
        if requested == "cuda" and not present:
            raise RuntimeError("CUDA requested, but NVIDIA GPU not detected by nvidia-smi.")
        return "cuda" if (requested == "cuda" or (requested == "auto" and present)) else "cpu"

    @staticmethod
    def build(device: str, seed: int, select_k: int = 512,
              reg_lambda: float = 5.0, reg_alpha: float = 0.1,
              subsample: float = 1.0, colsample_bytree: float = 1.0) -> Pipeline:
        from xgboost import XGBClassifier
        if select_k < 0:
            raise ValueError("select_k must be nonnegative")
        steps: list[tuple[str, Any]] = [("vectorizer", DictVectorizer(sparse=True, sort=True))]
        if select_k:
            # Every stage is fitted on the training portion of each CV split.
            # Match the previous DecisionTree/full XGBoost pipeline for comparisons.
            steps.extend([
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("variance", VarianceThreshold(threshold=0.0)),
                ("selector", SelectKBest(score_func=f_classif, k=select_k)),
            ])
        # No imputation if select_k=0: XGBoost accepts NaN values directly.
        steps.append(("tree", XGBClassifier(
            n_estimators=1, learning_rate=1.0, objective="binary:logistic",
            eval_metric="logloss", tree_method="hist", device=device,
            max_bin=256, random_state=seed, n_jobs=1,
            importance_type="gain", validate_parameters=True,
            reg_lambda=reg_lambda, reg_alpha=reg_alpha,
            subsample=subsample, colsample_bytree=colsample_bytree,
        )))
        return Pipeline(steps)

    @staticmethod
    def grid(depths: list[int], min_weights: list[float], gammas: list[float]) -> dict:
        if not depths or any(v < 1 for v in depths):
            raise ValueError("max_depth values must be >=1")
        if not min_weights or any(v < 0 for v in min_weights):
            raise ValueError("min_child_weight values must be >=0")
        if not gammas or any(v < 0 for v in gammas):
            raise ValueError("gamma values must be >=0")
        return {"tree__max_depth": depths,
                "tree__min_child_weight": min_weights,
                "tree__gamma": gammas}
