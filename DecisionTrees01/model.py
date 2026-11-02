"""Two deliberately different single-tree models: sklearn CART and XGBoost GPU."""
from __future__ import annotations
import shutil
import subprocess
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.tree import DecisionTreeClassifier


class DecisionTreeModelFactory:
    @staticmethod
    def gpu_detected() -> bool:
        if not shutil.which("nvidia-smi"):
            return False
        try:
            result = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                                    text=True, timeout=10, check=False)
            return result.returncode == 0 and "GPU" in result.stdout
        except (OSError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def choose_backend(value: str) -> str:
        if value == "cpu":
            return "cpu"
        gpu = DecisionTreeModelFactory.gpu_detected()
        if value == "cuda":
            if not gpu:
                raise RuntimeError("No NVIDIA GPU found via nvidia-smi. Use --backend cpu.")
            try:
                import xgboost  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("Install requirements-cuda.txt to use CUDA.") from exc
            return "cuda"
        if gpu:
            try:
                import xgboost  # noqa: F401
                return "cuda"
            except ImportError:
                pass
        return "cpu"

    @staticmethod
    def create(backend: str, seed: int = 42, select_k: int = 512):
        steps = [("vectorizer", DictVectorizer(sparse=True, sort=True))]
        if backend == "cpu":
            steps.append(("imputer", SimpleImputer(strategy="median", keep_empty_features=True)))
            if select_k > 0:
                # Univariate filtering is fitted *inside* each training fold.
                steps.append(("variance", VarianceThreshold(threshold=0.0)))
                steps.append(("selector", SelectKBest(score_func=f_classif, k=select_k)))
            steps.append(("tree", DecisionTreeClassifier(random_state=seed, criterion="gini")))
        elif backend == "cuda":
            try:
                from xgboost import XGBClassifier
            except ImportError as exc:
                raise RuntimeError("Missing xgboost; install requirements-cuda.txt") from exc
            # XGBoost supports NaNs directly; NaN cannot pass through f_classif.
            # Keep missing features as the supplied source intended on CUDA.
            if select_k > 0:
                steps.append(("imputer", SimpleImputer(strategy="median", keep_empty_features=True)))
                steps.append(("variance", VarianceThreshold(threshold=0.0)))
                steps.append(("selector", SelectKBest(score_func=f_classif, k=select_k)))
            steps.append(("tree", XGBClassifier(
                n_estimators=1, learning_rate=1.0, objective="binary:logistic",
                eval_metric="logloss", tree_method="hist", device="cuda",
                max_bin=256, random_state=seed, n_jobs=1,
                importance_type="gain", validate_parameters=True,
            )))
            # XGBoost handles NaNs itself; sparse absent entries are interpreted
            # as missing rather than numeric zero (different from sklearn CART).
        else:
            raise ValueError(f"Unknown backend: {backend}")
        return Pipeline(steps)

    @staticmethod
    def distributions(backend: str) -> dict:
        if backend == "cpu":
            return {
                "tree__criterion": ["gini", "entropy"],
                "tree__max_depth": [3, 4, 5, 6, 8, 10, 12, None],
                "tree__min_samples_leaf": [5, 10, 20, 40, 80],
                "tree__min_samples_split": [10, 20, 50],
                "tree__ccp_alpha": [0.0, 0.0001, 0.0005, 0.002],
            }
        return {
            "tree__max_depth": [3, 4, 5, 6, 8, 10, 12],
            "tree__min_child_weight": [1, 3, 7, 15, 30],
            "tree__gamma": [0.0, 0.1, 0.5, 1.0, 3.0],
            "tree__reg_lambda": [0.1, 1.0, 5.0, 15.0],
            "tree__reg_alpha": [0.0, 0.1, 1.0],
            "tree__subsample": [0.8, 1.0],
            "tree__colsample_bytree": [0.75, 1.0],
        }
