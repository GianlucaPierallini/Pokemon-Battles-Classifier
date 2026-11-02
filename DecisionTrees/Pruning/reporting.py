"""Save reproducible experiment results, CV figures and interpretable tree data."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
    classification_report, ConfusionMatrixDisplay,
)


class PruningReporter:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, name, obj):
        path = self.output_dir / name
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                                   default=self._default), encoding="utf-8")
        return path

    @staticmethod
    def _default(v):
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        raise TypeError(f"Cannot JSON serialize {type(v)}")

    def cv_results(self, search, metric):
        data = pd.DataFrame(search.cv_results_)
        data = data.sort_values("rank_test_score").reset_index(drop=True)
        data.to_csv(self.output_dir / "pruning_cv_search.csv", index=False)
        # Exploratory pruning diagnostics: averages over evaluated parameter
        # combinations. These do NOT represent controlled single-variable effects
        # when the search uses random samples of a non-factorial grid.
        self.pruning_effects(data, metric)
        n = min(15, len(data))
        top = data.head(n).iloc[::-1]
        fig, ax = plt.subplots(figsize=(10, max(4, n * 0.42)))
        ax.barh([f"Rank {int(x)}" for x in top["rank_test_score"]],
                top["mean_test_score"], xerr=top["std_test_score"], capsize=2)
        ax.set_xlabel(f"CV {metric} (mean ± fold SD)")
        ax.set_title("XGBoost single-tree pruning — best configurations")
        fig.tight_layout()
        fig.savefig(self.output_dir / "pruning_cv_top.png", dpi=160)
        plt.close(fig)
        folds = np.asarray([search.cv_results_[f"split{i}_test_score"][search.best_index_]
                            for i in range(search.n_splits_)], dtype=float)
        pd.DataFrame({"fold": np.arange(1, len(folds)+1), metric: folds}).to_csv(
            self.output_dir / "pruning_best_fold_scores.csv", index=False)
        return folds

    def pruning_effects(self, scores, metric):
        for param, name in [("param_tree__max_depth", "depth"),
                            ("param_tree__gamma", "gamma"),
                            ("param_tree__min_child_weight", "min_child_weight")]:
            if param not in scores.columns:
                continue
            data = scores[[param, "mean_test_score", "mean_train_score"]].copy()
            data[param] = pd.to_numeric(data[param], errors="coerce")
            avg = (data.groupby(param, dropna=True)[["mean_test_score", "mean_train_score"]]
                   .mean().sort_index())
            if avg.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(avg.index, avg["mean_train_score"], marker="o", label="Training CV")
            ax.plot(avg.index, avg["mean_test_score"], marker="o", label="Validation CV")
            ax.set_title(f"Pruning sensitivity — {name}")
            ax.set_xlabel(name)
            ax.set_ylabel(f"Mean {metric} across evaluated configurations")
            ax.legend()
            fig.tight_layout()
            fig.savefig(self.output_dir / f"pruning_effect_{name}.png", dpi=160)
            plt.close(fig)

    def holdout(self, actual, pred, probability):
        cm = confusion_matrix(actual, pred, labels=[0, 1])
        auc = float(roc_auc_score(actual, probability)) if len(set(actual)) > 1 else None
        metrics = {
            "accuracy": float(accuracy_score(actual, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(actual, pred)),
            "precision": float(precision_score(actual, pred, zero_division=0)),
            "recall": float(recall_score(actual, pred, zero_division=0)),
            "f1": float(f1_score(actual, pred, zero_division=0)),
            "roc_auc": auc,
            "confusion_matrix_labels_0_1": cm.tolist(),
            "classification_report": classification_report(actual, pred, output_dict=True,
                                                            zero_division=0),
        }
        self.write_json("pruning_holdout_metrics.json", metrics)
        disp = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
        disp.plot(values_format="d", cmap="Blues")
        plt.title("Holdout confusion matrix — pruned single tree")
        plt.tight_layout()
        plt.savefig(self.output_dir / "pruning_holdout_confusion_matrix.png", dpi=160)
        plt.close()
        return metrics

    def features(self, model):
        # Importance vectors correspond to selected columns, not all vectorized features.
        names = model.named_steps["vectorizer"].get_feature_names_out()
        if "variance" in model.named_steps:
            names = names[model.named_steps["variance"].get_support()]
        if "selector" in model.named_steps:
            names = names[model.named_steps["selector"].get_support()]
        importances = np.asarray(model.named_steps["tree"].feature_importances_)
        if len(names) != len(importances):
            raise ValueError("Feature names and importances don't align")
        frame = pd.DataFrame({"feature": names, "gain_importance": importances})
        frame.sort_values("gain_importance", ascending=False, inplace=True)
        frame.to_csv(self.output_dir / "pruning_feature_importances.csv", index=False)
        nonzero = frame[frame.gain_importance > 0].head(20).iloc[::-1]
        if not nonzero.empty:
            fig, ax = plt.subplots(figsize=(11, max(5, 0.33*len(nonzero))))
            ax.barh(nonzero.feature, nonzero.gain_importance)
            ax.set_xlabel("XGBoost normalized gain importance")
            ax.set_title("Feature importance — selected pruned tree")
            fig.tight_layout()
            fig.savefig(self.output_dir / "pruning_feature_importances.png", dpi=160)
            plt.close(fig)
        tree = model.named_steps["tree"]
        (self.output_dir / "pruning_best_tree.json").write_text(
            tree.get_booster().get_dump(dump_format="json")[0], encoding="utf-8")
        return frame.head(20).to_dict(orient="records"), len(names)
