"""Artifacts: metrics, fold scores, confusion matrix, importances, model and CSV."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.tree import export_text


class ExperimentReporter:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def dump_json(self, name: str, data: dict):
        path = self.output_dir / name
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=self._json_default), encoding="utf-8")
        return path

    @staticmethod
    def _json_default(o):
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError(f"Unsupported JSON value: {type(o)}")

    def save_search(self, search, score_name: str):
        df = pd.DataFrame(search.cv_results_)
        columns = [c for c in df.columns if c.startswith("param_")]
        columns += ["mean_test_score", "std_test_score", "mean_train_score", "rank_test_score", "mean_fit_time"]
        columns += [c for c in df.columns if c.startswith("split") and c.endswith("_test_score")]
        df = df.loc[:, [c for c in columns if c in df.columns]].sort_values("rank_test_score")
        df.to_csv(self.output_dir / "cv_search_results.csv", index=False)
        selected = pd.DataFrame({
            "fold": list(range(1, search.n_splits_ + 1)),
            score_name: [search.cv_results_[f"split{i}_test_score"][search.best_index_]
                         for i in range(search.n_splits_)],
        })
        selected.to_csv(self.output_dir / "cv_best_fold_scores.csv", index=False)
        self.plot_search(df, score_name)
        return selected

    def plot_search(self, df, metric):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return
        top = df.head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(np.arange(len(top)), top["mean_test_score"], xerr=top["std_test_score"], capsize=2)
        ax.set_yticks(np.arange(len(top)), labels=[f"Rank {v}" for v in top["rank_test_score"]])
        ax.set_xlabel(f"CV {metric}")
        ax.set_title("Best hyperparameter configurations (CV)")
        fig.tight_layout()
        fig.savefig(self.output_dir / "cv_top_configs.png", dpi=140)
        plt.close(fig)

    def holdout(self, y_true, pred, proba):
        score = {
            "accuracy": float(accuracy_score(y_true, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, proba)) if len(set(y_true)) == 2 else None,
            "confusion_matrix_labels_0_1": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
            "classification_report": classification_report(y_true, pred, zero_division=0, output_dict=True),
        }
        self.dump_json("holdout_metrics.json", score)
        self.plot_confusion(score["confusion_matrix_labels_0_1"])
        return score

    def plot_confusion(self, cm):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["0", "1"],
               yticklabels=["0", "1"], xlabel="Predicted", ylabel="Actual", title="Holdout confusion matrix")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i][j]), ha="center", va="center", color="black")
        fig.tight_layout()
        fig.savefig(self.output_dir / "holdout_confusion_matrix.png", dpi=140)
        plt.close(fig)

    def tree_details(self, model, backend: str):
        tree = model.named_steps["tree"]
        names = model.named_steps["vectorizer"].get_feature_names_out()
        if "variance" in model.named_steps:
            names = names[model.named_steps["variance"].get_support()]
        if "selector" in model.named_steps:
            names = names[model.named_steps["selector"].get_support()]
        importances = np.asarray(tree.feature_importances_)
        if len(names) != len(importances):
            raise RuntimeError("Importance dimension does not match encoded feature names")
        importance = pd.DataFrame({"feature": names, "importance": importances})
        importance.sort_values("importance", ascending=False, inplace=True)
        importance.to_csv(self.output_dir / "feature_importances.csv", index=False)
        if backend == "cpu":
            text = export_text(tree, feature_names=list(names), max_depth=4, decimals=3)
            (self.output_dir / "tree_rules.txt").write_text(text, encoding="utf-8")
            tree_info = {"depth": tree.get_depth(), "leaves": tree.get_n_leaves()}
        else:
            tree_info = {"depth": None, "leaves": None, "note": "XGBoost single tree (not CART)"}
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            top = importance.head(20).iloc[::-1]
            fig, ax = plt.subplots(figsize=(11, 7))
            ax.barh(top.feature.astype(str), top.importance)
            ax.set_title("Top 20 feature importances")
            fig.tight_layout()
            fig.savefig(self.output_dir / "feature_importances.png", dpi=140)
            plt.close(fig)
        except ImportError:
            pass
        return tree_info, importance.head(15).to_dict(orient="records")
