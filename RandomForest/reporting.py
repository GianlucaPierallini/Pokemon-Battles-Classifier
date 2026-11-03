"""Holdout-first reporting and verifiable feature-family coverage."""
from __future__ import annotations
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)


def feature_family(name: str) -> str:
    """Interpret encoded feature names, without assuming any feature is useful."""
    n = str(name).lower()
    if n.startswith("t") and re.match(r"^t\d{2}_", n):
        if "_move" in n:
            return "turn_moves"
        if "_status" in n or "_boost" in n or "_effect" in n:
            return "turn_status_boosts"
        if "_pokemon=" in n:
            return "turn_pokemon_identity"
        if "_hp" in n:
            return "turn_hp"
        return "turn_other"
    if n.startswith("team_slot_") or n.startswith("p2_lead_name=") or n.startswith("p2_lead_type_"):
        if "_name=" in n or "_pokemon=" in n or n.startswith("p2_lead_name="):
            return "static_species_identity"
        if "_item=" in n:
            return "static_items"
        if "_type_" in n:
            return "static_types"
        return "static_slots"
    if n.startswith(("team_", "opponent_lead_", "p1_total_base_", "p1_team_", "p1_leftovers_", "p1_mintberry_")):
        return "static_stats"
    if n.startswith("difference_"):
        if "hp" in n or "fainted" in n:
            return "aggregate_relative_hp"
        return "aggregate_relative_other"
    if "hp" in n or "fainted" in n:
        return "aggregate_hp"
    if "move" in n:
        return "aggregate_moves"
    if "status" in n or "boost" in n:
        return "aggregate_status_boosts"
    if "switch" in n or "species" in n:
        return "aggregate_switch_species"
    return "aggregate_other"


class RandomForestReporter:
    def __init__(self, output_dir: Path):
        self.dir = Path(output_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def json(self, name, obj):
        path = self.dir / name
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=self._serialize), encoding="utf-8")

    @staticmethod
    def _serialize(v):
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        raise TypeError(f"Cannot JSON encode {type(v)}")

    def search(self, search):
        df = pd.DataFrame(search.cv_results_)
        df.sort_values(["rank_test_score", "mean_test_score"],
                       ascending=[True, False]).to_csv(self.dir / "cv_search_results.csv", index=False)
        best_i = search.best_index_
        pd.DataFrame({
            "fold": np.arange(1, search.n_splits_ + 1),
            "accuracy": [search.cv_results_[f"split{i}_test_score"][best_i]
                         for i in range(search.n_splits_)],
        }).to_csv(self.dir / "cv_best_fold_scores.csv", index=False)
        df_top = df.sort_values("mean_test_score", ascending=False).head(12).iloc[::-1]
        fig, ax = plt.subplots(figsize=(10, max(4.5, 0.45 * len(df_top))))
        ax.barh(range(len(df_top)), df_top.mean_test_score,
                xerr=df_top.std_test_score, capsize=3)
        ax.set_yticks(range(len(df_top)), labels=[f"Rank {int(r)}" for r in df_top.rank_test_score])
        ax.set_xlim(left=0)
        ax.set_xlabel("CV accuracy (mean ± fold SD)")
        ax.set_title("Random Forest — hyperparameter configurations")
        fig.tight_layout()
        fig.savefig(self.dir / "cv_top_configs.png", dpi=160)
        plt.close(fig)

    def holdout(self, ids, actual, predicted, proba):
        values = np.asarray(actual)
        pred = np.asarray(predicted)
        prob = np.asarray(proba)
        matrix = confusion_matrix(values, pred, labels=[0, 1])
        # Wilson interval (descriptive sampling uncertainty; it does NOT undo
        # bias introduced by repeatedly viewing this holdout in past runs).
        correct = int(np.sum(values == pred))
        total = len(values)
        z = 1.95996398454
        fraction = correct / total
        denominator = 1 + z*z/total
        midpoint = (fraction + z*z/(2*total)) / denominator
        radius = z*np.sqrt(fraction*(1-fraction)/total + z*z/(4*total*total))/denominator
        metrics = {
            "correct": correct,
            "samples": total,
            "accuracy_wilson_95ci": [float(max(0, midpoint-radius)), float(min(1, midpoint+radius))],
            "majority_class_baseline_accuracy": float(max(np.mean(values == 0), np.mean(values == 1))),
            "accuracy": float(accuracy_score(values, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(values, pred)),
            "precision": float(precision_score(values, pred, zero_division=0)),
            "recall": float(recall_score(values, pred, zero_division=0)),
            "f1": float(f1_score(values, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(values, prob)),
            "confusion_matrix_labels_0_1": matrix.tolist(),
            "classification_report": classification_report(values, pred,
                                          output_dict=True, zero_division=0),
        }
        self.json("holdout_metrics.json", metrics)
        pd.DataFrame({"battle_id": ids, "actual": values,
                      "predicted": pred, "prob_win": prob}).to_csv(
                          self.dir / "holdout_predictions.csv", index=False)
        fig, ax = plt.subplots(figsize=(5.5, 4.6))
        ax.imshow(matrix, cmap="Blues")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=[0, 1],
               yticklabels=[0, 1], xlabel="Predicted", ylabel="Actual",
               title="Random Forest — holdout confusion matrix")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i,j]}", ha="center", va="center", fontsize=13,
                        color="white" if matrix[i,j] > matrix.max()/2 else "black")
        fig.tight_layout()
        fig.savefig(self.dir / "holdout_confusion_matrix.png", dpi=160)
        plt.close(fig)
        return metrics

    def diversity(self, fitted_pipeline, prefix="development"):
        """Count features actually USED in splits, not only offered to trees."""
        names = fitted_pipeline.named_steps["vectorizer"].get_feature_names_out()
        forest = fitted_pipeline.named_steps["forest"]
        gains = np.asarray(forest.feature_importances_, dtype=float)
        if len(names) != len(gains):
            raise RuntimeError("Feature name / forest importance mismatch")
        families = np.asarray([feature_family(n) for n in names])
        rows = []
        family_tree_counts = {}
        for j, tree in enumerate(forest.estimators_, start=1):
            used_nodes = tree.tree_.feature
            used_nodes = used_nodes[used_nodes >= 0]
            unique_used = np.unique(used_nodes)
            count_by_family = pd.Series(families[unique_used]).value_counts().to_dict()
            family_tree_counts[j] = count_by_family
            rows.append({
                "tree": j, "depth": tree.get_depth(),
                "leaves": tree.get_n_leaves(), "splits": len(used_nodes),
                "unique_features_used": len(unique_used),
                "feature_families_used": len(count_by_family),
                **{f"used_{f}": c for f, c in count_by_family.items()},
            })
        pd.DataFrame(rows).fillna(0).to_csv(self.dir / f"{prefix}_per_tree_diversity.csv", index=False)
        usage_counts = np.zeros(len(names), dtype=int)
        split_counts = np.zeros(len(names), dtype=int)
        for tree in forest.estimators_:
            used = tree.tree_.feature
            used = used[used >= 0]
            if len(used):
                usage_counts[np.unique(used)] += 1
                split_counts += np.bincount(used, minlength=len(names))
        df = pd.DataFrame({
            "feature": names, "family": families,
            "importance_gini": gains,
            "trees_using_feature": usage_counts,
            "total_splits_using_feature": split_counts,
        }).sort_values("importance_gini", ascending=False)
        df.to_csv(self.dir / f"{prefix}_feature_importances.csv", index=False)
        by_family = df.groupby("family", sort=True).agg(
            available_features=("feature", "count"),
            actually_used_features=("trees_using_feature", lambda v: int((v > 0).sum())),
            total_splits=("total_splits_using_feature", "sum"),
            gini_importance_sum=("importance_gini", "sum"),
        ).reset_index().sort_values("gini_importance_sum", ascending=False)
        by_family["trees_using_family"] = [
            sum(int(row.get(f, 0) > 0) for row in family_tree_counts.values())
            for f in by_family.family
        ]
        by_family.to_csv(self.dir / f"{prefix}_feature_family_coverage.csv", index=False)
        top = df.head(20).iloc[::-1]
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.barh(top.feature, top.importance_gini)
        ax.set_xlabel("Impurity-based feature importance (Gini)")
        ax.set_title("Random Forest — top 20 features")
        fig.tight_layout()
        fig.savefig(self.dir / f"{prefix}_feature_importances.png", dpi=160)
        plt.close(fig)
        return {
            "encoded_features": int(len(names)),
            "features_actually_used": int((usage_counts > 0).sum()),
            "features_never_used": int((usage_counts == 0).sum()),
            "number_of_trees": len(forest.estimators_),
            "mean_unique_features_per_tree": float(np.mean([r["unique_features_used"] for r in rows])),
            "families_used": int((by_family.actually_used_features > 0).sum()),
            "families_available": int(len(by_family)),
            "family_importance": dict(zip(by_family.family, by_family.gini_importance_sum)),
        }
