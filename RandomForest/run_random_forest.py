"""Run the holdout-first Random Forest study from Windows or Linux."""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from sklearn.metrics import accuracy_score

from RandomForest.dataset import BattleDataset
from RandomForest.improved_features import ImprovedBattleFeatureExtractor
from RandomForest.model import RandomForestFactory
from RandomForest.reporting import RandomForestReporter
from RandomForest.training import RandomForestTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def int_csv(v):
    try:
        values = [int(s.strip()) for s in v.split(",") if s.strip()]
        if not values or any(x < 1 for x in values):
            raise ValueError()
        return list(dict.fromkeys(values))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated positive integers") from exc


def depth_csv(v):
    try:
        values = [None if s.strip().lower() == "none" else int(s.strip())
                  for s in v.split(",") if s.strip()]
        if not values or any(x is not None and x < 1 for x in values):
            raise ValueError()
        return list(dict.fromkeys(values))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use e.g. 8,14,22,None") from exc


def feature_csv(v):
    try:
        values = []
        for s in v.split(","):
            s = s.strip().lower()
            if not s:
                continue
            if s in ("sqrt", "log2"):
                values.append(s)
            else:
                x = float(s)
                if not (0 < x <= 1):
                    raise ValueError()
                values.append(x)
        if not values:
            raise ValueError()
        return list(dict.fromkeys(values))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use e.g. sqrt,log2,0.02,0.05") from exc


def parser():
    p = argparse.ArgumentParser(
        description="Random Forest on all Pokémon battle features; holdout-primary evaluation")
    p.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--output-dir", type=Path, default=HERE / "results")
    p.add_argument("--feature-set", choices=list(ImprovedBattleFeatureExtractor.PROFILES), default="rich_compact")
    p.add_argument("--turns", type=int, default=30)
    p.add_argument("--folds", type=int, default=3,
                   help="CV folds for tuning on the development subset")
    p.add_argument("--trials", type=int, default=8,
                   help="RandomizedSearchCV candidate configurations")
    p.add_argument("--jobs", type=int, default=2,
                   help="CPU threads used within each forest; search remains sequential")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--holdout-fraction", type=float, default=0.2)
    p.add_argument("--group-cv", action="store_true",
                   help="Separate identical six-species teams in both holdout and CV")
    p.add_argument("--n-estimators", type=int_csv, default=[200, 400])
    p.add_argument("--max-depths", type=depth_csv, default=[10, 20, 30, None])
    p.add_argument("--min-leaf", type=int_csv, default=[1, 2, 4, 8])
    p.add_argument("--max-features", type=feature_csv,
                   default=["sqrt", "log2", 0.02, 0.05, 0.1, 0.2],
                   help="Features examined at each split; all encoded features remain available")
    p.add_argument("--max-train", type=int, default=None,
                   help="Smoke test on first N training battles (no official submission)")
    p.add_argument("--skip-submission", action="store_true")
    p.add_argument("--no-refit-all", action="store_true",
                   help="Skip final full-data refit and test submission")
    return p


def main():
    a = parser().parse_args()
    if not 1 <= a.turns <= 30 or a.folds < 2 or a.trials < 1 or a.jobs < 1:
        raise ValueError("Invalid turns/folds/trials/jobs")
    if a.max_train is not None and a.max_train < 30:
        raise ValueError("--max-train requires >= 30")
    t0 = time.perf_counter()
    print("Model: sklearn RandomForestClassifier (CPU; NOT a neural network)")
    print("Full feature vocabulary; no supervised top-512 selection")
    extractor = ImprovedBattleFeatureExtractor(a.feature_set, a.turns)
    data = BattleDataset(a.data_dir, extractor)
    X, y, groups, ids = data.training(max_train=a.max_train)
    print(f"Loaded {len(X)} labeled battles; positive class proportion {y.mean():.3f}")
    print("NOTE: holdout is for reporting, never for parameter selection.")
    print("CAUTION: earlier experiments already viewed this seed=42 holdout; ")
    print("do not treat repeated holdout checks as an unbiased model-selection metric.")
    distributions = RandomForestFactory.distributions(
        a.n_estimators, a.max_depths, a.min_leaf, a.max_features
    )
    trainer = RandomForestTrainer(
        seed=a.seed, folds=a.folds, trials=a.trials, jobs=a.jobs,
        group_cv=a.group_cv, holdout_fraction=a.holdout_fraction,
        distributions=distributions,
    )
    report = RandomForestReporter(a.output_dir)
    result = trainer.run(X, y, groups)
    report.search(result.search)
    hold = result.holdout_indices
    metrics = report.holdout(np.asarray(ids)[hold], y[hold],
                             result.holdout_pred, result.holdout_proba)
    selected = result.search.best_estimator_
    development_coverage = report.diversity(selected, prefix="development")
    train_dev_acc = accuracy_score(y[result.development_indices],
                                   selected.predict([X[int(i)] for i in result.development_indices]))
    submission_created = False
    if not a.no_refit_all:
        model = trainer.refit_all(X, y, result.search.best_params_)
        final_coverage = report.diversity(model, prefix="final")
        bundle = {
            "model": model, "feature_set": a.feature_set,
            "turns": a.turns, "seed": a.seed,
            "best_params": result.search.best_params_,
        }
        joblib.dump(bundle, report.dir / "random_forest_model.joblib", compress=3)
        if a.max_train is None and not a.skip_submission:
            X_test, test_ids = data.unlabeled()
            preds = np.asarray(model.predict(X_test), dtype=int)
            filename, size = data.submission(test_ids, preds,
                                              report.dir / "random_forest_submission.csv")
            print(f"Submission saved: {filename} ({size} predictions)")
            submission_created = True
    else:
        final_coverage = None
    output = {
        "model": "RandomForestClassifier (bagged decision trees, non-neural)",
        "feature_set": a.feature_set, "turns": a.turns,
        "feature_selection": "none; entire DictVectorizer feature space",
        "device": "cpu", "seed": a.seed,
        "train_samples": len(X),
        "development_samples": len(result.development_indices),
        "holdout_samples": len(hold),
        "group_split": bool(a.group_cv),
        "folds": a.folds, "candidates": len(result.search.cv_results_["params"]),
        "best_params": result.search.best_params_,
        "development_best_cv_accuracy": float(result.search.best_score_),
        "development_training_accuracy": float(train_dev_acc),
        "holdout": metrics,
        "development_feature_diversity": development_coverage,
        "final_feature_diversity": final_coverage,
        "submission_generated": submission_created,
        "smoke_test": a.max_train is not None,
        "holdout_reuse_warning": "Seed 42 holdout previously inspected in earlier experiments; repeated use biases model selection.",
        "cv_selection_warning": "The best CV score is selection-biased; the holdout is independent of this run's parameter search.",
        "elapsed_seconds": float(time.perf_counter() - t0),
    }
    report.json("metrics.json", output)
    print(f"\nHOLDOUT PRIMARY: accuracy={metrics['accuracy']:.4%}, "
          f"ROC-AUC={metrics['roc_auc']:.4f}, F1={metrics['f1']:.4f}")
    print(f"FEATURE COVERAGE (dev): {development_coverage['features_actually_used']} "
          f"/ {development_coverage['encoded_features']} encoded features used by forest splits")
    print(f"Features families used: {development_coverage['families_used']} "
          f"/ {development_coverage['families_available']}")
    print("All outputs:", report.dir.resolve())


if __name__ == "__main__":
    main()
