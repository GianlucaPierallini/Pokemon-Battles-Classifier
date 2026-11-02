r"""Train, cross-validate and export single-tree Pokemon battle classifier.

Example (from the project root):
    .\.venv\Scripts\python.exe .\DecisionTree\run_decision_tree.py --backend cpu
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Support both 'python -m DecisionTree.run_decision_tree' and direct Windows paths.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from DecisionTree.dataset import BattleDataset
from DecisionTree.features import BattleFeatureExtractor
from DecisionTree.model import DecisionTreeModelFactory
from DecisionTree.training import DecisionTreeTrainer
from DecisionTree.reporting import ExperimentReporter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THIS_DIR = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description="FDS single Decision Tree (CART CPU or single-tree XGBoost CUDA)")
    p.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--output-dir", type=Path, default=THIS_DIR / "results")
    p.add_argument("--backend", choices=["cpu", "cuda", "auto"], default="cpu")
    p.add_argument("--feature-set", choices=["engineered", "full"], default="full")
    p.add_argument("--turns", type=int, default=30)
    p.add_argument("--select-k", type=int, default=512,
                   help="Select top K features within each fold (0 disables; recommended for full)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--trials", type=int, default=12)
    p.add_argument("--search", choices=["random", "grid"], default="random")
    p.add_argument("--metric", choices=["accuracy", "roc_auc", "f1"], default="accuracy")
    p.add_argument("--group-cv", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--jobs", type=int, default=1, help="CPU parallel processes; keep 1 for GPU")
    p.add_argument("--max-train", type=int, default=None, help="Smoke-test first N battles (NOT for submission)")
    p.add_argument("--skip-submission", action="store_true", help="Do not load test set or write CSV")
    return p.parse_args()


def main():
    args = parse_args()
    if args.max_train is not None and args.max_train < 30:
        raise ValueError("--max-train must be >= 30")
    if args.search == "grid":
        print("WARNING: exhaustive grid is large. Prefer --search random for first experiments.")
    if args.backend == "cuda" and args.jobs != 1:
        raise ValueError("For CUDA use --jobs 1 to avoid multiple competing GPU fits")
    if args.select_k < 0:
        raise ValueError("--select-k must be >= 0")
    if args.turns < 1 or args.turns > 30:
        raise ValueError("--turns must be between 1 and 30")
    start = time.perf_counter()
    backend = DecisionTreeModelFactory.choose_backend(args.backend)
    if backend != args.backend:
        print(f"Backend selected: {backend} (requested: {args.backend})")
    if backend == "cuda" and args.jobs != 1:
        raise ValueError("Auto-selected CUDA requires --jobs 1")
    extractor = BattleFeatureExtractor(feature_set=args.feature_set, max_turns=args.turns)
    dataset = BattleDataset(args.data_dir, extractor)
    reporter = ExperimentReporter(args.output_dir)
    X, y, groups, ids = dataset.training(max_train=args.max_train)
    print(f"Read {len(X)} labeled battles (win rate: {y.mean():.3f}), feature set={args.feature_set}")
    trainer = DecisionTreeTrainer(
        backend=backend, folds=args.folds, trials=args.trials, seed=args.seed,
        metric=args.metric, group_cv=args.group_cv, search_mode=args.search,
        jobs=args.jobs, select_k=args.select_k,
    )
    result = trainer.fit_validate(X, y, groups)
    fold_scores = reporter.save_search(result.search, args.metric)
    hold = reporter.holdout(y[result.holdout_indices], result.pred_holdout, result.prob_holdout)
    pd.DataFrame({
        "battle_id": np.asarray(ids)[result.holdout_indices],
        "actual": y[result.holdout_indices],
        "predicted": result.pred_holdout,
        "prob_win": result.prob_holdout,
    }).to_csv(reporter.output_dir / "holdout_predictions.csv", index=False)
    tree_info, top_importances = reporter.tree_details(result.best_estimator, backend)
    final_model = trainer.refit_all(X, y, result.search.best_params_)
    joblib.dump({"model": final_model, "feature_set": args.feature_set,
                 "turns": args.turns, "backend": backend}, reporter.output_dir / "decision_tree_model.joblib")
    smoke = args.max_train is not None
    should_submit = not args.skip_submission and not smoke
    if should_submit:
        X_test, test_ids = dataset.unlabeled()
        predictions = final_model.predict(X_test).astype(int)
        sub_path, n_test = dataset.submission(test_ids, predictions,
                                             reporter.output_dir / "decision_tree_submission.csv")
        print(f"Submission: {sub_path} ({n_test} predictions)")
    elif smoke:
        print("Smoke test: submission disabled because --max-train was specified")
    else:
        print("Submission skipped by request")
    out = {
        "model_family": "CART decision tree" if backend == "cpu" else "XGBoost with exactly one tree (NOT CART)",
        "backend": backend, "feature_set": args.feature_set, "turns": args.turns,
        "select_k": args.select_k,
        "seed": args.seed, "group_cv": args.group_cv, "cv_folds": args.folds,
        "metric_used_for_selection": args.metric, "search": args.search,
        "candidates_tested": len(result.search.cv_results_["params"]),
        "n_train": len(X), "n_dev": len(result.dev_indices), "n_holdout": len(result.holdout_indices),
        "n_encoded_features": len(result.best_estimator.named_steps["vectorizer"].get_feature_names_out()),
        "n_selected_features": len(result.best_estimator.named_steps["tree"].feature_importances_),
        "best_cv_score_on_development": float(result.search.best_score_),
        "best_cv_fold_scores": fold_scores[args.metric].tolist(),
        "best_cv_accuracy_is_selection_biased": True,
        "holdout": hold, "best_params": result.search.best_params_,
        "tree_info": tree_info, "top_15_importances": top_importances,
        "submission_generated": should_submit, "smoke_test": smoke,
        "elapsed_seconds": time.perf_counter() - start,
    }
    reporter.dump_json("metrics.json", out)
    print(f"Done. Holdout accuracy={hold['accuracy']:.4f}, ROC-AUC={hold['roc_auc']:.4f}")
    print(f"Files in: {reporter.output_dir.resolve()}")
    print("NOTE: holdout is unbiased relative to the hyperparameter search; best CV score is not.")


if __name__ == "__main__":
    main()
