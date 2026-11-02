r"""Windows/Linux entry point: XGBoost *single* tree + pruning CV.

From project root:
    .\.venv\Scripts\python.exe -m DecisionTree.Pruning.run_pruning --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Direct execution from Windows as well as `python -m DecisionTree.Pruning.run_pruning`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd

from DecisionTree.dataset import BattleDataset
from DecisionTree.features import BattleFeatureExtractor
from DecisionTree.Pruning.model import PrunedXGBoostFactory
from DecisionTree.Pruning.reporting import PruningReporter
from DecisionTree.Pruning.training import PruningCrossValidator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def csv_numbers(text, caster):
    try:
        values = [caster(v.strip()) for v in text.split(",") if v.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected comma-separated numbers, got {text!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError("Specify at least one value")
    return list(dict.fromkeys(values))


def parser():
    p = argparse.ArgumentParser(description="XGBoost single-tree pruning search (no neural networks)")
    p.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--output-dir", type=Path, default=HERE / "results")
    p.add_argument("--device", "--backend", choices=["cpu", "cuda", "auto"], default="cpu")
    p.add_argument("--feature-set", choices=["full", "engineered"], default="full")
    p.add_argument("--turns", type=int, default=30)
    p.add_argument("--select-k", type=int, default=512,
                   help="Feature selector per CV fold; 0 keeps all features")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--search", choices=["random", "grid"], default="random")
    p.add_argument("--trials", type=int, default=18,
                   help="Random search trials; ignored for exhaustive grid")
    p.add_argument("--metric", choices=["accuracy", "roc_auc", "f1"], default="accuracy")
    p.add_argument("--group-cv", action="store_true",
                   help="Separate team compositions in holdout and inner folds")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--reg-lambda", type=float, default=5.0)
    p.add_argument("--reg-alpha", type=float, default=0.1)
    p.add_argument("--subsample", type=float, default=1.0)
    p.add_argument("--colsample-bytree", type=float, default=1.0)
    p.add_argument("--depths", default="2,3,4,5,6")
    p.add_argument("--min-child-weights", default="5,15,30,50")
    p.add_argument("--gammas", default="0,0.5,1,2,5")
    p.add_argument("--compare-previous", action="store_true",
                   help="CV of fixed previous model on the SAME development folds")
    p.add_argument("--max-train", type=int, default=None,
                   help="Development smoke test: process first N samples, do not submit")
    p.add_argument("--skip-submission", action="store_true")
    return p


def main():
    args = parser().parse_args()
    if not 1 <= args.turns <= 30:
        raise ValueError("--turns must be between 1 and 30")
    if args.max_train is not None and args.max_train < 30:
        raise ValueError("--max-train must be >= 30")
    if args.select_k < 0:
        raise ValueError("--select-k must be >= 0")
    if args.reg_lambda < 0 or args.reg_alpha < 0:
        raise ValueError("L1 and L2 regularization must be nonnegative")
    if not 0 < args.subsample <= 1 or not 0 < args.colsample_bytree <= 1:
        raise ValueError("Sampling fractions must be in (0,1]")
    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")

    started = time.perf_counter()
    device = PrunedXGBoostFactory.device(args.device)
    if device == "cuda" and args.jobs != 1:
        raise ValueError("Only --jobs 1 is allowed for CUDA to avoid GPU contention")
    if device != args.device:
        print(f"Selected {device} for --device {args.device}")

    depths = csv_numbers(args.depths, int)
    weights = csv_numbers(args.min_child_weights, float)
    gammas = csv_numbers(args.gammas, float)
    param_grid = PrunedXGBoostFactory.grid(depths, weights, gammas)
    extractor = BattleFeatureExtractor(feature_set=args.feature_set, max_turns=args.turns)
    dataset = BattleDataset(args.data_dir, extractor)
    features, labels, groups, battle_ids = dataset.training(max_train=args.max_train)
    print(f"Dataset: {len(features)} labeled battles, positive fraction {labels.mean():.3f}")
    print(f"Feature set: {args.feature_set}, first {args.turns} turns; selected max K={args.select_k}")
    reporter = PruningReporter(args.output_dir)

    model = PrunedXGBoostFactory.build(
        device, args.seed, args.select_k, args.reg_lambda, args.reg_alpha,
        args.subsample, args.colsample_bytree)
    tester = PruningCrossValidator(folds=args.folds, seed=args.seed,
                                  group_cv=args.group_cv, metric=args.metric,
                                  search=args.search, trials=args.trials, jobs=args.jobs)
    # Previous experiment's exact fixed hyperparameters. This is an OPTIONAL
    # development-only cross-validation comparison using the SAME split.
    baseline_estimator = None
    baseline_params = None
    if args.compare_previous:
        if args.feature_set != "full" or args.select_k != 512 or args.turns != 30:
            print("WARNING: Previous-model CV comparison is only feature-equivalent with full/512/30.")
        baseline_estimator = PrunedXGBoostFactory.build(
            device, args.seed, args.select_k, reg_lambda=1.0, reg_alpha=0.1,
            subsample=0.8, colsample_bytree=0.75)
        baseline_params = {
            "tree__max_depth": 6, "tree__min_child_weight": 30,
            "tree__gamma": 0.0,
        }
    result = tester.run(features, labels, groups, model, param_grid,
                        baseline_estimator=baseline_estimator,
                        baseline_params=baseline_params)
    fold_scores = reporter.cv_results(result.search, args.metric)
    holdout_idx = result.holdout_indices
    holdout_metrics = reporter.holdout(labels[holdout_idx],
                                      result.holdout_pred, result.holdout_prob)
    pd.DataFrame({
        "battle_id": np.asarray(battle_ids)[holdout_idx],
        "actual": labels[holdout_idx],
        "predicted": result.holdout_pred,
        "prob_win": result.holdout_prob,
    }).to_csv(reporter.output_dir / "pruning_holdout_predictions.csv", index=False)
    important, n_features = reporter.features(result.search.best_estimator_)

    # Refit only with labels in train.jsonl after evaluating the development holdout.
    final = tester.refit_all(model, result.search.best_params_, features, labels)
    joblib.dump({
        "model": final, "feature_set": args.feature_set,
        "turns": args.turns, "select_k": args.select_k,
        "device": device,
    }, reporter.output_dir / "pruned_single_tree_model.joblib")
    do_submission = args.max_train is None and not args.skip_submission
    if do_submission:
        test_features, test_ids = dataset.unlabeled()
        predicted = np.asarray(final.predict(test_features), dtype=int)
        output, rows = dataset.submission(test_ids, predicted,
                                         reporter.output_dir / "pruning_submission.csv")
        print(f"Submission written: {output} ({rows} samples)")
    else:
        print("Submission skipped (smoke test or --skip-submission)")
    best = result.search.best_estimator_
    report = {
        "model": "XGBoost single tree (not CART and not a boosted ensemble)",
        "device": device, "feature_set": args.feature_set,
        "turns": args.turns, "select_k": args.select_k,
        "n_train": len(features), "n_dev": len(result.dev_indices),
        "n_holdout": len(result.holdout_indices),
        "n_encoded_features": len(best.named_steps["vectorizer"].get_feature_names_out()),
        "n_selected_features": n_features,
        "search_mode": args.search,
        "candidates_tested": len(result.search.cv_results_["params"]),
        "folds": args.folds, "group_cv": args.group_cv,
        "random_state": args.seed, "scoring": args.metric,
        "fixed_params": {"reg_lambda": args.reg_lambda,
                         "reg_alpha": args.reg_alpha,
                         "subsample": args.subsample,
                         "colsample_bytree": args.colsample_bytree},
        "best_params": result.search.best_params_,
        "best_cv_score": float(result.search.best_score_),
        "best_cv_fold_scores": fold_scores.tolist(),
        "best_cv_train_score": float(result.search.cv_results_["mean_train_score"][result.search.best_index_]),
        "dev_fit_accuracy": result.train_accuracy,
        "holdout": holdout_metrics,
        "previous_fixed_config_cv": result.baseline_cv,
        "top_20_importances": important,
        "holdout_prior_exposure_warning": "The project has already inspected a holdout of this dataset; repeated use for experiment selection biases the final performance estimate.",
        "cv_selection_warning": "The best CV score is selection-biased; use holdout or nested CV for assessment.",
        "submission_generated": do_submission,
        "smoke_test": args.max_train is not None,
        "elapsed_seconds": time.perf_counter()-started,
    }
    reporter.write_json("pruning_metrics.json", report)
    print(f"\nBest CV {args.metric}: {result.search.best_score_:.4f}")
    print(f"Best params: {result.search.best_params_}")
    print(f"Development fit accuracy: {result.train_accuracy:.4f}")
    print(f"Holdout accuracy: {holdout_metrics['accuracy']:.4f}  ROC-AUC: {holdout_metrics['roc_auc']:.4f}")
    if result.baseline_cv:
        print(f"Previous fixed configuration CV {args.metric}: {result.baseline_cv['mean']:.4f}")
    print(f"Results saved to: {reporter.output_dir.resolve()}")


if __name__ == "__main__":
    main()
