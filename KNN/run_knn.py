r"""Terminal entrypoint: kNN tuning, full refit, model export and test submission.

Example (PowerShell):
    .\.venv\Scripts\python.exe run_knn.py --data-dir .\data --feature-set summary
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # save plots without requiring a desktop/interactive notebook
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from KNN.features import FEATURE_SETS, read_jsonl
from KNN.model import BattleKNNClassifier, NeighborEngine
from KNN.training import KNNCrossValidator

LOG = logging.getLogger("KNN")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pokémon battle prediction: classical kNN + cross-validation")
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="Folder with train.jsonl and test.jsonl")
    p.add_argument("--output-dir", type=Path, default=Path("results"), help="Folder for plots, CV scores, model and CSV")
    p.add_argument("--feature-set", choices=FEATURE_SETS, default="summary",
                   help="static / summary (recommended for first run) / full (notebook per-turn)")
    p.add_argument("--scaling", choices=["numeric", "all", "none"], default="numeric",
                   help="numeric leaves one-hot categories unchanged; all reproduces original notebook")
    p.add_argument("--category-weight", type=float, default=1.0, help="Multiplier for categorical one-hot features")
    p.add_argument("--backend", choices=["auto", "cpu", "cuda"], default="auto",
                   help="auto tries CuPy CUDA, and falls back to CPU")
    p.add_argument("--gpu-batch-size", type=int, default=64)
    p.add_argument("--n-jobs", type=int, default=-1, help="scikit-learn CPU threads; -1 means all available")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k-max", type=int, default=51, help="Maximum odd k: 1, 3, ..., k_max")
    p.add_argument("--group-cv", action="store_true", help="Keep same (unordered) P1 team composition in one fold")
    p.add_argument("--limit-train", type=int, default=None, help="Smoke test on first N train rows (NOT for final model)")
    p.add_argument("--limit-test", type=int, default=None, help="Smoke test on first N test rows")
    p.add_argument("--skip-test", action="store_true", help="Cross-validation and model export only")
    p.add_argument("--no-save-model", action="store_true", help="Avoid saving the fitted model")
    return p.parse_args()


def save_plot(df: pd.DataFrame, best_k: int, target: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(df["k"], df["mean_accuracy"], yerr=df["std_accuracy"],
                fmt="o-", capsize=2, label="Mean CV accuracy ± fold SD")
    ax.axvline(best_k, linestyle="--", linewidth=1.0, label=f"Best k = {best_k}")
    ax.set(xlabel="Number of neighbors (k)", ylabel="Accuracy",
           title="Pokémon kNN — cross-validation")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    if args.k_max < 1 or args.k_max % 2 == 0:
        raise SystemExit("--k-max must be a positive odd number (e.g. 51)")
    if args.gpu_batch_size < 1:
        raise SystemExit("--gpu-batch-size must be positive")
    if args.category_weight <= 0:
        raise SystemExit("--category-weight must be > 0")
    if args.limit_train is not None and args.limit_train < 1:
        raise SystemExit("--limit-train must be >= 1")
    if args.limit_test is not None and args.limit_test < 1:
        raise SystemExit("--limit-test must be >= 1")

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    train_path = data_dir / "train.jsonl"
    test_path = data_dir / "test.jsonl"
    submission_path = data_dir / "sample_submission.csv"
    if not train_path.is_file():
        raise SystemExit(f"Missing {train_path}; copy the competition files into the data folder")
    output_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("Reading train: %s", train_path)
    records = read_jsonl(train_path, args.limit_train)
    y = np.asarray([int(r["player_won"]) for r in records], dtype=np.int8)
    if len(np.unique(y)) != 2:
        raise SystemExit("Training sample needs both win and loss classes")
    LOG.info("Battles: %d, wins: %d, losses: %d", len(records), int(y.sum()), int(len(y)-y.sum()))

    engine = NeighborEngine(args.backend, args.gpu_batch_size, args.n_jobs)
    validator = KNNCrossValidator(
        feature_set=args.feature_set,
        scaling=args.scaling,
        category_weight=args.category_weight,
        k_values=list(range(1, args.k_max + 1, 2)),
        folds=args.folds,
        seed=args.seed,
        group_cv=args.group_cv,
        engine=engine,
    )
    cv_result = validator.run(records, y)
    cv_result.folds.to_csv(output_dir / "cv_fold_scores.csv", index=False)
    cv_result.summary.to_csv(output_dir / "cv_summary.csv", index=False)
    save_plot(cv_result.summary, cv_result.best_k, output_dir / "cv_accuracy.png")

    best_row = cv_result.summary.loc[cv_result.summary.k == cv_result.best_k].iloc[0]
    LOG.info("BEST k=%d, accuracy=%.4f ± %.4f, ROC-AUC=%.4f, CV seconds=%.1f",
             cv_result.best_k, best_row.mean_accuracy, best_row.std_accuracy,
             best_row.mean_roc_auc, cv_result.elapsed_seconds)
    if cv_result.best_k == args.k_max:
        LOG.warning("Best k is at the search boundary; consider increasing --k-max")

    # Refit from scratch on ALL labeled examples, never on the competition test labels.
    LOG.info("Refitting final kNN with full labeled training set...")
    model = BattleKNNClassifier(cv_result.best_k, args.feature_set, args.scaling, args.category_weight)
    model.fit(records, y)
    if not args.no_save_model:
        joblib.dump(model, output_dir / "knn_model.joblib", compress=3)
        LOG.info("Saved trained model to %s", output_dir / "knn_model.joblib")
    info = {
        "feature_set": args.feature_set,
        "scaling": args.scaling,
        "category_weight": args.category_weight,
        "backend_requested": args.backend,
        "folds": args.folds,
        "group_cv": args.group_cv,
        "seed": args.seed,
        "k_values": validator.k_values,
        "best_k": cv_result.best_k,
        "cv_mean_accuracy": float(best_row.mean_accuracy),
        "cv_std_accuracy": float(best_row.std_accuracy),
        "cv_mean_roc_auc": float(best_row.mean_roc_auc),
        "cv_mean_f1": float(best_row.mean_f1),
        "train_samples": len(records),
        "encoded_features": int(model.X_train.shape[1]),
        "cv_elapsed_seconds": cv_result.elapsed_seconds,
        "is_smoke_test": args.limit_train is not None or args.limit_test is not None,
    }
    if args.limit_train is not None:
        LOG.warning("SMOKE TEST MODE: model trained only on %d battles, do not submit.", len(records))
    (output_dir / "metrics.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    if args.skip_test:
        LOG.info("--skip-test: not generating a submission")
        return
    if not test_path.is_file():
        LOG.warning("No test.jsonl under %s; CV complete, no submission created.", data_dir)
        return
    LOG.info("Reading test: %s", test_path)
    tests = read_jsonl(test_path, args.limit_test)
    probs = model.predict_proba(tests, engine)
    predictions = (probs > 0.5).astype(int)
    submission = pd.DataFrame({
        "battle_id": [r["battle_id"] for r in tests],
        "player_won": predictions,
    })
    if submission["battle_id"].duplicated().any():
        raise ValueError("Duplicate battle_id in test.jsonl")
    if submission_path.is_file() and args.limit_test is None:
        template = pd.read_csv(submission_path, usecols=["battle_id"])
        if set(template.battle_id) != set(submission.battle_id):
            raise ValueError("The sample submission IDs differ from test.jsonl")
        submission = template.merge(submission, on="battle_id", how="left", validate="one_to_one")
        if submission.player_won.isna().any():
            raise ValueError("Submission has missing predictions")
        submission["player_won"] = submission["player_won"].astype(int)
    outfile = output_dir / "knn_submission.csv"
    submission.to_csv(outfile, index=False)
    LOG.info("Submission saved: %s (rows=%d; backend=%s)", outfile, len(submission), engine.last_backend)
    if info["is_smoke_test"]:
        LOG.warning("DO NOT SUBMIT: limit-train / limit-test was provided")


if __name__ == "__main__":
    main()
