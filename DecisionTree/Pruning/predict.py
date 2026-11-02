"""Use a previously saved pruned XGBoost single-tree model on the test JSONL."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import joblib
import numpy as np
from DecisionTree.dataset import BattleDataset
from DecisionTree.features import BattleFeatureExtractor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description="Predict with the saved pruning model")
    p.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--model", type=Path, default=HERE / "results" / "pruned_single_tree_model.joblib")
    p.add_argument("--output", type=Path, default=HERE / "results" / "pruning_submission_again.csv")
    args = p.parse_args()
    payload = joblib.load(args.model)  # Only load files you trust.
    extractor = BattleFeatureExtractor(payload["feature_set"], payload["turns"])
    dataset = BattleDataset(args.data_dir, extractor)
    X, ids = dataset.unlabeled()
    pred = np.asarray(payload["model"].predict(X), dtype=int)
    path, n = dataset.submission(ids, pred, args.output)
    print(f"Saved {n} predictions to {path.resolve()}")

if __name__ == "__main__":
    main()
