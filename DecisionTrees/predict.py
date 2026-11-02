"""Regenerate a submission with a saved, already-fitted pipeline."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import joblib
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from DecisionTree.features import BattleFeatureExtractor
from DecisionTree.dataset import BattleDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    p.add_argument("--model", type=Path, default=Path(__file__).resolve().parent / "results" / "decision_tree_model.joblib")
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results" / "decision_tree_submission.csv")
    args = p.parse_args()
    artifact = joblib.load(args.model)  # Only load your own trusted joblib files.
    extractor = BattleFeatureExtractor(artifact["feature_set"], artifact["turns"])
    data = BattleDataset(args.data_dir, extractor)
    X, ids = data.unlabeled()
    preds = artifact["model"].predict(X)
    path, count = data.submission(ids, preds, args.output)
    print(f"Saved {count} predictions to {path}")


if __name__ == "__main__":
    main()
