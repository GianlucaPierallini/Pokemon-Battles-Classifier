"""Reload a trusted fitted forest and regenerate its submission."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
from RandomForest.dataset import BattleDataset
from RandomForest.improved_features import ImprovedBattleFeatureExtractor

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description="Predict from an already fitted forest")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--model", type=Path, default=HERE / "results" / "random_forest_model.joblib")
    p.add_argument("--output", type=Path, default=HERE / "results" / "random_forest_submission_reloaded.csv")
    a = p.parse_args()
    # joblib uses pickle: load only files you personally trained/trust.
    bundle = joblib.load(a.model)
    extractor = ImprovedBattleFeatureExtractor(bundle["feature_set"], bundle["turns"])
    dataset = BattleDataset(a.data_dir, extractor)
    X_test, ids = dataset.unlabeled()
    predictions = bundle["model"].predict(X_test)
    output, n = dataset.submission(ids, predictions, a.output)
    print(f"Saved {n} predictions to {output.resolve()}")


if __name__ == "__main__":
    main()
