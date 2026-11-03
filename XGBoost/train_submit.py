"""Reproduce the fixed XGBoost v2 configuration and generate Kaggle CSV.

Run from a repository root with:
python -m XGBoostBaseline.train_submit --data-dir data --output submission.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .features import RichBattleFeatureExtractor
from .dataset import BattleDataset
from .model import BEST_XGB_PARAMS, build_model


def run(data_dir: Path, output: Path, jobs: int = 4, device: str = "cpu", smoke_test: bool = False) -> Path:
    dataset = BattleDataset(Path(data_dir), RichBattleFeatureExtractor(max_turns=30))
    X_train, y_train, _, train_ids = dataset.training()
    if not smoke_test and len(train_ids) != 10_000:
        raise ValueError(f"Expected 10,000 labeled battles, found {len(train_ids)}")
    print("Train:", len(train_ids), "positive fraction:", round(float(np.mean(y_train)), 4))
    model = build_model(seed=42, jobs=jobs, device=device)
    print("Fixed XGBoost v2 parameters:", BEST_XGB_PARAMS)
    model.fit(X_train, y_train)
    print("Training complete. Generating test predictions...")
    X_test, test_ids = dataset.unlabeled()
    if not smoke_test and len(test_ids) != 5_000:
        raise ValueError(f"Expected 5,000 test battles, found {len(test_ids)}")
    pred = model.predict(X_test)
    path, n = dataset.submission(test_ids, pred, output)
    submission = pd.read_csv(path)
    template = pd.read_csv(Path(data_dir) / 'sample_submission.csv')
    assert submission.columns.tolist() == ["battle_id", "player_won"]
    assert submission["battle_id"].equals(template["battle_id"])
    assert set(submission["player_won"].unique()).issubset({0, 1})
    assert len(submission) == n == len(test_ids)
    print("Valid Kaggle CSV:", path.resolve())
    print(submission.head().to_string(index=False))
    return path


def main():
    p = argparse.ArgumentParser(description="XGBoost v2 fixed baseline → Kaggle submission CSV")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output", type=Path, default=Path("submission.csv"))
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--smoke-test", action="store_true", help="Allow reduced local fixture; do not use for Kaggle")
    a = p.parse_args()
    run(a.data_dir, a.output, a.jobs, a.device, a.smoke_test)


if __name__ == "__main__":
    main()
