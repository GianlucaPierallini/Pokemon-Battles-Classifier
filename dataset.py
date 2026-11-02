"""Read only competition files; never use test labels for training/tuning."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .features import BattleFeatureExtractor, team_group_key


class BattleDataset:
    def __init__(self, data_dir: Path, extractor: BattleFeatureExtractor):
        self.data_dir = Path(data_dir)
        self.extractor = extractor

    @staticmethod
    def _stream(path: Path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing dataset file: {path}")
        with path.open("r", encoding="utf-8") as file:
            for line_no, line in enumerate(file, 1):
                if line.strip():
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON in {path} at line {line_no}") from exc
                    if not isinstance(obj, dict):
                        raise ValueError(f"Expected JSON object in {path} line {line_no}")
                    yield obj

    def training(self, max_train: int | None = None):
        features, labels, ids, groups = [], [], [], []
        for b in self._stream(self.data_dir / "train.jsonl"):
            if "player_won" not in b:
                raise ValueError("train.jsonl contains a battle without player_won")
            y = b["player_won"]
            if y not in (True, False, 0, 1):
                raise ValueError(f"Invalid player_won value: {y!r}")
            labels.append(int(y))
            ids.append(b["battle_id"])
            groups.append(team_group_key(b))
            features.append(self.extractor.extract(b))
            if max_train and len(labels) >= max_train:
                break
        if len(features) < 30 or len(set(labels)) < 2:
            raise ValueError("Need at least 30 labeled battles and both classes")
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate train battle_id")
        return features, np.asarray(labels, dtype=np.int32), np.asarray(groups), ids

    def unlabeled(self):
        features, ids = [], []
        for b in self._stream(self.data_dir / "test.jsonl"):
            if "player_won" in b:
                raise ValueError("Test data unexpectedly contains player_won")
            ids.append(b["battle_id"])
            features.append(self.extractor.extract(b))
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Empty test set or duplicate test IDs")
        return features, ids

    def submission(self, test_ids, predictions, output: Path):
        template = pd.read_csv(self.data_dir / "sample_submission.csv")
        if list(template.columns) != ["battle_id", "player_won"]:
            raise ValueError("Unexpected columns in sample_submission.csv")
        if template.battle_id.duplicated().any() or len(template) != len(test_ids):
            raise ValueError("Duplicate IDs or length mismatch in sample submission")
        pred_map = pd.Series(np.asarray(predictions, dtype=int), index=pd.Index(test_ids))
        labels = template.battle_id.map(pred_map)
        if labels.isna().any() or not set(labels.astype(int).unique()).issubset({0, 1}):
            raise ValueError("Missing or invalid predictions for sample submission IDs")
        result = template.copy()
        result["player_won"] = labels.astype(int)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
        return output, len(result)
