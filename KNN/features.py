"""Feature engineering from the supplied Pokémon battle JSONL records.

The `full` profile follows the uploaded pokemon_knn_cuda_cv.ipynb, except
for two small correctness fixes: missing numeric indicators are consistently
emitted and missing feature values are handled explicitly.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
import json
import math
from pathlib import Path

import numpy as np

STAT_FIELDS = ("base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe")
BOOST_FIELDS = ("atk", "def", "spa", "spd", "spe")
FEATURE_SETS = ("static", "summary", "full")


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict]:
    """Read records in original file order; limit is for quick smoke tests only."""
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}, line {line_no}") from exc
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    return records


def numeric(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def add_category(out: dict[str, object], name: str, value: object) -> None:
    out[name] = str(value if value is not None else "unknown")


class BattleFeatureExtractor:
    """Turn one nested battle into a numeric/categorical DictVectorizer dictionary.

    static:   static team/lead details only
    summary:  static + order-independent temporal counts and HP aggregates
    full:     summary + each turn's detailed snapshots (notebook-equivalent)
    """

    def __init__(self, feature_set: str = "summary") -> None:
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {FEATURE_SETS}")
        self.feature_set = feature_set

    def transform_one(self, battle: dict) -> dict[str, object]:
        f: dict[str, object] = {}
        team = battle.get("p1_team_details") or []
        for i, mon in enumerate(team, start=1):
            prefix = f"p1_team{i}"
            f[f"{prefix}_level"] = numeric(mon.get("level"))
            for stat in STAT_FIELDS:
                f[f"{prefix}_{stat}"] = numeric(mon.get(stat))
            add_category(f, f"{prefix}_name", mon.get("name"))
            add_category(f, f"{prefix}_item", mon.get("item"))
            for j, typ in enumerate(mon.get("types") or [], start=1):
                add_category(f, f"{prefix}_type{j}", typ)
            species = mon.get("name") or "unknown"
            key = f"team_species_count::{species}"
            f[key] = int(f.get(key, 0)) + 1
            for typ in mon.get("types") or []:
                if typ and typ != "notype":
                    key = f"team_type_count::{typ}"
                    f[key] = int(f.get(key, 0)) + 1
        for stat in STAT_FIELDS:
            f[f"p1_team_mean_{stat}"] = (
                float(np.mean([numeric(mon.get(stat)) for mon in team])) if team else 0.0
            )
        lead = battle.get("p2_lead_details") or {}
        f["p2_lead_level"] = numeric(lead.get("level"))
        for stat in STAT_FIELDS:
            f[f"p2_lead_{stat}"] = numeric(lead.get(stat))
        add_category(f, "p2_lead_name", lead.get("name"))
        for j, typ in enumerate(lead.get("types") or [], start=1):
            add_category(f, f"p2_lead_type{j}", typ)

        if self.feature_set == "static":
            return f

        timeline = battle.get("battle_timeline") or []
        f["observed_turns"] = len(timeline)
        for side in ("p1", "p2"):
            hp_history: list[float] = []
            move_powers: list[float] = []
            observed_names: set[str] = set()
            n_missing_moves = 0
            for t_idx, turn in enumerate(timeline, start=1):
                prefix = f"{side}_t{t_idx:02d}"
                state = turn.get(f"{side}_pokemon_state") or {}
                move = turn.get(f"{side}_move_details")
                hp = numeric(state.get("hp_pct"), default=float("nan"))
                if self.feature_set == "full":
                    f[f"{prefix}_hp_missing"] = int(np.isnan(hp))
                    f[f"{prefix}_hp_pct"] = 0.0 if np.isnan(hp) else hp
                if np.isfinite(hp):
                    hp_history.append(hp)
                name = state.get("name") or "unknown"
                observed_names.add(str(name))
                status = state.get("status") or "unknown"
                status_key = f"{side}_status_count::{status}"
                f[status_key] = int(f.get(status_key, 0)) + 1
                if self.feature_set == "full":
                    add_category(f, f"{prefix}_name", name)
                    add_category(f, f"{prefix}_status", status)
                effects = state.get("effects") or []
                for effect in effects:
                    effect_key = f"{side}_effect_count::{effect}"
                    f[effect_key] = int(f.get(effect_key, 0)) + 1
                    if self.feature_set == "full":
                        f[f"{prefix}_effect::{effect}"] = 1
                boosts = state.get("boosts") or {}
                for bst in BOOST_FIELDS:
                    value = numeric(boosts.get(bst))
                    if self.feature_set == "full":
                        f[f"{prefix}_boost_{bst}"] = value
                    # A compact statistic useful in summary mode, too.
                    key = f"{side}_boost_sum_{bst}"
                    f[key] = float(f.get(key, 0.0)) + value
                if self.feature_set == "full":
                    f[f"{prefix}_move_missing"] = int(move is None)
                if move is None:
                    n_missing_moves += 1
                    if self.feature_set == "full":
                        add_category(f, f"{prefix}_move_name", "missing")
                else:
                    for cat_field in ("name", "type", "category"):
                        val = move.get(cat_field)
                        if self.feature_set == "full":
                            add_category(f, f"{prefix}_move_{cat_field}", val)
                        key = f"{side}_move_{cat_field}_count::{val}"
                        f[key] = int(f.get(key, 0)) + 1
                    for num_field in ("base_power", "accuracy", "priority"):
                        val = numeric(move.get(num_field), default=float("nan"))
                        if self.feature_set == "full":
                            f[f"{prefix}_{num_field}_missing"] = int(np.isnan(val))
                            f[f"{prefix}_{num_field}"] = 0.0 if np.isnan(val) else val
                    power = numeric(move.get("base_power"), default=float("nan"))
                    if np.isfinite(power):
                        move_powers.append(power)
            f[f"{side}_unique_active_pokemon"] = len(observed_names)
            f[f"{side}_missing_moves"] = n_missing_moves
            for label, seq in (("hp", hp_history), ("move_power", move_powers)):
                f[f"{side}_{label}_mean"] = float(np.mean(seq)) if seq else 0.0
                f[f"{side}_{label}_min"] = float(np.min(seq)) if seq else 0.0
                f[f"{side}_{label}_max"] = float(np.max(seq)) if seq else 0.0
            f[f"{side}_hp_first"] = hp_history[0] if hp_history else 0.0
            f[f"{side}_hp_last"] = hp_history[-1] if hp_history else 0.0
            f[f"{side}_hp_delta"] = f[f"{side}_hp_last"] - f[f"{side}_hp_first"]
            for bst in BOOST_FIELDS:
                key = f"{side}_boost_sum_{bst}"
                f[f"{side}_boost_mean_{bst}"] = float(f.pop(key, 0.0)) / max(1, len(timeline))
        f["last_active_hp_difference"] = f["p1_hp_last"] - f["p2_hp_last"]
        return f

    def transform(self, records: Iterable[dict]) -> Iterator[dict[str, object]]:
        for battle in records:
            yield self.transform_one(battle)
