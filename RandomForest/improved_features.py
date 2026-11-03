"""Battle-state and momentum features using only visible turn information."""
from __future__ import annotations

from collections import Counter
import copy
import re
import numpy as np

from .features import BattleFeatureExtractor, RichBattleFeatureExtractor, number, summarize

SIDES = ("p1", "p2")
PHASES = {"early": (0, 10), "middle": (10, 20), "late": (20, 30)}


def hp_fraction(value):
    """HP is normalized by the battle-wide scale before feature extraction."""
    x = number(value)
    return float(np.clip(x, 0, 1)) if np.isfinite(x) else np.nan


def normalized_battle(battle, max_turns):
    """Convert 0–100 HP readings to fractions without using labels or future turns."""
    item = dict(battle)
    turns = copy.deepcopy((battle.get("battle_timeline") or [])[:max_turns])
    readings = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        for side in SIDES:
            state = turn.get(f"{side}_pokemon_state") or {}
            h = number(state.get("hp_pct"))
            if np.isfinite(h):
                readings.append(h)
    # A percentage reading >1 establishes a 0–100 scale for this battle.
    scale = 100.0 if any(h > 1 for h in readings) else 1.0
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        for side in SIDES:
            state = turn.get(f"{side}_pokemon_state")
            if isinstance(state, dict) and "hp_pct" in state:
                h = number(state["hp_pct"])
                state["hp_pct"] = float(np.clip(h / scale, 0, 1)) if np.isfinite(h) else None
    item["battle_timeline"] = turns
    return item


def states_through(turns, side):
    """Last known state per species; repeated species cannot be distinguished."""
    latest = {}
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        state = turn.get(f"{side}_pokemon_state") or {}
        name = str(state.get("name") or "").strip().lower()
        if not name:
            continue
        hp = hp_fraction(state.get("hp_pct"))
        if name not in latest:
            latest[name] = {"hp": np.nan, "status": "nostatus"}
        if np.isfinite(hp):
            latest[name]["hp"] = hp
        if state.get("status") is not None:
            latest[name]["status"] = str(state.get("status") or "nostatus").lower()
    return latest


def observed_switches(turns, side):
    names = [str((t.get(f"{side}_pokemon_state") or {}).get("name") or "").lower()
             for t in turns if isinstance(t, dict)]
    names = [n for n in names if n]
    return sum(a != b for a, b in zip(names, names[1:]))


def phase_stats(turns, side, prefix, out):
    """Aggregate each phase without assuming that the active Pokemon is unchanged."""
    hps = []
    boost_values = []
    statuses = 0
    for t in turns:
        if not isinstance(t, dict):
            continue
        state = t.get(f"{side}_pokemon_state") or {}
        hps.append(hp_fraction(state.get("hp_pct")))
        if str(state.get("status") or "nostatus").lower() not in ("nostatus", "none", ""):
            statuses += 1
        boosts = state.get("boosts") or {}
        boost_values.append(sum(number(boosts.get(k), 0) for k in ("atk", "def", "spa", "spd", "spe")))
    summarize(hps, f"{prefix}_active_hp", out)
    summarize(boost_values, f"{prefix}_boost", out)
    out[f"{prefix}_status_turns"] = statuses
    out[f"{prefix}_switches"] = observed_switches(turns, side)
    last = states_through(turns, side)
    known = [entry["hp"] for entry in last.values() if np.isfinite(entry["hp"])]
    out[f"{prefix}_known_hp_sum"] = float(sum(known))
    out[f"{prefix}_known_fainted"] = sum(h <= 0 for h in known)
    out[f"{prefix}_observed_count"] = len(last)


def add_state_features(battle, out, max_turns):
    turns = (battle.get("battle_timeline") or [])[:max_turns]
    team = battle.get("p1_team_details") or []
    states = {side: states_through(turns, side) for side in SIDES}
    for side in SIDES:
        s = states[side]
        known = [v["hp"] for v in s.values() if np.isfinite(v["hp"])]
        out[f"{side}_known_hp_sum"] = float(sum(known))
        out[f"{side}_known_hp_count"] = len(known)
        out[f"{side}_observed_count"] = len(s)
        out[f"{side}_unknown_hp_count"] = len(s) - len(known)
        out[f"{side}_known_fainted"] = sum(h <= 0 for h in known)
        out[f"{side}_known_below_25"] = sum(0 < h < .25 for h in known)
        out[f"{side}_known_below_50"] = sum(0 < h < .5 for h in known)
        out[f"{side}_observed_hp_mean"] = float(np.mean(known)) if known else np.nan
    counts = Counter(str(p.get("name") or "").strip().lower() for p in team)
    counts.pop("", None)
    unknown_p1 = sum(max(n - 1, 0) for name, n in counts.items() if name in states["p1"])
    out["p1_duplicate_species_ambiguity"] = unknown_p1
    out["p1_unobserved_team_slots_lower_bound"] = sum(n for name, n in counts.items() if name not in states["p1"])
    out["p2_unobserved_team_slots_unknown"] = 1
    out["known_hp_sum_difference"] = out["p1_known_hp_sum"] - out["p2_known_hp_sum"]
    out["known_fainted_advantage"] = out["p2_known_fainted"] - out["p1_known_fainted"]
    out["known_below_25_advantage"] = out["p2_known_below_25"] - out["p1_known_below_25"]
    out["observed_count_difference"] = out["p1_observed_count"] - out["p2_observed_count"]
    out["known_hp_count_difference"] = out["p1_known_hp_count"] - out["p2_known_hp_count"]

    # Do not infer the unseen P2 team HP or assign it an artificial zero.
    out.pop("p1_full_team_hp_proxy", None)
    out.pop("p1_full_team_hp_proxy_sum", None)

    for phase, (start, end) in PHASES.items():
        subset = turns[start:min(end, max_turns)]
        for side in SIDES:
            phase_stats(subset, side, f"{side}_{phase}", out)
        for metric in ("active_hp_mean", "known_hp_sum", "known_fainted", "switches", "status_turns", "boost_mean"):
            p1 = out.get(f"p1_{phase}_{metric}", np.nan)
            p2 = out.get(f"p2_{phase}_{metric}", np.nan)
            out[f"{phase}_{metric}_difference"] = p1 - p2

    for earlier, later in (("early", "middle"), ("middle", "late"), ("early", "late")):
        for metric in ("active_hp_mean", "known_hp_sum", "known_fainted", "switches", "status_turns", "boost_mean"):
            for side in SIDES:
                a = out.get(f"{side}_{earlier}_{metric}", np.nan)
                b = out.get(f"{side}_{later}_{metric}", np.nan)
                out[f"{side}_{earlier}_to_{later}_{metric}_change"] = b - a
            a = out.get(f"{earlier}_{metric}_difference", np.nan)
            b = out.get(f"{later}_{metric}_difference", np.nan)
            out[f"{earlier}_to_{later}_{metric}_gap_change"] = b - a

    # More recent observations can be more relevant than the first turns.
    for side in SIDES:
        window = turns[-min(10, len(turns)):]
        out[f"{side}_recent_switches"] = observed_switches(window, side)
        recent_hps = [hp_fraction((t.get(f"{side}_pokemon_state") or {}).get("hp_pct"))
                      for t in window if isinstance(t, dict)]
        summarize(recent_hps, f"{side}_recent_hp", out)
    out["recent_switches_difference"] = out["p1_recent_switches"] - out["p2_recent_switches"]
    out["recent_active_hp_gap"] = out["p1_recent_hp_mean"] - out["p2_recent_hp_mean"]


class ImprovedBattleFeatureExtractor:
    """Profiles share improved state features; compact omits raw per-turn columns."""
    PROFILES = ("engineered", "full", "rich_full", "rich_compact", "rich_no_identity")

    def __init__(self, feature_set="rich_compact", max_turns=30):
        if feature_set not in self.PROFILES:
            raise ValueError(f"Choose one of {self.PROFILES}")
        if not 1 <= max_turns <= 30:
            raise ValueError("max_turns must be within 1..30")
        self.feature_set = feature_set
        self.max_turns = int(max_turns)

    def extract(self, battle):
        normalized = normalized_battle(battle, self.max_turns)
        if self.feature_set == "engineered":
            out = BattleFeatureExtractor("engineered", self.max_turns).extract(normalized)
        elif self.feature_set == "full":
            out = BattleFeatureExtractor("full", self.max_turns).extract(normalized)
        else:
            out = RichBattleFeatureExtractor(self.max_turns).extract(normalized)
        add_state_features(normalized, out, self.max_turns)
        if self.feature_set == "rich_compact":
            out = {k: v for k, v in out.items() if not re.match(r"^t\d{2}_", k)}
        elif self.feature_set == "rich_no_identity":
            out = {
                k: v for k, v in out.items()
                if not (
                    re.match(r"^t\d{2}_p[12]_pokemon$", k)
                    or re.match(r"^team_slot_\d{2}_name$", k)
                    or k == "p2_lead_name"
                )
            }
        return out
