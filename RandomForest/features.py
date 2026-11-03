
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

RANDOM_STATE = 42
STATS = ("hp", "atk", "def", "spa", "spd", "spe")
BOOSTS = ("atk", "def", "spa", "spd", "spe")
STATUS = ("brn", "par", "slp", "frz", "psn", "tox")


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def number(value, default=np.nan):
    """Convert JSON numbers safely, including None and numeric strings."""
    try:
        val = float(value)
        return val if np.isfinite(val) else default
    except (TypeError, ValueError):
        return default


def summarize(values, prefix, out):
    values = np.asarray([number(v) for v in values], dtype=float)
    values = values[np.isfinite(values)]
    out[f"{prefix}_count"] = len(values)
    for stat, func in (("mean", np.mean), ("min", np.min),
                       ("max", np.max), ("std", np.std), ("sum", np.sum)):
        out[f"{prefix}_{stat}"] = float(func(values)) if len(values) else np.nan


def build_features(battle):
    """Use only pre-battle information and the observed 30-turn timeline."""
    out = {}
    team = battle.get("p1_team_details") or []
    lead = battle.get("p2_lead_details") or {}
    timeline = (battle.get("battle_timeline") or [])[:30]

    # Starting strength: all six known p1 Pokemon, but only p2's known lead.
    out["p1_team_size"] = len(team)
    out["p1_team_unique_species"] = len({p.get("name") for p in team if p.get("name")})
    for stat in STATS:
        summarize([p.get("base_" + stat) for p in team], "team_" + stat, out)
        out["opponent_lead_" + stat] = number(lead.get("base_" + stat))
        out["team_vs_lead_" + stat] = out[f"team_{stat}_mean"] - out[f"opponent_lead_{stat}"]
    out["p1_total_base_stats_mean"] = np.nanmean(
        [sum(number(p.get("base_" + s), 0) for s in STATS) for p in team]
    ) if team else np.nan
    out["opponent_lead_total_base_stats"] = sum(number(lead.get("base_" + s), 0) for s in STATS)
    out["p1_leftovers_count"] = sum(str(p.get("item", "")).lower() == "leftovers" for p in team)
    out["p1_mintberry_count"] = sum(str(p.get("item", "")).lower() == "mintberry" for p in team)
    team_types = Counter(t for p in team for t in (p.get("types") or []) if t != "notype")
    out["p1_distinct_types"] = len(team_types)
    out["p1_duplicate_type_slots"] = sum(max(v - 1, 0) for v in team_types.values())
    lead_types = set(lead.get("types") or []) - {"notype"}
    out["p1_count_lead_shared_type"] = sum(
        bool(set(p.get("types") or []) & lead_types) for p in team
    )

    # Keep latest observed state for EACH observed Pokemon. This is important:
    # only tracking the active Pokemon on turn 30 loses bench information.
    sides = {}
    for side in ("p1", "p2"):
        states = {}
        active_hp, active_status, damage, healing = [], [], [], []
        move_types, move_names = Counter(), Counter()
        categories = Counter()
        power, accuracy, boosts = [], [], []
        switches = 0
        prev_name = None
        for turn in timeline:
            if not isinstance(turn, dict):
                continue
            state = turn.get(f"{side}_pokemon_state") or {}
            name = state.get("name")
            if name:
                if prev_name and prev_name != name:
                    switches += 1
                prev_name = name
                hp = number(state.get("hp_pct"))
                if np.isfinite(hp):
                    active_hp.append(hp)
                    if name in states and np.isfinite(states[name]["hp"]):
                        diff = hp - states[name]["hp"]
                        if diff < 0:
                            damage.append(-diff)
                        elif diff > 0:
                            healing.append(diff)
                statuses = str(state.get("status") or "nostatus").lower()
                if statuses != "nostatus":
                    active_status.append(statuses)
                boosts_dict = state.get("boosts") or {}
                boosts.append(sum(number(boosts_dict.get(key), 0) for key in BOOSTS))
                states[name] = {"hp": hp, "status": statuses,
                                "boost": sum(number(boosts_dict.get(key), 0) for key in BOOSTS)}

            move = turn.get(f"{side}_move_details") or {}
            if move:
                if move.get("name"):
                    move_names[str(move["name"]).lower()] += 1
                if move.get("type"):
                    move_types[str(move["type"]).lower()] += 1
                if move.get("category"):
                    categories[str(move["category"]).lower()] += 1
                power.append(number(move.get("base_power"), 0))
                accuracy.append(number(move.get("accuracy")))

        hp_values = [v["hp"] for v in states.values() if np.isfinite(v["hp"])]
        prefix = side
        out[prefix + "_observed_species"] = len(states)
        out[prefix + "_switches"] = switches
        out[prefix + "_status_events"] = len(active_status)
        out[prefix + "_distinct_moves"] = len(move_names)
        out[prefix + "_distinct_move_types"] = len(move_types)
        out[prefix + "_damaging_moves"] = categories["physical"] + categories["special"]
        out[prefix + "_status_moves"] = categories["status"]
        for status in STATUS:
            out[prefix + "_latest_" + status] = sum(v["status"] == status for v in states.values())
        for key, vals in (("latest_hp", hp_values), ("active_hp", active_hp),
                          ("damage_events", damage), ("healing_events", healing),
                          ("move_power", power), ("move_accuracy", accuracy),
                          ("active_boost", boosts)):
            summarize(vals, prefix + "_" + key, out)
        out[prefix + "_latest_hp_below_50pct"] = sum(h < .5 for h in hp_values)
        out[prefix + "_latest_hp_below_25pct"] = sum(h < .25 for h in hp_values)
        out[prefix + "_latest_fainted"] = sum(h <= 0 for h in hp_values)
        out[prefix + "_last_active_hp"] = active_hp[-1] if active_hp else np.nan
        out[prefix + "_first_active_hp"] = active_hp[0] if active_hp else np.nan
        out[prefix + "_hp_change_active"] = (
            active_hp[-1] - active_hp[0] if len(active_hp) >= 2 else np.nan
        )
        out[prefix + "_latest_boost_sum"] = sum(v["boost"] for v in states.values())
        sides[side] = states

    # Team HP estimate: species never observed are not assumed dead.
    # Only for p1 (whose complete team is known), an unobserved team member
    # may be treated as still at 100% for a *proxy*, not a guaranteed fact.
    p1_states = sides["p1"]
    team_hp_proxy = []
    for p in team:
        pokemon_name = p.get("name")
        hp = p1_states.get(pokemon_name, {}).get("hp", 1.0)
        team_hp_proxy.append(hp)
    out["p1_full_team_hp_proxy"] = float(np.mean(team_hp_proxy)) if team_hp_proxy else np.nan
    out["p1_full_team_hp_proxy_sum"] = float(np.sum(team_hp_proxy)) if team_hp_proxy else np.nan

    # Opponent team members that never appear are UNKNOWN; do not assume
    # they have been knocked out or that there are fewer than six Pokemon.
    for suffix in ("observed_species", "switches", "latest_hp_mean",
                   "latest_hp_min", "latest_fainted", "last_active_hp",
                   "latest_hp_below_50pct", "status_events", "move_power_mean",
                   "damage_events_sum", "healing_events_sum", "latest_boost_sum"):
        a = out.get("p1_" + suffix, np.nan)
        b = out.get("p2_" + suffix, np.nan)
        out["difference_" + suffix] = a - b

    # Trends are calculated over observed states, not future battle turns.
    out["number_observed_turns"] = len(timeline)
    for side in ("p1", "p2"):
        for window in (5, 10):
            window_hps = [number((turn.get(f"{side}_pokemon_state") or {}).get("hp_pct"))
                          for turn in timeline[-window:] if isinstance(turn, dict)]
            summarize(window_hps, f"{side}_last_{window}_turn_hp", out)
    out["recent_hp_difference"] = out.get("p1_last_5_turn_hp_mean", np.nan) - out.get("p2_last_5_turn_hp_mean", np.nan)
    return out



class BattleFeatureExtractor:
    """Extract battle-level feature dictionaries for sklearn DictVectorizer."""

    def __init__(self, feature_set: str = "full", max_turns: int = 30):
        if feature_set not in ("engineered", "full"):
            raise ValueError("feature_set must be 'engineered' or 'full'")
        if max_turns < 1 or max_turns > 30:
            raise ValueError("max_turns must be between 1 and 30")
        self.feature_set = feature_set
        self.max_turns = int(max_turns)

    def extract(self, battle: dict) -> dict:
        # build_features() from the uploaded script is unchanged for all 30 turns.
        # When a prefix is requested, we use only the observations up to that turn.
        truncated = dict(battle)
        truncated["battle_timeline"] = (battle.get("battle_timeline") or [])[:self.max_turns]
        out = build_features(truncated)
        if self.feature_set == "engineered":
            return out
        team = battle.get("p1_team_details") or []
        lead = battle.get("p2_lead_details") or {}
        # Categorical features are strings (not arbitrary ordinal integer codes).
        for j, mon in enumerate(team[:6]):
            pre = f"team_slot_{j + 1:02d}"
            out[f"{pre}_name"] = str(mon.get("name") or "__missing__").lower()
            out[f"{pre}_item"] = str(mon.get("item") or "__missing__").lower()
            out[f"{pre}_level"] = number(mon.get("level"))
            for ti, typ in enumerate((mon.get("types") or [])[:2]):
                out[f"{pre}_type_{ti+1}"] = str(typ or "notype").lower()
        out["p2_lead_name"] = str(lead.get("name") or "__missing__").lower()
        for ti, typ in enumerate((lead.get("types") or [])[:2]):
            out[f"p2_lead_type_{ti+1}"] = str(typ or "notype").lower()
        for i, turn in enumerate(truncated["battle_timeline"], start=1):
            if not isinstance(turn, dict):
                continue
            for side in ("p1", "p2"):
                pre = f"t{i:02d}_{side}"
                state = turn.get(f"{side}_pokemon_state") or {}
                move = turn.get(f"{side}_move_details") or {}
                out[f"{pre}_pokemon"] = str(state.get("name") or "__missing__").lower()
                out[f"{pre}_hp_pct"] = number(state.get("hp_pct"))
                out[f"{pre}_status"] = str(state.get("status") or "nostatus").lower()
                eff = state.get("effects") or []
                out[f"{pre}_effect_count"] = sum(str(e).lower() != "noeffect" for e in eff)
                boosts = state.get("boosts") or {}
                for stat in BOOSTS:
                    out[f"{pre}_boost_{stat}"] = number(boosts.get(stat), 0)
                out[f"{pre}_move"] = str(move.get("name") or "__none__").lower()
                out[f"{pre}_move_type"] = str(move.get("type") or "__none__").lower()
                out[f"{pre}_move_category"] = str(move.get("category") or "__none__").lower()
                out[f"{pre}_move_power"] = number(move.get("base_power"))
                out[f"{pre}_move_accuracy"] = number(move.get("accuracy"))
                out[f"{pre}_move_priority"] = number(move.get("priority"))
                out[f"{pre}_move_missing"] = int(not bool(move))
            # Side-relative feature captures the evolving score.
            p1 = number((turn.get("p1_pokemon_state") or {}).get("hp_pct"))
            p2 = number((turn.get("p2_pokemon_state") or {}).get("hp_pct"))
            out[f"t{i:02d}_hp_gap"] = p1 - p2
        return out


def team_group_key(battle: dict) -> str:
    """Same six species, regardless of lead order -> same group."""
    names = [str(p.get("name") or "__missing__").lower()
             for p in (battle.get("p1_team_details") or [])]
    return "|".join(sorted(names))


class RichBattleFeatureExtractor(BattleFeatureExtractor):
    """A strict superset of `full`, with count/phase representations.

    Raw per-turn data alone creates many rare categorical coordinates. This
    second representation exposes repeated actions and phase trends in compact
    features, using only the battle JSON and no external dataset.
    """
    def __init__(self, max_turns: int = 30):
        super().__init__(feature_set="full", max_turns=max_turns)

    def extract(self, battle: dict) -> dict:
        out = super().extract(battle)
        turns = (battle.get("battle_timeline") or [])[:self.max_turns]
        phases = ("early", "middle", "late")
        for side in ("p1", "p2"):
            for phase in phases:
                lo, hi = {
                    "early": (0, 10), "middle": (10, 20), "late": (20, 30),
                }[phase]
                phase_turns = [t for t in turns[lo:hi] if isinstance(t, dict)]
                hp = [number((t.get(f"{side}_pokemon_state") or {}).get("hp_pct"))
                      for t in phase_turns]
                summarize(hp, f"{side}_{phase}_active_hp", out)
                powers = [number((t.get(f"{side}_move_details") or {}).get("base_power"))
                          for t in phase_turns]
                summarize(powers, f"{side}_{phase}_move_power", out)
                out[f"{side}_{phase}_status_turns"] = sum(
                    (t.get(f"{side}_pokemon_state") or {}).get("status")
                    not in (None, "", "nostatus") for t in phase_turns
                )
            move_names, move_types, move_categories = Counter(), Counter(), Counter()
            pokemon_presence, statuses, effects = Counter(), Counter(), Counter()
            for t in turns:
                if not isinstance(t, dict):
                    continue
                state = t.get(f"{side}_pokemon_state") or {}
                move = t.get(f"{side}_move_details") or {}
                if state.get("name"):
                    pokemon_presence[str(state["name"]).lower()] += 1
                status = str(state.get("status") or "nostatus").lower()
                if status != "nostatus":
                    statuses[status] += 1
                for effect in state.get("effects") or []:
                    if isinstance(effect, str) and effect.lower() != "noeffect":
                        effects[effect.lower()] += 1
                for key, counter in (("name", move_names), ("type", move_types),
                                     ("category", move_categories)):
                    value = move.get(key)
                    if value:
                        counter[str(value).lower()] += 1
            for kind, counter in (("move_name", move_names),
                                  ("move_type", move_types),
                                  ("move_category", move_categories),
                                  ("pokemon_presence", pokemon_presence),
                                  ("status_frequency", statuses),
                                  ("effect_frequency", effects)):
                for value, count in counter.items():
                    out[f"{side}_{kind}__{value}"] = count
        for phase in phases:
            a = out.get(f"p1_{phase}_active_hp_mean", np.nan)
            b = out.get(f"p2_{phase}_active_hp_mean", np.nan)
            out[f"difference_{phase}_hp_mean"] = a - b
        return out
