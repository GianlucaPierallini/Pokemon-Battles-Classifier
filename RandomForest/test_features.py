"""Unit checks for cutoff, HP scale, and missing-opponent information."""
from __future__ import annotations
import copy
import unittest
import math
from .improved_features import ImprovedBattleFeatureExtractor


def sample(scale=1):
    team = [{"name": f"Mon{i}", "base_hp": 90, "base_atk": 80, "base_def": 70,
             "base_spa": 60, "base_spd": 80, "base_spe": 60, "types": ["water"], "item": "leftovers"}
            for i in range(6)]
    timeline = []
    for i in range(30):
        timeline.append({
            "p1_pokemon_state": {"name": "Mon0" if i < 15 else "Mon1", "hp_pct": (.95 - .02*i)*scale, "status": "nostatus", "boosts": {}},
            "p2_pokemon_state": {"name": "Rival0" if i < 12 else "Rival1", "hp_pct": (.9 - .025*i)*scale, "status": "par" if i > 20 else "nostatus", "boosts": {}},
            "p1_move_details": {"name": "tackle", "type": "normal", "category": "physical", "base_power": 40},
            "p2_move_details": {"name": "rest", "type": "psychic", "category": "status", "base_power": 0}
        })
    return {"battle_id": 1, "player_won": True, "p1_team_details": team,
            "p2_lead_details": {"name": "Rival0", "base_hp": 100, "types": ["grass"]},
            "battle_timeline": timeline}


class FeatureTests(unittest.TestCase):
    def test_hp_scaling(self):
        a = ImprovedBattleFeatureExtractor("rich_compact").extract(sample(1))
        b = ImprovedBattleFeatureExtractor("rich_compact").extract(sample(100))
        self.assertAlmostEqual(a["p1_known_hp_sum"], b["p1_known_hp_sum"])
        self.assertAlmostEqual(a["recent_active_hp_gap"], b["recent_active_hp_gap"])

    def test_cutoff_and_label(self):
        a = sample()
        extractor = ImprovedBattleFeatureExtractor("rich_full", 30)
        before = extractor.extract(a)
        changed = copy.deepcopy(a)
        changed["player_won"] = False
        changed["battle_timeline"].append({"p1_pokemon_state": {"name": "Late", "hp_pct": 0}})
        after = extractor.extract(changed)
        self.assertEqual(before.keys(), after.keys())
        for key in before:
            x, y = before[key], after[key]
            if isinstance(x, (float, int)) and isinstance(y, (float, int)) and math.isnan(float(x)) and math.isnan(float(y)):
                continue
            self.assertEqual(x, y, key)

    def test_compact_has_no_raw_turn_columns(self):
        compact = ImprovedBattleFeatureExtractor("rich_compact").extract(sample())
        full = ImprovedBattleFeatureExtractor("rich_full").extract(sample())
        self.assertNotIn("t01_p1_pokemon", compact)
        self.assertIn("t01_p1_pokemon", full)
        self.assertIn("early_to_late_active_hp_mean_gap_change", compact)
        self.assertIn("known_fainted_advantage", compact)

    def test_no_identity_removes_only_requested_columns(self):
        full = ImprovedBattleFeatureExtractor("rich_full").extract(sample())
        reduced = ImprovedBattleFeatureExtractor("rich_no_identity").extract(sample())
        excluded = {
            k for k in full if (
                k.startswith("team_slot_") and k.endswith("_name")
            ) or k == "p2_lead_name" or (
                len(k) >= 14 and k.startswith("t") and k[1:3].isdigit()
                and k.endswith("_pokemon")
            )}
        self.assertTrue(excluded)
        self.assertEqual(set(reduced), set(full) - excluded)
        self.assertIn("t01_p1_move", reduced)
        self.assertIn("t01_p1_hp_pct", reduced)
        self.assertIn("p1_pokemon_presence__mon0", reduced)
        self.assertIn("known_fainted_advantage", reduced)

    def test_unseen_opponent_is_not_assumed_fainted(self):
        features = ImprovedBattleFeatureExtractor().extract(sample())
        self.assertEqual(features["p2_known_fainted"], 0)
        self.assertNotIn("p2_full_team_hp_proxy", features)
        self.assertEqual(features["p1_unobserved_team_slots_lower_bound"], 4)


if __name__ == "__main__":
    unittest.main()
