"""Feature engineering for the Sapienza FDS Gen 2 battle challenge.

Inputs: ONLY the competition's JSONL records and, optionally, its supplied
Gen 2 type chart. No external Pokemon databases or replay information.

Caution: use for study/experimentation; adapt, understand, and document AI
assistance in accordance with your course's academic-integrity rules.
"""

from collections import Counter
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

STATS = ('hp', 'atk', 'def', 'spa', 'spd', 'spe')
BOOSTS = ('atk', 'def', 'spa', 'spd', 'spe')
MISSING = {'', 'none', 'null', 'nan', 'unknown', 'notype'}


def norm(v):
    if v is None:
        return 'unknown'
    return str(v).lower().strip().replace(' ', '')


def safe_key(s):
    return re.sub('[^a-z0-9_]+', '_', norm(s))


def fnum(x, default=0.0):
    try:
        x = float(x)
        return x if np.isfinite(x) else default
    except (ValueError, TypeError):
        return default


def dct(x):
    return x if isinstance(x, dict) else {}


def clean_types(x):
    return [norm(t) for t in (x if isinstance(x, list) else [])
            if norm(t) not in MISSING]


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def build_species_lookup(train_records):
    """Species metadata copied from provided records; NEVER includes labels or items."""
    lookup = {}
    for record in train_records:
        candidates = list(record.get('p1_team_details') or [])
        if record.get('p2_lead_details'):
            candidates.append(record['p2_lead_details'])
        for pokemon in candidates:
            p = dct(pokemon)
            name = norm(p.get('name'))
            if name == 'unknown':
                continue
            if name not in lookup:
                lookup[name] = {
                    **{f'base_{s}': fnum(p.get(f'base_{s}')) for s in STATS},
                    'types': clean_types(p.get('types')),
                }
    return lookup


def build_vocab(train_records, max_species=55, max_moves=35):
    """Unsupervised token vocabulary learned from TRAIN only."""
    species = Counter()
    moves = Counter()
    types = Counter()
    items = Counter()
    for b in train_records:
        for p in b.get('p1_team_details') or []:
            q = dct(p)
            species[norm(q.get('name'))] += 1
            items[norm(q.get('item'))] += 1
            types.update(clean_types(q.get('types')))
        lead = dct(b.get('p2_lead_details'))
        species[norm(lead.get('name'))] += 1
        types.update(clean_types(lead.get('types')))
        for row in b.get('battle_timeline') or []:
            for side in ('p1', 'p2'):
                state = dct(row.get(f'{side}_pokemon_state'))
                move = dct(row.get(f'{side}_move_details'))
                if state.get('name'):
                    species[norm(state['name'])] += 1
                if move.get('name'):
                    moves[norm(move['name'])] += 1
                if move.get('type'):
                    types[norm(move['type'])] += 1
    return {
        'species': [s for s, _ in species.most_common(max_species) if s not in MISSING],
        'moves': [m for m, _ in moves.most_common(max_moves) if m not in MISSING],
        'types': [t for t, _ in types.most_common() if t not in MISSING],
        'items': [i for i, _ in items.most_common(12) if i not in MISSING],
    }


def load_type_chart(path):
    """Optional: supplied CSV/JSON type chart; rows are *attacking* types.

    CSV supported: long (attack, defense, multiplier) or wide attack-type rows.
    A Water->Fire sanity check comes from the challenge handout itself.
    """
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    chart = {}
    if path.suffix.lower() == '.json':
        obj = json.loads(path.read_text(encoding='utf-8'))
        for atk, defenses in obj.items():
            if isinstance(defenses, dict):
                for df, value in defenses.items():
                    chart[(norm(atk), norm(df))] = fnum(value, 1.0)
    else:
        raw = pd.read_csv(path)
        raw.columns = [norm(c) for c in raw.columns]
        a = next((c for c in raw if c in ('attack', 'attacking', 'attacker', 'attack_type', 'attacking_type')), None)
        d = next((c for c in raw if c in ('defense', 'defending', 'defender', 'defense_type', 'defending_type')), None)
        v = next((c for c in raw if c in ('multiplier', 'effectiveness', 'damage', 'value')), None)
        if a and d and v:
            for _, r in raw.iterrows():
                chart[(norm(r[a]), norm(r[d]))] = fnum(r[v], 1.0)
        else:
            atk_col = a or raw.columns[0]
            for _, r in raw.iterrows():
                atk = norm(r[atk_col])
                for df in raw.columns:
                    if df != atk_col:
                        chart[(atk, df)] = fnum(r[df], 1.0)
    # Water attacks should do double damage against Fire according to the PDF.
    if chart and chart.get(('water', 'fire')) != 2.0:
        if chart.get(('fire', 'water')) == 2.0:
            chart = {(b, a): x for (a, b), x in chart.items()}
        else:
            print(f'WARNING: Cannot verify type-chart orientation ({path.name}); disabling chart features.')
            return {}
    print(f'Loaded {len(chart)} type matchups from {path.name}')
    return chart


def effectiveness(attack_type, defense_types, chart):
    if not chart or norm(attack_type) in MISSING or not defense_types:
        return np.nan
    score = 1.0
    for defense_type in defense_types:
        key = (norm(attack_type), norm(defense_type))
        if key not in chart:
            return np.nan
        score *= chart[key]
    return float(score)


def best_stab_matchup(attacker, defender, chart):
    ats = clean_types(dct(attacker).get('types'))
    dfs = clean_types(dct(defender).get('types'))
    values = [effectiveness(a, dfs, chart) for a in ats]
    values = [v for v in values if np.isfinite(v)]
    return max(values) if values else np.nan


def summary(vals, prefix, out):
    vals = np.asarray([fnum(x, np.nan) for x in vals], dtype=float)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        for op in ('mean', 'min', 'max', 'std'):
            out[f'{prefix}_{op}'] = 0.0
    else:
        out[f'{prefix}_mean'] = float(np.mean(vals))
        out[f'{prefix}_min'] = float(np.min(vals))
        out[f'{prefix}_max'] = float(np.max(vals))
        out[f'{prefix}_std'] = float(np.std(vals))


def slope(y):
    if len(y) < 2:
        return 0.0
    a = np.asarray(y, dtype=float)
    x = np.arange(len(a), dtype=float)
    return float(np.sum((x - x.mean()) * (a - a.mean())) /
                 max(np.sum((x - x.mean()) ** 2), 1e-12))


def extract_battle(battle, lookup, vocab, chart=None):
    """One battle -> descriptive, turn-aware features; no target/ID is consumed."""
    chart = chart or {}
    out = {}
    team = [dct(p) for p in (battle.get('p1_team_details') or [])]
    lead2 = dct(battle.get('p2_lead_details'))
    tline = sorted((dct(t) for t in (battle.get('battle_timeline') or [])),
                   key=lambda x: fnum(x.get('turn')))[:30]
    team_by_name = {norm(p.get('name')): p for p in team}
    team_names = list(team_by_name)
    team_name_set = set(team_names)

    out['team_size'] = len(team)
    for stat in STATS:
        summary([p.get(f'base_{stat}') for p in team], f'team_{stat}', out)
    team_total = [sum(fnum(p.get(f'base_{s}')) for s in STATS) for p in team]
    summary(team_total, 'team_base_total', out)
    for s in ('atk', 'spa', 'def', 'spd'):
        out[f'team_top2_{s}'] = sum(sorted([fnum(p.get(f'base_{s}')) for p in team], reverse=True)[:2])
    out['team_physical_bias'] = out['team_atk_mean'] - out['team_spa_mean']
    out['team_defensive_bias'] = out['team_def_mean'] - out['team_spd_mean']
    out['team_fast_count'] = sum(fnum(p.get('base_spe')) >= 100 for p in team)
    out['team_slow_count'] = sum(fnum(p.get('base_spe')) < 60 for p in team)
    out['team_bulk_mean'] = out['team_hp_mean'] + out['team_def_mean'] + out['team_spd_mean']
    out['team_offense_mean'] = out['team_atk_mean'] + out['team_spa_mean']
    for typ in vocab['types']:
        out[f'team_type_{safe_key(typ)}'] = sum(typ in clean_types(p.get('types')) for p in team)
    for item in vocab['items']:
        out[f'team_item_{safe_key(item)}'] = sum(norm(p.get('item')) == item for p in team)
    out['team_number_unique_types'] = len(set(t for p in team for t in clean_types(p.get('types'))))
    out['team_number_unique_items'] = len(set(norm(p.get('item')) for p in team))

    for stat in STATS:
        out[f'lead_p2_base_{stat}'] = fnum(lead2.get(f'base_{stat}'))
    out['lead_p2_name'] = norm(lead2.get('name'))
    out['lead_p2_types'] = '_'.join(sorted(clean_types(lead2.get('types')))) or 'unknown'
    out['lead_p2_total'] = sum(out[f'lead_p2_base_{stat}'] for stat in STATS)
    out['lead_p1_name'] = norm(dct(dct(tline[0]).get('p1_pokemon_state')).get('name')) if tline else 'unknown'
    lead1 = team_by_name.get(out['lead_p1_name'], lookup.get(out['lead_p1_name'], {}))
    out['lead_p1_item'] = norm(lead1.get('item'))
    out['lead_p1_types'] = '_'.join(sorted(clean_types(lead1.get('types')))) or 'unknown'
    for stat in STATS:
        v1 = fnum(lead1.get(f'base_{stat}'))
        v2 = fnum(lead2.get(f'base_{stat}'))
        out[f'lead_stat_diff_{stat}'] = v1 - v2
        out[f'lead_teammean_diff_{stat}'] = out[f'team_{stat}_mean'] - v2
    out['lead_p1_stab_effectiveness'] = best_stab_matchup(lead1, lead2, chart) if chart else np.nan
    out['lead_p2_stab_effectiveness'] = best_stab_matchup(lead2, lead1, chart) if chart else np.nan

    states = {'p1': {}, 'p2': {}}
    trace = []
    moves = {'p1': Counter(), 'p2': Counter()}
    side_counts = {s: Counter() for s in ('p1', 'p2')}
    move_powers = {'p1': [], 'p2': []}
    move_acc = {'p1': [], 'p2': []}
    move_effects = {'p1': [], 'p2': []}
    hp_events = {'p1': [], 'p2': []}
    previous_active = {'p1': None, 'p2': None}
    first_active = {}
    last_active = {}
    last_move = {}
    seen_moves_by_name = {'p1': set(), 'p2': set()}

    for i, turn in enumerate(tline):
        turn_n = fnum(turn.get('turn'), i + 1)
        row = {'turn': turn_n}
        for side in ('p1', 'p2'):
            current = dct(turn.get(f'{side}_pokemon_state'))
            move = dct(turn.get(f'{side}_move_details'))
            name = norm(current.get('name'))
            hp = fnum(current.get('hp_pct'), 1.0)
            hp = float(np.clip(hp, 0., 1.))
            status = norm(current.get('status'))
            if status in ('', 'none', 'unknown', 'nostatus'):
                status = 'nostatus'
            effects = [norm(e) for e in (current.get('effects') or [])]
            boosts = dct(current.get('boosts'))
            old = states[side].get(name)
            if name not in MISSING and name != 'unknown':
                if old is None:
                    # Approximate initial HP=100%; actual HP may already be reduced on the first observed turn.
                    delta = hp - 1.0
                    side_counts[side]['newly_revealed'] += 1
                    if side not in first_active:
                        first_active[side] = name
                else:
                    delta = hp - old['hp']
                    if old['status'] != status and status != 'nostatus':
                        side_counts[side]['new_status_events'] += 1
                hp_events[side].append(delta)
                if delta < -0.001:
                    side_counts[side]['hp_lost'] -= delta
                elif delta > 0.001:
                    side_counts[side]['hp_healed'] += delta
                states[side][name] = {
                    'hp': hp, 'status': status, 'turn': turn_n,
                    'boosts': {s: fnum(boosts.get(s)) for s in BOOSTS},
                    'effects': effects,
                }
                last_active[side] = name
                if previous_active[side] is not None and name != previous_active[side]:
                    side_counts[side]['switches'] += 1
                previous_active[side] = name
            if status != 'nostatus':
                side_counts[side]['status_turns'] += 1
            for e in effects:
                if e not in ('noeffect', 'none', 'unknown'):
                    side_counts[side]['effect_turns'] += 1
            row[f'{side}_hp'] = hp
            row[f'{side}_status'] = int(status != 'nostatus')
            row[f'{side}_boost_total'] = sum(fnum(boosts.get(s)) for s in BOOSTS)
            row[f'{side}_boost_positive'] = sum(max(0., fnum(boosts.get(s))) for s in BOOSTS)
            row[f'{side}_boost_negative'] = sum(min(0., fnum(boosts.get(s))) for s in BOOSTS)

            if move and move.get('name'):
                mvname = norm(move['name'])
                moves[side][mvname] += 1
                seen_moves_by_name[side].add(mvname)
                last_move[side] = mvname
                category = norm(move.get('category'))
                move_type = norm(move.get('type'))
                side_counts[side]['move_' + category] += 1
                side_counts[side]['move_type_' + move_type] += 1
                bp = fnum(move.get('base_power'))
                ac = fnum(move.get('accuracy'), 1.0)
                move_powers[side].append(bp)
                move_acc[side].append(ac)
                side_counts[side]['nominal_damage'] += bp * ac
                user_info = team_by_name.get(name, lookup.get(name, {}))
                if move_type in clean_types(user_info.get('types')):
                    side_counts[side]['stab_moves'] += 1
                if chart:
                    other = 'p2' if side == 'p1' else 'p1'
                    victim = dct(turn.get(f'{other}_pokemon_state'))
                    victim_name = norm(victim.get('name'))
                    defender_info = team_by_name.get(victim_name, lookup.get(victim_name, {}))
                    eff = effectiveness(move_type, clean_types(defender_info.get('types')), chart)
                    if np.isfinite(eff):
                        move_effects[side].append(eff)
                        side_counts[side]['eff_power'] += bp * ac * eff
                        side_counts[side]['super_effective_moves'] += int(eff > 1.)
                        side_counts[side]['immune_moves'] += int(eff == 0.)
            else:
                side_counts[side]['no_move_turns'] += 1

        # Only known/revealed information is used for both players. Unrevealed
        # opponents are *assumed* at full HP for the six-slot estimate.
        p1hp = [states['p1'][n]['hp'] if n in states['p1'] else 1.0 for n in team_names]
        p2revealed = [s['hp'] for s in states['p2'].values()]
        p2hp = p2revealed + [1.] * max(0, 6 - len(p2revealed))
        row['p1_team_hp_est'] = float(np.mean(p1hp)) if p1hp else 1.0
        row['p2_team_hp_est'] = float(np.mean(p2hp)) if p2hp else 1.0
        row['p1_fainted'] = sum(h <= 0.001 for h in p1hp)
        row['p2_fainted'] = sum(h <= 0.001 for h in p2revealed)
        row['p2_revealed_count'] = len(states['p2'])
        row['active_hp_advantage'] = row['p1_hp'] - row['p2_hp']
        row['team_hp_advantage'] = row['p1_team_hp_est'] - row['p2_team_hp_est']
        row['fainted_advantage'] = row['p2_fainted'] - row['p1_fainted']
        trace.append(row)

    out['battle_observed_turns'] = len(trace)
    out['battle_p2_revealed_count'] = len(states['p2'])
    out['battle_p1_revealed_count'] = len(states['p1'])
    out['battle_opponent_hidden_count'] = max(0, 6 - len(states['p2']))
    out['battle_novel_p2_after10'] = len({norm(dct(x.get('p2_pokemon_state')).get('name'))
                                         for x in tline[10:] if dct(x.get('p2_pokemon_state')).get('name')})
    for side in ('p1', 'p2'):
        known = states[side]
        out[f'{side}_last_name'] = last_active.get(side, 'unknown')
        out[f'{side}_last_move'] = last_move.get(side, 'unknown')
        out[f'{side}_last_status'] = known.get(last_active.get(side), {}).get('status', 'nostatus')
        out[f'{side}_known_fainted'] = sum(s['hp'] <= 0.001 for s in known.values())
        out[f'{side}_known_healthy'] = sum(s['hp'] >= 0.75 for s in known.values())
        out[f'{side}_known_critical'] = sum(s['hp'] <= 0.25 for s in known.values())
        out[f'{side}_known_bad_status'] = sum(s['status'] != 'nostatus' for s in known.values())
        out[f'{side}_known_toxic'] = sum(s['status'] == 'tox' for s in known.values())
        out[f'{side}_known_paralyzed'] = sum(s['status'] in ('par', 'paralysis') for s in known.values())
        out[f'{side}_known_sleeping'] = sum(s['status'] in ('slp', 'sleep') for s in known.values())
        out[f'{side}_known_burned'] = sum(s['status'] in ('brn', 'burn') for s in known.values())
        out[f'{side}_known_poisoned'] = sum(s['status'] in ('psn', 'tox', 'poison') for s in known.values())
        out[f'{side}_known_boost_positive'] = sum(max(0, v) for s in known.values() for v in s['boosts'].values())
        out[f'{side}_known_boost_negative'] = sum(min(0, v) for s in known.values() for v in s['boosts'].values())
        out[f'{side}_known_effects'] = sum(len([x for x in s['effects'] if x not in ('noeffect', 'unknown')]) for s in known.values())
        out[f'{side}_status_distinct'] = len({s['status'] for s in known.values() if s['status'] != 'nostatus'})
        hp_known = [v['hp'] for v in known.values()]
        hp_est = ([known[n]['hp'] if n in known else 1.0 for n in team_names]
                  if side == 'p1' else hp_known + [1.] * max(0, 6 - len(hp_known)))
        summary(hp_known, f'{side}_revealed_hp', out)
        summary(hp_est, f'{side}_six_hp_est', out)
        out[f'{side}_remaining_slots_est'] = 6 - out[f'{side}_known_fainted']
        out[f'{side}_revealed_below_half'] = sum(h < 0.5 for h in hp_known)
        out[f'{side}_revealed_above_half'] = sum(h > 0.5 for h in hp_known)
        out[f'{side}_move_unique'] = len(seen_moves_by_name[side])
        out[f'{side}_move_count'] = sum(moves[side].values())
        out[f'{side}_move_spam_ratio'] = (max(moves[side].values()) / max(1, sum(moves[side].values()))
                                           if moves[side] else 0.)
        for s in ('switches', 'newly_revealed', 'new_status_events', 'status_turns',
                  'effect_turns', 'hp_lost', 'hp_healed', 'no_move_turns',
                  'nominal_damage', 'stab_moves', 'eff_power',
                  'super_effective_moves', 'immune_moves'):
            out[f'{side}_{s}'] = side_counts[side][s]
        for category in ('physical', 'special', 'status'):
            out[f'{side}_move_{category}'] = side_counts[side]['move_' + category]
        for typ in vocab['types']:
            out[f'{side}_move_type_{safe_key(typ)}'] = side_counts[side]['move_type_' + typ]
        summary(move_powers[side], f'{side}_move_basepower', out)
        summary(move_acc[side], f'{side}_move_accuracy', out)
        summary(move_effects[side], f'{side}_move_type_effect', out)
        summary(hp_events[side], f'{side}_hp_change', out)
        for stat in BOOSTS:
            last = known.get(last_active.get(side), {})
            out[f'{side}_last_boost_{stat}'] = last.get('boosts', {}).get(stat, 0.)
        for name in vocab['species']:
            out[f'{side}_species_{safe_key(name)}'] = int(name in (team_name_set if side == 'p1' else known))
        for name in vocab['moves']:
            out[f'{side}_used_{safe_key(name)}'] = int(moves[side][name] > 0)
            # Move counts are useful for repeated Recover/Toxic/etc.
            out[f'{side}_count_{safe_key(name)}'] = moves[side][name]

        current_name = last_active.get(side)
        current_info = team_by_name.get(current_name, lookup.get(current_name, {}))
        for stat in STATS:
            out[f'{side}_last_base_{stat}'] = fnum(current_info.get(f'base_{stat}'))
        out[f'{side}_last_types'] = '_'.join(sorted(clean_types(current_info.get('types')))) or 'unknown'
        for stat in ('atk', 'spa', 'def', 'spd', 'spe'):
            weighted = [(fnum((team_by_name.get(n) or lookup.get(n, {})).get(f'base_{stat}')), state['hp'])
                        for n, state in known.items()]
            out[f'{side}_revealed_healthy_base_{stat}'] = sum(v * hp for v, hp in weighted) / max(1., sum(hp for _, hp in weighted))

    paired_metrics = ('known_fainted', 'known_critical', 'known_bad_status', 'known_boost_positive',
                      'known_boost_negative', 'move_count', 'switches', 'hp_lost', 'hp_healed',
                      'move_physical', 'move_special', 'move_status', 'nominal_damage',
                      'revealed_hp_mean', 'six_hp_est_mean', 'move_unique', 'status_turns',
                      'remaining_slots_est')
    for metric in paired_metrics:
        out[f'adv_{metric}'] = out.get(f'p1_{metric}', 0) - out.get(f'p2_{metric}', 0)
    out['adv_damage_dealt_proxy'] = out['p2_hp_lost'] - out['p1_hp_lost']
    out['adv_healing'] = out['p1_hp_healed'] - out['p2_hp_healed']
    out['adv_healthy_remaining'] = out['p1_remaining_slots_est'] - out['p2_remaining_slots_est']
    out['adv_last_speed'] = out['p1_last_base_spe'] - out['p2_last_base_spe']
    out['adv_last_physical_attack_vs_defense'] = out['p1_last_base_atk'] - out['p2_last_base_def']
    out['adv_last_special_attack_vs_defense'] = out['p1_last_base_spa'] - out['p2_last_base_spd']
    out['adv_last_stab_effectiveness'] = (best_stab_matchup(
        team_by_name.get(last_active.get('p1'), lookup.get(last_active.get('p1'), {})),
        lookup.get(last_active.get('p2'), {}), chart) if chart else np.nan)
    out['adv_last_stab_effectiveness_opponent'] = (best_stab_matchup(
        lookup.get(last_active.get('p2'), {}),
        team_by_name.get(last_active.get('p1'), lookup.get(last_active.get('p1'), {})),
        chart) if chart else np.nan)

    # Time windows: early / mid / late / very late. Trends describe momentum.
    ranges = {
        'early': trace[:10], 'middle': trace[10:20], 'late': trace[20:30],
        'last5': trace[-5:],
    }
    measurements = ('p1_hp', 'p2_hp', 'active_hp_advantage', 'p1_team_hp_est',
                    'p2_team_hp_est', 'team_hp_advantage', 'fainted_advantage',
                    'p1_status', 'p2_status', 'p1_boost_total', 'p2_boost_total',
                    'p2_revealed_count')
    for prefix, rows in ranges.items():
        for measure in measurements:
            vals = [r[measure] for r in rows]
            out[f'window_{prefix}_{measure}_mean'] = float(np.mean(vals)) if vals else 0.
            out[f'window_{prefix}_{measure}_last'] = float(vals[-1]) if vals else 0.
    for measure in measurements:
        vals = [r[measure] for r in trace]
        out[f'trend_{measure}_slope'] = slope(vals)
        out[f'trend_{measure}_change'] = vals[-1] - vals[0] if vals else 0.
        out[f'trend_{measure}_late_minus_early'] = (
            out[f'window_late_{measure}_mean'] - out[f'window_early_{measure}_mean'])
    out['trend_team_hp_gap_final'] = trace[-1]['team_hp_advantage'] if trace else 0.
    out['trend_active_hp_gap_final'] = trace[-1]['active_hp_advantage'] if trace else 0.
    out['trend_fainted_gap_final'] = trace[-1]['fainted_advantage'] if trace else 0.
    out['trend_p2_revealed_last'] = len(states['p2'])

    # Optional supplied Gen2 type chart: type-only proxies, not exact damage.
    if chart:
        p2_revealed_info = [lookup.get(name, {}) for name in states['p2']]
        for group_name, attackers, defenders in (
            ('team_vs_p2_revealed', team, p2_revealed_info),
            ('p2_revealed_vs_team', p2_revealed_info, team),
        ):
            matches = [best_stab_matchup(a, b, chart) for a in attackers for b in defenders]
            matches = [m for m in matches if np.isfinite(m)]
            out[f'type_{group_name}_mean'] = float(np.mean(matches)) if matches else 1.
            out[f'type_{group_name}_max'] = max(matches) if matches else 1.
            out[f'type_{group_name}_weak_count'] = sum(m > 1. for m in matches)
    return out


def make_features(records, lookup, vocab, chart=None, verbose=True):
    if verbose:
        print(f'Extracting temporal and team features for {len(records):,} battles...')
    rows = [extract_battle(r, lookup, vocab, chart) for r in records]
    df = pd.DataFrame(rows)
    cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    for c in cat_cols:
        df[c] = df[c].fillna('unknown').astype(str)
    num_cols = df.columns.difference(cat_cols)
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(-1.).astype('float32')
    return df
