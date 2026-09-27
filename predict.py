#!/usr/bin/env python3
import sqlite3
import pandas as pd
import numpy as np
import os
import argparse
import json
from catboost import CatBoostRegressor

from rating_system_SHARED import parse_role, is_seed_layer

DB_PATH = 'squadDB.sqlite'
CB_MODEL_PATH = 'layer_effect_model.cbm'
CB_META_PATH = 'layer_effect_model_meta.json'

import urllib.request
import os

STEAM_API_KEY = os.environ.get('STEAM_API_KEY', 'YOUR_STEAM_API_KEY_HERE')

def fetch_steam_names(player_ids, db_path):
    if not player_ids:
        return {}

    conn = sqlite3.connect(db_path)
    placeholders = ','.join('?' for _ in player_ids)
    
    query = f"""
        SELECT canonical_id, alias FROM player_aliases 
        WHERE canonical_id IN ({placeholders}) AND alias LIKE '7656119%'
    """
    df = pd.read_sql_query(query, conn, params=player_ids)
    conn.close()

    id_to_steamid = {}
    for pid in player_ids:
        if str(pid).startswith('7656119') and len(str(pid)) == 17:
            id_to_steamid[pid] = str(pid)

    for _, row in df.iterrows():
        cid = row['canonical_id']
        alias = str(row['alias'])
        if cid not in id_to_steamid and alias.startswith('7656119') and len(alias) == 17:
            id_to_steamid[cid] = alias

    if not id_to_steamid:
        return {}

    steam_ids = list(set(id_to_steamid.values()))
    
    url = f"http://api.steampowered.com/ISteamUser/GetPlayerSummaries/v0002/?key={STEAM_API_KEY}&steamids={','.join(steam_ids)}"
    steamid_to_name = {}

    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            for player in data.get('response', {}).get('players', []):
                steamid_to_name[player['steamid']] = player.get('personaname', 'Unknown')
    except Exception as e:
        print(f"  [Warning] Could not fetch Steam names: {e}")

    final_map = {}
    for cid, sid in id_to_steamid.items():
        final_map[cid] = steamid_to_name.get(sid, 'Unknown')

    return final_map


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def parse_faction_type(faction_type):
    """Parse 'WPMC+LightInfantry' → ('WPMC', 'WPMC+LightInfantry', 'LightInfantry')"""
    if pd.isna(faction_type) or str(faction_type).strip() == "":
        return "Unknown", "Unknown", ""
    parts = str(faction_type).split('+')
    if len(parts) == 2:
        return parts[0].strip(), faction_type.strip(), parts[1].strip()
    return faction_type.strip(), faction_type.strip(), ""


def extract_version(layer_name):
    if pd.isna(layer_name):
        return 'v1'
    for part in reversed(str(layer_name).split()):
        if part.startswith('v') and part[1:].isdigit():
            return part
    return 'v1'


def load_catboost_model():
    """Load CatBoost model and metadata."""
    if not os.path.exists(CB_MODEL_PATH):
        return None, None, None
    model = CatBoostRegressor()
    model.load_model(CB_MODEL_PATH)
    with open(CB_META_PATH, 'r') as f:
        meta = json.load(f)
    return model, meta['features'], meta['cat_features']


def get_match_data(db_path, match_id):
    conn = sqlite3.connect(db_path)

    match_df = pd.read_sql_query("""
        SELECT matchId, gamemode, layerName, winningTeamID, winningTeam, losingTeam,
               winningSubfaction, losingSubfaction, duration,
               faction1_type, faction2_type
        FROM matches WHERE matchId = ?
    """, conn, params=(match_id,))

    if match_df.empty:
        conn.close()
        raise ValueError(f"Match {match_id} not found")

    teams_df = pd.read_sql_query("""
        SELECT matchId, teamID, canonical_id, steamID, role
        FROM teams WHERE matchId = ?
    """, conn, params=(match_id,))

    try:
        ratings_df = pd.read_sql_query("SELECT player_id, general_skill, sigma FROM player_ratings", conn)
    except:
        ratings_df = pd.DataFrame(columns=['player_id', 'general_skill', 'sigma'])

    try:
        role_ratings_df = pd.read_sql_query("SELECT player_id, role, role_skill FROM player_role_ratings", conn)
    except:
        role_ratings_df = pd.DataFrame(columns=['player_id', 'role', 'role_skill'])

    conn.close()
    return match_df.iloc[0].to_dict(), teams_df, ratings_df, role_ratings_df


def compute_skill_for_match(match_row, teams_df, ratings_df, role_ratings_df):
    skill_dict = {}
    sigma_dict = {}
    if len(ratings_df) > 0:
        ratings_df = ratings_df.copy()
        ratings_df['player_id'] = ratings_df['player_id'].astype(str)
        skill_dict = dict(zip(ratings_df['player_id'], ratings_df['general_skill']))
        sigma_dict = dict(zip(ratings_df['player_id'], ratings_df['sigma']))

    role_skill_dict = {}
    if len(role_ratings_df) > 0:
        role_ratings_df = role_ratings_df.copy()
        role_ratings_df['player_id'] = role_ratings_df['player_id'].astype(str)
        role_ratings_df['role'] = role_ratings_df['role'].astype(str)
        role_skill_dict = dict(zip(
            role_ratings_df['player_id'] + '|' + role_ratings_df['role'],
            role_ratings_df['role_skill']
        ))

    players = []
    for _, row in teams_df.iterrows():
        pid = None
        if pd.notna(row.get('canonical_id')):
            pid = str(row['canonical_id'])
        elif pd.notna(row.get('steamID')):
            sid = row['steamID']
            try:
                pid = str(int(sid)) if float(sid).is_integer() else str(sid)
            except:
                pid = str(sid)
        
        if not pid or pid == 'nan':
            continue
            
        role = parse_role(row.get('role'))
        team = int(row['teamID'])
        
        g_skill = float(skill_dict.get(pid, 0.0))
        sigma = float(sigma_dict.get(pid, 2.0))
        r_skill = float(role_skill_dict.get(f"{pid}|{role}", 0.0))
        
        players.append({
            'player_id': pid, 'role': role, 'team': team,
            'general_skill': g_skill, 'role_skill': r_skill,
            'total_skill': g_skill + r_skill, 'sigma': sigma,
        })

    if not players:
        return {
            'players': [],
            'team1_skill': 0.0, 'team2_skill': 0.0,
            'skill_diff': 0.0, 'skill_pred': 0.5,
            'skill_uncertainty': 2.828,
            'team1_sigma': 2.0, 'team2_sigma': 2.0,
        }

    t1_skill = sum(p['total_skill'] for p in players if p['team'] == 1)
    t2_skill = sum(p['total_skill'] for p in players if p['team'] == 2)
    t1_sigmas = [p['sigma'] for p in players if p['team'] == 1]
    t2_sigmas = [p['sigma'] for p in players if p['team'] == 2]
    t1_sigma = np.mean(t1_sigmas) if t1_sigmas else 2.0
    t2_sigma = np.mean(t2_sigmas) if t2_sigmas else 2.0

    skill_diff = t1_skill - t2_skill
    skill_pred = sigmoid(skill_diff)
    skill_uncertainty = np.sqrt(t1_sigma**2 + t2_sigma**2)

    return {
        'players': players,
        'team1_skill': t1_skill, 'team2_skill': t2_skill,
        'skill_diff': skill_diff, 'skill_pred': skill_pred,
        'skill_uncertainty': skill_uncertainty,
        'team1_sigma': t1_sigma, 'team2_sigma': t2_sigma,
    }

def build_features_for_match(match_row, skill_info):
    f1_type = match_row.get('faction1_type', '')
    f2_type = match_row.get('faction2_type', '')

    if pd.isna(f1_type) or str(f1_type).strip() == "":
        if match_row['winningTeamID'] == 1:
            f1_type = match_row.get('winningSubfaction', 'Unknown')
            f2_type = match_row.get('losingSubfaction', 'Unknown')
        else:
            f1_type = match_row.get('losingSubfaction', 'Unknown')
            f2_type = match_row.get('winningSubfaction', 'Unknown')

    t1_faction, t1_subfaction, t1_tag = parse_faction_type(f1_type)
    t2_faction, t2_subfaction, t2_tag = parse_faction_type(f2_type)

    layer_name = match_row['layerName']
    parts = str(layer_name).split()
    map_name = parts[0] if len(parts) > 0 else 'Unknown'
    game_mode_full = ' '.join(parts[-2:]) if len(parts) >= 2 else 'Unknown'
    gamemode = match_row.get('gamemode') or (parts[-2] if len(parts) >= 2 else 'Unknown')
    duration = float(match_row.get('duration', 60) or 60)

    return {
        'gamemode': gamemode, 'map': map_name, 'game_mode_full': game_mode_full,
        'layerName': layer_name, 'layer_version': extract_version(layer_name),
        'team1_faction': t1_faction, 'team2_faction': t2_faction,
        'team1_subfaction': t1_subfaction, 'team2_subfaction': t2_subfaction,
        'team1_tag': t1_tag, 'team2_tag': t2_tag,
        'matchup': f"{t1_subfaction} vs {t2_subfaction}",
        'duration': duration,
        'skill_pred': skill_info['skill_pred'],
        'skill_uncertainty': skill_info['skill_uncertainty'],
    }


def predict_matchid(db_path, match_id):
    print("=" * 80)
    print(f"  MATCH PREDICTION: {match_id}")
    print("=" * 80)

    match_row, teams_df, ratings_df, role_ratings_df = get_match_data(db_path, match_id)

    print(f"\n  Layer:       {match_row['layerName']}")
    print(f"  Gamemode:    {match_row.get('gamemode', 'N/A')}")
    print(f"  Duration:    {match_row.get('duration', 'N/A')} sec")
    print(f"  Winner:      Team {match_row['winningTeamID']}")

    n_t1 = (teams_df['teamID'] == 1).sum()
    n_t2 = (teams_df['teamID'] == 2).sum()
    print(f"  Roster:      {n_t1}v{n_t2}")

    skill_info = compute_skill_for_match(match_row, teams_df, ratings_df, role_ratings_df)

    model, features, cat_features = load_catboost_model()
    if model is not None:
        feat_dict = build_features_for_match(match_row, skill_info)
        X = pd.DataFrame([feat_dict])[features]
        layer_effect = float(model.predict(X)[0])
    else:
        print("\n  ⚠ CatBoost model not found — layer effect = 0")
        feat_dict = build_features_for_match(match_row, skill_info)
        layer_effect = 0.0

    eps = 1e-7
    skill_pred_clipped = np.clip(skill_info['skill_pred'], eps, 1 - eps)
    skill_logit = np.log(skill_pred_clipped / (1 - skill_pred_clipped))
    combined_logit = skill_logit + layer_effect
    combined_prob = sigmoid(combined_logit)

    print(f"\n{'─' * 80}")
    print(f"  1. PLAYER SKILL EFFECT")
    print(f"{'─' * 80}")
    print(f"  Team 1 skill sum:   {skill_info['team1_skill']:+.3f}")
    print(f"  Team 2 skill sum:   {skill_info['team2_skill']:+.3f}")
    print(f"  Skill difference:   {skill_info['skill_diff']:+.3f}  (logit space)")
    print(f"  Skill win prob:     {skill_info['skill_pred']:.3f}  (Team 1)")
    print(f"  Uncertainty (σ):    {skill_info['skill_uncertainty']:.3f}")

    print(f"\n{'─' * 80}")
    print(f"  2. LAYER EFFECT (CatBoost)")
    print(f"{'─' * 80}")
    print(f"  Faction 1:          {feat_dict['team1_subfaction']}")
    print(f"  Faction 2:          {feat_dict['team2_subfaction']}")
    print(f"  Matchup:            {feat_dict['matchup']}")
    print(f"  Layer effect:       {layer_effect:+.4f}  (logit adjustment)")
    if abs(layer_effect) < 0.01:
        print(f"  → Neutral layer")
    elif layer_effect > 0:
        print(f"  → Favors Team 1 by {layer_effect:.4f} logits ({sigmoid(layer_effect)-0.5:+.1%} win shift)")
    else:
        print(f"  → Favors Team 2 by {abs(layer_effect):.4f} logits ({0.5-sigmoid(abs(layer_effect)):+.1%} win shift)")

    print(f"\n{'─' * 80}")
    print(f"  3. COMBINED EFFECT")
    print(f"{'─' * 80}")
    print(f"  Skill logit:        {skill_logit:+.4f}")
    print(f"  Layer effect:       {layer_effect:+.4f}")
    print(f"  Combined logit:     {combined_logit:+.4f}")
    print(f"  ────────────────────────────────")
    print(f"  Combined win prob:  {combined_prob:.1%}  Team 1")
    print(f"                       {1-combined_prob:.1%}  Team 2")

    actual = int(match_row['winningTeamID'])
    predicted = 1 if combined_prob > 0.5 else 2
    print(f"\n  Predicted winner:   Team {predicted}")
    print(f"  Actual winner:      Team {actual}")
    print(f"  Correct:            {'✓ YES' if predicted == actual else '✗ NO'}")

    print(f"\n{'─' * 190}")
    print(f"  PLAYER IMPACT RANKINGS")
    print(f"{'─' * 190}")

    t1 = sorted([p for p in skill_info['players'] if p['team'] == 1],
                key=lambda x: x['total_skill'], reverse=True)
    t2 = sorted([p for p in skill_info['players'] if p['team'] == 2],
                key=lambda x: x['total_skill'], reverse=True)

    all_player_ids = [p['player_id'] for p in skill_info['players']]
    steam_names = fetch_steam_names(all_player_ids, db_path)

    print(f"\n  {'Rank':<5} {'─ TEAM 1 ─':^86}  │  {'─ TEAM 2 ─':^86}")
    print(f"  {'':5} {'ID':<22} {'Username':<20} {'Role':<13} {'GenSkill':>9} {'RoleSkill':>10} {'Total':>8}  │  "
          f"{'ID':<22} {'Username':<20} {'Role':<13} {'GenSkill':>9} {'RoleSkill':>10} {'Total':>8}")
    print(f"  {'─'*5} {'─'*22} {'─'*20} {'─'*13} {'─'*9} {'─'*10} {'─'*8}  ┼  {'─'*22} {'─'*20} {'─'*13} {'─'*9} {'─'*10} {'─'*8}")

    max_rows = max(len(t1), len(t2))
    for i in range(max_rows):
        def fmt(p):
            if p is None:
                return f"{'':22} {'':20} {'':13} {'':>9} {'':>10} {'':>8}"
            name = steam_names.get(p['player_id'], 'Unknown')[:20]
            return (f"{p['player_id'][:22]:<22} {name:<20} {p['role']:<13} "
                    f"{p['general_skill']:>+9.3f} {p['role_skill']:>+10.3f} {p['total_skill']:>+8.3f}")
        p1 = t1[i] if i < len(t1) else None
        p2 = t2[i] if i < len(t2) else None
        print(f"  {i+1:<5} {fmt(p1)}  │  {fmt(p2)}")

    print(f"\n{'─' * 80}")
    print(f"  TEAM SUMMARY")
    print(f"{'─' * 80}")
    print(f"  {'Team':<8} {'Players':<9} {'Skill Sum':>12} {'Avg Skill':>12} {'Top Player':>12} {'σ (avg)':>10}")
    print(f"  {'─'*8} {'─'*9} {'─'*12} {'─'*12} {'─'*12} {'─'*10}")
    for num, players in [(1, t1), (2, t2)]:
        n = len(players)
        s = sum(p['total_skill'] for p in players)
        avg = s / n if n else 0
        top = players[0]['total_skill'] if players else 0
        sig = np.mean([p['sigma'] for p in players]) if players else 2.0
        print(f"  Team {num:<3} {n:<9} {s:>+12.3f} {avg:>+12.3f} {top:>+12.3f} {sig:>10.3f}")



def predict_from_file(db_path, file_path):
    print("=" * 110)
    print(f"  ROTATION FILE PREDICTION: {file_path}")
    print("=" * 110)

    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f"  Loaded {len(lines)} lines\n")

    conn = sqlite3.connect(db_path)

    has_table = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='layer_effects'", conn
    ).shape[0] > 0
    if not has_table:
        print("  ⚠ layer_effects table not found in DB. Results will be CatBoost predictions only.\n")

    model, features, cat_features = load_catboost_model()
    if model is None:
        print("  ⚠ CatBoost model not found. Only DB lookups will be shown.\n")

    print(f"  {'#':<4} {'Rotation':<58} {'Effect':>8} {'WinProb':>8} {'Actual':>8} {'N':>5} {'Status':<12}")
    print(f"  {'─'*4} {'─'*58} {'─'*8} {'─'*8} {'─'*8} {'─'*5} {'─'*12}")

    found_count = 0
    predicted_count = 0
    notfound_count = 0

    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 3:
            print(f"  {i+1:<4} {line[:58]:<58} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>5} {'PARSE_ERR':<12}")
            notfound_count += 1
            continue

        rotation_format = line

        if has_table:
            df = pd.read_sql_query(
                "SELECT predicted_effect, actual_team1_winrate, sample_count FROM layer_effects WHERE rotation_format = ?",
                conn, params=(rotation_format,)
            )
        else:
            df = pd.DataFrame()

        if not df.empty:
            row = df.iloc[0]
            effect = float(row['predicted_effect'])
            win_prob = sigmoid(effect)  
            actual = row['actual_team1_winrate']
            samples = int(row['sample_count'])
            actual_str = f"{actual:.3f}" if pd.notna(actual) else "  N/A"
            print(f"  {i+1:<4} {rotation_format[:58]:<58} {effect:>+8.4f} {win_prob:>8.1%} {actual_str:>8} {samples:>5} {'FOUND':<12}")
            found_count += 1
        elif model is not None:
            layer_underscored = parts[0]
            f1_type = parts[1]
            f2_type = parts[2] if len(parts) > 2 else ""

            layer_name = layer_underscored.replace('_', ' ')
            t1_faction, t1_subfaction, t1_tag = parse_faction_type(f1_type)
            t2_faction, t2_subfaction, t2_tag = parse_faction_type(f2_type)

            ln_parts = layer_name.split()
            map_name = ln_parts[0] if len(ln_parts) > 0 else 'Unknown'
            game_mode_full = ' '.join(ln_parts[-2:]) if len(ln_parts) >= 2 else 'Unknown'
            gamemode = ln_parts[-2] if len(ln_parts) >= 2 else 'Unknown'

            feat_dict = {
                'gamemode': gamemode, 'map': map_name, 'game_mode_full': game_mode_full,
                'layerName': layer_name, 'layer_version': extract_version(layer_name),
                'team1_faction': t1_faction, 'team2_faction': t2_faction,
                'team1_subfaction': t1_subfaction, 'team2_subfaction': t2_subfaction,
                'team1_tag': t1_tag, 'team2_tag': t2_tag,
                'matchup': f"{t1_subfaction} vs {t2_subfaction}",
                'duration': 60.0, 'skill_pred': 0.5, 'skill_uncertainty': 2.0,
            }
            try:
                X = pd.DataFrame([feat_dict])[features]
                effect = float(model.predict(X)[0])
                win_prob = sigmoid(effect)
                print(f"  {i+1:<4} {rotation_format[:58]:<58} {effect:>+8.4f} {win_prob:>8.1%} {'N/A':>8} {'0':>5} {'PREDICTED':<12}")
                predicted_count += 1
            except Exception as e:
                print(f"  {i+1:<4} {rotation_format[:58]:<58} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>5} {'ERROR':<12}")
                notfound_count += 1
        else:
            print(f"  {i+1:<4} {rotation_format[:58]:<58} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>5} {'NOT_FOUND':<12}")
            notfound_count += 1

    conn.close()

    print(f"\n  {'─'*4} {'─'*58} {'─'*8} {'─'*8} {'─'*8} {'─'*5} {'─'*12}")
    print(f"\n  Summary: {found_count} found in DB, {predicted_count} predicted by CatBoost, {notfound_count} not found")
    print(f"  Note: WinProb assumes equal teams (skill_pred=0.5). For match-specific predictions, use --matchid.")



def main():
    parser = argparse.ArgumentParser(
        description='Match Predictor — predict match outcomes and player impact',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python predict_match.py --matchid 12345
            python predict_match.py --matchid 12345 --db custom.db
            python predict_match.py --file rotations.txt
        """
    )
    parser.add_argument('--matchid', type=str, help='Match ID to predict')
    parser.add_argument('--file', type=str, help='File with rotation-format lines (one per line)')
    parser.add_argument('--db', type=str, default=DB_PATH, help='Database path')
    args = parser.parse_args()

    if args.matchid:
        predict_matchid(args.db, args.matchid)
    elif args.file:
        predict_from_file(args.db, args.file)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()