#!/usr/bin/env python3

import sqlite3
import pandas as pd
import numpy as np
import os
import json
import gc
from catboost import CatBoostRegressor
from sklearn.metrics import log_loss
from rating_system_SHARED import parse_role, DURATION_CAP, DURATION_WEIGHT_SCALE, USE_DURATION_WEIGHT

DB_PATH = "squadDB.sqlite"
CB_MODEL_PATH = "layer_effect_model.cbm"
CB_META_PATH = "layer_effect_model_meta.json"

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _parse_faction_type(faction_type):
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

def load_and_aggregate_data():
    print("Loading and aggregating data in chunks...")
    conn = sqlite3.connect(DB_PATH)
    
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(matches)")
    columns = [col[1] for col in cursor.fetchall()]
    has_faction_types = 'faction1_type' in columns and 'faction2_type' in columns

    select_cols = """matchId, gamemode, layerName, winningTeamID, winningTeam,
                      winningSubfaction, losingTeam, losingSubfaction, duration"""
    if has_faction_types:
        select_cols += ", faction1_type, faction2_type"

    matches_df = pd.read_sql_query(
        f"SELECT {select_cols} FROM matches WHERE teamdataCollected = 1 ORDER BY matchId ASC", conn)
    matches_df['actual_win'] = (matches_df['winningTeamID'] == 1).astype(int)

    try:
        ratings_df = pd.read_sql_query("SELECT player_id, general_skill, sigma FROM player_ratings", conn)
    except:
        ratings_df = pd.DataFrame(columns=['player_id', 'general_skill', 'sigma'])
        
    try:
        role_ratings_df = pd.read_sql_query("SELECT player_id, role, role_skill FROM player_role_ratings", conn)
    except:
        role_ratings_df = pd.DataFrame(columns=['player_id', 'role', 'role_skill'])
        
    skill_dict = ratings_df.set_index('player_id')['general_skill'].to_dict() if len(ratings_df) > 0 else {}
    sigma_dict = ratings_df.set_index('player_id')['sigma'].to_dict() if len(ratings_df) > 0 else {}
    
    if len(role_ratings_df) > 0:
        role_ratings_df['key'] = role_ratings_df['player_id'] + '|' + role_ratings_df['role']
        role_skill_dict = role_ratings_df.set_index('key')['role_skill'].to_dict()
    else:
        role_skill_dict = {}

    chunksize = 500_000
    query = "SELECT matchId, teamID, canonical_id, role FROM teams WHERE canonical_id IS NOT NULL"
    
    agg_list = []
    for chunk in pd.read_sql_query(query, conn, chunksize=chunksize):
        chunk['gen_skill'] = chunk['canonical_id'].map(skill_dict).fillna(0.0)
        chunk['sigma'] = chunk['canonical_id'].map(sigma_dict).fillna(2.0)
        chunk['parsed_role'] = chunk['role'].apply(parse_role)
        chunk['role_key'] = chunk['canonical_id'].astype(str) + '|' + chunk['parsed_role']
        chunk['role_skill'] = chunk['role_key'].map(role_skill_dict).fillna(0.0)
        chunk['total_skill'] = chunk['gen_skill'] + chunk['role_skill']
        
        agg_chunk = chunk.groupby(['matchId', 'teamID']).agg(
            skill_sum=('total_skill', 'sum'),
            sigma_sum=('sigma', 'sum'),
            n_players=('total_skill', 'size')
        ).reset_index()
        
        agg_list.append(agg_chunk)
        del chunk
        gc.collect()
        
    conn.close()
    
    if not agg_list:
        return matches_df
        
    full_agg = pd.concat(agg_list, ignore_index=True)
    teams_agg = full_agg.groupby(['matchId', 'teamID']).sum().reset_index()
    teams_agg['sigma_mean'] = teams_agg['sigma_sum'] / teams_agg['n_players']
    
    t1 = teams_agg[teams_agg['teamID'] == 1][['matchId', 'skill_sum', 'sigma_mean']].rename(
        columns={'skill_sum': 't1_skill', 'sigma_mean': 't1_sigma'})
    t2 = teams_agg[teams_agg['teamID'] == 2][['matchId', 'skill_sum', 'sigma_mean']].rename(
        columns={'skill_sum': 't2_skill', 'sigma_mean': 't2_sigma'})
    
    matches_df = matches_df.merge(t1, on='matchId', how='left').merge(t2, on='matchId', how='left')
    
    matches_df['t1_skill'] = matches_df['t1_skill'].fillna(0.0)
    matches_df['t2_skill'] = matches_df['t2_skill'].fillna(0.0)
    matches_df['t1_sigma'] = matches_df['t1_sigma'].fillna(2.0)
    matches_df['t2_sigma'] = matches_df['t2_sigma'].fillna(2.0)

    
    matches_df['skill_diff'] = matches_df['t1_skill'] - matches_df['t2_skill']
    matches_df['skill_pred'] = sigmoid(matches_df['skill_diff'])
    matches_df['skill_uncertainty'] = np.sqrt(matches_df['t1_sigma']**2 + matches_df['t2_sigma']**2)
    
    return matches_df

def prepare_features(matches_df):
    parts = matches_df['layerName'].str.split()
    matches_df['map'] = parts.str[0]
    matches_df['game_mode_full'] = parts.str[-2:].str.join(' ')

    if 'faction1_type' in matches_df.columns:
        matches_df['team1_faction'], matches_df['team1_subfaction'], matches_df['team1_tag'] = \
            zip(*matches_df['faction1_type'].apply(_parse_faction_type))
        matches_df['team2_faction'], matches_df['team2_subfaction'], matches_df['team2_tag'] = \
            zip(*matches_df['faction2_type'].apply(_parse_faction_type))
    else:
        matches_df['team1_faction'] = np.where(matches_df['winningTeamID'] == 1, matches_df['winningTeam'], matches_df['losingTeam'])
        matches_df['team2_faction'] = np.where(matches_df['winningTeamID'] == 1, matches_df['losingTeam'], matches_df['winningTeam'])
        matches_df['team1_subfaction'] = np.where(matches_df['winningTeamID'] == 1, matches_df['winningSubfaction'], matches_df['losingSubfaction'])
        matches_df['team2_subfaction'] = np.where(matches_df['winningTeamID'] == 1, matches_df['losingSubfaction'], matches_df['winningSubfaction'])
        matches_df['team1_tag'] = ""
        matches_df['team2_tag'] = ""

    matches_df['matchup'] = matches_df['team1_subfaction'] + ' vs ' + matches_df['team2_subfaction']
    matches_df['layer_version'] = matches_df['layerName'].apply(extract_version)
    matches_df['duration'] = matches_df['duration'].fillna(60).clip(lower=10, upper=120)
    
    return matches_df

def main():
    matches_df = load_and_aggregate_data()
    if matches_df.empty:
        print("No matches found.")
        return

    matches_df = prepare_features(matches_df)
    

    model = CatBoostRegressor()
    model.load_model(CB_MODEL_PATH)
    with open(CB_META_PATH, 'r') as f:
        meta = json.load(f)
    features = meta['features']


    df_layer_only = matches_df.copy()
    df_layer_only['skill_pred'] = 0.5
    df_layer_only['skill_uncertainty'] = 2.0
    pure_layer_effects = model.predict(df_layer_only[features])

    actual_layer_effects = model.predict(matches_df[features])
    
    matches_df['skill_prob'] = sigmoid(matches_df['skill_diff'])
    matches_df['layer_prob'] = sigmoid(pure_layer_effects)
    matches_df['combined_logit'] = matches_df['skill_diff'] + actual_layer_effects
    matches_df['combined_prob'] = sigmoid(matches_df['combined_logit'])
    
    y_true = matches_df['actual_win'].values

    weights = (matches_df['duration'] / DURATION_WEIGHT_SCALE).clip(upper=DURATION_CAP / DURATION_WEIGHT_SCALE).values if USE_DURATION_WEIGHT else np.ones(len(matches_df))

    def calc_metrics(y_pred):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        brier = np.average((y_pred - y_true) ** 2, weights=weights)
        acc = np.mean((y_pred > 0.5) == (y_true == 1))
        logloss = log_loss(y_true, y_pred_clipped, sample_weight=weights)
        return brier, acc, logloss

    brier_s, acc_s, ll_s = calc_metrics(matches_df['skill_prob'].values)
    brier_l, acc_l, ll_l = calc_metrics(matches_df['layer_prob'].values)
    brier_c, acc_c, ll_c = calc_metrics(matches_df['combined_prob'].values)

    print(f"\n{'='*70}")
    print(f"COMBINED MODEL EVALUATION")
    print(f"{'='*70}")
    print(f"Evaluated on {len(matches_df):,} matches.\n")
    
    print(f"{'Model':<25} | {'Accuracy':>10} | {'Brier Score':>12} | {'Log Loss':>10}")
    print(f"{'-'*25}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}")
    print(f"{'Player Skill Only':<25} | {acc_s:>10.2%} | {brier_s:>12.4f} | {ll_s:>10.4f}")
    print(f"{'Layer Effect Only':<25} | {acc_l:>10.2%} | {brier_l:>12.4f} | {ll_l:>10.4f}")
    print(f"{'Combined (Skill+Layer)':<25} | {acc_c:>10.2%} | {brier_c:>12.4f} | {ll_c:>10.4f}")

if __name__ == "__main__":
    main()