#!/usr/bin/env python3
"""
Full Training Pipeline
======================
Merges: update_faction_entries → u_rating_system → effects
"""

import sqlite3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import json
import csv
import gc
import warnings
import argparse
from collections import defaultdict, Counter
from catboost import CatBoostRegressor, Pool

warnings.filterwarnings('ignore')

from rating_system_SHARED import *



def ensure_faction_type_columns(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(matches)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'faction1_type' not in columns:
        cursor.execute("ALTER TABLE matches ADD COLUMN faction1_type TEXT")
        print("  Added faction1_type column")
    if 'faction2_type' not in columns:
        cursor.execute("ALTER TABLE matches ADD COLUMN faction2_type TEXT")
        print("  Added faction2_type column")
    conn.commit()
    conn.close()


FACTION_TYPES = {
    "Irregular Militia Forces": {
        "shortname": "IMF",
        "subfactions": {
            "Support Brigade": "Support", "Mechanized Brigade": "Mechanized",
            "Light Battalion": "LightInfantry", "Irregular Motorized Platoon": "Motorized",
            "Irregular Mechanized Platoon": "Mechanized", "Irregular Light Infantry": "LightInfantry",
            "Irregular Fire Support Group": "Support", "Irregular Battle Group": "CombinedArms",
            "Irregular Armored Squadron": "Armored", "Hoplite Battalion": "LightInfantry",
            "Combined Arms Brigade": "CombinedArms", "Armor Battalion": "Armored",
        }
    },
    "Turkish Land Forces": {
        "shortname": "TLF",
        "subfactions": {
            "1st Army Battle Group": "CombinedArms",
            "66th Mechanized Infantry Brigade Battle Group": "Mechanized",
            "1st Commando Brigade Battle Group": "AirAssault",
            "Land Forces Logistics Command Battle Group": "Support",
            "51st Motorized Infantry Brigade Battle Group": "Motorized",
        }
    },
    "Canadian Armed Forces": {
        "shortname": "CAF",
        "subfactions": {"3rd Battalion, Royal Canadian Regiment": "AirAssault"}
    },
    "Australian Defense Force": {
        "shortname": "CAF",
        "subfactions": {"3rd Battalion, Royal Australian Regiment": "AirAssault"}
    },
    "Russian Airborne Forces": {
        "shortname": "VDV",
        "subfactions": {"150th Support Battalion": "Support"}
    },
    "Insurgent Forces": {
        "shortname": "MEI",
        "subfactions": {
            "Support Brigade": "Support", "Mechanized Brigade": "Mechanized",
            "Light Battalion": "LightInfantry", "Irregular Motorized Platoon": "Motorized",
            "Irregular Mechanized Platoon": "Mechanized", "Irregular Light Infantry": "LightInfantry",
            "Irregular Fire Support Group": "Support", "Irregular Battle Group": "CombinedArms",
            "Irregular Armored Squadron": "Armored", "Hoplite Battalion": "LightInfantry",
            "Combined Arms Brigade": "CombinedArms", "Armor Battalion": "Armored",
        }
    },
}

POSSIBLE_FTYPES = {
    'CombinedArms', 'LightInfantry', 'AirAssault', 'Armored', 'Support',
    'Motorized', 'Mechanized', 'AmphibiousAssault', 'FSTemplate', 'LobbyFactionSetup'
}


def get_faction_type(layer_name, faction, faction_name, map_and_disp_to_type):
    fd = FACTION_TYPES.get(faction_name)
    if fd and faction in fd["subfactions"]:
        return f'{fd["shortname"]}+{fd["subfactions"][faction]}'
    return map_and_disp_to_type[layer_name][faction]


def step1_update_factions(db_path, faction_info_path, translation_table_path):
    print("\n" + "=" * 60)
    print("STEP 1: Update Faction Entries")
    print("=" * 60)

    ensure_faction_type_columns(db_path)

    translation_table = {}
    if os.path.exists(translation_table_path):
        with open(translation_table_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                translation_table[row["OLD"]] = row["NEW"]
        print(f"  Loaded {len(translation_table)} translations")
    else:
        print(f"  Warning: translation table not found at {translation_table_path}")

    if not os.path.exists(faction_info_path):
        print(f"  ERROR: {faction_info_path} not found — skipping faction update")
        return

    with open(faction_info_path) as f:
        faction_data = json.load(f)

    map_and_disp_to_type = {}
    for k, faction in faction_data['factionSetups'].items():
        ftype = next((x for x in POSSIBLE_FTYPES if x in k), None)
        disp = faction['displayName']
        shortname = faction['shortName']
        if not ftype:
            continue
        for m in faction['mapLayers']:
            m = m.replace("Fool's Road", "FoolsRoad").replace("Goose Bay", "GooseBay")
            m = m.replace("Tallil Outskirts", "Tallil").replace("Sumari Bala", "Sumari")
            m = m.replace("Al Basrah", "AlBasrah").replace("Black Coast", "BlackCoast")
            m = m.replace("Kohat Toi", "Kohat")
            if m not in map_and_disp_to_type:
                map_and_disp_to_type[m] = {}
            map_and_disp_to_type[m][disp] = shortname + "+" + ftype

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if translation_table:
        for old, new in translation_table.items():
            cursor.execute("""
                UPDATE matches
                SET
                    winningTeam = CASE WHEN winningTeam = ? THEN ? ELSE winningTeam END,
                    losingTeam = CASE WHEN losingTeam = ? THEN ? ELSE losingTeam END,
                    winningSubfaction = CASE WHEN winningSubfaction = ? THEN ? ELSE winningSubfaction END,
                    losingSubfaction = CASE WHEN losingSubfaction = ? THEN ? ELSE losingSubfaction END
                WHERE winningTeam = ? OR losingTeam = ?
                   OR winningSubfaction = ? OR losingSubfaction = ?
            """, (old, new, old, new, old, new, old, new, old, old, old, old))
        conn.commit()
        print(f"  Applied translations")

    cursor.execute("""
        SELECT matchId, winningTeamID, winningSubfaction, losingSubfaction,
               layerName, winningTeam, losingTeam
        FROM matches
    """)
    rows = cursor.fetchall()

    cnt = 0
    total = 0
    for match_id, win_id, f1, f2, layer_name, F1, F2 in rows:
        total += 2
        if F1 == "Middle Eastern Alliance" or F2 == "Middle Eastern Alliance":
            continue

        if win_id == 1:
            faction1, faction2 = f1, f2
            faction1_name, faction2_name = F1, F2
        else:
            faction1, faction2 = f2, f1
            faction1_name, faction2_name = F2, F1

        f1t = f2t = None
        try:
            f1t = get_faction_type(layer_name, faction1, faction1_name, map_and_disp_to_type)
            cnt += 1
        except:
            pass
        try:
            f2t = get_faction_type(layer_name, faction2, faction2_name, map_and_disp_to_type)
            cnt += 1
        except:
            pass

        cursor.execute("""
            UPDATE matches SET faction1_type = ?, faction2_type = ? WHERE matchId = ?
        """, (f1t, f2t, match_id))

    conn.commit()
    conn.close()
    print(f"  Updated {cnt}/{total} faction types")


def init_rating_tables(db_path):
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS player_ratings (
            player_id TEXT PRIMARY KEY,
            games_played INTEGER,
            general_skill REAL,
            sigma REAL DEFAULT 2.0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS player_role_ratings (
            player_id TEXT,
            role TEXT,
            role_skill REAL,
            games_in_role INTEGER,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (player_id, role)
        );
        CREATE TABLE IF NOT EXISTS trained_matches (
            matchId INTEGER PRIMARY KEY,
            tag TEXT DEFAULT 'default',
            trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()


def log_trained_matches(match_ids, db_path, tag='default'):
    if not match_ids:
        return
    conn = sqlite3.connect(db_path)
    rows = [(str(m), tag) for m in match_ids]
    conn.executemany(
        "INSERT OR REPLACE INTO trained_matches (matchId, tag, trained_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
        rows
    )
    conn.commit()
    conn.close()
    print(f"  Logged {len(rows):,} matches as trained (tag='{tag}')")


def load_and_build_tensors(db_path):
    conn = sqlite3.connect(db_path)

    print("  Finding seed layers...")
    rows = conn.execute("SELECT DISTINCT layerName FROM matches").fetchall()
    seed_names = [r[0] for r in rows if is_seed_layer(r[0])]
    placeholders = ",".join("?" * len(seed_names)) if seed_names else ""
    seed_clause = f"AND m.layerName NOT IN ({placeholders})" if seed_names else ""

    print("  Filtering matches in SQL...")
    matches_df = pd.read_sql_query(f"""
        SELECT m.matchId, m.duration, m.winningTeamID
        FROM matches m
        JOIN (
            SELECT matchId, MIN(cnt) AS min_size
            FROM (SELECT matchId, teamID, COUNT(*) AS cnt FROM teams GROUP BY matchId, teamID) s
            GROUP BY matchId
        ) t ON t.matchId = m.matchId
        WHERE m.teamdataCollected = 1 AND t.min_size >= ? {seed_clause}
    """, conn, params=[MIN_PLAYERS_PER_SIDE, *seed_names])

    if matches_df.empty:
        conn.close()
        return None

    valid_ids = matches_df['matchId'].tolist()
    print(f"  Filtered to {len(valid_ids):,} valid matches")

    unique_matches = matches_df['matchId'].to_numpy()
    label_arr = (matches_df['winningTeamID'].to_numpy() == 1).astype(np.float32)
    duration_arr = matches_df['duration'].fillna(60).clip(lower=20, upper=DURATION_CAP).to_numpy().astype(np.float32)

    if USE_TIME_WEIGHT:
        ids = unique_matches.astype(np.float64)
        lo, hi = ids.min(), ids.max()
        rng = max(hi - lo, 1.0)
        age = (hi - ids) / rng
        time_weight_arr = np.exp(-TIME_DECAY_LAMBDA * age).astype(np.float32)
    else:
        time_weight_arr = np.ones_like(label_arr)

    conn.execute("CREATE TEMP TABLE _valid_matches (matchId INTEGER PRIMARY KEY)")
    conn.executemany("INSERT INTO _valid_matches VALUES (?)", [(int(x),) for x in valid_ids])

    print("  Loading distinct roles...")
    roles_df = pd.read_sql_query("SELECT DISTINCT role FROM teams WHERE role IS NOT NULL", conn)
    unique_roles = sorted(list(set([parse_role(r) for r in roles_df['role'].tolist()] + ['Unknown'])))
    role_to_idx = {r: i for i, r in enumerate(unique_roles)}

    try:
        aliases_df = pd.read_sql_query("SELECT alias, canonical_id FROM player_aliases", conn)
        id_map = pd.Series(aliases_df['canonical_id'].values, index=aliases_df['alias'].astype(str)).to_dict()
    except:
        id_map = {}

    print("  Streaming teams table in chunks...")
    chunksize = 500_000
    query = """
        SELECT t.matchId, t.teamID, t.steamID, t.eosID, t.mssID,
               t.canonical_id, t.role
        FROM teams t JOIN _valid_matches v ON v.matchId = t.matchId
        ORDER BY t.matchId
    """

    player_to_idx = {}
    players_arrs, roles_arrs, teams_arrs, match_id_arrs = [], [], [], []

    for chunk in pd.read_sql_query(query, conn, chunksize=chunksize):
        chunk['player_id'] = chunk['canonical_id']
        missing_mask = chunk['player_id'].isna()
        if missing_mask.any():
            for col in ['steamID', 'eosID', 'mssID']:
                still_missing = chunk['player_id'].isna()
                if not still_missing.any():
                    break
                mask = still_missing & chunk[col].notna()
                if not mask.any():
                    continue
                mapped = chunk.loc[mask, col].astype(str).map(id_map)
                found = mapped.notna()
                chunk.loc[mask & found, 'player_id'] = mapped[found]
            still_missing = chunk['player_id'].isna()
            if still_missing.any():
                valid_fb = still_missing & chunk['steamID'].notna()
                chunk.loc[valid_fb, 'player_id'] = chunk.loc[valid_fb, 'steamID'].astype(str)

        chunk = chunk.dropna(subset=['player_id']).copy()
        if len(chunk) == 0:
            continue

        chunk['player_id'] = chunk['player_id'].astype(str)
        chunk['parsed_role'] = chunk['role'].apply(parse_role)
        chunk['p_idx'] = chunk['player_id'].map(player_to_idx)
        new_mask = chunk['p_idx'].isna()
        if new_mask.any():
            new_pids = chunk.loc[new_mask, 'player_id'].unique()
            new_indices = range(len(player_to_idx), len(player_to_idx) + len(new_pids))
            player_to_idx.update(zip(new_pids, new_indices))
            chunk['p_idx'] = chunk['player_id'].map(player_to_idx).astype(np.int32)
        else:
            chunk['p_idx'] = chunk['p_idx'].astype(np.int32)

        chunk['r_idx'] = chunk['parsed_role'].map(role_to_idx).fillna(role_to_idx['Unknown']).astype(np.int16)
        players_arrs.append(chunk['p_idx'].to_numpy(dtype=np.int32))
        roles_arrs.append(chunk['r_idx'].to_numpy(dtype=np.int16))
        teams_arrs.append(chunk['teamID'].to_numpy(dtype=np.int8))
        match_id_arrs.append(chunk['matchId'].to_numpy(dtype=np.int64))

    conn.execute("DROP TABLE _valid_matches")
    conn.close()

    players = np.concatenate(players_arrs)
    roles = np.concatenate(roles_arrs)
    teams = np.concatenate(teams_arrs)
    match_ids_all = np.concatenate(match_id_arrs)
    del players_arrs, roles_arrs, teams_arrs, match_id_arrs, chunk, matches_df
    gc.collect()

    print("  Building player-role pairs...")
    pr_view = np.stack([players, roles], axis=1).astype(np.int32)
    unique_pr, inverse_indices = np.unique(pr_view, axis=0, return_inverse=True)
    pr_indices = inverse_indices.astype(np.int32)
    pr_to_p_idx = unique_pr[:, 0]
    pr_to_r_idx = unique_pr[:, 1]
    player_role_to_idx = {(int(p), int(r)): i for i, (p, r) in enumerate(unique_pr)}

    unique_matches, counts = np.unique(match_ids_all, return_counts=True)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    flat = {
        'players': players, 'roles': roles, 'teams': teams,
        'pr_indices': pr_indices, 'offsets': offsets, 'labels': label_arr,
        'durations': duration_arr, 'time_weights': time_weight_arr,
        'match_ids': unique_matches, 'pr_to_p_idx': pr_to_p_idx, 'pr_to_r_idx': pr_to_r_idx,
    }

    print(f"  Dataset: {len(unique_matches):,} matches, {len(players):,} player-records")
    print(f"  Unique players: {len(player_to_idx):,}, player-role pairs: {len(unique_pr):,}")

    return {
        'flat': flat, 'player_to_idx': player_to_idx, 'role_to_idx': role_to_idx,
        'player_role_to_idx': player_role_to_idx, 'num_players': len(player_to_idx),
        'num_roles': len(unique_roles), 'unique_players': list(player_to_idx.keys()),
        'unique_roles': unique_roles,
    }


class SquadRatingModel(nn.Module):
    def __init__(self, num_players, num_pr_pairs, player_role_to_idx, player_to_idx, role_to_idx):
        super().__init__()
        self.player_to_idx = player_to_idx
        self.role_to_idx = role_to_idx
        self.player_role_to_idx = player_role_to_idx
        self.general_skill = nn.Parameter(torch.zeros(num_players))
        self.role_skill = nn.Parameter(torch.zeros(num_pr_pairs))

    def forward(self, flat, batch_idx):
        device = self.general_skill.device
        
        if isinstance(batch_idx, np.ndarray):
            batch_idx_np = batch_idx
        else:
            batch_idx_np = batch_idx.cpu().numpy()
            
        starts = flat['offsets'][batch_idx_np]
        ends = flat['offsets'][batch_idx_np + 1]

        logits = []
        for i in range(len(batch_idx_np)):
            s, e = starts[i], ends[i]
            players = torch.from_numpy(flat['players'][s:e]).long().to(device)
            pr_indices = torch.from_numpy(flat['pr_indices'][s:e]).long().to(device)
            teams = torch.from_numpy(flat['teams'][s:e]).long().to(device)

            contrib = self.general_skill[players] + self.role_skill[pr_indices]
            signs = (teams == 1).float() - (teams == 2).float()
            logits.append((contrib * signs).sum())

        return torch.stack(logits)

    def regularization(self):
        return (REG_GENERAL * (self.general_skill ** 2).sum() +
                REG_ROLE * (self.role_skill ** 2).sum())

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)


def train_model(model, flat, train_idx, val_idx, epochs=EPOCHS,
                batch_size=BATCH_SIZE, lr=LEARNING_RATE):
    import time
    device = next(model.parameters()).device
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    early_stop = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)

    val_labels = torch.from_numpy(flat['labels'][val_idx]).to(device)
    if USE_DURATION_WEIGHT:
        val_weights = torch.from_numpy(flat['durations'][val_idx]).to(device) / DURATION_WEIGHT_SCALE
        val_weights = val_weights.clamp(max=DURATION_CAP / DURATION_WEIGHT_SCALE)
    else:
        val_weights = torch.ones(len(val_idx), device=device)
    if USE_TIME_WEIGHT:
        val_weights = val_weights * torch.from_numpy(flat['time_weights'][val_idx]).to(device)

    n_train = len(train_idx)
    total_start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        indices = np.random.permutation(train_idx)
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_train, batch_size):
            batch_idx = indices[i:i+batch_size]
            labels = torch.from_numpy(flat['labels'][batch_idx]).to(device)
            logits = model(flat, batch_idx)
            if USE_DURATION_WEIGHT:
                durations = torch.from_numpy(flat['durations'][batch_idx]).to(device)
                weights = (durations / DURATION_WEIGHT_SCALE).clamp(max=DURATION_CAP / DURATION_WEIGHT_SCALE)
            else:
                weights = torch.ones(len(batch_idx), device=device)
            if USE_TIME_WEIGHT:
                weights = weights * torch.from_numpy(flat['time_weights'][batch_idx]).to(device)
            loss = (criterion(logits, labels) * weights).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                model.general_skill.data -= model.general_skill.data.mean()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches
        
        model.eval()
        with torch.no_grad():
            val_logits = model(flat, val_idx)
            val_losses = criterion(val_logits, val_labels)
            val_loss = (val_losses * val_weights).mean().item()
            val_acc = ((val_logits > 0).float() == val_labels).float().mean().item()
            
        scheduler.step(val_loss)
        stopped = early_stop(val_loss, model)
        
        elapsed_time = time.time() - total_start_time
        avg_time_per_epoch = elapsed_time / (epoch + 1)
        remaining_epochs = epochs - (epoch + 1)
        eta_seconds = avg_time_per_epoch * remaining_epochs
        eta_mins = int(eta_seconds // 60)
        eta_secs = int(eta_seconds % 60)
        
        print(f"  Epoch {epoch+1:3d}/{epochs} | train={avg_train_loss:.4f} | val={val_loss:.4f} | acc={val_acc:.3f} | ETA: {eta_mins}m {eta_secs:02d}s")
        
        if stopped:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    early_stop.restore_best(model)
    return model

def evaluate_model(model, flat, idx):
    device = next(model.parameters()).device
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    labels = torch.from_numpy(flat['labels'][idx]).to(device)
    if USE_DURATION_WEIGHT:
        weights = torch.from_numpy(flat['durations'][idx]).to(device) / DURATION_WEIGHT_SCALE
        weights = weights.clamp(max=DURATION_CAP / DURATION_WEIGHT_SCALE)
    else:
        weights = torch.ones(len(idx), device=device)
    if USE_TIME_WEIGHT:
        weights = weights * torch.from_numpy(flat['time_weights'][idx]).to(device)
    with torch.no_grad():
        logits = model(flat, idx)
        loss = (criterion(logits, labels) * weights).mean().item()
        probs = torch.sigmoid(logits)
        acc = ((logits > 0).float() == labels).float().mean().item()
        brier = ((probs - labels) ** 2).mean().item()
    return {'loss': loss, 'accuracy': acc, 'brier': brier}


def export_ratings_to_db(model, model_data, db_path, min_games=MIN_GAMES_FOR_RATING, player_sigma=None):
    player_to_idx = model_data['player_to_idx']
    role_to_idx = model_data['role_to_idx']
    idx_to_role = {v: k for k, v in role_to_idx.items()}
    idx_to_player = {v: k for k, v in player_to_idx.items()}
    flat = model_data['flat']

    player_games = np.bincount(flat['players'], minlength=model_data['num_players'])
    pr_games = np.bincount(flat['pr_indices'], minlength=len(flat['pr_to_p_idx']))

    core_rows, role_rows = [], []
    for player_id, p_idx in player_to_idx.items():
        games = player_games[p_idx]
        if games < min_games:
            continue
        general = model.general_skill[p_idx].item()
        sigma_val = float(round(player_sigma[p_idx], 4)) if player_sigma is not None else 2.0
        core_rows.append((player_id, int(games), float(round(general, 6)), sigma_val))

    for pr_idx, games in enumerate(pr_games):
        if games == 0:
            continue
        p_idx = flat['pr_to_p_idx'][pr_idx]
        if player_games[p_idx] < min_games:
            continue
        r_idx = flat['pr_to_r_idx'][pr_idx]
        role_name = idx_to_role[r_idx]
        rs = model.role_skill[pr_idx].item()
        role_rows.append((idx_to_player[p_idx], role_name, float(round(rs, 6)), int(games)))

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("BEGIN TRANSACTION")
    try:
        cursor.execute("DROP TABLE IF EXISTS player_ratings")
        cursor.execute("DROP TABLE IF EXISTS player_role_ratings")
        cursor.execute("""CREATE TABLE player_ratings (
            player_id TEXT PRIMARY KEY, games_played INTEGER, general_skill REAL,
            sigma REAL DEFAULT 2.0, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        cursor.execute("""CREATE TABLE player_role_ratings (
            player_id TEXT, role TEXT, role_skill REAL, games_in_role INTEGER,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (player_id, role))""")
        
        cursor.executemany("INSERT INTO player_ratings (player_id, games_played, general_skill, sigma) VALUES (?, ?, ?, ?)",
                           core_rows)
        cursor.executemany("INSERT INTO player_role_ratings (player_id, role, role_skill, games_in_role) VALUES (?, ?, ?, ?)",
                           role_rows)
        
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()
    print(f"  Exported {len(core_rows):,} players, {len(role_rows):,} player-roles")


def step2_train_ratings(db_path):
    print("\n" + "=" * 60)
    print("STEP 2: Train Rating System")
    print("=" * 60)

    init_rating_tables(db_path)
    device = torch.device("cpu")
    print(f"  Device: {device}")
    print(f"  Duration weighting: {'ON' if USE_DURATION_WEIGHT else 'OFF'}")
    print(f"  Time weighting: {'ON' if USE_TIME_WEIGHT else 'OFF'} (lambda={TIME_DECAY_LAMBDA})")

    model_data = load_and_build_tensors(db_path)
    if model_data is None or len(model_data['flat']['match_ids']) == 0:
        print("  No new matches found. Skipping.")
        return

    flat = model_data['flat']
    n = len(flat['match_ids'])
    n_test = max(1, int(n * TEST_FRAC))
    n_val = max(1, int(n * VAL_FRAC))
    perm = np.random.permutation(n)
    test_idx = perm[:n_test]
    val_idx = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]
    print(f"  Split: {len(train_idx):,} train | {len(val_idx):,} val | {len(test_idx):,} test")

    model = SquadRatingModel(
        num_players=model_data['num_players'],
        num_pr_pairs=len(model_data['player_role_to_idx']),
        player_role_to_idx=model_data['player_role_to_idx'],
        player_to_idx=model_data['player_to_idx'],
        role_to_idx=model_data['role_to_idx'],
    ).to(device)

    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    model = train_model(model, flat, train_idx, val_idx)

    print("\n  --- Final Evaluation ---")
    test_metrics = evaluate_model(model, flat, test_idx)
    print(f"  Test — loss: {test_metrics['loss']:.4f}, acc: {test_metrics['accuracy']:.3f}, brier: {test_metrics['brier']:.4f}")


    player_games = np.bincount(flat['players'], minlength=model_data['num_players'])
    player_sigma = np.zeros(model_data['num_players'], dtype=np.float32)
    skill_std = float(torch.std(model.general_skill).cpu())
    sigma_floor = skill_std * 0.2
    valid_mask = player_games > 0
    player_sigma[valid_mask] = np.maximum(sigma_floor, skill_std / np.sqrt(player_games[valid_mask]))
    print(f"  Sigma: mean={player_sigma.mean():.3f}, median={np.median(player_sigma):.3f}")

    export_ratings_to_db(model, model_data, db_path, player_sigma=player_sigma)
    log_trained_matches(flat['match_ids'].tolist(), db_path, tag='batch')



def load_effects_data(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(matches)")
    columns = [col[1] for col in cursor.fetchall()]
    has_faction_types = 'faction1_type' in columns and 'faction2_type' in columns

    select_cols = """matchId, gamemode, layerName, winningTeamID, winningTeam,
                      winningSubfaction, losingTeam, losingSubfaction,
                      winningTickets, losingTickets, duration"""
    if has_faction_types:
        select_cols += ", faction1_type, faction2_type"

    matches_df = pd.read_sql_query(
        f"SELECT {select_cols} FROM matches WHERE teamdataCollected = 1 ORDER BY matchId ASC", conn)
    parts = matches_df['layerName'].str.split()
    matches_df['map'] = parts.str[0]
    matches_df['game_mode_full'] = parts.str[-2:].str.join(' ')

    teams_df = pd.read_sql_query(
        "SELECT matchId, teamID, canonical_id, role FROM teams WHERE canonical_id IS NOT NULL", conn)
    try:
        ratings_df = pd.read_sql_query("SELECT player_id, general_skill, sigma FROM player_ratings", conn)
    except:
        ratings_df = pd.DataFrame(columns=['player_id', 'general_skill', 'sigma'])
    try:
        role_ratings_df = pd.read_sql_query("SELECT player_id, role, role_skill FROM player_role_ratings", conn)
    except:
        role_ratings_df = pd.DataFrame(columns=['player_id', 'role', 'role_skill'])
    conn.close()

    print(f"  Loaded {len(matches_df):,} matches, {len(teams_df):,} team records")
    print(f"  Loaded {len(ratings_df):,} player ratings, {len(role_ratings_df):,} role ratings")
    return matches_df, teams_df, ratings_df, role_ratings_df


def compute_skill_predictions(matches_df, teams_df, ratings_df, role_ratings_df):
    if len(ratings_df) == 0:
        print("  Warning: No player ratings. Using neutral predictions.")
        matches_df['skill_pred'] = 0.5
        matches_df['team1_win'] = (matches_df['winningTeamID'] == 1).astype(int)
        matches_df['residual'] = matches_df['team1_win'] - matches_df['skill_pred']
        return matches_df

    print("  Building skill lookup dictionaries...")
    skill_dict = ratings_df.set_index('player_id')['general_skill'].to_dict()
    sigma_dict = ratings_df.set_index('player_id')['sigma'].to_dict()

    if len(role_ratings_df) > 0:
        role_ratings_df['key'] = role_ratings_df['player_id'] + '|' + role_ratings_df['role']
        role_skill_dict = role_ratings_df.set_index('key')['role_skill'].to_dict()
    else:
        role_skill_dict = {}

    print("  Aggregating team strengths...")
    agg_dict = {}
    for row in teams_df[['matchId', 'teamID', 'canonical_id', 'role']].itertuples(index=False, name=None):
        match_id, team_id, canonical_id, role_str = row
        p_skill = skill_dict.get(canonical_id, 0.0)
        p_sigma = sigma_dict.get(canonical_id, 2.0)
        role = parse_role(role_str)
        r_skill = role_skill_dict.get(f"{canonical_id}|{role}", 0.0)
        total_skill = p_skill + r_skill
        key = (match_id, team_id)
        if key in agg_dict:
            agg_dict[key][0] += total_skill
            agg_dict[key][1] += p_sigma
            agg_dict[key][2] += 1
        else:
            agg_dict[key] = [total_skill, p_sigma, 1]

    del teams_df
    gc.collect()

    agg_rows = [{'matchId': mid, 'teamID': tid, 'team_skill_sum': s, 'team_sigma_mean': sig/c, 'num_players': c}
                for (mid, tid), (s, sig, c) in agg_dict.items()]
    del agg_dict
    gc.collect()

    team_strength = pd.DataFrame(agg_rows)
    t1 = team_strength[team_strength['teamID'] == 1][['matchId', 'team_skill_sum', 'team_sigma_mean']].rename(
        columns={'team_skill_sum': 'team1_skill', 'team_sigma_mean': 'team1_sigma'})
    t2 = team_strength[team_strength['teamID'] == 2][['matchId', 'team_skill_sum', 'team_sigma_mean']].rename(
        columns={'team_skill_sum': 'team2_skill', 'team_sigma_mean': 'team2_sigma'})
    matches_df = matches_df.merge(t1, on='matchId', how='left').merge(t2, on='matchId', how='left')
    matches_df['team1_skill'] = matches_df['team1_skill'].fillna(0.0)
    matches_df['team2_skill'] = matches_df['team2_skill'].fillna(0.0)
    matches_df['team1_sigma'] = matches_df['team1_sigma'].fillna(2.0)
    matches_df['team2_sigma'] = matches_df['team2_sigma'].fillna(2.0)
    matches_df['skill_diff'] = matches_df['team1_skill'] - matches_df['team2_skill']
    matches_df['skill_pred'] = 1 / (1 + np.exp(-matches_df['skill_diff']))
    matches_df['skill_uncertainty'] = np.sqrt(matches_df['team1_sigma']**2 + matches_df['team2_sigma']**2)
    matches_df['team1_win'] = (matches_df['winningTeamID'] == 1).astype(int)
    matches_df['residual'] = matches_df['team1_win'] - matches_df['skill_pred']

    print(f"  Mean skill_pred: {matches_df['skill_pred'].mean():.3f}")
    print(f"  Residual std: {matches_df['residual'].std():.3f}")
    return matches_df


def _parse_faction_type(faction_type):
    if pd.isna(faction_type) or str(faction_type).strip() == "":
        return "Unknown", "Unknown", ""
    parts = str(faction_type).split('+')
    if len(parts) == 2:
        return parts[0].strip(), faction_type.strip(), parts[1].strip()
    return faction_type.strip(), faction_type.strip(), ""


def prepare_features(matches_df):
    if 'faction1_type' in matches_df.columns:
        matches_df['team1_faction'], matches_df['team1_subfaction'], matches_df['team1_tag'] = \
            zip(*matches_df['faction1_type'].apply(_parse_faction_type))
        matches_df['team2_faction'], matches_df['team2_subfaction'], matches_df['team2_tag'] = \
            zip(*matches_df['faction2_type'].apply(_parse_faction_type))
    else:
        matches_df['team1_faction'] = np.where(matches_df['winningTeamID'] == 1,
                                                matches_df['winningTeam'], matches_df['losingTeam'])
        matches_df['team2_faction'] = np.where(matches_df['winningTeamID'] == 1,
                                                matches_df['losingTeam'], matches_df['winningTeam'])
        matches_df['team1_subfaction'] = np.where(matches_df['winningTeamID'] == 1,
                                                   matches_df['winningSubfaction'], matches_df['losingSubfaction'])
        matches_df['team2_subfaction'] = np.where(matches_df['winningTeamID'] == 1,
                                                   matches_df['losingSubfaction'], matches_df['winningSubfaction'])
        matches_df['team1_tag'] = ""
        matches_df['team2_tag'] = ""

    matches_df['matchup'] = matches_df['team1_subfaction'] + ' vs ' + matches_df['team2_subfaction']

    def extract_version(layer_name):
        if pd.isna(layer_name):
            return 'v1'
        for part in reversed(str(layer_name).split()):
            if part.startswith('v') and part[1:].isdigit():
                return part
        return 'v1'
    matches_df['layer_version'] = matches_df['layerName'].apply(extract_version)

    matches_df['duration'] = matches_df['duration'].fillna(60).clip(lower=10, upper=CB_DURATION_CAP)
    if CB_USE_TIME_WEIGHT and len(matches_df) > 1:
        match_ids = matches_df['matchId'].astype(float).values
        min_id, max_id = float(match_ids.min()), float(match_ids.max())
        id_range = max_id - min_id if max_id > min_id else 1.0
        age_norm = (max_id - match_ids) / id_range
        matches_df['time_weight'] = np.exp(-CB_TIME_DECAY_LAMBDA * age_norm)
    else:
        matches_df['time_weight'] = 1.0
    matches_df['duration_weight'] = (matches_df['duration'] / CB_DURATION_WEIGHT_SCALE).clip(
        upper=CB_DURATION_CAP / CB_DURATION_WEIGHT_SCALE)
    matches_df['sample_weight'] = matches_df['duration_weight'] * matches_df['time_weight']
    return matches_df


def get_feature_columns():
    features = [
        'gamemode', 'map', 'game_mode_full', 'layerName', 'layer_version',
        'team1_faction', 'team2_faction', 'team1_subfaction', 'team2_subfaction',
        'team1_tag', 'team2_tag', 'matchup', 'duration', 'skill_pred', 'skill_uncertainty',
    ]
    cat_features = [
        'gamemode', 'map', 'game_mode_full', 'layerName', 'layer_version',
        'team1_faction', 'team2_faction', 'team1_subfaction', 'team2_subfaction',
        'team1_tag', 'team2_tag', 'matchup'
    ]
    return features, cat_features


def train_catboost_model(matches_df, existing_model_path=None):
    features, cat_features = get_feature_columns()
    X = matches_df[features].copy()
    y = matches_df['residual'].values
    weights = matches_df['sample_weight'].values

    n = len(matches_df)
    indices = np.random.RandomState(CB_RANDOM_SEED).permutation(n)
    n_test = max(1, int(n * CB_TEST_FRAC))
    n_val = max(1, int(n * CB_VAL_FRAC))
    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    train_pool = Pool(X.iloc[train_idx], y[train_idx], cat_features=cat_features, weight=weights[train_idx])
    val_pool = Pool(X.iloc[val_idx], y[val_idx], cat_features=cat_features, weight=weights[val_idx])
    test_pool = Pool(X.iloc[test_idx], y[test_idx], cat_features=cat_features, weight=weights[test_idx])

    print(f"  Split: {len(train_idx):,} train | {len(val_idx):,} val | {len(test_idx):,} test")

    feature_weight_dict = {f: 1.0 for f in features}
    for f in ['team1_subfaction', 'team2_subfaction', 'matchup']:
        if f in feature_weight_dict:
            feature_weight_dict[f] = 0.7
    feature_weights_list = [feature_weight_dict[f] for f in features]

    model = CatBoostRegressor(
        iterations=CB_ITERATIONS, learning_rate=CB_LEARNING_RATE, depth=CB_DEPTH,
        l2_leaf_reg=CB_L2_LEAF_REG, loss_function='RMSE', random_seed=CB_RANDOM_SEED,
        verbose=100, early_stopping_rounds=CB_EARLY_STOPPING_ROUNDS,
        bagging_temperature=CB_BAGGING_TEMPERATURE, cat_features=cat_features,
        feature_border_type='GreedyLogSum', leaf_estimation_method='Newton',
        leaf_estimation_iterations=10, feature_weights=feature_weights_list
    )

    if existing_model_path and os.path.exists(existing_model_path):
        print(f"  Loading existing model from {existing_model_path}")
        model.load_model(existing_model_path)
        model.set_params(learning_rate=CB_LEARNING_RATE * 0.3)

    print(f"  Training CatBoost...")
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    print("\n  --- Evaluation ---")
    from sklearn.metrics import r2_score
    for name, pool, y_true, w in [("Train", train_pool, y[train_idx], weights[train_idx]),
                                   ("Val", val_pool, y[val_idx], weights[val_idx]),
                                   ("Test", test_pool, y[test_idx], weights[test_idx])]:
        pred = model.predict(pool)
        rmse = np.sqrt(np.average((y_true - pred)**2, weights=w))
        r2 = r2_score(y_true, pred, sample_weight=w)
        print(f"  {name:5s} RMSE: {rmse:.4f}  R²: {r2:.4f}")

    importance = model.get_feature_importance()
    feat_imp = pd.DataFrame({'feature': features, 'importance': importance}).sort_values('importance', ascending=False)
    print("\n  Feature Importance:")
    for _, row in feat_imp.iterrows():
        print(f"    {row['feature']:25s} {row['importance']:6.2f}%")

    return model, features, cat_features


def save_catboost_model(model, features, cat_features, path, meta_path):
    model.save_model(path)
    meta = {
        'features': features, 'cat_features': cat_features,
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'iterations': model.tree_count_,
        'learning_rate': CB_LEARNING_RATE, 'depth': CB_DEPTH
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved model to {path}, metadata to {meta_path}")


def export_layer_effects_to_db(model, matches_df, db_path, table_name=LAYER_EFFECTS_TABLE):
    combos = matches_df[[
        'gamemode', 'map', 'game_mode_full', 'layerName', 'layer_version',
        'team1_faction', 'team2_faction', 'team1_subfaction', 'team2_subfaction',
        'team1_tag', 'team2_tag', 'matchup'
    ]].drop_duplicates()

    features, _ = get_feature_columns()
    combos['skill_pred'] = 0.5
    combos['skill_uncertainty'] = 2.0
    combos['duration'] = 60
    combos['predicted_effect'] = model.predict(combos[features])

    combo_counts = matches_df.groupby(['layerName', 'team1_subfaction', 'team2_subfaction']).size().reset_index(name='sample_count')
    combos = combos.merge(combo_counts, on=['layerName', 'team1_subfaction', 'team2_subfaction'], how='left')
    combos['sample_count'] = combos['sample_count'].fillna(0).astype(int)

    actual_wr = matches_df.groupby(['layerName', 'team1_subfaction', 'team2_subfaction']).agg(
        actual_team1_winrate=('team1_win', 'mean'), actual_count=('team1_win', 'count')).reset_index()
    combos = combos.merge(actual_wr, on=['layerName', 'team1_subfaction', 'team2_subfaction'], how='left')

    combos['rotation_format'] = (
        combos['layerName'].str.replace(' ', '_') + ' ' +
        combos['team1_subfaction'].astype(str) + ' ' +
        combos['team2_subfaction'].astype(str)
    )

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    cursor.execute(f"""CREATE TABLE {table_name} (
        id INTEGER PRIMARY KEY AUTOINCREMENT, gamemode TEXT, map TEXT, game_mode_full TEXT,
        layerName TEXT, layer_version TEXT, team1_faction TEXT, team2_faction TEXT,
        team1_subfaction TEXT, team2_subfaction TEXT, team1_tag TEXT, team2_tag TEXT,
        matchup TEXT, predicted_effect REAL, actual_team1_winrate REAL, sample_count INTEGER,
        rotation_format TEXT, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")

    for _, row in combos.iterrows():
        cursor.execute(f"""INSERT INTO {table_name}
            (gamemode, map, game_mode_full, layerName, layer_version, team1_faction, team2_faction,
             team1_subfaction, team2_subfaction, team1_tag, team2_tag, matchup,
             predicted_effect, actual_team1_winrate, sample_count, rotation_format)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
            row['gamemode'], row['map'], row['game_mode_full'], row['layerName'], row['layer_version'],
            row['team1_faction'], row['team2_faction'], row['team1_subfaction'], row['team2_subfaction'],
            row['team1_tag'], row['team2_tag'], row['matchup'],
            float(row['predicted_effect']),
            float(row['actual_team1_winrate']) if pd.notna(row['actual_team1_winrate']) else None,
            int(row['sample_count']), row['rotation_format']))

    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_layer ON {table_name} (layerName, team1_subfaction, team2_subfaction)")
    conn.commit()
    conn.close()
    print(f"  Exported {len(combos):,} layer effect estimates to '{table_name}'")


def step3_train_effects(db_path, update_existing=False):
    print("\n" + "=" * 60)
    print("STEP 3: Train Layer Effect Model (CatBoost)")
    print("=" * 60)

    matches_df, teams_df, ratings_df, role_ratings_df = load_effects_data(db_path)
    if len(matches_df) == 0:
        print("  No matches found. Skipping.")
        return

    print("\n  Computing skill predictions...")
    matches_df = compute_skill_predictions(matches_df, teams_df, ratings_df, role_ratings_df)

    print("\n  Preparing features...")
    matches_df = prepare_features(matches_df)

    print("\n  Training CatBoost...")
    existing_path = CB_MODEL_PATH if update_existing else None
    model, features, cat_features = train_catboost_model(matches_df, existing_path)

    save_catboost_model(model, features, cat_features, CB_MODEL_PATH, CB_META_PATH)
    export_layer_effects_to_db(model, matches_df, db_path)
    log_trained_matches(matches_df['matchId'].tolist(), db_path, tag='catboost_layer')


def main():
    global DB_PATH

    parser = argparse.ArgumentParser(description='Full Training Pipeline')
    parser.add_argument('--db', type=str, default=DB_PATH)
    parser.add_argument('--faction-info', type=str, default=FACTION_INFO_PATH)
    parser.add_argument('--translation-table', type=str, default=TRANSLATION_TABLE_PATH)
    parser.add_argument('--skip-factions', action='store_true')
    parser.add_argument('--skip-ratings', action='store_true')
    parser.add_argument('--skip-effects', action='store_true')
    parser.add_argument('--update-effects', action='store_true', help='Continue training existing CatBoost model')
    args = parser.parse_args()

    DB_PATH = args.db

    print("=" * 60)
    print("FULL TRAINING PIPELINE")
    print("=" * 60)
    print(f"Database: {args.db}")
    print(f"Steps: {'Factions' if not args.skip_factions else 'SKIP'} → "
          f"{'Ratings' if not args.skip_ratings else 'SKIP'} → "
          f"{'Effects' if not args.skip_effects else 'SKIP'}")

    if not args.skip_factions:
        step1_update_factions(args.db, args.faction_info, args.translation_table)
    if not args.skip_ratings:
        step2_train_ratings(args.db)
    if not args.skip_effects:
        step3_train_effects(args.db, update_existing=args.update_effects)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()