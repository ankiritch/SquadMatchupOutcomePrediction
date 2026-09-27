import pandas as pd

DB_PATH = 'squadDB.sqlite'
MIN_PLAYERS_PER_SIDE = 35
MIN_GAMES_FOR_RATING = 1

# Training
EPOCHS = 35
BATCH_SIZE = 512
LEARNING_RATE = 0.005 # 0.01
EARLY_STOPPING_PATIENCE = 5
VAL_FRAC = 0.1
TEST_FRAC = 0.05

# Duration weighting
DURATION_CAP = 120.0           # Cut off anything above 2 hrs
DURATION_WEIGHT_SCALE = 60.0   # 60 min match = weight 1.0, 120 min = 2.0, 30 min = 0.5
USE_DURATION_WEIGHT = True

# Time weighting (by matchId: higher = more recent)
USE_TIME_WEIGHT = True
TIME_DECAY_LAMBDA = 0.7 #2        # e^-3 ≈ 0.05 for oldest match; 1.0 for newest
                               # Increase to decay faster, decrease for flatter weights

# Regularization
REG_GENERAL = 0.0 # 0.001 both
REG_ROLE = 0.0
WEIGHT_DECAY = 0.005

# Role merging
ROLE_MERGE_MAP = {
    'Sapper': 'Engineer',
    'Jandarma': 'Raider',
    'Pathfinder': 'Raider',
    'Infiltrator': 'Raider',
    'Saboteur': 'Raider',
    'Ambusher': 'Raider',
    'SLCrewman': 'Crewman',
    'SLPilot': 'Pilot',
    'SquadLeader': 'SL',
    'Officer': 'SL',
}

SEED_KEYWORDS = ['seed', 'seeding', 'seedlayer']

VALID_ROLES = {
    'Rifleman', 'SL', 'Medic', 'LAT', 'Crewman', 'Grenadier',
    'HAT', 'AR', 'Engineer', 'MachineGunner', 'Pilot', 'Raider',
    'Sniper', 'Scout', 'Recruit', 'Unarmed', 'Unknown', 'Marksman'
}

VALID_ROLES_SORTED = sorted(VALID_ROLES, key=len, reverse=True)
ROLE_MERGE_KEYS_SORTED = sorted(ROLE_MERGE_MAP.keys(), key=len, reverse=True)


def is_seed_layer(layer_name):
    if not layer_name:
        return False
    return any(kw in str(layer_name).lower() for kw in SEED_KEYWORDS)


def parse_role(role_str):
    if not role_str or pd.isna(role_str):
        return 'Unknown'
    raw = str(role_str).strip().rstrip(',').rstrip().strip('"').strip("'")
    parts = raw.split('_')
    if len(parts) == 5:
        role = parts[3]
    elif len(parts) == 4:
        role = parts[2]
    elif len(parts) in (3, 2):
        role = parts[1]
    elif len(parts) == 1:
        role = parts[0]
    else:
        role = ""
    role = ROLE_MERGE_MAP.get(role, role)
    if role in VALID_ROLES:
        return role
    for cand in VALID_ROLES_SORTED:
        if cand in raw:
            return cand
    for key in ROLE_MERGE_KEYS_SORTED:
        if key in raw:
            merged = ROLE_MERGE_MAP[key]
            if merged in VALID_ROLES:
                return merged
    return 'Unknown'



# =============================================================================
# CONFIGURATION
# =============================================================================

#DB_PATH = 'squadDB.sqlite'
FACTION_INFO_PATH = 'factionInfo.json'
TRANSLATION_TABLE_PATH = 'translation_table.csv'

# CatBoost model
CB_MODEL_PATH = 'layer_effect_model.cbm'
CB_META_PATH = 'layer_effect_model_meta.json'
LAYER_EFFECTS_TABLE = 'layer_effects'

CB_ITERATIONS = 2000
CB_LEARNING_RATE = 0.03
CB_DEPTH = 7
CB_L2_LEAF_REG = 3.0
CB_RANDOM_SEED = 42
CB_EARLY_STOPPING_ROUNDS = 150
CB_BAGGING_TEMPERATURE = 0.5

CB_TEST_FRAC = 0.10
CB_VAL_FRAC = 0.10

CB_DURATION_CAP = 120
CB_DURATION_WEIGHT_SCALE = 60
CB_USE_TIME_WEIGHT = True
CB_TIME_DECAY_LAMBDA = 0.3