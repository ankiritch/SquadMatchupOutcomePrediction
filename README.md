# Setup
## Notes
It's possible 'pip' should be replaced with 'pip3' or 'python' with 'python3' in the commands below

## Required
Install the necessary packages by running this in the project directory:
pip install -r requirements.txt

For data collection create the file 'basicMatchdataCollectorVariables' with the following contents:
```
oldest_checked_page:1
matchdata_request_count:0
checked_pages:1
```

## Optional
If you want the steam usernames to be printed correctly:
on linux set the steam api key like this:
export STEAM_API_KEY="your_steam_api_key_here"

you can create one here: https://steamcommunity.com/dev/apikey
as the domain you can use 'localhost', without the quotation marks.

# File Contents
- requirements.txt: required python packages
- squadDB.sqlite: the database containing the relevant data
- full_model-update.py: update the model that predicts layer & player effects
- predict.py: Can give data about a historical match or a rotation
- factionInfo.json: data about factions
- translation_table.csv: translation data, incomplete but still results in more useable data
- rating_system_SHARED.py: some config and reused values
- layer_effect_model.cbm: CatBoost layer effect model
- layer_effect_model_meta.json: CatBoost model metadata
- fetch_data.py: collect new match data
- basicMatchdataCollectorVariables: some variables for collecting match data from mysquadstats

# Usage:
    python fetch_data.py
    python fetch_data.py --loop                    # keeps collecting data by checking if new matches appeared regularly, usually the normal mode is just fine

    python predict.py --matchid 12345
    python predict.py --matchid 12345 --db squadDB.sqlite
    python predict.py --file rotations.txt

    python full_model_update.py
    python full_model_update.py --skip-factions
    python full_model_update.py --skip-ratings     # this step takes the longest (60m ETA is normal, due to early stopping it can finish within 20-40m)
    python full_model_update.py --skip-effects
    python full_model_update.py --update-effects   # continue training existing CatBoost model (layer effects)
    python full_model_update.py --db custom.db


# The database
If you want to look through the database you can use something like sqlitebrowser to open "squadDB.sqlite"
You can also query the database using sqlite3

The matchid is the same one as on mysquadstats. So if you're looking for a specific server/match you can look there for the ID first

Example queries:
    - Most balanced matchups on a certain layer
        SELECT *
        FROM layer_effects
        WHERE layerName = "Narva RAAS v1"
        ORDER BY ABS(predicted_effect);
    - Most balanced maps for a certain matchup
        SELECT *
        FROM layer_effects
        WHERE 
        (team1_subfaction = "CAF+AirAssault" AND team2_subfaction = "WPMC+AirAssault")
        OR (team2_subfaction = "CAF+AirAssault" AND team1_subfaction = "WPMC+AirAssault")
        ORDER BY ABS(predicted_effect);

Database schema:
```
CREATE TABLE matches (
    matchId INTEGER PRIMARY KEY,
    gamemode TEXT NOT NULL,
    mapClassname TEXT NOT NULL,
    layerName TEXT NOT NULL,
    duration INTEGER NOT NULL,

    winningTeamID INTEGER NOT NULL,
    winningTeam TEXT NOT NULL,
    winningSubfaction TEXT NOT NULL,
    winningTickets INTEGER NOT NULL,

    losingTeam TEXT NOT NULL,
    losingSubfaction TEXT NOT NULL,
    losingTickets INTEGER NOT NULL,

    teamdataCollected INTEGER NOT NULL DEFAULT 0
, faction1_type TEXT, faction2_type TEXT);
CREATE TABLE teams (
                matchId INTEGER,
                teamID INTEGER,
                squadID INTEGER,
                steamID TEXT,
                eosID TEXT,
                mssID TEXT,
                canonical_id TEXT,
                isSL INTEGER,
                role TEXT,
                UNIQUE(matchId, teamID, canonical_id)
            );
CREATE TABLE player_aliases (
                alias TEXT NOT NULL,
                alias_type TEXT NOT NULL CHECK(alias_type IN ('steam', 'eos', 'mss')),
                canonical_id TEXT NOT NULL,
                PRIMARY KEY (alias, alias_type)
            );
CREATE TABLE IF NOT EXISTS "trained_matches" (
matchId INTEGER PRIMARY KEY,
tag TEXT DEFAULT 'default',
trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE sqlite_sequence(name,seq);
CREATE INDEX idx_matches_teamdataCollected
ON matches(teamdataCollected);
CREATE INDEX idx_player_aliases_canonical
            ON player_aliases(canonical_id);
CREATE INDEX idx_matches_teamdata ON matches(teamdataCollected, matchId);
CREATE TABLE player_ratings (
            player_id TEXT PRIMARY KEY, games_played INTEGER, general_skill REAL,
            sigma REAL DEFAULT 2.0, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE player_role_ratings (
            player_id TEXT, role TEXT, role_skill REAL, games_in_role INTEGER,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (player_id, role));
CREATE TABLE layer_effects (
        id INTEGER PRIMARY KEY AUTOINCREMENT, gamemode TEXT, map TEXT, game_mode_full TEXT,
        layerName TEXT, layer_version TEXT, team1_faction TEXT, team2_faction TEXT,
        team1_subfaction TEXT, team2_subfaction TEXT, team1_tag TEXT, team2_tag TEXT,
        matchup TEXT, predicted_effect REAL, actual_team1_winrate REAL, sample_count INTEGER,
        rotation_format TEXT, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
CREATE INDEX idx_layer_effects_layer ON layer_effects (layerName, team1_subfaction, team2_subfaction);
```


=============

Credits to MySquadStats for providing historical match data on their website. (mysquadstats.com)