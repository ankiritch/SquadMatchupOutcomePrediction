#!/usr/bin/env python3
"""
Squad stats collector — Python port with cross-platform player identity resolution.
Requires: requests  (pip install requests)
"""
import argparse
import json
import os
import signal
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import requests

shutdown_event = threading.Event()
match_list_done = False
loop_mode = False


DB_NAME = "squadDB.sqlite"
OLDEST_CHECKED_PAGE = 1380
STATE_FILE = "basicMatchdataCollectorVariables"

API_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:142.0) Gecko/20100101 Firefox/142.0",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "identity",
    "Origin": "https://mysquadstats.com",
    "Connection": "close",
    "Referer": "https://mysquadstats.com/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
}

TEAM_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:142.0) Gecko/20100101 Firefox/142.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "identity",
    "Connection": "close",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "cross-site",
    "Priority": "u=0, i",
}


def init_db() -> None:
    with sqlite3.connect(DB_NAME) as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS teams (
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
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS player_aliases (
                alias TEXT NOT NULL,
                alias_type TEXT NOT NULL CHECK(alias_type IN ('steam', 'eos', 'mss')),
                canonical_id TEXT NOT NULL,
                PRIMARY KEY (alias, alias_type)
            );
        """)
        db.execute("""
            CREATE INDEX IF NOT EXISTS idx_player_aliases_canonical
            ON player_aliases(canonical_id);
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS matches (
                matchId INTEGER PRIMARY KEY,
                gamemode TEXT,
                mapClassname TEXT,
                layerName TEXT,
                duration INTEGER,
                winningTeamID INTEGER,
                winningTeam TEXT,
                winningSubfaction TEXT,
                winningTickets INTEGER,
                losingTeam TEXT,
                losingSubfaction TEXT,
                losingTickets INTEGER,
                teamdataCollected INTEGER DEFAULT 0,
                faction1_type TEXT,
                faction2_type TEXT
            );
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS player_ratings (
                player_id TEXT PRIMARY KEY,
                sigma REAL DEFAULT 2.0
            );
        """)
        db.commit()


def api_request(session: requests.Session, url: str, headers: Dict[str, str], timeout: int = 10) -> Tuple[Any, int]:
    resp = session.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    remaining = int(resp.headers.get("X-RateLimit-Remaining", "100"))
    return resp.json(), remaining


def remove_unassigned_squads(data: List[Any]) -> List[Any]:
    if not isinstance(data, list):
        return []
    for team in data:
        if isinstance(team, dict) and isinstance(team.get("squads"), list):
            team["squads"] = [
                sq for sq in team["squads"]
                if not (
                    isinstance(sq, dict)
                    and str(sq.get("squadID")) == "0"
                    and sq.get("squadName") == "Unassigned"
                )
            ]
    return data



def team_data_collector(thread_count: int = 10) -> None:
    queues: List[List[int]] = [[] for _ in range(thread_count)]
    queues_lock = threading.Lock()
    active_workers = 0

    def worker(worker_id: int) -> None:
        nonlocal active_workers
        db = sqlite3.connect(DB_NAME, check_same_thread=False)
        db.execute("PRAGMA journal_mode=WAL;")
        db.execute("PRAGMA busy_timeout = 30000;")
        db.execute("PRAGMA synchronous = NORMAL;")

        session = requests.Session()

        insert_team = """
            INSERT INTO teams
            (matchId, teamID, squadID, steamID, eosID, mssID, canonical_id, isSL, role)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        update_match = "UPDATE matches SET teamdataCollected = ? WHERE matchId = ?;"

        identity_cache: dict[tuple, str] = {}

        def resolve_cached(cur, steam_id, eos_id, mss_id) -> Optional[str]:
            present: List[tuple] = []
            if steam_id:
                present.append(("steam", str(steam_id)))
            if eos_id:
                present.append(("eos", str(eos_id)))
            if mss_id is not None:
                present.append(("mss", str(mss_id)))

            if not present:
                return None

            for key in present:
                if key in identity_cache:
                    canonical = identity_cache[key]
                    for k in present:
                        identity_cache[k] = canonical
                    return canonical

            aliases = [a for _, a in present]
            placeholders = ",".join("?" for _ in aliases)
            cur.execute(
                f"SELECT canonical_id FROM player_aliases WHERE alias IN ({placeholders})",
                aliases,
            )
            rows = cur.fetchall()
            if rows:
                canonical = rows[0][0]
            else:
                canonical = str(uuid.uuid4())
                for alias_type, alias in present:
                    cur.execute(
                        "INSERT OR IGNORE INTO player_aliases (alias, alias_type, canonical_id) VALUES (?, ?, ?)",
                        (alias, alias_type, canonical),
                    )

            for key in present:
                identity_cache[key] = canonical
            return canonical

        def fetch_match(match_id: int):
            url = f"https://mysquadstats.com/matchesPlayers/{match_id}.json"
            for attempt in range(1, 4):
                try:
                    resp, _ = api_request(session, url, TEAM_HEADERS, timeout=10)
                    return ("ok", match_id, resp)
                except requests.exceptions.JSONDecodeError:
                    return ("empty", match_id, None)
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                    if attempt < 3:
                        time.sleep(0.1)
                        continue
                    return ("retry", match_id, None)
                except Exception as exc:
                    return ("error", match_id, exc)
            return ("error", match_id, "max retries")

        try:
            while not shutdown_event.is_set():
                with queues_lock:
                    if queues[worker_id]:
                        batch = queues[worker_id]
                        queues[worker_id] = []
                        active_workers += 1
                    else:
                        batch = None

                if not batch:
                    if shutdown_event.wait(1.0):
                        break
                    continue

                try:
                    print(f"[T{worker_id}] Batch of {len(batch)} matches starting...")

                    t0_fetch = time.perf_counter()
                    results: List[tuple] = []
                    pool = ThreadPoolExecutor(max_workers=4)
                    futures = {pool.submit(fetch_match, mid): mid for mid in batch}
                    try:
                        for fut in as_completed(futures):
                            if shutdown_event.is_set():
                                break
                            results.append(fut.result())
                    finally:
                        pool.shutdown(wait=False, cancel_futures=True)

                    print(f"[T{worker_id}] Fetched {len(results)} matches in {time.perf_counter()-t0_fetch:.1f}s")

                    t0_sql = time.perf_counter()
                    cur = db.cursor()
                    cur.execute("BEGIN TRANSACTION;")
                    pending = 0

                    try:
                        for status, match_id, payload in results:
                            if shutdown_event.is_set():
                                break

                            if status == "empty":
                                cur.execute(update_match, (-1, match_id))
                                pending += 1
                                continue
                            if status != "ok":
                                continue

                            response = remove_unassigned_squads(payload)

                            player_team = 1
                            for team in response:
                                if not isinstance(team, dict):
                                    continue
                                for squad in team.get("squads", []):
                                    if not isinstance(squad, dict):
                                        continue
                                    for player in squad.get("players", []):
                                        if not isinstance(player, dict):
                                            continue

                                        canonical_id = resolve_cached(
                                            cur,
                                            player.get("steamID"),
                                            player.get("eosID"),
                                            player.get("mssID"),
                                        )
                                        if canonical_id is None:
                                            continue

                                        is_leader = bool(player.get("isLeader", False))
                                        is_command = bool(squad.get("isCommandSquad", False))
                                        leader_value = 2 if (is_leader and is_command) else (1 if is_leader else 0)

                                        try:
                                            cur.execute(insert_team, (
                                                match_id, player_team, player.get("squadID", 0),
                                                player.get("steamID"), player.get("eosID"), player.get("mssID"),
                                                canonical_id, leader_value, player.get("role", ""),
                                            ))
                                        except sqlite3.IntegrityError:
                                            pass
                                player_team += 1

                            cur.execute(update_match, (1, match_id))
                            pending += 1

                            if pending >= 10:
                                cur.execute("COMMIT;")
                                cur.execute("BEGIN TRANSACTION;")
                                pending = 0

                        if pending:
                            cur.execute("COMMIT;")
                        else:
                            try:
                                cur.execute("COMMIT;")
                            except Exception:
                                pass

                    except Exception as exc:
                        try:
                            cur.execute("ROLLBACK;")
                        except Exception:
                            pass
                        print(f"[T{worker_id}] Exception during SQLite write: {exc}")
                        raise

                    print(f"[T{worker_id}] SQLite done in {time.perf_counter()-t0_sql:.1f}s, cache size: {len(identity_cache)}")
                finally:
                    with queues_lock:
                        active_workers -= 1
        finally:
            db.close()
            session.close()


    workers = []
    for i in range(thread_count):
        t = threading.Thread(target=worker, args=(i,), daemon=True)
        t.start()
        workers.append(t)


    db = sqlite3.connect(DB_NAME, check_same_thread=False)
    db.execute("PRAGMA journal_mode=WAL;")
    db.execute("PRAGMA busy_timeout = 30000;")
    highest_queued_rowid = 0

    try:
        while not shutdown_event.is_set():
            # Health check
            if any(not t.is_alive() for t in workers):
                print("Worker died unexpectedly. Shutting down.")
                shutdown_event.set()
                break

            with queues_lock:
                shortest_idx = min(range(thread_count), key=lambda i: len(queues[i]))
                shortest_len = len(queues[shortest_idx])

            if shortest_len > 50:
                if shutdown_event.wait(0.2):
                    break
                continue

            cur = db.cursor()
            cur.execute(
                "SELECT ROWID, matchId FROM matches "
                "WHERE teamdataCollected = 0 AND ROWID > ? "
                "ORDER BY ROWID LIMIT 100;",
                (highest_queued_rowid,),
            )
            rows = cur.fetchall()

            if not rows:
                if not loop_mode and match_list_done:
                    with queues_lock:
                        total_queued = sum(len(q) for q in queues)
                        current_active = active_workers
                    if total_queued == 0 and current_active == 0:
                        print("All matches processed. Draining complete.")
                        shutdown_event.set()
                        break
                    else:
                        if shutdown_event.wait(1.0):
                            break
                        continue
                else:
                    if shutdown_event.wait(1.0):
                        break
                    continue

            to_assign: List[int] = []
            for rowid, match_id in rows:
                highest_queued_rowid = rowid
                to_assign.append(match_id)

            if to_assign:
                cur.execute("SELECT COUNT(*) FROM matches WHERE teamdataCollected = 0;")
                remaining = cur.fetchone()[0]
                print(f"Distributor: assigned {len(to_assign)} to queue {shortest_idx}, {remaining} total remaining")

            with queues_lock:
                queues[shortest_idx].extend(to_assign)
    finally:
        shutdown_event.set()
        for t in workers:
            t.join(timeout=10)
        db.close()


def match_data_collector() -> None:
    global match_list_done
    checked_pages: set[int] = set()
    current_page = OLDEST_CHECKED_PAGE
    matchdata_request_count = 0

    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                if key == "checked_pages":
                    for page_str in value.split(","):
                        if page_str.strip():
                            checked_pages.add(int(page_str.strip()))
                elif key == "oldest_checked_page":
                    current_page = int(value)
                elif key == "matchdata_request_count":
                    matchdata_request_count = int(value)

    while current_page in checked_pages:
        current_page += 1

    db = sqlite3.connect(DB_NAME, check_same_thread=False)
    db.execute("PRAGMA journal_mode=WAL;")
    db.execute("PRAGMA busy_timeout = 30000;")

    insert_sql = """
        INSERT OR IGNORE INTO matches (
            matchId, gamemode, mapClassname, layerName, duration,
            winningTeamID, winningTeam, winningSubfaction, winningTickets,
            losingTeam, losingSubfaction, losingTickets, teamdataCollected
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0);
    """

    next_allowed_list_request = 0

    team_thread = threading.Thread(target=team_data_collector, args=(1,), daemon=True)
    team_thread.start()

    session = requests.Session()

    try:
        while not shutdown_event.is_set():
            now = time.time()
            if next_allowed_list_request > now:
                if shutdown_event.wait(next_allowed_list_request - now):
                    break
                continue

            url = (
                "https://api.mysquadstats.com/matchData?"
                f"draw=1&page={current_page}&pageSize=100"
                "&sortColumn=matchId&sortDirection=asc&search="
            )

            retries = 0
            while True:
                try:
                    res, remaining = api_request(session, url, API_HEADERS)
                    break
                except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError) as exc:
                    retries += 1
                    wait = min(2 ** retries, 60)
                    print(f"Matchlist connection error (attempt {retries}), waiting {wait}s...")
                    if shutdown_event.wait(wait):
                        return
                    continue
                except Exception as exc:
                    print(f"Unexpected matchlist error: {exc}")
                    raise RuntimeError("failed matchdata collection")

            if remaining <= 5:
                print("Reached rate limit")
                next_allowed_list_request = time.time() + 60

            status = res.get("status")
            message = res.get("message")
            if status == "Error" and message == "No match data found. Match may still be in progress.":
                print("Reached last page")
                if not loop_mode:
                    print("Reached end of match list in single-run mode. Waiting for team data collector to finish...")
                    match_list_done = True
                    break
                next_allowed_list_request = time.time() + 60 * 5
                continue

            matches_list = res.get("data", [])
            if len(matches_list) != 100:
                next_allowed_list_request = time.time() + 60 * 5
                print("Reached last page. Page is not full yet.")
                if not loop_mode:
                    print("Reached end of match list in single-run mode. Waiting for team data collector to finish...")
                    match_list_done = True
                    break
                continue

            for match_value in matches_list:
                match_obj = match_value if isinstance(match_value, dict) else {}
                if match_obj.get("mod") != "Vanilla":
                    continue

                fields = [
                    "matchId", "gamemode", "mapClassname", "layerName", "duration",
                    "winningTeamID", "winningTeam", "winningSubfaction", "winningTickets",
                    "losingTeam", "losingSubfaction", "losingTickets",
                ]

                values = []
                broken = False
                for field in fields:
                    val = match_obj.get(field)
                    if val is None:
                        broken = True
                        break
                    values.append(val)

                if broken:
                    continue

                try:
                    db.execute(insert_sql, tuple(values))
                except sqlite3.IntegrityError as exc:
                    if "UNIQUE constraint failed: matches.matchId" in str(exc):
                        continue
                    raise

            db.commit()

            checked_pages.add(current_page)

            with open(STATE_FILE, "w", encoding="utf-8") as fh:
                fh.write(f"oldest_checked_page:{current_page}\n")
                fh.write(f"matchdata_request_count:{matchdata_request_count}\n")
                fh.write("checked_pages:")
                fh.write(",".join(str(p) for p in sorted(checked_pages)))
                fh.write("\n")

            print(f"Checked matchlist page: {current_page}")
            current_page += 1
            matchdata_request_count += 1
    finally:
        if loop_mode or not match_list_done:
            shutdown_event.set()
        team_thread.join(timeout=3600)
        session.close()
        db.close()


def main() -> None:
    global loop_mode
    parser = argparse.ArgumentParser(description="Squad stats collector")
    parser.add_argument("--loop", action="store_true", help="Run in continuous loop mode")
    args = parser.parse_args()
    loop_mode = args.loop

    init_db()

    with sqlite3.connect(DB_NAME) as db:
        mode = db.execute("PRAGMA journal_mode=WAL;").fetchone()[0]
        print(f"Journal mode: {mode}")
        if mode != "wal":
            print("ERROR: WAL mode could not be enabled. Exiting.")
            return
        db.execute("PRAGMA busy_timeout = 30000;")

    match_thread = threading.Thread(target=match_data_collector, daemon=False)
    match_thread.start()

    try:
        match_thread.join()
    except KeyboardInterrupt:
        print("\nCtrl+C received — shutting down gracefully...")
        shutdown_event.set()
        match_thread.join(timeout=30)
        print("Done.")


if __name__ == "__main__":
    main()