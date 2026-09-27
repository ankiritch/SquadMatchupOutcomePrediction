#!/usr/bin/env python3
import sqlite3
import argparse

from rating_system_SHARED import *


def main():
    parser = argparse.ArgumentParser()

    source = parser.add_mutually_exclusive_group()

    source.add_argument(
        "--steam",
        metavar="ALIAS",
        help="look up a Steam alias",
    )
    source.add_argument(
        "--eos",
        metavar="ALIAS",
        help="look up an EOS alias",
    )
    source.add_argument(
        "--mss",
        metavar="ALIAS",
        help="look up an MSS alias",
    )

    parser.add_argument("--id", type=int)

    args = parser.parse_args()

    if args.id is not None:
        query = """
            SELECT
                pr.player_id,
                pr.games_played,
                pr.general_skill,
                pr.sigma,
                prr.role,
                prr.role_skill,
                prr.games_in_role
            FROM player_ratings AS pr
            LEFT JOIN player_role_ratings AS prr
                ON prr.player_id = pr.player_id
            WHERE pr.player_id = ?
        """

        params = (args.id,)

    else:
        selected_aliases = [
            ("steam", args.steam),
            ("eos", args.eos),
            ("mss", args.mss),
        ]

        alias_type, alias = next(
            ((alias_type, alias)
             for alias_type, alias in selected_aliases
             if alias is not None),
            (None, None),
        )

        if alias is None:
            parser.error(
                "provide either --id, or one of --steam, --eos, or --mss"
            )

        query = """
            SELECT
                pr.player_id,
                pr.games_played,
                pr.general_skill,
                pr.sigma,
                prr.role,
                prr.role_skill,
                prr.games_in_role
            FROM player_aliases AS pa
            JOIN player_ratings AS pr
                ON pr.player_id = pa.canonical_id
            LEFT JOIN player_role_ratings AS prr
                ON prr.player_id = pr.player_id
            WHERE pa.alias = ?
              AND pa.alias_type = ?
        """

        params = (alias, alias_type)

    conn = sqlite3.connect(DB_PATH)

    try:
        rows = conn.execute(query, params).fetchall()
    finally:
        conn.close()

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
