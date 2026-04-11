#!/usr/bin/env python3
"""
Inspect the voice message transcript cache.

Usage (run from project root):
    uv run python scripts/inspect_cache.py
    uv run python scripts/inspect_cache.py --chat <chat_id>
    uv run python scripts/inspect_cache.py --limit 50
"""

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

DB_PATH = "message_cache.db"


def _connect():
    path = Path(DB_PATH)
    if not path.exists():
        print(f"No cache file found: {DB_PATH}")
        print("No voice messages have been cached yet.")
        sys.exit(1)
    return sqlite3.connect(DB_PATH)


def cmd_stats(conn: sqlite3.Connection) -> None:
    total = conn.execute("SELECT COUNT(*) FROM voice_transcripts").fetchone()[0]
    chats = conn.execute("SELECT COUNT(DISTINCT chat_id) FROM voice_transcripts").fetchone()[0]
    print(f"  total cached  : {total}")
    print(f"  unique chats  : {chats}")


def cmd_last(conn: sqlite3.Connection, chat_id: int | None, limit: int) -> None:
    query = "SELECT chat_id, message_id, transcript, created_at FROM voice_transcripts"
    params: list = []
    if chat_id is not None:
        query += " WHERE chat_id = ?"
        params.append(chat_id)
    query += " ORDER BY rowid DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    if not rows:
        print("  (no entries)")
        return

    for chat, msg_id, transcript, created_at in rows:
        ts = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d %H:%M:%S") if created_at else "?"
        preview = transcript.replace("\n", " ")
        if len(preview) > 120:
            preview = preview[:117] + "..."
        print(f"  [{ts}] chat={chat} msg={msg_id}")
        print(f"    {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect voice message transcript cache")
    parser.add_argument("--chat", type=int, metavar="CHAT_ID", help="Filter by chat ID")
    parser.add_argument("--limit", type=int, default=20, metavar="N", help="Number of entries to show (default: 20)")
    args = parser.parse_args()

    print(f"Cache file: {DB_PATH}")
    print()

    conn = _connect()

    print("Stats:")
    cmd_stats(conn)

    print(f"\nLast {args.limit} cached messages:")
    cmd_last(conn, args.chat, args.limit)

    conn.close()
    print()


if __name__ == "__main__":
    main()
