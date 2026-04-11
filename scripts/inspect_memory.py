#!/usr/bin/env python3
"""
Inspect the KnowledgeGraph memory for a character.

Usage (run from project root):
    uv run python scripts/inspect_memory.py <character_name>
    uv run python scripts/inspect_memory.py <character_name> --chat <chat_id>
    uv run python scripts/inspect_memory.py <character_name> --person <name>
"""

import argparse
import sqlite3
import sys
from pathlib import Path


def _connect(db_path: str):
    path = Path(db_path)
    if not path.exists():
        print(f"No memory file found: {db_path}")
        print("The bot hasn't stored any memories yet, or the character name is wrong.")
        sys.exit(1)
    return sqlite3.connect(db_path)


def cmd_stats(db_path: str) -> None:
    conn = _connect(db_path)
    entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
    current = conn.execute("SELECT COUNT(*) FROM triples WHERE valid_to IS NULL").fetchone()[0]
    predicates = [r[0] for r in conn.execute(
        "SELECT DISTINCT predicate FROM triples ORDER BY predicate"
    ).fetchall()]

    names = [r[0] for r in conn.execute("SELECT name FROM entities").fetchall()]
    chats = sorted({n.split("::", 1)[0] for n in names if "::" in n})

    conn.close()

    print(f"  entities      : {entities}")
    print(f"  total facts   : {total}")
    print(f"  current facts : {current}")
    print(f"  expired facts : {total - current}")
    print(f"  predicates    : {', '.join(predicates) or '(none)'}")
    print(f"  chats         : {', '.join(chats) or '(none)'}")


def cmd_list(db_path: str, chat_id: str | None, person: str | None) -> None:
    conn = _connect(db_path)
    query = """
        SELECT s.name as subject, t.predicate, o.name as object,
               t.valid_from, t.valid_to
        FROM triples t
        JOIN entities s ON t.subject = s.id
        JOIN entities o ON t.object = o.id
    """
    conditions, params = [], []
    if chat_id:
        conditions.append("s.name LIKE ?")
        params.append(f"{chat_id}::%")
    if person:
        conditions.append("(s.name LIKE ? OR s.name LIKE ?)")
        params += [f"%::{person}", f"%::{person}%"]
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY s.name, t.predicate"

    rows = conn.execute(query, params).fetchall()
    conn.close()

    if not rows:
        print("  (no facts found)")
        return

    current_chat = None
    current_subject = None
    for subject, predicate, obj, valid_from, valid_to in rows:
        if "::" in subject:
            chat, display = subject.split("::", 1)
        else:
            chat, display = "(unknown)", subject

        if chat != current_chat:
            current_chat = chat
            print(f"\n  chat {chat}")

        if display != current_subject:
            current_subject = display
            print(f"    {display}")

        status = "expired" if valid_to else "current"
        since = f" (since {valid_from})" if valid_from else ""
        print(f"      {predicate}: {obj}  [{status}{since}]")


def cmd_entries(char_name: str, db_path: str, limit: int = 100) -> None:
    """Print up to `limit` current facts as readable one-liners."""
    conn = _connect(db_path)
    rows = conn.execute(
        """
        SELECT s.name, t.predicate, o.name
        FROM triples t
        JOIN entities s ON t.subject = s.id
        JOIN entities o ON t.object = o.id
        WHERE t.valid_to IS NULL
        ORDER BY t.rowid DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()

    if not rows:
        print("  (no facts)")
        return

    for subject, predicate, obj in rows:
        person = subject.split("::", 1)[1] if "::" in subject else subject
        print(f"  {char_name} knows: {person}  {predicate} → {obj}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect character memory")
    parser.add_argument("character", help="Character name (matches <character>_memory.db)")
    parser.add_argument("--chat", metavar="CHAT_ID", help="Filter by chat ID")
    parser.add_argument("--person", metavar="NAME", help="Filter by person name")
    args = parser.parse_args()

    db_path = f"{args.character}_memory.db"
    print(f"Memory file: {db_path}")
    print()

    print("Stats:")
    cmd_stats(db_path)

    print(f"\nEntries (up to 100 current facts):")
    cmd_entries(args.character, db_path, limit=100)

    print("\nAll facts (grouped):")
    cmd_list(db_path, args.chat, args.person)
    print()


if __name__ == "__main__":
    main()
