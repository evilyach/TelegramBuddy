"""
Shared SQLite cache mapping (chat_id, message_id) -> voice message transcript.

All bots write to this cache when they send a voice message, and read from it
when they receive a voice message — enabling bot-to-bot communication over voice.
"""

import asyncio
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS voice_transcripts (
    chat_id    INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    transcript TEXT    NOT NULL,
    created_at REAL    NOT NULL DEFAULT (unixepoch('now')),
    PRIMARY KEY (chat_id, message_id)
)
"""


class MessageCache:
    """
    Thread-safe (check_same_thread=False) SQLite cache for voice message transcripts.
    Uses a single shared DB file so every CharacterBot instance can read each other's
    cached transcripts without any IPC.
    """

    def __init__(self, db_path: str = "message_cache.db") -> None:
        path = Path(db_path)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()
        self._transcription_locks: dict[tuple[int, int], asyncio.Lock] = {}
        logger.debug("MessageCache opened at %s", path.resolve())

    def transcription_lock(self, chat_id: int, message_id: int) -> asyncio.Lock:
        """Return a per-message lock so only one bot transcribes a given voice message."""
        key = (chat_id, message_id)
        if key not in self._transcription_locks:
            self._transcription_locks[key] = asyncio.Lock()
        return self._transcription_locks[key]

    def store(self, chat_id: int, message_id: int, transcript: str) -> None:
        """Cache the transcript for a voice message that was just sent."""
        self._conn.execute(
            "INSERT OR REPLACE INTO voice_transcripts (chat_id, message_id, transcript) VALUES (?, ?, ?)",
            (chat_id, message_id, transcript),
        )
        self._conn.commit()
        logger.debug("MessageCache: stored transcript for chat=%d msg=%d", chat_id, message_id)

    def lookup(self, chat_id: int, message_id: int) -> str | None:
        """Return the cached transcript, or None if not found."""
        row = self._conn.execute(
            "SELECT transcript FROM voice_transcripts WHERE chat_id = ? AND message_id = ?",
            (chat_id, message_id),
        ).fetchone()
        if row:
            logger.debug("MessageCache: hit for chat=%d msg=%d", chat_id, message_id)
            return row[0]
        logger.debug("MessageCache: miss for chat=%d msg=%d", chat_id, message_id)
        return None

    def close(self) -> None:
        self._conn.close()
