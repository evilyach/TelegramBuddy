from collections import deque
from dataclasses import dataclass
from datetime import datetime

from mempalace.knowledge_graph import KnowledgeGraph


@dataclass
class ChatMessage:
    sender_name: str
    text: str
    timestamp: datetime


class MemoryManager:
    """
    Per-character memory manager.

    Short-term: in-process deque of the last 50 messages per chat.
    Long-term:  KnowledgeGraph (SQLite) storing facts the bot learns about
                group members. Subjects are namespaced as "{chat_id}::{name}"
                to keep per-group facts isolated.
    """

    def __init__(self, char_name: str, db_path: str = "memory.db"):
        self._char_name = char_name
        self._short_term: dict[int, deque[ChatMessage]] = {}
        self._kg = KnowledgeGraph(db_path=db_path)

    # ------------------------------------------------------------------
    # Short-term (in-memory)
    # ------------------------------------------------------------------

    def add_message(self, chat_id: int, sender_name: str, text: str) -> None:
        if chat_id not in self._short_term:
            self._short_term[chat_id] = deque(maxlen=50)
        self._short_term[chat_id].append(
            ChatMessage(sender_name, text, datetime.utcnow())
        )

    def get_recent_messages(self, chat_id: int) -> list[ChatMessage]:
        return list(self._short_term.get(chat_id, []))

    # ------------------------------------------------------------------
    # Long-term (KnowledgeGraph)
    # ------------------------------------------------------------------

    def _scoped(self, chat_id: int, name: str) -> str:
        """Namespace an entity name to a specific chat."""
        return f"{chat_id}::{name}"

    def store_fact(
        self,
        chat_id: int,
        subject: str,
        predicate: str,
        obj: str,
        valid_from: str | None = None,
    ) -> None:
        self._kg.add_triple(
            self._scoped(chat_id, subject),
            predicate,
            obj,
            valid_from=valid_from or datetime.utcnow().date().isoformat(),
        )

    def recall_facts(self, chat_id: int, person_name: str) -> list[dict]:
        """Retrieve known facts about a person in a specific chat."""
        scoped_name = self._scoped(chat_id, person_name)
        return self._kg.query_entity(scoped_name)

    # ------------------------------------------------------------------
    # Context block for LLM
    # ------------------------------------------------------------------

    def build_context_block(self, chat_id: int) -> str:
        history = self.get_recent_messages(chat_id)
        lines = [f"{m.sender_name}: {m.text}" for m in history]

        parts = []
        if lines:
            parts.append("## Recent chat history\n" + "\n".join(lines))
        return "\n\n".join(parts)
