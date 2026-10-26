"""Minimal per-session conversation memory.

Stores the most recent turns per session id so a caller can show history
or build follow-up context. In-memory only, trimmed to ``max_turns`` —
persistence and summarisation are deliberately out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Turn:
    query: str
    answer: str


@dataclass
class ConversationMemory:
    max_turns: int = 20
    _sessions: dict[str, list[Turn]] = field(default_factory=dict)

    def add_turn(self, session_id: str, query: str, answer: str) -> None:
        history = self._sessions.setdefault(session_id, [])
        history.append(Turn(query=query, answer=answer))
        if len(history) > self.max_turns:
            del history[: len(history) - self.max_turns]

    def get_history(self, session_id: str) -> list[Turn]:
        return list(self._sessions.get(session_id, []))

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def session_ids(self) -> list[str]:
        return list(self._sessions.keys())
