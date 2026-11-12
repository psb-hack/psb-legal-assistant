"""Фейковая модель для тестов: отвечает по сценарию, в сеть не ходит."""

from __future__ import annotations

from collections.abc import Sequence

from lex_models import ChatTurn, FinishReason, JsonCompletion


class ScriptedJsonModel:
    """Отвечает заготовленными ответами по очереди и запоминает запросы."""

    def __init__(self, *replies: str | JsonCompletion | Exception) -> None:
        self._replies = list(replies)
        self.requests: list[tuple[ChatTurn, ...]] = []

    def complete_json(self, turns: Sequence[ChatTurn]) -> JsonCompletion:
        self.requests.append(tuple(turns))
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, str):
            return JsonCompletion(content=reply, finish=FinishReason.STOP)
        return reply
