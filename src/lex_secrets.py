"""Обёртка ключа: в repr и str значения нет."""

from __future__ import annotations


class Secret:
    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def get(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "Secret(***)"

    def __str__(self) -> str:
        return "***"
