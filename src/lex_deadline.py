"""Бюджет прогона: глобальный дедлайн, доля на вопрос, лимит вызова."""

from __future__ import annotations

from dataclasses import dataclass

from lex_retry import Clock, WallClock

DEFAULT_RUN_TIMEOUT_S = 20 * 60
DEFAULT_RESERVE_S = 30.0
MIN_CALL_TIMEOUT_S = 0.05


@dataclass(frozen=True, slots=True)
class Deadline:
    end_monotonic: float
    reserve_s: float = 0.0

    @classmethod
    def from_now(
        cls,
        total_s: float,
        *,
        clock: Clock | None = None,
        reserve_s: float = DEFAULT_RESERVE_S,
    ) -> Deadline:
        now = (clock or WallClock()).monotonic()
        return cls(end_monotonic=now + total_s, reserve_s=max(0.0, reserve_s))

    @property
    def work_end(self) -> float:
        return self.end_monotonic - self.reserve_s

    def remaining(self, clock: Clock) -> float:
        return self.work_end - clock.monotonic()

    def expired(self, clock: Clock) -> bool:
        return self.remaining(clock) <= 0

    def call_timeout(self, clock: Clock, own_s: float) -> float:
        left = self.remaining(clock)
        if left <= 0:
            return 0.0
        return max(MIN_CALL_TIMEOUT_S, min(own_s, left))

    def for_question(self, clock: Clock, remaining_questions: int) -> Deadline:
        left = max(0.0, self.remaining(clock))
        n = max(1, remaining_questions)
        slice_s = left / n
        now = clock.monotonic()
        return Deadline(end_monotonic=now + slice_s, reserve_s=0.0)
