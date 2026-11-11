"""Политика повторов: статусы, полный джиттер, Retry-After, дедлайн."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

T = TypeVar("T")

RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


class Sleeper(Protocol):
    def sleep(self, seconds: float) -> None: ...


class Clock(Protocol):
    def monotonic(self) -> float: ...


class WallClock:
    def monotonic(self) -> float:
        return time.monotonic()


class WallSleeper:
    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int
    base_delay: float
    max_delay: float
    deadline_monotonic: float | None
    rng: random.Random

    @classmethod
    def standard(
        cls,
        *,
        deadline_monotonic: float | None = None,
        rng: random.Random | None = None,
    ) -> RetryPolicy:
        return cls(
            max_attempts=4,
            base_delay=0.25,
            max_delay=8.0,
            deadline_monotonic=deadline_monotonic,
            rng=rng if rng is not None else random.Random(),
        )

    def remaining(self, clock: Clock) -> float:
        if self.deadline_monotonic is None:
            return 1e9
        return self.deadline_monotonic - clock.monotonic()

    def delay_after(self, attempt_index: int, retry_after: float | None) -> float:
        cap = min(self.max_delay, self.base_delay * (2**attempt_index))
        wait = self.rng.random() * cap
        if retry_after is not None and retry_after > 0:
            wait = max(wait, retry_after)
        return wait


def parse_retry_after(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def run_with_retries(
    policy: RetryPolicy,
    clock: Clock,
    sleeper: Sleeper,
    attempt: Callable[[], T],
    *,
    should_retry: Callable[[Exception], tuple[bool, float | None]],
) -> T:
    last_error: Exception | None = None
    for index in range(policy.max_attempts):
        if policy.remaining(clock) <= 0:
            break
        try:
            return attempt()
        except Exception as exc:
            last_error = exc
            retry, retry_after = should_retry(exc)
            if not retry or index + 1 >= policy.max_attempts:
                raise
            wait = policy.delay_after(index, retry_after)
            if wait > policy.remaining(clock):
                raise
            sleeper.sleep(wait)
    if last_error is not None:
        raise last_error
    raise RuntimeError("retry loop exited without attempt")
