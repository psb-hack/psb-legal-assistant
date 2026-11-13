from __future__ import annotations

from lex_deadline import MIN_CALL_TIMEOUT_S, Deadline


class FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def monotonic(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def test_remaining_subtracts_reserve() -> None:
    clock = FakeClock(10.0)
    deadline = Deadline.from_now(100.0, clock=clock, reserve_s=30.0)
    assert deadline.work_end == 80.0
    assert deadline.remaining(clock) == 70.0
    clock.advance(70.0)
    assert deadline.expired(clock)


def test_call_timeout_caps_to_remaining() -> None:
    clock = FakeClock(0.0)
    deadline = Deadline.from_now(5.0, clock=clock, reserve_s=0.0)
    assert deadline.call_timeout(clock, 90.0) == 5.0
    clock.advance(4.98)
    assert deadline.call_timeout(clock, 90.0) == MIN_CALL_TIMEOUT_S
    clock.advance(1.0)
    assert deadline.call_timeout(clock, 90.0) == 0.0


def test_for_question_splits_remaining() -> None:
    clock = FakeClock(0.0)
    deadline = Deadline.from_now(90.0, clock=clock, reserve_s=0.0)
    slice_deadline = deadline.for_question(clock, 3)
    assert slice_deadline.remaining(clock) == 30.0
    assert slice_deadline.reserve_s == 0.0
