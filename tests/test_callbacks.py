"""Conformance of the adaptive-timeout callback helpers.

The gate has three legs:

- **schedule replay** — the helpers reproduce captured per-iteration solver
  settings over a representative schedule. Bootstrap also replays the captured
  retirement floors exactly. Point estimation shares bounded-phase floors but
  has its own terminal policy: no sentinel after the bounded schedule is spent.
- **capability** — a callback reaches a configurable target's settings; on a
  non-configurable target it returns the floor and raises nothing.
- **determinism** — the same ``(schedule, iteration)`` yields the same
  ``(settings, floor)`` on every call.

The helpers are tested as pure functions against the hook contract
``(iteration, oracle) -> int | None``; no live row-generation loop runs here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from combrum.callbacks import (
    Phase,
    Schedule,
    bootstrap_timeout_callback,
    point_timeout_callback,
)
from combrum.solver_settings import SolverConfigurable, SolverSettings

TESTS = Path(__file__).resolve().parent
CALLBACK_FIXTURE_DIR = TESTS / "fixtures" / "callbacks"
REPLAY_FIXTURE = "adaptive_timeout_replay"


def _load_replay_fixture() -> dict:
    return json.loads(
        (CALLBACK_FIXTURE_DIR / f"{REPLAY_FIXTURE}.json").read_text()
    )


def _schedule_from_fixture(spec: list[dict]) -> Schedule:
    """Rebuild the typed Schedule from the fixture's plain-dict shape."""
    return Schedule(
        [
            Phase(
                timeout=phase["timeout"],
                iters=phase.get("iters"),
                retire=phase.get("retire", False),
            )
            for phase in spec
        ]
    )


class _Recorder:
    """A SolverConfigurable target that records applied settings."""

    def __init__(self) -> None:
        self.received: list[SolverSettings] = []

    def apply_solver_settings(self, settings: SolverSettings) -> None:
        self.received.append(settings)

    @property
    def last(self) -> SolverSettings:
        return self.received[-1]


class _Plain:
    """A target without apply_solver_settings: the capability gate must skip it."""


@pytest.fixture
def schedule() -> Schedule:
    return _schedule_from_fixture(_load_replay_fixture()["schedule"])


@pytest.fixture
def n_iters() -> int:
    return int(_load_replay_fixture()["n_iters"])


# ---- typed schedule validation ---------------------------------------------


def test_phase_validates_at_construction() -> None:
    Phase(timeout=1.0)  # terminal, defaults fine
    Phase(timeout=2.5, iters=3, retire=True)
    # iters=1 is the smallest legal bounded span (iters=0 rejected below)
    smallest = Phase(timeout=1.0, iters=1)
    assert smallest.iters == 1
    assert type(smallest.iters) is int
    # np.int64 spans are normalised to builtin int via operator.index
    normalised = Phase(timeout=1.0, iters=np.int64(2))
    assert normalised.iters == 2
    assert type(normalised.iters) is int
    # non-integral spans are rejected, not truncated (int() would accept 2.5 as 2)
    with pytest.raises((TypeError, ValueError)):
        Phase(timeout=1.0, iters=2.5)  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        Phase(timeout=1.0, iters="3")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Phase(timeout=0.0)  # budget must be > 0 (shares SolverSettings rule)
    with pytest.raises(ValueError):
        Phase(timeout=math.inf)
    with pytest.raises(ValueError):
        Phase(timeout=1.0, iters=0)  # a bounded phase needs a positive span
    # retire is bool-only: retire=1/0 is rejected, not coerced
    with pytest.raises(ValueError):
        Phase(timeout=1.0, retire="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Phase(timeout=1.0, retire=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Phase(timeout=1.0, retire=0)  # type: ignore[arg-type]
    assert Phase(timeout=1.0, iters=2, retire=True).retire is True
    assert Phase(timeout=1.0, iters=2, retire=False).retire is False


def test_smallest_bounded_span_floors_at_one() -> None:
    # a lone iters=1 bounded phase owns exactly iteration 0, so the non-retiring
    # floor is its cumulative boundary 1. The span is built from np.int64 so the
    # returned floor exercises normalisation to builtin int (hook contract is
    # int | None).
    sched = Schedule([Phase(timeout=1.0, iters=np.int64(1)), Phase(timeout=5.0)])
    for helper, terminal_floor in (
        (point_timeout_callback, None),
        (bootstrap_timeout_callback, 10**9),
    ):
        callback = helper(sched)
        rec = _Recorder()
        floor0 = callback(0, rec)
        assert floor0 == 1, helper.__name__
        assert type(floor0) is int, helper.__name__  # bounded floor is builtin int
        assert rec.last.time_limit_seconds == 1.0
        assert rec.last.mip_focus == 1  # cold-start hint at iter 0
        rec_terminal = _Recorder()
        assert callback(1, rec_terminal) == terminal_floor, helper.__name__
        assert rec_terminal.last.time_limit_seconds == 5.0


def test_retiring_terminal_phase_floors_at_zero_for_both_helpers() -> None:
    # a retiring TERMINAL phase floors at 0 for both helpers: the retire rule
    # wins over each helper's terminal policy (bootstrap sentinel, point None).
    #   phase0 iters {0,1}   @1.0s  non-retire -> boundary 2
    #   phase1 iters {2,3,4} @5.0s  retire     -> 0
    #   terminal iters {5,6} @600s  retire     -> 0  (both helpers)
    sched = Schedule(
        [
            Phase(timeout=1.0, iters=2),
            Phase(timeout=5.0, iters=3, retire=True),
            Phase(timeout=600.0, retire=True),
        ]
    )
    n = 7
    expected_timeout = [1.0, 1.0, 5.0, 5.0, 5.0, 600.0, 600.0]
    expected_focus = [1, 0, 0, 0, 0, 0, 0]
    expected_floor = [2, 2, 0, 0, 0, 0, 0]

    for helper in (point_timeout_callback, bootstrap_timeout_callback):
        callback = helper(sched)
        got_floor, got_timeout, got_focus = [], [], []
        for it in range(n):
            rec = _Recorder()
            got_floor.append(callback(it, rec))
            got_timeout.append(rec.last.time_limit_seconds)
            got_focus.append(rec.last.mip_focus)
        assert got_floor == expected_floor, helper.__name__
        assert got_timeout == expected_timeout, helper.__name__
        assert got_focus == expected_focus, helper.__name__
        # retiring-terminal floors are 0, not the sentinel/None
        assert callback(5, _Recorder()) == 0, helper.__name__
        assert callback(6, _Recorder()) == 0, helper.__name__

    # with a non-retiring terminal each helper keeps its own terminal policy
    sched_nonretire = Schedule(
        [
            Phase(timeout=1.0, iters=2),
            Phase(timeout=5.0, iters=3, retire=True),
            Phase(timeout=600.0),
        ]
    )
    assert bootstrap_timeout_callback(sched_nonretire)(6, _Recorder()) == 10**9
    assert point_timeout_callback(sched_nonretire)(6, _Recorder()) is None


def test_schedule_requires_one_terminal_phase_last() -> None:
    # A lone terminal phase is a valid schedule.
    Schedule([Phase(timeout=5.0)])
    Schedule([Phase(timeout=1.0, iters=2), Phase(timeout=5.0)])
    with pytest.raises(ValueError):
        Schedule([])  # empty
    with pytest.raises(ValueError):
        Schedule([Phase(timeout=1.0, iters=2)])  # last not terminal
    with pytest.raises(ValueError):
        # A terminal phase before the end leaves a bounded phase no boundary.
        Schedule([Phase(timeout=1.0), Phase(timeout=5.0)])


# ---- schedule replay: the gate ----------------------------------------------


def test_bootstrap_helper_matches_captured_schedule_field_by_field(
    schedule, n_iters
) -> None:
    """Bootstrap replays the fixture's (settings, floor) sequence -- the
    fixture covers boundary, retiring, and terminal-sentinel floors.
    """
    fixture = _load_replay_fixture()
    callback = bootstrap_timeout_callback(schedule)

    expected = fixture["bootstrap"]
    assert len(expected) == n_iters
    for it in range(n_iters):
        # Return value is the floor; settings arrive as a side effect on the target.
        recorder = _Recorder()
        floor = callback(it, recorder)  # type: ignore[arg-type]
        settings = recorder.last
        row = expected[it]
        assert settings.time_limit_seconds == row["TimeLimit"], (
            f"bootstrap iter {it}: TimeLimit mismatch"
        )
        assert settings.mip_focus == row["MIPFocus"], (
            f"bootstrap iter {it}: MIPFocus mismatch"
        )
        assert floor == row["min_iterations"], (
            f"bootstrap iter {it}: min_iterations floor mismatch"
        )


def test_point_helper_uses_bounded_floors_without_terminal_sentinel(
    schedule, n_iters
) -> None:
    # The terminal phase adds no convergence floor — otherwise a terminal
    # non-retiring phase would block every point estimate until max_iterations.
    fixture = _load_replay_fixture()
    callback = point_timeout_callback(schedule)
    # read the point block, not bootstrap (the two happen to share settings)
    expected = fixture["point"]
    assert len(expected) == n_iters

    # cumulative bounded boundaries are [2, 5, 9] (phases of iters 2, 3, 4):
    # non-retiring phases floor at their boundary, the retiring phase (iters
    # 5–8) floors at 0, the terminal phase (iters 9–11) gets no sentinel.
    expected_floors = [2, 2, 5, 5, 5, 0, 0, 0, 0, None, None, None]
    assert len(expected_floors) == n_iters

    # the point fixture block still records the bootstrap-style sentinel on the
    # terminal iterations; the point helper must override it to None.
    terminal_iters = [it for it, f in enumerate(expected_floors) if f is None]
    assert terminal_iters == [9, 10, 11]
    for it in terminal_iters:
        assert expected[it]["min_iterations"] == 10**9, (
            f"point fixture iter {it}: expected the captured terminal sentinel"
        )

    for it, expected_floor in enumerate(expected_floors):
        recorder = _Recorder()
        floor = callback(it, recorder)  # type: ignore[arg-type]
        row = expected[it]
        assert recorder.last.time_limit_seconds == row["TimeLimit"], (
            f"point iter {it}: TimeLimit mismatch"
        )
        assert recorder.last.mip_focus == row["MIPFocus"], (
            f"point iter {it}: MIPFocus mismatch"
        )
        assert floor == expected_floor, (
            f"point iter {it}: floor mismatch"
        )
        # bounded iters match the capture; terminal iters diverge (None vs
        # the recorded sentinel)
        if expected_floor is None:
            assert row["min_iterations"] != floor
        else:
            assert floor == row["min_iterations"], (
                f"point iter {it}: bounded floor must match point capture"
            )


# ---- determinism ------------------------------------------------------------


def test_same_schedule_and_iteration_is_deterministic(schedule, n_iters) -> None:
    for helper in (point_timeout_callback, bootstrap_timeout_callback):
        for it in range(n_iters):
            # Cross-instance: two fresh callbacks over the same schedule agree.
            a, b = _Recorder(), _Recorder()
            f1, f2 = helper(schedule), helper(schedule)
            floor_a = f1(it, a)  # type: ignore[arg-type]
            floor_b = f2(it, b)  # type: ignore[arg-type]
            assert floor_a == floor_b
            assert a.last.time_limit_seconds == b.last.time_limit_seconds
            assert a.last.mip_focus == b.last.mip_focus
            # same instance called twice at the same iteration: no per-call
            # state may leak (e.g. a seen-before flag bumping floor or focus)
            f3 = helper(schedule)
            c = _Recorder()
            floor_first = f3(it, c)  # type: ignore[arg-type]
            floor_again = f3(it, c)  # type: ignore[arg-type]
            assert floor_again == floor_first
            assert c.received[0].time_limit_seconds == c.received[1].time_limit_seconds
            assert c.received[0].mip_focus == c.received[1].mip_focus


def test_cold_start_hint_tracks_schedule_iteration_not_call_order(
    schedule, n_iters
) -> None:
    # MIPFocus=1 is the cold-start hint for schedule iteration 0, not for
    # whatever iteration a callback instance first happens to be invoked at.
    # A stateful "first call" flag would flip mip_focus=1 on the first call
    # regardless of iteration; pin the hint to iteration 0 by giving a fresh
    # callback its first call at a non-zero iteration and at 0.
    for helper in (point_timeout_callback, bootstrap_timeout_callback):
        for first_it in range(1, n_iters):
            cb = helper(schedule)
            rec = _Recorder()
            cb(first_it, rec)  # type: ignore[arg-type]
            assert rec.last.mip_focus == 0, (
                f"{helper.__name__}: first call at iter {first_it} must not"
                " carry the cold-start hint"
            )
        cb0 = helper(schedule)
        rec0 = _Recorder()
        cb0(0, rec0)  # type: ignore[arg-type]
        assert rec0.last.mip_focus == 1, (
            f"{helper.__name__}: a fresh callback first called at iter 0 must"
            " carry the cold-start hint"
        )


# ---- capability: applies, and no-ops ---------------------------------------


def test_callback_applies_settings_to_a_configurable_target(schedule) -> None:
    # A configurable target receives exactly the phase's settings, and the
    # callback still returns the floor.
    callback = point_timeout_callback(schedule)
    recorder = _Recorder()
    floor0 = callback(0, recorder)  # type: ignore[arg-type]
    # Exactly one apply per call: the side effect is applied once, not skipped
    # and not duplicated. A redundant re-apply (double solver reconfiguration)
    # dies here.
    assert len(recorder.received) == 1
    assert recorder.last.time_limit_seconds == 1.0
    assert recorder.last.mip_focus == 1  # cold-start hint at iter 0
    assert floor0 == 2
    floor2 = callback(2, recorder)  # type: ignore[arg-type]
    # A second call appends exactly one more: two calls => two records.
    assert len(recorder.received) == 2
    assert recorder.last.time_limit_seconds == 5.0
    assert recorder.last.mip_focus == 0
    assert floor2 == 5


def test_callback_skips_a_non_configurable_target(schedule) -> None:
    # The capability gate: a target without apply_solver_settings is skipped,
    # never an error, and the floor still comes back.
    target = _Plain()
    assert not isinstance(target, SolverConfigurable)
    for helper in (point_timeout_callback, bootstrap_timeout_callback):
        callback = helper(schedule)
        assert callback(0, target) == 2  # type: ignore[arg-type]
        assert callback(5, target) == 0  # type: ignore[arg-type]
