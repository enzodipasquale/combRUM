"""Adaptive per-phase solver-timeout callbacks for the row-generation hook."""

from __future__ import annotations

import operator
from bisect import bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from combrum.oracle import Oracle
from combrum.solver_settings import SolverConfigurable, SolverSettings

_TERMINAL_FLOOR = 10**9


@dataclass(frozen=True)
class Phase:
    """One scheduled phase: a solver time budget over ``iters`` iterations.

    Args:
        timeout: Wall budget in seconds, mapped to
            :attr:`SolverSettings.time_limit_seconds`.
        iters: Iteration span, or ``None`` for the terminal phase (which
            owns every iteration past the last bounded phase).
        retire: Whether reps may retire early once this phase is active.
    """

    timeout: float
    iters: int | None = None
    retire: bool = False

    def __post_init__(self) -> None:
        if self.timeout is None:
            raise ValueError("Phase.timeout must be a finite float > 0, got None")
        SolverSettings(time_limit_seconds=self.timeout)
        if self.iters is not None:
            iters = operator.index(self.iters)
            if iters < 1:
                raise ValueError(
                    f"Phase.iters must be >= 1 or None, got {self.iters!r}"
                )
            object.__setattr__(self, "iters", iters)
        if not isinstance(self.retire, bool):
            raise ValueError(f"Phase.retire must be a bool, got {self.retire!r}")

    @property
    def is_terminal(self) -> bool:
        return self.iters is None


@dataclass(frozen=True)
class TimeoutSchedule:
    """An ordered phase sequence ending in exactly one terminal phase."""

    phases: tuple[Phase, ...]

    def __init__(self, phases: Sequence[Phase]) -> None:
        phases = tuple(phases)
        if not phases:
            raise ValueError("TimeoutSchedule must have at least one phase")
        if not phases[-1].is_terminal:
            raise ValueError(
                "the last phase must be terminal (iters=None): it runs to"
                " convergence or the engine's iteration cap"
            )
        if any(p.is_terminal for p in phases[:-1]):
            raise ValueError(
                "only the last phase may be terminal; every earlier phase"
                " needs an iters span to own a boundary"
            )
        object.__setattr__(self, "phases", phases)


def _phase_lookup(
    schedule: TimeoutSchedule,
    *,
    terminal_floor: int | None,
) -> Callable[[int], tuple[SolverSettings, int | None]]:
    bounded = schedule.phases[:-1]
    terminal = schedule.phases[-1]
    boundaries: list[int] = []
    running = 0
    for phase in bounded:
        assert phase.iters is not None
        running += phase.iters
        boundaries.append(running)

    def lookup(iteration: int) -> tuple[SolverSettings, int | None]:
        idx = bisect_right(boundaries, iteration)
        phase = bounded[idx] if idx < len(bounded) else terminal
        settings = SolverSettings(
            time_limit_seconds=phase.timeout,
            mip_focus=1 if iteration == 0 else 0,
        )
        if phase.retire:
            floor = 0
        elif idx < len(bounded):
            floor = boundaries[idx]
        else:
            floor = terminal_floor
        return settings, floor

    return lookup


def _timeout_callback(
    schedule: TimeoutSchedule,
    *,
    terminal_floor: int | None,
) -> Callable[[int, Oracle], int | None]:
    lookup = _phase_lookup(schedule, terminal_floor=terminal_floor)

    def callback(iteration: int, oracle: Oracle) -> int | None:
        settings, floor = lookup(iteration)
        if isinstance(oracle, SolverConfigurable):
            oracle.apply_solver_settings(settings)
        return floor

    return callback


def point_timeout_callback(
    schedule: TimeoutSchedule,
) -> Callable[[int, Oracle], int | None]:
    """Build a per-iteration hook for point estimation over ``schedule``.

    The returned callback applies the active phase's solver settings to
    the oracle (iff configurable) and returns the active bounded phase's
    ``min_iterations`` floor. The terminal phase returns ``None`` so point
    estimation can converge normally after the bounded schedule is spent.
    """
    return _timeout_callback(schedule, terminal_floor=None)


def bootstrap_timeout_callback(
    schedule: TimeoutSchedule,
) -> Callable[[int, Oracle], int | None]:
    """Build a per-iteration hook for the distributed bootstrap over ``schedule``.

    The bootstrap floor keeps a rep from retiring before its phase boundary.
    A non-retiring terminal phase returns the sentinel floor; mark the terminal
    phase ``retire=True`` when reps may retire once that phase is active.
    """
    return _timeout_callback(schedule, terminal_floor=_TERMINAL_FLOOR)
