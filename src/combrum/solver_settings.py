"""Runtime-settings contract shared by the engine and configurable solvers.

:class:`SolverSettings` is the per-phase settings vocabulary an adaptive
schedule applies mid-solve; :class:`SolverConfigurable` is the opt-in
capability a solver advertises to receive them. Both are framework
contracts with no concrete-solver dependency, so the callback layer drives
them without naming any solver.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class SolverSettings:
    """Runtime solver settings a phase schedule may apply mid-solve.

    A per-phase time budget and a search-focus hint. ``None`` means
    "leave the solver's own default"; interpretation is solver-owned
    (a solver without a focus notion ignores ``mip_focus``).
    """

    time_limit_seconds: float | None = None
    mip_focus: int | None = None

    def __post_init__(self) -> None:
        if self.time_limit_seconds is not None:
            if isinstance(self.time_limit_seconds, bool) or not isinstance(
                self.time_limit_seconds, (int, float)
            ):
                raise ValueError(
                    "time_limit_seconds must be a finite float > 0 or"
                    f" None; got {self.time_limit_seconds!r}"
                )
            limit = float(self.time_limit_seconds)
            if not math.isfinite(limit) or limit <= 0.0:
                raise ValueError(
                    "time_limit_seconds must be a finite float > 0 or"
                    f" None; got {self.time_limit_seconds!r}"
                )
            object.__setattr__(self, "time_limit_seconds", limit)
        if self.mip_focus is not None:
            focus = operator.index(self.mip_focus)
            if focus < 0:
                raise ValueError(
                    f"mip_focus must be >= 0 or None; got {self.mip_focus!r}"
                )
            object.__setattr__(self, "mip_focus", focus)


@runtime_checkable
class SolverConfigurable(Protocol):
    """Opt-in capability of accepting :class:`SolverSettings`.

    Settings are applied iff ``isinstance(x, SolverConfigurable)``; a
    solver without the capability is left untouched, never an error.
    """

    def apply_solver_settings(self, settings: SolverSettings) -> None: ...
