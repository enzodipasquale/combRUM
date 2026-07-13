"""The generic solve-method contract, not specific to row generation.

Nothing presumes a master problem, cuts, or duals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from combrum.context import FitContext
from combrum.demand import Demand


def _readonly(arr: np.ndarray) -> np.ndarray:
    arr.setflags(write=False)
    return arr


def _staged_penalty(
    ref: np.ndarray, weight: float, K: int
) -> tuple[np.ndarray, float]:
    """Validate a proximal ``(ref, weight)`` pair for staging.

    ``ref`` is a frozen copy, detached from
    caller memory so a later mutation cannot leak into an already-staged
    solve.
    """
    ref_arr = np.asarray(ref, dtype=np.float64)
    if ref_arr.shape != (K,):
        raise ValueError(f"expected ref of shape ({K},), got shape {ref_arr.shape}")
    weight_value = float(weight)
    if not np.isfinite(weight_value):
        raise ValueError(f"expected finite weight, got {weight!r}")
    staged_ref = _readonly(np.array(ref_arr, dtype=np.float64, copy=True))
    return staged_ref, weight_value


def _require_owner_master(
    master: object, required: type, formulation: str
) -> None:
    """``required`` (the backend base class) rides in as an argument: this
    module deliberately has no import edge to the master layer.
    """
    if master is None:
        raise ValueError(
            f"{formulation} is master-based by definition:"
            " ctx.master_backend must be set on the owner rank"
        )
    if not isinstance(master, required):
        raise ValueError(
            f"ctx.master_backend does not implement {required.__name__}"
            f" (got {type(master).__name__})"
        )


@dataclass(frozen=True)
class Evaluation:
    """One iteration's evaluated step: progress measure plus method state.

    The stop rule is ``violation <= tolerance``; ``violation`` is the only
    field the caller reads.

    ``payload`` is method-owned evaluated state carried forward by the
    same object into :meth:`Formulation.update`; the caller never
    interprets it.
    """

    violation: float
    payload: object | None = None

    def __post_init__(self) -> None:
        violation = float(self.violation)
        if not violation >= 0.0:
            raise ValueError(f"violation must be >= 0; got {violation}")
        object.__setattr__(self, "violation", violation)


@dataclass(frozen=True)
class FormulationResult:
    """The answer a formulation publishes.

    ``theta_hat`` and ``objective`` are positional so a method cannot
    silently fall back to a last iterate. ``n_active_cuts`` counts cuts
    active in the answer; ``0`` is valid. ``active_set``, ``dual``, and
    ``metadata`` are method-owned and opaque to the caller.
    """

    theta_hat: np.ndarray
    objective: float
    n_active_cuts: int
    slack: np.ndarray | None = None
    active_set: object | None = None
    dual: object | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        theta_hat = np.asarray(self.theta_hat, dtype=np.float64)
        if theta_hat.ndim != 1:
            raise ValueError(
                f"expected one-dimensional (K,) theta_hat, got shape {theta_hat.shape}"
            )
        if np.any(~np.isfinite(theta_hat)):
            raise ValueError("theta_hat must be finite")
        object.__setattr__(self, "theta_hat", _readonly(theta_hat))
        object.__setattr__(self, "objective", float(self.objective))
        if self.n_active_cuts < 0:
            raise ValueError(f"n_active_cuts must be >= 0; got {self.n_active_cuts}")
        if self.slack is not None:
            object.__setattr__(
                self,
                "slack",
                _readonly(np.asarray(self.slack, dtype=np.float64)),
            )


class Formulation(ABC):
    """The generic solve-method contract.

    A driver runs every formulation through the same walk::

        setup(ctx)
        repeat: theta = solve(); evaluate(priced demands); update(step)
        result(); dispose()

    The published estimate comes from :meth:`result`, never the last
    :meth:`solve` output. Convergence is
    ``Evaluation.violation <= tolerance``. ``theta_init`` on
    :class:`~combrum.context.FitContext` is the only seed affordance.
    """

    @abstractmethod
    def setup(self, ctx: FitContext) -> None:
        """Bind the fit geometry and interfaces; called once before the walk."""

    @abstractmethod
    def solve(self) -> np.ndarray:
        """Produce the next theta to price, shape ``(K,)``.

        A query point, not the answer (which lives in :meth:`result`); it
        need not be any master's solution.
        """

    @abstractmethod
    def evaluate(self, demands: Mapping[int, Demand]) -> Evaluation:
        """Fold the subproblems priced at the current theta into progress.

        Keys are GLOBAL agent ids.
        """

    @abstractmethod
    def update(self, step: Evaluation) -> int:
        """Advance internal state by consuming the evaluated step.

        ``step`` is the same object the preceding :meth:`evaluate`
        returned; the method reads ``step.payload`` from it. Returns a
        method-owned progress count; ``0`` is valid (a normal step for
        many methods), so the caller must not branch on it as a health
        signal.
        """

    @abstractmethod
    def result(self) -> FormulationResult:
        """The published answer."""

    def dispose(self) -> None:
        """Release method-held resources; default no-op."""
