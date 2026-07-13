"""Composable row-generation phase contract.

A row-generation method's per-iteration step splits into three phases the
engine drives, so the engine, not the method, owns the cross-rank
reduce and exchange. With the collective hoisted out of the method, one
engine can fold several live replications through one reduce/exchange
super-step instead of paying one collective per replication.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from combrum.demand import Demand
from combrum.transport.base import CutRow


@dataclass(frozen=True)
class MaxContribution:
    """Per-agent-slack reduction half: a MAX scalar plus local rows.

    ``worst`` is the rank's max reduced cost, floored at 0.0 so an empty
    shard contributes exactly 0.0 to the MAX. ``local_rows`` are the
    violated cut rows the engine's exchange routes to their owning rank.
    """

    worst: float
    local_rows: tuple[CutRow, ...]


@dataclass(frozen=True)
class SumContribution:
    """Aggregate-slack reduction half: weighted vectors to SUM.

    ``terms`` is ``(n_rows, M)`` weighted ``(phi | eps)`` rows and ``ids``
    the matching ``(n_rows,)`` sum keys: one row per local agent keyed by
    global agent id, or one pre-summed row under OneSlack's single-rank
    fast path. The engine SUMs these keyed on ``ids``; the reduction is
    reproducible, so the aggregate lands bitwise identical on every rank.
    """

    terms: np.ndarray
    ids: np.ndarray


Contribution = MaxContribution | SumContribution


@dataclass(frozen=True)
class MaxReduced:
    """Engine output for a :class:`MaxContribution`.

    ``global_worst`` is the MAX of every rank's ``worst``. ``received_rows``
    are the rows the exchange routed to this rank, in canonical order.
    """

    global_worst: float
    received_rows: tuple[CutRow, ...]


@dataclass(frozen=True)
class SumReduced:
    """Engine output for a :class:`SumContribution`.

    ``aggregate`` is the ``(M,)`` reproducible SUM of every rank's
    per-agent ``terms``, bitwise identical on every rank.
    """

    aggregate: np.ndarray


Reduced = MaxReduced | SumReduced


@dataclass(frozen=True)
class StepOutcome:
    """Rank-local result of :meth:`RowGenStep.finalise`.

    ``violation`` is the progress measure on the generic
    :class:`~combrum.formulation.Evaluation` distance; the stop rule reads
    only this. ``install_payload`` is the method-owned object
    :meth:`RowGenStep.apply_step` installs, opaque to the engine.
    """

    violation: float
    install_payload: object


class RowGenStep(Protocol):
    """One composable row-generation step, phased for engine ownership."""

    def contribute(self, demands: Mapping[int, Demand]) -> Contribution:
        """Fold this rank's priced demands into its :class:`Contribution`.

        Transport-free and rank-local (no collective); the engine reduces
        across ranks. ``demands`` keys are global agent ids.
        """
        ...

    def finalise(self, reduced: Reduced) -> StepOutcome:
        """Map the engine-reduced value onto a :class:`StepOutcome`.

        Transport-free and rank-local: every rank already holds the
        identical ``reduced`` value, so no further agreement round.
        """
        ...

    def apply_step(self, install_payload: object) -> int:
        """Install the payload on the owner rank, solve, and broadcast the
        master state.

        The method's one inherent owner-rooted collective. Returns the
        progress count (cuts newly installed); ``0`` is a valid step.
        """
        ...
