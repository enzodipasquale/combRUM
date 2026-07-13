"""The engine-owned one-iteration fit step.

The engine drives a row-generation method through its phases and owns the
cross-rank collectives between them, so the method stays transport-passive.

One iteration, for a :class:`~combrum.rowgen.RowGenStep` formulation:

1. price (comm-free): the scheduled local ids resolve to
   ``{id: Demand}`` via :func:`~combrum.interface_resolution.price_demands`.
2. contribute (transport-free): the formulation folds its priced
   demands into a :class:`~combrum.rowgen.Contribution`.
3. reduce + exchange (the one collective set per iteration): the
   engine reduces, dispatching on the concrete contribution type.
4. finalise (transport-free): the formulation maps the reduced value
   onto a :class:`~combrum.rowgen.StepOutcome`.
5. apply_step (owner-only): the formulation installs/solves on the owner
   rank and mirrors the master state in its one inherent owner-rooted bcast.

The reduce/exchange shapes match the formulations' bundled reduce exactly
(``allreduce_max`` + ``exchange_cuts`` for MAX, ``sum_reproducible`` for SUM),
so the phase path is bitwise-equal to the bundled path.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from combrum.demand import Demand
from combrum.interface_resolution import (
    Resolution,
    needs_conformance_guard,
    price_demands,
)
from combrum.rowgen import (
    Contribution,
    MaxContribution,
    MaxReduced,
    Reduced,
    RowGenStep,
    StepOutcome,
    SumContribution,
    SumReduced,
)
from combrum.transport.base import Transport


@dataclass(frozen=True)
class StepResult:
    """One fit-step's outcome plus the per-step collective tally."""

    violation: float
    progressed: int
    reduce_rounds: int
    exchange_rounds: int
    n_priced: int
    n_inexact: int
    n_candidates: int
    pricing_seconds: float
    master_seconds: float


def reduce_contribution(
    transport: Transport,
    contribution: Contribution,
    *,
    owner_rank: int = 0,
) -> tuple[Reduced, int, int]:
    """Reduce + exchange a contribution, dispatched on its concrete type."""
    if isinstance(contribution, MaxContribution):
        owners = np.array([int(owner_rank)], dtype=np.int64)
        global_worst = transport.allreduce_max(contribution.worst)
        received = transport.exchange_cuts(contribution.local_rows, owners)
        return MaxReduced(global_worst=global_worst, received_rows=received), 1, 1
    if isinstance(contribution, SumContribution):
        aggregate = np.asarray(
            transport.sum_reproducible(contribution.terms, contribution.ids),
            dtype=np.float64,
        )
        return SumReduced(aggregate=aggregate), 1, 0
    raise AssertionError(
        f"unhandled contribution type: {type(contribution).__name__};"
        " the engine dispatches the reduce by concrete Contribution type"
    )


def fit_step(
    formulation: RowGenStep,
    *,
    transport: Transport,
    price_resolution: Resolution,
    theta: np.ndarray,
    scheduled_local_ids: Sequence[int],
    owner_rank: int = 0,
    before_apply: Callable[[], None] | None = None,
    demand_sink: Callable[..., None] | None = None,
) -> StepResult:
    """Run one engine-owned fit-step over a transport-passive formulation.

    ``before_apply`` is an optional hook run after finalise and before
    apply_step. The penalty path uses it for formulations without
    ``prepare_penalty_solve``, re-solving the master so the install adopts
    the penalized iterate. ``None`` otherwise.

    ``demand_sink`` is an optional read-only observer called with the
    ``{id: Demand}`` mapping after pricing and before contribute. Read-only:
    it moves no wire and changes no reduction.
    """

    def _price_and_contribute():
        demands: Mapping[int, Demand] = price_demands(
            price_resolution, theta, scheduled_local_ids
        )
        n_priced = len(demands)
        gaps = getattr(demands, "gaps", None)
        if gaps is not None:
            gap_arr = np.asarray(gaps, dtype=np.float64)
            inexact = gap_arr > 0.0
            n_inexact = int(np.count_nonzero(inexact))
            worst_gap = float(np.max(gap_arr[inexact])) if n_inexact else 0.0
        else:
            n_inexact = 0
            worst_gap = 0.0
            for demand in demands.values():
                gap = float(demand.gap)
                if gap > 0.0:
                    n_inexact += 1
                    if gap > worst_gap:
                        worst_gap = gap
        if demand_sink is not None:
            demand_sink(
                demands,
                n_priced=n_priced,
                n_inexact=n_inexact,
                worst_gap=worst_gap,
            )
        contribution = formulation.contribute(demands)
        return contribution, n_priced, n_inexact

    price_t0 = time.perf_counter()
    guard_pricing = transport.size > 1 or needs_conformance_guard(
        price_resolution, getattr(formulation, "_features_res", None)
    )
    if guard_pricing:
        with transport.collective():
            contribution, n_priced, n_inexact = _price_and_contribute()
    else:
        contribution, n_priced, n_inexact = _price_and_contribute()
    price_t1 = time.perf_counter()
    reduced, reduce_rounds, exchange_rounds = reduce_contribution(
        transport, contribution, owner_rank=owner_rank
    )
    n_candidates = len(reduced.received_rows) if isinstance(reduced, MaxReduced) else 1
    outcome: StepOutcome = formulation.finalise(reduced)
    master_t0 = time.perf_counter()
    if before_apply is not None:
        before_apply()
    progressed = formulation.apply_step(outcome.install_payload)
    step_t1 = time.perf_counter()
    return StepResult(
        violation=float(outcome.violation),
        progressed=int(progressed),
        reduce_rounds=reduce_rounds,
        exchange_rounds=exchange_rounds,
        n_priced=int(n_priced),
        n_inexact=int(n_inexact),
        n_candidates=int(n_candidates),
        pricing_seconds=float(price_t1 - price_t0),
        master_seconds=float(step_t1 - master_t0),
    )
