"""Distributed multiplier bootstrap over a public ``Model``."""

from __future__ import annotations

import copy
import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

import numpy as np

from combrum.activity import (
    ActivityConfig,
    ActivityLevel,
    ActivityRun,
    BootstrapFinal,
    BootstrapRepFinal,
    BootstrapRound,
    BootstrapStart,
    _activity_details,
    _object_name,
    build_activity_run,
)
from combrum.context import ResultPublication
from combrum.demand import Demand
from combrum.dual import DualSolution
from combrum.dualstore import DualStoreWriter
from combrum.engine.agreement import (
    agree_public_bool,
    agree_public_choice,
    agree_public_float,
    agree_public_int,
    agree_public_optional_theta,
    callback_convergence_floor,
    collective_call,
    require_public_object_agreement,
)
from combrum.engine.certify import GapTally, certification_metadata
from combrum.interface_resolution import (
    Resolution,
    needs_conformance_guard,
    price_demands,
    resolve,
)
from combrum.engine.distributed_context import (
    DistributedObservedPrep,
    build_distributed_fit_context,
    prepare_distributed_observed,
)
from combrum.formulations import NSlack
from combrum.masters import resolve_master_backend
from combrum.model import Model
from combrum.oracle import Oracle
from combrum.parameters import Parameters
from combrum.policies import CutPolicy
from combrum.randomness import bootstrap_multiplier
from combrum.result import BootstrapResult
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
from combrum.transport.base import CutRow, Transport

class ThetaEstimate(Protocol):
    """A published estimate whose ``theta_hat`` anchors each rep's ``theta_init``.

    The warm start reads only ``theta_hat``; the cut set is supplied separately
    via ``warm_cuts``. Both ``FitResult`` and ``FormulationResult`` satisfy this.
    """

    theta_hat: np.ndarray


class _Closable(Protocol):
    def close(self) -> None: ...


_DEFAULT_MAX_LIVE_REPS = 64


def _coerce_max_live_reps(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("max_live_reps must be an integer >= 1; got bool")
    try:
        cap = operator.index(value)
    except TypeError as exc:
        raise TypeError(
            "max_live_reps must be an integer >= 1;"
            f" got {type(value).__name__}"
        ) from exc
    if cap < 1:
        raise ValueError(f"max_live_reps must be >= 1; got {value!r}")
    return int(cap)


def _max_live_reps_token(value: object) -> tuple[bool, int, str, str]:
    try:
        return (True, _coerce_max_live_reps(value), "", "")
    except (TypeError, ValueError) as exc:
        return (False, 0, type(exc).__name__, str(exc))


def _agree_max_live_reps(value: object, transport: Transport) -> int:
    if transport.size == 1:
        return _coerce_max_live_reps(value)

    local = _max_live_reps_token(value)

    with transport.collective():
        root = transport.bcast(local if transport.rank == 0 else None, root=0)
        if local != root:
            local_value = local[1] if local[0] else local[3]
            root_value = root[1] if root[0] else root[3]
            raise ValueError(
                "max_live_reps must match on every rank;"
                f" rank {transport.rank} has {local_value!r},"
                f" rank 0 has {root_value!r}"
            )
        if not local[0]:
            if local[2] == "TypeError":
                raise TypeError(local[3])
            raise ValueError(local[3])
    return local[1]


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _publication_label(publication: ResultPublication) -> str:
    if publication == ResultPublication.SUMMARY:
        return "summary"
    parts: list[str] = []
    if publication & ResultPublication.SLACK:
        parts.append("slack")
    if publication & ResultPublication.ACTIVE_SET:
        parts.append("active_set")
    if publication & ResultPublication.DUAL:
        parts.append("dual")
    return "+".join(parts)


def _restamp(rows: Sequence[CutRow], rep_id: int) -> list[CutRow]:
    """Re-key ``rows`` onto replication ``rep_id`` for the batched exchange.

    ``contribute`` builds rows under ``rep_id == 0``; the batched exchange routes
    by ``rep_id``, so each rep's rows are restamped with its slot. ``phi`` is
    shared read-only (frozen on the source row), so the moment vector is not
    copied.
    """
    return [
        CutRow(
            rep_id=rep_id,
            agent_id=row.agent_id,
            phi=row.phi,
            epsilon=row.epsilon,
            bundle_key=row.bundle_key,
        )
        for row in rows
    ]


def batched_reduce(
    transport: Transport,
    contributions: Mapping[int, Contribution],
    owners: np.ndarray,
) -> list[Reduced | None]:
    """Reduce every live replication's contribution in one batched super-step.

    ``contributions`` maps each live slot to that rank's local
    :class:`~combrum.rowgen.Contribution`; ``owners`` is the ``(B,)`` owner
    vector (identical on every rank) routing each rep's rows to its master's
    rank. Returns a length-``B`` list indexed by slot, holding the
    :class:`~combrum.rowgen.Reduced` for each live slot (``None`` otherwise).

    Dispatch is on the concrete contribution kind, exhaustive and uniform across
    the live set (one formulation drives the whole bootstrap):

    * MAX kind: one ``batched_max`` over the ``(B,)`` worsts + one
      ``exchange_cuts``. The result lands on EVERY rank (the batched worst is
      identical everywhere; routed rows land on the owner, empty elsewhere) so
      per-rep finalise/apply_step can run rank-wide.
    * SUM kind: one ``owner_sum`` over the ``(B, M)`` block; each owner receives
      only its owned reps' aggregates, so a non-owner slot is ``None``.

    One batched reduce (+ one exchange for MAX) per call, independent of B.
    """
    B = int(owners.shape[0])
    live_slots = sorted(contributions)
    if not live_slots:
        # Empty reduce desyncs the SPMD program (collective on some ranks only).
        raise AssertionError(
            "batched_reduce called with no live replication; the super-step"
            " loop must stop once the live set is empty rather than issuing"
            " an empty (rank-desyncing) reduce"
        )
    sample = contributions[live_slots[0]]

    if isinstance(sample, MaxContribution):
        worsts = np.zeros(B, dtype=np.float64)
        all_rows: list[CutRow] = []
        for slot in live_slots:
            contribution = contributions[slot]
            _require_kind(contribution, MaxContribution, slot)
            worsts[slot] = contribution.worst
            all_rows.extend(_restamp(contribution.local_rows, slot))
        global_worsts = np.asarray(
            transport.batched_max(worsts), dtype=np.float64
        )
        received = transport.exchange_cuts(all_rows, owners)
        # Rows arrive in canonical (rep_id, agent_id, bundle_key) order; one
        # pass preserves it per slot.
        rows_by_slot: dict[int, list[CutRow]] = {}
        for row in received:
            rows_by_slot.setdefault(row.rep_id, []).append(row)
        # MaxReduced is built on EVERY rank (batched worst is rank-uniform;
        # routed rows land only on the owner) so rank-wide finalise/apply_step
        # has a value at every live slot.
        reduced: list[Reduced | None] = [None] * B
        for slot in live_slots:
            reduced[slot] = MaxReduced(
                global_worst=float(global_worsts[slot]),
                received_rows=tuple(rows_by_slot.get(slot, ())),
            )
        return reduced

    if isinstance(sample, SumContribution):
        # An empty-shard rep still contributes a zero row so the (B, M) block
        # shape is rank-uniform.
        M = _sum_width(sample)
        block = np.zeros((B, M), dtype=np.float64)
        for slot in live_slots:
            contribution = contributions[slot]
            _require_kind(contribution, SumContribution, slot)
            terms = np.asarray(contribution.terms, dtype=np.float64)
            if terms.size:
                # Rank-local pre-combine; owner_sum reduces across ranks
                # reproducibly.
                block[slot] = terms.sum(axis=0)
        sums = transport.owner_sum(block, owners)
        reduced = [None] * B
        for slot, aggregate in sums.items():
            reduced[slot] = SumReduced(
                aggregate=np.asarray(aggregate, dtype=np.float64)
            )
        return reduced

    # Exhaustive by type: an unhandled kind is raised, never mis-reduced.
    raise AssertionError(
        f"unhandled contribution type: {type(sample).__name__};"
        " the B-fold reduce dispatches by concrete Contribution type"
    )


def _store_dual(
    writer: DualStoreWriter, rep_id: int, dual: DualSolution | None
) -> int:
    """Re-stamp one replication's dual onto its global id and persist it.

    Returns 1 when a dual was written, 0 when ``dual is None``. The dual is
    built under ``rep_id == 0`` and re-stamped before the write, else B of them
    would collide on the store's per-``rep_id`` key. One dual is in flight at a
    time, so the writer's resident set is one payload, never B.
    """
    if dual is None:
        return 0
    restamped = dual.with_rep_id(rep_id)
    writer.write(restamped)
    del restamped
    return 1


def _require_kind(
    contribution: Contribution, expected: type, slot: int
) -> None:
    if not isinstance(contribution, expected):
        raise AssertionError(
            "batched_reduce: mixed contribution kinds across the live set"
            f" (slot {slot} is {type(contribution).__name__}, expected"
            f" {expected.__name__}); one formulation drives the whole"
            " bootstrap"
        )


def _sum_width(contribution: SumContribution) -> int:
    terms = np.asarray(contribution.terms)
    return int(terms.shape[1]) if terms.ndim == 2 and terms.size else 0


def _observe_demands(tally: GapTally, demands: Mapping[int, Demand]) -> None:
    gaps = getattr(demands, "gaps", None)
    if gaps is not None:
        gap_arr = np.asarray(gaps, dtype=np.float64)
        inexact = gap_arr > 0.0
        n_inexact = int(np.count_nonzero(inexact))
        worst_gap = float(np.max(gap_arr[inexact])) if n_inexact else 0.0
        tally.observe_counts(len(demands), n_inexact, worst_gap)
        return
    tally.observe(demands)


@dataclass
class _Replica:
    """One replication's per-rep state: formulation, resolved price surface,
    and the local-id shard it prices each iteration."""

    rep_id: int
    formulation: RowGenStep
    price_resolution: Resolution
    scheduled_local_ids: np.ndarray
    master_backend: _Closable | None = None


def _run_bfold(
    replicas: Sequence[_Replica],
    *,
    oracle: Oracle,
    transport: Transport,
    owners: np.ndarray,
    K: int,
    parameters: Parameters,
    tolerance: float,
    max_iterations: int,
    min_iterations: int = 0,
    iteration_callback: Callable[[int, Oracle], int | None] | None = None,
    gap_tally: GapTally | None = None,
    dual_store_dir: Path | str | None = None,
    activity: ActivityRun | None = None,
    round_offset: int = 0,
) -> BootstrapResult:
    """Drive the formulation phases B-fold to per-replication convergence.

    ``replicas`` is the full set (one :class:`_Replica` per slot, in slot
    order); ``owners`` is the ``(B,)`` owner vector. Each iteration, every rank:

    1. for each live rep: solve, price its shard, contribute, then release the
       priced demands immediately (resident set stays O(live-reps x shard));
    2. issue the batched collective(s) for the live set;
    3. for each live rep: finalise, apply_step, then retire if it converged.

    Stops once the live set empties or the iteration budget is spent. The live
    mask is global and deterministic: a rep retires on its own pure-LP
    certificate, agreed because the batched worst is rank-uniform, so every
    rank loops the same live slots with no extra agreement round.

    ``dual_store_dir`` streams each rep's dual to disk from the root rank (one
    payload in flight, never B).

    ``iteration_callback`` runs once per wave-local iteration, before pricing.
    It may update oracle-owned solver settings and return a convergence floor.
    """
    n_reps = len(replicas)
    # Live mask: one bit per rep, True while converging. Global and
    # deterministic so the SPMD collectives stay aligned.
    live = np.ones(n_reps, dtype=bool)
    converged = np.zeros(n_reps, dtype=bool)
    iterations = 0
    log_details = (
        activity is not None
        and activity.enabled
        and _activity_details(activity.config.level)
    )
    diagnostic_reps = (
        activity is not None
        and activity.enabled
        and activity.config.level is ActivityLevel.DIAGNOSTIC
    )
    rep_iterations = np.zeros(n_reps, dtype=np.int64) if log_details else None
    rep_gaps = np.full(n_reps, np.nan, dtype=np.float64) if log_details else None

    # Dual store: one writer, built once; each dual is streamed and dropped.
    writer = (
        DualStoreWriter(dual_store_dir) if dual_store_dir is not None else None
    )
    stored = 0

    needs_prereduce_guard = transport.size > 1 or any(
        needs_conformance_guard(
            replica.price_resolution,
            getattr(replica.formulation, "_features_res", None),
        )
        for replica in replicas
    )

    try:
        for it in range(max_iterations):
            live_slots = np.flatnonzero(live)
            if live_slots.size == 0:
                break
            convergence_floor = callback_convergence_floor(
                name="iteration_callback floor",
                callback=iteration_callback,
                iteration=it,
                oracle=oracle,
                base_floor=int(min_iterations),
                transport=transport,
            )
            iterations += 1
            round_t0 = perf_counter() if log_details else None

            # Phase 1: solve + contribute each live rep, then release its
            # priced demands (no (B, n_agents) object ever resident).
            price_t0 = perf_counter() if log_details else None
            def _price_live() -> dict[int, Contribution]:
                contributions: dict[int, Contribution] = {}
                for slot in live_slots:
                    slot = int(slot)
                    replica = replicas[slot]
                    theta = replica.formulation.solve()
                    demands: Mapping[int, Demand] = price_demands(
                        replica.price_resolution,
                        theta,
                        replica.scheduled_local_ids,
                    )
                    if gap_tally is not None:
                        _observe_demands(gap_tally, demands)
                    contributions[slot] = replica.formulation.contribute(
                        demands
                    )
                    del demands  # release priced demands once folded
                return contributions

            if needs_prereduce_guard:
                with transport.collective():
                    contributions = _price_live()
            else:
                contributions = _price_live()
            price_seconds = perf_counter() - price_t0 if price_t0 is not None else None

            # Phase 2: one batched communication step over the live set, called
            # with the same live slots and owner vector on every rank.
            comm_t0 = perf_counter() if log_details else None
            reduced = batched_reduce(transport, contributions, owners)
            contributions.clear()
            comm_seconds = perf_counter() - comm_t0 if comm_t0 is not None else None

            # Phase 3: finalise + apply_step + retire, per live rep. The batched
            # worst is rank-uniform, so retirement is decided identically
            # everywhere with no extra round.
            master_t0 = perf_counter() if log_details else None
            retired_slots: list[int] = []
            max_gap: float | None = None
            for slot in live_slots:
                slot = int(slot)
                replica = replicas[slot]
                rep_reduced = reduced[slot]
                assert rep_reduced is not None, (
                    f"replication {slot} is live but the batched reduce"
                    " returned no value for it; a reduce/exchange routing"
                    " bug, not a convergence outcome"
                )
                outcome: StepOutcome = replica.formulation.finalise(rep_reduced)
                replica.formulation.apply_step(outcome.install_payload)
                violation = float(outcome.violation)
                max_gap = violation if max_gap is None else max(max_gap, violation)
                if rep_gaps is not None:
                    rep_gaps[slot] = violation
                # Stop rule: a full-shard violation at or under tolerance retires
                # the rep only after any callback-provided phase floor is met.
                if violation <= tolerance and iterations >= convergence_floor:
                    converged[slot] = True
                    live[slot] = False
                    retired_slots.append(slot)
                    if rep_iterations is not None:
                        rep_iterations[slot] = iterations
            master_seconds = (
                perf_counter() - master_t0 if master_t0 is not None else None
            )
            if log_details and activity is not None:
                activity.emit(
                    BootstrapRound(
                        run_id=activity.config.run_id,
                        label=activity.config.label,
                        round_index=round_offset + iterations - 1,
                        live_count=int(live_slots.size),
                        retired_count=len(retired_slots),
                        total_retired=n_reps - int(np.count_nonzero(live)),
                        total_converged=int(np.count_nonzero(converged)),
                        max_gap=max_gap,
                        live_rep_ids=(
                            tuple(
                                int(replicas[int(slot)].rep_id) for slot in live_slots
                            )
                            if diagnostic_reps
                            else None
                        ),
                        retired_rep_ids=(
                            tuple(int(replicas[slot].rep_id) for slot in retired_slots)
                            if diagnostic_reps
                            else None
                        ),
                        price_seconds=price_seconds,
                        comm_seconds=comm_seconds,
                        master_seconds=master_seconds,
                        round_seconds=(
                            perf_counter() - round_t0 if round_t0 is not None else None
                        ),
                    )
                )
        if rep_iterations is not None:
            for slot in np.flatnonzero(live):
                rep_iterations[int(slot)] = iterations
        # Publish each rep's result. theta/count are rank-local; optional
        # slack/dual artifacts land on rank 0 only, when requested.
        thetas = np.zeros((n_reps, K), dtype=np.float64)
        for slot, replica in enumerate(replicas):
            result = replica.formulation.result()
            thetas[slot] = result.theta_hat
            if writer is not None and transport.rank == 0:
                stored += _store_dual(writer, replica.rep_id, result.dual)
            if log_details and activity is not None:
                assert rep_iterations is not None
                assert rep_gaps is not None
                activity.emit(
                    BootstrapRepFinal(
                        run_id=activity.config.run_id,
                        label=activity.config.label,
                        rep_id=int(replica.rep_id),
                        slot=slot,
                        owner_rank=int(owners[slot]),
                        state=("computed" if bool(converged[slot]) else "nonconverged"),
                        converged=bool(converged[slot]),
                        iterations=int(rep_iterations[slot]),
                        final_gap=_finite_or_none(float(rep_gaps[slot])),
                        objective=float(result.objective),
                        active_cuts=int(result.n_active_cuts),
                    )
                )
    finally:
        for replica in replicas:
            try:
                replica.formulation.dispose()
            finally:
                if replica.master_backend is not None:
                    replica.master_backend.close()
    stored_total = (
        int(transport.bcast(stored if transport.rank == 0 else None, root=0))
        if writer is not None
        else 0
    )

    return BootstrapResult(
        thetas=thetas,
        converged=converged.copy(),
        parameters=parameters,
        iterations=iterations,
        dual_store_dir=Path(dual_store_dir) if writer is not None else None,
        n_duals_stored=stored_total,
        u_samples=None,
        duals=None,
    )


def _bootstrap_c_theta_and_normalizer(
    prep: DistributedObservedPrep,
    *,
    base_seed: int,
    rep_id: int,
    transport: Transport,
) -> tuple[np.ndarray, float]:
    """One observation-axis reduction for a rep's normalizer and ``c_theta``."""
    local = _bootstrap_local_rows(
        prep, base_seed=base_seed, rep_ids=[rep_id]
    )
    reduced = np.asarray(
        transport.sum_reproducible(local, prep.owned_obs), dtype=np.float64
    )
    c_theta, normalizer = _finish_bootstrap_reduction(
        prep, np.atleast_2d(reduced)
    )
    return c_theta[0], float(normalizer[0])


def _bootstrap_local_rows(
    prep: DistributedObservedPrep,
    *,
    base_seed: int,
    rep_ids: Sequence[int],
) -> np.ndarray:
    """Observation-keyed bootstrap rows for one live wave.

    Each row is keyed by the global observation id. Columns are grouped by
    replication as ``[raw_weight, c_theta_0, ..., c_theta_K]``. This keeps the
    global reduction row-distribution invariant while still reducing the whole
    live wave in one call.
    """
    rep_ids = tuple(int(rep_id) for rep_id in rep_ids)
    width = len(rep_ids) * (prep.K + 1)
    local = np.empty((prep.owned_obs.size, width), dtype=np.float64)
    for slot, rep_id in enumerate(rep_ids):
        raw = np.fromiter(
            (
                bootstrap_multiplier(base_seed, rep_id, int(obs_id))
                for obs_id in prep.owned_obs
            ),
            dtype=np.float64,
            count=int(prep.owned_obs.size),
        )
        offset = slot * (prep.K + 1)
        local[:, offset] = raw
        local[:, offset + 1 : offset + 1 + prep.K] = -float(prep.S) * (
            raw[:, None] * prep.phi_obs_local
        )
    return local


def _finish_bootstrap_reduction(
    prep: DistributedObservedPrep, reduced: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if reduced.shape != (reduced.shape[0], prep.K + 1):
        raise ValueError(
            "bootstrap observed reduction returned shape"
            f" {reduced.shape}; expected (B, {prep.K + 1})"
        )
    normalizers = np.asarray(reduced[:, 0], dtype=np.float64)
    if np.any(normalizers <= 0.0):
        bad = normalizers[normalizers <= 0.0]
        raise ValueError(
            "bootstrap multiplier normalizer must be positive;"
            f" got {bad.tolist()}"
        )
    scales = float(prep.N) / normalizers
    return reduced[:, 1:] * scales[:, None], normalizers


def _bootstrap_wave_c_theta_and_normalizers(
    prep: DistributedObservedPrep,
    rep_ids: Sequence[int],
    *,
    base_seed: int,
    transport: Transport,
) -> tuple[np.ndarray, np.ndarray]:
    """Observation-axis bootstrap reductions for one live wave."""
    local = _bootstrap_local_rows(
        prep, base_seed=base_seed, rep_ids=rep_ids
    )
    reduced = np.asarray(
        transport.sum_reproducible(local, prep.owned_obs), dtype=np.float64
    )
    return _finish_bootstrap_reduction(
        prep, reduced.reshape(len(rep_ids), prep.K + 1)
    )


def _bootstrap_slack_coef(
    *, n_observations: int, base_seed: int, rep_id: int, normalizer: float
) -> Callable[[int], float]:
    scale = float(n_observations) / float(normalizer)

    def _coef(agent_id: int) -> float:
        obs_id = int(agent_id) % int(n_observations)
        return bootstrap_multiplier(base_seed, rep_id, obs_id) * scale

    return _coef


def _build_distributed_replica(
    rep_id: int,
    *,
    prep: DistributedObservedPrep,
    model: Model,
    c_theta: np.ndarray,
    normalizer: float,
    price_resolution: Resolution,
    transport: Transport,
    owner_rank: int,
    backend: str,
    master_params: dict[str, object] | None,
    tolerance: float,
    cut_policy: CutPolicy | None,
    result_publication: ResultPublication,
    theta_init: np.ndarray | None,
    warm_cuts: Sequence[CutRow] | None,
    base_seed: int,
) -> _Replica:
    """Build one weighted distributed replication without dense weights."""
    formulation = model.formulation(model.features)
    built = build_distributed_fit_context(
        prep,
        model=model,
        formulation=formulation,
        c_theta=c_theta,
        slack_coef=_bootstrap_slack_coef(
            n_observations=prep.N,
            base_seed=base_seed,
            rep_id=rep_id,
            normalizer=normalizer,
        ),
        transport=transport,
        owner_rank=owner_rank,
        master_backend=backend,
        master_params=master_params,
        tolerance=tolerance,
        theta_init=theta_init,
        warm_cuts=warm_cuts,
        cut_policy=cut_policy,
        result_publication=result_publication,
    )
    formulation.setup(built.ctx)
    return _Replica(
        rep_id=rep_id,
        formulation=formulation,
        price_resolution=price_resolution,
        scheduled_local_ids=prep.local_ids,
        master_backend=built.ctx.master_backend,
    )


def _owner_vector(n_bootstrap: int, size: int) -> np.ndarray:
    """Cyclic replication-to-owner map ``owner(b) = b % size`` as a ``(B,)`` int
    array, spreading per-rep masters across ranks. A pure function of ``B`` and
    ``size``, so every rank agrees on ownership with no agreement round.
    """
    return np.arange(n_bootstrap, dtype=np.int64) % size


def bootstrap_distributed(
    model: Model,
    *,
    n_observations: int,
    n_simulations: int,
    n_bootstrap: int,
    base_seed: int,
    transport: Transport,
    max_live_reps: int = _DEFAULT_MAX_LIVE_REPS,
    master_backend: str = "auto",
    master_params: dict[str, object] | None = None,
    tolerance: float = 1e-6,
    max_iterations: int = 1000,
    min_iterations: int = 0,
    iteration_callback: Callable[[int, Oracle], int | None] | None = None,
    warm_start: ThetaEstimate | None = None,
    warm_cuts: Sequence[CutRow] | None = None,
    cut_policy: CutPolicy | None = None,
    dual_store_dir: Path | str | None = None,
    activity: ActivityConfig | None = None,
) -> BootstrapResult:
    """Run the distributed multiplier bootstrap for NSlack.

    ``n_observations`` is the observed row count ``N`` and ``n_simulations`` is
    the number of simulated pricing agents per observation. Pricing runs over
    global ids ``0, ..., N*S-1``; bootstrap multipliers are drawn on observed
    rows only and reused by every simulated agent with the same ``gid % N``.
    ``max_live_reps`` bounds the number of concurrently live bootstrap
    replications per wave; larger values use more memory and fewer waves, while
    smaller values use less memory and more wave setup rounds. Every rank must
    pass the same value. Draws use the distributed counter-based stream keyed by
    ``(base_seed, rep_id, obs_id)``; this is placement-invariant but not the
    same stream as serial ``bootstrap``'s default generator.

    The model must provide the distributed observed-feature surface described by
    :func:`combrum.engine.estimate.estimate_distributed`. The function does not
    accept dense ``Data`` or ``observed_bundles`` and does not collect dense
    per-replication slack arrays. ``metadata["certification"]`` is an aggregate
    pricing-exactness report across the run; per-replication convergence remains
    in ``converged``. ``iteration_callback``, when supplied, runs once per
    wave-local iteration before pricing; the iteration index resets for each
    bounded live-replication wave.
    """
    oracle = model.oracle
    parameters = model.parameters
    formulation = model.formulation

    supported_formulation = agree_public_bool(
        "model.formulation is NSlack", formulation is NSlack, transport
    )
    if not supported_formulation:
        raise NotImplementedError(
            "bootstrap_distributed currently supports model.formulation=NSlack"
            " only; serial bootstrap remains available for other formulations"
        )
    master_backend = str(
        agree_public_choice(
            "master_backend",
            master_backend,
            transport,
            choices=("auto", "gurobi", "highs"),
        )
    )
    n_bootstrap = agree_public_int(
        "n_bootstrap", n_bootstrap, transport, lower=1
    )
    max_iterations = agree_public_int(
        "max_iterations", max_iterations, transport, lower=1
    )
    min_iterations = agree_public_int(
        "min_iterations", min_iterations, transport, lower=0
    )
    if min_iterations > max_iterations:
        raise ValueError(
            "min_iterations must be <= max_iterations;"
            f" got min_iterations={min_iterations!r},"
            f" max_iterations={max_iterations!r}"
        )
    base_seed = agree_public_int("base_seed", base_seed, transport, lower=0)
    tolerance = agree_public_float(
        "tolerance", tolerance, transport, lower=0.0, strict_lower=True
    )
    live_cap = _agree_max_live_reps(max_live_reps, transport)
    master_params = require_public_object_agreement(
        "master_params", master_params, transport
    )
    warm_cuts = require_public_object_agreement(
        "warm_cuts", warm_cuts, transport
    )
    cut_policy = require_public_object_agreement(
        "cut_policy", cut_policy, transport
    )

    prep = prepare_distributed_observed(
        model,
        n_observations=n_observations,
        n_simulations=n_simulations,
        transport=transport,
    )
    K = prep.K
    # Warm-start theta is a public distributed input: every rank must agree
    # before any owner-local context validates or consumes it.
    theta_init = agree_public_optional_theta(
        "warm_start", warm_start, transport, K=K
    )

    # Place rep b's master on owner(b) = b % size; several reps may share an
    # owning rank, kept apart by the rep_id envelope.
    owners = _owner_vector(n_bootstrap, transport.size)
    rep_ids = list(range(n_bootstrap))
    resolved_master_backend = resolve_master_backend(
        master_backend,
        transport=transport,
        owner_ranks=np.unique(owners),
    )

    store_duals = agree_public_bool(
        "dual_store_dir is not None", dual_store_dir is not None, transport
    )
    root_dual_store_dir: Path | None = None
    if store_duals:
        local_store_dir = collective_call(
            transport,
            lambda: (
                Path(dual_store_dir) if transport.rank == 0 else None
            ),
        )
        root_dual_store_dir = transport.bcast(
            local_store_dir if transport.rank == 0 else None, root=0
        )

    result_publication = ResultPublication.SUMMARY
    if store_duals:
        result_publication |= ResultPublication.DUAL

    log = build_activity_run(activity, is_root=transport.rank == 0)
    try:
        run_t0 = perf_counter() if log.enabled else None
        log.emit(
            BootstrapStart(
                run_id=log.config.run_id,
                label=log.config.label,
                n_bootstrap=n_bootstrap,
                base_seed=base_seed,
                resampling="multiplier",
                tolerance=tolerance,
                max_iterations=max_iterations,
                min_iterations=min_iterations,
                n_obs=prep.N,
                n_simulations=prep.S,
                n_parameters=K,
                n_agents=prep.n_agents,
                master_backend=resolved_master_backend,
                formulation=_object_name(formulation),
                cut_policy=_object_name(cut_policy),
                result_publication=_publication_label(result_publication),
                transport=type(transport).__name__,
                warm_start=theta_init is not None,
                rank=transport.rank,
                world_size=transport.size,
                activity_level=log.config.level,
                dual_store_dir=(
                    str(root_dual_store_dir)
                    if root_dual_store_dir is not None
                    else None
                ),
            )
        )

        # The oracle and price surface are resolved once and shared across every
        # rep; only the weights differ per rep.
        collective_call(
            transport, lambda: oracle.setup(transport, prep.local_ids)
        )
        try:
            price_resolution = resolve(
                oracle,
                surface="price",
                default_name="price",
                optimized_name="price_batch",
                default_func=Oracle.price,
                optimized_func=Oracle.price_batch,
                transport=transport,
            )
            thetas = np.zeros((n_bootstrap, K), dtype=np.float64)
            converged = np.zeros(n_bootstrap, dtype=bool)
            run_iterations = 0
            run_duals_stored = 0
            run_dual_store_dir: Path | None = None
            gap_tally = GapTally()
            # Reps are independent, so a bounded live window gives the same draws
            # without hosting every per-rep master at once.
            wave_limit = min(live_cap, n_bootstrap)
            for start in range(0, len(rep_ids), wave_limit):
                wave_rep_ids = rep_ids[start : start + wave_limit]
                c_thetas, normalizers = _bootstrap_wave_c_theta_and_normalizers(
                    prep,
                    wave_rep_ids,
                    base_seed=base_seed,
                    transport=transport,
                )
                replicas: list[_Replica] = []
                for slot, rep_id in enumerate(wave_rep_ids):
                    replicas.append(
                        _build_distributed_replica(
                            rep_id,
                            prep=prep,
                            model=model,
                            c_theta=c_thetas[slot],
                            normalizer=float(normalizers[slot]),
                            price_resolution=price_resolution,
                            transport=transport,
                            owner_rank=int(owners[rep_id]),
                            backend=resolved_master_backend,
                            master_params=master_params,
                            tolerance=tolerance,
                            # Deep-copy per rep: a policy carries per-(agent_id,
                            # bundle_key) state, so a shared instance would let
                            # same-keyed cuts from different reps age each other.
                            # The caller's instance is a template, never mutated.
                            cut_policy=(
                                None
                                if cut_policy is None
                                else copy.deepcopy(cut_policy)
                            ),
                            result_publication=result_publication,
                            theta_init=theta_init,
                            warm_cuts=warm_cuts,
                            base_seed=base_seed,
                        )
                    )
                wave = _run_bfold(
                    replicas,
                    oracle=oracle,
                    transport=transport,
                    owners=owners[np.asarray(wave_rep_ids, dtype=np.int64)],
                    K=K,
                    parameters=parameters,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                    min_iterations=min_iterations,
                    iteration_callback=iteration_callback,
                    gap_tally=gap_tally,
                    dual_store_dir=root_dual_store_dir,
                    activity=log,
                    round_offset=run_iterations,
                )
                for slot, rep_id in enumerate(wave_rep_ids):
                    thetas[rep_id] = wave.thetas[slot]
                    converged[rep_id] = wave.converged[slot]
                run_iterations += wave.iterations
                run_duals_stored += wave.n_duals_stored
                if wave.dual_store_dir is not None:
                    run_dual_store_dir = wave.dual_store_dir
            certification = gap_tally.certify(transport)
        finally:
            oracle.teardown()

        log.emit(
            BootstrapFinal(
                run_id=log.config.run_id,
                label=log.config.label,
                n_requested=n_bootstrap,
                n_persisted=0,
                n_computed=n_bootstrap,
                n_nonconverged=n_bootstrap - int(np.count_nonzero(converged)),
                n_converged=int(np.count_nonzero(converged)),
                total_super_steps=run_iterations,
                wall_seconds=(perf_counter() - run_t0 if run_t0 is not None else None),
                n_duals_stored=run_duals_stored,
            )
        )
    finally:
        log.close()

    return BootstrapResult(
        thetas=thetas,
        converged=converged,
        parameters=parameters,
        iterations=run_iterations,
        dual_store_dir=run_dual_store_dir,
        n_duals_stored=run_duals_stored,
        u_samples=None,
        duals=None,
        metadata={"certification": certification_metadata(certification)},
    )
