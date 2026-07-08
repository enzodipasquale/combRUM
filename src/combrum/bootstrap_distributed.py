"""Distributed multiplier bootstrap over a public ``Model``."""

from __future__ import annotations

import copy
import hashlib
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
from combrum.engine.distributed_context import (
    DistributedObservedPrep,
    build_distributed_fit_context,
    prepare_distributed_observed,
)
from combrum.formulations import NSlack
from combrum.interface_resolution import (
    Resolution,
    needs_conformance_guard,
    price_demands,
    resolve,
)
from combrum.masters import resolve_master_backend
from combrum.model import Model
from combrum.oracle import Oracle
from combrum.parameters import Parameters
from combrum.policies import CutPolicy
from combrum.randomness import bootstrap_multipliers
from combrum.result import BootstrapResult
from combrum.rowgen import (
    Contribution,
    MaxContribution,
    MaxReduced,
    Reduced,
    RowGenStep,
    StepOutcome,
)
from combrum.transport.base import CutRow, Transport, _cut_row_nbytes


class _Closable(Protocol):
    def close(self) -> None: ...


_DEFAULT_MAX_LIVE_REPS = 64
_BOOTSTRAP_OBS_BLOCK_ELEMENTS = 1_000_000
_CUT_EXCHANGE_BLOCK_ELEMENTS = 1_000_000
_REQUIRED_TRANSPORT_OVERRIDES = (
    "owner_broadcast",
    "route_agent_values_batched",
)


def _require_distributed_transport(transport: Transport) -> None:
    missing = [
        name
        for name in _REQUIRED_TRANSPORT_OVERRIDES
        if getattr(type(transport), name, None) is getattr(Transport, name)
    ]
    if missing:
        names = ", ".join(missing)
        raise NotImplementedError(
            "bootstrap_distributed requires a transport implementing"
            f" {names}; use SerialTransport, MpiTransport,"
            " combrum.transport.LocalCluster, or another transport with the"
            " batched distributed-bootstrap primitives"
        )


def _coerce_max_live_reps(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("max_live_reps must be an integer >= 1; got bool")
    try:
        cap = operator.index(value)
    except TypeError as exc:
        raise TypeError(
            f"max_live_reps must be an integer >= 1; got {type(value).__name__}"
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


def _warm_cut_token(
    rows: tuple[CutRow, ...] | None,
) -> tuple[bool, tuple[int, str] | None, str, str]:
    if rows is None:
        return (True, None, "", "")
    h = hashlib.sha256()
    h.update(len(rows).to_bytes(8, "little", signed=False))
    for idx, row in enumerate(rows):
        if not isinstance(row, CutRow):
            return (
                False,
                None,
                "TypeError",
                f"warm_cuts[{idx}] must be CutRow; got {type(row).__name__}",
            )
        h.update(int(row.rep_id).to_bytes(8, "little", signed=True))
        h.update(int(row.agent_id).to_bytes(8, "little", signed=True))
        h.update(np.float64(row.epsilon).tobytes())
        phi = np.ascontiguousarray(row.phi, dtype=np.float64)
        h.update(phi.shape[0].to_bytes(8, "little", signed=False))
        h.update(phi.tobytes())
        h.update(len(row.bundle_key).to_bytes(8, "little", signed=False))
        h.update(row.bundle_key)
    return (True, (len(rows), h.hexdigest()), "", "")


def _agree_warm_cuts(
    value: Sequence[CutRow] | None, transport: Transport
) -> tuple[CutRow, ...] | None:
    try:
        rows = None if value is None else tuple(value)
    except TypeError as exc:
        rows = None
        local = (
            False,
            None,
            "TypeError",
            f"warm_cuts must be a sequence of CutRow; {type(exc).__name__}: {exc}",
        )
    else:
        local = _warm_cut_token(rows)
    if transport.size == 1:
        if not local[0]:
            if local[2] == "TypeError":
                raise TypeError(local[3])
            raise ValueError(local[3])
        return rows

    with transport.collective():
        root = transport.bcast(local if transport.rank == 0 else None, root=0)
        if local != root:
            raise ValueError("warm_cuts must match on every rank")
        if not local[0]:
            if local[2] == "TypeError":
                raise TypeError(local[3])
            raise ValueError(local[3])
    return rows


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
    return [row._replace(rep_id=rep_id) for row in rows]


def batched_reduce(
    transport: Transport,
    contributions: Mapping[int, Contribution],
    owners: np.ndarray,
) -> list[Reduced | None]:
    """Reduce every live replication's contribution in one batched super-step.

    ``contributions`` maps each live slot to that rank's local
    :class:`~combrum.rowgen.Contribution`; ``owners`` is the wave's owner vector
    (identical on every rank). Returns a length-``B`` list indexed by wave slot,
    holding the :class:`~combrum.rowgen.Reduced` for each live slot
    (``None`` otherwise).

    The current distributed bootstrap is NSlack-only, so the live set must emit
    ``MaxContribution``. One ``batched_max`` and one ``exchange_cuts`` cover the
    current live set; retired slots are absent from both payloads.
    """
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
        worsts = np.zeros(len(live_slots), dtype=np.float64)
        all_rows: list[CutRow] = []
        for pos, slot in enumerate(live_slots):
            contribution = contributions[slot]
            _require_kind(contribution, MaxContribution, slot)
            worsts[pos] = contribution.worst
            all_rows.extend(_restamp(contribution.local_rows, pos))
        reduced, _ = _reduce_live_max(transport, live_slots, owners, worsts, all_rows)
        return reduced

    raise AssertionError(
        f"unhandled contribution type: {type(sample).__name__};"
        " distributed bootstrap expects MaxContribution"
    )


def _reduce_live_max(
    transport: Transport,
    live_slots: Sequence[int],
    owners: np.ndarray,
    worsts: np.ndarray,
    all_rows: Sequence[CutRow],
    *,
    local_row_nbytes: int | None = None,
) -> tuple[list[Reduced | None], int | None]:
    live_slots = tuple(int(slot) for slot in live_slots)
    if not live_slots:
        raise AssertionError("cannot reduce an empty live-replication set")
    B = int(owners.shape[0])
    live_owners = np.asarray([owners[slot] for slot in live_slots], dtype=np.int64)
    # The row-width bookkeeping rides the batched max as one extra lane, so
    # agreeing it costs no additional round.
    lanes = (
        worsts
        if local_row_nbytes is None
        else np.append(worsts, float(local_row_nbytes))
    )
    global_lanes = np.asarray(transport.batched_max(lanes), dtype=np.float64)
    global_worsts = global_lanes[: len(live_slots)]
    observed_nbytes = (
        None if local_row_nbytes is None else int(global_lanes[len(live_slots)])
    )
    received = transport.exchange_cuts(all_rows, live_owners)
    rows_by_pos: dict[int, list[CutRow]] = {}
    for row in received:
        rows_by_pos.setdefault(row.rep_id, []).append(row)
    reduced: list[Reduced | None] = [None] * B
    for pos, slot in enumerate(live_slots):
        reduced[slot] = MaxReduced(
            global_worst=float(global_worsts[pos]),
            received_rows=tuple(_restamp(rows_by_pos.get(pos, ()), slot)),
        )
    return reduced, observed_nbytes


def _store_dual(writer: DualStoreWriter, rep_id: int, dual: DualSolution | None) -> int:
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


def _require_kind(contribution: Contribution, expected: type, slot: int) -> None:
    if not isinstance(contribution, expected):
        raise AssertionError(
            "batched_reduce: mixed contribution kinds across the live set"
            f" (slot {slot} is {type(contribution).__name__}, expected"
            f" {expected.__name__}); one formulation drives the whole"
            " bootstrap"
        )


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


def _cut_exchange_block_size(
    n_live: int, *, row_nbytes: int, max_scheduled_ids: int
) -> int:
    # Pure sizing: the agreed shard-size bound is wave-constant, so the wave
    # loop agrees it once instead of re-reducing it per block.
    if not n_live:
        return 0
    budget_bytes = max(1, _CUT_EXCHANGE_BLOCK_ELEMENTS * np.dtype(np.float64).itemsize)
    bytes_per_rep = max(1, int(max_scheduled_ids) * max(1, int(row_nbytes)))
    return max(
        1,
        min(
            int(n_live),
            max(1, budget_bytes // bytes_per_rep),
        ),
    )


def _observed_cut_row_nbytes(
    rows: Sequence[CutRow],
    *,
    transport: Transport,
) -> int | None:
    local = max((_cut_row_nbytes(row) for row in rows), default=0)
    observed = (
        local if transport.size == 1 else int(transport.allreduce_max(float(local)))
    )
    return observed if observed > 0 else None


@dataclass
class _Replica:
    """One replication's per-rep state: formulation, resolved price surface,
    and the local-id shard it prices each iteration."""

    rep_id: int
    formulation: RowGenStep
    price_resolution: Resolution
    scheduled_local_ids: np.ndarray
    master_backend: _Closable | None = None
    initial_state: object | None = None
    initial_full_u: dict[int, float] | None = None
    closed: bool = False


def _dispose_replicas(replicas: Sequence[_Replica]) -> str | None:
    errors: list[str] = []
    for replica in replicas:
        if replica.closed:
            continue
        formulation = getattr(replica, "formulation", None)
        try:
            if formulation is not None:
                formulation.dispose()
        except Exception as exc:
            errors.append(
                f"rep {replica.rep_id} formulation.dispose:"
                f" {type(exc).__name__}: {exc}"
            )
        finally:
            master_backend = getattr(replica, "master_backend", None)
            if master_backend is not None:
                try:
                    master_backend.close()
                except Exception as exc:
                    errors.append(
                        f"rep {replica.rep_id} master_backend.close:"
                        f" {type(exc).__name__}: {exc}"
                    )
                finally:
                    replica.master_backend = None
        replica.closed = True
    return "; ".join(errors) if errors else None


def _raise_dispose_error(dispose_error: str | None, transport: Transport) -> None:
    if transport.size > 1:
        with transport.collective():
            if dispose_error is not None:
                raise RuntimeError(dispose_error)
        return
    if dispose_error is not None:
        raise RuntimeError(dispose_error)


@dataclass(frozen=True)
class _PublishedMasterState:
    theta: np.ndarray
    objective: float
    n_installed: int
    progressed: int


def _nslack(replica: _Replica) -> NSlack:
    formulation = replica.formulation
    if not isinstance(formulation, NSlack):
        raise TypeError(
            "distributed bootstrap currently supports NSlack replicas only;"
            f" got {type(formulation).__name__}"
        )
    return formulation


def _pack_master_state(state: object, K: int) -> np.ndarray:
    row = np.zeros(int(K) + 4, dtype=np.float64)
    row[:K] = np.asarray(getattr(state, "theta"), dtype=np.float64)
    row[K] = float(getattr(state, "objective"))
    row[K + 1] = float(getattr(state, "n_installed"))
    row[K + 2] = float(getattr(state, "progressed"))
    row[K + 3] = 1.0
    return row


def _unpack_master_state(row: np.ndarray, K: int) -> _PublishedMasterState:
    arr = np.asarray(row, dtype=np.float64)
    if arr.shape != (int(K) + 4,):
        raise ValueError(
            f"master-state row has shape {arr.shape}; expected ({int(K) + 4},)"
        )
    if float(arr[K + 3]) != 1.0:
        raise RuntimeError("missing owner master-state row in batched publication")
    return _PublishedMasterState(
        theta=arr[:K].copy(),
        objective=float(arr[K]),
        n_installed=int(round(float(arr[K + 1]))),
        progressed=int(round(float(arr[K + 2]))),
    )


def _publish_nslack_states(
    replicas: Sequence[_Replica],
    slots: Sequence[int],
    *,
    owner_states: Sequence[object | None],
    local_us: Mapping[int, Mapping[int, float]],
    owners: np.ndarray,
    transport: Transport,
    K: int,
    bump_iteration: bool,
) -> None:
    slot_ids = [int(slot) for slot in slots]
    if not slot_ids:
        return
    owners_live = np.asarray([owners[slot] for slot in slot_ids], dtype=np.int64)
    block = np.zeros((len(slot_ids), int(K) + 4), dtype=np.float64)
    with transport.collective():
        for pos, slot in enumerate(slot_ids):
            state = owner_states[slot]
            if state is not None:
                block[pos] = _pack_master_state(state, K)

    published = np.asarray(transport.owner_broadcast(block, owners_live))
    for pos, slot in enumerate(slot_ids):
        formulation = _nslack(replicas[slot])
        formulation._adopt_owner_state(
            _unpack_master_state(published[pos], K),
            local_u=local_us.get(slot, {}),
            full_u=formulation._distributed_owner_u(),
            bump_iteration=bump_iteration,
        )


def _route_nslack_us(
    replicas: Sequence[_Replica],
    slots: Sequence[int],
    *,
    full_us: Mapping[int, Mapping[int, float] | None],
    owners: np.ndarray,
    transport: Transport,
) -> dict[int, dict[int, float]]:
    slot_ids = [int(slot) for slot in slots]
    if not slot_ids:
        return {}

    first = _nslack(replicas[slot_ids[0]])
    local_ids, n_agents = first._distributed_route_spec()
    owners_live = np.asarray([owners[slot] for slot in slot_ids], dtype=np.int64)
    payload: dict[int, Mapping[int, float] | None] = {}
    for pos, slot in enumerate(slot_ids):
        formulation = _nslack(replicas[slot])
        if transport.rank == int(owners[slot]):
            values = full_us.get(slot)
            payload[pos] = values
            formulation._distributed_set_owner_u(values)
    routed = transport.route_agent_values_batched(
        payload if payload else None,
        local_ids,
        owners=owners_live,
        n_agents=n_agents,
    )
    return {slot_ids[pos]: values for pos, values in routed.items()}


def _run_replica_wave(
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
    """Drive a bounded wave of bootstrap replicas to convergence."""
    n_reps = len(replicas)
    with transport.collective():
        if any(not isinstance(replica.formulation, NSlack) for replica in replicas):
            raise TypeError(
                "distributed bootstrap currently supports NSlack replicas only"
            )
    # The live mask is rank-identical; all collectives loop over this set.
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
    cut_row_nbytes_hint: int | None = None

    # Duals are streamed one at a time from rank 0.
    writer: DualStoreWriter | None = None
    stored = 0
    thetas = np.zeros((n_reps, K), dtype=np.float64)
    finalized = np.zeros(n_reps, dtype=bool)
    rep_objectives = np.full(n_reps, np.nan, dtype=np.float64)
    rep_active_cuts = np.zeros(n_reps, dtype=np.int64)

    dispose_error: str | None = None

    def _remember_dispose_error(error: str | None) -> None:
        nonlocal dispose_error
        if error:
            dispose_error = error if dispose_error is None else f"{dispose_error}; {error}"

    def _finalize_slots(slot_ids: Sequence[int]) -> None:
        nonlocal stored
        for raw_slot in slot_ids:
            slot = int(raw_slot)
            if finalized[slot]:
                continue
            replica = replicas[slot]
            result = replica.formulation.result()
            thetas[slot] = result.theta_hat
            rep_objectives[slot] = float(result.objective)
            rep_active_cuts[slot] = int(result.n_active_cuts)
            if writer is not None:
                with transport.collective():
                    if transport.rank == 0:
                        stored += _store_dual(writer, replica.rep_id, result.dual)
            finalized[slot] = True
            _remember_dispose_error(_dispose_replicas((replica,)))

    try:
        if dual_store_dir is not None:
            with transport.collective():
                writer = DualStoreWriter(dual_store_dir)
        needs_prereduce_guard = transport.size > 1 or any(
            needs_conformance_guard(
                replica.price_resolution,
                getattr(replica.formulation, "_features_res", None),
            )
            for replica in replicas
        )
        # Shard sizes never change within a wave: agree the block-sizing
        # bound once here rather than once per block.
        local_max_scheduled = max(
            (
                int(np.asarray(replica.scheduled_local_ids).size)
                for replica in replicas
            ),
            default=0,
        )
        max_scheduled = (
            local_max_scheduled
            if transport.size == 1
            else int(transport.allreduce_max(float(local_max_scheduled)))
        )
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

            live_slot_ids = tuple(int(slot) for slot in live_slots)

            price_seconds = 0.0 if log_details else None
            comm_seconds = 0.0 if log_details else None
            master_seconds = 0.0 if log_details else None
            retired_slots: list[int] = []
            max_gap: float | None = None

            block_start = 0
            while block_start < len(live_slot_ids):
                def _price_slots(
                    slot_ids: Sequence[int],
                ) -> tuple[np.ndarray, list[CutRow]]:
                    worsts = np.zeros(len(slot_ids), dtype=np.float64)
                    rows: list[CutRow] = []
                    for pos, slot in enumerate(slot_ids):
                        replica = replicas[slot]
                        theta = replica.formulation.solve()
                        demands: Mapping[int, Demand] = price_demands(
                            replica.price_resolution,
                            theta,
                            replica.scheduled_local_ids,
                        )
                        if gap_tally is not None:
                            _observe_demands(gap_tally, demands)
                        contribution = replica.formulation.contribute(demands)
                        _require_kind(contribution, MaxContribution, slot)
                        worsts[pos] = contribution.worst
                        rows.extend(_restamp(contribution.local_rows, pos))
                        del demands
                    return worsts, rows

                def _guarded_price(
                    slot_ids: Sequence[int],
                ) -> tuple[np.ndarray, list[CutRow]]:
                    if needs_prereduce_guard:
                        with transport.collective():
                            return _price_slots(slot_ids)
                    return _price_slots(slot_ids)

                # Price bounded live blocks; no dense (B, n_agents) payload and
                # no all-live cut list.
                price_t0 = perf_counter() if log_details else None
                if cut_row_nbytes_hint is None:
                    block_slots: list[int] = []
                    worst_parts: list[np.ndarray] = []
                    rows: list[CutRow] = []
                    while block_start + len(block_slots) < len(live_slot_ids):
                        probe_start = block_start + len(block_slots)
                        next_slot = live_slot_ids[probe_start : probe_start + 1]
                        next_worsts, next_rows = _guarded_price(next_slot)
                        block_slots.extend(int(slot) for slot in next_slot)
                        worst_parts.append(next_worsts)
                        # The single-slot probe stamped its rows at position 0;
                        # re-key them to the slot's position in the block the
                        # batched exchange routes by.
                        rows.extend(_restamp(next_rows, len(block_slots) - 1))
                        observed_row_nbytes = _observed_cut_row_nbytes(
                            next_rows, transport=transport
                        )
                        if observed_row_nbytes is not None:
                            cut_row_nbytes_hint = observed_row_nbytes
                            break
                    block_slot_ids = tuple(block_slots)
                    worsts = np.concatenate(worst_parts)
                else:
                    cut_block = _cut_exchange_block_size(
                        len(live_slot_ids) - block_start,
                        row_nbytes=cut_row_nbytes_hint,
                        max_scheduled_ids=max_scheduled,
                    )
                    block_slot_ids = live_slot_ids[block_start : block_start + cut_block]
                    worsts, rows = _guarded_price(block_slot_ids)
                if price_seconds is not None and price_t0 is not None:
                    price_seconds += perf_counter() - price_t0

                comm_t0 = perf_counter() if log_details else None
                reduced, observed_row_nbytes = _reduce_live_max(
                    transport,
                    block_slot_ids,
                    owners,
                    worsts,
                    rows,
                    local_row_nbytes=max(
                        (_cut_row_nbytes(row) for row in rows), default=0
                    ),
                )
                del rows
                if observed_row_nbytes:
                    cut_row_nbytes_hint = max(
                        cut_row_nbytes_hint or 0, observed_row_nbytes
                    )
                if comm_seconds is not None and comm_t0 is not None:
                    comm_seconds += perf_counter() - comm_t0

                master_t0 = perf_counter() if log_details else None
                block_outcomes: dict[int, StepOutcome] = {}
                for slot in block_slot_ids:
                    replica = replicas[slot]
                    reduced_slot = reduced[slot]
                    if reduced_slot is None:
                        raise AssertionError(f"missing reduced payload for slot {slot}")
                    outcome: StepOutcome = replica.formulation.finalise(reduced_slot)
                    block_outcomes[slot] = outcome
                    violation = float(outcome.violation)
                    max_gap = violation if max_gap is None else max(max_gap, violation)
                    if rep_gaps is not None:
                        rep_gaps[slot] = violation
                owner_states: list[object | None] = [None] * n_reps
                full_us: dict[int, Mapping[int, float] | None] = {}
                with transport.collective():
                    for slot in block_slot_ids:
                        formulation = _nslack(replicas[slot])
                        state, full_u = formulation._distributed_apply_owner_step(
                            block_outcomes[slot].install_payload
                        )
                        owner_states[slot] = state
                        full_us[slot] = full_u
                local_us = _route_nslack_us(
                    replicas,
                    list(block_slot_ids),
                    full_us=full_us,
                    owners=owners,
                    transport=transport,
                )
                _publish_nslack_states(
                    replicas,
                    list(block_slot_ids),
                    owner_states=owner_states,
                    local_us=local_us,
                    owners=owners,
                    transport=transport,
                    K=K,
                    bump_iteration=True,
                )
                block_retired_slots: list[int] = []
                for slot in block_slot_ids:
                    violation = float(block_outcomes[slot].violation)
                    if violation <= tolerance and iterations >= convergence_floor:
                        converged[slot] = True
                        live[slot] = False
                        retired_slots.append(slot)
                        block_retired_slots.append(slot)
                        if rep_iterations is not None:
                            rep_iterations[slot] = iterations
                if master_seconds is not None and master_t0 is not None:
                    master_seconds += perf_counter() - master_t0
                _finalize_slots(block_retired_slots)
                block_start += len(block_slot_ids)
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
        _finalize_slots(range(n_reps))
        for slot, replica in enumerate(replicas):
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
                        objective=float(rep_objectives[slot]),
                        active_cuts=int(rep_active_cuts[slot]),
                    )
                )
    finally:
        _remember_dispose_error(_dispose_replicas(replicas))
    stored_total = (
        int(transport.bcast(stored if transport.rank == 0 else None, root=0))
        if writer is not None
        else 0
    )
    _raise_dispose_error(dispose_error, transport)

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


def _bootstrap_local_rows(
    prep: DistributedObservedPrep,
    *,
    base_seed: int,
    rep_ids: Sequence[int],
    obs_start: int = 0,
    obs_stop: int | None = None,
) -> np.ndarray:
    """Observation-keyed bootstrap rows for one live wave.

    Each row is keyed by the global observation id. Columns are grouped by
    replication as ``[raw_weight, c_theta_0, ..., c_theta_K]``. This keeps the
    global reduction row-distribution invariant while still reducing the whole
    live wave in one call.
    """
    rep_ids = tuple(int(rep_id) for rep_id in rep_ids)
    obs_stop = prep.owned_obs.size if obs_stop is None else int(obs_stop)
    obs_start = int(obs_start)
    if not 0 <= obs_start <= obs_stop <= prep.owned_obs.size:
        raise ValueError(
            "observation block must satisfy"
            f" 0 <= start <= stop <= {prep.owned_obs.size};"
            f" got start={obs_start}, stop={obs_stop}"
        )
    owned_obs = prep.owned_obs[obs_start:obs_stop]
    phi_obs = prep.phi_obs_local[obs_start:obs_stop]
    width = len(rep_ids) * (prep.K + 1)
    local = np.empty((owned_obs.size, width), dtype=np.float64)
    for slot, rep_id in enumerate(rep_ids):
        raw = bootstrap_multipliers(base_seed, rep_id, owned_obs)
        offset = slot * (prep.K + 1)
        local[:, offset] = raw
        np.multiply(
            raw[:, None],
            phi_obs,
            out=local[:, offset + 1 : offset + 1 + prep.K],
        )
        local[:, offset + 1 : offset + 1 + prep.K] *= -float(prep.S)
    return local


def _finish_bootstrap_reduction(
    prep: DistributedObservedPrep, reduced: np.ndarray, *, n_reps: int
) -> tuple[np.ndarray, np.ndarray]:
    expected_shape = (int(n_reps), prep.K + 1)
    reduced = np.asarray(reduced, dtype=np.float64)
    if reduced.size != expected_shape[0] * expected_shape[1]:
        raise ValueError(
            "bootstrap observed reduction returned shape"
            f" {reduced.shape}; expected {expected_shape}"
        )
    reduced = reduced.reshape(expected_shape)
    normalizers = np.asarray(reduced[:, 0], dtype=np.float64)
    if np.any(normalizers <= 0.0):
        bad = normalizers[normalizers <= 0.0]
        raise ValueError(
            f"bootstrap multiplier normalizer must be positive; got {bad.tolist()}"
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
    rep_ids = tuple(int(rep_id) for rep_id in rep_ids)
    if not rep_ids:
        return (
            np.empty((0, prep.K), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    elems_per_rep = prep.N * (prep.K + 1)
    block = (
        len(rep_ids)
        if elems_per_rep == 0
        else max(1, _BOOTSTRAP_OBS_BLOCK_ELEMENTS // elems_per_rep)
    )
    c_thetas = np.empty((len(rep_ids), prep.K), dtype=np.float64)
    normalizers = np.empty(len(rep_ids), dtype=np.float64)
    for start in range(0, len(rep_ids), block):
        chunk = rep_ids[start : start + block]
        obs_block = max(
            1,
            _BOOTSTRAP_OBS_BLOCK_ELEMENTS // (len(chunk) * (prep.K + 1)),
        )
        reduced = np.zeros(len(chunk) * (prep.K + 1), dtype=np.float64)
        for obs_start in range(0, prep.N, obs_block):
            obs_stop = min(obs_start + obs_block, prep.N)
            local_start = int(np.searchsorted(prep.owned_obs, obs_start, side="left"))
            local_stop = int(np.searchsorted(prep.owned_obs, obs_stop, side="left"))
            local = _bootstrap_local_rows(
                prep,
                base_seed=base_seed,
                rep_ids=chunk,
                obs_start=local_start,
                obs_stop=local_stop,
            )
            reduced += np.asarray(
                transport.sum_reproducible(
                    local,
                    prep.owned_obs[local_start:local_stop],
                ),
                dtype=np.float64,
            )
        c_chunk, n_chunk = _finish_bootstrap_reduction(
            prep,
            reduced,
            n_reps=len(chunk),
        )
        stop = start + len(chunk)
        c_thetas[start:stop] = c_chunk
        normalizers[start:stop] = n_chunk
    return c_thetas, normalizers


def _bootstrap_slack_coef(
    *, n_observations: int, base_seed: int, rep_id: int, normalizer: float
) -> Callable[[int], float]:
    # One vectorized draw over the observation axis; the closure keeps O(N)
    # memory where a dense (N*S,) coefficient array would not, and the lazy
    # distributed masters read only the agents that actually cut.
    scale = float(n_observations) / float(normalizer)
    raw = bootstrap_multipliers(
        base_seed, rep_id, np.arange(int(n_observations), dtype=np.int64)
    )

    def _coef(agent_id: int) -> float:
        return float(raw[int(agent_id) % raw.size]) * scale

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
    setup_collective: bool = True,
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
        guard_master=setup_collective,
    )
    try:
        initial_state: object | None = None
        initial_full_u: dict[int, float] | None = None
        if setup_collective:
            formulation.setup(built.ctx)
        else:
            initial_state, initial_full_u = formulation._distributed_setup_owner_local(
                built.ctx
            )
    except Exception:
        try:
            formulation.dispose()
        except Exception:
            pass
        if built.ctx.master_backend is not None:
            try:
                built.ctx.master_backend.close()
            except Exception:
                pass
        raise
    return _Replica(
        rep_id=rep_id,
        formulation=formulation,
        price_resolution=price_resolution,
        scheduled_local_ids=prep.local_ids,
        master_backend=built.ctx.master_backend,
        initial_state=initial_state,
        initial_full_u=initial_full_u,
    )


def _node_interleaved_rank_order(transport: Transport) -> np.ndarray:
    rank_ids = np.array([transport.rank], dtype=np.int64)
    cols: list[np.ndarray] = []
    for value in (
        float(transport.node.node_rank),
        float(transport.node.node_id),
        float(transport.rank),
    ):
        gathered = transport.gather_agent_values(
            np.array([value], dtype=np.float64),
            rank_ids,
            transport.size,
            root=0,
        )
        column = transport.bcast(gathered if transport.rank == 0 else None, root=0)
        cols.append(np.asarray(column, dtype=np.float64))
    table = np.column_stack(cols)
    if table.shape != (transport.size, 3):
        raise ValueError(
            "rank-topology reduction returned shape"
            f" {table.shape}; expected ({transport.size}, 3)"
        )
    order = sorted(
        (
            (int(row[0]), int(row[1]), int(row[2]))
            for row in table
        )
    )
    return np.asarray([rank for _node_rank, _node_id, rank in order], dtype=np.int64)


def _owner_vector(n_bootstrap: int, transport: Transport) -> np.ndarray:
    """Node-interleaved replication owner map.

    Cycling by node rank first spreads small batches across nodes before
    placing multiple masters on the same node.
    """
    order = _node_interleaved_rank_order(transport)
    return order[np.arange(n_bootstrap, dtype=np.int64) % order.size]


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
    warm_start: object | None = None,
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

    The model must use ``model.formulation=NSlack`` and provide a distributed
    observed-feature surface: either ``model.observed_features`` or
    ``model.features`` must define
    ``observed_features_batch(observation_ids) -> float64[N_local, K]``.
    Optional ``setup_observed(transport, observation_ids)`` and
    ``setup_pricing_agents(transport, local_ids)`` hooks run once before the
    loop. ``model.oracle.setup(transport, local_ids)`` also runs once, and then
    pricing is called only on this rank's local simulated-agent ids.

    This entry point does not accept dense ``Data`` or ``observed_bundles`` and
    does not collect dense per-replication slack arrays. Public controls
    (including ``max_live_reps``, ``master_params``, ``cut_policy``, and
    warm-start inputs) must be rank-uniform. ``warm_start`` may be any object
    with a finite ``theta_hat`` vector; ``warm_cuts`` must be a rank-identical
    sequence of :class:`combrum.CutRow`. When ``dual_store_dir`` is provided,
    rank 0 writes one dual file per replication and ``BootstrapResult.duals``
    remains ``None``.

    ``metadata["certification"]`` is an aggregate pricing-exactness report
    across the run; per-replication convergence remains in ``converged``.
    ``iteration_callback``, when supplied, runs once per wave-local iteration
    before pricing; the iteration index resets for each bounded live-replication
    wave.
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
    if transport.size == 1:
        _require_distributed_transport(transport)
    else:
        with transport.collective():
            _require_distributed_transport(transport)
    master_backend = str(
        agree_public_choice(
            "master_backend",
            master_backend,
            transport,
            choices=("auto", "gurobi", "highs"),
        )
    )
    n_bootstrap = agree_public_int("n_bootstrap", n_bootstrap, transport, lower=1)
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
    warm_cuts = _agree_warm_cuts(warm_cuts, transport)
    cut_policy = require_public_object_agreement("cut_policy", cut_policy, transport)

    prep = prepare_distributed_observed(
        model,
        n_observations=n_observations,
        n_simulations=n_simulations,
        transport=transport,
    )
    K = prep.K
    # Warm-start theta is a public distributed input: every rank must agree
    # before any owner-local context validates or consumes it.
    theta_init = agree_public_optional_theta("warm_start", warm_start, transport, K=K)

    # Several reps may share an owner; rep_id keeps their rows separate.
    owners = _owner_vector(n_bootstrap, transport)
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
            lambda: Path(dual_store_dir) if transport.rank == 0 else None,
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
        collective_call(transport, lambda: oracle.setup(transport, prep.local_ids))
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
                wave_owns_replicas = False
                try:
                    feature_tokens: list[object] = []
                    with transport.collective():
                        for slot, rep_id in enumerate(wave_rep_ids):
                            replica = _build_distributed_replica(
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
                                # Policies carry row state and cannot be shared
                                # across independent replicas.
                                cut_policy=(
                                    None
                                    if cut_policy is None
                                    else copy.deepcopy(cut_policy)
                                ),
                                result_publication=result_publication,
                                theta_init=theta_init,
                                warm_cuts=warm_cuts,
                                base_seed=base_seed,
                                setup_collective=False,
                            )
                            replicas.append(replica)
                            feature_tokens.append(
                                _nslack(replica)._distributed_feature_token()
                            )
                    require_public_object_agreement(
                        "bootstrap replica feature tokens",
                        tuple(feature_tokens),
                        transport,
                    )
                    wave_owners = owners[np.asarray(wave_rep_ids, dtype=np.int64)]
                    initial_states = [replica.initial_state for replica in replicas]
                    initial_full_us = {
                        slot: replica.initial_full_u
                        for slot, replica in enumerate(replicas)
                    }
                    initial_us = _route_nslack_us(
                        replicas,
                        range(len(replicas)),
                        full_us=initial_full_us,
                        owners=wave_owners,
                        transport=transport,
                    )
                    _publish_nslack_states(
                        replicas,
                        range(len(replicas)),
                        owner_states=initial_states,
                        local_us=initial_us,
                        owners=wave_owners,
                        transport=transport,
                        K=K,
                        bump_iteration=False,
                    )
                    for replica in replicas:
                        replica.initial_state = None
                        replica.initial_full_u = None
                    wave_owns_replicas = True
                    wave = _run_replica_wave(
                        replicas,
                        oracle=oracle,
                        transport=transport,
                        owners=wave_owners,
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
                finally:
                    if not wave_owns_replicas:
                        _dispose_replicas(replicas)
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
