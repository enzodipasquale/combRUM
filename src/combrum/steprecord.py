"""Frozen, typed per-iteration snapshots of a row-gen step's pre-filter inputs.

A :class:`StepRecord` captures the quantities a formulation's filters read in
one iteration, over their full pre-filter domain, at the read site. Each
capture is keyed by the global ``agent_id`` (and bundle key bytes where a row
identity is involved) so runs that shard agents differently produce comparable
records. The trace sink is a no-op when ``None``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from combrum.demand import Demand


@dataclass(frozen=True)
class PricedReducedCost:
    """One priced agent's reduced cost before the emit threshold.

    ``rc = payoff - u_a``, captured for every priced agent, including those
    with ``rc <= tolerance`` that emit no row. ``bundle_key`` is the chosen
    bundle's canonical bytes.
    """

    agent_id: int
    bundle_key: bytes
    rc: float


@dataclass(frozen=True)
class AdmitViolation:
    """One received candidate's violation before ``policy.admit``.

    ``violation = phi . theta + eps - u_a``, the pre-admit signal handed to the
    admit policy, captured for every received candidate. Keyed by ``agent_id``
    + ``bundle_key``.
    """

    agent_id: int
    bundle_key: bytes
    violation: float


@dataclass(frozen=True)
class PurgeInput:
    """The dual and slack the retirement policy reads for one installed cut.

    Per installed ``(agent_id, bundle_key)``. ``dual`` / ``slack`` are ``None``
    when the last solve held no reading for that key (a just-admitted cut).
    """

    agent_id: int
    bundle_key: bytes
    dual: float | None
    slack: float | None


@dataclass(frozen=True)
class InstallSnapshot:
    """The install-gate inputs of one iteration.

    ``installed_before`` is the ``(agent_id, bundle_key)`` key set the master
    held before the add; ``admitted`` is the set passed to the master. The
    fresh-vs-duplicate decision is ``admitted - installed_before``.
    """

    installed_before: frozenset[tuple[int, bytes]]
    admitted: frozenset[tuple[int, bytes]]


@dataclass(frozen=True)
class PricedFeature:
    """One priced agent's demand stream and feature row.

    ``payoff`` / ``gap`` are the priced :class:`~combrum.demand.Demand`'s
    continuous fields, ``bundle_key`` its identity, and ``phi`` / ``eps`` the
    resolved feature row. Captured for every priced agent in price order.
    """

    agent_id: int
    bundle_key: bytes
    payoff: float
    gap: float
    phi: np.ndarray
    eps: float


@dataclass(frozen=True)
class StepRecord:
    """One iteration's pre-filter inputs over their full domain.

    A frozen, typed snapshot of the quantities a row-gen step's filters read
    in one iteration. Fields a formulation does not exercise stay at their
    empty default. ``iteration`` is the 0-based loop index.
    """

    iteration: int
    priced_reduced_costs: tuple[PricedReducedCost, ...] = ()
    admit_violations: tuple[AdmitViolation, ...] = ()
    purge_inputs: tuple[PurgeInput, ...] = ()
    install: InstallSnapshot | None = None
    aggregate_raw: float | None = None
    aggregate_bytes: bytes | None = None
    priced_features: tuple[PricedFeature, ...] = ()


@dataclass
class _Pending:
    """Mutable per-iteration accumulator; :meth:`seal` freezes it to a record."""

    iteration: int
    priced_reduced_costs: list[PricedReducedCost] = field(default_factory=list)
    admit_violations: list[AdmitViolation] = field(default_factory=list)
    purge_inputs: list[PurgeInput] = field(default_factory=list)
    install: InstallSnapshot | None = None
    aggregate_raw: float | None = None
    aggregate_bytes: bytes | None = None
    priced_features: list[PricedFeature] = field(default_factory=list)

    def seal(self) -> StepRecord:
        return StepRecord(
            iteration=self.iteration,
            priced_reduced_costs=tuple(self.priced_reduced_costs),
            admit_violations=tuple(self.admit_violations),
            purge_inputs=tuple(self.purge_inputs),
            install=self.install,
            aggregate_raw=self.aggregate_raw,
            aggregate_bytes=self.aggregate_bytes,
            priced_features=tuple(self.priced_features),
        )


class TraceSink(Protocol):
    """A sink a formulation emits one sealed record to per iteration.

    :class:`ListTraceSink` is the in-memory collector.
    """

    def emit(self, record: StepRecord) -> None:
        """Receive one iteration's sealed :class:`StepRecord`."""
        ...


@dataclass
class ListTraceSink:
    """A trace sink that appends every emitted record to ``records`` in order."""

    records: list[StepRecord] = field(default_factory=list)

    def emit(self, record: StepRecord) -> None:
        self.records.append(record)


def priced_features_from(
    demands: Mapping[int, Demand],
    ids: Sequence[int],
    rows: Sequence[tuple[np.ndarray, float]],
) -> tuple[PricedFeature, ...]:
    """Assemble the priced-feature stream from a demand map and feature rows.

    ``ids`` are the global agent ids in featurisation order, ``rows`` the
    matching ``(phi, eps)`` pairs, and ``demands`` the id-keyed priced
    outcomes. phi is copied read-only so the record cannot alias a buffer a
    later phase mutates.
    """
    out: list[PricedFeature] = []
    for agent_id, (phi, eps) in zip(ids, rows):
        a = int(agent_id)
        demand = demands[a]
        phi_arr = np.array(phi, dtype=np.float64)
        phi_arr.setflags(write=False)
        out.append(
            PricedFeature(
                agent_id=a,
                bundle_key=_bundle_identity(demand.bundle),
                payoff=float(demand.payoff),
                gap=float(demand.gap),
                phi=phi_arr,
                eps=float(eps),
            )
        )
    return tuple(out)


def _bundle_identity(bundle: np.ndarray) -> bytes:
    # Must stay byte-for-byte equal to the transport.base cut-identity codec
    # (dtype tag + raw bytes) so record keys compare to installed-row keys.
    # steprecord is deliberately kept free of a transport import.
    arr = np.ascontiguousarray(bundle)
    return arr.dtype.str.encode() + b":" + arr.tobytes()
