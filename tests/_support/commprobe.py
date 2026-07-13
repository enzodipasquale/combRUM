"""Counting wrapper around any :class:`~combrum.transport.base.Transport`.

Wrap a transport, run a scenario, read the per-method round and byte
tallies. The byte-accounting rules below are asserted by tests, so they
are contract, not debug aid.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeVar

import numpy as np

from combrum.transport import SerialTransport
from combrum.transport.base import CutRow, NodeTopology, Transport

_T = TypeVar("_T")

# CutRow fixed-envelope size: rep_id, agent_id, epsilon, phi length, and
# bundle-key length. Phi and bundle_key are measured exactly.
_ROW_HEADER_BYTES = 40


def _nbytes(value: object) -> int:
    return int(np.asarray(value).nbytes)


def spread_values(
    rng: np.random.Generator, shape: tuple[int, ...]
) -> np.ndarray:
    # Magnitudes spanning many orders, where summation order changes the
    # float result.
    magnitude = rng.uniform(-10.0, 10.0, size=shape)
    sign = rng.choice([-1.0, 1.0], size=shape)
    return sign * 10.0**magnitude


@dataclass(frozen=True)
class CommSnapshot:
    """Immutable copy of a :class:`CountingTransport`'s tallies."""

    counts: Mapping[str, int]
    bytes_moved: Mapping[str, int]

    def __post_init__(self) -> None:
        # Copy before proxying; a proxy over the live dict would track later mutations.
        object.__setattr__(self, "counts", MappingProxyType(dict(self.counts)))
        object.__setattr__(
            self, "bytes_moved", MappingProxyType(dict(self.bytes_moved))
        )


class CountingTransport(Transport):
    """Transport wrapper that tallies rounds and payload bytes per method.

    Each member delegates to the wrapped transport and returns its result
    unaltered (bitwise for arrays). Each collective call counts as one
    round of its kind; topology properties are rank-local reads, not
    rounds. Byte accounting per method:

    - ``bcast``: 0 — object payloads have no stable in-process wire size.
    - ``allreduce_max``: 8 — one float64 scalar from this rank.
    - ``sum_reproducible``: ``values.nbytes + global_ids.nbytes``.
    - ``sum_vectors_reproducible``: ``values.nbytes``.
    - ``scatter_by_agent``: received bytes (sum of returned arrays'
      ``nbytes``).
    - ``route_agent_values``: received sparse pairs, counted as one int64 id
      plus one float64 value per returned entry.
    - ``route_agent_values_batched``: received sparse triples, counted as two
      int64 ids plus one float64 value per returned entry.
    - ``node_shared``: published bytes (sum of input arrays' ``nbytes``)
      on the publishing rank only; 0 on peers.
    - ``batched_max``: ``values.nbytes``.
    - ``owner_broadcast``: ``values.nbytes + owners.nbytes``.
    - ``exchange_cuts``: sum over this rank's rows of
      ``phi.nbytes + len(bundle_key) +`` :data:`_ROW_HEADER_BYTES`.
    - ``collective()``: kind ``"collective_guard"``, 0 bytes.

    Data collectives tally on successful return; the guard tallies at
    call time, since its agreement round runs even if the body fails.
    """

    def __init__(self, inner: Transport) -> None:
        self._inner = inner
        self._counts: dict[str, int] = {}
        self._bytes: dict[str, int] = {}

    # --- tallies -------------------------------------------------------

    def _record(self, kind: str, nbytes: int) -> None:
        self._counts[kind] = self._counts.get(kind, 0) + 1
        self._bytes[kind] = self._bytes.get(kind, 0) + int(nbytes)

    def counts(self) -> dict[str, int]:
        """Rounds per method kind seen so far (a copy)."""
        return dict(self._counts)

    def bytes_moved(self) -> dict[str, int]:
        """Accounted payload bytes per method kind (a copy)."""
        return dict(self._bytes)

    def reset(self) -> None:
        """Zero every tally; the wrapped transport is untouched."""
        self._counts.clear()
        self._bytes.clear()

    def snapshot(self) -> CommSnapshot:
        """Freeze the current tallies as an immutable record."""
        return CommSnapshot(counts=self._counts, bytes_moved=self._bytes)

    # --- delegated Transport surface ------------------------------------

    @property
    def rank(self) -> int:
        return self._inner.rank

    @property
    def size(self) -> int:
        return self._inner.size

    @property
    def node(self) -> NodeTopology:
        return self._inner.node

    def collective(self) -> AbstractContextManager[None]:
        self._record("collective_guard", 0)
        # Return the inner guard as-is to preserve its error agreement.
        return self._inner.collective()

    def bcast(self, obj: _T | None, root: int = 0) -> _T:
        result = self._inner.bcast(obj, root)
        self._record("bcast", 0)
        return result

    def allreduce_max(self, value: float) -> float:
        result = self._inner.allreduce_max(value)
        self._record("allreduce_max", np.float64(0.0).nbytes)
        return result

    def sum_reproducible(
        self, values: np.ndarray, global_ids: np.ndarray
    ) -> np.ndarray | float:
        contributed = _nbytes(values) + _nbytes(global_ids)
        result = self._inner.sum_reproducible(values, global_ids)
        self._record("sum_reproducible", contributed)
        return result

    def sum_vectors_reproducible(self, values: np.ndarray) -> np.ndarray:
        contributed = _nbytes(values)
        result = self._inner.sum_vectors_reproducible(values)
        self._record("sum_vectors_reproducible", contributed)
        return result

    def send_to_root(
        self, obj: object | None, *, source: int, root: int = 0
    ) -> object | None:
        result = self._inner.send_to_root(obj, source=source, root=root)
        self._record("send_to_root", 0)
        return result

    def scatter_by_agent(
        self,
        arrays: dict[str, np.ndarray] | None,
        agent_ids: np.ndarray,
        *,
        root: int = 0,
    ) -> dict[str, np.ndarray]:
        result = self._inner.scatter_by_agent(arrays, agent_ids, root=root)
        # Count received (shard-sized) bytes, not root's full arrays.
        self._record(
            "scatter_by_agent",
            sum(arr.nbytes for arr in result.values()),
        )
        return result

    def gather_agent_values(
        self,
        values: np.ndarray,
        global_ids: np.ndarray,
        n_global: int,
        *,
        root: int = 0,
    ) -> np.ndarray | None:
        contributed = _nbytes(values) + _nbytes(global_ids)
        result = self._inner.gather_agent_values(
            values, global_ids, n_global, root=root
        )
        self._record("gather_agent_values", contributed)
        return result

    def route_agent_values(
        self,
        values: Mapping[int, float] | None,
        agent_ids: np.ndarray,
        *,
        source: int,
        n_agents: int,
    ) -> dict[int, float]:
        result = self._inner.route_agent_values(
            values,
            agent_ids,
            source=source,
            n_agents=n_agents,
        )
        self._record("route_agent_values", 16 * len(result))
        return result

    def route_agent_values_batched(
        self,
        values_by_rep: Mapping[int, Mapping[int, float]] | None,
        agent_ids: np.ndarray,
        *,
        owners: np.ndarray,
        n_agents: int,
    ) -> dict[int, dict[int, float]]:
        result = self._inner.route_agent_values_batched(
            values_by_rep,
            agent_ids,
            owners=owners,
            n_agents=n_agents,
        )
        received = sum(len(values) for values in result.values())
        self._record("route_agent_values_batched", 24 * received)
        return result

    def node_shared(
        self, arrays: dict[str, np.ndarray]
    ) -> Mapping[str, np.ndarray]:
        result = self._inner.node_shared(arrays)
        # Only the publishing rank moves content; peers' argument is ignored.
        if self._inner.node.node_rank == 0:
            published = sum(_nbytes(arr) for arr in arrays.values())
        else:
            published = 0
        self._record("node_shared", published)
        return result

    def batched_max(self, values: np.ndarray) -> np.ndarray:
        contributed = _nbytes(values)
        result = self._inner.batched_max(values)
        self._record("batched_max", contributed)
        return result

    def owner_broadcast(self, values: np.ndarray, owners: np.ndarray) -> np.ndarray:
        contributed = _nbytes(values) + _nbytes(owners)
        result = self._inner.owner_broadcast(values, owners)
        self._record("owner_broadcast", contributed)
        return result

    def exchange_cuts(
        self, rows: Sequence[CutRow], owners: np.ndarray
    ) -> tuple[CutRow, ...]:
        contributed = sum(
            row.phi.nbytes + len(row.bundle_key) + _ROW_HEADER_BYTES
            for row in rows
        )
        result = self._inner.exchange_cuts(rows, owners)
        self._record("exchange_cuts", contributed)
        return result


def _check_tally_readers_return_copies() -> None:
    """counts() and bytes_moved() must hand back fresh dicts.

    A caller may mutate what it gets back — or take before/after deltas —
    without corrupting the live tally. Runs at import, so every
    commprobe-using test file fails if the copy degrades to a live handle.
    """
    probe = CountingTransport(SerialTransport())
    probe.batched_max(np.zeros(2))  # float64 x 2 = 16 payload bytes

    for reader, expected in (
        (probe.counts, {"batched_max": 1}),
        (probe.bytes_moved, {"batched_max": 16}),
    ):
        name = reader.__name__
        if reader() != expected:
            raise AssertionError(f"{name} seed wrong: {reader()!r}")
        if reader() is reader():
            raise AssertionError(f"{name} handed back the same object twice")
        got = reader()
        got["batched_max"] = 999
        got["phantom"] = 1
        if reader() != expected:
            raise AssertionError(
                f"{name} is not a copy: mutating its result corrupted the "
                f"live tally to {reader()!r}"
            )


_check_tally_readers_return_copies()
