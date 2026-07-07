"""In-process reference transports.

:class:`SerialTransport` is the one-rank identity; :class:`LocalCluster`
runs R thread-ranks in one process and :class:`LocalMultirankTransport` is
one of its ranks. Both route cross-rank semantics through the shared
combiners below, so the two references cannot drift apart.

Determinism is structural: each collective writes per-rank payloads into
rank-indexed slots, rendezvouses at a barrier, then combines the
identical rank-ordered snapshot through canonical orders (global agent
id, rank index, canonical cut key). Arrival order cannot influence any
result, so results are bitwise reproducible under any thread scheduling.
"""

from __future__ import annotations

import copy
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from types import MappingProxyType
from typing import TypeVar

import numpy as np

from combrum.reductions import canonical_sum
from combrum.transport._common import (
    ids_validated as _ids_validated,
)
from combrum.transport._common import (
    route_bucket_for_rank as _route_bucket_for_rank,
)
from combrum.transport._common import (
    route_agent_axis_validated as _route_agent_axis_validated,
)
from combrum.transport._common import (
    route_local_ids_owned_validated as _route_local_ids_owned_validated,
)
from combrum.transport._common import (
    route_values_validated as _route_values_validated,
)
from combrum.transport._common import (
    scatter_arrays_validated as _scatter_arrays_validated,
)
from combrum.transport.base import (
    CutRow,
    NodeTopology,
    Transport,
    TransportError,
    canonical_cut_order,
)

_T = TypeVar("_T")


def _readonly(arr: np.ndarray) -> np.ndarray:
    arr.setflags(write=False)
    return arr


# Shared combiners: collective semantics on rank-ordered pooled inputs. Serial
# is the pool-of-one case, so serial and multirank cannot diverge.


def _combine_sum(
    values_by_rank: Sequence[np.ndarray],
    ids_by_rank: Sequence[np.ndarray],
) -> np.ndarray | float:
    arrays = [np.asarray(v, dtype=np.float64) for v in values_by_rank]
    ids_parts = [np.asarray(i) for i in ids_by_rank]
    non_empty_shapes = []
    for r, (arr, ids) in enumerate(zip(arrays, ids_parts)):
        if arr.ndim not in (1, 2):
            raise ValueError(
                "sum_reproducible: values must have shape (n,) or (n, M);"
                f" rank {r} passed shape {arr.shape}"
            )
        if ids.shape != (arr.shape[0],):
            raise ValueError(
                "sum_reproducible: one global id per contribution row"
                f" required; rank {r} has {arr.shape[0]} rows but"
                f" {ids.size} ids"
            )
        if arr.shape[0] > 0:
            width = int(arr.shape[1]) if arr.ndim == 2 else 1
            non_empty_shapes.append((arr.ndim == 2, width))
    agreed_shape = set(non_empty_shapes)
    if len(agreed_shape) > 1:
        raise ValueError(
            "sum_reproducible: non-empty ranks must agree on contribution"
            f" shape; got {sorted(agreed_shape)}"
        )
    if agreed_shape:
        is2d, width = next(iter(agreed_shape))
    else:
        is2d = any(arr.ndim == 2 for arr in arrays)
        width = max(
            (int(arr.shape[1]) for arr in arrays if arr.ndim == 2),
            default=1,
        )
    if is2d:
        arrays = [
            arr if arr.ndim == 2 else np.empty((0, width), dtype=np.float64)
            for arr in arrays
        ]
    values = np.concatenate(arrays, axis=0)
    ids = np.concatenate(ids_parts, axis=0)
    return canonical_sum(values, ids)


def _combine_vector_sum(values_by_rank: Sequence[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(v, dtype=np.float64) for v in values_by_rank]
    if not arrays:
        raise ValueError("sum_vectors_reproducible: at least one rank required")
    shape = arrays[0].shape
    for r, arr in enumerate(arrays):
        if arr.ndim not in (1, 2):
            raise ValueError(
                "sum_vectors_reproducible: values must have shape (M,) or"
                f" (B, M); rank {r} passed {arr.shape}"
            )
        if arr.shape != shape:
            raise ValueError(
                "sum_vectors_reproducible: every rank must pass the same"
                f" shape; rank 0 has {shape}, rank {r} has {arr.shape}"
            )
    flat = np.stack(
        [np.ascontiguousarray(arr).reshape(-1) for arr in arrays],
        axis=0,
    )
    rank_ids = np.arange(len(arrays), dtype=np.int64)
    reduced = np.asarray(canonical_sum(flat, rank_ids), dtype=np.float64)
    return reduced.reshape(shape)


def _combine_batched_max(
    values_by_rank: Sequence[np.ndarray],
) -> np.ndarray:
    arrays = [np.asarray(v, dtype=np.float64) for v in values_by_rank]
    shapes = {a.shape for a in arrays}
    if len(shapes) != 1 or arrays[0].ndim != 1:
        raise ValueError(
            "batched_max requires the same (B,) shape on every rank;"
            f" got shapes {sorted(shapes)}"
        )
    return np.max(np.stack(arrays, axis=0), axis=0)


def _agreed_owners(
    owners_by_rank: Sequence[np.ndarray], size: int, what: str
) -> np.ndarray:
    first = np.asarray(owners_by_rank[0])
    if first.ndim != 1 or not np.issubdtype(first.dtype, np.integer):
        raise ValueError(
            f"{what}: owners must be a 1-D integer array of ranks;"
            f" got shape {first.shape}, dtype {first.dtype}"
        )
    for r, other in enumerate(owners_by_rank[1:], start=1):
        other = np.asarray(other)
        if other.shape != first.shape or not np.array_equal(other, first):
            raise ValueError(
                f"{what}: owners must be identical on every rank;"
                f" rank {r} disagrees with rank 0"
            )
    if first.size and (int(first.min()) < 0 or int(first.max()) >= size):
        raise ValueError(
            f"{what}: owners must lie in [0, size) = [0, {size});"
            f" got range [{int(first.min())}, {int(first.max())}]"
        )
    return first


def _combine_owner_broadcast(
    values_by_rank: Sequence[np.ndarray], owners: np.ndarray
) -> np.ndarray:
    B = owners.shape[0]
    arrays: list[np.ndarray] = []
    for r, v in enumerate(values_by_rank):
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != B:
            raise ValueError(
                f"owner_broadcast: values must have shape (B, M) = ({B}, M);"
                f" rank {r} passed shape {arr.shape}"
            )
        arrays.append(arr)
    if len({a.shape for a in arrays}) != 1:
        raise ValueError(
            "owner_broadcast: every rank must pass the same (B, M) shape;"
            f" got shapes {sorted(a.shape for a in arrays)}"
        )
    out = np.zeros_like(arrays[0], dtype=np.float64)
    for b, owner in enumerate(owners):
        out[b, :] = arrays[int(owner)][b, :]
    return out


def _normalize_batched_agent_values(
    payloads_by_rank: Sequence[object],
    owners: np.ndarray,
    *,
    n_agents: int,
) -> dict[int, dict[int, float]]:
    by_rep: dict[int, dict[int, float]] = {}
    B = int(owners.shape[0])
    for rank, payload in enumerate(payloads_by_rank):
        if payload is None:
            payload = {}
        if not isinstance(payload, Mapping):
            raise ValueError(
                "route_agent_values_batched: values_by_rep must be a mapping"
                f" or None; rank {rank} passed {type(payload).__name__}"
            )
        for rep_key, values in payload.items():
            if (
                isinstance(rep_key, (bool, np.bool_))
                or not isinstance(rep_key, (int, np.integer))
                or int(rep_key) < 0
                or int(rep_key) >= B
            ):
                raise ValueError(
                    "route_agent_values_batched: replication ids must lie in"
                    f" [0, {B}); rank {rank} passed {rep_key!r}"
                )
            rep = int(rep_key)
            if int(owners[rep]) != rank:
                raise ValueError(
                    "route_agent_values_batched: only the owner rank may pass"
                    f" values for replication {rep}; owner is"
                    f" {int(owners[rep])}, rank {rank} passed values"
                )
            by_rep[rep] = _route_values_validated(
                values,
                n_agents=n_agents,
                rank=rank,
                source=rank,
                what="route_agent_values_batched",
            )
    return by_rep


def _combine_batched_route_agent_values(
    payloads_by_rank: Sequence[object],
    owners: np.ndarray,
    *,
    n_agents: int,
    size: int,
    my_rank: int,
) -> dict[int, dict[int, float]]:
    by_rep = _normalize_batched_agent_values(
        payloads_by_rank,
        owners,
        n_agents=n_agents,
    )
    out: dict[int, dict[int, float]] = {}
    for rep, values in by_rep.items():
        bucket = _route_bucket_for_rank(
            values,
            n_agents=n_agents,
            size=size,
            rank=my_rank,
        )
        if bucket:
            out[int(rep)] = bucket
    return out


def _combine_cuts(
    rows_by_rank: Sequence[Sequence[CutRow]],
    owners: np.ndarray,
    my_rank: int,
) -> tuple[CutRow, ...]:
    B = owners.shape[0]
    mine: list[CutRow] = []
    # Rank-major pooling + stable canonical sort: deterministic duplicate
    # order regardless of thread finish order.
    for r, rows in enumerate(rows_by_rank):
        for row in rows:
            if not isinstance(row, CutRow):
                raise ValueError(
                    "exchange_cuts: rows must be CutRow instances;"
                    f" rank {r} passed {type(row).__name__}"
                )
            if row.rep_id >= B:
                raise ValueError(
                    f"exchange_cuts: rep_id {row.rep_id} out of range for"
                    f" {B} live replications (contributed by rank {r})"
                )
            if int(owners[row.rep_id]) == my_rank:
                mine.append(row)
    return canonical_cut_order(mine)


def _combine_agent_values(
    values_by_rank: Sequence[np.ndarray],
    ids_by_rank: Sequence[np.ndarray],
    n_global: int,
) -> np.ndarray:
    if n_global < 0:
        raise ValueError(f"n_global must be >= 0; got {n_global}")
    values_parts: list[np.ndarray] = []
    ids_parts: list[np.ndarray] = []
    for r, (values, ids) in enumerate(zip(values_by_rank, ids_by_rank)):
        vals = np.asarray(values, dtype=np.float64)
        gids = np.asarray(ids)
        if vals.ndim != 1:
            raise ValueError(
                "gather_agent_values: values must be one-dimensional;"
                f" rank {r} passed shape {vals.shape}"
            )
        if gids.ndim != 1 or not np.issubdtype(gids.dtype, np.integer):
            raise ValueError(
                "gather_agent_values: global_ids must be a 1-D integer array;"
                f" rank {r} passed shape {gids.shape}, dtype {gids.dtype}"
            )
        if gids.shape != vals.shape:
            raise ValueError(
                "gather_agent_values: values and global_ids must have the same"
                f" shape on each rank; rank {r} has {vals.shape} and"
                f" {gids.shape}"
            )
        if gids.size and (int(gids.min()) < 0 or int(gids.max()) >= n_global):
            raise ValueError(
                "gather_agent_values: global_ids must lie in"
                f" [0, {n_global}); rank {r} got range"
                f" [{int(gids.min())}, {int(gids.max())}]"
            )
        values_parts.append(vals)
        ids_parts.append(gids.astype(np.int64, copy=False))
    all_ids = (
        np.concatenate(ids_parts, axis=0) if ids_parts else np.empty(0, dtype=np.int64)
    )
    if all_ids.size:
        unique, counts = np.unique(all_ids, return_counts=True)
        if unique.size != all_ids.size:
            raise ValueError(
                "gather_agent_values: duplicate global_ids:"
                f" {unique[counts > 1].tolist()}"
            )
        all_values = np.concatenate(values_parts, axis=0)
    else:
        all_values = np.empty(0, dtype=np.float64)
    out = np.zeros(int(n_global), dtype=np.float64)
    out[all_ids] = all_values
    return _readonly(out)


def _scatter_select(
    arrays: dict[str, np.ndarray], ids: np.ndarray
) -> dict[str, np.ndarray]:
    # Fancy indexing yields fresh arrays; flipping them read-only cannot
    # touch the root's originals.
    return {key: _readonly(full[ids]) for key, full in arrays.items()}


def _publish_node_arrays(arrays: object) -> dict[str, np.ndarray]:
    if not isinstance(arrays, dict):
        raise ValueError(
            "node_shared: the publishing rank must pass a dict of arrays;"
            f" got {type(arrays).__name__}"
        )
    published: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        if not isinstance(key, str):
            raise ValueError(f"node_shared: array keys must be str; got {key!r}")
        buf = np.array(value, copy=True)
        if buf.dtype == object:
            # Read-only flags do not protect object-array contents, so
            # sharing one would break immutability.
            raise ValueError(
                f"node_shared: array {key!r} must be numeric; got dtype object"
            )
        published[key] = _readonly(buf)
    return published


def _node_view(
    published: dict[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    # Per-rank views over one read-only base: the in-process analogue of a
    # node-shared memory window.
    return MappingProxyType({key: value.view() for key, value in published.items()})


# single-rank transport


class SerialTransport(Transport):
    """Rank 0 of 1, one node; every collective is the local computation."""

    _NODE = NodeTopology(node_id=0, node_rank=0, node_size=1, n_nodes=1)

    @property
    def rank(self) -> int:
        return 0

    @property
    def size(self) -> int:
        return 1

    @property
    def node(self) -> NodeTopology:
        return self._NODE

    @contextmanager
    def collective(self) -> Iterator[None]:
        try:
            yield
        except TransportError:
            # Re-raise unchanged: origin rank and message must survive
            # re-guarding.
            raise
        except Exception as exc:
            raise TransportError(0, f"{type(exc).__name__}: {exc}") from exc

    def bcast(self, obj: _T | None, root: int = 0) -> _T:
        if root != 0:
            raise ValueError(f"root must lie in [0, 1); got {root}")
        return obj  # type: ignore[return-value]

    def send_to_root(self, obj: _T | None, *, source: int, root: int = 0) -> _T | None:
        if source != 0:
            raise ValueError(f"source must lie in [0, 1); got {source}")
        if root != 0:
            raise ValueError(f"root must lie in [0, 1); got {root}")
        return obj

    def allreduce_max(self, value: float) -> float:
        return float(value)

    def sum_reproducible(
        self, values: np.ndarray, global_ids: np.ndarray
    ) -> np.ndarray | float:
        return _combine_sum([values], [global_ids])

    def sum_vectors_reproducible(self, values: np.ndarray) -> np.ndarray:
        return _combine_vector_sum([values])

    def scatter_by_agent(
        self,
        arrays: dict[str, np.ndarray] | None,
        local_ids: np.ndarray,
        *,
        root: int = 0,
    ) -> dict[str, np.ndarray]:
        if root != 0:
            raise ValueError(f"root must lie in [0, 1); got {root}")
        if arrays is None:
            raise ValueError(
                "scatter_by_agent: rank 0 must pass the full arrays; got None"
            )
        normalized, n_global = _scatter_arrays_validated(arrays)
        ids = _ids_validated(local_ids, n_global, "scatter_by_agent")
        return _scatter_select(normalized, ids)

    def gather_agent_values(
        self,
        values: np.ndarray,
        global_ids: np.ndarray,
        n_global: int,
        *,
        root: int = 0,
    ) -> np.ndarray | None:
        if root != 0:
            raise ValueError(f"root must lie in [0, 1); got {root}")
        return _combine_agent_values([values], [global_ids], int(n_global))

    def route_agent_values(
        self,
        values: Mapping[int, float] | None,
        local_ids: np.ndarray,
        *,
        source: int,
        n_agents: int,
    ) -> dict[int, float]:
        n_agents_i, src = _route_agent_axis_validated(
            n_agents,
            size=1,
            source=source,
            what="route_agent_values",
        )
        _route_local_ids_owned_validated(
            local_ids,
            n_agents=n_agents_i,
            rank=0,
            size=1,
            what="route_agent_values",
        )
        normalized = _route_values_validated(
            values,
            n_agents=n_agents_i,
            rank=0,
            source=src,
            what="route_agent_values",
        )
        bucket = _route_bucket_for_rank(
            normalized, n_agents=n_agents_i, size=1, rank=0
        )
        return bucket

    def route_agent_values_batched(
        self,
        values_by_rep: Mapping[int, Mapping[int, float]] | None,
        local_ids: np.ndarray,
        *,
        owners: np.ndarray,
        n_agents: int,
    ) -> dict[int, dict[int, float]]:
        n_agents_i, _src = _route_agent_axis_validated(
            n_agents,
            size=1,
            source=0,
            what="route_agent_values_batched",
        )
        _route_local_ids_owned_validated(
            local_ids,
            n_agents=n_agents_i,
            rank=0,
            size=1,
            what="route_agent_values_batched",
        )
        agreed = _agreed_owners([owners], size=1, what="route_agent_values_batched")
        return _combine_batched_route_agent_values(
            [values_by_rep],
            agreed,
            n_agents=n_agents_i,
            size=1,
            my_rank=0,
        )

    def node_shared(self, arrays: dict[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        return _node_view(_publish_node_arrays(arrays))

    def batched_max(self, values: np.ndarray) -> np.ndarray:
        return _combine_batched_max([values])

    def owner_broadcast(self, values: np.ndarray, owners: np.ndarray) -> np.ndarray:
        agreed = _agreed_owners([owners], size=1, what="owner_broadcast")
        return _combine_owner_broadcast([values], agreed)

    def exchange_cuts(
        self, rows: Sequence[CutRow], owners: np.ndarray
    ) -> tuple[CutRow, ...]:
        agreed = _agreed_owners([owners], size=1, what="exchange_cuts")
        return _combine_cuts([tuple(rows)], agreed, my_rank=0)


# in-process multirank transport


class _Rendezvous:
    """Rank-indexed slot exchange behind a reusable barrier.

    One collective = write own slot, barrier, snapshot all slots, barrier.
    The snapshot sits between the two waits, when no rank can be writing,
    so every rank reads the identical rank-ordered tuple. The second wait
    fences slot reuse: round k+1 writes begin only after every rank
    snapshotted round k. The timeout turns a would-be deadlock into a
    loud failure.
    """

    def __init__(self, size: int, timeout: float) -> None:
        self._slots: list[tuple[str, object] | None] = [None] * size
        self._barrier = threading.Barrier(size, timeout=timeout)

    def exchange(self, rank: int, tag: str, payload: object) -> tuple[object, ...]:
        self._slots[rank] = (tag, payload)
        self._wait(rank)
        snapshot = tuple(self._slots)
        self._wait(rank)
        tags = sorted({entry[0] for entry in snapshot})  # type: ignore[index]
        if len(tags) != 1:
            # Same snapshot on every rank, so all raise identically and
            # none hangs; otherwise unrelated payloads combine silently.
            raise TransportError(
                rank,
                f"misaligned collectives in one round: {tags}",
            )
        return tuple(entry[1] for entry in snapshot)  # type: ignore[index]

    def abort(self) -> None:
        self._barrier.abort()

    def _wait(self, rank: int) -> None:
        try:
            self._barrier.wait()
        except threading.BrokenBarrierError:
            raise TransportError(
                rank,
                "collective rendezvous broken: a peer rank died outside a"
                " guarded section or timed out",
            ) from None


class LocalMultirankTransport(Transport):
    """One in-process rank of a :class:`LocalCluster`.

    Constructed by the cluster; all instances of one run share a single
    :class:`_Rendezvous`.
    """

    def __init__(
        self,
        rank: int,
        size: int,
        node: NodeTopology,
        rendezvous: _Rendezvous,
    ) -> None:
        self._rank = int(rank)
        self._size = int(size)
        self._node = node
        self._rendezvous = rendezvous

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def size(self) -> int:
        return self._size

    @property
    def node(self) -> NodeTopology:
        return self._node

    @contextmanager
    def collective(self) -> Iterator[None]:
        exc: Exception | None = None
        try:
            yield
        except Exception as e:
            exc = e
        if exc is None:
            verdict: tuple[int, str] | None = None
        elif isinstance(exc, TransportError):
            # Preserve the true origin through nested guards, not the rank
            # that re-guarded.
            verdict = (exc.rank, exc.message)
        else:
            verdict = (self._rank, f"{type(exc).__name__}: {exc}")
        gathered = self._rendezvous.exchange(self._rank, "collective.agree", verdict)
        failures = [v for v in gathered if v is not None]
        if failures:
            # Rank-ordered slots: lowest-ranked report is the same agreed
            # verdict on every rank.
            origin_rank, message = failures[0]  # type: ignore[misc]
            raise TransportError(origin_rank, message) from exc

    def bcast(self, obj: _T | None, root: int = 0) -> _T:
        if not 0 <= root < self._size:
            raise ValueError(f"root must lie in [0, {self._size}); got {root}")
        gathered = self._rendezvous.exchange(
            self._rank, "bcast", obj if self._rank == root else None
        )
        if self._rank == root:
            return obj  # type: ignore[return-value]
        # Deep copy: without it, in-process aliasing would let one rank
        # mutate another's state, a coupling no real transport has.
        return copy.deepcopy(gathered[root])  # type: ignore[return-value]

    def send_to_root(self, obj: _T | None, *, source: int, root: int = 0) -> _T | None:
        if not 0 <= source < self._size:
            raise ValueError(f"source must lie in [0, {self._size}); got {source}")
        if not 0 <= root < self._size:
            raise ValueError(f"root must lie in [0, {self._size}); got {root}")
        gathered = self._rendezvous.exchange(
            self._rank,
            f"send_to_root:{source}:{root}",
            obj if self._rank == source else None,
        )
        if self._rank != root:
            return None
        payload = gathered[source]
        return obj if source == root else copy.deepcopy(payload)  # type: ignore[return-value]

    def allreduce_max(self, value: float) -> float:
        gathered = self._rendezvous.exchange(self._rank, "allreduce_max", float(value))
        return float(np.max(np.asarray(gathered, dtype=np.float64)))

    def sum_reproducible(
        self, values: np.ndarray, global_ids: np.ndarray
    ) -> np.ndarray | float:
        payload = (
            np.asarray(values, dtype=np.float64),
            np.asarray(global_ids),
        )
        gathered = self._rendezvous.exchange(self._rank, "sum_reproducible", payload)
        return _combine_sum(
            [entry[0] for entry in gathered],  # type: ignore[index]
            [entry[1] for entry in gathered],  # type: ignore[index]
        )

    def sum_vectors_reproducible(self, values: np.ndarray) -> np.ndarray:
        gathered = self._rendezvous.exchange(
            self._rank,
            "sum_vectors_reproducible",
            np.asarray(values, dtype=np.float64),
        )
        return _combine_vector_sum(gathered)  # type: ignore[arg-type]

    def scatter_by_agent(
        self,
        arrays: dict[str, np.ndarray] | None,
        local_ids: np.ndarray,
        *,
        root: int = 0,
    ) -> dict[str, np.ndarray]:
        if not 0 <= root < self._size:
            raise ValueError(f"root must lie in [0, {self._size}); got {root}")
        gathered = self._rendezvous.exchange(
            self._rank, f"scatter_by_agent:{root}", (arrays, local_ids)
        )
        root_arrays = gathered[root][0]  # type: ignore[index]
        for r in range(self._size):
            if r == root:
                continue
            if gathered[r][0] is not None:  # type: ignore[index]
                raise ValueError(
                    f"scatter_by_agent: only rank {root} holds the full arrays;"
                    f" rank {r} passed a non-None payload"
                )
        if root_arrays is None:
            raise ValueError(
                f"scatter_by_agent: rank {root} must pass the full arrays; got None"
            )
        normalized, n_global = _scatter_arrays_validated(root_arrays)
        # Validate every rank's ids on every rank: divergent caller state
        # raises the same error everywhere instead of stranding peers.
        ids_by_rank = [
            _ids_validated(entry[1], n_global, "scatter_by_agent")  # type: ignore[index]
            for entry in gathered
        ]
        return _scatter_select(normalized, ids_by_rank[self._rank])

    def gather_agent_values(
        self,
        values: np.ndarray,
        global_ids: np.ndarray,
        n_global: int,
        *,
        root: int = 0,
    ) -> np.ndarray | None:
        if not 0 <= root < self._size:
            raise ValueError(f"root must lie in [0, {self._size}); got {root}")
        payload = (
            np.asarray(values, dtype=np.float64),
            np.asarray(global_ids),
            int(n_global),
        )
        gathered = self._rendezvous.exchange(
            self._rank, f"gather_agent_values:{root}", payload
        )
        sizes = {entry[2] for entry in gathered}  # type: ignore[index]
        if len(sizes) != 1:
            raise ValueError(
                "gather_agent_values: every rank must pass the same"
                f" n_global; got {sorted(sizes)}"
            )
        if self._rank != root:
            return None
        return _combine_agent_values(
            [entry[0] for entry in gathered],  # type: ignore[index]
            [entry[1] for entry in gathered],  # type: ignore[index]
            int(gathered[0][2]),  # type: ignore[index]
        )

    def route_agent_values(
        self,
        values: Mapping[int, float] | None,
        local_ids: np.ndarray,
        *,
        source: int,
        n_agents: int,
    ) -> dict[int, float]:
        payload = (
            values,
            local_ids,
            n_agents,
            source,
        )
        gathered = self._rendezvous.exchange(self._rank, "route_agent_values", payload)
        geometries = {
            (entry[2], entry[3])  # type: ignore[index]
            for entry in gathered
        }
        if len(geometries) != 1:
            raise ValueError(
                "route_agent_values: n_agents and source must be identical"
                " on every rank"
            )
        first = gathered[0]  # type: ignore[index]
        n_agents_i, src = _route_agent_axis_validated(
            first[2],
            size=self._size,
            source=first[3],
            what="route_agent_values",
        )
        for r, entry in enumerate(gathered):
            _route_local_ids_owned_validated(
                entry[1],
                n_agents=n_agents_i,
                rank=r,
                size=self._size,
                what="route_agent_values",  # type: ignore[index]
            )
        normalized_by_rank = [
            _route_values_validated(
                entry[0],  # type: ignore[index]
                n_agents=n_agents_i,
                rank=r,
                source=src,
                what="route_agent_values",
            )
            for r, entry in enumerate(gathered)
        ]
        source_values = normalized_by_rank[src]
        bucket = _route_bucket_for_rank(
            source_values,
            n_agents=n_agents_i,
            size=self._size,
            rank=self._rank,
        )
        return bucket

    def route_agent_values_batched(
        self,
        values_by_rep: Mapping[int, Mapping[int, float]] | None,
        local_ids: np.ndarray,
        *,
        owners: np.ndarray,
        n_agents: int,
    ) -> dict[int, dict[int, float]]:
        payload = (
            values_by_rep,
            local_ids,
            owners,
            n_agents,
        )
        gathered = self._rendezvous.exchange(
            self._rank, "route_agent_values_batched", payload
        )
        geometries = {entry[3] for entry in gathered}  # type: ignore[index]
        if len(geometries) != 1:
            raise ValueError(
                "route_agent_values_batched: n_agents must be identical"
                " on every rank"
            )
        first = gathered[0]  # type: ignore[index]
        n_agents_i, _src = _route_agent_axis_validated(
            first[3],
            size=self._size,
            source=0,
            what="route_agent_values_batched",
        )
        for r, entry in enumerate(gathered):
            _route_local_ids_owned_validated(
                entry[1],
                n_agents=n_agents_i,
                rank=r,
                size=self._size,
                what="route_agent_values_batched",  # type: ignore[index]
            )
        agreed = _agreed_owners(
            [entry[2] for entry in gathered],  # type: ignore[index]
            self._size,
            "route_agent_values_batched",
        )
        return _combine_batched_route_agent_values(
            [entry[0] for entry in gathered],  # type: ignore[index]
            agreed,
            n_agents=n_agents_i,
            size=self._size,
            my_rank=self._rank,
        )

    def node_shared(self, arrays: dict[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        if self._node.node_rank == 0:
            try:
                payload: tuple[str, object] = (
                    "published",
                    _publish_node_arrays(arrays),
                )
            except Exception as exc:
                # Publish the failure rather than raise before the
                # rendezvous, or peers wait out the barrier timeout.
                payload = ("error", f"{type(exc).__name__}: {exc}")
        else:
            payload = ("peer", None)
        gathered = self._rendezvous.exchange(self._rank, "node_shared", payload)
        errors = [
            (r, entry[1])  # type: ignore[index]
            for r, entry in enumerate(gathered)
            if entry[0] == "error"  # type: ignore[index]
        ]
        if errors:
            r0, message = errors[0]
            raise ValueError(f"node_shared: publishing failed on rank {r0}: {message}")
        # Nodes are contiguous rank blocks, so the publisher sits
        # node_rank slots below this rank.
        leader_rank = self._rank - self._node.node_rank
        published = gathered[leader_rank][1]  # type: ignore[index]
        return _node_view(published)  # type: ignore[arg-type]

    def batched_max(self, values: np.ndarray) -> np.ndarray:
        gathered = self._rendezvous.exchange(
            self._rank, "batched_max", np.asarray(values, dtype=np.float64)
        )
        return _combine_batched_max(gathered)  # type: ignore[arg-type]

    def owner_broadcast(self, values: np.ndarray, owners: np.ndarray) -> np.ndarray:
        payload = (
            np.asarray(values, dtype=np.float64),
            np.asarray(owners),
        )
        gathered = self._rendezvous.exchange(self._rank, "owner_broadcast", payload)
        agreed = _agreed_owners(
            [entry[1] for entry in gathered],  # type: ignore[index]
            self._size,
            "owner_broadcast",
        )
        return _combine_owner_broadcast(
            [entry[0] for entry in gathered],  # type: ignore[index]
            agreed,
        )

    def exchange_cuts(
        self, rows: Sequence[CutRow], owners: np.ndarray
    ) -> tuple[CutRow, ...]:
        payload = (tuple(rows), np.asarray(owners))
        gathered = self._rendezvous.exchange(self._rank, "exchange_cuts", payload)
        agreed = _agreed_owners(
            [entry[1] for entry in gathered],  # type: ignore[index]
            self._size,
            "exchange_cuts",
        )
        return _combine_cuts(
            [entry[0] for entry in gathered],  # type: ignore[index]
            agreed,
            self._rank,
        )


class LocalCluster:
    """In-process multirank runner: R thread-ranks, barrier rendezvous.

    :meth:`run` executes ``fn(transport)`` once per rank, each on its own
    thread, and returns the per-rank results in rank order. Every call
    builds a fresh rendezvous and transports, so runs are independent.

    ``ranks_per_node`` carves ranks into contiguous nodes (the last may be
    smaller); the default is a single node holding every rank.
    ``rendezvous_timeout`` bounds every barrier wait, turning a rank dying
    outside a guarded section into a :class:`TransportError` on its peers
    instead of a hang.
    """

    def __init__(
        self,
        size: int,
        ranks_per_node: int | None = None,
        rendezvous_timeout: float = 10.0,
    ) -> None:
        if size < 1:
            raise ValueError(f"size must be >= 1; got {size}")
        if ranks_per_node is None:
            ranks_per_node = size
        if ranks_per_node < 1:
            raise ValueError(f"ranks_per_node must be >= 1; got {ranks_per_node}")
        if not rendezvous_timeout > 0:
            raise ValueError(
                f"rendezvous_timeout must be > 0; got {rendezvous_timeout}"
            )
        self._size = int(size)
        self._ranks_per_node = int(ranks_per_node)
        self._timeout = float(rendezvous_timeout)

    def _topology(self, rank: int) -> NodeTopology:
        n_nodes = -(-self._size // self._ranks_per_node)
        node_id = rank // self._ranks_per_node
        node_size = min(
            self._ranks_per_node,
            self._size - node_id * self._ranks_per_node,
        )
        return NodeTopology(
            node_id=node_id,
            node_rank=rank % self._ranks_per_node,
            node_size=node_size,
            n_nodes=n_nodes,
        )

    def run(self, fn: Callable[[Transport], _T]) -> list[_T]:
        rendezvous = _Rendezvous(self._size, self._timeout)
        transports = [
            LocalMultirankTransport(
                rank=r,
                size=self._size,
                node=self._topology(r),
                rendezvous=rendezvous,
            )
            for r in range(self._size)
        ]
        results: list[_T | None] = [None] * self._size
        errors: list[BaseException | None] = [None] * self._size

        def _run_rank(r: int) -> None:
            try:
                results[r] = fn(transports[r])
            except BaseException as exc:  # relayed below
                errors[r] = exc

        threads = [
            threading.Thread(
                target=_run_rank,
                args=(r,),
                name=f"local-rank-{r}",
                daemon=True,
            )
            for r in range(self._size)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=self._timeout + 5.0)
        alive = [thread.name for thread in threads if thread.is_alive()]
        if alive:
            # Free peers parked at the barrier, then fail: a hung run must
            # never look like a passing one.
            rendezvous.abort()
            raise RuntimeError(
                f"local cluster ranks still running after the rendezvous"
                f" timeout: {alive}; aborting the run instead of hanging"
            )
        first_error = next((e for e in errors if e is not None), None)
        if first_error is not None:
            # Rank order makes the relayed error deterministic; agreement
            # semantics belong to collective(), not here.
            raise first_error
        return list(results)  # type: ignore[arg-type]
