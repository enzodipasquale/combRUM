"""Distributed :class:`combrum.transport.base.Transport` over mpi4py.

Row-keyed reductions and cut exchange preserve the same canonical contracts as
the in-process references: global row id for row sums, and
``(rep_id, agent_id, bundle_key)`` for cuts. Fixed-rank aggregate reductions
such as :meth:`sum_vectors_reproducible` are deterministic for a fixed rank
layout; they are not a row-distribution-invariant replacement for
:meth:`sum_reproducible`.

mpi4py is an optional dependency (the ``mpi`` extra); it is imported at
instantiation, not module load, so the package imports without it.

Cross-rank validation is agreed before data movement where possible, so ranks
raise together instead of leaving peers in later collectives.
"""

from __future__ import annotations

import math
import struct
import hashlib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from combrum.reductions import canonical_sum, canonical_sum_window_rows
from combrum.transport._common import (
    agent_owner_ranks as _agent_owner_ranks,
)
from combrum.transport._common import (
    ids_validated as _ids_validated,
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
    _CUT_ROW_WIRE_HEADER_BYTES,
    _cut_row_nbytes,
)

if TYPE_CHECKING:
    from mpi4py.MPI import Intracomm, Win
else:
    Intracomm = Any
    Win = Any

_T = TypeVar("_T")


def _mpi() -> Any:
    """Import mpi4py on first use and return its ``MPI`` namespace."""
    try:
        from mpi4py import MPI
    except ImportError as exc:
        raise ModuleNotFoundError(
            "MpiTransport requires mpi4py, which is not importable here;"
            " install the optional extra: pip install combrum[mpi]"
        ) from exc
    return MPI


# scatter_by_agent streaming chunk. Bounds the root's in-flight transient at
# chunk_bytes x (size - 1); large enough to amortize per-message latency.
_SCATTER_CHUNK_BYTES: int = 8 * 2**20

# One tag suffices: MPI's non-overtaking rule orders messages per
# (source, destination, tag), and root/receiver walk keys and windows in the
# same order, so the k-th send on a channel is the k-th receive.
_SCATTER_STREAM_TAG: int = 92

# Separate from scatter streams so a result object is never confused with row
# shards on the same channel.
_OBJECT_TO_ROOT_TAG: int = 93

# Cache-line alignment satisfies every numeric dtype and keeps adjacent arrays
# off the same line.
_WINDOW_ALIGN: int = 64

_ROUTE_VALUE_DTYPE = np.dtype([("gid", np.int64), ("value", np.float64)])
_ROUTE_BATCH_VALUE_DTYPE = np.dtype(
    [("rep", np.int64), ("gid", np.int64), ("value", np.float64)]
)


def _chunk_spans(
    n_rows: int, row_nbytes: int, chunk_bytes: int
) -> list[tuple[int, int]]:
    """Half-open row windows of at most ``chunk_bytes`` each (>= 1 row).

    Pure function of (row count, row width, chunk size), so root and
    receiver derive identical schedules and the k-th send pairs the k-th
    receive. A row wider than the chunk still travels whole, degrading
    the in-flight bound to one row per destination.
    """
    if n_rows == 0 or row_nbytes == 0:
        return []
    rows_per = max(1, chunk_bytes // row_nbytes)
    return [
        (start, min(start + rows_per, n_rows)) for start in range(0, n_rows, rows_per)
    ]


# Cut-row wire header: rep_id, agent_id (int64), epsilon (float64), phi length,
# bundle-key length (int64); native order, no padding. Followed by phi's raw
# float64 bytes then the bundle key. Native order is unambiguous within one MPI
# job; raw bytes preserve exact bit patterns.
_CUT_HEADER = struct.Struct("=qqdqq")
assert _CUT_HEADER.size == _CUT_ROW_WIRE_HEADER_BYTES


def _write_cut(buf: np.ndarray, offset: int, row: CutRow) -> int:
    header = _CUT_HEADER.pack(
        row.rep_id,
        row.agent_id,
        row.epsilon,
        row.phi.shape[0],
        len(row.bundle_key),
    )
    pos = int(offset)
    buf[pos : pos + _CUT_HEADER.size] = np.frombuffer(header, dtype=np.uint8)
    pos += _CUT_HEADER.size
    phi_bytes = row.phi.view(np.uint8)
    buf[pos : pos + phi_bytes.size] = phi_bytes
    pos += int(phi_bytes.size)
    key_bytes = np.frombuffer(row.bundle_key, dtype=np.uint8)
    buf[pos : pos + key_bytes.size] = key_bytes
    return pos + int(key_bytes.size)


def _cut_record_dtype(k: int, key_len: int) -> np.dtype:
    # Mirrors _CUT_HEADER + phi payload + key byte-for-byte: numpy packs
    # structured dtypes without padding, so a records array views as the
    # exact wire bytes of the equivalent _write_cut sequence.
    dtype = np.dtype(
        [
            ("rep", np.int64),
            ("agent", np.int64),
            ("eps", np.float64),
            ("k", np.int64),
            ("klen", np.int64),
            ("phi", np.float64, (k,)),
            ("key", np.void, key_len),
        ]
    )
    assert dtype.itemsize == _CUT_HEADER.size + 8 * k + key_len
    return dtype


def _pack_cuts(
    rows: tuple[CutRow, ...], canon: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Destination-grouped wire buffer plus per-destination byte counts.

    Within one fit every row shares the phi width and bundle-key length, so
    the common case packs as one fixed-stride records array; heterogeneous
    rows fall back to the per-row writer. Both produce identical bytes:
    rows grouped by destination, input order within each destination.
    """
    if not rows:
        return np.empty(0, dtype=np.uint8), np.zeros(size, dtype=np.int64)
    k = int(rows[0].phi.shape[0])
    key_len = len(rows[0].bundle_key)
    uniform = k > 0 and all(
        row.phi.shape[0] == k and len(row.bundle_key) == key_len for row in rows
    )
    if not uniform:
        send_counts = np.zeros(size, dtype=np.int64)
        for row in rows:
            send_counts[int(canon[row.rep_id])] += _cut_row_nbytes(row)
        send_buf = np.empty(int(send_counts.sum()), dtype=np.uint8)
        cursors = np.concatenate(([0], np.cumsum(send_counts)[:-1]))
        for row in rows:
            dest = int(canon[row.rep_id])
            cursors[dest] = _write_cut(send_buf, int(cursors[dest]), row)
        return send_buf, send_counts
    n = len(rows)
    rep_ids = np.fromiter((row.rep_id for row in rows), np.int64, count=n)
    dests = canon[rep_ids]
    order = np.argsort(dests, kind="stable")
    ordered = [rows[i] for i in order.tolist()]
    rec = np.empty(n, dtype=_cut_record_dtype(k, key_len))
    rec["rep"] = rep_ids[order]
    rec["agent"] = np.fromiter((row.agent_id for row in ordered), np.int64, count=n)
    rec["eps"] = np.fromiter((row.epsilon for row in ordered), np.float64, count=n)
    rec["k"] = k
    rec["klen"] = key_len
    rec["phi"] = np.stack([row.phi for row in ordered])
    rec["key"] = np.frombuffer(
        b"".join(row.bundle_key for row in ordered),
        dtype=np.dtype((np.void, key_len)),
    )
    send_counts = np.bincount(dests, minlength=size).astype(np.int64)
    send_counts *= rec.dtype.itemsize
    return rec.view(np.uint8), send_counts


def _unpack_cuts(block: bytes | memoryview) -> list[CutRow]:
    view = memoryview(block)
    if not len(view):
        return []
    _rep0, _agent0, _eps0, k, key_len = _CUT_HEADER.unpack_from(view, 0)
    row_nbytes = _CUT_HEADER.size + 8 * k + key_len
    if k > 0 and key_len > 0 and len(view) % row_nbytes == 0:
        rec = np.frombuffer(view, dtype=_cut_record_dtype(k, key_len))
        if bool((rec["k"] == k).all()) and bool((rec["klen"] == key_len).all()):
            # The fixed-stride reading is exact, not heuristic: rows sit
            # back-to-back, so matching k/klen at every stride position pins
            # each row's true offset by induction from row 0.
            # One matrix copy detaches every phi from the receive buffer.
            phis = np.array(rec["phi"], dtype=np.float64)
            phis.setflags(write=False)
            keys = rec["key"].tobytes()
            return [
                CutRow._from_parts(
                    rep_id=rep,
                    agent_id=agent,
                    phi=phis[i],
                    epsilon=eps,
                    bundle_key=keys[i * key_len : (i + 1) * key_len],
                )
                for i, (rep, agent, eps) in enumerate(
                    zip(
                        rec["rep"].tolist(),
                        rec["agent"].tolist(),
                        rec["eps"].tolist(),
                    )
                )
            ]
    rows: list[CutRow] = []
    offset = 0
    while offset < len(view):
        rep_id, agent_id, epsilon, k, key_len = _CUT_HEADER.unpack_from(view, offset)
        offset += _CUT_HEADER.size
        # Copy detaches phi so a row does not pin the whole receive buffer.
        phi = np.frombuffer(view, dtype=np.float64, count=k, offset=offset).copy()
        phi.setflags(write=False)
        offset += 8 * k
        key = bytes(view[offset : offset + key_len])
        offset += key_len
        rows.append(
            CutRow._from_parts(
                rep_id=rep_id,
                agent_id=agent_id,
                phi=phi,
                epsilon=epsilon,
                bundle_key=key,
            )
        )
    return rows


def _scatter_verdict(
    arrays: dict[str, np.ndarray] | None,
    gathered: Sequence[tuple[bool, np.ndarray]],
    size: int,
    root: int,
) -> tuple[dict[str, np.ndarray], tuple[str, Any]]:
    """Root-side verdict: the validated arrays and the wire header.

    The verdict travels as data, not as a root-only raise, so every rank
    raises the identical message and no rank is stranded mid-collective.
    On success the header carries each key's dtype and trailing shape
    (enough to preallocate a shard), never the rows.
    """
    try:
        if arrays is None:
            raise ValueError(
                f"scatter_by_agent: rank {root} must pass the full arrays; got None"
            )
        for r in range(size):
            if r == root:
                continue
            if gathered[r][0]:
                raise ValueError(
                    f"scatter_by_agent: only rank {root} holds the full arrays;"
                    f" rank {r} passed a non-None payload"
                )
        normalized, n_global = _scatter_arrays_validated(arrays)
        for _, ids in gathered:
            _ids_validated(ids, n_global, "scatter_by_agent")
        header = [(key, full.dtype, full.shape[1:]) for key, full in normalized.items()]
        return normalized, ("ok", header)
    except ValueError as exc:
        return {}, ("error", str(exc))


def _node_arrays_validated(arrays: object) -> dict[str, np.ndarray]:
    """Publisher-side validation.

    Returns ``asarray`` views, not copies: the single publish copy
    happens straight into the shared window, with no staging duplicate.
    """
    if not isinstance(arrays, dict):
        raise ValueError(
            "node_shared: the publishing rank must pass a dict of arrays;"
            f" got {type(arrays).__name__}"
        )
    staged: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        if not isinstance(key, str):
            raise ValueError(f"node_shared: array keys must be str; got {key!r}")
        arr = np.asarray(value)
        if arr.dtype == object:
            # Read-only flags cannot protect object-array contents, so a
            # shared object array would break the immutable-copy promise.
            raise ValueError(
                f"node_shared: array {key!r} must be numeric; got dtype object"
            )
        staged[key] = arr
    return staged


def _window_layout(
    staged: dict[str, np.ndarray],
) -> tuple[list[tuple[str, np.dtype, tuple[int, ...], int]], int]:
    """Deterministic packing of one call's arrays into one window.

    Entries are ``(key, dtype, shape, byte offset)`` in dict order,
    offsets rounded up to :data:`_WINDOW_ALIGN`; second value is the
    total byte size. Pure function of the staged arrays, so publisher
    and peers reconstruct identical views from the broadcast layout.
    """
    layout: list[tuple[str, np.dtype, tuple[int, ...], int]] = []
    offset = 0
    for key, arr in staged.items():
        offset = -(-offset // _WINDOW_ALIGN) * _WINDOW_ALIGN
        layout.append((key, arr.dtype, arr.shape, offset))
        offset += int(arr.nbytes)
    return layout, offset


class MpiTransport(Transport):
    """:class:`Transport` over an MPI communicator.

    ``comm`` defaults to ``COMM_WORLD``; a caller-provided communicator
    stays caller-owned (:meth:`close` frees only the node communicator
    and any node-shared windows the transport created).

    ``scatter_chunk_bytes`` overrides the :meth:`scatter_by_agent`
    streaming window; the root's value travels with the scatter header,
    so every receiver follows the same schedule.

    :meth:`counts` / :meth:`reset` expose invocation tallies for diagnostics.
    """

    def __init__(
        self,
        comm: Intracomm | None = None,
        scatter_chunk_bytes: int = _SCATTER_CHUNK_BYTES,
    ) -> None:
        mpi = _mpi()
        if scatter_chunk_bytes < 1:
            raise ValueError(
                f"scatter_chunk_bytes must be >= 1; got {scatter_chunk_bytes}"
            )
        self._mpi: Any = mpi
        self._comm: Intracomm = mpi.COMM_WORLD if comm is None else comm
        self._scatter_chunk_bytes: int = int(scatter_chunk_bytes)
        self._windows: list[Win] = []
        self._counts: dict[str, int] = {}
        self._rank: int = int(self._comm.Get_rank())
        self._size: int = int(self._comm.Get_size())
        # Ranks sharing a memory domain form one node.
        self._tick("split_type")
        self._node_comm: Intracomm | None = self._comm.Split_type(mpi.COMM_TYPE_SHARED)
        node_rank = int(self._node_comm.Get_rank())
        node_size = int(self._node_comm.Get_size())
        # Split_type breaks key ties by parent rank, so in-node rank 0 is the
        # node's lowest world rank; broadcasting it identifies each node.
        self._tick("bcast")
        leader = int(self._node_comm.bcast(self._rank, root=0))
        self._tick("allgather")
        leaders: list[int] = self._comm.allgather(leader)
        # Node ids ordered by lowest world rank: a pure function of the rank
        # layout, independent of MPI's shared-domain enumeration order.
        ordered = sorted(set(leaders))
        self._node = NodeTopology(
            node_id=ordered.index(leader),
            node_rank=node_rank,
            node_size=node_size,
            n_nodes=len(ordered),
        )

    # --- introspection --------------------------------------------------

    def _tick(self, kind: str) -> None:
        self._counts[kind] = self._counts.get(kind, 0) + 1

    def counts(self) -> dict[str, int]:
        """Tallies of MPI primitive invocations by kind.

        Kinds are lowercase primitive names (``allreduce``, ``alltoall``,
        ``alltoallv``, ``allgather``, ``barrier``, ``bcast``, ``gather``,
        ``p2p_recv``, ``p2p_send``, ``split_type``,
        ``win_allocate_shared``, ``win_lock_all``, ``win_sync``), pooled
        over all communicators since construction. Point-to-point kinds
        count one tick per scatter-stream chunk.
        """
        return dict(self._counts)

    def reset(self) -> None:
        """Zero the invocation tallies; the transport itself is untouched."""
        self._counts.clear()

    # --- topology -------------------------------------------------------

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def size(self) -> int:
        return self._size

    @property
    def node(self) -> NodeTopology:
        return self._node

    # --- collectives ------------------------------------------------------

    @contextmanager
    def collective(self) -> Iterator[None]:
        verdict: tuple[int, str] | None = None
        cause: Exception | None = None
        try:
            yield
        except Exception as exc:
            cause = exc
            if isinstance(exc, TransportError):
                # Preserve the true origin through nested guards.
                verdict = (exc.rank, exc.message)
            else:
                verdict = (self._rank, f"{type(exc).__name__}: {exc}")
        # One word-sized round at every guard exit, success or failure, so the
        # reduction is rank-uniform and no rank waits on a failed peer. MIN over
        # "my rank if I hold a verdict, else size" elects the lowest REPORTING
        # rank, which holds a verdict and can broadcast it. (Reducing the
        # carried origin could elect a rank with nothing to say: a body may
        # re-raise an agreed TransportError whose origin rank is healthy here.)
        code = np.array(
            [self._rank if verdict is not None else self._size],
            dtype=np.int64,
        )
        agreed = np.empty_like(code)
        self._tick("allreduce")
        self._comm.Allreduce(code, agreed, op=self._mpi.MIN)
        reporter = int(agreed[0])
        if reporter == self._size:
            return  # no rank failed
        # Failure path only: the elected reporter broadcasts its verdict, which
        # cannot ride a reduction. Keeps the success path at a single round.
        self._tick("bcast")
        origin, message = self._comm.bcast(
            verdict if self._rank == reporter else None, root=reporter
        )
        raise TransportError(origin, message) from cause

    def bcast(self, obj: _T | None, root: int = 0) -> _T:
        if not 0 <= root < self._size:
            raise ValueError(f"root must lie in [0, {self._size}); got {root}")
        self._tick("bcast")
        # mpi4py's object path unpickles a fresh graph on every receiver, so
        # received copies are private; the root gets its own object back.
        return self._comm.bcast(obj if self._rank == root else None, root=root)

    def send_to_root(self, obj: _T | None, *, source: int, root: int = 0) -> _T | None:
        if not 0 <= source < self._size:
            raise ValueError(f"source must lie in [0, {self._size}); got {source}")
        if not 0 <= root < self._size:
            raise ValueError(f"root must lie in [0, {self._size}); got {root}")
        if source == root:
            return obj if self._rank == root else None
        if self._rank == source:
            self._tick("p2p_send")
            self._comm.send(obj, dest=root, tag=_OBJECT_TO_ROOT_TAG)
            return None
        if self._rank == root:
            self._tick("p2p_recv")
            return self._comm.recv(source=source, tag=_OBJECT_TO_ROOT_TAG)
        return None

    def allreduce_max(self, value: float) -> float:
        local = float(value)
        # MPI_MAX over NaN is implementation-defined, so NaN is handled
        # explicitly: lane 0 reduces a 0/1 NaN flag, lane 1 reduces the value
        # with NaN replaced by -inf (inert under MAX). One round.
        is_nan = math.isnan(local)
        lanes = np.array(
            [1.0 if is_nan else 0.0, -math.inf if is_nan else local],
            dtype=np.float64,
        )
        out = np.empty_like(lanes)
        self._tick("allreduce")
        self._comm.Allreduce(lanes, out, op=self._mpi.MAX)
        return math.nan if out[0] > 0.0 else float(out[1])

    def sum_reproducible(
        self, values: np.ndarray, global_ids: np.ndarray
    ) -> np.ndarray | float:
        # Global-id windows -> root canonical_sum per bounded window -> broadcast.
        # The root never materializes the full row pool; each window is sized by
        # canonical_sum_window_rows(width), and the window order matches
        # canonical_sum's deterministic grouping.
        comm, mpi, root = self._comm, self._mpi, 0
        vals = np.asarray(values, dtype=np.float64)
        if vals.ndim not in (1, 2):
            raise ValueError(
                "sum_reproducible: values must have shape (n,) or (n, M);"
                f" rank {self._rank} passed shape {vals.shape}"
            )
        vals = np.ascontiguousarray(vals)
        ids0 = np.asarray(global_ids)
        if ids0.ndim != 1 or not np.issubdtype(ids0.dtype, np.integer):
            raise ValueError(
                "sum_reproducible: global_ids must be a 1-D integer array;"
                f" rank {self._rank} passed shape {ids0.shape}, dtype {ids0.dtype}"
            )
        ids = np.ascontiguousarray(ids0, dtype=np.int64)
        is2d = vals.ndim == 2
        width = int(vals.shape[1]) if is2d else 1
        n_local = int(vals.shape[0])
        if ids.shape != (n_local,):
            raise ValueError(
                f"sum_reproducible: one global id per contribution row required;"
                f" rank {self._rank} has {n_local} rows but {ids.size} ids"
            )

        if n_local:
            local_min = int(ids.min())
            local_max = int(ids.max())
            if n_local == 1 or bool(np.all(ids[1:] >= ids[:-1])):
                sorted_ids = ids
                sorted_vals = vals
            else:
                order = np.argsort(ids, kind="stable")
                sorted_ids = ids[order]
                sorted_vals = vals[order]
        else:
            local_min = 0
            local_max = -1
            sorted_ids = ids
            sorted_vals = vals

        # One small allgather agrees the layout and id span, O(size) not O(N):
        # each rank's row count + (is2d, width, min_id, max_id).
        meta_local = np.array(
            [n_local, int(is2d), width, local_min, local_max], dtype=np.int64
        )
        meta = np.empty((self._size, 5), dtype=np.int64)
        self._tick("allgather")
        comm.Allgather([meta_local, mpi.INT64_T], [meta, mpi.INT64_T])
        counts = meta[:, 0]
        non_empty = counts > 0
        non_empty_shapes = {(bool(row[1]), int(row[2])) for row in meta[non_empty]}
        if len(non_empty_shapes) > 1:
            raise ValueError(
                "sum_reproducible: non-empty ranks must agree on"
                f" contribution shape; got {sorted(non_empty_shapes)}"
            )
        if non_empty_shapes:
            any2d, agreed_width = next(iter(non_empty_shapes))
        else:
            any2d = bool(meta[:, 1].any())
            agreed_width = int(meta[:, 2].max()) if any2d else 1
        out = np.empty(agreed_width, dtype=np.float64)
        out.fill(0.0)
        if self._rank == root:
            chunk_counts = np.empty(self._size, dtype=np.int64)
        else:
            chunk_counts = None

        total = int(counts.sum())
        if total:
            non_empty_meta = meta[non_empty]
            global_min = int(non_empty_meta[:, 3].min())
            global_max = int(non_empty_meta[:, 4].max())
            window_rows = canonical_sum_window_rows(agreed_width)
            lo = global_min
            stop = global_max + 1
            while lo < stop:
                hi = lo + window_rows
                left = int(np.searchsorted(sorted_ids, lo, side="left"))
                right = int(np.searchsorted(sorted_ids, hi, side="left"))
                chunk_ids = sorted_ids[left:right]
                chunk_vals = sorted_vals[left:right]
                n_chunk = np.array([chunk_ids.size], dtype=np.int64)

                self._tick("gather")
                comm.Gather(
                    [n_chunk, mpi.INT64_T],
                    [chunk_counts, mpi.INT64_T] if self._rank == root else None,
                    root=root,
                )
                flat_chunk = np.ascontiguousarray(chunk_vals).ravel()
                if self._rank == root:
                    assert chunk_counts is not None
                    win_total = int(chunk_counts.sum())
                    recv_vals = np.empty(win_total * agreed_width, dtype=np.float64)
                    recv_ids = np.empty(win_total, dtype=np.int64)
                    val_counts = chunk_counts * agreed_width
                    val_displs = np.concatenate(([0], np.cumsum(val_counts)[:-1]))
                    id_displs = np.concatenate(([0], np.cumsum(chunk_counts)[:-1]))
                    recv_val_spec = [recv_vals, val_counts, val_displs, mpi.DOUBLE]
                    recv_id_spec = [recv_ids, chunk_counts, id_displs, mpi.INT64_T]
                else:
                    recv_val_spec = recv_id_spec = None

                self._tick("gather")
                comm.Gatherv([flat_chunk, mpi.DOUBLE], recv_val_spec, root=root)
                self._tick("gather")
                comm.Gatherv([chunk_ids, mpi.INT64_T], recv_id_spec, root=root)
                error: str | None = None
                if self._rank == root and win_total:
                    pooled = (
                        recv_vals.reshape(win_total, agreed_width)
                        if any2d
                        else recv_vals
                    )
                    try:
                        out += np.atleast_1d(
                            np.asarray(
                                canonical_sum(pooled, recv_ids), dtype=np.float64
                            )
                        )
                    except ValueError as exc:
                        error = f"sum_reproducible: {exc}"
                self._tick("bcast")
                error = comm.bcast(error if self._rank == root else None, root=root)
                if error is not None:
                    raise ValueError(error)
                lo = hi

        self._tick("bcast")
        comm.Bcast([out, mpi.DOUBLE], root=root)
        return out if any2d else float(out[0])

    def sum_vectors_reproducible(self, values: np.ndarray) -> np.ndarray:
        vals = np.ascontiguousarray(values, dtype=np.float64)
        if vals.ndim == 0:
            raise ValueError("sum_vectors_reproducible: values must be an array")
        if vals.ndim > 2:
            raise ValueError(
                "sum_vectors_reproducible: values must have shape (M,) or"
                f" (B, M); got {vals.shape}"
            )
        dim1 = int(vals.shape[1]) if vals.ndim == 2 else 0
        meta_local = np.array([vals.ndim, int(vals.shape[0]), dim1], dtype=np.int64)
        meta = np.empty((self._size, 3), dtype=np.int64)
        self._tick("allgather")
        self._comm.Allgather([meta_local, self._mpi.INT64_T], [meta, self._mpi.INT64_T])
        if np.any(meta != meta[0]):
            shapes = [
                (int(row[1]), int(row[2])) if int(row[0]) == 2 else (int(row[1]),)
                for row in meta
            ]
            raise ValueError(
                "sum_vectors_reproducible: every rank must pass the same"
                f" shape; got {shapes}"
            )
        flat = vals.ravel()
        if flat.size == 0:
            return vals.copy()
        gathered = np.empty((self._size, flat.size), dtype=np.float64)
        self._tick("allgather")
        self._comm.Allgather([flat, self._mpi.DOUBLE], [gathered, self._mpi.DOUBLE])
        rank_ids = np.arange(self._size, dtype=np.int64)
        reduced = np.asarray(canonical_sum(gathered, rank_ids), dtype=np.float64)
        return reduced.reshape(vals.shape)

    def scatter_by_agent(
        self,
        arrays: dict[str, np.ndarray] | None,
        local_ids: np.ndarray,
        *,
        root: int = 0,
    ) -> dict[str, np.ndarray]:
        if not 0 <= root < self._size:
            raise ValueError(f"root must lie in [0, {self._size}); got {root}")
        ids = np.asarray(local_ids)
        # Only (presence flag, ids) travel to root; the full arrays never leave
        # the publishing rank whole.
        self._tick("gather")
        gathered: list[tuple[bool, np.ndarray]] | None = self._comm.gather(
            (arrays is not None, ids), root=root
        )
        # Root validates once and broadcasts one verdict (the wire header or a
        # shared error text), so a bad call raises identically on every rank.
        verdict: tuple[str, Any] | None = None
        normalized: dict[str, np.ndarray] = {}
        if self._rank == root and gathered is not None:
            normalized, (tag, payload) = _scatter_verdict(
                arrays, gathered, self._size, root
            )
            verdict = (tag, (self._scatter_chunk_bytes, payload))
        self._tick("bcast")
        tag, payload = self._comm.bcast(verdict, root=root)
        chunk_bytes, header = payload
        if tag == "error":
            raise ValueError(header)
        if self._rank == root and gathered is not None:
            return self._scatter_stream_root(normalized, gathered, chunk_bytes, root)
        return self._scatter_recv(header, int(ids.shape[0]), chunk_bytes, root)

    def _scatter_stream_root(
        self,
        normalized: dict[str, np.ndarray],
        gathered: Sequence[tuple[bool, np.ndarray]],
        chunk_bytes: int,
        root: int,
    ) -> dict[str, np.ndarray]:
        """Stream each key to each destination in bounded chunk windows.

        Point-to-point ``Isend`` per (destination, chunk), not a per-chunk
        ``Scatterv``: shard sizes are non-uniform, and independent channels let
        each destination's stream end when its rows do. A Waitall between
        windows caps the root's in-flight transient at one chunk per
        destination.
        """
        byte = self._mpi.BYTE
        out: dict[str, np.ndarray] = {}
        for key, full in normalized.items():
            row_nbytes = int(full.dtype.itemsize) * int(
                np.prod(full.shape[1:], dtype=np.int64)
            )
            # Root's own shard never touches the wire; the fancy-index yields
            # fresh rows, so nothing aliases the caller's originals.
            own = full[np.asarray(gathered[root][1])]
            own.setflags(write=False)
            out[key] = own
            spans = [
                _chunk_spans(int(np.asarray(ids).shape[0]), row_nbytes, chunk_bytes)
                for _, ids in gathered
            ]
            depth = max(
                (len(s) for r, s in enumerate(spans) if r != root),
                default=0,
            )
            for w in range(depth):
                in_flight: list[np.ndarray] = []
                requests: list[Any] = []
                for r in range(self._size):
                    if r == root:
                        continue
                    if w >= len(spans[r]):
                        continue  # destination r's stream already ended
                    start, stop = spans[r][w]
                    rank_ids = np.asarray(gathered[r][1])
                    # The chunk buffer must outlive its Isend, hence the
                    # in-flight list dropped only after Waitall.
                    chunk = np.ascontiguousarray(full[rank_ids[start:stop]])
                    in_flight.append(chunk)
                    self._tick("p2p_send")
                    requests.append(
                        self._comm.Isend(
                            [chunk, byte],
                            dest=r,
                            tag=_SCATTER_STREAM_TAG,
                        )
                    )
                self._mpi.Request.Waitall(requests)
        return out

    def _scatter_recv(
        self,
        header: Sequence[tuple[str, np.dtype, tuple[int, ...]]],
        n_local: int,
        chunk_bytes: int,
        root: int,
    ) -> dict[str, np.ndarray]:
        byte = self._mpi.BYTE
        out: dict[str, np.ndarray] = {}
        for key, dtype, trailing in header:
            row_nbytes = int(dtype.itemsize) * int(np.prod(trailing, dtype=np.int64))
            recv = np.empty((n_local, *trailing), dtype=dtype)
            for start, stop in _chunk_spans(n_local, row_nbytes, chunk_bytes):
                self._tick("p2p_recv")
                # Axis-0 slices of a fresh C-contiguous array are contiguous, so
                # each chunk lands in place with no extra copy. Raw bytes
                # preserve the dtype's exact bit patterns.
                self._comm.Recv(
                    [recv[start:stop], byte],
                    source=root,
                    tag=_SCATTER_STREAM_TAG,
                )
            recv.setflags(write=False)
            out[key] = recv
        return out

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
        if n_global < 0:
            raise ValueError(f"n_global must be >= 0; got {n_global}")
        vals = np.ascontiguousarray(values, dtype=np.float64)
        ids0 = np.asarray(global_ids)
        if ids0.ndim != 1 or not np.issubdtype(ids0.dtype, np.integer):
            raise ValueError(
                "gather_agent_values: global_ids must be a 1-D integer array;"
                f" rank {self._rank} passed shape {ids0.shape}, dtype {ids0.dtype}"
            )
        ids = np.ascontiguousarray(ids0, dtype=np.int64)
        if vals.ndim != 1:
            raise ValueError(
                "gather_agent_values: values must be one-dimensional;"
                f" rank {self._rank} passed shape {vals.shape}"
            )
        if ids.shape != vals.shape:
            raise ValueError(
                "gather_agent_values: values and global_ids must have the same"
                f" shape; rank {self._rank} has {vals.shape} and {ids.shape}"
            )
        if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= int(n_global)):
            raise ValueError(
                "gather_agent_values: global_ids must lie in"
                f" [0, {n_global}); rank {self._rank} got range"
                f" [{int(ids.min())}, {int(ids.max())}]"
            )

        meta_local = np.array([vals.size, int(n_global)], dtype=np.int64)
        meta = np.empty((self._size, 2), dtype=np.int64)
        self._tick("allgather")
        self._comm.Allgather([meta_local, self._mpi.INT64_T], [meta, self._mpi.INT64_T])
        sizes = set(int(v) for v in meta[:, 1])
        if len(sizes) != 1:
            raise ValueError(
                "gather_agent_values: every rank must pass the same"
                f" n_global; got {sorted(sizes)}"
            )
        counts = meta[:, 0]
        total = int(counts.sum())
        displs = np.concatenate(([0], np.cumsum(counts)[:-1]))
        if self._rank == root:
            recv_vals = np.empty(total, dtype=np.float64)
            recv_ids = np.empty(total, dtype=np.int64)
            recv_val_spec = [recv_vals, counts, displs, self._mpi.DOUBLE]
            recv_id_spec = [recv_ids, counts, displs, self._mpi.INT64_T]
        else:
            recv_val_spec = recv_id_spec = None
        self._tick("gather")
        self._comm.Gatherv([vals, self._mpi.DOUBLE], recv_val_spec, root=root)
        self._tick("gather")
        self._comm.Gatherv([ids, self._mpi.INT64_T], recv_id_spec, root=root)
        error: str | None = None
        if self._rank == root and recv_ids.size:
            unique, dup_counts = np.unique(recv_ids, return_counts=True)
            if unique.size != recv_ids.size:
                error = (
                    "gather_agent_values: duplicate global_ids:"
                    f" {unique[dup_counts > 1].tolist()}"
                )
        self._tick("bcast")
        error = self._comm.bcast(error if self._rank == root else None, root=root)
        if error is not None:
            raise ValueError(error)
        if self._rank != root:
            return None
        out = np.zeros(int(n_global), dtype=np.float64)
        out[recv_ids] = recv_vals
        out.setflags(write=False)
        return out

    def _agree_route_preflight(
        self,
        *,
        n_agents: object,
        source: object,
        local_error: str | None,
    ) -> tuple[int, int]:
        try:
            n_agents_i, src = _route_agent_axis_validated(
                n_agents,
                size=self._size,
                source=source,
                what="route_agent_values",
            )
        except ValueError as exc:
            n_agents_i, src = 1, 0
            local_error = str(exc) if local_error is None else local_error
        reporter = self._rank if local_error is not None else self._size
        lanes = np.array(
            [n_agents_i, -n_agents_i, src, -src, reporter],
            dtype=np.int64,
        )
        out = np.empty_like(lanes)
        self._tick("allreduce")
        self._comm.Allreduce(lanes, out, op=self._mpi.MIN)
        if int(out[0]) != -int(out[1]):
            raise ValueError("route_agent_values: n_agents must be identical on every rank")
        if int(out[2]) != -int(out[3]):
            raise ValueError(
                "route_agent_values: source must be identical on every rank"
            )
        agreed_reporter = int(out[4])
        if agreed_reporter != self._size:
            self._tick("bcast")
            message = self._comm.bcast(
                local_error if self._rank == agreed_reporter else None,
                root=agreed_reporter,
            )
            raise ValueError(message)
        return int(out[0]), int(out[2])

    def _agree_route_error(self, local_error: str | None) -> None:
        reporter = self._rank if local_error is not None else self._size
        lane = np.array([reporter], dtype=np.int64)
        out = np.empty_like(lane)
        self._tick("allreduce")
        self._comm.Allreduce(lane, out, op=self._mpi.MIN)
        agreed_reporter = int(out[0])
        if agreed_reporter == self._size:
            return
        self._tick("bcast")
        message = self._comm.bcast(
            local_error if self._rank == agreed_reporter else None,
            root=agreed_reporter,
        )
        raise ValueError(message)

    def _agree_owner_vector(self, owners: np.ndarray, *, what: str) -> None:
        """Verify rank-identical owner vectors with fixed-size wire payload."""
        if owners.size == 0:
            return
        canon = np.ascontiguousarray(owners, dtype="<i8")
        h = hashlib.blake2b(digest_size=16, person=b"combrum-owners")
        h.update(struct.pack("<Q", int(canon.size)))
        h.update(canon.tobytes())
        digest = h.digest()
        sig0 = int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)
        sig1 = int.from_bytes(digest[8:], "little") & ((1 << 63) - 1)
        lanes = np.array([sig0, -sig0, sig1, -sig1], dtype=np.int64)
        out = np.empty_like(lanes)
        self._tick("allreduce")
        self._comm.Allreduce(lanes, out, op=self._mpi.MIN)
        if int(out[0]) != -int(out[1]) or int(out[2]) != -int(out[3]):
            raise ValueError(f"{what}: owners must be identical on every rank")

    def route_agent_values(
        self,
        values: Mapping[int, float] | None,
        local_ids: np.ndarray,
        *,
        source: int,
        n_agents: int,
    ) -> dict[int, float]:
        what = "route_agent_values"
        local_error: str | None = None
        try:
            n_agents0, src0 = _route_agent_axis_validated(
                n_agents,
                size=self._size,
                source=source,
                what=what,
            )
            _route_local_ids_owned_validated(
                local_ids,
                n_agents=n_agents0,
                rank=self._rank,
                size=self._size,
                what=what,
            )
            normalized = _route_values_validated(
                values,
                n_agents=n_agents0,
                rank=self._rank,
                source=src0,
                what=what,
            )
        except Exception as exc:
            normalized = {}
            if local_error is None:
                local_error = f"{type(exc).__name__}: {exc}"
        n_agents_i, src = self._agree_route_preflight(
            n_agents=n_agents,
            source=source,
            local_error=local_error,
        )

        if self._rank == src:
            gids = np.fromiter(normalized.keys(), dtype=np.int64, count=len(normalized))
            vals = np.fromiter(
                normalized.values(), dtype=np.float64, count=len(normalized)
            )
            order = np.argsort(gids)
            gids = gids[order]
            vals = vals[order]
            # Owner rank is monotone in gid under contiguous sharding, so gid
            # order is already owner-bucketed wire order.
            owners = _agent_owner_ranks(gids, n_agents_i, self._size)
            send_counts = np.bincount(owners, minlength=self._size).astype(np.int64)
            send = np.empty(gids.size, dtype=_ROUTE_VALUE_DTYPE)
            send["gid"] = gids
            send["value"] = vals
        else:
            send_counts = np.zeros(self._size, dtype=np.int64)
            send = np.empty(0, dtype=_ROUTE_VALUE_DTYPE)

        recv_counts = np.empty(self._size, dtype=np.int64)
        self._tick("alltoall")
        self._comm.Alltoall(send_counts, recv_counts)
        itemsize = int(_ROUTE_VALUE_DTYPE.itemsize)
        send_byte_counts = send_counts * itemsize
        recv_byte_counts = recv_counts * itemsize
        send_byte_displs = np.concatenate(([0], np.cumsum(send_byte_counts)[:-1]))
        recv_byte_displs = np.concatenate(([0], np.cumsum(recv_byte_counts)[:-1]))
        recv = np.empty(int(recv_counts.sum()), dtype=_ROUTE_VALUE_DTYPE)
        byte = self._mpi.BYTE
        self._tick("alltoallv")
        self._comm.Alltoallv(
            [send.view(np.uint8), send_byte_counts, send_byte_displs, byte],
            [recv.view(np.uint8), recv_byte_counts, recv_byte_displs, byte],
        )

        out: dict[int, float] = {}
        local_error = None
        gids = recv["gid"]
        oob = (gids < 0) | (gids >= n_agents_i)
        owners = _agent_owner_ranks(
            np.clip(gids, 0, n_agents_i - 1), n_agents_i, self._size
        )
        misrouted = ~oob & (owners != self._rank)
        bad = oob | misrouted
        if bad.any():
            first = int(np.flatnonzero(bad)[0])
            gid = int(gids[first])
            if oob[first]:
                local_error = f"{what}: received out-of-range agent id {gid}"
            else:
                local_error = (
                    f"{what}: received agent {gid} on non-owner rank {self._rank}"
                )
        else:
            out = dict(zip(gids.tolist(), recv["value"].tolist()))
        self._agree_route_error(local_error)
        return out

    def _checked_route_batch_inputs(
        self,
        values_by_rep: Mapping[int, Mapping[int, float]] | None,
        local_ids: np.ndarray,
        *,
        owners: np.ndarray,
        n_agents: int,
    ) -> tuple[int, np.ndarray, dict[int, dict[int, float]]]:
        what = "route_agent_values_batched"
        n_agents_lane = -1
        try:
            n_agents0, _src0 = _route_agent_axis_validated(
                n_agents,
                size=self._size,
                source=0,
                what=what,
            )
            _route_local_ids_owned_validated(
                local_ids,
                n_agents=n_agents0,
                rank=self._rank,
                size=self._size,
                what=what,
            )
            owner_arr = np.asarray(owners)
            if owner_arr.ndim != 1 or not np.issubdtype(owner_arr.dtype, np.integer):
                raise ValueError(
                    f"{what}: owners must be a 1-D integer array of ranks;"
                    f" got shape {owner_arr.shape}, dtype {owner_arr.dtype}"
                )
            canon = np.ascontiguousarray(owner_arr, dtype=np.int64)
            if canon.size and (
                int(canon.min()) < 0 or int(canon.max()) >= self._size
            ):
                raise ValueError(
                    f"{what}: owners must lie in [0, size) = [0, {self._size});"
                    f" got range [{int(canon.min())}, {int(canon.max())}]"
                )
            payload: object = {} if values_by_rep is None else values_by_rep
            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"{what}: values_by_rep must be a mapping or None;"
                    f" got {type(payload).__name__}"
                )
            normalized: dict[int, dict[int, float]] = {}
            B = int(canon.size)
            for rep_key, values in payload.items():
                if (
                    isinstance(rep_key, (bool, np.bool_))
                    or not isinstance(rep_key, (int, np.integer))
                    or int(rep_key) < 0
                    or int(rep_key) >= B
                ):
                    raise ValueError(
                        f"{what}: replication ids must lie in [0, {B});"
                        f" got {rep_key!r}"
                    )
                rep = int(rep_key)
                if int(canon[rep]) != self._rank:
                    raise ValueError(
                        f"{what}: only owner rank {int(canon[rep])} may pass"
                        f" values for replication {rep}; rank {self._rank}"
                        " passed values"
                    )
                normalized[rep] = _route_values_validated(
                    values,
                    n_agents=n_agents0,
                    rank=self._rank,
                    source=self._rank,
                    what=what,
                )
            n_agents_lane = n_agents0
            local_error: str | None = None
        except Exception as exc:
            canon = np.empty(0, dtype=np.int64)
            normalized = {}
            local_error = f"{type(exc).__name__}: {exc}"
            try:
                n_agents_lane = int(n_agents)
            except (TypeError, ValueError):
                n_agents_lane = -1
        B = int(canon.size)
        reporter = self._rank if local_error is not None else self._size
        lanes = np.array(
            [
                n_agents_lane,
                -n_agents_lane,
                B,
                -B,
                reporter,
            ],
            dtype=np.int64,
        )
        out = np.empty_like(lanes)
        self._tick("allreduce")
        self._comm.Allreduce(lanes, out, op=self._mpi.MIN)
        agreed_reporter = int(out[4])
        if agreed_reporter != self._size:
            self._tick("bcast")
            message = self._comm.bcast(
                local_error if self._rank == agreed_reporter else None,
                root=agreed_reporter,
            )
            raise ValueError(message)
        if int(out[0]) != -int(out[1]):
            raise ValueError(f"{what}: n_agents must be identical on every rank")
        if int(out[2]) != -int(out[3]):
            raise ValueError(f"{what}: owners must have the same length on every rank")
        self._agree_owner_vector(canon, what=what)
        return int(out[0]), canon, normalized

    def route_agent_values_batched(
        self,
        values_by_rep: Mapping[int, Mapping[int, float]] | None,
        local_ids: np.ndarray,
        *,
        owners: np.ndarray,
        n_agents: int,
    ) -> dict[int, dict[int, float]]:
        (
            n_agents_i,
            agreed,
            normalized,
        ) = self._checked_route_batch_inputs(
            values_by_rep,
            local_ids,
            owners=owners,
            n_agents=n_agents,
        )
        if normalized:
            rep_parts = []
            gid_parts = []
            val_parts = []
            for rep in sorted(normalized):
                values = normalized[rep]
                gids = np.fromiter(values.keys(), dtype=np.int64, count=len(values))
                vals = np.fromiter(values.values(), dtype=np.float64, count=len(values))
                order = np.argsort(gids)
                rep_parts.append(np.full(gids.size, rep, dtype=np.int64))
                gid_parts.append(gids[order])
                val_parts.append(vals[order])
            reps = np.concatenate(rep_parts)
            gids = np.concatenate(gid_parts)
            vals = np.concatenate(val_parts)
            owners = _agent_owner_ranks(gids, n_agents_i, self._size)
            # Stable sort by owner keeps (rep asc, gid asc) order within each
            # destination bucket, matching the per-bucket append wire order.
            order = np.argsort(owners, kind="stable")
            send_counts = np.bincount(owners, minlength=self._size).astype(np.int64)
            send = np.empty(gids.size, dtype=_ROUTE_BATCH_VALUE_DTYPE)
            send["rep"] = reps[order]
            send["gid"] = gids[order]
            send["value"] = vals[order]
        else:
            send_counts = np.zeros(self._size, dtype=np.int64)
            send = np.empty(0, dtype=_ROUTE_BATCH_VALUE_DTYPE)

        recv_counts = np.empty(self._size, dtype=np.int64)
        self._tick("alltoall")
        self._comm.Alltoall(send_counts, recv_counts)
        itemsize = int(_ROUTE_BATCH_VALUE_DTYPE.itemsize)
        send_byte_counts = send_counts * itemsize
        recv_byte_counts = recv_counts * itemsize
        send_byte_displs = np.concatenate(([0], np.cumsum(send_byte_counts)[:-1]))
        recv_byte_displs = np.concatenate(([0], np.cumsum(recv_byte_counts)[:-1]))
        recv = np.empty(int(recv_counts.sum()), dtype=_ROUTE_BATCH_VALUE_DTYPE)
        byte = self._mpi.BYTE
        self._tick("alltoallv")
        self._comm.Alltoallv(
            [send.view(np.uint8), send_byte_counts, send_byte_displs, byte],
            [recv.view(np.uint8), recv_byte_counts, recv_byte_displs, byte],
        )

        out: dict[int, dict[int, float]] = {}
        local_error = None
        B = int(agreed.size)
        reps = recv["rep"]
        gids = recv["gid"]
        rep_bad = (reps < 0) | (reps >= B)
        oob = (gids < 0) | (gids >= n_agents_i)
        owners = _agent_owner_ranks(
            np.clip(gids, 0, n_agents_i - 1), n_agents_i, self._size
        )
        misrouted = ~oob & (owners != self._rank)
        bad = rep_bad | oob | misrouted
        if bad.any():
            first = int(np.flatnonzero(bad)[0])
            if rep_bad[first]:
                local_error = (
                    f"route_agent_values_batched: received rep {int(reps[first])}"
                )
            elif oob[first]:
                local_error = (
                    "route_agent_values_batched: received out-of-range"
                    f" agent id {int(gids[first])}"
                )
            else:
                local_error = (
                    "route_agent_values_batched: received agent"
                    f" {int(gids[first])} on non-owner rank {self._rank}"
                )
        else:
            for rep, gid, value in zip(
                reps.tolist(), gids.tolist(), recv["value"].tolist()
            ):
                out.setdefault(rep, {})[gid] = value
        self._agree_route_error(local_error)
        return out

    def node_shared(self, arrays: dict[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        node_comm = self._node_comm
        if node_comm is None:
            raise RuntimeError(
                "node_shared: the transport is closed; shared windows"
                " need the node communicator"
            )
        mpi = self._mpi
        # Verdict agreed across the WHOLE communicator: any node's failure must
        # abort every node before anyone enters the window collectives. This is
        # a publish-once path, so one small allgather is affordable.
        verdict: tuple[int, str] | None = None
        staged: dict[str, np.ndarray] = {}
        if self._node.node_rank == 0:
            try:
                staged = _node_arrays_validated(arrays)
            except Exception as exc:
                verdict = (self._rank, f"{type(exc).__name__}: {exc}")
        self._tick("allgather")
        verdicts: list[tuple[int, str] | None] = self._comm.allgather(verdict)
        failure = next((v for v in verdicts if v is not None), None)
        if failure is not None:
            # allgather is rank-ordered, so `next` elects the lowest failer.
            origin, message = failure
            raise ValueError(
                f"node_shared: publishing failed on rank {origin}: {message}"
            )
        # The publisher alone defines the layout; peers learn it in one
        # node-local broadcast.
        self._tick("bcast")
        layout, total = node_comm.bcast(
            _window_layout(staged) if self._node.node_rank == 0 else None,
            root=0,
        )
        # One window per call, allocated wholly on the publisher: peers request
        # ZERO bytes and map the publisher's segment, so the node holds one
        # physical copy.
        self._tick("win_allocate_shared")
        win: Win = mpi.Win.Allocate_shared(
            total if self._node.node_rank == 0 else 0, 1, comm=node_comm
        )
        self._windows.append(win)
        # Lifetime-long passive epoch (shared lock, NOCHECK since no rank ever
        # takes an exclusive lock); close() ends it before freeing.
        self._tick("win_lock_all")
        win.Lock_all(mpi.MODE_NOCHECK)
        buf, _ = win.Shared_query(0)
        if self._node.node_rank == 0:
            writable = memoryview(buf)
            for (_key, dtype, shape, offset), src in zip(layout, staged.values()):
                count = int(np.prod(shape, dtype=np.int64))
                dest = np.frombuffer(
                    writable, dtype=dtype, count=count, offset=offset
                ).reshape(shape)
                # The single publish copy: caller's array straight into the
                # shared pages, no staging duplicate.
                np.copyto(dest, src, casting="no")
        # Write-once-then-read-only needs one sync episode: sync (stores reach
        # the window), barrier (no read before every store), sync (readers see
        # them). Sufficient on the unified shared-window model; no rank writes
        # afterward, so no per-access synchronization.
        self._tick("win_sync")
        win.Sync()
        self._tick("barrier")
        node_comm.Barrier()
        self._tick("win_sync")
        win.Sync()
        # Read-only views over the same physical pages on every rank; the
        # read-only memoryview makes the arrays' write flag un-flippable.
        ro = memoryview(buf).toreadonly()
        out: dict[str, np.ndarray] = {}
        for key, dtype, shape, offset in layout:
            count = int(np.prod(shape, dtype=np.int64))
            out[key] = np.frombuffer(
                ro, dtype=dtype, count=count, offset=offset
            ).reshape(shape)
        return MappingProxyType(out)

    def batched_max(self, values: np.ndarray) -> np.ndarray:
        error: str | None = None
        try:
            batch = np.ascontiguousarray(values, dtype=np.float64)
            if batch.ndim != 1:
                error = (
                    "batched_max requires the same (B,) shape on every rank;"
                    f" rank {self._rank} passed shape {batch.shape}"
                )
        except (TypeError, ValueError) as exc:
            batch = np.empty(0, dtype=np.float64)
            error = f"batched_max: values must be numeric; {exc}"
        B = int(batch.size) if error is None else -1
        reporter = self._rank if error is not None else self._size
        lanes = np.array([reporter, B, -B], dtype=np.int64)
        out_i = np.empty_like(lanes)
        self._tick("allreduce")
        self._comm.Allreduce(lanes, out_i, op=self._mpi.MIN)
        agreed_reporter = int(out_i[0])
        if agreed_reporter != self._size:
            self._tick("bcast")
            message = self._comm.bcast(
                error if self._rank == agreed_reporter else None,
                root=agreed_reporter,
            )
            raise TransportError(agreed_reporter, message)
        if int(out_i[1]) != -int(out_i[2]):
            raise ValueError(
                "batched_max requires the same (B,) shape on every rank;"
                " ranks disagree on B"
            )
        if B == 0:
            return batch.copy()
        # One vector reduction for the whole batch. MPI_MAX over NaN is
        # implementation-defined, so each slot carries a NaN flag lane and a
        # value lane with NaNs made inert under MAX (matching scalar max).
        flags = np.isnan(batch).astype(np.float64)
        vals = np.where(flags > 0.0, -math.inf, batch)
        lanes = np.concatenate((flags, vals))
        out = np.empty_like(lanes)
        self._tick("allreduce")
        self._comm.Allreduce(lanes, out, op=self._mpi.MAX)
        result = out[batch.size :].copy()
        result[out[: batch.size] > 0.0] = math.nan
        return result

    def _checked_owner_matrix_inputs(
        self, values: np.ndarray, owners: np.ndarray, *, what: str
    ) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(owners)
        error: str | None = None
        if arr.ndim != 1 or not np.issubdtype(arr.dtype, np.integer):
            error = (
                f"{what}: owners must be a 1-D integer array of ranks;"
                f" got shape {arr.shape}, dtype {arr.dtype}"
            )
            canon = np.empty(0, dtype=np.int64)
        else:
            canon = np.ascontiguousarray(arr, dtype=np.int64)
            if canon.size and (int(canon.min()) < 0 or int(canon.max()) >= self._size):
                error = (
                    f"{what}: owners must lie in [0, size) ="
                    f" [0, {self._size}); got range"
                    f" [{int(canon.min())}, {int(canon.max())}]"
                )
        try:
            vals = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            vals = np.empty((0, 0), dtype=np.float64)
            if error is None:
                error = f"{what}: values must be numeric; {exc}"
        B = int(canon.shape[0])
        if error is None and (vals.ndim != 2 or vals.shape[0] != B):
            error = (
                f"{what}: values must have shape (B, M) = ({B}, M);"
                f" rank {self._rank} passed shape {vals.shape}"
            )
        M = int(vals.shape[1]) if error is None else -1
        reporter = self._rank if error is not None else self._size
        lanes = np.array([reporter, B, -B, M, -M], dtype=np.int64)
        out = np.empty_like(lanes)
        self._tick("allreduce")
        self._comm.Allreduce(lanes, out, op=self._mpi.MIN)
        agreed_reporter = int(out[0])
        if agreed_reporter != self._size:
            self._tick("bcast")
            message = self._comm.bcast(
                error if self._rank == agreed_reporter else None,
                root=agreed_reporter,
            )
            raise TransportError(agreed_reporter, message)
        if int(out[1]) != -int(out[2]) or int(out[3]) != -int(out[4]):
            raise ValueError(
                f"{what}: values must have the same (B, M) shape on every rank"
            )
        self._agree_owner_vector(canon, what=what)
        return np.ascontiguousarray(vals, dtype=np.float64), canon

    def _checked_exchange_cut_inputs(
        self, rows: Sequence[CutRow], owners: np.ndarray
    ) -> tuple[tuple[CutRow, ...], np.ndarray]:
        error: str | None = None
        row_tuple: tuple[CutRow, ...] = ()
        arr = np.asarray(owners)
        if arr.ndim != 1 or not np.issubdtype(arr.dtype, np.integer):
            error = (
                "exchange_cuts: owners must be a 1-D integer array of ranks;"
                f" got shape {arr.shape}, dtype {arr.dtype}"
            )
            canon = np.empty(0, dtype=np.int64)
        else:
            canon = np.ascontiguousarray(arr, dtype=np.int64)
            if canon.size and (int(canon.min()) < 0 or int(canon.max()) >= self._size):
                error = (
                    "exchange_cuts: owners must lie in"
                    f" [0, size) = [0, {self._size}); got range"
                    f" [{int(canon.min())}, {int(canon.max())}]"
                )
        if error is None:
            try:
                row_tuple = tuple(rows)
            except TypeError as exc:
                error = (
                    "exchange_cuts: rows must be an iterable of CutRow instances;"
                    f" {type(exc).__name__}: {exc}"
                )
        if error is None:
            B = int(canon.shape[0])
            for row in row_tuple:
                if not isinstance(row, CutRow):
                    error = (
                        "exchange_cuts: rows must be CutRow instances;"
                        f" rank {self._rank} passed {type(row).__name__}"
                    )
                    break
                if row.rep_id >= B:
                    error = (
                        f"exchange_cuts: rep_id {row.rep_id} out of range"
                        f" for {B} live replications (contributed by rank"
                        f" {self._rank})"
                    )
                    break

        B = int(canon.shape[0])
        reporter = self._rank if error is not None else self._size
        lanes = np.array([reporter, B, -B], dtype=np.int64)
        out = np.empty_like(lanes)
        self._tick("allreduce")
        self._comm.Allreduce(lanes, out, op=self._mpi.MIN)
        agreed_reporter = int(out[0])
        if agreed_reporter != self._size:
            self._tick("bcast")
            message = self._comm.bcast(
                error if self._rank == agreed_reporter else None,
                root=agreed_reporter,
            )
            raise TransportError(agreed_reporter, message)
        if int(out[1]) != -int(out[2]):
            raise ValueError(
                "exchange_cuts: owners must have the same length on every rank"
            )
        self._agree_owner_vector(canon, what="exchange_cuts")
        return row_tuple, canon

    def owner_broadcast(self, values: np.ndarray, owners: np.ndarray) -> np.ndarray:
        vals, agreed = self._checked_owner_matrix_inputs(
            values, owners, what="owner_broadcast"
        )
        B = int(vals.shape[0])
        M = int(vals.shape[1])
        if B == 0 or M == 0:
            return np.empty_like(vals)
        mine = np.flatnonzero(agreed == self._rank)
        send = np.ascontiguousarray(vals[mine, :], dtype=np.float64).ravel()
        rows_by_rank = np.bincount(agreed, minlength=self._size).astype(np.int64)
        recv_counts = rows_by_rank * M
        recv_displs = np.concatenate(([0], np.cumsum(recv_counts)[:-1]))
        packed = np.empty(int(recv_counts.sum()), dtype=np.float64)
        self._tick("allgather")
        self._comm.Allgatherv(
            [send, self._mpi.DOUBLE],
            [packed, recv_counts, recv_displs, self._mpi.DOUBLE],
        )
        out = np.empty_like(vals)
        for rank, n_rows in enumerate(rows_by_rank):
            if int(n_rows) == 0:
                continue
            start = int(recv_displs[rank])
            stop = start + int(n_rows) * M
            slots = np.flatnonzero(agreed == rank)
            out[slots, :] = packed[start:stop].reshape(int(n_rows), M)
        return out

    def exchange_cuts(
        self, rows: Sequence[CutRow], owners: np.ndarray
    ) -> tuple[CutRow, ...]:
        rows, canon = self._checked_exchange_cut_inputs(rows, owners)
        send_buf, send_counts = _pack_cuts(rows, canon, self._size)
        recv_counts = np.empty_like(send_counts)
        # Per-destination byte counts depend on what every peer generated, so
        # exchange is one counts Alltoall then one packed-payload Alltoallv:
        # a constant round count regardless of how many rows are live.
        self._tick("alltoall")
        self._comm.Alltoall(send_counts, recv_counts)
        send_displs = np.concatenate(([0], np.cumsum(send_counts)[:-1]))
        recv_buf = np.empty(int(recv_counts.sum()), dtype=np.uint8)
        recv_displs = np.concatenate(([0], np.cumsum(recv_counts)[:-1]))
        byte = self._mpi.BYTE
        self._tick("alltoallv")
        self._comm.Alltoallv(
            [send_buf, send_counts, send_displs, byte],
            [recv_buf, recv_counts, recv_displs, byte],
        )
        received: list[CutRow] = []
        recv_view = memoryview(recv_buf)
        for r in range(self._size):
            start = int(recv_displs[r])
            block = recv_view[start : start + int(recv_counts[r])]
            received.extend(_unpack_cuts(block))
        return canonical_cut_order(received)

    # --- lifecycle --------------------------------------------------------

    def close(self) -> None:
        """Release transport-owned MPI resources; idempotent.

        Frees only what the transport created: every node-shared window, then
        the node communicator. The caller's communicator (or ``COMM_WORLD``) is
        left alone. Freeing a window unmaps the node's shared pages, so views
        from :meth:`node_shared` dangle afterward; keeping readers off a closed
        transport's views is the caller's responsibility.
        """
        windows, self._windows = self._windows, []
        for win in windows:
            # End the passive epoch before the collective Free; SPMD reaches
            # close() on every rank in the same order, so the Frees match up.
            win.Unlock_all()
            win.Free()
        node_comm, self._node_comm = self._node_comm, None
        if node_comm is not None:
            node_comm.Free()
