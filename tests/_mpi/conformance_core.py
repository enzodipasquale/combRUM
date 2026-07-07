"""Shared conformance battery and the mpirun rank programs.

One battery definition serves both sides of the conformance check:
under ``mpirun`` every rank runs it against ``MpiTransport`` and prints
one JSON record; the pytest wrapper replays the same function on the
in-process references and compares records. Every result is rendered
as a digest or bit-hex string — never as a decimal float — so equality
of records is bitwise equality of results.

Beyond the default battery, ``argv[1]`` selects a dedicated rank
program for the memory gates (all of them emit one JSON record per
rank):

* ``chunked CHUNK_BYTES`` — a scatter whose shards span several chunk
  windows at a test-reduced chunk size, checked bitwise against the
  locally regenerated selection, with the per-chunk send/recv tallies;
* ``window-smoke`` — node_shared structural proofs: peers' window
  segments are zero bytes (single physical copy), contents match the
  publisher's bytes across processes, views are un-flippably read-only,
  and close() returns (clean exit is the no-leak/no-hang proof);
* ``rss-ladder N_ROWS CHUNK_BYTES`` — one ladder point of the per-rank
  peak-RSS measurement; the root scatters an (N_ROWS, 16) float64 table
  served from a stream-written memmap so generation never makes the
  table RSS-resident;
* ``node-rss`` — before/after RSS accounting around one ~16 MiB
  node_shared publication.
* ``exchange-bad-row`` — one rank passes an invalid cut row and every rank
  must report the same agreed transport error instead of hanging.
* ``sum-gather-bad-inputs`` — local validation for row-keyed sum/gather
  shape and id-dtype errors before native MPI calls.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

# Executed as a plain script under mpirun this file sits outside the
# package, so resolve the in-repo sources explicitly — and ahead of any
# installed combrum, so the run measures this tree. The tests root carries
# the _support package the skeleton vehicle loads.
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_TESTS = Path(__file__).resolve().parents[1]
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from combrum.transport.base import (  # noqa: E402
    CutRow,
    Transport,
    TransportError,
)
from combrum.transport.mpi import MpiTransport  # noqa: E402

_SEED = 20260612

# Keys whose value must be identical on every rank; the rest are
# rank-dependent by design (topology, and owner-only deliveries). The
# node_shared keys are invariant because every launch here is
# single-node, so all ranks read the same published copy.
INVARIANT_KEYS: tuple[str, ...] = (
    "bcast_root0",
    "bcast_root_last",
    "allreduce_max",
    "allreduce_max_all_negative",
    "allreduce_max_nan",
    "sum_matrix_contiguous",
    "sum_matrix_scrambled",
    "sum_vector_with_empty_rank",
    "sum_vectors_matrix",
    "batched_max",
    "owner_broadcast",
    "scatter_read_only",
    "node_shared",
    "node_shared_read_only",
    "node_shared_no_alias",
    "node_shared_isolated",
    "node_shared_error",
    "guard_clean",
    "guard_failure",
    "guard_nested",
)

# Distribution-invariant pooled sums: the serial reference computes the
# same bytes from the undivided tables (canonical_sum's whole point).
POOLED_SUM_KEYS: tuple[str, ...] = (
    "sum_matrix_contiguous",
    "sum_matrix_scrambled",
    "sum_vector_with_empty_rank",
)


def _rng(stream: int) -> np.random.Generator:
    # One independent, fixed stream per scenario: editing one scenario's
    # draws can never shift another scenario's inputs.
    return np.random.default_rng([_SEED, stream])


def _spread(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    # Magnitudes spread over many orders: the regime where addition order
    # moves last ulps, so digest equality is a real bitwise claim.
    magnitude = rng.uniform(-9.0, 9.0, size=shape)
    sign = rng.choice([-1.0, 1.0], size=shape)
    return sign * 10.0**magnitude


def _digest(arr: np.ndarray) -> dict[str, Any]:
    a = np.ascontiguousarray(arr)
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
    }


def _bits(x: float) -> str:
    return np.float64(x).tobytes().hex()


def _local_agent_ids(
    n_observations: int, n_simulations: int, rank: int, size: int
) -> np.ndarray:
    n_agents = int(n_observations) * int(n_simulations)
    base, extra = divmod(n_agents, int(size))
    start = int(rank) * base + min(int(rank), extra)
    stop = start + base + (1 if int(rank) < extra else 0)
    return np.arange(start, stop, dtype=np.int64)


def _cut_sig(row: CutRow) -> list[Any]:
    return [
        row.rep_id,
        row.agent_id,
        row.bundle_key.hex(),
        list(row.phi.shape),
        hashlib.sha256(row.phi.tobytes()).hexdigest(),
        _bits(row.epsilon),
    ]


def battery(t: Transport) -> dict[str, Any]:
    """The fixed scenario battery: a pure function of (rank, size).

    Every input derives from seeded tables shared by all ranks, so the
    same size yields the same per-rank records on any conforming
    transport.
    """
    rank, size = t.rank, t.size
    out: dict[str, Any] = {}

    out["topology"] = [
        rank,
        size,
        t.node.node_id,
        t.node.node_rank,
        t.node.node_size,
        t.node.n_nodes,
    ]

    theta = _spread(_rng(1), (5,))
    got = t.bcast({"theta": theta, "step": 7} if rank == 0 else None, root=0)
    out["bcast_root0"] = {
        "theta": _digest(got["theta"]),
        "step": int(got["step"]),
    }
    last = size - 1
    mark = t.bcast(
        {"mark": float(last) + 0.5} if rank == last else None, root=last
    )
    out["bcast_root_last"] = _bits(mark["mark"])

    peaks = _spread(_rng(2), (size,))
    out["allreduce_max"] = _bits(t.allreduce_max(float(peaks[rank])))
    out["allreduce_max_all_negative"] = _bits(
        t.allreduce_max(-abs(float(peaks[rank])) - 1.0)
    )
    nan_rank = size // 2
    out["allreduce_max_nan"] = _bits(
        t.allreduce_max(
            float("nan") if rank == nan_rank else float(peaks[rank])
        )
    )

    n, m = 97, 3
    table = _spread(_rng(3), (n, m))
    ids = _rng(4).permutation(n)
    contiguous = np.array_split(np.arange(n), size)[rank]
    out["sum_matrix_contiguous"] = _digest(
        np.asarray(t.sum_reproducible(table[contiguous], ids[contiguous]))
    )
    scrambled = _rng(5).permutation(n)[rank::size]
    out["sum_matrix_scrambled"] = _digest(
        np.asarray(t.sum_reproducible(table[scrambled], ids[scrambled]))
    )
    vector = _spread(_rng(6), (n,))
    assign = np.arange(n) % size
    if size > 1:
        assign[assign == 1] = 0  # rank 1 contributes nothing: empty is legal
    chunk = np.flatnonzero(assign == rank)
    out["sum_vector_with_empty_rank"] = _bits(
        float(t.sum_reproducible(vector[chunk], ids[chunk]))
    )

    rank_vectors = _spread(_rng(17), (size, 2, 4))
    out["sum_vectors_matrix"] = _digest(
        t.sum_vectors_reproducible(rank_vectors[rank])
    )

    batch = _spread(_rng(7), (size, 6))
    out["batched_max"] = _digest(t.batched_max(batch[rank]))

    B = size + 2  # more replications than ranks: owners repeat
    owners = (np.arange(B) % size).astype(np.int64)
    owner_rows = np.full((B, 3), 12345.0 + rank, dtype=np.float64)
    for b, owner in enumerate(owners):
        if int(owner) == rank:
            owner_rows[b] = np.array([float(b), float(rank), float(size + b)])
    out["owner_broadcast"] = _digest(t.owner_broadcast(owner_rows, owners))

    route_n, route_s = 7, 3
    route_source = size - 1
    route_values = {
        0: 1.25,
        6: -2.5,
        8: np.float64(3.75),
        14: -4.5,
        20: 5.25,
    }
    routed = t.route_agent_values(
        route_values if rank == route_source else None,
        _local_agent_ids(route_n, route_s, rank, size),
        source=route_source,
        n_agents=route_n * route_s,
    )
    out["route_agent_values"] = {
        str(gid): _bits(value) for gid, value in sorted(routed.items())
    }
    route_batch_owners = ((np.arange(B) + 1) % size).astype(np.int64)
    route_batch_payload = {
        rep: {
            int((rep * 3) % (route_n * route_s)): float(rep + 1),
            int(route_n * route_s - 1 - rep): float(-(rep + 1)),
        }
        for rep, owner in enumerate(route_batch_owners)
        if int(owner) == rank
    }
    routed_batch = t.route_agent_values_batched(
        route_batch_payload,
        _local_agent_ids(route_n, route_s, rank, size),
        owners=route_batch_owners,
        n_agents=route_n * route_s,
    )
    out["route_agent_values_batched"] = {
        str(rep): {
            str(gid): _bits(value) for gid, value in sorted(values.items())
        }
        for rep, values in sorted(routed_batch.items())
    }

    cut_owners = ((np.arange(B) + 1) % size).astype(np.int64)
    phi_table = _spread(_rng(9), (size, B, 4))
    rows = [
        # The same canonical key from every rank: the owner must receive
        # all of them, ordered by contributing rank.
        CutRow(
            rep_id=0,
            agent_id=5,
            phi=np.full(2, float(rank + 1)),
            epsilon=float(rank),
            bundle_key=b"DUP",
        ),
    ]
    for rep in ((rank + 3) % B, B - 1, rank % B):
        rows.append(
            CutRow(
                rep_id=int(rep),
                agent_id=100 + rank,
                phi=phi_table[rank, int(rep)],
                epsilon=float(rank) / 7.0,
                bundle_key=bytes([65 + rank % 26]),
            )
        )
    received = t.exchange_cuts(rows, cut_owners)
    out["exchange_cuts"] = [_cut_sig(row) for row in received]

    n_global = 11
    full = {
        "x": _spread(_rng(10), (n_global, 2)),
        "w": _spread(_rng(11), (n_global,)),
    }
    share = _rng(12).permutation(n_global)[rank::size]
    pieces = t.scatter_by_agent(full if rank == 0 else None, share)
    out["scatter_by_agent"] = {
        key: _digest(value) for key, value in sorted(pieces.items())
    }
    out["scatter_read_only"] = [
        bool(not value.flags.writeable) for _, value in sorted(pieces.items())
    ]
    # The same pooled tables under contiguous block ids: the mapping
    # contract must be insensitive to how callers slice the agent axis.
    block = np.array_split(np.arange(n_global), size)[rank]
    pieces_block = t.scatter_by_agent(full if rank == 0 else None, block)
    out["scatter_by_agent_contiguous"] = {
        key: _digest(value) for key, value in sorted(pieces_block.items())
    }

    # node_shared: the node's publisher provides the content; every rank
    # regenerates the same seeded tables, so peers' bitwise digests prove
    # the published copy carries the publisher's bytes.
    pub_a = _spread(_rng(13), (17, 3))
    pub_b = _rng(14).integers(-1000, 1000, size=29).astype(np.int32)
    shared = t.node_shared(
        {"a": pub_a, "b": pub_b} if t.node.node_rank == 0 else {}
    )
    out["node_shared"] = {
        key: _digest(np.asarray(value))
        for key, value in sorted(shared.items())
    }
    read_only: list[bool] = []
    for _, value in sorted(shared.items()):
        flipped = True
        try:
            value.setflags(write=True)
            flipped = False
        except ValueError:
            pass
        read_only.append(bool(not value.flags.writeable) and flipped)
    out["node_shared_read_only"] = read_only
    out["node_shared_no_alias"] = bool(
        not np.shares_memory(pub_a, np.asarray(shared["a"]))
    )
    # Publishing copies the input once: mutating the source afterwards
    # must not reach the shared data.
    if t.node.node_rank == 0:
        pub_a[:] = 0.0
    out["node_shared_isolated"] = _digest(np.asarray(shared["a"]))

    try:
        bad = (
            {"a": np.array([1.0]), 2: np.array([2.0])}
            if t.node.node_rank == 0
            else {}
        )
        t.node_shared(bad)
        out["node_shared_error"] = ["no-error"]
    except ValueError as exc:
        out["node_shared_error"] = ["caught", str(exc)]

    with t.collective():
        token = float(rank)
    # A collective right after a clean guard proves the guard consumed
    # exactly its own round on every rank.
    out["guard_clean"] = _bits(t.allreduce_max(token))

    failing = 1 if size > 1 else 0
    try:
        with t.collective():
            if rank == failing:
                raise ValueError("boom-from-the-failing-rank")
        out["guard_failure"] = ["no-error"]
    except TransportError as exc:
        out["guard_failure"] = ["caught", exc.rank, exc.message]

    try:
        with t.collective():
            with t.collective():
                if rank == failing:
                    raise KeyError("nested-origin")
        out["guard_nested"] = ["no-error"]
    except TransportError as exc:
        out["guard_nested"] = ["caught", exc.rank, exc.message]

    return out


def counter_scenario(t: MpiTransport) -> dict[str, dict[str, int]]:
    """Round-shape audit through the transport's own invocation tallies.

    Needs the MpiTransport introspection surface (counts/reset) — the
    in-process references have no MPI primitives to count — so it runs
    only under mpirun and is asserted against fixed expectations, not
    against the references.
    """
    size = t.size
    B = size + 2
    owners = (np.arange(B) % size).astype(np.int64)
    out: dict[str, dict[str, int]] = {}

    t.reset()
    t.sum_reproducible(np.ones(3), np.arange(3, dtype=np.int64) + 3 * t.rank)
    out["sum_reproducible"] = t.counts()

    t.reset()
    t.sum_vectors_reproducible(np.ones((B, 2)))
    out["sum_vectors_reproducible"] = t.counts()

    t.reset()
    values = np.full((B, 2), -1.0)
    values[np.flatnonzero(owners == t.rank)] = float(t.rank)
    t.owner_broadcast(values, owners)
    out["owner_broadcast"] = t.counts()

    t.reset()
    route_n, route_s = 7, 3
    source = size - 1
    t.route_agent_values(
        (
            {0: 1.0, route_n + 1: 2.0, route_n * route_s - 1: 3.0}
            if t.rank == source
            else None
        ),
        _local_agent_ids(route_n, route_s, t.rank, size),
        source=source,
        n_agents=route_n * route_s,
    )
    out["route_agent_values"] = t.counts()

    t.reset()
    batch_owners = ((np.arange(B) + 1) % size).astype(np.int64)
    t.route_agent_values_batched(
        {
            rep: {rep: float(rep)}
            for rep, owner in enumerate(batch_owners)
            if int(owner) == t.rank
        },
        _local_agent_ids(route_n, route_s, t.rank, size),
        owners=batch_owners,
        n_agents=route_n * route_s,
    )
    out["route_agent_values_batched"] = t.counts()

    t.reset()
    t.exchange_cuts(
        [
            CutRow(
                rep_id=0,
                agent_id=t.rank,
                phi=np.ones(1),
                epsilon=0.0,
                bundle_key=b"c",
            )
        ],
        owners,
    )
    out["exchange_cuts"] = t.counts()

    t.reset()
    with t.collective():
        pass
    out["guard_success"] = t.counts()

    t.reset()
    failing = 1 if size > 1 else 0
    try:
        with t.collective():
            if t.rank == failing:
                raise RuntimeError("counted-failure")
    except TransportError:
        pass
    out["guard_failure"] = t.counts()

    t.reset()
    tiny = {"x": np.full((5, 2), 0.5), "w": np.arange(5.0)}
    t.scatter_by_agent(
        tiny if t.rank == 0 else None,
        np.array([t.rank % 5], dtype=np.int64),
    )
    out["scatter_small"] = t.counts()

    t.reset()
    t.node_shared({"s": np.arange(6.0)} if t.node.node_rank == 0 else {})
    out["node_shared_small"] = t.counts()

    return out


def _emit(transport: MpiTransport, record: dict[str, Any]) -> None:
    """Print one record per rank in rank order.

    Rank k writes only after rank k-1's round completes, so the
    launcher's forwarding sees whole lines instead of interleaved
    multi-kilobyte writes.
    """
    line = json.dumps(record, sort_keys=True)
    for r in range(transport.size):
        if transport.rank == r:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
        transport.allreduce_max(float(r))


def main() -> int:
    transport = MpiTransport()
    record = {
        "rank": transport.rank,
        "size": transport.size,
        "results": battery(transport),
        "counters": counter_scenario(transport),
    }
    _emit(transport, record)
    transport.close()
    return 0


# Multi-chunk gate fixture: ~3.5 chunks of (·, 32) float64 per rank at
# the launch-provided chunk size, shard sizes deliberately non-uniform
# so destinations' window counts differ (the regime a per-chunk
# collective would have to pad).
_CHUNK_ROW_NBYTES = {"m": 32 * 8, "v": 4}


def _chunked_partition(
    size: int, chunk_bytes: int
) -> tuple[list[int], np.ndarray]:
    base_rows = int(3.5 * chunk_bytes) // _CHUNK_ROW_NBYTES["m"]
    rows = [base_rows + 173 * r for r in range(size)]
    bounds = np.concatenate(([0], np.cumsum(rows)))
    return rows, bounds


def _install_overfragment_probe() -> None:
    """Split the scatter stream into twice as many windows as bytes need.

    The regression the byte-ceiling bound must reject: a chunk
    schedule that fragments every stream into extra windows (here, half
    the rows per window) while still delivering the payload bit-for-bit.
    The mirror ``_n_windows`` uses the same ``chunk // row`` formula, so a
    mirror-vs-transport equality echoes the inflation; only a ceiling
    derived from bytes catches it. Installed under an env flag so the
    bound's coverage can be proved without editing src on disk.
    """
    import combrum.transport.mpi as _mpi

    def _fragmented(
        n_rows: int, row_nbytes: int, chunk_bytes: int
    ) -> list[tuple[int, int]]:
        if n_rows == 0 or row_nbytes == 0:
            return []
        rows_per = max(1, (chunk_bytes // row_nbytes) // 2)
        return [
            (start, min(start + rows_per, n_rows))
            for start in range(0, n_rows, rows_per)
        ]

    _mpi._chunk_spans = _fragmented  # type: ignore[assignment]


def chunked_main(chunk_bytes: int, root: int = 0) -> int:
    if os.environ.get("COMBRUM_SCATTER_PROBE_OVERFRAGMENT"):
        _install_overfragment_probe()
    transport = MpiTransport(scatter_chunk_bytes=chunk_bytes)
    rank, size = transport.rank, transport.size
    rows, bounds = _chunked_partition(size, chunk_bytes)
    if root != 0:
        rows[0] += 173 * size * 8
        bounds = np.concatenate(([0], np.cumsum(rows)))
    n_global = int(bounds[-1])
    rng = np.random.default_rng([_SEED, 90])
    big = rng.standard_normal((n_global, 32))
    vec = rng.standard_normal(n_global).astype(np.float32)
    perm = rng.permutation(n_global)
    ids = perm[int(bounds[rank]) : int(bounds[rank + 1])]
    transport.reset()
    got = transport.scatter_by_agent(
        {"m": big, "v": vec} if rank == root else None, ids, root=root
    )
    counters = transport.counts()
    # Every rank regenerated the full tables from the seed, so the
    # bitwise expectation needs no second transport: equality of bytes
    # and dtype is exactly the contract's mapping.
    record = {
        "rank": rank,
        "size": size,
        "root": root,
        "chunk_bytes": chunk_bytes,
        "n_local": int(ids.shape[0]),
        "n_local_by_rank": [int(r) for r in rows],
        "bitwise": {
            "m": bool(
                got["m"].tobytes() == big[ids].tobytes()
                and got["m"].dtype == big.dtype
            ),
            "v": bool(
                got["v"].tobytes() == vec[ids].tobytes()
                and got["v"].dtype == vec.dtype
            ),
        },
        "read_only": [
            bool(not got[key].flags.writeable) for key in ("m", "v")
        ],
        "counters": counters,
    }
    _emit(transport, record)
    transport.close()
    return 0


def window_smoke_main() -> int:
    transport = MpiTransport()
    rng = np.random.default_rng([_SEED, 91])
    a = rng.standard_normal((4096, 8))
    b = rng.integers(-128, 127, size=2048).astype(np.int16)
    shared = transport.node_shared(
        {"a": a, "b": b} if transport.node.node_rank == 0 else {}
    )
    # White-box reach into the window registry: the public contract
    # cannot see allocation placement, so ask MPI directly how many bytes
    # this rank's segment holds — zero on every non-publisher proves the
    # single physical copy.
    own_segment, _ = transport._windows[-1].Shared_query(
        transport.node.node_rank
    )
    flip_raises = False
    try:
        shared["a"].setflags(write=True)
    except ValueError:
        flip_raises = True
    rejects_set = False
    try:
        shared["new"] = np.zeros(1)  # type: ignore[index]
    except TypeError:
        rejects_set = True
    record = {
        "rank": transport.rank,
        "size": transport.size,
        "node_rank": transport.node.node_rank,
        # Peers regenerated a and b locally: bitwise equality on a peer
        # proves the mapped pages carry the publisher's bytes across
        # process boundaries (np.shares_memory cannot say this).
        "content": {
            "a": bool(shared["a"].tobytes() == a.tobytes()),
            "b": bool(shared["b"].tobytes() == b.tobytes()),
        },
        "dtype": {"a": str(shared["a"].dtype), "b": str(shared["b"].dtype)},
        "shape": {"a": list(shared["a"].shape), "b": list(shared["b"].shape)},
        "read_only": [
            bool(not shared[key].flags.writeable) for key in ("a", "b")
        ],
        "flip_raises": flip_raises,
        "mapping_rejects_set": rejects_set,
        "own_segment_bytes": len(own_segment),
        "payload_nbytes": int(a.nbytes + b.nbytes),
    }
    _emit(transport, record)
    transport.close()
    return 0


# ru_maxrss is bytes on Darwin and KiB on Linux.
_RU_MAXRSS_UNIT = 1 if sys.platform == "darwin" else 1024

# The scatter tag the root's Isend chunks carry; mirrors mpi._SCATTER_STREAM_TAG.
_SCATTER_STREAM_TAG = 92


class _CommProxy:
    """Transparent comm wrapper that meters scatter-chunk Isends.

    mpi4py's Intracomm and Request are immutable C types, so the meter
    cannot patch ``Isend``/``Waitall`` in place. Instead the transport's
    ``_comm`` handle is swapped for this proxy, which forwards every
    attribute to the real comm and only intercepts ``Isend`` to record
    the chunk bytes it puts on the wire.
    """

    def __init__(self, real: Any, meter: "_InFlightMeter") -> None:
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_meter", meter)

    def Isend(self, buf: Any, *args: Any, **kwargs: Any) -> Any:
        req = self._real.Isend(buf, *args, **kwargs)
        if kwargs.get("tag") == _SCATTER_STREAM_TAG:
            self._meter._on_isend(req, buf[0])
        return req

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_real"), name)


class _RequestProxy:
    """Forwards ``Request`` static methods, metering scatter Waitalls."""

    def __init__(self, real: Any, meter: "_InFlightMeter") -> None:
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_meter", meter)

    def Waitall(self, requests: Any) -> Any:
        result = self._real.Waitall(requests)
        self._meter._on_waitall(requests)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_real"), name)


class _MpiProxy:
    """Forwards the MPI module, swapping in the metering ``Request``."""

    def __init__(self, real: Any, meter: "_InFlightMeter") -> None:
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "Request", _RequestProxy(real.Request, meter))

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_real"), name)


class _InFlightMeter:
    """Track the root's peak concurrently-held scatter-chunk bytes.

    Each scatter Isend adds its buffer's bytes to a running total keyed
    by the returned request; the matching Waitall drops them. The peak of
    that total is the concurrent in-flight footprint the chunk bound
    claims to cap at one window (``chunk_bytes * (size - 1)``) regardless
    of row count.

    When ``defer_waitall`` is set (the injected regression: root
    lifts the Waitall out of the per-window loop and only drains once at
    the end), the per-window drops are suppressed so the accounting
    accumulates every window's chunks — exactly what the root would hold
    if the in-flight bound were removed. The real MPI completion still
    runs so the transfer does not deadlock.
    """

    def __init__(self, transport: Any, *, defer_waitall: bool) -> None:
        self._transport = transport
        self._defer = defer_waitall
        self._bytes_of: dict[int, int] = {}
        self._outstanding = 0
        self.peak = 0
        self._orig_comm = transport._comm
        self._orig_mpi = transport._mpi

    def _on_isend(self, req: Any, chunk: Any) -> None:
        nbytes = int(getattr(chunk, "nbytes", 0))
        self._bytes_of[id(req)] = nbytes
        self._outstanding += nbytes
        if self._outstanding > self.peak:
            self.peak = self._outstanding

    def _on_waitall(self, requests: Any) -> None:
        if self._defer:
            return
        for req in requests:
            self._outstanding -= self._bytes_of.pop(id(req), 0)

    def __enter__(self) -> "_InFlightMeter":
        self._transport._comm = _CommProxy(self._orig_comm, self)
        self._transport._mpi = _MpiProxy(self._orig_mpi, self)
        return self

    def __exit__(self, *exc: Any) -> None:
        self._transport._comm = self._orig_comm
        self._transport._mpi = self._orig_mpi


def _install_double_recv_probe() -> None:
    """Make each receiver retain a full second copy of its shard.

    The regression the receiver RSS band must reject: a
    ``_scatter_recv`` that keeps a redundant duplicate of every received
    shard alive for the call's duration, roughly doubling the resident
    shard at peak. Installed only when the injection env flag is set so
    the band's coverage can be proved without editing src on disk.
    """
    orig = MpiTransport._scatter_recv

    def _doubled(
        self: MpiTransport,
        header: Any,
        n_local: int,
        chunk_bytes: int,
        root: int,
    ) -> dict[str, np.ndarray]:
        out = orig(self, header, n_local, chunk_bytes, root)
        leak = self.__dict__.setdefault("_double_recv_leak", [])
        leak.append({key: value.copy() for key, value in out.items()})
        return out

    MpiTransport._scatter_recv = _doubled  # type: ignore[method-assign]


def rss_ladder_main(n_rows: int, chunk_bytes: int) -> int:
    import resource
    import shutil

    import psutil

    if os.environ.get("COMBRUM_RSS_PROBE_DOUBLE_RECV"):
        _install_double_recv_probe()
    transport = MpiTransport(scatter_chunk_bytes=chunk_bytes)
    rank, size = transport.rank, transport.size
    rng = np.random.default_rng([_SEED, 92])
    perm = rng.permutation(n_rows)
    bounds = [(r * n_rows) // size for r in range(size + 1)]
    ids = perm[bounds[rank] : bounds[rank + 1]]
    source_nbytes = n_rows * 16 * 8
    tmpdir: str | None = None
    arrays: dict[str, np.ndarray] | None = None
    if rank == 0:
        tmpdir = tempfile.mkdtemp(prefix="combrum-rss-")
        path = os.path.join(tmpdir, "source.bin")
        # Stream the synthetic table to disk in ~1 MiB slabs so generation
        # never makes it RSS-resident: the root's measured peak then
        # reflects the scatter plus faulted-in file pages, not a
        # full-table allocation that would mask the thing under test.
        # Row i is i * [1..16], so each rank can verify sample rows
        # without holding the table.
        slab_rows = max(1, (2**20) // 128)
        with open(path, "wb") as f:
            for start in range(0, n_rows, slab_rows):
                stop = min(start + slab_rows, n_rows)
                block = (
                    np.arange(start, stop, dtype=np.float64)[:, None]
                    * np.arange(1.0, 17.0)[None, :]
                )
                f.write(block.tobytes())
        source = np.memmap(
            path, dtype=np.float64, mode="r", shape=(n_rows, 16)
        )
        arrays = {"x": source}
    rss_before = int(psutil.Process().memory_info().rss)
    defer = bool(os.environ.get("COMBRUM_RSS_PROBE_DEFER_WAITALL"))
    with _InFlightMeter(transport, defer_waitall=defer) as meter:
        got = transport.scatter_by_agent(arrays, ids)
    peak = int(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _RU_MAXRSS_UNIT
    )
    sample = [0, int(ids.shape[0]) // 2, int(ids.shape[0]) - 1]
    expected = np.asarray(ids[sample], dtype=np.float64)[:, None] * np.arange(
        1.0, 17.0
    )
    record = {
        "rank": rank,
        "size": size,
        "n_rows": n_rows,
        "shard_nbytes": int(got["x"].nbytes),
        "source_nbytes": int(source_nbytes) if rank == 0 else 0,
        "rss_before_bytes": rss_before,
        "peak_rss_bytes": peak,
        # Root's peak concurrently-outstanding scatter-chunk bytes; on
        # non-root ranks there are no Isends so this stays 0.
        "peak_in_flight_bytes": int(meter.peak),
        "sample_ok": bool(
            np.array_equal(np.asarray(got["x"][sample]), expected)
        ),
    }
    _emit(transport, record)
    transport.close()
    if tmpdir is not None:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return 0


def node_rss_main() -> int:
    import psutil

    transport = MpiTransport()
    n = (16 * 2**20) // 8
    rng = np.random.default_rng([_SEED, 93])
    # Every rank regenerates the payload before the baseline sample: the
    # deltas below then isolate the publication itself.
    payload = rng.standard_normal(n)
    proc = psutil.Process()
    rss_before = int(proc.memory_info().rss)
    shared = transport.node_shared(
        {"big": payload} if transport.node.node_rank == 0 else {}
    )
    rss_after = int(proc.memory_info().rss)
    own_segment, _ = transport._windows[-1].Shared_query(
        transport.node.node_rank
    )
    # Content check strictly after the rss sample: comparing faults the
    # shared pages into this rank's resident set, smearing the delta it
    # is meant to corroborate.
    content_ok = bool(shared["big"].tobytes() == payload.tobytes())
    record = {
        "rank": transport.rank,
        "size": transport.size,
        "node_rank": transport.node.node_rank,
        "payload_nbytes": int(payload.nbytes),
        "rss_before_bytes": rss_before,
        "rss_after_bytes": rss_after,
        "publish_delta_bytes": rss_after - rss_before,
        "own_segment_bytes": len(own_segment),
        "content_ok": content_ok,
    }
    _emit(transport, record)
    transport.close()
    return 0


def skeleton_e2e_main(family: str) -> int:
    """One rank of the walking-skeleton end-to-end over real MPI.

    Runs the vehicle's family solve through ``MpiTransport`` and reports
    the estimate as bit-hex, so the pytest wrapper can demand it match
    the in-process serial solve byte for byte: the whole distributed
    walk — scatter, canonical reductions, cut exchange, guard-driven
    agreement — reaches the same theta_hat a single rank does, at every
    rank count.
    """
    from _support.families import load_family
    from _support.skeleton import run_skeleton

    fixtures = Path(__file__).resolve().parents[1] / "fixtures" / "families"
    transport = MpiTransport()
    result = run_skeleton(
        load_family(family, fixtures), transport, family=family
    )
    record = {
        "rank": transport.rank,
        "size": transport.size,
        "family": family,
        "theta_hat_hex": result.theta_hat.tobytes().hex(),
        "objective": repr(result.objective),
        "n_active_cuts": result.n_active_cuts,
        "n_iterations": result.metadata["n_iterations"],
        "best_total_regret": repr(result.metadata["best_total_regret"]),
    }
    _emit(transport, record)
    transport.close()
    return 0


def split_axis_smoke(t: Transport) -> dict[str, Any]:
    """Solver-free split-axis distributed context/bootstrap smoke."""
    from combrum.bootstrap_distributed import (
        _bootstrap_wave_c_theta_and_normalizers,
    )
    from combrum.engine.distributed_context import (
        distributed_c_theta,
        owned_observation_ids,
        prepare_distributed_observed,
    )
    from combrum.formulations import NSlack
    from combrum.model import Model
    from combrum.parameters import Parameters

    N, S, K = 11, 4, 3
    table = np.column_stack(
        [
            np.arange(1, N + 1, dtype=np.float64),
            10.0 + 3.0 * np.arange(N, dtype=np.float64),
            100.0 + 5.0 * np.arange(N, dtype=np.float64),
        ]
    )
    weights = np.linspace(0.5, 2.0, N, dtype=np.float64)

    class _ObservedSurface:
        def __init__(self) -> None:
            self.owned: tuple[int, ...] | None = None

        def setup_observed(
            self, transport: Transport, observation_ids: np.ndarray
        ) -> None:
            ids = tuple(map(int, observation_ids))
            owned = tuple(
                map(int, owned_observation_ids(N, transport.rank, transport.size))
            )
            if any(obs_id < 0 or obs_id >= N for obs_id in ids):
                raise AssertionError("setup_observed received agent ids")
            if ids != owned:
                raise AssertionError("setup_observed must receive only owned rows")
            self.owned = owned

        def observed_features_batch(self, observation_ids: np.ndarray) -> np.ndarray:
            ids = tuple(map(int, observation_ids))
            if self.owned is None or ids != self.owned:
                raise AssertionError("observed_features_batch crossed shards")
            if ids and max(ids) >= N:
                raise AssertionError("observed_features_batch received agent ids")
            return np.ascontiguousarray(table[np.asarray(ids, dtype=np.int64)])

    surface = _ObservedSurface()
    model = Model(
        object(),  # type: ignore[arg-type]
        Parameters({"theta": (-10.0, 10.0, K)}),
        features=object(),
        observed_features=surface,
        formulation=NSlack,
    )
    prep = prepare_distributed_observed(
        model,
        n_observations=N,
        n_simulations=S,
        transport=t,
    )
    if os.environ.get("COMBRUM_SMOKE_PROBE_MOMENT_SCALE"):
        # Regression for the empirical-moment oracle: scale the
        # pooled moment by N/(N+1). Applied identically on MpiTransport and
        # the LocalCluster reference (both run this function), so the
        # record equality still holds while the hand-computed [6, 25, 125]
        # oracle no longer matches. Injected under an env flag; src is not
        # edited on disk.
        scaled = np.asarray(prep.empirical_moment, dtype=np.float64) * (
            N / (N + 1)
        )
        object.__setattr__(prep, "empirical_moment", scaled)
    c_theta = distributed_c_theta(
        prep,
        obs_weights_local=weights[prep.owned_obs],
        transport=t,
    )
    c_boot, normalizers = _bootstrap_wave_c_theta_and_normalizers(
        prep,
        [2, 3],
        base_seed=77,
        transport=t,
    )
    return {
        "owned_obs": list(map(int, prep.owned_obs)),
        "local_ids": list(map(int, prep.local_ids)),
        "empirical_hex": prep.empirical_moment.tobytes().hex(),
        "c_theta_hex": c_theta.tobytes().hex(),
        "bootstrap_c_theta_hex": c_boot.tobytes().hex(),
        "normalizers_hex": normalizers.tobytes().hex(),
    }


def split_axis_smoke_main() -> int:
    transport = MpiTransport()
    record = {
        "rank": transport.rank,
        "size": transport.size,
        "results": split_axis_smoke(transport),
    }
    _emit(transport, record)
    transport.close()
    return 0


_BOOT_N = 5
_BOOT_S = 2
_BOOT_K = 2
_BOOT_B = 3
_BOOT_SEED = 123
_BOOT_TOL = 1e-8
_BOOT_MAX_ITER = 20


def _bootstrap_fixture_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.array(
        [
            [1.0, 0.2],
            [0.4, 1.1],
            [1.2, -0.3],
            [-0.5, 0.9],
            [0.7, 0.6],
        ],
        dtype=np.float64,
    )
    observed = np.array(
        [
            [True, False],
            [False, True],
            [True, True],
            [False, True],
            [True, False],
        ],
        dtype=bool,
    )
    base = np.array(
        [
            [0.2, -0.1],
            [-0.3, 0.4],
            [0.1, 0.2],
            [0.5, -0.2],
            [-0.4, 0.3],
        ],
        dtype=np.float64,
    )
    shocks = np.empty((_BOOT_N, _BOOT_S, _BOOT_K), dtype=np.float64)
    for sim_id in range(_BOOT_S):
        shocks[:, sim_id, :] = base + float(sim_id) * np.array([0.15, -0.05])
    return r, observed, shocks


def _bootstrap_model_and_data() -> tuple[Any, Any]:
    import combrum as cb

    r, observed, shocks = _bootstrap_fixture_arrays()

    class _Surface(cb.Oracle, cb.FeatureMap):
        def __init__(self) -> None:
            self.r = r
            self.observed = observed
            self.shocks = shocks

        def setup(self, transport: Transport, local_ids: np.ndarray) -> None:
            self.local_ids = np.asarray(local_ids, dtype=np.int64)

        def setup_observed(
            self, transport: Transport, observation_ids: np.ndarray
        ) -> None:
            self.observation_ids = np.asarray(observation_ids, dtype=np.int64)

        def _obs_sim(self, agent_id: int) -> tuple[int, int]:
            aid = int(agent_id)
            return aid % _BOOT_N, aid // _BOOT_N

        def price(self, theta: np.ndarray, agent_id: int) -> Any:
            obs_id, sim_id = self._obs_sim(agent_id)
            scores = self.r[obs_id] * np.asarray(theta, dtype=np.float64)
            scores = scores + self.shocks[obs_id, sim_id]
            bundle = scores > 0.0
            return cb.Demand.exact(
                bundle=bundle,
                payoff=float(np.where(bundle, scores, 0.0).sum()),
            )

        def price_batch(
            self, theta: np.ndarray, local_ids: np.ndarray
        ) -> dict[int, Any]:
            return {
                int(agent_id): self.price(theta, int(agent_id))
                for agent_id in np.asarray(local_ids, dtype=np.int64)
            }

        def features(
            self, agent_id: int, bundle: np.ndarray
        ) -> tuple[np.ndarray, float]:
            obs_id, sim_id = self._obs_sim(agent_id)
            b = np.asarray(bundle, dtype=np.float64)
            return b * self.r[obs_id], float(b @ self.shocks[obs_id, sim_id])

        def features_batch(
            self, ids: np.ndarray, bundles: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            ids = np.asarray(ids, dtype=np.int64)
            bundles = np.asarray(bundles, dtype=np.float64)
            obs_ids = ids % _BOOT_N
            sim_ids = ids // _BOOT_N
            phi = np.ascontiguousarray(bundles * self.r[obs_ids])
            eps = np.ascontiguousarray(
                np.einsum(
                    "nk,nk->n",
                    bundles,
                    self.shocks[obs_ids, sim_ids],
                    optimize=True,
                )
            )
            return phi, eps

        def __call__(self, agent_id: int, bundle: np.ndarray) -> np.ndarray:
            return self.features(agent_id, bundle)[0]

        def observed_features_batch(
            self, observation_ids: np.ndarray
        ) -> np.ndarray:
            ids = np.asarray(observation_ids, dtype=np.int64)
            return np.ascontiguousarray(
                self.observed[ids].astype(np.float64) * self.r[ids],
                dtype=np.float64,
            )

    surface = _Surface()
    model = cb.Model(
        surface,
        cb.Parameters({"theta": (-5.0, 5.0, _BOOT_K)}),
        features=surface,
        observed_features=surface,
        formulation=cb.NSlack,
    )
    data = cb.Data(
        observed_bundles=observed,
        shocks=shocks,
        observables=list(range(_BOOT_N)),
    )
    return model, data


class _BootstrapObservationWeights:
    def weights_for(self, rep_id: int) -> np.ndarray:
        from combrum.randomness import bootstrap_observation_weights

        return bootstrap_observation_weights(_BOOT_N, _BOOT_SEED, rep_id)


def nslack_bootstrap_serial_oracle() -> dict[str, Any]:
    import combrum as cb

    model, data = _bootstrap_model_and_data()
    result = cb.bootstrap(
        model,
        data,
        n_bootstrap=_BOOT_B,
        weights=_BootstrapObservationWeights(),
        transport=cb.SerialTransport(),
        master_backend="highs",
        tolerance=_BOOT_TOL,
        max_iterations=_BOOT_MAX_ITER,
    )
    return {
        "theta_hex": result.thetas.tobytes().hex(),
        "converged": [bool(v) for v in result.converged],
        "n_bootstrap": _BOOT_B,
        "n_agents": _BOOT_N * _BOOT_S,
    }


def nslack_bootstrap_main() -> int:
    import combrum as cb

    transport = MpiTransport()
    model, _data = _bootstrap_model_and_data()
    transport.reset()
    result = cb.bootstrap_distributed(
        model,
        n_observations=_BOOT_N,
        n_simulations=_BOOT_S,
        n_bootstrap=_BOOT_B,
        base_seed=_BOOT_SEED,
        transport=transport,
        max_live_reps=_BOOT_B,
        master_backend="highs",
        tolerance=_BOOT_TOL,
        max_iterations=_BOOT_MAX_ITER,
    )
    record = {
        "rank": transport.rank,
        "size": transport.size,
        "theta_hex": result.thetas.tobytes().hex(),
        "converged": [bool(v) for v in result.converged],
        "iterations": int(result.iterations or 0),
        "n_bootstrap": _BOOT_B,
        "n_agents": _BOOT_N * _BOOT_S,
        "counters": transport.counts(),
    }
    _emit(transport, record)
    transport.close()
    return 0


def exchange_bad_row_main() -> int:
    transport = MpiTransport()
    owners = np.zeros(1, dtype=np.int64)
    rows = (
        [object()]
        if transport.rank == min(1, transport.size - 1)
        else [
            CutRow(
                rep_id=0,
                agent_id=transport.rank,
                phi=np.ones(1),
                epsilon=0.0,
                bundle_key=b"good",
            )
        ]
    )
    transport.reset()
    try:
        transport.exchange_cuts(rows, owners)  # type: ignore[arg-type]
        outcome: list[object] = ["no-error"]
    except TransportError as exc:
        outcome = ["caught", exc.rank, exc.message]
    record = {
        "rank": transport.rank,
        "size": transport.size,
        "outcome": outcome,
        "counters": transport.counts(),
    }
    _emit(transport, record)
    transport.close()
    return 0


def route_bad_value_main() -> int:
    transport = MpiTransport()
    source = min(1, transport.size - 1)
    transport.reset()
    try:
        transport.route_agent_values(
            {0: object()} if transport.rank == source else None,
            _local_agent_ids(4, 2, transport.rank, transport.size),
            source=source,
            n_agents=8,
        )
        outcome: list[object] = ["no-error"]
    except ValueError as exc:
        outcome = ["caught", str(exc)]
    record = {
        "rank": transport.rank,
        "size": transport.size,
        "outcome": outcome,
        "counters": transport.counts(),
    }
    _emit(transport, record)
    transport.close()
    return 0


def batched_max_bad_shape_main() -> int:
    transport = MpiTransport()
    values = np.zeros(3 if transport.rank == 0 else 4, dtype=np.float64)
    transport.reset()
    try:
        transport.batched_max(values)
        outcome: list[object] = ["no-error"]
    except ValueError as exc:
        outcome = ["caught", str(exc)]
    record = {
        "rank": transport.rank,
        "size": transport.size,
        "outcome": outcome,
        "counters": transport.counts(),
    }
    _emit(transport, record)
    transport.close()
    return 0


def sum_gather_bad_inputs_main() -> int:
    transport = MpiTransport()
    outcomes: dict[str, str] = {}
    transport.reset()
    for name, call in (
        (
            "sum_float_ids",
            lambda: transport.sum_reproducible(
                np.ones(2, dtype=np.float64),
                np.array([0.0, 1.0], dtype=np.float64),
            ),
        ),
        (
            "sum_3d_values",
            lambda: transport.sum_reproducible(
                np.zeros((1, 1, 1), dtype=np.float64),
                np.array([0], dtype=np.int64),
            ),
        ),
        (
            "gather_float_ids",
            lambda: transport.gather_agent_values(
                np.ones(2, dtype=np.float64),
                np.array([0.0, 1.0], dtype=np.float64),
                2,
            ),
        ),
    ):
        try:
            call()
        except ValueError as exc:
            outcomes[name] = str(exc)
        else:
            outcomes[name] = "no-error"
    record = {
        "rank": transport.rank,
        "size": transport.size,
        "outcomes": outcomes,
        "counters": transport.counts(),
    }
    _emit(transport, record)
    transport.close()
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "skeleton-e2e":
        raise SystemExit(skeleton_e2e_main(sys.argv[2]))
    if len(sys.argv) > 1 and sys.argv[1] == "split-axis-smoke":
        raise SystemExit(split_axis_smoke_main())
    if len(sys.argv) > 1 and sys.argv[1] == "nslack-bootstrap":
        raise SystemExit(nslack_bootstrap_main())
    if len(sys.argv) > 1 and sys.argv[1] == "exchange-bad-row":
        raise SystemExit(exchange_bad_row_main())
    if len(sys.argv) > 1 and sys.argv[1] == "route-bad-value":
        raise SystemExit(route_bad_value_main())
    if len(sys.argv) > 1 and sys.argv[1] == "batched-max-bad-shape":
        raise SystemExit(batched_max_bad_shape_main())
    if len(sys.argv) > 1 and sys.argv[1] == "sum-gather-bad-inputs":
        raise SystemExit(sum_gather_bad_inputs_main())
    if len(sys.argv) > 1 and sys.argv[1] == "chunked":
        root = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        raise SystemExit(chunked_main(int(sys.argv[2]), root))
    if len(sys.argv) > 1 and sys.argv[1] == "window-smoke":
        raise SystemExit(window_smoke_main())
    if len(sys.argv) > 1 and sys.argv[1] == "rss-ladder":
        raise SystemExit(rss_ladder_main(int(sys.argv[2]), int(sys.argv[3])))
    if len(sys.argv) > 1 and sys.argv[1] == "node-rss":
        raise SystemExit(node_rss_main())
    raise SystemExit(main())
