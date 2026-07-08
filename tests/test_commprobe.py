from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from _support.commprobe import (
    _ROW_HEADER_BYTES,
    CommSnapshot,
    CountingTransport,
)
from combrum.transport import (
    CutRow,
    LocalCluster,
    SerialTransport,
    TransportError,
)


def make_row(
    rep_id: int = 0,
    agent_id: int = 1,
    key: bytes = b"kk",
    phi: np.ndarray | None = None,
) -> CutRow:
    return CutRow(
        rep_id=rep_id,
        agent_id=agent_id,
        phi=np.arange(3.0) if phi is None else phi,
        epsilon=0.5,
        bundle_key=key,
    )


def spread_magnitudes(
    rng: np.random.Generator, shape: tuple[int, ...]
) -> np.ndarray:
    # values spanning many orders of magnitude, where summation order
    # changes the float result
    magnitude = rng.uniform(-10.0, 10.0, size=shape)
    sign = rng.choice([-1.0, 1.0], size=shape)
    return sign * 10.0**magnitude


# --- counting --------------------------------------------------------------


def test_counts_communication_rounds_and_skips_local_topology_reads() -> None:
    probe = CountingTransport(SerialTransport())
    assert (probe.rank, probe.size) == (0, 1)
    assert probe.node.n_nodes == 1  # local read, not counted
    k = 4
    for i in range(k):
        probe.sum_reproducible(np.arange(2.0), np.arange(2) + 2 * i)
    probe.sum_vectors_reproducible(np.ones(2))
    probe.batched_max(np.zeros(3))
    assert probe.counts() == {
        "sum_reproducible": k,
        "sum_vectors_reproducible": 1,
        "batched_max": 1,
    }


def test_counting_transport_implements_full_abc() -> None:
    assert CountingTransport.__abstractmethods__ == frozenset()


# --- byte accounting -------------------------------------------------------


def test_bytes_moved_matches_nbytes_arithmetic() -> None:
    probe = CountingTransport(SerialTransport())
    values = np.arange(5.0)  # 5 float64 = 40 bytes
    # int32 ids so the value (40 B) and id (20 B) terms differ in width
    ids = np.arange(5, dtype=np.int32)  # 5 int32 = 20 bytes
    probe.sum_reproducible(values, ids)
    aggregate = np.ones((2, 4))  # 8 float64 = 64 bytes
    probe.sum_vectors_reproducible(aggregate)
    probe.batched_max(np.zeros(3))  # 3 float64 = 24 bytes
    owner_values = np.ones((2, 3))  # 6 float64 = 48 bytes
    owners = np.zeros(2, dtype=np.int64)  # 2 int64 = 16 bytes
    probe.owner_broadcast(owner_values, owners)
    probe.route_agent_values(
        {0: 1.0, 2: -2.0},
        np.array([0, 1, 2], dtype=np.int64),
        source=0,
        n_agents=3,
    )
    full = {"x": np.arange(8.0).reshape(4, 2), "w": np.arange(4.0)}
    shard_ids = np.array([2, 0])
    probe.scatter_by_agent(full, shard_ids)
    probe.node_shared({"a": np.zeros(6)})  # 6 float64 = 48 bytes
    probe.bcast({"payload": "opaque"})
    # single-rank max is the identity: 1.5 comes back untouched
    assert probe.allreduce_max(1.5) == 1.5
    assert probe.bytes_moved() == {
        "sum_reproducible": 40 + 20,  # values (40) + int32 ids (20)
        "sum_vectors_reproducible": 64,  # (2, 4) float64
        "batched_max": 24,  # (3,) float64 input
        "owner_broadcast": 48 + 16,  # values (48) + int64 owners (16)
        "route_agent_values": 2 * 16,  # two received pairs, 16 B each
        # Received shard: x rows (2, 2) float64 (32) + w rows (2,) float64 (16).
        "scatter_by_agent": 32 + 16,
        # Serial rank is its node's publisher, so input bytes count.
        "node_shared": 48,  # (6,) float64
        "bcast": 0,
        "allreduce_max": 8,  # one float64 scalar
    }


def test_gather_agent_values_counts_one_round_and_contributed_bytes() -> None:
    probe = CountingTransport(SerialTransport())
    values = np.array([3.0, 5.0, 7.0])  # 3 float64 = 24 bytes
    # int32 ids (12 bytes) deliberately differ in width from the values so the
    # value and id contributions never collide. With equal-width int64 ids a
    # faulty implementation that ignored global_ids and counted values twice would still hit
    # 48; here correct = 24 + 12 = 36 while that faulty implementation reports 48.
    ids = np.array([1, 3, 4], dtype=np.int32)  # 3 int32 = 12 bytes
    n_global = 6
    out = probe.gather_agent_values(values, ids, n_global)
    # Independent oracle: dense n_global vector, value at each id, zeros else.
    expected = np.zeros(n_global)
    expected[ids] = values
    assert np.array_equal(np.asarray(out), expected)
    # Pin the whole tally, not one entry: exactly one round, and contributed =
    # 3 float64 + 3 int32 hand-counted, keyed under gather_agent_values only.
    assert probe.counts() == {"gather_agent_values": 1}
    assert probe.bytes_moved() == {"gather_agent_values": 3 * 8 + 3 * 4}


def test_send_to_root_counts_one_round_zero_bytes_and_is_transparent() -> None:
    probe = CountingTransport(SerialTransport())
    payload = {"final": np.arange(10.0)}
    result = probe.send_to_root(payload, source=0, root=0)
    # Delivery primitive: the wrapper hands back the inner object untouched.
    assert result is payload
    assert probe.counts()["send_to_root"] == 1
    # Documented zero-byte rule: no accountable payload for this primitive.
    assert probe.bytes_moved()["send_to_root"] == 0


def test_exchange_cuts_bytes_use_documented_header() -> None:
    assert _ROW_HEADER_BYTES == 40
    probe = CountingTransport(SerialTransport())
    # Row 0 gets a 5-float64 phi (40 bytes), matching the header; row 1 keeps a
    # 3-float64 phi (24 bytes). With the two phi widths separated, dropping the
    # header or doubling phi no longer lands on the correct total.
    rows = [
        make_row(key=b"kk", phi=np.arange(5.0)),
        make_row(agent_id=2, key=b"longer-key"),
    ]
    probe.exchange_cuts(rows, np.zeros(1, dtype=np.int64))
    # Hand-computed from phi widths, key lengths, and the 40-byte header, all
    # written as literal integers so no term is read back through the source rule.
    # row 0: phi 40 + key 2 + header 40 = 82; row 1: phi 24 + key 10 + 40 = 74.
    assert probe.bytes_moved()["exchange_cuts"] == (40 + 2 + 40) + (24 + 10 + 40)
    assert probe.bytes_moved()["exchange_cuts"] == 156


def test_route_agent_values_bytes_count_received_not_sent() -> None:
    # size=3, n_agents=6. Under contiguous agent sharding, agent g lands on
    # rank g // 2, so the source's 6 pairs split evenly: two land on each rank.
    # The received
    # basis (16 * len(result)) is 32 everywhere; a sent basis (16 * 6 = 96
    # at the source, 0 elsewhere) would be visibly wrong.
    def fn(t):
        probe = CountingTransport(t)
        source = 0
        values = (
            {gid: float(gid) for gid in range(6)}
            if probe.rank == source
            else None
        )
        result = probe.route_agent_values(
            values,
            np.arange(2 * probe.rank, 2 * probe.rank + 2, dtype=np.int64),
            source=source,
            n_agents=6,
        )
        return (
            probe.rank,
            dict(result),
            probe.bytes_moved()["route_agent_values"],
        )

    outs = LocalCluster(3).run(fn)
    # Pin the whole routed payload, not just the keys: the source shipped
    # {gid: float(gid) for gid in range(6)}, so under contiguous-obs sharding
    # each rank must receive exactly its two owned (gid -> float(gid)) pairs.
    # Recomputed independently from the sent map restricted to each rank's
    # agents; this kills owner-permutations (wrong keys) AND any wrapper that
    # offsets/scales the routed value payload (wrong values).
    expected_payloads = [{0: 0.0, 1: 1.0}, {2: 2.0, 3: 3.0}, {4: 4.0, 5: 5.0}]
    assert [payload for _, payload, _ in outs] == expected_payloads
    # Received basis: 16 bytes per routed pair, so 32 everywhere. A sent
    # basis (16 * 6 = 96 at the source, 0 elsewhere) would be visibly wrong.
    assert [b for _, _, b in outs] == [32, 32, 32]


def test_node_shared_bytes_counted_on_publisher_only() -> None:
    def fn(t):
        probe = CountingTransport(t)
        # Peers pass a NON-empty dict the transport ignores; only the
        # publisher's bytes may be counted. A peer that wrongly tallied its
        # own input would report 72 (9 float64) instead of 0.
        payload = (
            {"d": np.zeros(4)}
            if t.node.node_rank == 0
            else {"junk": np.zeros(9)}
        )
        probe.node_shared(payload)
        return (
            probe.counts()["node_shared"],
            probe.bytes_moved()["node_shared"],
        )

    outs = LocalCluster(4, ranks_per_node=2).run(fn)
    assert [c for c, _ in outs] == [1, 1, 1, 1]  # one round everywhere
    # Publishers (node_rank 0) count 4 float64 = 32 bytes; peers count 0
    # despite passing a 9-element array (would be 72 if counted).
    assert [b for _, b in outs] == [32, 0, 32, 0]


# --- transparency ----------------------------------------------------------


def test_sum_reproducible_bitwise_identical_wrapped_vs_unwrapped() -> None:
    rng = np.random.default_rng(20260612)
    inner = SerialTransport()
    probe = CountingTransport(inner)
    ids = rng.permutation(31)
    values = spread_magnitudes(rng, (31, 2))
    wrapped = np.asarray(probe.sum_reproducible(values, ids))

    # Independent oracle: the reproducibility contract is invariance under
    # any permutation of the (id, value) rows. Compare against the wrapped
    # sum of a freshly shuffled copy; a canonical_sum that reduces in input
    # order instead of ascending-id order breaks this in the spread regime.
    perm = rng.permutation(31)
    reshuffled = np.asarray(probe.sum_reproducible(values[perm], ids[perm]))
    assert wrapped.tobytes() == reshuffled.tobytes()

    # Transparency: the wrapper hands back the inner result untouched. Not
    # the sole oracle — the permutation check above carries the src signal.
    raw = np.asarray(inner.sum_reproducible(values, ids))
    assert wrapped.tobytes() == raw.tobytes()


def test_sum_vectors_reproducible_bitwise_identical_wrapped_vs_unwrapped() -> None:
    rng = np.random.default_rng(20260626)
    inner = SerialTransport()
    probe = CountingTransport(inner)
    values = spread_magnitudes(rng, (2, 3))
    wrapped = np.asarray(probe.sum_vectors_reproducible(values))

    # Independent oracle: over a pool of one rank the reproducible vector sum
    # is the identity, so the (B, M) result must be the input verbatim. A
    # combiner that reduces the wrong way (scaling, wrong axis, column mix)
    # fails this without any reference to combrum's own reduction.
    assert wrapped.tobytes() == values.astype(np.float64).tobytes()

    # Transparency: wrapper returns the inner array unaltered.
    raw = np.asarray(inner.sum_vectors_reproducible(values))
    assert wrapped.tobytes() == raw.tobytes()


def test_sum_vectors_reproducible_multirank_matches_hand_sum() -> None:
    # Rank r contributes full((2, 3), r + 1); the reproducible sum across the
    # ranks must be full((2, 3), sum(r + 1)) = 1 + 2 + 3 = 6, hand-computed
    # independently of combrum. A combiner that averages, drops a rank, or
    # otherwise miscombines the per-rank contributions fails.
    n_ranks = 3

    def fn(t):
        probe = CountingTransport(t)
        out = probe.sum_vectors_reproducible(np.full((2, 3), float(t.rank + 1)))
        return np.asarray(out)

    outs = LocalCluster(n_ranks).run(fn)
    expected = np.full((2, 3), float(sum(r + 1 for r in range(n_ranks))))
    for out in outs:
        assert np.array_equal(out, expected)


def test_exchange_cuts_rows_identical_wrapped_vs_unwrapped() -> None:
    inner = SerialTransport()
    probe = CountingTransport(inner)
    rows = [make_row(agent_id=7), make_row(agent_id=3)]
    owners = np.zeros(1, dtype=np.int64)
    raw = inner.exchange_cuts(rows, owners)
    wrapped = probe.exchange_cuts(rows, owners)
    # CutRow identity is by object (eq=False): equality here means the
    # wrapper handed back the very rows the inner transport delivered.
    assert wrapped == raw
    assert [row.agent_id for row in wrapped] == [3, 7]


# --- inside the in-process multirank transport ------------------------------


def test_symmetric_scenario_counts_equal_on_every_rank() -> None:
    def fn(t):
        probe = CountingTransport(t)
        # int32 ids (16 B) deliberately differ in width from the 32-B float64
        # values so the value and id terms of the sum_reproducible byte rule
        # are separable. With equal-width int64 ids a faulty implementation that dropped the
        # ids term and doubled the values term would still land on 32*2 == 64.
        ids = (np.arange(4) + 4 * probe.rank).astype(np.int32)  # globally unique
        probe.sum_reproducible(np.full(4, float(probe.rank)), ids)
        probe.sum_vectors_reproducible(np.full(3, float(probe.rank)))
        probe.batched_max(np.zeros(2))
        probe.allreduce_max(float(probe.rank))
        return probe.counts(), probe.bytes_moved()

    outs = LocalCluster(3).run(fn)
    expected_counts = {
        "sum_reproducible": 1,
        "sum_vectors_reproducible": 1,
        "batched_max": 1,
        "allreduce_max": 1,
    }
    # Hand-computed absolute byte tally from the fixture widths, identical on
    # every rank because every rank feeds the same shapes:
    #   sum_reproducible: values full(4) float64 32 + ids arange(4) int32 16 = 48
    #   sum_vectors_reproducible: full(3) float64 = 24
    #   batched_max: zeros(2) float64 = 16
    #   allreduce_max: one float64 scalar = 8
    # Pinning the whole dict against this independent oracle kills any
    # symmetric byte-rule error (dropped/doubled/swapped term) that would
    # survive a mere cross-rank equality check. The int32 ids keep the value
    # and id terms separable, so drop-ids-double-values (48 -> 64) fails too.
    expected_bytes = {
        "sum_reproducible": 32 + 16,
        "sum_vectors_reproducible": 24,
        "batched_max": 16,
        "allreduce_max": 8,
    }
    for counts, bytes_moved in outs:
        assert counts == expected_counts
        assert bytes_moved == expected_bytes


def test_two_reps_one_owner_exchange_is_one_round_on_every_rank() -> None:
    owners = np.array([0, 0])  # two live reps, one owning rank

    def fn(t):
        probe = CountingTransport(t)
        rows = [make_row(rep_id=t.rank % 2, agent_id=t.rank)]
        received = probe.exchange_cuts(rows, owners)
        return (
            probe.counts()["exchange_cuts"],
            len(received),
            probe.bytes_moved()["exchange_cuts"],
        )

    outs = LocalCluster(3).run(fn)
    assert [rounds for rounds, _, _ in outs] == [1, 1, 1]
    assert [n_received for _, n_received, _ in outs] == [3, 0, 0]  # routing unchanged by probe
    # exchange_cuts bytes are the SENT rows, not the received ones. Each rank
    # contributes exactly one row: phi (3 float64 = 24) + key b"kk" (2) +
    # header (40) = 66. A received-counting tally would report [198, 0, 0]
    # because rank 0 receives all three rows and ranks 1,2 receive none.
    assert [b for _, _, b in outs] == [66, 66, 66]


def test_exchange_cuts_routes_each_rep_to_its_nonzero_owner() -> None:
    # The all-zero-owner scenarios above never route a rep to a non-zero
    # rank, so a combiner that hard-delivers every row to rank 0 would still
    # pass them. Here owners=[0, 1]: rep 0 belongs to rank 0, rep 1 to rank 1.
    # Rank r contributes one row for rep r, so correct routing gives each
    # rank exactly its own rep back; a route-to-0 faulty implementation hands rank 0 both
    # rows and rank 1 none.
    owners = np.array([0, 1], dtype=np.int64)

    def fn(t):
        probe = CountingTransport(t)
        rows = [make_row(rep_id=t.rank, agent_id=10 + t.rank)]
        received = probe.exchange_cuts(rows, owners)
        return (
            probe.rank,
            [row.rep_id for row in received],
            probe.bytes_moved()["exchange_cuts"],
        )

    outs = LocalCluster(2).run(fn)
    assert [reps for _, reps, _ in outs] == [[0], [1]]
    # Sent basis is unchanged by owner routing: each rank ships its one row
    # (phi 24 + key 2 + header 40 = 66) regardless of where it lands.
    assert [b for _, _, b in outs] == [66, 66]


# --- collective guard ------------------------------------------------------


def test_collective_guard_counted_and_error_agreement_intact() -> None:
    def fn(t):
        probe = CountingTransport(t)
        try:
            with probe.collective():
                if probe.rank == 1:
                    raise ValueError("boom-through-probe")
        except TransportError as exc:
            return probe.counts(), exc.rank, exc.message
        return probe.counts(), None, ""

    outs = LocalCluster(3).run(fn)  # returning at all proves no rank hung
    for counts, origin, message in outs:
        assert counts == {"collective_guard": 1}
        assert origin == 1
        assert "boom-through-probe" in message


def test_collective_guard_transparent_when_clean() -> None:
    probe = CountingTransport(SerialTransport())
    with probe.collective():
        pass
    assert probe.counts() == {"collective_guard": 1}
    assert probe.bytes_moved() == {"collective_guard": 0}


# --- reset / snapshot ------------------------------------------------------


def test_reset_zeroes_tallies() -> None:
    probe = CountingTransport(SerialTransport())
    # Independent oracle: over a single-rank pool the cross-rank max is the
    # input verbatim, so the tally-seeding call also carries combrum signal.
    seed = np.array([3.0, 1.0])
    assert np.array_equal(np.asarray(probe.batched_max(seed)), seed)
    assert probe.counts()
    probe.reset()
    assert probe.counts() == {}
    assert probe.bytes_moved() == {}
    assert np.array_equal(np.asarray(probe.batched_max(seed)), seed)
    assert probe.counts() == {"batched_max": 1}


def test_snapshot_immutable_and_does_not_drift() -> None:
    probe = CountingTransport(SerialTransport())
    # Independent oracle: single-rank cross-rank max is the input verbatim.
    # The seed carries 2 float64 = 16 payload bytes, matching the snapshot.
    seed = np.array([3.0, 1.0])
    assert np.array_equal(np.asarray(probe.batched_max(seed)), seed)
    snap = probe.snapshot()
    assert isinstance(snap, CommSnapshot)
    assert snap.counts == {"batched_max": 1}
    assert snap.bytes_moved == {"batched_max": 16}
    with pytest.raises(TypeError):
        snap.counts["batched_max"] = 99  # type: ignore[index]
    with pytest.raises(TypeError):
        snap.bytes_moved["new"] = 1  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.counts = {}  # type: ignore[misc]
    probe.batched_max(np.zeros(2))  # live tallies move on...
    assert snap.counts == {"batched_max": 1}  # ...the snapshot does not
