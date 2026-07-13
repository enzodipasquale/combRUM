from __future__ import annotations

import dataclasses
import threading
from collections import Counter
from unittest import mock

import numpy as np
import pytest

import combrum.transport.reference as reference
from _support.commprobe import CountingTransport, spread_values
from combrum.reductions import canonical_sum
from combrum.transport import (
    CutRow,
    LocalCluster,
    SerialTransport,
    TransportError,
)
from combrum.transport.base import (
    NodeTopology,
    Transport,
    _pack_bundle,
    _unpack_bundle,
    canonical_cut_order,
)
from combrum.transport.reference import LocalMultirankTransport


def bits(x: object) -> bytes:
    return np.asarray(x, dtype=np.float64).tobytes()


def row_sig(row: CutRow) -> tuple[int, int, bytes, bytes, bytes]:
    return (
        row.rep_id,
        row.agent_id,
        row.bundle_key,
        row.phi.tobytes(),
        bits(row.epsilon),
    )


def agent_owner(agent_id: int, n_agents: int, size: int) -> int:
    """Rank owning agent_id under the block partition (larger blocks first)."""
    base, extra = divmod(n_agents, size)
    front = (base + 1) * extra
    if agent_id < front:
        return agent_id // (base + 1)
    return extra + (agent_id - front) // base


def local_agent_axis_ids(
    n_observations: int, n_simulations: int, rank: int, size: int
) -> np.ndarray:
    n_agents = n_observations * n_simulations
    return np.array(
        [
            gid
            for gid in range(n_agents)
            if agent_owner(gid, n_agents, size) == rank
        ],
        dtype=np.int64,
    )


def run_counting_exchange_tags(
    cluster: LocalCluster, fn
) -> tuple[list, Counter]:
    """Run ``fn`` across the cluster, tallying the transport's own
    ``_Rendezvous.exchange`` super-steps by tag. Unlike the CountingTransport
    wrapper (one tally per outer call), this sees internal handshakes."""
    tally: Counter = Counter()
    lock = threading.Lock()
    real_exchange = reference._Rendezvous.exchange

    def counting_exchange(self, rank, tag, payload):
        with lock:
            tally[tag] += 1
        return real_exchange(self, rank, tag, payload)

    with mock.patch.object(
        reference._Rendezvous, "exchange", counting_exchange
    ):
        outs = cluster.run(fn)
    return outs, tally


class _DenseAllocation(AssertionError):
    """Raised when a guarded run allocates a dense pricing-axis vector."""


def run_route_alloc_guarded(
    cluster: LocalCluster,
    values: dict[int, float],
    *,
    source: int,
    n_agents: int,
) -> list[dict[int, float]]:
    """Route ``values`` across ``cluster`` with numpy's bulk allocators
    (``zeros``/``empty``/``full``/``ones``) intercepted: any single
    allocation of >= n_agents elements raises :class:`_DenseAllocation`
    before delegating, so the memory is never reserved. Enforces the base.py
    contract that implementations "must route sparse payloads without
    materializing the full agent axis". agent_ids is a single-element array
    so nothing dense can leak in through the caller's pricing axis."""
    dense_threshold = int(n_agents)
    real = {name: getattr(np, name) for name in ("zeros", "empty", "full", "ones")}

    def _count(shape: object) -> int:
        if isinstance(shape, (int, np.integer)):
            return int(shape)
        try:
            return int(np.prod(shape))
        except (TypeError, ValueError):
            return 0

    def guard(realfn):
        def wrapper(shape, *args, **kwargs):
            if _count(shape) >= dense_threshold:
                raise _DenseAllocation(
                    f"dense allocation of {_count(shape)} elements"
                    f" (>= n_agents={dense_threshold})"
                )
            return realfn(shape, *args, **kwargs)

        return wrapper

    def one_owned_id(rank: int, size: int) -> int:
        base, extra = divmod(dense_threshold, size)
        return rank * base + min(rank, extra)

    def fn(t):
        return t.route_agent_values(
            values if t.rank == source else None,
            np.array([one_owned_id(t.rank, t.size)], dtype=np.int64),
            source=source,
            n_agents=dense_threshold,
        )

    with mock.patch.object(np, "zeros", guard(real["zeros"])), mock.patch.object(
        np, "empty", guard(real["empty"])
    ), mock.patch.object(np, "full", guard(real["full"])), mock.patch.object(
        np, "ones", guard(real["ones"])
    ):
        return cluster.run(fn)


# --- envelope types --------------------------------------------------------


def test_node_topology_validated() -> None:
    NodeTopology(node_id=1, node_rank=0, node_size=2, n_nodes=2)
    bad = [
        dict(node_id=0, node_rank=0, node_size=1, n_nodes=0),
        dict(node_id=0, node_rank=0, node_size=0, n_nodes=1),
        dict(node_id=2, node_rank=0, node_size=1, n_nodes=2),
        dict(node_id=0, node_rank=1, node_size=1, n_nodes=1),
        dict(node_id=-1, node_rank=0, node_size=1, n_nodes=1),
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            NodeTopology(**kwargs)


def test_transport_error_carries_rank_and_message() -> None:
    err = TransportError(3, "boom")
    assert isinstance(err, RuntimeError)
    assert err.rank == 3
    assert err.message == "boom"
    assert "3" in str(err) and "boom" in str(err)


def test_cutrow_validation_and_immutability() -> None:
    row = CutRow(
        rep_id=np.int64(1),
        agent_id=np.int64(2),
        phi=[1.0, 2.0],
        epsilon=np.float64(0.5),
        bundle_key=b"k1",
    )
    assert row.rep_id == 1 and type(row.rep_id) is int
    assert row.agent_id == 2 and type(row.agent_id) is int
    assert type(row.epsilon) is float
    assert row.phi.dtype == np.float64
    assert not row.phi.flags.writeable
    assert row.canonical_key == (1, 2, b"k1")
    source = np.array([1.0, 2.0])
    detached = CutRow(
        rep_id=0, agent_id=0, phi=source[:], epsilon=0.0, bundle_key=b"k"
    )
    source[0] = 99.0
    assert detached.phi[0] == 1.0
    frozen = np.array([3.0, 4.0], dtype=np.float64)
    frozen.setflags(write=False)
    reused = CutRow(rep_id=0, agent_id=0, phi=frozen, epsilon=0.0, bundle_key=b"k")
    assert not np.shares_memory(reused.phi, frozen)
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.rep_id = 5  # type: ignore[misc]
    with pytest.raises(ValueError):
        row.phi[0] = 9.0
    valid = dict(
        rep_id=0, agent_id=0, phi=np.ones(2), epsilon=0.0, bundle_key=b"k"
    )
    for override in (
        dict(rep_id=-1),
        dict(agent_id=-1),
        dict(bundle_key=b""),
        dict(bundle_key="s"),
        dict(phi=np.ones((2, 2))),
    ):
        kwargs = dict(valid)
        kwargs.update(override)
        with pytest.raises(ValueError):
            CutRow(**kwargs)  # type: ignore[arg-type]


def test_cutrow_from_parts_validates_without_copying_trusted_phi() -> None:
    phi = np.array([1.0, 2.0], dtype=np.float64)
    phi.setflags(write=False)
    row = CutRow._from_parts(
        rep_id=np.int64(1),
        agent_id=np.int64(2),
        phi=phi,
        epsilon=np.float64(0.25),
        bundle_key=b"k",
    )
    assert row.rep_id == 1 and type(row.rep_id) is int
    assert row.agent_id == 2 and type(row.agent_id) is int
    assert type(row.epsilon) is float
    assert row.phi is phi
    assert not row.phi.flags.writeable
    for override in (
        dict(rep_id=-1),
        dict(agent_id=-1),
        dict(bundle_key=b""),
        dict(bundle_key="s"),
        dict(phi=np.ones((2, 2))),
    ):
        kwargs = dict(
            rep_id=0, agent_id=0, phi=phi, epsilon=0.0, bundle_key=b"k"
        )
        kwargs.update(override)
        with pytest.raises(ValueError):
            CutRow._from_parts(**kwargs)  # type: ignore[arg-type]


def test_bundle_key_round_trip_preserves_shape_and_dtype() -> None:
    shaped = np.array([[True, False, True], [False, True, False]])
    flat = shaped.reshape(-1)

    shaped_key = _pack_bundle(shaped)
    flat_key = _pack_bundle(flat)

    assert shaped_key != flat_key
    recovered = _unpack_bundle(shaped_key)
    assert recovered.dtype == np.bool_
    assert recovered.shape == shaped.shape
    np.testing.assert_array_equal(recovered, shaped)
    assert not recovered.flags.writeable
    assert _pack_bundle(recovered) == shaped_key


def test_cutrow_preserves_bundle_key_bytes() -> None:
    bundle = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    prior_key = bundle.dtype.str.encode("ascii") + b":" + bundle.tobytes()
    row = CutRow(
        rep_id=0,
        agent_id=0,
        phi=np.ones(1),
        epsilon=0.0,
        bundle_key=prior_key,
    )

    assert row.bundle_key == prior_key
    np.testing.assert_array_equal(row.bundle, bundle)
    np.testing.assert_array_equal(row.bundle, bundle)  # second read serves the memoized decode


def _reference_cut_order(
    rows: list[CutRow],
) -> tuple[CutRow, ...]:
    """Sort ``rows`` by (rep_id, agent_id, bundle_key) without combrum's sort
    or the ``canonical_key`` accessor: decorate-sort-undecorate over the raw
    fields, input index appended so ties keep input order."""
    decorated = [
        ((row.rep_id, row.agent_id, row.bundle_key, i), row)
        for i, row in enumerate(rows)
    ]
    decorated.sort(key=lambda pair: pair[0])
    return tuple(row for _key, row in decorated)


def test_canonical_cut_order_sorts_and_is_stable() -> None:
    # rep-major, then agent, then bundle_key, stable ties. The agent-5 rows
    # share rep_id and agent_id, so bundle_key alone decides p/q/r/s; p and q
    # also share bundle_key b"m", so only input stability fixes their order.
    u = CutRow(
        rep_id=0, agent_id=2, phi=np.ones(1), epsilon=0.0, bundle_key=b"z"
    )
    r = CutRow(
        rep_id=0, agent_id=5, phi=np.ones(1), epsilon=0.0, bundle_key=b"a"
    )
    p = CutRow(
        rep_id=0, agent_id=5, phi=np.ones(1), epsilon=0.0, bundle_key=b"m"
    )
    q = CutRow(
        rep_id=0, agent_id=5, phi=np.ones(1), epsilon=9.0, bundle_key=b"m"
    )
    s = CutRow(
        rep_id=0, agent_id=5, phi=np.ones(1), epsilon=0.0, bundle_key=b"z"
    )
    a = CutRow(
        rep_id=1, agent_id=0, phi=np.ones(1), epsilon=0.0, bundle_key=b"z"
    )
    # Input is scrambled and bundle-key-reversed within the agent-5 tie: a
    # bundle_key-blind or key-reversed sort would both emit (s, p, q, r)
    # there, not the canonical (r, p, q, s).
    rows_in = [a, s, p, q, r, u]
    ordered = canonical_cut_order(rows_in)

    expected = (u, r, p, q, s, a)
    assert ordered == expected
    assert ordered == _reference_cut_order(rows_in)
    assert [
        (row.rep_id, row.agent_id, row.bundle_key) for row in ordered
    ] == [
        (0, 2, b"z"),
        (0, 5, b"a"),
        (0, 5, b"m"),
        (0, 5, b"m"),
        (0, 5, b"z"),
        (1, 0, b"z"),
    ]


# --- topology --------------------------------------------------------------


def test_serial_topology() -> None:
    t = SerialTransport()
    assert (t.rank, t.size) == (0, 1)
    assert t.node == NodeTopology(
        node_id=0, node_rank=0, node_size=1, n_nodes=1
    )


def test_local_cluster_topology_two_nodes_of_two() -> None:
    outs = LocalCluster(4, ranks_per_node=2).run(
        lambda t: (t.rank, t.size, t.node)
    )
    assert [o[0] for o in outs] == [0, 1, 2, 3]
    assert all(o[1] == 4 for o in outs)
    assert [o[2] for o in outs] == [
        NodeTopology(node_id=0, node_rank=0, node_size=2, n_nodes=2),
        NodeTopology(node_id=0, node_rank=1, node_size=2, n_nodes=2),
        NodeTopology(node_id=1, node_rank=0, node_size=2, n_nodes=2),
        NodeTopology(node_id=1, node_rank=1, node_size=2, n_nodes=2),
    ]


def test_local_cluster_topology_ragged_last_node() -> None:
    outs = LocalCluster(3, ranks_per_node=2).run(lambda t: t.node)
    assert outs[2] == NodeTopology(
        node_id=1, node_rank=0, node_size=1, n_nodes=2
    )


def test_local_cluster_topology_default_single_node() -> None:
    outs = LocalCluster(3).run(lambda t: t.node)
    # On the default (ranks_per_node=size) path each rank keeps its own
    # node_rank, which selects the node_shared publisher.
    assert outs == [
        NodeTopology(node_id=0, node_rank=r, node_size=3, n_nodes=1)
        for r in range(3)
    ]


# --- bcast / allreduce_max -------------------------------------------------


def test_bcast_delivers_private_copies() -> None:
    base = np.arange(4.0)

    def fn(t):
        payload = {"v": base} if t.rank == 0 else None
        got = t.bcast(payload, root=0)
        if t.rank == 1:
            got["v"][0] = 99.0
        return float(got["v"][0])

    assert LocalCluster(3).run(fn) == [0.0, 99.0, 0.0]
    assert base[0] == 0.0  # root state untouched by rank 1's mutation


def test_bcast_from_nonzero_root_and_root_validation() -> None:
    outs = LocalCluster(3).run(
        lambda t: t.bcast("payload" if t.rank == 2 else None, root=2)
    )
    assert outs == ["payload"] * 3
    assert SerialTransport().bcast({"k": 1}) == {"k": 1}
    with pytest.raises(ValueError, match="root"):
        SerialTransport().bcast("x", root=1)


def test_allreduce_max_across_ranks_and_serial() -> None:
    outs = LocalCluster(3).run(
        lambda t: t.allreduce_max([-5.0, 3.5, 2.0][t.rank])
    )
    assert outs == [3.5, 3.5, 3.5]
    assert SerialTransport().allreduce_max(7.25) == 7.25

    # A NaN on any single rank must propagate to every rank (base.py
    # contract); a nanmax would return the finite 3.5 here.
    nan_outs = LocalCluster(3).run(
        lambda t: t.allreduce_max([np.nan, 3.5, 2.0][t.rank])
    )
    assert all(np.isnan(o) for o in nan_outs)
    assert np.isnan(SerialTransport().allreduce_max(np.nan))


# --- sum_reproducible ------------------------------------------------------


@pytest.mark.parametrize("shape", [(97,), (97, 3)], ids=["vector", "matrix"])
def test_sum_reproducible_rank_and_distribution_invariance(
    shape: tuple[int, ...],
) -> None:
    rng = np.random.default_rng(20260612)
    n = shape[0]
    ids = rng.permutation(n)
    values = spread_values(rng, shape)
    expected = bits(canonical_sum(values, ids))

    serial = SerialTransport().sum_reproducible(values, ids)
    assert bits(serial) == expected

    for size in (2, 4):
        contiguous = np.array_split(np.arange(n), size)
        perm = rng.permutation(n)
        scrambled = [perm[r::size] for r in range(size)]
        for parts in (contiguous, scrambled):

            def fn(t, parts=parts):
                idx = parts[t.rank]
                return t.sum_reproducible(values[idx], ids[idx])

            for result in LocalCluster(size).run(fn):
                assert bits(result) == expected


def test_sum_reproducible_empty_contribution_ranks() -> None:
    rng = np.random.default_rng(31)
    n = 9
    ids = np.arange(n)
    values = spread_values(rng, (n,))
    expected = bits(canonical_sum(values, ids))
    parts = [np.arange(0, 5), np.empty(0, dtype=np.int64), np.arange(5, 9)]

    def fn(t):
        idx = parts[t.rank]
        return t.sum_reproducible(values[idx], ids[idx])

    for result in LocalCluster(3).run(fn):
        assert bits(result) == expected


def test_sum_reproducible_empty_matrix_rank_adopts_width() -> None:
    values = np.array(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float64
    )
    ids = np.array([0, 1, 2], dtype=np.int64)
    expected = bits(canonical_sum(values, ids))
    parts = [np.array([0, 2]), np.empty(0, dtype=np.int64), np.array([1])]

    def fn(t):
        idx = parts[t.rank]
        local_values = (
            values[idx] if t.rank != 1 else np.empty(0, dtype=np.float64)
        )
        return t.sum_reproducible(local_values, ids[idx])

    for result in LocalCluster(3).run(fn):
        assert bits(result) == expected


def test_sum_reproducible_mismatched_nonempty_widths_raise() -> None:
    def fn(t):
        values = (
            np.ones((1, 2), dtype=np.float64)
            if t.rank == 0
            else np.ones((1, 3), dtype=np.float64)
        )
        ids = np.array([t.rank], dtype=np.int64)
        try:
            t.sum_reproducible(values, ids)
        except ValueError as exc:
            return "contribution shape" in str(exc)
        return False

    assert LocalCluster(2).run(fn) == [True, True]


def test_sum_reproducible_duplicate_ids_raise_on_every_rank() -> None:
    def fn(t):
        ids = np.array([7]) if t.rank in (0, 2) else np.array([1])
        try:
            t.sum_reproducible(np.array([1.0]), ids)
        except ValueError as exc:
            return ("raised", "7" in str(exc))
        return ("no-error", False)

    assert LocalCluster(3).run(fn) == [("raised", True)] * 3


def test_sum_vectors_reproducible_rank_ordered_sum() -> None:
    rng = np.random.default_rng(20260626)
    size = 4
    values = spread_values(rng, (size, 2, 3))
    expected = np.asarray(
        canonical_sum(values.reshape(size, -1), np.arange(size, dtype=np.int64))
    ).reshape(2, 3)

    def fn(t):
        return t.sum_vectors_reproducible(values[t.rank])

    for result in LocalCluster(size).run(fn):
        assert bits(result) == bits(expected)


def test_sum_vectors_reproducible_mismatched_shapes_raise() -> None:
    def fn(t):
        values = np.ones((2,), dtype=np.float64) if t.rank == 0 else np.ones((3,))
        try:
            t.sum_vectors_reproducible(values)
        except ValueError as exc:
            return "same shape" in str(exc)
        return False

    assert LocalCluster(2).run(fn) == [True, True]


# --- scatter_by_agent ------------------------------------------------------


def test_scatter_by_agent_mapping_three_ranks() -> None:
    n_global = 10
    full = {
        "x": np.arange(20.0).reshape(n_global, 2),
        "w": np.arange(n_global) * 0.5,
    }
    perm = np.random.default_rng(5).permutation(n_global)
    parts = [perm[r::3] for r in range(3)]

    def fn(t):
        arrays = full if t.rank == 0 else None
        return t.scatter_by_agent(arrays, parts[t.rank])

    outs = LocalCluster(3).run(fn)
    for r, out in enumerate(outs):
        assert set(out) == {"x", "w"}
        np.testing.assert_array_equal(out["x"], full["x"][parts[r]])
        np.testing.assert_array_equal(out["w"], full["w"][parts[r]])
        assert not out["x"].flags.writeable
        with pytest.raises(ValueError):
            out["w"][0] = 99.0


def test_scatter_by_agent_nonzero_root() -> None:
    n_global = 9
    root = 2
    full = {"x": np.arange(n_global, dtype=np.float64) + 10.0}
    parts = [
        np.array([0, 4], dtype=np.int64),
        np.array([1, 8, 3], dtype=np.int64),
        np.array([2, 5, 6, 7], dtype=np.int64),
    ]

    def fn(t):
        arrays = full if t.rank == root else None
        return t.scatter_by_agent(arrays, parts[t.rank], root=root)

    outs = LocalCluster(3).run(fn)
    for r, out in enumerate(outs):
        np.testing.assert_array_equal(out["x"], full["x"][parts[r]])


def test_scatter_by_agent_serial_identity() -> None:
    full = {"x": np.arange(8.0).reshape(4, 2)}
    ids = np.array([2, 0])
    out = SerialTransport().scatter_by_agent(full, ids)
    np.testing.assert_array_equal(out["x"], full["x"][ids])
    assert not out["x"].flags.writeable
    with pytest.raises(ValueError, match="must pass the full arrays"):
        SerialTransport().scatter_by_agent(None, ids)


def test_scatter_by_agent_payload_contract_enforced_everywhere() -> None:
    full = {"x": np.arange(4.0)}

    def root_none(t):
        try:
            t.scatter_by_agent(None, np.array([0]))
        except ValueError as exc:
            return "rank 0 must pass" in str(exc)
        return False

    assert LocalCluster(2).run(root_none) == [True, True]

    def nonroot_dict(t):
        try:
            t.scatter_by_agent(full, np.array([0]))
        except ValueError as exc:
            return "only rank 0" in str(exc)
        return False

    assert LocalCluster(2).run(nonroot_dict) == [True, True]

    def out_of_range_ids(t):
        arrays = full if t.rank == 0 else None
        ids = np.array([0]) if t.rank == 0 else np.array([7])
        try:
            t.scatter_by_agent(arrays, ids)
        except ValueError as exc:
            return "[0, 4)" in str(exc)
        return False

    assert LocalCluster(2).run(out_of_range_ids) == [True, True]

    # id == n_global must hit the [0, 4) ValueError, not fall through to a
    # raw IndexError from full[ids]; id == n_global - 1 is valid.
    def id_equals_n_global(t):
        arrays = full if t.rank == 0 else None
        ids = np.array([0]) if t.rank == 0 else np.array([4])
        try:
            t.scatter_by_agent(arrays, ids)
        except ValueError as exc:
            return "[0, 4)" in str(exc)
        return False

    assert LocalCluster(2).run(id_equals_n_global) == [True, True]

    def id_at_top_of_range(t):
        arrays = full if t.rank == 0 else None
        ids = np.array([0]) if t.rank == 0 else np.array([3])
        out = t.scatter_by_agent(arrays, ids)
        return float(out["x"][0])

    # rank 0 selects full["x"][0] == 0.0; rank 1 selects full["x"][3] == 3.0.
    assert LocalCluster(2).run(id_at_top_of_range) == [0.0, 3.0]


def test_send_to_root_delivers_only_to_root() -> None:
    def fn(t):
        payload = {"rank": t.rank, "x": np.array([1.0, 2.0])}
        return t.send_to_root(payload if t.rank == 2 else None, source=2)

    out = LocalCluster(4).run(fn)
    assert out[0]["rank"] == 2
    np.testing.assert_array_equal(out[0]["x"], np.array([1.0, 2.0]))
    assert out[1:] == [None, None, None]


def test_send_to_root_source_equals_root_returns_own_object() -> None:
    # source == root takes the identity-return branch: base.py mirrors bcast,
    # so the root gets its own object back (no deepcopy) and peers get None.
    sentinels = [np.array([float(r), float(r) + 0.5]) for r in range(4)]

    def fn(t):
        payload = {"rank": t.rank, "x": sentinels[t.rank]}
        got = t.send_to_root(payload if t.rank == 0 else None, source=0, root=0)
        # Check identity in-thread, where this rank's object lives.
        own = got is not None and got["x"] is sentinels[t.rank]
        return got, own

    out = LocalCluster(4).run(fn)
    root_got, root_own = out[0]
    assert root_got["rank"] == 0
    np.testing.assert_array_equal(root_got["x"], sentinels[0])
    assert root_own is True  # root receives its own object, not a copy
    assert [payload for payload, _own in out[1:]] == [None, None, None]


def test_gather_agent_values_root_only_dense_vector() -> None:
    # Values sit non-monotonically against their ids (rank 0 puts 40.0 at
    # id 0 and 10.0 at id 4), so a sort-and-pack of the values would differ
    # from the correct positional scatter. ids 5, 6 stay uncontributed to
    # check the documented zero-fill.
    n_global = 7
    ids = [
        np.array([0, 4], dtype=np.int64),
        np.array([1, 3], dtype=np.int64),
        np.array([2], dtype=np.int64),
    ]
    vals = [
        np.array([40.0, 10.0]),
        np.array([20.0, 30.0]),
        np.array([25.0]),
    ]
    expected = np.zeros(n_global, dtype=np.float64)
    for rank_ids, rank_vals in zip(ids, vals):
        expected[rank_ids] = rank_vals
    assert expected.tolist() == [40.0, 20.0, 25.0, 30.0, 10.0, 0.0, 0.0]

    def fn(t):
        return t.gather_agent_values(vals[t.rank], ids[t.rank], n_global)

    out = LocalCluster(3).run(fn)
    np.testing.assert_array_equal(out[0], expected)
    assert out[1:] == [None, None]
    assert not out[0].flags.writeable


def test_route_agent_values_uses_agent_axis_owner_mapping() -> None:
    n_observations, n_simulations, size, source = 5, 4, 3, 2
    n_agents = n_observations * n_simulations
    source_values = {
        0: 1.25,
        4: -3.5,
        6: np.float64(7.75),
        13: 9.0,
        19: -11.0,
    }

    def fn(t):
        ids = local_agent_axis_ids(n_observations, n_simulations, t.rank, t.size)
        values = source_values if t.rank == source else None
        return t.route_agent_values(
            values,
            ids,
            source=source,
            n_agents=n_agents,
        )

    outs = LocalCluster(size).run(fn)
    for rank, got in enumerate(outs):
        expected = {
            gid: value
            for gid, value in source_values.items()
            if agent_owner(gid, n_agents, size) == rank
        }
        assert sorted(got) == sorted(expected)
        for gid, value in expected.items():
            assert bits(got[gid]) == bits(value)


def test_route_agent_values_emits_keys_in_increasing_order() -> None:
    n_observations, n_simulations, size, source = 5, 3, 2, 1
    n_agents = n_observations * n_simulations
    source_values = {
        14: 14.0,
        0: 0.0,
        11: 11.0,
        3: 3.0,
        5: 5.0,
        2: 2.0,
    }
    # Derived from the block partition by hand; both ranks land non-empty
    # buckets, so the ordering check below has real input to order.
    expected_buckets = [{}, {}]  # type: list[dict[int, float]]
    for gid, value in source_values.items():
        owner = agent_owner(gid, n_agents, size)
        expected_buckets[owner][gid] = value
    assert all(bucket for bucket in expected_buckets)  # fixture keeps both live

    def fn(t):
        ids = local_agent_axis_ids(n_observations, n_simulations, t.rank, t.size)
        values = source_values if t.rank == source else None
        return t.route_agent_values(
            values,
            ids,
            source=source,
            n_agents=n_agents,
        )

    outs = LocalCluster(size).run(fn)
    for rank, got in enumerate(outs):
        assert dict(got) == expected_buckets[rank]
        assert list(got) == sorted(got)
    assert {gid for got in outs for gid in got} == set(source_values)
    assert sum(len(got) for got in outs) == len(source_values)


def test_route_agent_values_serial_identity() -> None:
    values = {0: 0.5, 2: -1.25}
    out = SerialTransport().route_agent_values(
        values,
        np.array([0, 1, 2], dtype=np.int64),
        source=0,
        n_agents=3,
    )
    assert out == values
    with pytest.raises(ValueError, match="source rank 0 must pass"):
        SerialTransport().route_agent_values(
            None,
            np.array([0], dtype=np.int64),
            source=0,
            n_agents=1,
        )


def test_route_agent_values_validation_errors_agree_across_ranks() -> None:
    def non_source_payload(t):
        try:
            t.route_agent_values(
                {0: 1.0},
                local_agent_axis_ids(4, 2, t.rank, t.size),
                source=1,
                n_agents=8,
            )
        except ValueError as exc:
            return "only source rank 1" in str(exc)
        return False

    assert LocalCluster(2).run(non_source_payload) == [True, True]

    def bad_agent_ids_shape(t):
        ids = (
            np.zeros((1, 1), dtype=np.int64)
            if t.rank == 0
            else local_agent_axis_ids(4, 2, t.rank, t.size)
        )
        try:
            t.route_agent_values(
                {0: 1.0} if t.rank == 0 else None,
                ids,
                source=0,
                n_agents=8,
            )
        except ValueError as exc:
            return "1-D integer array" in str(exc)
        return False

    assert LocalCluster(2).run(bad_agent_ids_shape) == [True, True]

    def bad_source_value(t):
        try:
            t.route_agent_values(
                {0: object()} if t.rank == 0 else None,
                local_agent_axis_ids(4, 2, t.rank, t.size),
                source=0,
                n_agents=8,
            )
        except ValueError as exc:
            return "float64" in str(exc)
        return False

    assert LocalCluster(2).run(bad_source_value) == [True, True]

    with pytest.raises(ValueError, match=r"\[0, 2\)"):
        SerialTransport().route_agent_values(
            {2: 1.0},
            np.array([0], dtype=np.int64),
            source=0,
            n_agents=2,
        )


def test_route_agent_values_sparse_delivery_one_round_and_partition() -> None:
    n_observations, n_simulations, size, source = 13, 20, 4, 3
    values = {0: 1.0, 27: 2.0, 259: 3.0}
    # divmod(260, 4) = (65, 0): ids 0 and 27 land on rank 0, id 259 on rank 3.
    expected_buckets = [
        {0: 1.0, 27: 2.0},
        {},
        {},
        {259: 3.0},
    ]

    def fn(t):
        probe = CountingTransport(t)
        got = probe.route_agent_values(
            values if t.rank == source else None,
            local_agent_axis_ids(n_observations, n_simulations, t.rank, t.size),
            source=source,
            n_agents=n_observations * n_simulations,
        )
        return got, probe.counts(), probe.bytes_moved()

    outs, exchange_tags = run_counting_exchange_tags(LocalCluster(size), fn)
    delivered = [got for got, _counts, _bytes in outs]

    for rank, got in enumerate(delivered):
        assert got == expected_buckets[rank]
        assert list(got) == sorted(got)
        for gid, value in expected_buckets[rank].items():
            assert bits(got[gid]) == bits(value)
    assert sum(len(got) for got in delivered) == len(values)
    assert {gid for got in delivered for gid in got} == set(values)

    # The reference does a single counts+payload super-step, so exactly
    # `size` rendezvous exchanges carry this tag. Counting at the rendezvous
    # (not the wrapper) exposes any extra internal handshake.
    assert exchange_tags["route_agent_values"] == size
    assert [counts for _got, counts, _bytes in outs] == [
        {"route_agent_values": 1}
    ] * size
    # Sparse wire size: one id + one value (16 B) per delivered pair, far
    # below a dense n_obs*n_sims float64 vector.
    assert sum(
        bytes_["route_agent_values"] for _got, _counts, bytes_ in outs
    ) == 16 * len(values)
    assert n_observations * n_simulations * 8 > 16 * len(values)


def test_route_agent_values_batched_routes_many_sparse_maps_once() -> None:
    n_observations, n_simulations, size = 5, 4, 3
    owners = np.array([1, 0, 2], dtype=np.int64)
    payloads = {
        0: {0: 1.0, 19: 2.0},
        1: {6: 3.0},
        2: {13: 4.0},
    }
    expected = [
        {0: {0: 1.0}, 1: {6: 3.0}},
        {2: {13: 4.0}},
        {0: {19: 2.0}},
    ]

    def fn(t):
        probe = CountingTransport(t)
        owned_payload = {
            rep: values
            for rep, values in payloads.items()
            if int(owners[rep]) == t.rank
        }
        got = probe.route_agent_values_batched(
            owned_payload,
            local_agent_axis_ids(n_observations, n_simulations, t.rank, t.size),
            owners=owners,
            n_agents=n_observations * n_simulations,
        )
        return got, probe.counts(), probe.bytes_moved()

    outs, exchange_tags = run_counting_exchange_tags(LocalCluster(size), fn)
    assert [got for got, _counts, _bytes in outs] == expected
    assert exchange_tags["route_agent_values_batched"] == size
    assert [counts for _got, counts, _bytes in outs] == [
        {"route_agent_values_batched": 1}
    ] * size
    assert sum(
        bytes_["route_agent_values_batched"] for _got, _counts, bytes_ in outs
    ) == 24 * sum(len(values) for values in payloads.values())


def test_route_agent_values_never_scans_a_dense_pricing_axis() -> None:
    # 1e9 agents (8 GB dense) with three routed pairs: run_route_alloc_guarded
    # aborts the run if any bulk allocator is asked for >= n_agents elements,
    # so a "materialize the dense axis, then read the pairs back out" path
    # blows up here rather than silently scanning.
    n_observations, n_simulations, size, source = 1_000_000, 1000, 4, 3
    values = {0: 1.0, 27: 2.0, 500_012: 3.0}
    # Block partition of 1e9 agents over 4 ranks: the first block holds 250M
    # agents, so all three ids belong to rank 0.
    expected_buckets = [{0: 1.0, 27: 2.0, 500_012: 3.0}, {}, {}, {}]

    outs = run_route_alloc_guarded(
        LocalCluster(size),
        values,
        source=source,
        n_agents=n_observations * n_simulations,
    )
    for rank, got in enumerate(outs):
        assert got == expected_buckets[rank]
        assert list(got) == sorted(got)
    assert {gid for got in outs for gid in got} == set(values)
    assert sum(len(got) for got in outs) == len(values)


def test_scatter_by_agent_arrays_must_share_agent_axis() -> None:
    with pytest.raises(ValueError, match="axis-0"):
        SerialTransport().scatter_by_agent(
            {"a": np.zeros(3), "b": np.zeros(4)}, np.array([0])
        )


# --- node_shared -----------------------------------------------------------


def test_node_shared_one_copy_per_node() -> None:
    def fn(t):
        return t.node_shared({"d": np.full(3, float(t.rank))})

    maps = LocalCluster(4, ranks_per_node=2).run(fn)
    d = [m["d"] for m in maps]
    assert np.shares_memory(d[0], d[1])
    assert np.shares_memory(d[2], d[3])
    assert not np.shares_memory(d[0], d[2])
    # Content comes from each node's publishing member (ranks 0 and 2).
    np.testing.assert_array_equal(d[1], np.full(3, 0.0))
    np.testing.assert_array_equal(d[3], np.full(3, 2.0))
    for arr in d:
        assert not arr.flags.writeable
        with pytest.raises(ValueError):
            arr.setflags(write=True)
    with pytest.raises(TypeError):
        maps[0]["new"] = np.zeros(1)  # type: ignore[index]


def test_node_shared_publish_is_a_copy_and_serial_shares() -> None:
    src = np.zeros(3)
    shared = SerialTransport().node_shared({"a": src})
    src[0] = 5.0  # later mutation of the source must not leak
    assert shared["a"][0] == 0.0
    assert not shared["a"].flags.writeable


def test_node_shared_publish_error_agrees_across_ranks() -> None:
    def fn(t):
        bad = np.empty(1, dtype=object)
        try:
            t.node_shared({"a": bad} if t.node.node_rank == 0 else {})
        except ValueError as exc:
            return "publishing failed on rank 0" in str(exc)
        return False

    assert LocalCluster(2).run(fn) == [True, True]


# --- batched_max -----------------------------------------------------------


def test_batched_max_matches_np_max_across_ranks() -> None:
    rng = np.random.default_rng(77)
    vals = [spread_values(rng, (6,)) for _ in range(3)]
    expected = np.max(np.stack(vals), axis=0)

    def fn(t):
        return t.batched_max(vals[t.rank])

    first = LocalCluster(3).run(fn)
    second = LocalCluster(3).run(fn)
    for a, b in zip(first, second):
        assert bits(a) == bits(expected)
        assert bits(a) == bits(b)
    serial = SerialTransport().batched_max(np.array([1.0, -2.0]))
    assert serial.dtype == np.float64
    np.testing.assert_array_equal(serial, [1.0, -2.0])


def test_batched_max_propagates_nan_per_slot() -> None:
    vals = [
        np.array([1.0, np.nan, 3.0]),
        np.array([2.0, 4.0, np.nan]),
    ]

    def fn(t):
        return t.batched_max(vals[t.rank])

    out = LocalCluster(2).run(fn)
    for arr in out:
        assert arr[0] == 2.0
        assert np.isnan(arr[1])
        assert np.isnan(arr[2])


def test_batched_max_shape_mismatch_agreed() -> None:
    def fn(t):
        try:
            t.batched_max(np.zeros(3 if t.rank == 0 else 4))
        except ValueError as exc:
            return "same (B,)" in str(exc)
        return False

    assert LocalCluster(2).run(fn) == [True, True]


def test_owner_broadcast_publishes_owner_rows_to_every_rank() -> None:
    owners = np.array([2, 0, 2, 1], dtype=np.int64)

    def fn(t):
        values = np.full((owners.size, 2), 9999.0 + t.rank, dtype=np.float64)
        for rep, owner in enumerate(owners):
            if int(owner) == t.rank:
                values[rep] = [float(rep), float(t.rank + 10)]
        probe = CountingTransport(t)
        got = probe.owner_broadcast(values, owners)
        return got, probe.counts()

    outs = LocalCluster(3).run(fn)
    expected = np.array(
        [
            [0.0, 12.0],
            [1.0, 10.0],
            [2.0, 12.0],
            [3.0, 11.0],
        ],
        dtype=np.float64,
    )
    for got, counts in outs:
        np.testing.assert_array_equal(got, expected)
        assert counts == {"owner_broadcast": 1}


# --- exchange_cuts ---------------------------------------------------------


def test_exchange_cuts_routing_and_canonical_order() -> None:
    size = 3
    owners = np.array([2, 0, 1])

    def make_rows(rank: int) -> list[CutRow]:
        # Scrambled local order, several reps per rank.
        return [
            CutRow(
                rep_id=(rank + 1) % 3,
                agent_id=30 - rank,
                phi=np.array([float(rank), 1.0]),
                epsilon=0.1 * rank,
                bundle_key=b"b" + bytes([65 + rank]),
            ),
            CutRow(
                rep_id=rank,
                agent_id=rank,
                phi=np.array([2.0, float(rank)]),
                epsilon=1.0,
                bundle_key=b"a",
            ),
        ]

    all_rows = [make_rows(r) for r in range(size)]
    outs = LocalCluster(size).run(
        lambda t: t.exchange_cuts(all_rows[t.rank], owners)
    )
    pooled = [row for rows in all_rows for row in rows]
    for r in range(size):
        expected = sorted(
            (row_sig(row) for row in pooled if owners[row.rep_id] == r),
            key=lambda sig: sig[:3],
        )
        got = [row_sig(row) for row in outs[r]]
        assert got == expected
        keys = [row.canonical_key for row in outs[r]]
        assert keys == sorted(keys)


def test_exchange_cuts_two_reps_one_owner_interleaved() -> None:
    owners = np.array([0, 0])  # two live reps, both owned by rank 0

    def fn(t):
        rows = [
            CutRow(
                rep_id=1,
                agent_id=10 + t.rank,
                phi=np.array([1.0]),
                epsilon=0.5,
                bundle_key=b"x",
            ),
            CutRow(
                rep_id=0,
                agent_id=20 + t.rank,
                phi=np.array([2.0]),
                epsilon=0.25,
                bundle_key=b"y",
            ),
            CutRow(
                rep_id=1,
                agent_id=t.rank,
                phi=np.array([3.0]),
                epsilon=0.75,
                bundle_key=b"z",
            ),
        ]
        return t.exchange_cuts(rows, owners)

    outs = LocalCluster(3).run(fn)
    assert outs[1] == () and outs[2] == ()
    got = outs[0]
    assert len(got) == 9  # every row of both reps, from all three ranks
    assert [row.rep_id for row in got] == [0, 0, 0, 1, 1, 1, 1, 1, 1]
    assert [row.agent_id for row in got] == [20, 21, 22, 0, 1, 2, 10, 11, 12]


def test_exchange_cuts_duplicate_keys_both_delivered_in_rank_order() -> None:
    owners = np.array([0])

    def fn(t):
        if t.rank == 0:
            rows: list[CutRow] = []
        else:
            rows = [
                CutRow(
                    rep_id=0,
                    agent_id=5,
                    phi=np.array([float(t.rank)]),
                    epsilon=float(t.rank),
                    bundle_key=b"K",
                )
            ]
        return t.exchange_cuts(rows, owners)

    outs = LocalCluster(3).run(fn)
    got = outs[0]
    assert len(got) == 2
    assert got[0].canonical_key == got[1].canonical_key == (0, 5, b"K")
    # Rank-major stability: rank 1's duplicate precedes rank 2's.
    assert [row.epsilon for row in got] == [1.0, 2.0]


def test_exchange_cuts_bundle_key_breaks_ties_across_ranks() -> None:
    # Every delivered row shares (rep_id, agent_id), so bundle_key is the
    # sole canonical tiebreak. Each rank contributes its two rows bundle-key-
    # descending; the canonical order returns the pooled six ascending, which
    # neither a key-blind nor a key-reversed sort reproduces.
    owners = np.array([0])

    def fn(t):
        hi = bytes([90 - t.rank])  # rank 0 -> b'Z', rank 1 -> b'Y', ...
        lo = bytes([65 + t.rank])  # rank 0 -> b'A', rank 1 -> b'B', ...
        rows = [
            CutRow(
                rep_id=0,
                agent_id=7,
                phi=np.array([float(t.rank)]),
                epsilon=0.0,
                bundle_key=hi,
            ),
            CutRow(
                rep_id=0,
                agent_id=7,
                phi=np.array([float(t.rank)]),
                epsilon=0.0,
                bundle_key=lo,
            ),
        ]
        return t.exchange_cuts(rows, owners)

    got = LocalCluster(3).run(fn)[0]
    delivered = [row.bundle_key for row in got]
    pooled_keys = [bytes([90 - r]) for r in range(3)] + [
        bytes([65 + r]) for r in range(3)
    ]
    assert delivered == sorted(pooled_keys)
    assert delivered == [b"A", b"B", b"C", b"X", b"Y", b"Z"]

    # Serial path: two rows sharing (rep, agent) fed descending must come
    # back ascending; a key-blind stable sort would return [b"zzz", b"aaa"].
    r_zzz = CutRow(
        rep_id=0, agent_id=5, phi=[1.0], epsilon=0.0, bundle_key=b"zzz"
    )
    r_aaa = CutRow(
        rep_id=0, agent_id=5, phi=[2.0], epsilon=0.0, bundle_key=b"aaa"
    )
    serial = SerialTransport().exchange_cuts([r_zzz, r_aaa], np.array([0]))
    assert [row.bundle_key for row in serial] == [b"aaa", b"zzz"]


def test_exchange_cuts_rep_id_range_enforced() -> None:
    row = CutRow(
        rep_id=5,
        agent_id=0,
        phi=np.array([1.0]),
        epsilon=0.0,
        bundle_key=b"k",
    )
    with pytest.raises(ValueError, match="out of range"):
        SerialTransport().exchange_cuts([row], np.zeros(2, dtype=np.int64))

    # rep_id == B (2, against length-2 owners) must raise the contracted
    # 'out of range' ValueError, not fall through to a raw IndexError from
    # owners[row.rep_id].
    boundary = CutRow(
        rep_id=2,
        agent_id=0,
        phi=np.array([1.0]),
        epsilon=0.0,
        bundle_key=b"k",
    )
    with pytest.raises(ValueError, match="out of range"):
        SerialTransport().exchange_cuts(
            [boundary], np.zeros(2, dtype=np.int64)
        )
    # rep_id == B - 1 (1) is the top of the valid range and is delivered.
    top = CutRow(
        rep_id=1,
        agent_id=0,
        phi=np.array([1.0]),
        epsilon=0.0,
        bundle_key=b"k",
    )
    out = SerialTransport().exchange_cuts([top], np.zeros(2, dtype=np.int64))
    assert len(out) == 1 and out[0].rep_id == 1


# --- collective() error agreement ------------------------------------------


def test_collective_error_agreement_three_ranks() -> None:
    def fn(t):
        try:
            with t.collective():
                if t.rank == 1:
                    raise ValueError("boom-from-rank-1")
        except TransportError as exc:
            return ("caught", exc.rank, exc.message)
        return ("no-error", None, None)

    outs = LocalCluster(3).run(fn)  # returning at all proves no rank hangs
    assert [o[0] for o in outs] == ["caught"] * 3
    assert {o[1] for o in outs} == {1}
    for o in outs:
        assert "boom-from-rank-1" in o[2] and "ValueError" in o[2]


def test_collective_multiple_failures_agree_on_lowest_rank() -> None:
    def fn(t):
        try:
            with t.collective():
                if t.rank >= 1:
                    raise RuntimeError(f"fail-{t.rank}")
        except TransportError as exc:
            return (exc.rank, exc.message)
        return None

    outs = LocalCluster(3).run(fn)
    assert outs == [(1, "RuntimeError: fail-1")] * 3


def test_collective_failing_rank_chains_its_cause() -> None:
    def fn(t):
        try:
            with t.collective():
                if t.rank == 1:
                    raise ValueError("origin")
        except TransportError as exc:
            return isinstance(exc.__cause__, ValueError)
        return None

    assert LocalCluster(3).run(fn) == [False, True, False]


def test_collective_transparent_when_no_rank_fails() -> None:
    def fn(t):
        with t.collective():
            local = float(t.rank)
        # A later collective still lines up, proving the agreement round
        # left the rendezvous clean on every rank.
        return t.allreduce_max(local)

    assert LocalCluster(3).run(fn) == [2.0, 2.0, 2.0]


def test_collective_serial_reraises_as_transport_error() -> None:
    t = SerialTransport()
    with pytest.raises(TransportError) as info:
        with t.collective():
            raise ValueError("solo")
    assert info.value.rank == 0
    assert "solo" in info.value.message and "ValueError" in info.value.message
    assert isinstance(info.value.__cause__, ValueError)
    with t.collective():
        pass  # the no-error path is transparent
    inner = TransportError(0, "already agreed")
    with pytest.raises(TransportError) as info2:
        with t.collective():
            raise inner
    assert info2.value is inner  # agreed form passes through unchanged


# --- determinism replay ----------------------------------------------------


def test_determinism_replay_multi_collective_scenario() -> None:
    def scenario(t):
        rng = np.random.default_rng(1000 + t.rank)
        trace: list[bytes] = []
        theta = t.bcast(np.array([1.5, -2.5, 3.25]) if t.rank == 0 else None)
        trace.append(bits(theta))
        ids = np.arange(23)[t.rank :: t.size]
        trace.append(
            bits(t.sum_reproducible(spread_values(rng, (ids.size, 4)), ids))
        )
        trace.append(bits(t.sum_vectors_reproducible(spread_values(rng, (2, 4)))))
        trace.append(bits(t.batched_max(spread_values(rng, (5,)))))
        owners = np.array([0, 1, 2, 0, 1])
        owner_rows = np.full((owners.size, 3), float(t.rank), dtype=np.float64)
        owner_rows[np.flatnonzero(owners == t.rank)] = spread_values(
            rng, (np.count_nonzero(owners == t.rank), 3)
        )
        trace.append(bits(t.owner_broadcast(owner_rows, owners)))
        rows = [
            CutRow(
                rep_id=int(rng.integers(0, 5)),
                agent_id=int(rng.integers(0, 50)),
                phi=spread_values(rng, (3,)),
                epsilon=float(rng.uniform()),
                bundle_key=bytes([65 + t.rank]),
            )
            for _ in range(4)
        ]
        received = t.exchange_cuts(rows, owners)
        trace.append(
            b"|".join(row.phi.tobytes() + row.bundle_key for row in received)
        )
        with t.collective():
            pass
        trace.append(bits(t.allreduce_max(float(rng.uniform(-1e6, 1e6)))))
        return tuple(trace)

    first = LocalCluster(3, ranks_per_node=2).run(scenario)
    second = LocalCluster(3, ranks_per_node=2).run(scenario)
    assert first == second  # bitwise: every trace entry is raw bytes


# --- cluster runner ------------------------------------------------------------


def test_run_relays_uncaught_rank_errors() -> None:
    def fn(t):
        raise RuntimeError(f"dead-{t.rank}")

    with pytest.raises(RuntimeError, match="dead-0"):
        LocalCluster(3).run(fn)


def test_doubles_implement_every_abstract_method() -> None:
    assert "route_agent_values_batched" not in Transport.__abstractmethods__
    assert "owner_broadcast" not in Transport.__abstractmethods__
    assert SerialTransport.__abstractmethods__ == frozenset()
    assert LocalMultirankTransport.__abstractmethods__ == frozenset()


def test_local_cluster_validates_construction() -> None:
    for kwargs in (
        dict(size=0),
        dict(size=2, ranks_per_node=0),
        dict(size=2, rendezvous_timeout=0.0),
    ):
        with pytest.raises(ValueError):
            LocalCluster(**kwargs)  # type: ignore[arg-type]
