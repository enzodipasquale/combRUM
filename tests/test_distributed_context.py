from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from _support.commprobe import CountingTransport
from combrum.context import ResultPublication
from combrum.cut_policies import SlackStrip
from combrum.engine.distributed_context import (
    build_distributed_fit_context,
    distributed_c_theta,
    owned_agent_ids,
    owned_observation_ids,
    prepare_distributed_observed,
)
from combrum.formulations import NSlack
from combrum.model import Model
from combrum.parameters import Parameters
from combrum.policies import policy_profile
from combrum.transport import (
    CutRow,
    LocalCluster,
    SerialTransport,
    TransportError,
)


class _CapturingTransport(CountingTransport):
    def __init__(self, inner) -> None:
        super().__init__(inner)
        self.sum_ids: list[np.ndarray] = []
        self.sum_shapes: list[tuple[int, ...]] = []

    def sum_reproducible(self, values: np.ndarray, global_ids: np.ndarray):
        self.sum_shapes.append(tuple(np.asarray(values).shape))
        self.sum_ids.append(np.asarray(global_ids, dtype=np.int64).copy())
        return super().sum_reproducible(values, global_ids)


class _StubOracle:
    pass


class _ObservedSurface:
    def __init__(self, K: int, *, noncontiguous: bool = False) -> None:
        self.K = K
        self.noncontiguous = noncontiguous
        self.setup_calls: list[tuple[int, tuple[int, ...]]] = []

    def setup_observed(self, transport, observation_ids: np.ndarray) -> None:
        self.setup_calls.append(
            (transport.rank, tuple(map(int, observation_ids)))
        )

    def observed_features_batch(
        self, observation_ids: np.ndarray
    ) -> np.ndarray:
        ids = np.asarray(observation_ids, dtype=np.float64)
        rows = np.column_stack([ids + j for j in range(self.K)]).astype(
            np.float64, copy=False
        )
        if self.noncontiguous:
            padded = np.empty(
                (rows.shape[0], rows.shape[1] * 2), dtype=np.float64
            )
            padded[:, ::2] = rows
            return padded[:, ::2]
        return np.ascontiguousarray(rows)


class _BadPhiSurface:
    """Surface whose observed_features_batch returns a value tripping one leg
    of _checked_distributed_phi (type / shape / dtype), for negative tests."""

    def __init__(self, K: int, *, mode: str) -> None:
        self.K = K
        self.mode = mode

    def setup_observed(self, transport, observation_ids: np.ndarray) -> None:
        pass

    def observed_features_batch(self, observation_ids: np.ndarray):
        ids = np.asarray(observation_ids, dtype=np.float64)
        if self.mode == "list":
            # non-ndarray: a nested Python list of the right shape
            return [[float(i) + j for j in range(self.K)] for i in ids]
        if self.mode == "wrong_shape":
            # K+1 columns so the shape guard, not a downstream reduction, fires
            cols = [ids + j for j in range(self.K + 1)]
            return np.ascontiguousarray(np.column_stack(cols))
        if self.mode == "float32":
            cols = [ids + j for j in range(self.K)]
            return np.ascontiguousarray(np.column_stack(cols)).astype(np.float32)
        raise AssertionError(f"unknown mode {self.mode!r}")


class _StrictShardObservedSurface:
    def __init__(self, table: np.ndarray) -> None:
        self.table = np.ascontiguousarray(table, dtype=np.float64)
        self.K = int(self.table.shape[1])
        self.N = int(self.table.shape[0])
        self.owned: tuple[int, ...] | None = None
        self.setup_ids: tuple[int, ...] | None = None

    def setup_observed(self, transport, observation_ids: np.ndarray) -> None:
        ids = tuple(map(int, observation_ids))
        owned = tuple(owned_observation_ids(self.N, transport.rank, transport.size))
        if any(obs_id < 0 or obs_id >= self.N for obs_id in ids):
            raise AssertionError("setup_observed received agent-axis ids")
        if ids != owned:
            raise AssertionError("setup_observed must receive only owned observations")
        self.owned = owned
        self.setup_ids = ids

    def observed_features_batch(self, observation_ids: np.ndarray) -> np.ndarray:
        ids = tuple(map(int, observation_ids))
        if self.owned is None:
            raise AssertionError("observed_features_batch called before setup_observed")
        if ids != self.owned:
            raise AssertionError("observed_features_batch requested non-owned rows")
        if any(obs_id < 0 or obs_id >= self.N for obs_id in ids):
            raise AssertionError("observation ids must stay on the observed axis")
        return np.ascontiguousarray(self.table[np.asarray(ids, dtype=np.int64)])


class _SplitSetupSurface(_ObservedSurface):
    def __init__(self, K: int) -> None:
        super().__init__(K)
        self.pricing_setup_calls: list[tuple[int, tuple[int, ...]]] = []

    def setup_pricing_agents(self, transport, agent_ids: np.ndarray) -> None:
        self.pricing_setup_calls.append((transport.rank, tuple(map(int, agent_ids))))


class _PricingSetupFeatures:
    def __init__(self) -> None:
        self.pricing_setup_calls: list[tuple[int, tuple[int, ...]]] = []

    def setup_pricing_agents(self, transport, agent_ids: np.ndarray) -> None:
        self.pricing_setup_calls.append((transport.rank, tuple(map(int, agent_ids))))


class _NoPricingSetupFeatures:
    pass


def _model(surface: object, *, features: object | None = None) -> Model:
    return Model(
        _StubOracle(),  # type: ignore[arg-type]
        Parameters({"a": (-10.0, 10.0, 2), "b": (-5.0, 5.0, 1)}),
        features=object() if features is None else features,
        observed_features=surface,
        formulation=NSlack,
    )


def test_observation_owned_geometry() -> None:
    # Exact shard contents, not just lengths: round-robin, an off-by-one
    # start, and remainder-to-trailing-ranks all keep the sizes. 7 over 3:
    np.testing.assert_array_equal(
        owned_observation_ids(7, 0, 3), np.array([0, 1, 2], dtype=np.int64)
    )
    np.testing.assert_array_equal(
        owned_observation_ids(7, 1, 3), np.array([3, 4], dtype=np.int64)
    )
    np.testing.assert_array_equal(
        owned_observation_ids(7, 2, 3), np.array([5, 6], dtype=np.int64)
    )
    # 252 over 5 ranks: the remainder lands on the leading ranks and every
    # shard stays a contiguous arange.
    shards = [owned_observation_ids(252, rank, 5) for rank in range(5)]
    assert [shard.size for shard in shards] == [51, 51, 50, 50, 50]
    np.testing.assert_array_equal(shards[0], np.arange(0, 51, dtype=np.int64))
    np.testing.assert_array_equal(shards[1], np.arange(51, 102, dtype=np.int64))
    np.testing.assert_array_equal(shards[2], np.arange(102, 152, dtype=np.int64))
    np.testing.assert_array_equal(shards[3], np.arange(152, 202, dtype=np.int64))
    np.testing.assert_array_equal(shards[4], np.arange(202, 252, dtype=np.int64))
    # Shards tile [0, N) exactly once, in order, with no gaps or overlaps.
    np.testing.assert_array_equal(
        np.concatenate(shards), np.arange(252, dtype=np.int64)
    )
    agent_shards = [owned_agent_ids(20, rank, 3) for rank in range(3)]
    assert [shard.size for shard in agent_shards] == [7, 7, 6]
    np.testing.assert_array_equal(agent_shards[0], np.arange(0, 7, dtype=np.int64))
    np.testing.assert_array_equal(agent_shards[1], np.arange(7, 14, dtype=np.int64))
    np.testing.assert_array_equal(agent_shards[2], np.arange(14, 20, dtype=np.int64))
    np.testing.assert_array_equal(
        owned_agent_ids(0, 2, 5), np.empty(0, dtype=np.int64)
    )


def test_prepare_distributed_observed_splits_observed_and_agent_axes() -> None:
    N, S, K = 7, 3, 3

    def run(transport):
        return prepare_distributed_observed(
            _model(_ObservedSurface(K)),
            n_observations=N,
            n_simulations=S,
            transport=transport,
        )

    preps = LocalCluster(3).run(run)
    for rank, prep in enumerate(preps):
        owned = owned_observation_ids(N, rank, 3)
        np.testing.assert_array_equal(prep.owned_obs, owned)
        np.testing.assert_array_equal(
            prep.local_ids,
            owned_agent_ids(N * S, rank, 3),
        )
        assert prep.phi_obs_local.shape == (owned.size, K)
        # phi_obs_local is frozen by _checked_distributed_phi; the other three
        # only by DistributedObservedPrep.__post_init__, so check all four.
        assert not prep.phi_obs_local.flags.writeable
        assert not prep.owned_obs.flags.writeable
        assert not prep.local_ids.flags.writeable
        assert not prep.empirical_moment.flags.writeable

    expected_sum = np.sum(
        np.column_stack(
            [np.arange(N, dtype=np.float64) + j for j in range(K)]
        ),
        axis=0,
    )
    for prep in preps:
        np.testing.assert_allclose(prep.empirical_moment, expected_sum / N)


def test_prepare_distributed_observed_uses_separate_pricing_setup_hook() -> None:
    N, S, K = 7, 3, 3

    def run(transport):
        observed = _SplitSetupSurface(K)
        pricing = _PricingSetupFeatures()
        prep = prepare_distributed_observed(
            _model(observed, features=pricing),
            n_observations=N,
            n_simulations=S,
            transport=transport,
        )
        return (
            tuple(observed.setup_calls),
            tuple(observed.pricing_setup_calls),
            tuple(pricing.pricing_setup_calls),
            tuple(map(int, prep.owned_obs)),
            tuple(map(int, prep.local_ids)),
        )

    results = LocalCluster(3).run(run)
    for rank, (
        obs_calls,
        observed_pricing_calls,
        pricing_calls,
        owned_obs,
        local_ids,
    ) in enumerate(results):
        assert obs_calls == ((rank, owned_obs),)
        assert observed_pricing_calls == ()
        assert pricing_calls == ((rank, local_ids),)
        assert all(0 <= obs_id < N for obs_id in owned_obs)
        assert all(0 <= agent_id < N * S for agent_id in local_ids)


def test_prepare_distributed_observed_requires_rank_uniform_pricing_setup_hook() -> None:
    def run(transport):
        pricing = (
            _PricingSetupFeatures()
            if transport.rank == 0
            else _NoPricingSetupFeatures()
        )
        try:
            prepare_distributed_observed(
                _model(_ObservedSurface(2), features=pricing),
                n_observations=5,
                n_simulations=2,
                transport=transport,
            )
        except TransportError as exc:
            return exc.message
        return "no error"

    assert all(
        "distributed pricing feature surface differs across ranks" in message
        for message in LocalCluster(2).run(run)
    )


def test_prepare_distributed_observed_requires_rank_uniform_geometry() -> None:
    def mismatched_n(transport):
        try:
            prepare_distributed_observed(
                _model(_ObservedSurface(3)),
                n_observations=7 if transport.rank == 0 else 8,
                n_simulations=3,
                transport=transport,
            )
        except TransportError as exc:
            return exc.message
        return "no error"

    assert all(
        "n_observations must match" in message
        for message in LocalCluster(2).run(mismatched_n)
    )

    def mismatched_s(transport):
        try:
            prepare_distributed_observed(
                _model(_ObservedSurface(3)),
                n_observations=7,
                n_simulations=3 if transport.rank == 0 else 4,
                transport=transport,
            )
        except TransportError as exc:
            return exc.message
        return "no error"

    assert all(
        "n_simulations must match" in message
        for message in LocalCluster(2).run(mismatched_s)
    )


def test_prepare_distributed_observed_requires_rank_uniform_k() -> None:
    def run(transport):
        K = 2 if transport.rank == 0 else 3
        model = Model(
            _StubOracle(),  # type: ignore[arg-type]
            Parameters({"theta": (-1.0, 1.0, K)}),
            features=object(),
            observed_features=_ObservedSurface(K),
            formulation=NSlack,
        )
        try:
            prepare_distributed_observed(
                model,
                n_observations=5,
                n_simulations=2,
                transport=transport,
            )
        except TransportError as exc:
            return exc.message
        return "no error"

    assert all(
        "model.parameters must match" in message
        for message in LocalCluster(2).run(run)
    )


def test_distributed_c_theta_reduces_over_observations_not_agents() -> None:
    N, S, K = 5, 4, 3
    weights = np.arange(1, N + 1, dtype=np.float64)

    def run(transport):
        prep = prepare_distributed_observed(
            _model(_ObservedSurface(K)),
            n_observations=N,
            n_simulations=S,
            transport=transport,
        )
        local_w = weights[prep.owned_obs]
        return distributed_c_theta(
            prep,
            obs_weights_local=local_w,
            transport=transport,
        )

    out = LocalCluster(2).run(run)
    phi = np.column_stack(
        [np.arange(N, dtype=np.float64) + j for j in range(K)]
    )
    expected = -S * np.sum(weights[:, None] * phi, axis=0)
    for c_theta in out:
        np.testing.assert_allclose(c_theta, expected)


def test_distributed_observed_reductions_are_keyed_by_observations() -> None:
    transport = _CapturingTransport(SerialTransport())
    prep = prepare_distributed_observed(
        _model(_ObservedSurface(3)),
        n_observations=11,
        n_simulations=5,
        transport=transport,
    )
    c_theta = distributed_c_theta(prep, transport=transport)

    assert c_theta.shape == (3,)
    # obs_weights_local is left at its None default here (nowhere else), so
    # this exercises the unit-weight branch. Column j sums to
    # sum_i(i + j) = 55 + 11*j over i in [0, 11); c_theta = -S * that, S=5.
    np.testing.assert_array_equal(
        c_theta, np.array([-275.0, -330.0, -385.0])
    )
    counts = transport.counts()
    assert counts["sum_reproducible"] == 2
    assert "sum_vectors_reproducible" not in counts
    assert transport.sum_shapes == [(11, 3), (11, 3)]
    for ids in transport.sum_ids:
        np.testing.assert_array_equal(ids, np.arange(11, dtype=np.int64))
        assert ids.min() >= 0
        assert ids.max() < 11
    assert transport.bytes_moved()["sum_reproducible"] == 2 * (
        11 * 3 * 8 + 11 * 8
    )


def test_distributed_observed_reductions_are_bitwise_rank_invariant() -> None:
    N, S, K = 252, 20, 3
    weights = np.linspace(0.25, 2.75, N, dtype=np.float64)

    def run(transport):
        prep = prepare_distributed_observed(
            _model(_ObservedSurface(K)),
            n_observations=N,
            n_simulations=S,
            transport=transport,
        )
        c_theta = distributed_c_theta(
            prep,
            obs_weights_local=weights[prep.owned_obs],
            transport=transport,
        )
        return prep.empirical_moment.tobytes(), c_theta.tobytes()

    expected = run(SerialTransport())
    for size in (2, 4, 5):
        for got in LocalCluster(size).run(run):
            assert got == expected


def test_observed_feature_surface_never_receives_agent_axis_ids() -> None:
    N, S = 7, 4
    table = np.column_stack(
        [
            np.arange(1, N + 1, dtype=np.float64),
            10.0 + 3.0 * np.arange(N, dtype=np.float64),
            100.0 + 5.0 * np.arange(N, dtype=np.float64),
        ]
    )
    weights = np.linspace(0.5, 2.0, N, dtype=np.float64)

    def run(transport):
        surface = _StrictShardObservedSurface(table)
        prep = prepare_distributed_observed(
            _model(surface),
            n_observations=N,
            n_simulations=S,
            transport=transport,
        )
        c_theta = distributed_c_theta(
            prep,
            obs_weights_local=weights[prep.owned_obs],
            transport=transport,
        )
        return (
            tuple(map(int, prep.owned_obs)),
            tuple(map(int, prep.local_ids)),
            prep.empirical_moment.copy(),
            c_theta.copy(),
        )

    results = LocalCluster(3).run(run)

    expected_empirical = table.sum(axis=0) / float(N)
    expected_c_theta = -float(S) * np.sum(weights[:, None] * table, axis=0)
    assert [x.hex() for x in expected_empirical] == [
        "0x1.0000000000000p+2",
        "0x1.3000000000000p+4",
        "0x1.cc00000000000p+6",
    ]
    assert [x.hex() for x in expected_c_theta] == [
        "-0x1.5000000000000p+7",
        "-0x1.7680000000000p+9",
        "-0x1.0450000000000p+12",
    ]
    seen_obs: list[int] = []
    seen_agents: list[int] = []
    for owned, local_ids, empirical, c_theta in results:
        seen_obs.extend(owned)
        seen_agents.extend(local_ids)
        np.testing.assert_allclose(empirical, expected_empirical)
        np.testing.assert_allclose(c_theta, expected_c_theta)
    assert sorted(seen_obs) == list(range(N))
    assert sorted(seen_agents) == list(range(N * S))


def test_prepare_distributed_observed_rejects_noncontiguous_phi() -> None:
    with pytest.raises(TransportError, match="C-contiguous"):
        prepare_distributed_observed(
            _model(_ObservedSurface(3, noncontiguous=True)),
            n_observations=3,
            n_simulations=2,
            transport=SerialTransport(),
        )


def test_prepare_distributed_observed_validates_phi_surface_return() -> None:
    # Each match string is specific to its leg of _checked_distributed_phi:
    # without the ndarray check a list fails later with AttributeError, the
    # K+1-column array would instead trip the reduction-shape guard inside
    # sum_reproducible, and float32 would coerce to float64 silently.
    cases = [
        ("list", r"observed_features_batch must return a numpy\.ndarray"),
        ("wrong_shape", r"observed_features_batch returned shape \(3, 4\)"),
        ("float32", r"observed_features_batch must return float64 rows"),
    ]
    for mode, pattern in cases:
        with pytest.raises(TransportError, match=pattern):
            prepare_distributed_observed(
                _model(_BadPhiSurface(3, mode=mode)),
                n_observations=3,
                n_simulations=2,
                transport=SerialTransport(),
            )


def test_build_distributed_fit_context_has_no_dense_weight_arrays(
    monkeypatch,
) -> None:
    import combrum.engine.distributed_context as dc

    made: list[dict[str, object]] = []

    class _Master:
        def reinstall(self, rows):
            pass

    def fake_make_master(
        K, bounds, c_theta, u_coef, *, backend, params, n_agents
    ):
        lb, ub = bounds
        made.append(
            {
                "K": K,
                "lb": np.asarray(lb).copy(),
                "ub": np.asarray(ub).copy(),
                "c_theta": np.asarray(c_theta).copy(),
                "u0": u_coef(0),
                "u4": u_coef(4),
                "backend": backend,
                "params": params,
                "n_agents": n_agents,
            }
        )
        return _Master()

    monkeypatch.setattr(dc, "make_master", fake_make_master)
    prep = prepare_distributed_observed(
        _model(_ObservedSurface(3)),
        n_observations=3,
        n_simulations=2,
        transport=SerialTransport(),
    )
    c_theta = distributed_c_theta(
        prep,
        obs_weights_local=np.ones(prep.owned_obs.size, dtype=np.float64),
        transport=SerialTransport(),
    )
    built = build_distributed_fit_context(
        prep,
        model=_model(_ObservedSurface(3)),
        c_theta=c_theta,
        slack_coef=lambda gid: float((gid % prep.N) + 10),
        transport=SerialTransport(),
        owner_rank=0,
        master_backend="highs",
        master_params=None,
        tolerance=1e-6,
        result_publication=["summary"],
    )

    assert built.ctx.weight_mode == "distributed"
    assert built.ctx.theta_coef is None
    assert built.ctx.agent_weights is None
    assert built.ctx.slack_weight(4) == 11.0
    # ['summary'] must land as SUMMARY (== 0), not get promoted to a streamed
    # payload flag; nslack reads this to pick which final artifacts to publish.
    assert built.ctx.result_publication == ResultPublication.SUMMARY
    assert built.ctx.result_publication & ResultPublication.DUAL == 0
    assert made[0]["n_agents"] is None
    assert made[0]["u0"] == 10.0
    assert made[0]["u4"] == 11.0
    assert made[0]["backend"] == "highs"
    # K and theta bounds reach make_master via prep.K and the model's
    # Parameters, a path distinct from the FitContext K; lb/ub follow the
    # (-10,10,2)+(-5,5,1) block spec.
    assert made[0]["K"] == 3
    np.testing.assert_array_equal(made[0]["lb"], np.array([-10.0, -10.0, -5.0]))
    np.testing.assert_array_equal(made[0]["ub"], np.array([10.0, 10.0, 5.0]))
    # phi column sums [3,6,9] over rows [0,1,2],[1,2,3],[2,3,4], times -S=-2
    # with unit weights.
    expected_c_theta = np.array([-6.0, -12.0, -18.0])
    np.testing.assert_array_equal(made[0]["c_theta"], expected_c_theta)
    np.testing.assert_array_equal(np.asarray(built.c_theta), expected_c_theta)

    # A 'dual' request lands as DUAL, so the SUMMARY mapping above is not a
    # constant.
    dual_built = build_distributed_fit_context(
        prep,
        model=_model(_ObservedSurface(3)),
        c_theta=c_theta,
        slack_coef=lambda gid: float((gid % prep.N) + 10),
        transport=SerialTransport(),
        owner_rank=0,
        master_backend="highs",
        master_params=None,
        tolerance=1e-6,
        result_publication="dual",
    )
    assert dual_built.ctx.result_publication == ResultPublication.DUAL


def test_build_distributed_fit_context_supports_nonzero_owner(
    monkeypatch,
) -> None:
    import combrum.engine.distributed_context as dc

    class _Master:
        def reinstall(self, rows):
            pass

    monkeypatch.setattr(
        dc,
        "make_master",
        lambda *args, **kwargs: _Master(),
    )

    def run(transport):
        prep = prepare_distributed_observed(
            _model(_ObservedSurface(3)),
            n_observations=3,
            n_simulations=2,
            transport=transport,
        )
        c_theta = distributed_c_theta(
            prep,
            obs_weights_local=np.ones(prep.owned_obs.size, dtype=np.float64),
            transport=transport,
        )
        built = build_distributed_fit_context(
            prep,
            model=_model(_ObservedSurface(3)),
            c_theta=c_theta,
            slack_coef=lambda gid: float(gid),
            transport=transport,
            owner_rank=1,
            master_backend="highs",
            master_params=None,
            tolerance=1e-6,
            result_publication="summary",
        )
        return (
            built.ctx.owner_rank,
            built.ctx.master_backend is not None,
            built.ctx.result_publication == ResultPublication.SUMMARY,
        )

    assert LocalCluster(2).run(run) == [(1, False, True), (1, True, True)]


def test_build_distributed_fit_context_keeps_gurobi_warm_start_defaults(
    monkeypatch,
) -> None:
    import combrum.engine.distributed_context as dc

    captured: list[dict[str, object] | None] = []

    class _Master:
        def reinstall(self, rows) -> None:
            raise AssertionError("warm cuts are not part of this test")

    def fake_make_master(
        K,
        bounds,
        c_theta,
        u_coef,
        *,
        backend,
        params,
        n_agents,
    ):
        captured.append(None if params is None else dict(params))
        return _Master()

    monkeypatch.setattr(dc, "make_master", fake_make_master)
    transport = SerialTransport()
    prep = prepare_distributed_observed(
        _model(_ObservedSurface(3)),
        n_observations=3,
        n_simulations=2,
        transport=transport,
    )
    c_theta = distributed_c_theta(
        prep,
        obs_weights_local=np.ones(prep.owned_obs.size, dtype=np.float64),
        transport=transport,
    )
    user_params = {"TimeLimit": 3.0, "LPWarmStart": 1}

    build_distributed_fit_context(
        prep,
        model=_model(_ObservedSurface(3)),
        c_theta=c_theta,
        slack_coef=lambda gid: float(gid + 1),
        transport=transport,
        owner_rank=0,
        master_backend="gurobi",
        master_params=user_params,
        tolerance=1e-6,
        result_publication="summary",
    )

    assert user_params == {"TimeLimit": 3.0, "LPWarmStart": 1}
    assert captured == [{"Method": 0, "LPWarmStart": 1, "TimeLimit": 3.0}]


def _cut_row(agent_id: int, bundle_key: bytes) -> CutRow:
    return CutRow(
        rep_id=0,
        agent_id=agent_id,
        phi=np.zeros(3, dtype=np.float64),
        epsilon=0.0,
        bundle_key=bundle_key,
    )


def test_nslack_validates_lazy_slack_strip_with_no_installed_rows() -> None:
    formulation = NSlack(lambda agent_id, bundle: (bundle, 0.0))
    formulation._ctx = SimpleNamespace(weight_mode="distributed", K=3)
    policy = SlackStrip(hard_threshold=2)

    with pytest.raises(ValueError, match="K \\+ installed_agents"):
        formulation._purge(policy, policy_profile(policy), ())


def test_nslack_lazy_slack_strip_counts_distinct_installed_agents() -> None:
    # hard_threshold == K, so only the installed_agents term can push the
    # total over the guard: the empty case passes (3 >= 3), and each populated
    # case below must raise on that term alone.
    formulation = NSlack(lambda agent_id, bundle: (bundle, 0.0))
    formulation._ctx = SimpleNamespace(weight_mode="distributed", K=3)
    policy = SlackStrip(hard_threshold=3)
    profile = policy_profile(policy)

    assert formulation._purge(policy, profile, ()) == set()

    with pytest.raises(
        ValueError,
        match=r"hard_threshold=3, K=3, installed_agents=2, K \+ installed_agents=5",
    ):
        formulation._purge(
            policy, profile, (_cut_row(0, b"a"), _cut_row(1, b"b"))
        )

    # Two rows on one agent dedup to installed_agents=1: distinct agents are
    # counted, not rows.
    with pytest.raises(
        ValueError,
        match=r"hard_threshold=3, K=3, installed_agents=1, K \+ installed_agents=4",
    ):
        formulation._purge(
            policy, profile, (_cut_row(0, b"a"), _cut_row(0, b"b"))
        )
