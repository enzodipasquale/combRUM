from __future__ import annotations

import numpy as np
import pytest

from combrum.interface_resolution import FeatureMap
from combrum.engine.observed import observed_objective, observed_phi_rows
from combrum.transport import LocalCluster, SerialTransport


class _RowAndBatchFeatureMap(FeatureMap):
    def __init__(self) -> None:
        self.features_calls = 0
        self.batch_calls = 0

    def features(self, agent_id: int, bundle: np.ndarray):
        self.features_calls += 1
        return np.asarray(bundle, dtype=np.float64), float(agent_id)

    def features_batch(self, ids: np.ndarray, bundles: np.ndarray):
        self.batch_calls += 1
        ids = np.asarray(ids, dtype=np.float64)
        Phi = np.asarray(bundles, dtype=np.float64) + ids[:, None]
        Eps = ids + 0.5
        return Phi, Eps


class _AggregatingFeatureMap(FeatureMap):
    def __init__(self) -> None:
        self.row_calls = 0
        self.aggregate_calls = 0

    def features_batch(
        self,
        ids: np.ndarray,
        bundles: np.ndarray,
        *,
        weights: np.ndarray | None = None,
        K: int | None = None,
        aggregate: bool = False,
    ):
        ids = np.asarray(ids, dtype=np.float64)
        Phi = np.asarray(bundles, dtype=np.float64) + ids[:, None]
        if aggregate:
            self.aggregate_calls += 1
            weighted = np.asarray(weights, dtype=np.float64)[:, None] * Phi
            return np.sum(weighted, axis=0), 0.0
        self.row_calls += 1
        return Phi, ids + 0.5


def test_observed_phi_rows_prefers_features_batch_over_per_row() -> None:
    # non-square observed (3 bundles x 2 features); ids wrap past the row count
    observed = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    ids = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    fmap = _RowAndBatchFeatureMap()

    Phi = observed_phi_rows(
        K=2,
        observed_bundles=observed,
        local_ids=ids,
        features=fmap,
        observed_features=None,
    )

    # ids mod 3 -> rows [0,1,2,0,1]
    selected = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(Phi, selected + ids[:, None].astype(np.float64))
    assert fmap.batch_calls == 1
    assert fmap.features_calls == 0


def test_observed_phi_rows_validates_features_batch_eps_shape() -> None:
    # the observed-row path discards Eps but must still shape-check it
    class _BadEpsBatchMap(FeatureMap):
        def features(self, agent_id: int, bundle: np.ndarray):
            return np.asarray(bundle, dtype=np.float64), float(agent_id)

        def features_batch(self, ids: np.ndarray, bundles: np.ndarray):
            ids = np.asarray(ids, dtype=np.float64)
            Phi = np.asarray(bundles, dtype=np.float64) + ids[:, None]
            # Phi is the right shape; Eps is one element too long
            Eps = np.zeros(ids.size + 1, dtype=np.float64)
            return Phi, Eps

    observed = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    ids = np.array([0, 1, 2, 3, 4], dtype=np.int64)

    with pytest.raises(ValueError) as excinfo:
        observed_phi_rows(
            K=2,
            observed_bundles=observed,
            local_ids=ids,
            features=_BadEpsBatchMap(),
            observed_features=None,
        )
    # 5 ids -> expected (5,), got (6,)
    message = str(excinfo.value)
    assert "Eps" in message
    assert "(6,)" in message
    assert "(5,)" in message


def test_explicit_observed_features_are_phi_only_and_skip_priced_features() -> None:
    # distinct rows; ids [1,2,5] mod 3 select rows [1,2,2]
    observed = np.array([[2.0, 0.0], [0.0, 3.0], [5.0, 7.0]], dtype=np.float64)
    ids = np.array([1, 2, 5], dtype=np.int64)
    fmap = _RowAndBatchFeatureMap()
    calls: list[tuple[int, tuple[float, ...]]] = []

    def observed_features(agent_id: int, bundle: np.ndarray) -> np.ndarray:
        calls.append((agent_id, tuple(np.asarray(bundle, dtype=np.float64))))
        return 2.0 * np.asarray(bundle, dtype=np.float64)

    Phi = observed_phi_rows(
        K=2,
        observed_bundles=observed,
        local_ids=ids,
        features=fmap,
        observed_features=observed_features,
    )

    # rows [1,2,2], phi = 2*bundle
    np.testing.assert_array_equal(
        Phi,
        np.array([[0.0, 6.0], [10.0, 14.0], [10.0, 14.0]], dtype=np.float64),
    )
    # the hook sees each (agent_id, selected bundle) pair
    assert calls == [
        (1, (0.0, 3.0)),
        (2, (5.0, 7.0)),
        (5, (5.0, 7.0)),
    ]
    assert fmap.batch_calls == 0
    assert fmap.features_calls == 0


def test_observed_objective_infers_from_features_with_modulo_indexing() -> None:
    # non-square observed (3x2); ids 4,5 wrap past the row count
    observed = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]], dtype=np.float64)
    # only ids {0,1} are observed (< N=4): the empirical-moment denominator is
    # N, not the number of present rows
    ids = np.array([0, 1, 4, 5], dtype=np.int64)
    N = 4
    theta_coef = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    fmap = _RowAndBatchFeatureMap()

    c_theta, empirical_moment = observed_objective(
        K=2,
        N=N,
        theta_coef=theta_coef,
        observed_bundles=observed,
        local_ids=ids,
        transport=SerialTransport(),
        features=fmap,
        observed_features=None,
    )

    # ids [0,1,4,5] mod 3 -> rows [0,1,1,2]; Phi rows add ids[:,None];
    # c_theta = -sum(theta[id] * phi_row)
    Phi = np.array(
        [
            [1.0 + 0.0, 0.0 + 0.0],
            [0.0 + 1.0, 1.0 + 1.0],
            [0.0 + 4.0, 1.0 + 4.0],
            [2.0 + 5.0, 2.0 + 5.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(c_theta, np.array([-65.0, -71.0]))
    np.testing.assert_array_equal(
        c_theta, -np.sum(theta_coef[ids, None] * Phi, axis=0)
    )
    # empirical moment divides by N=4, not by the two present rows
    obs_sum = np.zeros(2, dtype=np.float64)
    for local_id, phi_row in zip(ids.tolist(), Phi):
        if local_id < N:
            obs_sum += phi_row
    np.testing.assert_array_equal(empirical_moment, obs_sum / float(N))
    np.testing.assert_array_equal(empirical_moment, np.array([0.5, 0.5]))
    assert fmap.batch_calls == 1
    assert fmap.features_calls == 0


def test_observed_objective_uses_features_batch_aggregate_mode() -> None:
    # observed has more rows than N, so replica rows index observed mod N.
    # ids 1,2 are the observed rows; 7,8 are replicas that wrap; id 3 sits on
    # the boundary (== N) and must be excluded by the `ids < N` observed mask.
    observed = np.array(
        [[1.0, 0.0], [0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
        dtype=np.float64,
    )
    N = 3
    ids = np.array([1, 2, 3, 7, 8], dtype=np.int64)
    # theta_coef indexed by id, so it must span up to max(id)=8.
    theta_coef = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float64
    )
    fmap = _AggregatingFeatureMap()

    c_theta, empirical_moment = observed_objective(
        K=2,
        N=N,
        theta_coef=theta_coef,
        observed_bundles=observed,
        local_ids=ids,
        transport=SerialTransport(),
        features=fmap,
        observed_features=None,
    )

    # the aggregate path indexes observed mod N
    Phi = observed[ids % N] + ids[:, None].astype(np.float64)
    np.testing.assert_allclose(c_theta, -np.sum(theta_coef[ids, None] * Phi, axis=0))
    # unit-weight sum of observed[obs_ids] (fixture adds obs_ids), divided by N
    obs_ids = ids[ids < N]
    obs_sum = (observed[obs_ids] + obs_ids[:, None].astype(np.float64)).sum(axis=0)
    np.testing.assert_allclose(empirical_moment, obs_sum / float(N))
    assert fmap.aggregate_calls == 2
    assert fmap.row_calls == 0


def test_observed_aggregate_fast_path_is_serial_only_for_rank_invariance() -> None:
    class _PassThroughAggregate(FeatureMap):
        def __init__(self) -> None:
            self.aggregate_calls = 0
            self.row_calls = 0

        def features_batch(
            self,
            ids: np.ndarray,
            bundles: np.ndarray,
            *,
            weights: np.ndarray | None = None,
            K: int | None = None,
            aggregate: bool = False,
        ):
            Phi = np.asarray(bundles, dtype=np.float64)
            if aggregate:
                self.aggregate_calls += 1
                return np.sum(np.asarray(weights)[:, None] * Phi, axis=0), 0.0
            self.row_calls += 1
            return Phi, np.zeros(len(ids), dtype=np.float64)

    observed = np.array([[1e16], [1.0], [-1e16], [1.0]], dtype=np.float64)
    theta_coef = np.ones(4, dtype=np.float64)

    def run_objective(transport):
        fmap = _PassThroughAggregate()
        ids = np.arange(transport.rank, 4, transport.size, dtype=np.int64)
        c_theta, empirical_moment = observed_objective(
            K=1,
            N=4,
            theta_coef=theta_coef,
            observed_bundles=observed,
            local_ids=ids,
            transport=transport,
            features=fmap,
            observed_features=None,
        )
        return c_theta, empirical_moment, fmap.aggregate_calls, fmap.row_calls

    serial_c, serial_m, serial_agg, serial_rows = run_objective(SerialTransport())
    assert serial_agg == 2
    assert serial_rows == 0

    # order-sensitive fixture: an id-order left fold gives 1e16 + 1 - 1e16 + 1
    # = 1.0, while e.g. 1e16 - 1e16 + 1 + 1 gives 2.0, so the exact value
    # depends on the summation order. c_theta = -1.0; moment = 1.0 / 4 = 0.25.
    canonical_reduce = float(np.add.reduce(observed[:, 0]))
    np.testing.assert_array_equal(serial_c, np.array([-canonical_reduce]))
    np.testing.assert_array_equal(serial_m, np.array([canonical_reduce / 4.0]))
    assert serial_c[0] == -1.0
    assert serial_m[0] == 0.25

    outcomes = LocalCluster(2).run(run_objective)
    for c_theta, empirical_moment, aggregate_calls, row_calls in outcomes:
        assert c_theta.tobytes() == serial_c.tobytes()
        assert empirical_moment.tobytes() == serial_m.tobytes()
        assert aggregate_calls == 0
        assert row_calls == 1


def test_observed_objective_hook_takes_precedence_over_priced_features() -> None:
    transport = SerialTransport()

    class _HookedMap:
        def __init__(self) -> None:
            self.hook_calls = 0
            self.priced_calls = 0
            self.seen: dict[str, object] | None = None

        def observed_objective(
            self, K, N, theta_coef, observed_bundles, local_ids, transport
        ):
            self.hook_calls += 1
            self.seen = {
                "K": K,
                "N": N,
                "theta_coef": np.asarray(theta_coef, dtype=np.float64).copy(),
                "observed_bundles": np.asarray(
                    observed_bundles, dtype=np.float64
                ).copy(),
                "local_ids": np.asarray(local_ids, dtype=np.int64).copy(),
                "transport": transport,
            }
            # returns depend on every forwarded argument
            theta_coef = np.asarray(theta_coef, dtype=np.float64)
            observed_bundles = np.asarray(observed_bundles, dtype=np.float64)
            local_ids = np.asarray(local_ids, dtype=np.int64)
            c_theta = theta_coef * float(K + N)
            empirical_moment = observed_bundles.sum(axis=0) + float(local_ids.sum())
            return c_theta, empirical_moment

        def __call__(self, agent_id: int, bundle: np.ndarray):
            self.priced_calls += 1
            return np.asarray(bundle, dtype=np.float64), 0.0

    features = _HookedMap()
    # K != N so a swapped forward is visible
    K = 2
    N = 5
    theta_coef = np.array([1.0, 2.0], dtype=np.float64)
    observed_bundles = np.array([[3.0, 5.0], [7.0, 11.0]], dtype=np.float64)
    local_ids = np.array([0, 1], dtype=np.int64)

    c_theta, empirical_moment = observed_objective(
        K=K,
        N=N,
        theta_coef=theta_coef,
        observed_bundles=observed_bundles,
        local_ids=local_ids,
        transport=transport,
        features=features,
        observed_features=None,
    )

    # hook wins over the priced-feature backstop
    assert features.hook_calls == 1
    assert features.priced_calls == 0

    # every argument should reach the hook as passed
    seen = features.seen
    assert seen is not None
    assert seen["K"] == K
    assert seen["N"] == N
    np.testing.assert_array_equal(seen["theta_coef"], theta_coef)
    np.testing.assert_array_equal(seen["observed_bundles"], observed_bundles)
    np.testing.assert_array_equal(seen["local_ids"], local_ids)
    assert seen["transport"] is transport

    # c_theta = theta_coef * (K + N) = [1,2] * 7 = [7, 14];
    # empirical_moment = observed_bundles.sum(axis=0) + local_ids.sum()
    #                  = [10, 16] + 1 = [11, 17]
    np.testing.assert_allclose(c_theta, np.array([7.0, 14.0]))
    np.testing.assert_allclose(empirical_moment, np.array([11.0, 17.0]))
