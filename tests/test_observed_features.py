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
    # Non-square observed (3 bundles x 2 features) so the row modulus shape[0]=3
    # differs from the feature dimension shape[1]=2. ids wrap past 3, so the
    # correct `% shape[0]` selection [0,1,2,0,1] diverges from a mutated
    # `% shape[1]` selection [0,1,0,1,0].
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

    # Hand-derived selection [0,1,2,0,1] (ids mod 3, the observed-row count),
    # written out literally so the oracle does not reuse the code's shape[0].
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
    # The optimized observed-row path discards the priced-error Eps but must
    # still shape-check it: a features_batch that returns a correctly-shaped Phi
    # yet a wrong-length Eps is a malformed conformance surface and must raise.
    class _BadEpsBatchMap(FeatureMap):
        def features(self, agent_id: int, bundle: np.ndarray):
            return np.asarray(bundle, dtype=np.float64), float(agent_id)

        def features_batch(self, ids: np.ndarray, bundles: np.ndarray):
            ids = np.asarray(ids, dtype=np.float64)
            Phi = np.asarray(bundles, dtype=np.float64) + ids[:, None]
            # Phi is the right shape; Eps is deliberately one element too long.
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
    # The message pins the Eps shape mismatch (5 ids -> expected (5,), got (6,)),
    # so a Phi-only shape check passing silently would not satisfy this.
    message = str(excinfo.value)
    assert "Eps" in message
    assert "(6,)" in message
    assert "(5,)" in message


def test_explicit_observed_features_are_phi_only_and_skip_priced_features() -> None:
    # Non-square observed (3x2) with distinct rows so the row modulus shape[0]=3
    # lands on different bundles than a mutated shape[1]=2. ids [1,2,5] select
    # rows [1,2,2] under `% 3` but rows [1,0,1] under `% 2`.
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

    # Hand-derived selection: ids [1,2,5] mod 3 -> rows [1,2,2], phi = 2*bundle.
    np.testing.assert_array_equal(
        Phi,
        np.array([[0.0, 6.0], [10.0, 14.0], [10.0, 14.0]], dtype=np.float64),
    )
    # The observed_features hook sees each (agent_id, selected_bundle) pair, so
    # pinning the exact call list also pins the row selection.
    assert calls == [
        (1, (0.0, 3.0)),
        (2, (5.0, 7.0)),
        (5, (5.0, 7.0)),
    ]
    assert fmap.batch_calls == 0
    assert fmap.features_calls == 0


def test_observed_objective_infers_from_features_with_modulo_indexing() -> None:
    # Non-square observed (3 bundles x 2 features): the row modulus shape[0]=3
    # differs from the feature dimension shape[1]=2, so a mutated `% shape[1]`
    # selects different rows for the wrapping ids 4,5.
    observed = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]], dtype=np.float64)
    # Partial observed shard: N=4 but only ids {0,1} are observed (< N), so the
    # observed-row count differs from N (4). This separates the correct
    # empirical-moment denominator (1/N) from a per-present-row mean (1/count).
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

    # Hand-derived selection: ids [0,1,4,5] mod 3 -> rows [0,1,1,2]. Phi rows
    # add ids[:,None]; c_theta = -sum(theta[id] * phi_row). Both operands are
    # written literally so the oracle never reuses the code's shape[0] modulus.
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
    # Empirical moment = (sum of observed phi rows) / N, divided by N=4 (NOT by
    # the two present rows). Plain-Python accumulation, independent of combrum.
    obs_sum = np.zeros(2, dtype=np.float64)
    for local_id, phi_row in zip(ids.tolist(), Phi):
        if local_id < N:
            obs_sum += phi_row
    np.testing.assert_array_equal(empirical_moment, obs_sum / float(N))
    np.testing.assert_array_equal(empirical_moment, np.array([0.5, 0.5]))
    assert fmap.batch_calls == 1
    assert fmap.features_calls == 0


def test_observed_objective_uses_features_batch_aggregate_mode() -> None:
    # observed has more rows than N so `% N` (the correct aggregate modulus) and
    # `% observed.shape[0]` select different rows for out-of-range ids. Rows are
    # distinct so a wrong modulus lands on a different bundle. ids 7 and 8 are
    # non-observed replicas whose `% N` (1, 2) differs from `% shape[0]` (2, 3);
    # ids 1 and 2 are the observed rows feeding the empirical moment. id 3 sits
    # exactly on the boundary (== N): it is a non-observed replica, so the
    # aggregate observed-row mask `ids < N` must exclude it. A `ids <= N` off-by-
    # one would fold observed[3] into the empirical moment.
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

    # Oracle uses `% N` — the modulus the aggregate fast path MUST use. This is a
    # distinct invariant from the code path (which computes the modulus inline),
    # so swapping it for `% observed.shape[0]` moves rows 7,8 off [1,2] onto
    # [2,3] and c_theta changes.
    Phi = observed[ids % N] + ids[:, None].astype(np.float64)
    np.testing.assert_allclose(c_theta, -np.sum(theta_coef[ids, None] * Phi, axis=0))
    # Empirical moment: aggregate path sums observed[obs_ids] (raw ids) with unit
    # weight, the fixture adds obs_ids, then divides by N. obs_ids = {1, 2} pick
    # distinct rows so the sum pins the selected rows, not just their count. id 3
    # (== N) is strictly excluded here, so `ids <= N` instead of `ids < N` would
    # add observed[3] + 3 = [7, 8] and break this assertion.
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

    # Independent canonical-order oracle: a left-fold over the id-sorted observed
    # values (np.add.reduce is order-defined) is what a deterministic reduction
    # must produce. With this order-sensitive fixture, 1e16 + 1 - 1e16 + 1 = 1.0
    # (the 1s survive only because the +1e16 term is added before -1e16). c_theta
    # = -reduce = -1.0; empirical_moment = reduce / N = 1.0 / 4 = 0.25. A
    # non-canonical order (e.g. 1e16 - 1e16 + 1 + 1) would give 2.0, so pinning
    # the value also pins the summation order.
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
            # Derive returns from the forwarded arguments so a mis-forwarded,
            # reordered, or mangled positional changes the emitted result and
            # the caller's assertions below fail.
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
    # Distinct K and N so a K<->N transposition at the forward boundary is
    # visible in seen["K"]/seen["N"] (with K==N a swap is invisible).
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

    # Hook wins over the priced-feature backstop and is invoked exactly once.
    assert features.hook_calls == 1
    assert features.priced_calls == 0

    # The src forwards every positional in the documented order
    # (K, N, theta_coef, observed_bundles, local_ids, transport). Pin each one
    # so a dropped/reordered/stale argument is caught at the boundary.
    seen = features.seen
    assert seen is not None
    assert seen["K"] == K
    assert seen["N"] == N
    np.testing.assert_array_equal(seen["theta_coef"], theta_coef)
    np.testing.assert_array_equal(seen["observed_bundles"], observed_bundles)
    np.testing.assert_array_equal(seen["local_ids"], local_ids)
    assert seen["transport"] is transport

    # Independently hand-computed from the fixture (not from combrum's objective
    # math): c_theta = theta_coef * (K + N) = [1,2] * 7 = [7, 14];
    # empirical_moment = observed_bundles.sum(axis=0) + local_ids.sum()
    #                  = [10, 16] + 1 = [11, 17].
    np.testing.assert_allclose(c_theta, np.array([7.0, 14.0]))
    np.testing.assert_allclose(empirical_moment, np.array([11.0, 17.0]))
