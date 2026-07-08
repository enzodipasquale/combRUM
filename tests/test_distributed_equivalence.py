from __future__ import annotations

from collections.abc import Mapping

import combrum as cb
import numpy as np
import pytest

from _support.commprobe import CountingTransport
from _support.constants import MAX_ITERATIONS, THETA_BOUND, TOLERANCE
from _support.families import toy_family
from combrum.bootstrap_distributed import _owner_vector
from combrum.engine.certify import GapTally, certification_metadata
from combrum.engine.distributed_context import (
    owned_agent_ids,
    prepare_distributed_observed,
)
from combrum.masters import highs as highs_backend
from combrum.randomness import bootstrap_observation_weights
from combrum.transport import LocalCluster


needs_highs = pytest.mark.skipif(
    not highs_backend.available(), reason="highspy missing or broken"
)

_N = 6
_K = 2
_S = 2
_FAMILY_SEED = 20260629
# The shared 20260629 fixture yields a degenerate observed moment [1.0, 0.0];
# a zero coordinate hides any reduction error confined to it, so the full-fit
# test uses a seed with a distinct nonzero moment in every coordinate.
_MOMENT_SEED = 20260630
_BOOT_SEED = 99
_B = 4


def _simulation_shocks(
    arrays: Mapping[str, np.ndarray], n_simulations: int
) -> np.ndarray:
    base = np.asarray(arrays["shocks"], dtype=np.float64)[:, 0, :]
    shocks = np.empty((base.shape[0], int(n_simulations), base.shape[1]))
    offsets = 0.2 * (1.0 + 0.1 * np.arange(base.shape[1], dtype=np.float64))
    for sim_id in range(int(n_simulations)):
        shocks[:, sim_id, :] = base + float(sim_id) * offsets
    return np.ascontiguousarray(shocks, dtype=np.float64)


class _SplitToySurface(cb.Oracle, cb.FeatureMap):
    """Toy surface where agent_id splits into obs = agent_id % N, sim = agent_id // N."""

    def __init__(
        self, arrays: Mapping[str, np.ndarray], n_simulations: int
    ) -> None:
        self.r = np.asarray(arrays["observables"], dtype=np.float64)
        self.nu = _simulation_shocks(arrays, int(n_simulations))
        self.observed = np.asarray(arrays["observed"], dtype=bool)
        self.N = int(self.r.shape[0])
        self.S = int(n_simulations)
        self.K = int(self.r.shape[1])
        self.local_ids = np.empty(0, dtype=np.int64)
        self.observation_ids = np.empty(0, dtype=np.int64)

    def setup(self, transport, local_ids: np.ndarray) -> None:
        self.local_ids = np.asarray(local_ids, dtype=np.int64).copy()

    def setup_observed(self, transport, observation_ids: np.ndarray) -> None:
        self.observation_ids = np.asarray(observation_ids, dtype=np.int64).copy()

    def _obs_sim(self, agent_id: int) -> tuple[int, int]:
        aid = int(agent_id)
        return aid % self.N, aid // self.N

    def price(self, theta: np.ndarray, agent_id: int) -> cb.Demand:
        obs_id, sim_id = self._obs_sim(agent_id)
        scores = (
            self.r[obs_id] * np.asarray(theta, dtype=np.float64)
            + self.nu[obs_id, sim_id]
        )
        bundle = scores > 0.0
        return cb.Demand.exact(
            bundle=bundle,
            payoff=float(np.where(bundle, scores, 0.0).sum()),
        )

    def price_batch(
        self, theta: np.ndarray, local_ids: np.ndarray
    ) -> Mapping[int, cb.Demand]:
        return {
            int(agent_id): self.price(theta, int(agent_id))
            for agent_id in np.asarray(local_ids, dtype=np.int64)
        }

    def features(
        self, agent_id: int, bundle: np.ndarray
    ) -> tuple[np.ndarray, float]:
        obs_id, sim_id = self._obs_sim(agent_id)
        b = np.asarray(bundle, dtype=np.float64)
        return b * self.r[obs_id], float(b @ self.nu[obs_id, sim_id])

    def features_batch(
        self, ids: np.ndarray, bundles: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        ids = np.asarray(ids, dtype=np.int64)
        bundles = np.asarray(bundles, dtype=np.float64)
        obs_ids = ids % self.N
        sim_ids = ids // self.N
        phi = np.ascontiguousarray(
            bundles * self.r[obs_ids], dtype=np.float64
        )
        eps = np.ascontiguousarray(
            np.einsum(
                "nm,nm->n",
                bundles,
                self.nu[obs_ids, sim_ids],
                optimize=True,
            ),
            dtype=np.float64,
        )
        return phi, eps

    def __call__(self, agent_id: int, bundle: np.ndarray) -> np.ndarray:
        obs_id = int(agent_id) % self.N
        return np.asarray(bundle, dtype=np.float64) * self.r[obs_id]

    def observed_features_batch(
        self, observation_ids: np.ndarray
    ) -> np.ndarray:
        obs_ids = np.asarray(observation_ids, dtype=np.int64)
        return np.ascontiguousarray(
            self.observed[obs_ids].astype(np.float64) * self.r[obs_ids],
            dtype=np.float64,
        )


class _ObservationWeightDraws:
    def __init__(self, n_observations: int, base_seed: int) -> None:
        self.n_obs = int(n_observations)
        self.base_seed = int(base_seed)

    def weights_for(self, rep_id: int) -> np.ndarray:
        return bootstrap_observation_weights(
            self.n_obs, self.base_seed, rep_id
        )


def _arrays() -> dict[str, np.ndarray]:
    return toy_family(_N, _K, _FAMILY_SEED)


def _model(arrays: Mapping[str, np.ndarray]) -> cb.Model:
    surface = _SplitToySurface(arrays, _S)
    return cb.Model(
        surface,
        cb.Parameters({"theta": (-THETA_BOUND, THETA_BOUND, surface.K)}),
        features=surface,
        observed_features=surface,
        formulation=cb.NSlack,
    )


def _data(arrays: Mapping[str, np.ndarray]) -> cb.Data:
    return cb.Data(
        observed_bundles=np.asarray(arrays["observed"]),
        shocks=_simulation_shocks(arrays, _S),
        observables=list(range(_N)),
    )


def _observed_moment_by_hand(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    """Per-column observed moment ``(1/N) sum_i observed[i,k] * r[i,k]``."""
    r = np.asarray(arrays["observables"], dtype=np.float64)
    observed = np.asarray(arrays["observed"], dtype=bool)
    n_obs, n_items = r.shape
    acc = [0.0] * n_items
    for i in range(n_obs):
        for k in range(n_items):
            acc[k] += (1.0 if observed[i, k] else 0.0) * float(r[i, k])
    return np.array([total / float(n_obs) for total in acc], dtype=np.float64)


def test_surface_uses_distinct_simulation_ids() -> None:
    arrays = _arrays()
    surface = _SplitToySurface(arrays, _S)
    theta = np.zeros(surface.K, dtype=np.float64)
    bundle = np.ones(surface.K, dtype=bool)

    demand0 = surface.price(theta, 0)
    demand1 = surface.price(theta, _N)
    _phi0, eps0 = surface.features(0, bundle)
    _phi1, eps1 = surface.features(_N, bundle)

    assert not np.array_equal(surface.nu[:, 0, :], surface.nu[:, 1, :])
    assert demand0.payoff != demand1.payoff
    assert eps0 != eps1

    # The flattened agent axis lays out agent_id = sim * N + obs; check the
    # full one-rank ownership array against that layout.
    agent_ids = owned_agent_ids(_N * _S, 0, 1)
    expected_agent_ids = [
        sim * _N + obs_id
        for sim in range(_S)
        for obs_id in range(_N)
    ]
    assert agent_ids.dtype == np.int64
    np.testing.assert_array_equal(
        agent_ids, np.array(expected_agent_ids, dtype=np.int64)
    )
    # The agents priced above (obs 0 on sims 0 and 1) sit at slots 0 and _N.
    assert expected_agent_ids[0] == 0 and expected_agent_ids[_N] == _N


@needs_highs
def test_estimate_distributed_matches_serial_split_axis_fit() -> None:
    arrays = toy_family(_N, _K, _MOMENT_SEED)
    expected_moment = _observed_moment_by_hand(arrays)

    # The per-column assertions below need nonzero, distinct coordinates.
    assert np.all(np.abs(expected_moment) > 1e-6)
    assert abs(abs(expected_moment[0]) - abs(expected_moment[1])) > 1e-6

    serial = cb.estimate(
        _model(arrays),
        _data(arrays),
        transport=cb.SerialTransport(),
        master_backend="highs",
        tolerance=TOLERANCE,
        max_iterations=MAX_ITERATIONS,
    )

    assert serial.metadata["converged"] is True

    # The data is rationalised by theta_true (min regret 0), so the fit should
    # recover its sign per coordinate; sign(0) is 0, which also rules out a
    # theta stuck at the zero cold start.
    theta_true_sign = np.sign(np.asarray(arrays["theta_true"], dtype=np.float64))
    assert not np.allclose(serial.theta_hat, 0.0, atol=1e-6)
    np.testing.assert_array_equal(
        np.sign(serial.theta_hat), theta_true_sign
    )

    np.testing.assert_allclose(
        serial.empirical_moment, expected_moment, rtol=0.0, atol=1e-12
    )
    for size in (2, 3):
        results = LocalCluster(size).run(
            lambda transport: cb.estimate_distributed(
                _model(arrays),
                n_observations=_N,
                n_simulations=_S,
                transport=transport,
                master_backend="highs",
                tolerance=TOLERANCE,
                max_iterations=MAX_ITERATIONS,
            )
        )

        for result in results:
            assert result.metadata["converged"] is True
            np.testing.assert_allclose(
                result.theta_hat, serial.theta_hat, rtol=1e-10, atol=1e-10
            )
            np.testing.assert_allclose(
                result.empirical_moment,
                expected_moment,
                rtol=0.0,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                result.empirical_moment,
                serial.empirical_moment,
                rtol=1e-12,
                atol=1e-12,
            )


@needs_highs
def test_bootstrap_distributed_matches_serial_with_observation_weights() -> None:
    arrays = _arrays()
    serial = cb.bootstrap(
        _model(arrays),
        _data(arrays),
        n_bootstrap=_B,
        weights=_ObservationWeightDraws(_N, _BOOT_SEED),
        transport=cb.SerialTransport(),
        master_backend="highs",
        tolerance=TOLERANCE,
        max_iterations=MAX_ITERATIONS,
    )

    assert serial.converged.all()
    assert np.max(np.ptp(serial.thetas, axis=0)) > 1e-7
    clusters = (
        (LocalCluster(3), np.array([0, 1, 2, 0], dtype=np.int64)),
        (LocalCluster(4, ranks_per_node=2), np.array([0, 2, 1, 3], dtype=np.int64)),
    )
    for cluster, expected_owners in clusters:
        # Every expected owner rank should own at least one rep; the two-node
        # case interleaves ownership across nodes.
        owners = cluster.run(lambda transport: _owner_vector(_B, transport))[0]
        np.testing.assert_array_equal(owners, expected_owners)

        results = cluster.run(
            lambda transport: cb.bootstrap_distributed(
                _model(arrays),
                n_observations=_N,
                n_simulations=_S,
                n_bootstrap=_B,
                base_seed=_BOOT_SEED,
                transport=transport,
                master_backend="highs",
                tolerance=TOLERANCE,
                max_iterations=MAX_ITERATIONS,
            )
        )

        # Each super-step prices the full N*S agent set once per live rep
        # (split across ranks), so the run-wide count is N*S * _B * iterations.
        expected_n_priced = _N * _S * _B * int(results[0].iterations)
        for result in results:
            np.testing.assert_array_equal(result.converged, serial.converged)
            np.testing.assert_allclose(
                result.thetas, serial.thetas, rtol=1e-10, atol=1e-10
            )
            certification = result.metadata["certification"]
            assert certification["n_priced"] == expected_n_priced
            assert certification["n_inexact"] == 0
            assert certification["worst_gap_unknown"] is False


@needs_highs
def test_split_axis_fit_routes_agent_values_without_dense_scatter() -> None:
    arrays = _arrays()
    serial = cb.estimate(
        _model(arrays),
        _data(arrays),
        transport=cb.SerialTransport(),
        master_backend="highs",
        tolerance=TOLERANCE,
        max_iterations=MAX_ITERATIONS,
    )

    def run_rank(transport):
        probe = CountingTransport(transport)
        result = cb.estimate_distributed(
            _model(arrays),
            n_observations=_N,
            n_simulations=_S,
            transport=probe,
            master_backend="highs",
            tolerance=TOLERANCE,
            max_iterations=MAX_ITERATIONS,
        )
        # this rank's owned shard size, for the byte bounds below
        owned_here = owned_agent_ids(_N * _S, transport.rank, transport.size)
        return (
            result,
            probe.counts(),
            probe.bytes_moved(),
            int(owned_here.shape[0]),
        )

    size = 2
    results = LocalCluster(size).run(run_rank)

    # Contiguous sharding gives each rank at most ceil(N*S / size) owned
    # agents; a dense broadcast would move the full N*S map to every rank.
    owned_agents = (_N * _S + size - 1) // size
    for result, counts, bytes_moved, owned_here in results:
        # An empty route never installs the routed u/slack values, so the fit
        # would hit max_iterations without converging.
        assert result.metadata["converged"] is True
        np.testing.assert_allclose(
            result.theta_hat, serial.theta_hat, rtol=1e-10, atol=1e-10
        )
        assert counts["sum_reproducible"] == 3
        assert counts["route_agent_values"] > 0
        # A converging fit routes each rank its owned shard at least once, so
        # routed bytes are at least one owned-shard delivery (owned_here*16).
        assert owned_here > 0
        assert bytes_moved["route_agent_values"] >= owned_here * 16
        # Sparse routing: each rank only receives its owned shard, so per-rank
        # bytes stay under route_calls*owned*16; a dense broadcast would need
        # route_calls*N*S*16.
        assert owned_agents < _N * _S
        assert (
            bytes_moved["route_agent_values"]
            <= counts["route_agent_values"] * owned_agents * 16
        )


# Observed-feature rows with a distinct nonzero moment in every column, so
# the cross-rank observed reduction can be checked per coordinate.
_REDUCTION_ROWS = np.array(
    [
        [1.0, 2.0],
        [2.0, 3.0],
        [1.0, 1.0],
        [2.0, 4.0],
        [1.0, 2.0],
        [2.0, 3.0],
    ],
    dtype=np.float64,
)


class _ObservedRowSurface(cb.Oracle, cb.FeatureMap):
    """Minimal observed-feature surface returning fixed rows per observation."""

    def __init__(self, rows: np.ndarray) -> None:
        self.rows = np.ascontiguousarray(rows, dtype=np.float64)
        self.observation_ids = np.empty(0, dtype=np.int64)

    def setup_observed(self, transport, observation_ids: np.ndarray) -> None:
        self.observation_ids = np.asarray(observation_ids, dtype=np.int64).copy()

    def observed_features_batch(self, observation_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(observation_ids, dtype=np.int64)
        return np.ascontiguousarray(self.rows[ids], dtype=np.float64)

    def price(self, theta: np.ndarray, agent_id: int) -> cb.Demand:
        raise NotImplementedError

    def features(
        self, agent_id: int, bundle: np.ndarray
    ) -> tuple[np.ndarray, float]:
        raise NotImplementedError

    def __call__(self, agent_id: int, bundle: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def _column_mean_by_hand(rows: np.ndarray, n_obs: int) -> np.ndarray:
    """Per-column mean via an explicit block loop."""
    k = int(rows.shape[1])
    acc = [0.0] * k
    for i in range(int(n_obs)):
        for col in range(k):
            acc[col] += float(rows[i, col])
    return np.array([total / float(n_obs) for total in acc], dtype=np.float64)


def test_distributed_observed_reduction_recovers_per_column_moment() -> None:
    rows = _REDUCTION_ROWS
    n_obs = int(rows.shape[0])
    expected = _column_mean_by_hand(rows, n_obs)

    def _model_rows() -> cb.Model:
        surface = _ObservedRowSurface(rows)
        return cb.Model(
            surface,
            cb.Parameters({"theta": (-THETA_BOUND, THETA_BOUND, rows.shape[1])}),
            observed_features=surface,
        )

    for size in (2, 3):
        moments = LocalCluster(size).run(
            lambda transport: np.asarray(
                prepare_distributed_observed(
                    _model_rows(),
                    n_observations=n_obs,
                    n_simulations=1,
                    transport=transport,
                ).empirical_moment,
                dtype=np.float64,
            )
        )
        for moment in moments:
            np.testing.assert_allclose(moment, expected, rtol=0.0, atol=1e-12)


# The toy surfaces above price only exact demands (gap 0), so certification
# there never leaves the all-zero case; these tallies give each rank a distinct
# nonzero worst gap to exercise the count SUM and worst-gap MAX.
def _certify_across_ranks(
    per_rank: dict[int, tuple[int, int, float]], size: int
) -> list[dict[str, object]]:
    def _run(transport):
        tally = GapTally()
        priced, inexact, worst = per_rank[transport.rank]
        tally.observe_counts(priced, inexact, worst)
        return certification_metadata(tally.certify(transport))

    return LocalCluster(size).run(_run)


def test_distributed_certify_reduces_inexact_counts_and_worst_gap() -> None:
    size = 3
    # Each rank inexact with a distinct worst gap, so the global MAX (0.3)
    # differs from the rank-local worst on ranks 0 and 1.
    per_rank = {0: (10, 1, 0.1), 1: (10, 2, 0.2), 2: (10, 3, 0.3)}
    expected_priced = sum(row[0] for row in per_rank.values())  # 30
    expected_inexact = sum(row[1] for row in per_rank.values())  # 6
    expected_worst = max(row[2] for row in per_rank.values())  # 0.3

    for meta in _certify_across_ranks(per_rank, size):
        assert meta["n_priced"] == expected_priced
        assert meta["n_inexact"] == expected_inexact
        assert meta["worst_gap_unknown"] is False
        assert meta["worst_gap"] == pytest.approx(expected_worst, abs=1e-15)


def test_distributed_certify_propagates_unknown_gap() -> None:
    size = 3
    # One rank reports an unknown (inf) bound; it must reach every rank via
    # the cross-rank MAX, and the lone inexact call via the count SUM.
    per_rank = {0: (5, 1, float("inf")), 1: (5, 0, 0.0), 2: (5, 0, 0.0)}
    expected_priced = sum(row[0] for row in per_rank.values())  # 15

    for meta in _certify_across_ranks(per_rank, size):
        assert meta["n_priced"] == expected_priced
        assert meta["n_inexact"] == 1
        assert meta["worst_gap_unknown"] is True
        assert meta["worst_gap"] is None
