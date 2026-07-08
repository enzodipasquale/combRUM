from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

import combrum
from combrum.bootstrap import NativeDraws
from combrum.formulation import FormulationResult
from combrum.interface_resolution import FeatureMap
from combrum.model import Data, Model
from combrum.oracle import Oracle
from combrum.parameters import Parameters
from combrum.transport import CutRow, LocalCluster, SerialTransport


class _Oracle(Oracle):
    pass


class _Formulation:
    def __init__(self, features) -> None:  # type: ignore[no-untyped-def]
        self.features = features


def _run_fake_bootstrap(monkeypatch, model, data, *, n_bootstrap=4):
    """Drive ``bootstrap`` with mocked engine hooks.

    Replication ``b`` reports ``theta_hat = full(K, b)`` and converges on every
    rep except rep 1, so per-rep outcomes are distinguishable. The returned
    namespace records the ``weights`` and ``observed_cache`` each
    ``build_fit_context`` saw.
    """
    bootstrap_mod = importlib.import_module("combrum.bootstrap")
    resolved_calls: list[str] = []
    build_resolved: list[str] = []
    build_caches: list[object] = []
    captured_weights: list[np.ndarray] = []
    rep_counter = {"n": 0}

    def fake_resolve(requested, **kwargs):  # type: ignore[no-untyped-def]
        resolved_calls.append(requested)
        return "highs"

    def fake_build_fit_context(*args, **kwargs):  # type: ignore[no-untyped-def]
        build_resolved.append(kwargs["resolved_master_backend"])
        build_caches.append(kwargs["observed_cache"])
        captured_weights.append(np.asarray(kwargs["weights"], dtype=np.float64))
        return SimpleNamespace(ctx=SimpleNamespace())

    def fake_run_fit(*args, **kwargs):  # type: ignore[no-untyped-def]
        b = rep_counter["n"]
        rep_counter["n"] += 1
        return SimpleNamespace(
            result=FormulationResult(
                theta_hat=np.full(model.parameters.K, float(b), dtype=np.float64),
                objective=0.0,
                n_active_cuts=0,
            ),
            diagnostics=SimpleNamespace(converged=(b != 1), iterations=1),
        )

    monkeypatch.setattr(bootstrap_mod, "resolve_master_backend", fake_resolve)
    monkeypatch.setattr(bootstrap_mod, "build_fit_context", fake_build_fit_context)
    monkeypatch.setattr(bootstrap_mod, "run_fit", fake_run_fit)

    result = bootstrap_mod.bootstrap(
        model,
        data,
        n_bootstrap=n_bootstrap,
        weights=NativeDraws(n_obs=len(data.observables), base_seed=7),
        transport=SerialTransport(),
        master_backend="auto",
    )
    return SimpleNamespace(
        result=result,
        resolved_calls=resolved_calls,
        build_resolved=build_resolved,
        build_caches=build_caches,
        captured_weights=captured_weights,
    )


def _theta_model():
    parameters = Parameters({"theta": (-1.0, 1.0, 1)})
    return Model(
        _Oracle(),
        parameters,
        features=lambda agent_id, bundle: (np.zeros(1), 0.0),
        formulation=_Formulation,
    )


def _zeros_data(n_obs=2):
    return Data(
        observed_bundles=np.zeros((n_obs, 1), dtype=bool),
        shocks=np.zeros((n_obs, 1, 1)),
        observables=np.arange(n_obs),
    )


def test_serial_bootstrap_passes_warm_start_to_every_replication(
    monkeypatch,
) -> None:
    bootstrap_mod = importlib.import_module("combrum.bootstrap")
    captured: list[tuple[object, object]] = []

    def fake_build_fit_context(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured.append((kwargs["warm_start"], kwargs["warm_cuts"]))
        return SimpleNamespace(ctx=SimpleNamespace())

    def fake_run_fit(*args, **kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(
            result=FormulationResult(
                theta_hat=np.zeros(1), objective=0.0, n_active_cuts=0
            ),
            diagnostics=SimpleNamespace(converged=True, iterations=1),
        )

    monkeypatch.setattr(
        bootstrap_mod, "resolve_master_backend", lambda requested, **kw: "highs"
    )
    monkeypatch.setattr(bootstrap_mod, "build_fit_context", fake_build_fit_context)
    monkeypatch.setattr(bootstrap_mod, "run_fit", fake_run_fit)

    point = SimpleNamespace(theta_hat=np.zeros(1))
    rows = (
        CutRow(
            rep_id=0,
            agent_id=0,
            phi=np.zeros(1),
            epsilon=0.0,
            bundle_key=b"k",
        ),
    )
    bootstrap_mod.bootstrap(
        _theta_model(),
        _zeros_data(),
        n_bootstrap=3,
        weights=NativeDraws(n_obs=2, base_seed=7),
        warm_start=point,
        warm_cuts=rows,
    )
    assert captured == [(point, rows)] * 3


def test_serial_bootstrap_resolves_backend_once(monkeypatch) -> None:
    model = _theta_model()
    data = _zeros_data()
    run = _run_fake_bootstrap(monkeypatch, model, data)

    # rep 1 alone reports nonconvergence; rep b's theta lands in row b
    assert run.result.converged.tolist() == [True, False, True, True]
    assert run.result.thetas.tolist() == [[0.0], [1.0], [2.0], [3.0]]
    assert run.resolved_calls == ["auto"]
    assert run.build_resolved == ["highs", "highs", "highs", "highs"]

    # same base_seed as the run; weights_for(b) is rep b's RNG substream
    oracle = NativeDraws(n_obs=len(data.observables), base_seed=7)
    assert len(run.captured_weights) == 4
    for b, seen in enumerate(run.captured_weights):
        assert np.array_equal(seen, oracle.weights_for(b))
    # consecutive reps draw distinct weight rows
    for b in range(3):
        assert not np.array_equal(
            run.captured_weights[b], run.captured_weights[b + 1]
        )


def test_serial_bootstrap_rejects_non_integer_replication_count(monkeypatch) -> None:
    model = _theta_model()
    data = _zeros_data()

    for value, exc_type in (
        (True, TypeError),
        (np.bool_(True), TypeError),
        (3.5, TypeError),
        (0, ValueError),
    ):
        with pytest.raises(exc_type, match="n_bootstrap"):
            _run_fake_bootstrap(monkeypatch, model, data, n_bootstrap=value)

    run = _run_fake_bootstrap(monkeypatch, model, data, n_bootstrap=np.int64(2))
    assert run.result.thetas.shape == (2, 1)
    assert run.result.thetas.tolist() == [[0.0], [1.0]]
    assert run.result.converged.tolist() == [True, False]

    # n_bootstrap=1 is the documented minimum and must be accepted
    run_one = _run_fake_bootstrap(monkeypatch, model, data, n_bootstrap=1)
    assert run_one.result.thetas.shape == (1, 1)
    assert run_one.result.thetas.tolist() == [[0.0]]
    assert run_one.result.converged.tolist() == [True]
    assert run_one.build_resolved == ["highs"]


def test_serial_bootstrap_rejects_multirank_dense_transport() -> None:
    bootstrap_mod = importlib.import_module("combrum.bootstrap")
    parameters = Parameters({"theta": (-1.0, 1.0, 1)})
    model = Model(
        _Oracle(),
        parameters,
        features=lambda agent_id, bundle: (np.zeros(1), 0.0),
        formulation=_Formulation,
    )
    data = Data(
        observed_bundles=np.zeros((2, 1), dtype=bool),
        shocks=np.zeros((2, 1, 1)),
        observables=np.arange(2),
    )

    def run(transport):
        try:
            bootstrap_mod.bootstrap(
                model,
                data,
                n_bootstrap=1,
                weights=NativeDraws(n_obs=2, base_seed=7),
                transport=transport,
            )
        except ValueError as exc:
            return str(exc)
        return None

    # full message: entry point, release, and the distributed alternative
    expected = (
        f"bootstrap does not support non-serial transport in combRUM {combrum.__version__};"
        " use bootstrap_distributed for distributed runs"
    )
    messages = LocalCluster(2).run(run)
    assert messages == [expected, expected]


def test_serial_bootstrap_validates_data_shapes_before_cache_build() -> None:
    bootstrap_mod = importlib.import_module("combrum.bootstrap")
    model = _theta_model()
    weights = NativeDraws(n_obs=2, base_seed=7)

    bad_shocks = Data(
        observed_bundles=np.zeros((2, 1), dtype=bool),
        shocks=np.zeros(2),
        observables=np.arange(2),
    )
    with pytest.raises(ValueError, match=r"shocks must have shape \(N, S, \.\.\.\)"):
        bootstrap_mod.bootstrap(model, bad_shocks, n_bootstrap=1, weights=weights)

    bad_observed = Data(
        observed_bundles=np.zeros(2, dtype=bool),
        shocks=np.zeros((2, 1, 1)),
        observables=np.arange(2),
    )
    with pytest.raises(ValueError, match="observed_bundles must be 2-D"):
        bootstrap_mod.bootstrap(model, bad_observed, n_bootstrap=1, weights=weights)


def test_serial_bootstrap_reuses_observed_rows_for_materialized_path(
    monkeypatch,
) -> None:
    parameters = Parameters({"theta": (-1.0, 1.0, 1)})
    calls: list[tuple[int, float]] = []

    # phi depends on both the bundle and the agent id, so agents 2, 3 (which
    # reuse bundle rows 0, 1 via a % N) still get distinct phi rows
    def features(agent_id, bundle):  # type: ignore[no-untyped-def]
        calls.append((int(agent_id), float(np.asarray(bundle)[0])))
        value = float(np.asarray(bundle)[0]) + 10.0 * int(agent_id)
        return np.asarray([value], dtype=np.float64), 0.0

    model = Model(
        _Oracle(),
        parameters,
        features=features,
        formulation=_Formulation,
    )
    data = Data(
        observed_bundles=np.array([[1.0], [2.0]], dtype=np.float64),
        shocks=np.zeros((2, 2, 1), dtype=np.float64),
        observables=np.arange(2),
    )

    run = _run_fake_bootstrap(monkeypatch, model, data, n_bootstrap=3)

    assert calls == [(0, 1.0), (1, 2.0), (2, 1.0), (3, 2.0)]
    assert all(cache is run.build_caches[0] for cache in run.build_caches)
    cache = run.build_caches[0]
    assert cache is not None

    # shocks (2, 2, 1) => n_sim=2, n_agents = N * n_sim = 4; bundle of agent a
    # is observed_bundles[a % N] and phi row = bundle + 10*a
    N = 2
    expected_phi = np.array([[1.0], [12.0], [21.0], [32.0]])
    assert np.array_equal(cache.phi_local, expected_phi)
    # only rows 0, 1 are observed; the divisor is n_obs, not len(local_ids)
    expected_empirical = (expected_phi[0] + expected_phi[1]) / float(N)
    assert np.array_equal(cache.empirical_moment, expected_empirical)
    assert np.array_equal(cache.empirical_moment, np.array([6.5]))
    # differs from the all-rows / n_agents mean
    assert not np.allclose(
        cache.empirical_moment, expected_phi.sum(axis=0) / float(len(expected_phi))
    )


def test_serial_bootstrap_keeps_observed_objective_hooks_per_rep(
    monkeypatch,
) -> None:
    class _HookedFeatures:
        def observed_objective(
            self, K, N, theta_coef, observed_bundles, local_ids, transport
        ):
            return np.zeros(K, dtype=np.float64), np.zeros(K, dtype=np.float64)

        def __call__(self, agent_id, bundle):  # type: ignore[no-untyped-def]
            raise AssertionError("hooked observed objective should not be cached")

    parameters = Parameters({"theta": (-1.0, 1.0, 1)})
    model = Model(
        _Oracle(),
        parameters,
        features=_HookedFeatures(),
        formulation=_Formulation,
    )
    data = Data(
        observed_bundles=np.zeros((2, 1), dtype=np.float64),
        shocks=np.zeros((2, 1, 1), dtype=np.float64),
        observables=np.arange(2),
    )

    run = _run_fake_bootstrap(monkeypatch, model, data, n_bootstrap=3)

    assert run.build_caches == [None, None, None]


def test_serial_bootstrap_keeps_aggregate_features_per_rep(
    monkeypatch,
) -> None:
    class _AggregateFeatures(FeatureMap):
        def features_batch(
            self,
            ids,
            bundles,
            *,
            weights=None,
            K=None,
            aggregate=False,
        ):  # type: ignore[no-untyped-def]
            if aggregate:
                return np.zeros(1, dtype=np.float64), 0.0
            return np.asarray(bundles, dtype=np.float64), np.zeros(len(ids))

    parameters = Parameters({"theta": (-1.0, 1.0, 1)})
    model = Model(
        _Oracle(),
        parameters,
        features=_AggregateFeatures(),
        formulation=_Formulation,
    )
    data = Data(
        observed_bundles=np.zeros((2, 1), dtype=np.float64),
        shocks=np.zeros((2, 1, 1), dtype=np.float64),
        observables=np.arange(2),
    )

    run = _run_fake_bootstrap(monkeypatch, model, data, n_bootstrap=3)

    # this map satisfies every clause of the cache-skip gate (serial transport,
    # OPTIMIZED resolution, aggregate support), so no shared cache is built and
    # each rep recomputes; the _PlainBatch test below covers the converse
    from combrum.interface_resolution import (
        resolve_features,
        supports_feature_batch_aggregate,
    )

    resolution = resolve_features(model.features)
    assert resolution.runs_optimized
    assert supports_feature_batch_aggregate(resolution.active)
    assert run.build_caches == [None, None, None]


def test_serial_bootstrap_caches_optimized_batch_without_aggregate(
    monkeypatch,
) -> None:
    # OPTIMIZED FeatureMap whose features_batch takes only (ids, bundles) -- no
    # weights/aggregate kwargs, so supports_feature_batch_aggregate is False and
    # the shared cache must still be built
    class _PlainBatch(FeatureMap):
        def features_batch(self, ids, bundles):  # type: ignore[no-untyped-def]
            ids = np.asarray(ids)
            rows = np.asarray(bundles, dtype=np.float64) + 10.0 * ids[:, None]
            return rows, np.zeros(len(ids))

    parameters = Parameters({"theta": (-1.0, 1.0, 1)})
    model = Model(
        _Oracle(),
        parameters,
        features=_PlainBatch(),
        formulation=_Formulation,
    )
    data = Data(
        # shocks (2, 2, 1) => n_sim=2, n_agents=4: agents 2, 3 fall outside the
        # observed rows
        observed_bundles=np.array([[3.0], [5.0]], dtype=np.float64),
        shocks=np.zeros((2, 2, 1), dtype=np.float64),
        observables=np.arange(2),
    )

    run = _run_fake_bootstrap(monkeypatch, model, data, n_bootstrap=3)

    cache = run.build_caches[0]
    assert cache is not None
    assert all(c is cache for c in run.build_caches)
    # n_agents=4; bundle of agent a is observed_bundles[a % N] and
    # phi row = bundle + 10*a
    expected_phi = np.array([[3.0], [15.0], [23.0], [35.0]])
    assert np.array_equal(cache.phi_local, expected_phi)
    # only the observed rows (local_ids < N) enter the moment, divided by n_obs=2
    N = 2
    expected_empirical = (expected_phi[0] + expected_phi[1]) / float(N)
    assert np.array_equal(cache.empirical_moment, expected_empirical)
    assert np.array_equal(cache.empirical_moment, np.array([9.0]))
    assert not np.allclose(
        cache.empirical_moment, expected_phi.sum(axis=0) / float(len(expected_phi))
    )
