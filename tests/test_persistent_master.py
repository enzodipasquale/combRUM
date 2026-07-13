"""PersistentMasterFit and the run_fit suppress_close flag.

A ``PersistentMasterFit`` holds one master across an outer ψ search: it
guards reuse validity fail-closed, RHS-rewrites the carried cuts, and
warm-solves. Covered here:

* ``run_fit(suppress_close=True)`` retains the master; the default closes it.
* construction state and the ``PersistentFitResult`` surface.
* cold ``fit`` + ``reevaluate`` over a valid shocks-only ψ, end to end.
* the reuse guards: observed-bundle φ/c_theta drift → G1 (b1),
  geometry-signature drift → G2 (b2), weight drift → G1 (b3), and θ-box
  drift → G3 when the geometry signature is omitted.
* NSlack-only: any other formulation → TypeError at first use.

Most tests pass the formulation explicitly rather than relying on the
driver's lazy NSlack default.
"""

from __future__ import annotations

import dataclasses
import importlib
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from _family_oracles import FamilyProblem, toy_problem
from combrum.engine import (
    LoopConfig,
    PersistentFitResult,
    PersistentMasterFit,
    build_fit_context,
    run_fit,
)
from combrum.formulations import NSlack, OneSlack
from _support.constants import MAX_ITERATIONS, THETA_BOUND, TOLERANCE
from _support.families import FAMILY_DIR, load_family
from combrum.master import MasterBackend
from combrum.masters import gurobi as gurobi_backend
from combrum.masters import highs as highs_backend
from combrum.parameters import Parameters
from combrum.transport import LocalCluster, SerialTransport, Transport, TransportError
from combrum.transport.base import CutRow

HIGHS_AVAILABLE = highs_backend.available()
needs_highs = pytest.mark.skipif(
    not HIGHS_AVAILABLE, reason="highspy missing or broken"
)

GUROBI_AVAILABLE = gurobi_backend.available()
needs_gurobi = pytest.mark.skipif(
    not GUROBI_AVAILABLE, reason="gurobipy missing or no environment starts"
)


# --- shared toy fixture wiring ------------------------------------------------


def _toy_inputs():
    """The toy fixture's (params, observables, observed, shocks, problem).

    ``observables`` is a stand-in index list — the fit context reads only
    ``len(observables)`` — not the family's ``observables`` feature array
    (which the geometry-signature helpers load separately).
    """
    arrays = load_family("toy", FAMILY_DIR)
    problem: FamilyProblem = toy_problem(arrays)
    observed = np.asarray(arrays["observed"])
    n_obs, n_items = observed.shape
    params = Parameters({"theta": (-THETA_BOUND, THETA_BOUND, n_items)})
    observables = list(range(n_obs))
    shocks = np.asarray(arrays["shocks"], dtype=np.float64)
    return params, observables, observed, shocks, problem


def _config() -> LoopConfig:
    return LoopConfig(max_iterations=MAX_ITERATIONS)


def _psi_problem(psi: float) -> FamilyProblem:
    """A ψ-coherent toy problem: oracle + features over shocks_ψ = ψ·shocks_ψ0.

    The caller owns ψ→(oracle_ψ, features_ψ, shocks_ψ); this builds all three
    from one ψ. The toy features are φ = b·r (ψ-invariant — r is the
    ψ-invariant geometry) and eps = b·ν, so scaling the shocks ν by ψ scales
    every cut's ε by exactly ψ (ε@ψ = ψ·ε@ψ0) and the priced demand by the
    same ν. c_theta / the θ-box / weights stay ψ-invariant.
    """
    arrays = dict(load_family("toy", FAMILY_DIR))
    arrays["shocks"] = psi * np.asarray(arrays["shocks"], dtype=np.float64)
    return toy_problem(arrays)


# A fixed-bytes geometry fingerprint over the toy feature arrays: the toy
# features are phi = b * r_a, so the ψ-invariant geometry IS the observables r.
# The canonical shocks-only ψ never touches r, so this signature is ψ-stable.
def _geometry_signature_factory(observables: np.ndarray):
    fixed = np.asarray(observables, dtype=np.float64)

    def geometry_signature(_psi: object) -> bytes:
        return fixed.tobytes()

    return geometry_signature


# The canonical bundle-free closed-form RHS rule: ε@ψ = ψ · ε@ψ0.
def _rhs_transform(row: CutRow, psi: float) -> float:
    return float(psi) * float(row.epsilon)


# --- a spy master: wraps a real backend, records close() ----------------------


class _CloseSpyMaster(MasterBackend):
    """Wraps a real MasterBackend and counts close() calls.

    Subclasses the ABC because NSlack.setup ``isinstance``-checks the master.
    Every contract method forwards verbatim to the wrapped backend.
    """

    def __init__(self, inner: MasterBackend) -> None:
        self._inner = inner
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        self._inner.close()

    def add_cuts(self, rows):
        return self._inner.add_cuts(rows)

    def solve(self) -> None:
        self._inner.solve()

    def theta(self):
        return self._inner.theta()

    def objective(self) -> float:
        return self._inner.objective()

    def u_values(self):
        return self._inner.u_values()

    def dual_values(self):
        return self._inner.dual_values()

    def set_penalty(self, ref, weight) -> None:
        self._inner.set_penalty(ref, weight)

    def extract_cuts(self):
        return self._inner.extract_cuts()

    @property
    def n_active_cuts(self) -> int:
        # HighsMaster tracks its installed set independently of
        # extract_cuts(); forward to the property, not a derived count.
        return self._inner.n_active_cuts

    def reinstall(self, rows) -> None:
        self._inner.reinstall(rows)

    def set_rhs(self, updates) -> None:
        self._inner.set_rhs(updates)

    def bound_duals(self):
        return self._inner.bound_duals()


def _built_with_spy(formulation):
    """A toy BuiltContext whose master is wrapped in a _CloseSpyMaster."""
    params, observables, observed, shocks, _problem = _toy_inputs()
    built = build_fit_context(
        params,
        observables=observables,
        observed_bundles=observed,
        shocks=shocks,
        formulation=formulation,
        features=_problem_features(),
        observed_features=_problem_observed_features(),
        transport=SerialTransport(),
        master_backend="highs",
        tolerance=TOLERANCE,
    )
    spy = _CloseSpyMaster(built.ctx.master_backend)
    ctx = dataclasses.replace(built.ctx, master_backend=spy)
    return ctx, spy


def _spy_on_live_master(driver: PersistentMasterFit) -> _CloseSpyMaster:
    """Swap the driver's live master for a _CloseSpyMaster.

    Any later ``master.close()`` — guard teardown or context-manager exit —
    is counted; a teardown that only nils ``self._master`` without releasing
    the backend leaves ``close_calls == 0``.
    """
    spy = _CloseSpyMaster(driver._master)
    driver._master = spy
    return spy


def _problem_features():
    return toy_problem(load_family("toy", FAMILY_DIR)).features


def _problem_observed_features():
    return toy_problem(load_family("toy", FAMILY_DIR)).observed_features


def _hand_empirical_moment() -> np.ndarray:
    """empirical_moment recomputed from the fixture arrays.

    empirical_moment is the row-observed feature mean
    (1/N)·Σ_i observed_features(i, observed_i); the toy family has
    observed_features(i, b) = b·r_i, so the mean comes straight off the
    fixture arrays with no combrum accessor. φ is ψ-invariant, so this is the
    value at every ψ.
    """
    arrays = load_family("toy", FAMILY_DIR)
    observed = np.asarray(arrays["observed"], dtype=np.float64)
    r = np.asarray(arrays["observables"], dtype=np.float64)
    n_obs = observed.shape[0]
    return (observed * r).sum(axis=0) / float(n_obs)


# --- suppress_close: the lifecycle flag --------------------------------------


@needs_highs
def test_run_fit_default_closes_master() -> None:
    """Default run_fit (suppress_close unset) closes the master exactly once."""
    _params, _obs, _observed, _shocks, problem = _toy_inputs()
    ctx, spy = _built_with_spy(NSlack(problem.features))
    run_fit(ctx, problem.oracle, NSlack(problem.features), _config())
    assert spy.close_calls == 1


def _cold_rebuild_objective(formulation_factory) -> float:
    """The toy row-generation objective from a fresh master built from scratch.

    A distinct build/solve path from the retained master, so the retained
    master's re-solve is checked against it rather than against its own
    carried value.
    """
    params, observables, observed, shocks, _problem = _toy_inputs()
    built = build_fit_context(
        params,
        observables=observables,
        observed_bundles=observed,
        shocks=shocks,
        formulation=formulation_factory(),
        features=_problem_features(),
        observed_features=_problem_observed_features(),
        transport=SerialTransport(),
        master_backend="highs",
        tolerance=TOLERANCE,
    )
    outcome = run_fit(
        built.ctx, _toy_inputs()[4].oracle, formulation_factory(), _config()
    )
    return float(outcome.result.objective)


@needs_highs
def test_run_fit_suppress_close_retains_master() -> None:
    """suppress_close=True skips close() — the retained master stays live for reuse."""
    _params, _obs, _observed, _shocks, problem = _toy_inputs()
    ctx, spy = _built_with_spy(NSlack(problem.features))
    run_fit(
        ctx, problem.oracle, NSlack(problem.features), _config(),
        suppress_close=True,
    )
    assert spy.close_calls == 0

    # Retained must mean reusable: a fresh re-solve reproduces the objective
    # of a cold rebuild, so a torn-down or degraded master fails here.
    ref_objective = _cold_rebuild_objective(lambda: NSlack(problem.features))
    spy.solve()
    assert abs(spy.objective() - ref_objective) <= 1e-9

    # The carried cut set is intact: extract_cuts() returns every row the
    # backend's own installed-set count reports.
    installed = spy.n_active_cuts
    assert installed > 0
    assert len(spy.extract_cuts()) == installed
    spy.close()


# --- PersistentMasterFit: the construction surface ----------------------------


def test_persistent_master_fit_construction_surface() -> None:
    """PersistentMasterFit constructs to a clean pre-fit state and
    PersistentFitResult carries the documented field set."""
    params, observables, observed, _shocks, _problem = _toy_inputs()
    driver = PersistentMasterFit(
        params,
        observables=observables,
        observed_bundles=observed,
        transport=SerialTransport(),
        config=_config(),
        rhs_transform=_rhs_transform,
        master_backend="auto",
        tolerance=TOLERANCE,
    )
    # Pre-fit: no master built, backend unresolved, no ψ0 signature stashed.
    assert driver._master is None
    assert driver._resolved_master_backend is None
    assert driver._c_theta0 is None
    assert driver._agent_weights0 is None
    assert driver._theta_bounds0 is None
    assert driver._geometry0 is None

    # The published result surface the outer search drives on.
    assert [f.name for f in dataclasses.fields(PersistentFitResult)] == [
        "theta_hat",
        "objective",
        "empirical_moment",
        "dual",
        "converged",
        "iterations",
        "n_active_cuts",
    ]


def test_persistent_fit_resolves_backend_once(monkeypatch) -> None:
    persistent_mod = importlib.import_module("combrum.engine.persistent")
    params, observables, observed, shocks0, problem = _toy_inputs()
    resolved_calls: list[str] = []
    build_resolved: list[str] = []

    class FakeMaster:
        def extract_cuts(self):
            return ()

        def set_rhs(self, updates) -> None:  # type: ignore[no-untyped-def]
            assert updates == {}

        def close(self) -> None:
            pass

    fake_master = FakeMaster()

    def fake_resolve(requested, **kwargs):  # type: ignore[no-untyped-def]
        resolved_calls.append(requested)
        return "highs"

    def fake_build_fit_context(*args, **kwargs):  # type: ignore[no-untyped-def]
        build_resolved.append(kwargs["resolved_master_backend"])
        master = fake_master if kwargs["master"] is None else kwargs["master"]
        return SimpleNamespace(
            ctx=SimpleNamespace(
                master_backend=master,
                owner_rank=0,
                agent_weights=np.ones(observed.shape[0], dtype=np.float64),
                theta_bounds=params.bounds(),
            ),
            c_theta=np.zeros(params.K, dtype=np.float64),
            empirical_moment=np.zeros(params.K, dtype=np.float64),
        )

    def fake_run_fit(*args, **kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(
            result=SimpleNamespace(
                theta_hat=np.zeros(params.K, dtype=np.float64),
                objective=0.0,
                dual=None,
                n_active_cuts=0,
            ),
            diagnostics=SimpleNamespace(converged=True, iterations=1),
        )

    monkeypatch.setattr(persistent_mod, "resolve_master_backend", fake_resolve)
    monkeypatch.setattr(persistent_mod, "build_fit_context", fake_build_fit_context)
    monkeypatch.setattr(persistent_mod, "run_fit", fake_run_fit)

    driver = PersistentMasterFit(
        params,
        observables=observables,
        observed_bundles=observed,
        transport=SerialTransport(),
        config=_config(),
        rhs_transform=_rhs_transform,
        master_backend="auto",
        tolerance=TOLERANCE,
    )
    try:
        driver.fit(
            1.0,
            oracle=problem.oracle,
            formulation=NSlack(problem.features),
            features=problem.features,
            observed_features=problem.observed_features,
            shocks=shocks0,
        )
        # Second cold fit: resolve_master_backend must not run again — the
        # cached ψ0-resolved backend is reused.
        driver.fit(
            2.0,
            oracle=problem.oracle,
            formulation=NSlack(problem.features),
            features=problem.features,
            observed_features=problem.observed_features,
            shocks=2.0 * shocks0,
        )
        driver.reevaluate(
            1.5,
            oracle=problem.oracle,
            formulation=NSlack(problem.features),
            features=problem.features,
            observed_features=problem.observed_features,
            shocks=1.5 * shocks0,
        )
    finally:
        driver.close()

    # One resolve across two fits; every build (both fits + the reevaluate)
    # threads the cached backend through.
    assert resolved_calls == ["auto"]
    assert build_resolved == ["highs", "highs", "highs"]


def _config_with_penalty(qp_weight: float, qp_iterations: int) -> LoopConfig:
    """A LoopConfig with a (qp_weight, qp_iterations) pair its validator may reject.

    ``LoopConfig`` requires qp_iterations>=1 whenever qp_weight>0, but the
    require_quadratic guard reads the two attributes at runtime, and
    separating its ``and`` clauses needs the qp_weight>0 / qp_iterations=0
    corner. So build a valid frozen config and override the two fields
    directly; nothing else in fit() reads them before resolve.
    """
    cfg = LoopConfig(max_iterations=1, qp_weight=1.0, qp_iterations=1)
    object.__setattr__(cfg, "qp_weight", qp_weight)
    object.__setattr__(cfg, "qp_iterations", qp_iterations)
    return cfg


def test_persistent_fit_positive_penalty_requires_quadratic_backend(
    monkeypatch,
) -> None:
    persistent_mod = importlib.import_module("combrum.engine.persistent")
    params, observables, observed, shocks0, problem = _toy_inputs()

    def fail_build_fit_context(*args, **build_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("build_fit_context should not run")

    monkeypatch.setattr(
        persistent_mod, "build_fit_context", fail_build_fit_context
    )

    # Three configs vary the two clauses of
    #   require_quadratic = (qp_weight > 0.0 and qp_iterations > 0)
    # independently. The captured vector must match this truth table exactly:
    # neither clause may be dropped and the `and` may not become an `or`.
    cases = [
        (_config_with_penalty(1.0, 1), True),
        (_config_with_penalty(1.0, 0), False),
        (_config_with_penalty(0.0, 1), False),
    ]
    expected = [want for _cfg, want in cases]

    captured: list[bool] = []

    def fake_resolve(requested, **kwargs):  # type: ignore[no-untyped-def]
        captured.append(bool(kwargs["require_quadratic"]))
        # Raise only on the quadratic branch so both outcomes are observable.
        if kwargs["require_quadratic"]:
            raise RuntimeError("quadratic backend required")
        return "highs"

    monkeypatch.setattr(persistent_mod, "resolve_master_backend", fake_resolve)

    for config, want_quadratic in cases:
        driver = PersistentMasterFit(
            params,
            observables=observables,
            observed_bundles=observed,
            transport=SerialTransport(),
            config=config,
            rhs_transform=_rhs_transform,
            master_backend="highs",
            tolerance=TOLERANCE,
        )
        if want_quadratic:
            with pytest.raises(RuntimeError, match="quadratic backend"):
                driver.fit(
                    1.0,
                    oracle=problem.oracle,
                    formulation=NSlack(problem.features),
                    features=problem.features,
                    observed_features=problem.observed_features,
                    shocks=shocks0,
                )
        else:
            # build_fit_context is stubbed to fail, so this AssertionError
            # proves fit() got past resolve without demanding a quadratic
            # backend.
            with pytest.raises(AssertionError, match="build_fit_context"):
                driver.fit(
                    1.0,
                    oracle=problem.oracle,
                    formulation=NSlack(problem.features),
                    features=problem.features,
                    observed_features=problem.observed_features,
                    shocks=shocks0,
                )

    assert captured == expected


def test_persistent_reevaluate_before_fit_raises_on_all_ranks_under_multirank() -> None:
    params, observables, observed, shocks0, problem = _toy_inputs()

    # LocalCluster.run re-raises only the first rank's error, which cannot
    # distinguish "all ranks raised" from "rank 0 raised, a worker returned or
    # stranded" — so record and assert each rank's own outcome.
    outcomes: dict[int, object] = {}
    lock = threading.Lock()

    def per_rank(transport: Transport) -> None:
        driver = PersistentMasterFit(
            params,
            observables=observables,
            observed_bundles=observed,
            transport=transport,
            config=_config(),
            rhs_transform=_rhs_transform,
            master_backend="highs",
            tolerance=TOLERANCE,
        )
        try:
            driver.reevaluate(
                1.5,
                oracle=problem.oracle,
                formulation=NSlack(problem.features),
                features=problem.features,
                observed_features=problem.observed_features,
                shocks=1.5 * shocks0,
            )
        except BaseException as exc:  # relay each rank's own outcome
            with lock:
                outcomes[transport.rank] = exc
            raise
        else:
            with lock:
                outcomes[transport.rank] = "returned-normally"

    with pytest.raises(TransportError, match="no live master"):
        LocalCluster(2).run(per_rank)

    # Both ranks raised the same agreed TransportError (origin rank 0, same
    # message): collective() propagates the failure to the worker rank rather
    # than leaving it to return or strand at a half-run collective.
    assert set(outcomes) == {0, 1}
    for rank, observed_outcome in outcomes.items():
        assert isinstance(observed_outcome, TransportError), (rank, observed_outcome)
        assert observed_outcome.rank == 0, (rank, observed_outcome.rank)
        assert "no live master" in observed_outcome.message, (
            rank,
            observed_outcome.message,
        )


def test_persistent_set_rhs_failure_raises_on_all_ranks_under_multirank(
    monkeypatch,
) -> None:
    persistent_mod = importlib.import_module("combrum.engine.persistent")
    params, observables, observed, shocks0, problem = _toy_inputs()

    class FailingMaster:
        def extract_cuts(self):
            return ()

        def set_rhs(self, updates) -> None:  # type: ignore[no-untyped-def]
            raise RuntimeError("bad rhs")

        def close(self) -> None:
            pass

    def fake_resolve(requested, **kwargs):  # type: ignore[no-untyped-def]
        return "highs"

    def fake_build_fit_context(*args, **kwargs):  # type: ignore[no-untyped-def]
        transport = kwargs["transport"]
        master = None
        if transport.rank == 0:
            master = kwargs["master"] if kwargs["master"] is not None else FailingMaster()
        return SimpleNamespace(
            ctx=SimpleNamespace(
                master_backend=master,
                owner_rank=0,
                agent_weights=np.ones(observed.shape[0], dtype=np.float64),
                theta_bounds=params.bounds(),
            ),
            c_theta=np.zeros(params.K, dtype=np.float64),
            empirical_moment=np.zeros(params.K, dtype=np.float64),
        )

    def fake_run_fit(*args, **kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(
            result=SimpleNamespace(
                theta_hat=np.zeros(params.K, dtype=np.float64),
                objective=0.0,
                dual=None,
                n_active_cuts=0,
            ),
            diagnostics=SimpleNamespace(converged=True, iterations=1),
        )

    monkeypatch.setattr(persistent_mod, "resolve_master_backend", fake_resolve)
    monkeypatch.setattr(persistent_mod, "build_fit_context", fake_build_fit_context)
    monkeypatch.setattr(persistent_mod, "run_fit", fake_run_fit)

    # As above: assert on each rank's own outcome, not just the error
    # LocalCluster.run re-raises for the owner.
    outcomes: dict[int, object] = {}
    lock = threading.Lock()

    def per_rank(transport: Transport) -> None:
        driver = PersistentMasterFit(
            params,
            observables=observables,
            observed_bundles=observed,
            transport=transport,
            config=_config(),
            rhs_transform=_rhs_transform,
            master_backend="auto",
            tolerance=TOLERANCE,
        )
        try:
            driver.fit(
                1.0,
                oracle=problem.oracle,
                formulation=NSlack(problem.features),
                features=problem.features,
                observed_features=problem.observed_features,
                shocks=shocks0,
            )
            driver.reevaluate(
                1.5,
                oracle=problem.oracle,
                formulation=NSlack(problem.features),
                features=problem.features,
                observed_features=problem.observed_features,
                shocks=1.5 * shocks0,
            )
        except BaseException as exc:  # relay each rank's own outcome
            with lock:
                outcomes[transport.rank] = exc
            raise
        else:
            with lock:
                outcomes[transport.rank] = "returned-normally"

    with pytest.raises(TransportError, match="bad rhs"):
        LocalCluster(2).run(per_rank)

    # The collective() around the RHS rewrite turns the owner's set_rhs
    # failure into an agreed verdict every rank raises.
    assert set(outcomes) == {0, 1}
    for rank, observed_outcome in outcomes.items():
        assert isinstance(observed_outcome, TransportError), (rank, observed_outcome)
        assert observed_outcome.rank == 0, (rank, observed_outcome.rank)
        assert "bad rhs" in observed_outcome.message, (rank, observed_outcome.message)


def _make_driver(
    observables,
    observed,
    params,
    *,
    check_geometry: bool = True,
    transport: Transport | None = None,
) -> PersistentMasterFit:
    kwargs = {
        "observables": observables,
        "observed_bundles": observed,
        "transport": SerialTransport() if transport is None else transport,
        "config": _config(),
        "rhs_transform": _rhs_transform,
        "master_backend": "highs",
        "tolerance": TOLERANCE,
    }
    if check_geometry:
        kwargs["geometry_signature"] = _geometry_signature_factory(
            np.asarray(load_family("toy", FAMILY_DIR)["observables"])
        )
    return PersistentMasterFit(params, **kwargs)


# --- cold fit + reevaluate over a valid shocks-only ψ -------------------------


@needs_highs
def test_cold_fit_then_reevaluate_runs_end_to_end() -> None:
    """A valid shocks-only ψ-search runs cold fit + reevaluate end-to-end.

    ψ scales the shocks: shocks_ψ = ψ·shocks_ψ0 and ε@ψ = ψ·ε@ψ0, while φ /
    c_theta / agent_weights / the θ-box are ψ-invariant — exactly the
    supported reuse class. The driver holds one master across ψ0 and ψ1,
    RHS-rewrites the carried cuts, warm-solves, and publishes a result on the
    estimate criterion scale.
    """
    params, observables, observed, shocks0, problem = _toy_inputs()

    # Cold-rebuild reference for theta_hat and iterations: the toy LP is
    # deterministic, so a fresh build converges to the byte-identical theta_hat.
    cold_rebuild = build_fit_context(
        params,
        observables=observables,
        observed_bundles=observed,
        shocks=shocks0,
        formulation=NSlack(problem.features),
        features=problem.features,
        observed_features=problem.observed_features,
        transport=SerialTransport(),
        master_backend="highs",
        tolerance=TOLERANCE,
    )
    ref_outcome = run_fit(
        cold_rebuild.ctx, problem.oracle, NSlack(problem.features), _config()
    )
    ref_theta_hat = np.array(ref_outcome.result.theta_hat, dtype=np.float64)
    ref_iterations = int(ref_outcome.diagnostics.iterations)

    hand_moment = _hand_empirical_moment()
    driver = _make_driver(observables, observed, params)
    try:
        cold = driver.fit(
            1.0,
            oracle=problem.oracle,
            features=problem.features,
            observed_features=problem.observed_features,
            shocks=shocks0,
        )
        assert isinstance(cold, PersistentFitResult)
        assert cold.converged
        assert cold.dual is not None
        n_cold_cuts = cold.n_active_cuts

        # theta_hat is the converged master estimate: the full (K,) vector
        # matches the cold rebuild byte for byte.
        np.testing.assert_array_equal(cold.theta_hat, ref_theta_hat)

        # empirical_moment is the observed feature mean, not theta_hat: it
        # matches the fixture-derived value and differs from theta_hat.
        np.testing.assert_allclose(cold.empirical_moment, hand_moment, atol=1e-9)
        assert not np.allclose(cold.theta_hat, cold.empirical_moment)

        # iterations comes from the diagnostics, matching a cold rebuild.
        assert cold.iterations == ref_iterations

        # A second valid ψ: a ψ-coherent oracle/features over shocks_ψ = ψ·shocks0,
        # and the cut ε rewritten by the same ψ via rhs_transform.
        psi = 1.5
        psi_problem = _psi_problem(psi)
        re = driver.reevaluate(
            psi,
            oracle=psi_problem.oracle,
            features=psi_problem.features,
            observed_features=psi_problem.observed_features,
            shocks=psi * shocks0,
        )
        assert isinstance(re, PersistentFitResult)
        assert re.converged
        assert re.dual is not None
        # Warm reuse only grows the installed set.
        assert re.n_active_cuts >= n_cold_cuts
        # φ is ψ-invariant, so the warm eval publishes the same moment.
        np.testing.assert_allclose(re.empirical_moment, hand_moment, atol=1e-9)
        # Warm reuse does strictly less row-generation than the cold fit.
        assert re.iterations < cold.iterations
    finally:
        driver.close()


@needs_highs
def test_cold_fit_then_reevaluate_runs_without_geometry_signature() -> None:
    """Omitting geometry_signature skips only G2; valid RHS-only reuse still runs."""
    params, observables, observed, shocks0, problem = _toy_inputs()
    driver = _make_driver(observables, observed, params, check_geometry=False)
    try:
        cold = driver.fit(
            1.0,
            oracle=problem.oracle,
            formulation=NSlack(problem.features),
            features=problem.features,
            observed_features=problem.observed_features,
            shocks=shocks0,
        )
        assert cold.converged
        assert cold.dual is not None

        psi = 1.5
        psi_problem = _psi_problem(psi)
        re = driver.reevaluate(
            psi,
            oracle=psi_problem.oracle,
            formulation=NSlack(psi_problem.features),
            features=psi_problem.features,
            observed_features=psi_problem.observed_features,
            shocks=psi * shocks0,
        )
        assert re.converged
        assert re.dual is not None
        assert re.n_active_cuts >= cold.n_active_cuts
    finally:
        driver.close()


@needs_highs
def test_cold_fit_then_reevaluate_runs_under_multirank_transport() -> None:
    """Only the owner rank holds the master; worker ranks still warm-reevaluate."""

    def run(transport: Transport) -> tuple[bool, bool, bool, int, int]:
        params, observables, observed, shocks0, problem = _toy_inputs()
        driver = _make_driver(
            observables,
            observed,
            params,
            check_geometry=False,
            transport=transport,
        )
        try:
            cold = driver.fit(
                1.0,
                oracle=problem.oracle,
                formulation=NSlack(problem.features),
                features=problem.features,
                observed_features=problem.observed_features,
                shocks=shocks0,
            )
            psi = 1.5
            psi_problem = _psi_problem(psi)
            warm = driver.reevaluate(
                psi,
                oracle=psi_problem.oracle,
                formulation=NSlack(psi_problem.features),
                features=psi_problem.features,
                observed_features=psi_problem.observed_features,
                shocks=psi * shocks0,
            )
            return (
                bool(cold.converged),
                bool(warm.converged),
                bool((warm.dual is not None) == (transport.rank == 0)),
                int(cold.n_active_cuts),
                int(warm.n_active_cuts),
            )
        finally:
            driver.close()

    results = LocalCluster(2).run(run)
    for cold_ok, warm_ok, dual_on_root, cold_cuts, warm_cuts in results:
        assert cold_ok
        assert warm_ok
        assert dual_on_root
        assert warm_cuts >= cold_cuts


@needs_highs
def test_persistent_criterion_matches_cold_rebuild_within_band() -> None:
    """The warm ψ-eval's Q matches a cold rebuild-per-eval within PARITY_BAND.

    The RHS-rewritten ψ0 cuts stay valid lower bounds at ψ, so warm row-gen
    converges to the same objective as a cold rebuild — Q banded, though the
    vertex/active set may differ. Single-ψ case; the ψ-panel version lives in
    test_persistent_master_rhs.py.
    """
    params, observables, observed, shocks0, problem = _toy_inputs()
    psi = 1.5
    psi_problem = _psi_problem(psi)
    driver = _make_driver(observables, observed, params)
    try:
        driver.fit(
            1.0,
            oracle=problem.oracle,
            formulation=NSlack(problem.features),
            features=problem.features,
            observed_features=problem.observed_features,
            shocks=shocks0,
        )
        warm = driver.reevaluate(
            psi,
            oracle=psi_problem.oracle,
            formulation=NSlack(psi_problem.features),
            features=psi_problem.features,
            observed_features=psi_problem.observed_features,
            shocks=psi * shocks0,
        )
    finally:
        driver.close()

    # Cold rebuild at the same ψ: same oracle/features/shocks, fresh master.
    built = build_fit_context(
        params,
        observables=observables,
        observed_bundles=observed,
        shocks=psi * shocks0,
        formulation=NSlack(psi_problem.features),
        features=psi_problem.features,
        observed_features=psi_problem.observed_features,
        transport=SerialTransport(),
        master_backend="highs",
        tolerance=TOLERANCE,
    )
    outcome = run_fit(
        built.ctx, psi_problem.oracle, NSlack(psi_problem.features), _config()
    )
    cold_objective = float(outcome.result.objective)
    assert abs(warm.objective - cold_objective) <= 1e-9


# --- positive quadratic-penalty persistent path (fixed-then-off -> pure LP) --


@needs_gurobi
@pytest.mark.slow
def test_persistent_penalty_decay_fit_then_reevaluate_runs_end_to_end() -> None:
    """A qp_weight>0 / qp_iterations>=1 config runs the persistent path on a real QP solve.

    ``require_quadratic`` resolves to True (qp_weight>0 and qp_iterations>0 in
    ``PersistentMasterFit.fit``'s ``resolve_master_backend`` call), so the
    driver builds a gurobi master. The proximal weight holds at ``qp_weight``
    for ``qp_iterations`` iterations and then drops to exactly zero, so the
    terminating solve is a pure LP and a dual is published. Both the cold fit
    and a valid shocks-only reevaluate run end-to-end over the carried master.
    """
    params, observables, observed, shocks0, problem = _toy_inputs()
    penalty_config = LoopConfig(
        max_iterations=MAX_ITERATIONS,
        qp_weight=10.0,
        qp_iterations=3,
        penalty_ref="static",
    )
    driver = PersistentMasterFit(
        params,
        observables=observables,
        observed_bundles=observed,
        transport=SerialTransport(),
        config=penalty_config,
        rhs_transform=_rhs_transform,
        geometry_signature=_geometry_signature_factory(
            np.asarray(load_family("toy", FAMILY_DIR)["observables"])
        ),
        master_backend="gurobi",
        tolerance=TOLERANCE,
    )
    try:
        cold = driver.fit(
            1.0,
            oracle=problem.oracle,
            formulation=NSlack(problem.features),
            features=problem.features,
            observed_features=problem.observed_features,
            shocks=shocks0,
        )
        assert isinstance(cold, PersistentFitResult)
        assert cold.converged
        # A published dual proves the terminating solve was a pure LP (the
        # penalty weight dropped to 0).
        assert cold.dual is not None
        n_cold_cuts = cold.n_active_cuts

        psi = 1.5
        psi_problem = _psi_problem(psi)
        re = driver.reevaluate(
            psi,
            oracle=psi_problem.oracle,
            formulation=NSlack(psi_problem.features),
            features=psi_problem.features,
            observed_features=psi_problem.observed_features,
            shocks=psi * shocks0,
        )
        assert isinstance(re, PersistentFitResult)
        assert re.converged
        assert re.dual is not None
        assert re.n_active_cuts >= n_cold_cuts
    finally:
        driver.close()


# --- perturbation: fail-closed guards (b1/b2/b3) -----------------------------------


@needs_highs
def test_perturbation_b1_observed_phi_drift_hard_errors_at_g1() -> None:
    """(b1) a ψ whose features change observed-bundle φ/c_theta → G1 hard-error.

    reevaluate is handed a different features map (φ perturbed), so c_theta@ψ
    no longer matches ψ0's — the objective is not ψ-invariant, out of the
    RHS-only class. The guard hard-errors at G1 and closes the master.

    Single case: the φ drift trips G1's c_theta sub-guard, which
    short-circuits before G2 is consulted, so parametrizing over
    check_geometry would run identical code twice. The G2 checks live in the
    b2 tests.
    """
    params, observables, observed, shocks0, problem = _toy_inputs()
    driver = _make_driver(observables, observed, params)
    driver.fit(
        1.0,
        oracle=problem.oracle,
        formulation=NSlack(problem.features),
        features=problem.features,
        observed_features=problem.observed_features,
        shocks=shocks0,
    )
    spy = _spy_on_live_master(driver)

    def perturbed_features(agent_id: int, bundle: np.ndarray):
        phi, eps = problem.features(agent_id, bundle)
        phi = np.asarray(phi, dtype=np.float64).copy()
        phi[0] += 1.0  # a gross φ drift → c_theta@ψ differs from ψ0
        return phi, eps

    def perturbed_observed_features(agent_id: int, bundle: np.ndarray):
        phi = np.asarray(
            problem.observed_features(agent_id, bundle), dtype=np.float64
        ).copy()
        phi[0] += 1.0
        return phi

    with pytest.raises(ValueError, match=r"G1: c_theta"):
        driver.reevaluate(
            1.5,
            oracle=problem.oracle,
            formulation=NSlack(perturbed_features),
            features=perturbed_features,
            observed_features=perturbed_observed_features,
            shocks=1.5 * shocks0,
        )
    # Fail-closed: the hard-error released the backend, not just the reference.
    assert spy.close_calls == 1
    assert driver._master is None


@needs_highs
@pytest.mark.parametrize(
    "drift_lower, drift_upper",
    [
        (True, True),  # both bounds drift (the symmetric case)
        (False, True),  # only the upper bound drifts
        (True, False),  # only the lower bound drifts
    ],
)
def test_no_geometry_signature_theta_bounds_drift_hard_errors_at_g3(
    drift_lower: bool, drift_upper: bool
) -> None:
    """Omitting geometry_signature does not disable the automatic θ-box guard.

    A drift on only the upper bound, or only the lower, must still hard-error;
    a one-sided compare would admit the opposite drift silently.
    """
    params, observables, observed, shocks0, problem = _toy_inputs()
    n_items = observed.shape[1]
    driver = _make_driver(observables, observed, params, check_geometry=False)
    driver.fit(
        1.0,
        oracle=problem.oracle,
        formulation=NSlack(problem.features),
        features=problem.features,
        observed_features=problem.observed_features,
        shocks=shocks0,
    )
    spy = _spy_on_live_master(driver)

    lower = -2.0 * THETA_BOUND if drift_lower else -THETA_BOUND
    upper = 2.0 * THETA_BOUND if drift_upper else THETA_BOUND
    driver._parameters = Parameters({"theta": (lower, upper, n_items)})
    psi = 1.5
    psi_problem = _psi_problem(psi)
    with pytest.raises(ValueError, match=r"G3: theta_bounds"):
        driver.reevaluate(
            psi,
            oracle=psi_problem.oracle,
            formulation=NSlack(psi_problem.features),
            features=psi_problem.features,
            observed_features=psi_problem.observed_features,
            shocks=psi * shocks0,
        )
    assert spy.close_calls == 1
    assert driver._master is None


@needs_highs
def test_perturbation_b2_geometry_signature_drift_hard_errors_at_g2() -> None:
    """(b2) a geometry_signature returning a different fingerprint → G2.

    c_theta / weights / bounds are all ψ-invariant (a valid shocks-only ψ),
    but the caller's geometry_signature(ψ) returns a different fingerprint —
    the cut geometry is declared non-invariant, so the guard hard-errors at
    G2 (covers the all-bundle φ the observed-only c_theta cannot reach).
    """
    params, observables, observed, shocks0, problem = _toy_inputs()
    # The fingerprint flips once ψ != ψ0. Both values are same-length bytes: a
    # real drift (array.tobytes() over a fixed-shape geometry) keeps the
    # length and changes only the content, so the compare must read content,
    # not length.
    def drifting_signature(psi: object) -> bytes:
        return b"psi0" if float(psi) == 1.0 else b"psi1"

    driver = PersistentMasterFit(
        params,
        observables=observables,
        observed_bundles=observed,
        transport=SerialTransport(),
        config=_config(),
        rhs_transform=_rhs_transform,
        geometry_signature=drifting_signature,
        master_backend="highs",
        tolerance=TOLERANCE,
    )
    driver.fit(
        1.0,
        oracle=problem.oracle,
        formulation=NSlack(problem.features),
        features=problem.features,
        observed_features=problem.observed_features,
        shocks=shocks0,
    )
    spy = _spy_on_live_master(driver)
    with pytest.raises(ValueError, match=r"G2: geometry_signature"):
        driver.reevaluate(
            1.5,
            oracle=problem.oracle,
            formulation=NSlack(problem.features),
            features=problem.features,
            observed_features=problem.observed_features,
            shocks=1.5 * shocks0,
        )
    assert spy.close_calls == 1
    assert driver._master is None


@needs_highs
@pytest.mark.parametrize(
    "outcome",
    [
        "equal_admits_reuse",
        "content_drift_raises_g2",
        "length_mismatch_raises_g2",
    ],
)
def test_perturbation_b2_tuple_geometry_signature_branch(outcome: str) -> None:
    """G2 over a tuple[np.ndarray, ...] geometry signature (the array branch).

    b2 above exercises only the bytes branch of ``_signature_equal``. The
    documented tuple contract — length shortcut, then per-element
    shape/dtype/tobytes compare — has three outcomes:

    * equal tuples at ψ0 and ψ → reuse admitted (only the absence of a G2
      verdict is asserted; the warm solve may or may not converge).
    * a same-length tuple with one drifted element → G2 (content compare).
    * a length mismatch (drop one element) → G2 (the length shortcut).

    Both tuples are built by hand from the fixture geometry.
    """
    params, observables, observed, shocks0, problem = _toy_inputs()
    base = np.asarray(load_family("toy", FAMILY_DIR)["observables"], dtype=np.float64)

    def geometry_signature(psi: object):
        # ψ0 fingerprint: a two-array tuple. At any later ψ the fingerprint
        # either stays equal, drifts one element's content, or drops an element.
        psi0 = (base.copy(), 2.0 * base.copy())
        if float(psi) == 1.0:
            return psi0
        if outcome == "equal_admits_reuse":
            return (base.copy(), 2.0 * base.copy())
        if outcome == "content_drift_raises_g2":
            drifted = 2.0 * base.copy()
            drifted[0, 0] += 1.0  # same shape/dtype, one element differs
            return (base.copy(), drifted)
        # length_mismatch_raises_g2: same first element, second dropped
        return (base.copy(),)

    driver = PersistentMasterFit(
        params,
        observables=observables,
        observed_bundles=observed,
        transport=SerialTransport(),
        config=_config(),
        rhs_transform=_rhs_transform,
        geometry_signature=geometry_signature,
        master_backend="highs",
        tolerance=TOLERANCE,
    )
    driver.fit(
        1.0,
        oracle=problem.oracle,
        formulation=NSlack(problem.features),
        features=problem.features,
        observed_features=problem.observed_features,
        shocks=shocks0,
    )
    try:
        if outcome == "equal_admits_reuse":
            spy = _spy_on_live_master(driver)
            # Equal fingerprints: G2 must not raise, and the master stays live.
            re = driver.reevaluate(
                1.5,
                oracle=problem.oracle,
                formulation=NSlack(problem.features),
                features=problem.features,
                observed_features=problem.observed_features,
                shocks=1.5 * shocks0,
            )
            assert isinstance(re, PersistentFitResult)
            assert spy.close_calls == 0
            assert driver._master is not None
        else:
            spy = _spy_on_live_master(driver)
            with pytest.raises(ValueError, match=r"G2: geometry_signature"):
                driver.reevaluate(
                    1.5,
                    oracle=problem.oracle,
                    formulation=NSlack(problem.features),
                    features=problem.features,
                    observed_features=problem.observed_features,
                    shocks=1.5 * shocks0,
                )
            assert spy.close_calls == 1
            assert driver._master is None
    finally:
        driver.close()


@needs_highs
def test_perturbation_b3_weight_drift_hard_errors_at_g1_end_to_end() -> None:
    """(b3) a ψ that drifts agent_weights alone (c_theta held) → G1 hard-error.

    End-to-end reevaluate: c_theta is held byte-identical to the stashed ψ0
    vector while one agent's weight drifts, so the raise comes from G1's
    agent_weights branch, not from c_theta.

    In the toy fit every agent receives at least one installed cut, so this
    path cannot separate a full-vector compare from a subset-over-installed-
    cuts one; that distinction is covered unit-level in
    ``test_reuse_guard_needs_full_weight_vector_over_uninstalled_agent``.
    """
    persistent_mod = importlib.import_module("combrum.engine.persistent")
    params, observables, observed, shocks0, problem = _toy_inputs()
    n_agents = observed.shape[0]
    driver = PersistentMasterFit(
        params,
        observables=observables,
        observed_bundles=observed,
        transport=SerialTransport(),
        config=_config(),
        rhs_transform=_rhs_transform,
        geometry_signature=_geometry_signature_factory(
            np.asarray(load_family("toy", FAMILY_DIR)["observables"])
        ),
        master_backend="highs",
        tolerance=TOLERANCE,
        weights=np.ones(n_agents, dtype=np.float64),
    )
    driver.fit(
        1.0,
        oracle=problem.oracle,
        formulation=NSlack(problem.features),
        features=problem.features,
        observed_features=problem.observed_features,
        shocks=shocks0,
    )

    # In the toy fit every agent has an installed cut, so the drifted index is
    # installed (see docstring).
    installed = {int(row.agent_id) for row in driver._master.extract_cuts()}
    drift_index = n_agents - 1
    assert drift_index in installed

    spy = _spy_on_live_master(driver)

    # The reevaluate build holds c_theta byte-identical to ψ0 and drifts
    # agent_weights only at drift_index.
    real_build = persistent_mod.build_fit_context

    def wrapped_build(*args, **kwargs):  # type: ignore[no-untyped-def]
        built = real_build(*args, **kwargs)
        c_theta_identical = np.array(driver._c_theta0, dtype=np.float64)
        drifted_weights = np.array(built.ctx.agent_weights, dtype=np.float64)
        drifted_weights[drift_index] = drifted_weights[drift_index] + 5.0
        new_ctx = dataclasses.replace(built.ctx, agent_weights=drifted_weights)
        return dataclasses.replace(built, ctx=new_ctx, c_theta=c_theta_identical)

    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(persistent_mod, "build_fit_context", wrapped_build)
            with pytest.raises(ValueError, match=r"G1: agent_weights at psi"):
                driver.reevaluate(
                    1.5,
                    oracle=problem.oracle,
                    formulation=NSlack(problem.features),
                    features=problem.features,
                    observed_features=problem.observed_features,
                    shocks=1.5 * shocks0,
                )
        assert spy.close_calls == 1
        assert driver._master is None
    finally:
        driver.close()


def test_reuse_guard_needs_full_weight_vector_over_uninstalled_agent() -> None:
    """G1's weight guard must compare the full vector, not just installed cuts.

    The master's u_coef closure is frozen at ψ0 (the G1 agent_weights guard in
    ``_assert_reuse_valid``) and serves ψ0's weight to any cut warm row-gen
    prices for the first time at ψ, so a weight drift on an agent with no
    installed cut must still hard-error.

    Driven at the guard directly — the toy fit installs every agent, so no
    end-to-end reevaluate can present an uninstalled-agent drift. A fake
    master's ``extract_cuts()`` covers a strict subset of agents, and the
    built's agent_weights drifts outside that subset.
    """
    params, observables, observed, shocks0, _problem = _toy_inputs()
    n_agents = observed.shape[0]
    n_items = observed.shape[1]
    driver = PersistentMasterFit(
        params,
        observables=observables,
        observed_bundles=observed,
        transport=SerialTransport(),
        config=_config(),
        rhs_transform=_rhs_transform,
        master_backend="highs",
        tolerance=TOLERANCE,
    )

    # Stash a ψ0 signature by hand (independent of any solve): unit weights, a
    # distinct c_theta, the toy θ-box.
    weights0 = np.ones(n_agents, dtype=np.float64)
    c_theta0 = np.arange(n_items, dtype=np.float64)
    lower = np.full(n_items, -THETA_BOUND, dtype=np.float64)
    upper = np.full(n_items, THETA_BOUND, dtype=np.float64)
    driver._c_theta0 = c_theta0.copy()
    driver._agent_weights0 = weights0.copy()
    driver._theta_bounds0 = (lower.copy(), upper.copy())
    driver._geometry0 = None

    # A fake master whose installed cuts cover only agents {0, 1, 2}.
    installed_ids = (0, 1, 2)
    cuts = tuple(SimpleNamespace(agent_id=i) for i in installed_ids)
    driver._master = SimpleNamespace(extract_cuts=lambda: cuts)
    installed = {int(row.agent_id) for row in driver._master.extract_cuts()}
    drift_index = n_agents - 1  # agent 11: outside the installed subset
    assert drift_index not in installed

    # A built that trips only the weight branch: c_theta / bounds held, weight
    # drifted on the uninstalled agent.
    drifted = weights0.copy()
    drifted[drift_index] += 5.0
    built = SimpleNamespace(
        c_theta=c_theta0.copy(),
        ctx=SimpleNamespace(
            agent_weights=drifted,
            theta_bounds=(lower.copy(), upper.copy()),
        ),
    )

    with pytest.raises(ValueError, match=r"G1: agent_weights at psi"):
        driver._assert_reuse_valid(1.5, built)

    # Control: the same guard admits an undrifted vector, so the raise above
    # is the drift.
    clean = SimpleNamespace(
        c_theta=c_theta0.copy(),
        ctx=SimpleNamespace(
            agent_weights=weights0.copy(),
            theta_bounds=(lower.copy(), upper.copy()),
        ),
    )
    driver._assert_reuse_valid(1.5, clean)


# --- NSlack-only: reject any non-"NSlack"-named formulation -------------------


@needs_highs
def test_nslack_only_reject_at_cold_fit() -> None:
    """A OneSlack into PersistentMasterFit.fit → TypeError (fail-closed).

    The reject is by the exact defining identity (module + qualname,
    import-free), at the first arrival of the formulation (cold fit). OneSlack's
    single aggregate cut has an RHS that depends on the priced joint selection,
    so a per-cut RHS rewrite is undefined for it.
    """
    params, observables, observed, shocks0, problem = _toy_inputs()
    driver = _make_driver(observables, observed, params)
    with pytest.raises(TypeError, match=r"NSlack-only"):
        driver.fit(
            1.0,
            oracle=problem.oracle,
            formulation=OneSlack(problem.features),
            features=problem.features,
            observed_features=problem.observed_features,
            shocks=shocks0,
        )
    # No master was ever built (reject preceded the build).
    assert driver._master is None


@needs_highs
def test_nslack_only_rejects_same_name_fake() -> None:
    """A foreign class merely named "NSlack" is rejected.

    The guard checks the defining identity (module + qualname), not the bare
    class name.
    """
    params, observables, observed, shocks0, problem = _toy_inputs()
    driver = _make_driver(observables, observed, params)
    # name "NSlack", but __module__ is this test module — not the real class.
    fake = type("NSlack", (), {})()
    assert type(fake).__qualname__ == "NSlack"
    assert type(fake).__module__ != "combrum.formulations.nslack"
    with pytest.raises(TypeError, match=r"NSlack-only"):
        driver.fit(
            1.0,
            oracle=problem.oracle,
            formulation=fake,
            features=problem.features,
            observed_features=problem.observed_features,
            shocks=shocks0,
        )
    assert driver._master is None


@needs_highs
def test_nslack_only_reject_at_reevaluate() -> None:
    """A OneSlack handed to reevaluate after a valid NSlack cold fit → TypeError.

    The NSlack-only check runs wherever the formulation arrives, reevaluate
    included — a caller cannot smuggle a non-NSlack in on a later ψ.
    """
    params, observables, observed, shocks0, problem = _toy_inputs()
    driver = _make_driver(observables, observed, params)
    try:
        driver.fit(
            1.0,
            oracle=problem.oracle,
            formulation=NSlack(problem.features),
            features=problem.features,
            observed_features=problem.observed_features,
            shocks=shocks0,
        )
        with pytest.raises(TypeError, match=r"NSlack-only"):
            driver.reevaluate(
                1.5,
                oracle=problem.oracle,
                formulation=OneSlack(problem.features),
                features=problem.features,
                observed_features=problem.observed_features,
                shocks=1.5 * shocks0,
            )
    finally:
        driver.close()


# --- context manager closes the master ---------------------------------------


@needs_highs
def test_context_manager_closes_master() -> None:
    """The driver is a context manager that closes the live master on exit."""
    params, observables, observed, shocks0, problem = _toy_inputs()
    with _make_driver(observables, observed, params) as driver:
        driver.fit(
            1.0,
            oracle=problem.oracle,
            formulation=NSlack(problem.features),
            features=problem.features,
            observed_features=problem.observed_features,
            shocks=shocks0,
        )
        assert driver._master is not None
        # __exit__ must call master.close(), not merely drop the reference.
        spy = _spy_on_live_master(driver)
    assert spy.close_calls == 1
    assert driver._master is None
