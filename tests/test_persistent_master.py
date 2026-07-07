"""Gate: the persistent-master driver + the suppress_close flag.

A ``PersistentMasterFit`` holds one master across an outer ψ search: it
fail-closed-guards reuse validity, RHS-rewrites the carried cuts, and
warm-solves. This file pins:

* ``run_fit(suppress_close=True)`` retains the master (a spy master records
  close() — default closes, suppress_close does not).
* ``PersistentMasterFit`` is importable from ``combrum.engine``.
* cold ``fit`` + ``reevaluate`` over a valid shocks-only ψ runs end-to-end
  and publishes a result on the estimate criterion scale.
* the fail-closed perturbation tests: (b1) a ψ that changes observed-bundle
  φ/c_theta → G1, with and without the optional geometry signature; (b2) a
  geometry_signature drift → G2 when supplied; (b3) a ψ that changes a
  not-yet-installed agent's weight → G1's full-vector compare; and θ-box drift
  → G3 when the geometry signature is omitted.
* NSlack-only: a non-"NSlack"-named formulation → TypeError at first use.

The persistent driver receives the oracle as an argument and lazily defaults
to NSlack only when the caller omits a formulation. These tests mostly pass
the formulation explicitly so the guard behavior stays pinned.
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
from _support.families import load_family
from combrum.master import MasterBackend
from combrum.masters import gurobi as gurobi_backend
from combrum.masters import highs as highs_backend
from combrum.parameters import Parameters
from combrum.transport import LocalCluster, SerialTransport, Transport, TransportError
from combrum.transport.base import CutRow

from pathlib import Path

FAMILY_DIR = Path(__file__).resolve().parent / "fixtures" / "families"

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
    """The toy family's (params, observables, observed, shocks, problem)."""
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
    """A real MasterBackend that wraps a backend and records close().

    Subclasses the ABC (NSlack.setup ``isinstance``-checks the master), so the
    toy fit runs through a real highs solve while this counts close() calls —
    default run_fit closes once, suppress_close=True closes zero times. Every
    contract method forwards verbatim to the wrapped backend.
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
        # Forward to the inner backend's own property (HighsMaster tracks its
        # installed set independently of extract_cuts()), so a degraded
        # extract_cuts() does not silently corrupt this count too.
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
    """Swap the driver's live master for a _CloseSpyMaster that records close().

    Wraps the real backend built by ``fit`` so that any later ``master.close()``
    (guard hard-error teardown or context-manager exit) is counted. Every
    contract method forwards verbatim, so the driver's own guard/rewrite path is
    unchanged. This gives the fail-closed lifecycle claim a meaningful signal: a leaky
    ``close`` that only nils ``self._master`` leaves ``close_calls == 0``.
    """
    spy = _CloseSpyMaster(driver._master)
    driver._master = spy
    return spy


def _problem_features():
    return toy_problem(load_family("toy", FAMILY_DIR)).features


def _problem_observed_features():
    return toy_problem(load_family("toy", FAMILY_DIR)).observed_features


def _hand_empirical_moment() -> np.ndarray:
    """The published empirical_moment, hand-derived from the fixture.

    empirical_moment is the row-observed feature mean:
    (1/N)·Σ_i observed_features(i, observed_i). For the toy family
    observed_features(i, b) = b·r_i (φ = b·r), so it is (1/N)·Σ_i observed_i·r_i
    — computed straight off the fixture arrays, independent of any combrum fit
    or accessor. φ is ψ-invariant, so this is the value at every ψ.
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
    """An independent cold-rebuild row-generation objective on the toy fit.

    A fresh master built and solved from scratch — a distinct path from the
    retained master, so it is a valid oracle for the retained master's
    re-solve objective (not the code-under-test's own carried value).
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

    # The retained master is genuinely reusable, not merely non-closed: a fresh
    # re-solve reproduces the converged row-generation objective. The oracle is
    # a cold rebuild (a distinct master/path), so a suppress_close path that
    # retains a torn-down/degraded master is caught rather than passing on a
    # unexercised `is not None`.
    ref_objective = _cold_rebuild_objective(lambda: NSlack(problem.features))
    spy.solve()
    assert abs(spy.objective() - ref_objective) <= 1e-9

    # The carried cut set survived intact. spy.n_active_cuts forwards to the
    # backend's installed-set count (independent of extract_cuts()), so a
    # degraded extract_cuts() that drops rows is caught by the mismatch — the
    # old `extract_cuts() is not None` (always a tuple) never tripped.
    installed = spy.n_active_cuts
    assert installed > 0
    assert len(spy.extract_cuts()) == installed
    spy.close()


# --- PersistentMasterFit: importable + the construction surface ---------------


def test_persistent_master_fit_importable_from_engine() -> None:
    """PersistentMasterFit constructs to a clean pre-fit state and publishes the
    documented result surface.

    The plain ``is`` identity checks are collection-time tautologies (both names
    resolve to the already-imported module attribute), so they add no meaningful signal over
    the file-level import. Give the test real signal by constructing a driver and
    pinning its post-``__init__`` invariants (a broken/raising constructor is
    then caught) plus the PersistentFitResult field set (a dropped/renamed
    published field is caught).
    """
    from combrum.engine import PersistentFitResult as R
    from combrum.engine import PersistentMasterFit as D

    assert D is PersistentMasterFit
    assert R is PersistentFitResult

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
    # Pre-fit state: no master built yet, backend unresolved, no psi0 signature
    # stashed. A constructor that eagerly built/resolved or corrupted these
    # would break the fail-closed reuse contract.
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
        # A second cold fit: resolve_master_backend must NOT run again — the
        # is-None caching guard holds the psi0-resolved backend. Without a
        # second fit the guard is never exercised (one call looks the same
        # whether or not it caches).
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

    # One resolve across two fits: the backend is resolved once and cached.
    assert resolved_calls == ["auto"]
    # Every build (both fits + the reevaluate) threads the cached resolved
    # backend into build_fit_context.
    assert build_resolved == ["highs", "highs", "highs"]


def _config_with_penalty(qp_weight: float, decay: int) -> LoopConfig:
    """A LoopConfig whose (qp_weight, decay) may be an otherwise-rejected pair.

    ``LoopConfig`` validates qp_weight>0 needs decay>=1 at construction, but the
    require_quadratic guard is a runtime read of the two attributes. To separate
    the ``and`` clauses we need the qp_weight>0 / decay=0 corner, which the
    validator forbids — so build a valid frozen config, then override the two
    fields directly. Nothing else in fit() reads these before resolve.
    """
    cfg = LoopConfig(max_iterations=1, qp_weight=1.0, decay=1)
    object.__setattr__(cfg, "qp_weight", qp_weight)
    object.__setattr__(cfg, "decay", decay)
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

    # Drive three configs that make the two sub-conditions of
    #   require_quadratic = (qp_weight > 0.0 and decay > 0)
    # vary independently, and pin the EXACT captured vector against a
    # hand-derived truth table. A one-clause regression (drop the decay clause,
    # drop the qp clause) or `and`->`or` changes at least one entry:
    #   (qp>0, decay>0):  and=T  or=T  qp-only=T  decay-only=T
    #   (qp>0, decay=0):  and=F  or=T  qp-only=T  decay-only=F
    #   (qp=0, decay>0):  and=F  or=T  qp-only=F  decay-only=T
    cases = [
        (_config_with_penalty(1.0, 1), True),
        (_config_with_penalty(1.0, 0), False),
        (_config_with_penalty(0.0, 1), False),
    ]
    expected = [want for _cfg, want in cases]

    captured: list[bool] = []

    def fake_resolve(requested, **kwargs):  # type: ignore[no-untyped-def]
        captured.append(bool(kwargs["require_quadratic"]))
        # A quadratic backend is demanded exactly when require_quadratic is True;
        # mirror resolve_master_backend's own contract so the raise arm is only
        # reached in that case.
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
            # No quadratic demanded: resolve returns "highs" and build is next.
            # build_fit_context is stubbed to fail, so a raise here proves we
            # got past resolve without a quadratic RuntimeError (i.e. the guard
            # did NOT spuriously demand a quadratic backend).
            with pytest.raises(AssertionError, match="build_fit_context"):
                driver.fit(
                    1.0,
                    oracle=problem.oracle,
                    formulation=NSlack(problem.features),
                    features=problem.features,
                    observed_features=problem.observed_features,
                    shocks=shocks0,
                )

    # The exact captured truth table pins that BOTH clauses are load-bearing.
    assert captured == expected


def test_persistent_reevaluate_before_fit_raises_on_all_ranks_under_multirank() -> None:
    params, observables, observed, shocks0, problem = _toy_inputs()

    # Record each rank's own outcome. LocalCluster.run re-raises only the first
    # rank's error, so asserting on that alone cannot tell "all ranks agreed on
    # the raise" from "rank 0 raised, a worker stranded/returned" — the exact
    # desync the collective() guard prevents. We assert on the per-rank vector.
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

    # Every rank raised the SAME agreed TransportError (origin rank 0, the same
    # message) — the collective() guard propagated the "no live master" failure
    # to the worker rank rather than leaving it to return or strand at a
    # half-run collective. A rank-0-only raise leaves rank 1 either
    # "returned-normally" or raising a "rendezvous broken" TransportError from
    # a different origin; both fail these assertions.
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

    # Per-rank outcome vector: LocalCluster.run re-raises only the owner's
    # error, so it cannot tell "both ranks agreed on the set_rhs failure" from
    # "owner raised, worker returned success" — the desync the collective()
    # wrapper around the RHS rewrite exists to prevent.
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

    # Both ranks raised the SAME agreed TransportError (origin rank 0, "bad
    # rhs"): the collective() around the RHS rewrite turned the owner's
    # set_rhs failure into an agreed verdict every rank raises. If only the
    # owner raised, the worker rank returns normally (or strands), which these
    # per-rank assertions catch.
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

    # Independent cold rebuild (a distinct master/path from the driver's own
    # _publish): the oracle for cold.theta_hat and cold.iterations below. The
    # toy row-generation LP is deterministic, so a fresh build converges to the
    # byte-identical theta_hat.
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

        # theta_hat is published from result.theta_hat, the converged master
        # estimate. Pin the FULL (K,) vector byte-for-byte against the
        # independent cold rebuild: a _publish that ships the scalar objective
        # (or any wrong source/index) as theta_hat is caught wholesale, not just
        # its distinctness from empirical_moment.
        np.testing.assert_array_equal(cold.theta_hat, ref_theta_hat)

        # empirical_moment is published from built.empirical_moment (the observed
        # feature mean), NOT theta_hat. Pin it to the hand-derived fixture value
        # so a swapped/wrong-source _publish mapping is caught, and pin that the
        # two published vectors are genuinely distinct.
        np.testing.assert_allclose(cold.empirical_moment, hand_moment, atol=1e-9)
        assert not np.allclose(cold.theta_hat, cold.empirical_moment)

        # iterations is published from the diagnostics, matching a cold rebuild.
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
        # The carried cut set is a superset of the cold fit's (warm reuse only
        # grows the installed set), so the count is monotone non-decreasing.
        assert re.n_active_cuts >= n_cold_cuts
        # φ is ψ-invariant, so the warm eval's empirical_moment is the same
        # hand-derived vector — a wrong-ψ recompute or swapped source is caught.
        np.testing.assert_allclose(re.empirical_moment, hand_moment, atol=1e-9)
        # Warm reuse does no more row-generation than the cold fit paid for:
        # the warm eval takes strictly fewer iterations than the cold fit.
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
    converges to the same row-generation objective as a cold rebuild (the
    warm-start property — Q banded, the vertex/active set may differ). This
    pins the property on a single ψ.
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

    # A cold rebuild at the same ψ (fresh master, warm_cuts=None): the
    # rebuild-per-eval reference the persistent path must band against. Same
    # ψ-coherent oracle/features/shocks — only the master is cold-built.
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


# --- positive quadratic-penalty persistent path (decay -> pure LP) -----------


@needs_gurobi
@pytest.mark.slow
def test_persistent_penalty_decay_fit_then_reevaluate_runs_end_to_end() -> None:
    """A qp_weight>0 / decay>=1 config runs the persistent path on a real QP solve.

    ``require_quadratic`` resolves to True (qp_weight>0 and decay>0 at
    persistent.py:169-170), so the driver builds a gurobi master. The proximal
    weight decays linearly to exactly zero over ``decay`` iterations, so the
    terminating solve is a pure LP and a dual is published. Both the cold fit
    and a valid shocks-only reevaluate run end-to-end over the carried master.
    """
    params, observables, observed, shocks0, problem = _toy_inputs()
    penalty_config = LoopConfig(
        max_iterations=MAX_ITERATIONS,
        qp_weight=10.0,
        decay=3,
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
        # penalty finished decaying to weight 0).
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
    RHS-only class. The guard hard-errors at G1 (and closes the master).

    Single case, geometry_signature supplied: the φ drift trips the c_theta
    sub-guard of G1, which short-circuits before G2 is ever consulted (the
    geometry callable is called 0 times on the perturbed reevaluate). A
    check_geometry parametrization here would run byte-identical code twice —
    the G2 tuple/bytes checks live in the b2 tests instead.
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
    # The guard closed the master on the hard-error (fail-closed lifecycle): the
    # backend was released (close_calls == 1), not merely dereferenced.
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

    Each half of the G3 OR must carry independent signal: a drift on ONLY the
    upper bound, or ONLY the lower bound, must still hard-error. A one-sided
    guard (compares only lower, or only upper) admits the opposite-side drift
    silently.
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
    # Fail-closed: the G3 hard-error released the backend, not just the ref.
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
    # A geometry_signature that flips fingerprint once ψ != ψ0 (the ψ0 cold-fit
    # value is stashed; this returns a different one at the reevaluate ψ). The
    # two fingerprints are SAME-LENGTH, different-content bytes — the real
    # geometry signature is array.tobytes() over a fixed-shape geometry, so a
    # genuine drift keeps the length and changes only the content. This forces
    # _signature_equal's content comparison (not a length shortcut) to carry the
    # Content-comparison signal.
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
    # Fail-closed: the G2 hard-error released the backend, not just the ref.
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

    b2 above only exercises the bytes branch of ``_signature_equal``; the
    documented tuple contract (length shortcut, then per-element shape/dtype/
    tobytes compare) is otherwise untested — it could be replaced by a constant
    True with no failure. This drives the three tuple outcomes directly:

    * equal tuples at ψ0 and ψ → reuse admitted (no G2 raise; the reevaluate is
      allowed to proceed and warm-solve, may or may not converge — only the
      absence of a G2 verdict is asserted).
    * a same-length tuple with ONE drifted element → G2 (content compare).
    * a length mismatch (drop one element) → G2 (the len shortcut).

    The stashed ψ0 tuple and the ψ tuple are built by hand from the fixture
    geometry, independent of any combrum accessor.
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
            # Equal tuple fingerprints: the G2 guard must NOT raise. The warm
            # reevaluate proceeds (a raise here would be a G2 false-positive, or
            # would tear down the master). A tuple branch mutated to `return
            # False` would raise G2 on this equal case.
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

    The full end-to-end reevaluate path: c_theta is held byte-identical to the
    stashed ψ0 vector (so the c_theta sub-guard passes untripped) while a single
    agent's weight is drifted, and the reevaluate must hard-error at the
    agent_weights branch of G1 and close the master. This pins that the weight
    sub-guard fires *on a real master reevaluate*, separately from c_theta.

    It does NOT isolate the full-vector-vs-subset distinction: in the toy fit
    every agent (id 0..N*S-1) receives at least one installed cut, so a
    subset-over-installed-cuts compare covers the whole vector too. That
    distinction is pinned unit-level in
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

    # Independently record the installed cut set from the live master. In the
    # toy fit this covers every agent, so the drifted index -1 is installed —
    # documenting why this end-to-end path cannot isolate full-vs-subset.
    installed = {int(row.agent_id) for row in driver._master.extract_cuts()}
    drift_index = n_agents - 1
    assert drift_index in installed

    # Count the fail-closed teardown: the G1 hard-error must release the backend.
    spy = _spy_on_live_master(driver)

    # Wrap build_fit_context so the reevaluate built holds a c_theta that is
    # byte-identical to the stashed ψ0 vector (so the c_theta sub-guard passes
    # untripped) while agent_weights is drifted only at drift_index.
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
        # Fail-closed: the G1 weight hard-error released the backend, not just
        # the reference.
        assert spy.close_calls == 1
        assert driver._master is None
    finally:
        driver.close()


def test_reuse_guard_needs_full_weight_vector_over_uninstalled_agent() -> None:
    """G1's weight guard must compare the FULL vector, not just installed cuts.

    The src comment (persistent.py:325-327) warns that the master's u_coef
    closure is frozen at ψ0 and serves ψ0's weight to any *newly-priced-agent*
    cut — an agent with no cut at ψ0 that warm row-gen prices for the first time
    at ψ. So a weight drift on an uninstalled agent must still hard-error, which
    a subset-over-installed-cuts compare would miss.

    Driven at the guard directly (the toy fit installs every agent, so no
    end-to-end reevaluate can present an uninstalled-agent drift): a fake master
    whose ``extract_cuts()`` covers only a strict subset of agents, and a built
    whose agent_weights drifts on an agent *outside* that subset. The installed
    set is read from the fake master itself, independent of the guard.
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

    # Control: the same guard admits an *undrifted* weight vector (so the raise
    # above is the drift, not a mis-set fixture).
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
    """A non-NSlack class merely named "NSlack" is rejected (not spoofable).

    The guard pins the exact defining identity (module + qualname), so a class
    whose name is "NSlack" but whose module is not
    combrum.formulations.nslack is fail-closed rejected — a bare class-name
    check would have admitted this spoof.
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
    # No master was ever built (reject preceded the build).
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
        # Wrap the live backend so __exit__ has to actually call master.close(),
        # not merely drop the reference. spy.close_calls pins the backend was
        # released (a solver/environment leak otherwise sails through `is None`).
        spy = _spy_on_live_master(driver)
    # __exit__ closed it via close(): the backend was released exactly once, and
    # the reference was cleared.
    assert spy.close_calls == 1
    assert driver._master is None
