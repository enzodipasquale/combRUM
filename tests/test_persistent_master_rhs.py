"""The RHS-rewrite persistent-master speed layer, end-to-end.

Self-contained: builds a toy ψ-problem and checks, over real solves, that
one master held across an outer ψ search tracks a cold rebuild-per-eval to
within ``PARITY_BAND`` while doing strictly less work.

The valid ψ-class is the shocks-only outer search: ``shocks_ψ = ψ·shocks_ψ0``
so each cut's ``ε@ψ = ψ·ε@ψ0``, while φ / ``c_theta`` / the θ-box /
``agent_weights`` are ψ-invariant. ``_Psi0BaselineRhs`` supplies the ε rule
(and why it must multiply the ψ0 baseline, not a live cut's epsilon);
``_geometry_signature_factory`` supplies the optional G2 fingerprint. The
observed bundles are perturbed off the oracle's argmax (two flipped rows) so
the objective varies across the panel while staying in the ψ-class.
"""

from __future__ import annotations

import time

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
from combrum.masters import gurobi as gurobi_backend
from combrum.masters import highs as highs_backend
from combrum.parameters import Parameters
from combrum.transport import SerialTransport
from combrum.transport.base import CutRow

# Shared parity band, defined locally so this module stays self-contained.
PARITY_BAND = 1e-9

GUROBI_AVAILABLE = gurobi_backend.available()
HIGHS_AVAILABLE = highs_backend.available()

needs_gurobi = pytest.mark.skipif(
    not GUROBI_AVAILABLE, reason="gurobipy missing or no environment starts"
)
needs_highs = pytest.mark.skipif(
    not HIGHS_AVAILABLE, reason="highspy missing or broken"
)

# The speed layer is backend-agnostic, so run both LP backends where
# available.
REAL_BACKENDS = (
    pytest.param("gurobi", marks=needs_gurobi),
    pytest.param("highs", marks=needs_highs),
)

# The ψ-panel: psi0=1.0 first (the cold fit) plus ψ≠ψ0 on both sides of 1,
# so the reevaluate sequence both tightens (ψ>1) and loosens (ψ<1) the
# carried cuts' RHS. This ψ-class prices no new cut, so the carried set
# stays the whole ψ0 set across the run.
PSI_PANEL = (1.0, 1.1, 0.9, 1.25, 0.8)


# --- self-contained toy ψ-fixtures (mirrors the persistent-master patterns) ---


def _config() -> LoopConfig:
    return LoopConfig(max_iterations=MAX_ITERATIONS)


def _toy_arrays() -> dict:
    return dict(load_family("toy", FAMILY_DIR))


def _perturbed_observed(arrays: dict) -> np.ndarray:
    """The data: the toy observed bundles with two rows flipped off-argmax.

    Off-argmax observed rows make the objective vary with ψ without leaving
    the shocks-only ψ-class -- observed bundles are data, not required to be
    the oracle's argmax, and only the shocks scale with ψ.
    """
    observed = np.asarray(arrays["observed"]).copy()
    observed[0] = ~observed[0]
    observed[5] = ~observed[5]
    return observed


def _psi_problem(psi: float) -> FamilyProblem:
    """A ψ-coherent toy problem over ``shocks_ψ = ψ·shocks_ψ0``.

    The toy features are φ = b·r (ψ-invariant -- r is the geometry) and
    eps = b·ν, so scaling ν by ψ scales every cut's ε by exactly ψ
    (``ε@ψ = ψ·ε@ψ0``) and the priced demand by the same ν. c_theta / the
    θ-box / weights stay ψ-invariant.
    """
    arrays = _toy_arrays()
    arrays["shocks"] = psi * np.asarray(arrays["shocks"], dtype=np.float64)
    return toy_problem(arrays)


def _geometry_signature_factory(observables: np.ndarray):
    """A ψ-invariant geometry fingerprint over the toy observables.

    The toy features are φ = b·r, so the ψ-invariant cut geometry is the
    observables r; the canonical shocks-only ψ never touches r, so this
    signature is byte-identical across ψ (the G2 reuse guard passes).
    """
    fixed = np.asarray(observables, dtype=np.float64)

    def geometry_signature(_psi: object) -> bytes:
        return fixed.tobytes()

    return geometry_signature


class _Psi0BaselineRhs:
    """The caller-owned closed-form ε rule on the ψ0 baseline: ``ε@ψ = ψ·ε@ψ0``.

    ``set_rhs`` updates the backend's ``_installed`` mirror in place, so a
    live cut's ``CutRow.epsilon`` reflects the last ψ's rewrite, not ε@ψ0. A
    naive ``ψ·row.epsilon`` would compound over the reevaluate sequence
    (``ψ_k·ψ_{k-1}·…·ε@ψ0``) and the master would diverge. So track ε@ψ0 per
    cut key and multiply ψ by that stable baseline. ``snapshot(ψ)`` runs
    after each fit/reevaluate: a cut newly priced at ψ carries ``ε@ψ``, so
    its ε@ψ0 = ``ε@ψ / ψ``; existing keys keep their stashed baseline.
    """

    def __init__(self) -> None:
        self._eps0: dict[tuple[int, bytes], float] = {}

    def __call__(self, row: CutRow, psi: object) -> float:
        return float(psi) * self._eps0[(row.agent_id, row.bundle_key)]

    def snapshot(self, master, psi: float) -> None:
        # Seed ε@ψ0 for any unseen cut: its stored epsilon is ε at the ψ it
        # was priced under, so ε@ψ_add / ψ recovers the ψ0 baseline.
        for row in master.extract_cuts():
            key = (row.agent_id, row.bundle_key)
            if key not in self._eps0:
                self._eps0[key] = float(row.epsilon) / float(psi)


def _make_driver(
    observed: np.ndarray,
    backend: str,
    rhs: _Psi0BaselineRhs,
    arrays: dict,
    config: LoopConfig | None = None,
) -> PersistentMasterFit:
    n_obs, n_items = observed.shape
    params = Parameters({"theta": (-THETA_BOUND, THETA_BOUND, n_items)})
    return PersistentMasterFit(
        params,
        observables=list(range(n_obs)),
        observed_bundles=observed,
        transport=SerialTransport(),
        config=_config() if config is None else config,
        rhs_transform=rhs,
        geometry_signature=_geometry_signature_factory(
            np.asarray(arrays["observables"])
        ),
        master_backend=backend,
        tolerance=TOLERANCE,
    )


def _q_cold_rebuild(
    observed: np.ndarray, backend: str, psi: float, shocks0: np.ndarray
) -> tuple[float, int, int]:
    """The rebuild-per-eval reference: a cold ``build_fit_context(master=None)``
    + ``run_fit`` at ψ. Returns (objective, iterations, n_active_cuts)."""
    n_obs, n_items = observed.shape
    params = Parameters({"theta": (-THETA_BOUND, THETA_BOUND, n_items)})
    psi_problem = _psi_problem(psi)
    built = build_fit_context(
        params,
        observables=list(range(n_obs)),
        observed_bundles=observed,
        shocks=psi * shocks0,
        formulation=NSlack(psi_problem.features),
        features=psi_problem.features,
        observed_features=psi_problem.observed_features,
        transport=SerialTransport(),
        master_backend=backend,
        tolerance=TOLERANCE,
    )
    outcome = run_fit(
        built.ctx, psi_problem.oracle, NSlack(psi_problem.features), _config()
    )
    return (
        float(outcome.result.objective),
        int(outcome.diagnostics.iterations),
        int(outcome.result.n_active_cuts),
    )


def _run_persistent_panel(
    observed: np.ndarray, backend: str, shocks0: np.ndarray, panel
):
    """Drive one cold fit then a reevaluate sequence over ``panel`` on a single
    held master. Returns the per-ψ :class:`PersistentFitResult` list."""
    rhs = _Psi0BaselineRhs()
    arrays = _toy_arrays()
    driver = _make_driver(observed, backend, rhs, arrays)
    results: list[PersistentFitResult] = []
    try:
        for i, psi in enumerate(panel):
            psi_problem = _psi_problem(psi)
            kwargs = dict(
                oracle=psi_problem.oracle,
                formulation=NSlack(psi_problem.features),
                features=psi_problem.features,
                observed_features=psi_problem.observed_features,
                shocks=psi * shocks0,
            )
            if i == 0:
                result = driver.fit(psi, **kwargs)
            else:
                result = driver.reevaluate(psi, **kwargs)
            rhs.snapshot(driver._master, psi)
            results.append(result)
    finally:
        driver.close()
    return results


# --- (a) criterion parity over the ψ-panel ------------------------------------


@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_a_criterion_parity_over_psi_panel(
    backend: str,
) -> None:
    """Q_persistent tracks Q_rebuild to ≤ PARITY_BAND at every ψ.

    One cold ``fit(ψ0)`` then a ``reevaluate`` sequence over the panel vs a
    cold rebuild-per-eval at each ψ. The RHS-rewritten ψ0 cuts stay valid
    lower bounds at ψ, so warm row-gen converges to the same master objective
    as the rebuild (the vertex / active set may differ; only the objective is
    gated). The objective must also vary across the panel, or the banded
    match would just be a flat criterion.
    """
    arrays = _toy_arrays()
    observed = _perturbed_observed(arrays)
    shocks0 = np.asarray(arrays["shocks"], dtype=np.float64)

    persistent = _run_persistent_panel(observed, backend, shocks0, PSI_PANEL)
    q_persistent = {psi: r.objective for psi, r in zip(PSI_PANEL, persistent)}
    q_rebuild = {
        psi: _q_cold_rebuild(observed, backend, psi, shocks0)[0]
        for psi in PSI_PANEL
    }

    max_abs = max(abs(q_persistent[psi] - q_rebuild[psi]) for psi in PSI_PANEL)
    spread = max(q_persistent.values()) - min(q_persistent.values())
    print(
        f"\nRHS parity (a) {backend}: max|obj_p-obj_c|={max_abs:.3e}"
        f" (band {PARITY_BAND:.1e}), objective spread={spread:.4f},"
        f" objective={[round(q_persistent[p], 5) for p in PSI_PANEL]}"
    )

    # Parity: every ψ banded against its cold rebuild.
    for psi in PSI_PANEL:
        assert abs(q_persistent[psi] - q_rebuild[psi]) <= PARITY_BAND, (
            f"RHS parity {backend} ψ={psi}: |Qp-Qc|"
            f" {abs(q_persistent[psi] - q_rebuild[psi])!r} exceeds band"
            f" {PARITY_BAND!r}"
        )
    # The spread must sit well above the band, or the parity check is flat.
    assert spread > 1e3 * PARITY_BAND, (
        f"RHS parity {backend}: objective spread {spread!r} too small --"
        " ψ barely moves the objective across the panel"
    )
    # Convergence flag, both states: True across the panel...
    assert all(r.converged for r in persistent)
    # ...and False for a cold fit at ψ0 capped at max_iterations=1, which
    # cannot converge (the toy needs several row-gen iterations; test_c
    # shows 7-8).
    capped = _make_driver(
        observed, backend, _Psi0BaselineRhs(), arrays,
        config=LoopConfig(max_iterations=1),
    )
    try:
        psi0_problem = _psi_problem(1.0)
        capped_result = capped.fit(
            1.0,
            oracle=psi0_problem.oracle,
            formulation=NSlack(psi0_problem.features),
            features=psi0_problem.features,
            observed_features=psi0_problem.observed_features,
            shocks=shocks0,
        )
    finally:
        capped.close()
    assert capped_result.converged is False, (
        f"RHS parity {backend}: a cold fit(ψ0) capped at 1 iteration reported"
        " converged=True; the toy needs several row-gen iterations to certify"
    )
    # The cap actually bit: the run stopped at exactly one iteration.
    assert capped_result.iterations == 1


# --- (c) speed proxy + cut-count monotonicity ---------------------------------


@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_c_warm_iterations_le_cold_and_cut_count_monotonic(backend: str) -> None:
    """Speed proxy: warm iters ≤ cold iters per ψ; cut count monotone non-dec.

    The warm ``reevaluate`` starts from the RHS-rewritten ψ0 cut superset, so
    it does no more row-gen than a cold rebuild at any ψ (here it converges in
    a single pass after the cold fit). The carried set is reused whole:
    ``n_active_cuts`` never decreases and, on this shocks-only ψ-class (which
    prices no new cut), stays at the ψ0 cold-fit count — itself checked
    against an independent cold rebuild at ψ0. Wall-time is printed but not
    gated; a timing assert would flake under host load.
    """
    arrays = _toy_arrays()
    observed = _perturbed_observed(arrays)
    shocks0 = np.asarray(arrays["shocks"], dtype=np.float64)

    # Warm path: time the whole held-master panel.
    t0 = time.perf_counter()
    persistent = _run_persistent_panel(observed, backend, shocks0, PSI_PANEL)
    warm_wall = time.perf_counter() - t0
    warm_iters = [r.iterations for r in persistent]
    cut_seq = [r.n_active_cuts for r in persistent]

    # Cold path: a rebuild-per-eval at each ψ (the reference work the speed
    # layer avoids), timed the same way.
    t0 = time.perf_counter()
    cold_rebuilds = [
        _q_cold_rebuild(observed, backend, psi, shocks0) for psi in PSI_PANEL
    ]
    cold_wall = time.perf_counter() - t0
    cold_iters = [it for _obj, it, _nc in cold_rebuilds]
    # ψ0 cut-count floor from a distinct build, not read off the driver.
    cold_psi0_cuts = cold_rebuilds[0][2]

    print(
        f"\nRHS speed (c) {backend}: warm_iters={warm_iters}"
        f" cold_iters={cold_iters} cut_seq={cut_seq}"
        f" | wall warm={warm_wall * 1e3:.1f}ms cold={cold_wall * 1e3:.1f}ms"
        " (recorded, not gated)"
    )

    # Bounded on both sides: a row-gen master always runs at least one
    # iteration, and warm reuse never does more row-gen than cold.
    for psi, warm_i, cold_i in zip(PSI_PANEL, warm_iters, cold_iters):
        assert warm_i >= 1, (
            f"RHS speed {backend} ψ={psi}: warm iters {warm_i} < 1 — a real"
            " solve runs at least one iteration"
        )
        assert cold_i >= 1, (
            f"RHS speed {backend} ψ={psi}: cold iters {cold_i} < 1"
        )
        assert warm_i <= cold_i, (
            f"RHS speed {backend} ψ={psi}: warm iters {warm_i} >"
            f" cold iters {cold_i} (warm reuse should do no more row-gen)"
        )
    # ψ0 anchor: the persistent cold fit and the independent cold rebuild run
    # identical row-gen from scratch, so their iteration counts must be equal.
    # cold_iters[0] comes straight from run_fit's diagnostics, never through
    # PersistentMasterFit._publish.
    assert warm_iters[0] == cold_iters[0], (
        f"RHS speed {backend}: persistent cold fit(ψ0) reported"
        f" {warm_iters[0]} iters but the independent cold rebuild at ψ0"
        f" reported {cold_iters[0]}"
    )
    # The full iteration vector has a closed-form expectation: on this ψ-class
    # the rewritten ψ0 superset already bounds each new ψ-optimum, so every
    # warm reevaluate solves once, prices, admits nothing, and stops
    # (driver.py: one pass, convergence on a sweep with no violated cut) —
    # exactly 1 iteration. The ψ0 slot is the cold fit, anchored above.
    expected_iters = [cold_iters[0]] + [1] * (len(PSI_PANEL) - 1)
    assert warm_iters == expected_iters, (
        f"RHS speed {backend}: published warm iterations {warm_iters} !="
        f" expected {expected_iters} — a warm reevaluate on the shocks-only"
        " ψ-class is a single solve+price+converge pass"
    )
    # Cut-count carry. Three checks, strongest last:
    # (1) monotone non-decreasing — no carried cut is ever dropped.
    for prev, nxt in zip(cut_seq, cut_seq[1:]):
        assert nxt >= prev, (
            f"RHS cut-count {backend}: not monotone non-decreasing:"
            f" {cut_seq}"
        )
    # (2) the cold fit(ψ0) reproduces the independent cold rebuild's count.
    assert cut_seq[0] == cold_psi0_cuts, (
        f"RHS cut-count {backend}: cold fit(ψ0) n_active_cuts {cut_seq[0]} !="
        f" independent cold rebuild at ψ0 {cold_psi0_cuts}"
    )
    # (3) warm reuse carries the ψ0 superset whole: this ψ-class prices no new
    # cut, so the count stays exactly at the ψ0 floor — reevaluate neither
    # rebuilds from scratch nor drops or grows the carried set.
    assert cut_seq == [cold_psi0_cuts] * len(PSI_PANEL), (
        f"RHS cut-count {backend}: warm reuse did not carry the ψ0 superset"
        f" whole: cut_seq={cut_seq}, expected constant at ψ0 floor"
        f" {cold_psi0_cuts}"
    )
    # At least one warm reevaluate beats its cold rebuild outright — the
    # warm-start payoff is real, not a tie everywhere.
    assert any(
        warm_i < cold_i for warm_i, cold_i in zip(warm_iters[1:], cold_iters[1:])
    )


# A ψ far enough from ψ0 that the rewritten ψ0 cuts no longer bound the new
# optimum, so a warm reevaluate must price new cuts, while staying in the valid
# ψ-class (ε scales by ψ; geometry/weights/box fixed). Chosen so both real
# backends grow their cut set here.
PSI_ROWGEN = 10.0


@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_c_warm_iterations_tied_to_cut_growth(backend: str) -> None:
    """A warm reevaluate that prices new cuts must publish >1 iteration.

    The large ψ step makes the rewritten ψ0 cuts under-bound the new optimum,
    forcing real warm row-gen. A pass installs a cut only when its reduced
    cost exceeds ``ctx.tolerance`` (nslack.py), so the driver's stop rule
    (driver.py) cannot certify convergence on that pass — a converged warm
    reevaluate whose active-cut count grew therefore ran at least two passes.
    That ties the published iteration count to the independently published
    cut count.
    """
    arrays = _toy_arrays()
    observed = _perturbed_observed(arrays)
    shocks0 = np.asarray(arrays["shocks"], dtype=np.float64)

    persistent = _run_persistent_panel(
        observed, backend, shocks0, (1.0, PSI_ROWGEN)
    )
    cold = persistent[0]
    warm = persistent[1]

    # Converged and in-band against a cold rebuild at ψ_rowgen: the extra
    # passes were real row-gen.
    q_cold = _q_cold_rebuild(observed, backend, PSI_ROWGEN, shocks0)
    assert warm.converged, (
        f"RHS row-gen {backend}: warm reevaluate at ψ={PSI_ROWGEN} did not"
        " converge; the case is only meaningful on a converged warm solve"
    )
    assert abs(warm.objective - q_cold[0]) <= PARITY_BAND, (
        f"RHS row-gen {backend}: warm objective {warm.objective!r} does not"
        f" match the cold rebuild {q_cold[0]!r} within the band — the warm"
        " reevaluate solved a different problem"
    )

    # The ψ-step must actually force new cuts, or the claim below is idle.
    cut_growth = warm.n_active_cuts - cold.n_active_cuts
    assert cut_growth > 0, (
        f"RHS row-gen {backend}: ψ={PSI_ROWGEN} did not grow the carried cut"
        f" set (cold {cold.n_active_cuts} -> warm {warm.n_active_cuts}); the"
        " case no longer exercises warm row-gen — retune PSI_ROWGEN"
    )

    # A converged warm reevaluate that grew its cut set ran at least two
    # passes (see docstring).
    assert warm.iterations >= 2, (
        f"RHS row-gen {backend}: warm reevaluate grew the cut set by"
        f" {cut_growth} yet published only {warm.iterations} iteration(s); a"
        " pass that installs a cut has violation > tolerance, so a converged"
        " solve that admitted cuts ran at least two passes"
    )

    # Even with the extra row-gen, warm reuse still runs fewer passes than a
    # cold rebuild.
    assert warm.iterations < q_cold[1], (
        f"RHS row-gen {backend}: warm reevaluate took {warm.iterations}"
        f" iterations, not fewer than the cold rebuild's {q_cold[1]} at"
        f" ψ={PSI_ROWGEN} — the warm-start advantage is gone"
    )


# --- (d) NSlack-only -----------------------------------------------------------


@needs_highs
def test_d_oneslack_rejected_at_cold_fit() -> None:
    """A OneSlack into the persistent driver raises TypeError before any build.

    OneSlack installs one aggregate cut whose RHS depends on the priced joint
    selection, so a per-cut ``set_rhs(ε@ψ)`` rewrite is undefined for it.
    """
    arrays = _toy_arrays()
    observed = _perturbed_observed(arrays)
    shocks0 = np.asarray(arrays["shocks"], dtype=np.float64)
    problem = toy_problem(arrays)
    driver = _make_driver(observed, "highs", _Psi0BaselineRhs(), arrays)
    with pytest.raises(TypeError, match=r"NSlack-only"):
        driver.fit(
            1.0,
            oracle=problem.oracle,
            formulation=OneSlack(problem.features),
            features=problem.features,
            observed_features=problem.observed_features,
            shocks=shocks0,
        )
    # The reject preceded any build — no master leaked.
    assert driver._master is None


# --- (e) end-to-end example: a numpy-only outer ψ-search ----------------------


@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_e_numpy_only_golden_section_outer_search(backend: str) -> None:
    """End-to-end: a numpy-only golden-section ψ-search on one held master.

    One cold ``fit`` at the first probe, then a ``reevaluate`` per later
    probe (pure numpy — scipy is not a dependency). The objective increases
    monotonically in ψ over the toy panel, so the minimiser lands at the
    bracket floor, and the driver's own iteration/cut diagnostics show the
    one-cold-build + N-warm-reuses decomposition.
    """
    arrays = _toy_arrays()
    observed = _perturbed_observed(arrays)
    shocks0 = np.asarray(arrays["shocks"], dtype=np.float64)

    rhs = _Psi0BaselineRhs()
    driver = _make_driver(observed, backend, rhs, arrays)
    calls = {"cold": 0, "warm": 0}
    started = {"flag": False}
    # Per-eval diagnostics the driver itself reports; `calls` is only the
    # test's own branch counter.
    eval_iters: list[int] = []
    eval_cuts: list[int] = []

    def objective(psi: float) -> float:
        # Cold fit at the first probe, warm reevaluate at every later one.
        psi_problem = _psi_problem(psi)
        kwargs = dict(
            oracle=psi_problem.oracle,
            formulation=NSlack(psi_problem.features),
            features=psi_problem.features,
            observed_features=psi_problem.observed_features,
            shocks=psi * shocks0,
        )
        if not started["flag"]:
            result = driver.fit(psi, **kwargs)
            calls["cold"] += 1
            started["flag"] = True
        else:
            result = driver.reevaluate(psi, **kwargs)
            calls["warm"] += 1
        rhs.snapshot(driver._master, psi)
        eval_iters.append(result.iterations)
        eval_cuts.append(result.n_active_cuts)
        return result.objective

    # Pure-numpy golden-section minimisation over [lo, hi].
    lo, hi = 0.8, 1.25
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0  # 1/φ, the golden ratio reciprocal
    n_steps = 12
    try:
        c = hi - inv_phi * (hi - lo)
        d = lo + inv_phi * (hi - lo)
        f_c = objective(c)
        f_d = objective(d)
        for _ in range(n_steps):
            if f_c < f_d:
                hi, d, f_d = d, c, f_c
                c = hi - inv_phi * (hi - lo)
                f_c = objective(c)
            else:
                lo, c, f_c = c, d, f_d
                d = lo + inv_phi * (hi - lo)
                f_d = objective(d)
        psi_star = 0.5 * (lo + hi)
        q_star = min(f_c, f_d)
    finally:
        driver.close()

    print(
        f"\nRHS search (e) {backend}: psi_star={psi_star:.5f} obj*={q_star:.5f}"
        f" | calls cold={calls['cold']} warm={calls['warm']}"
        f" | eval_iters={eval_iters} eval_cuts={eval_cuts}"
    )

    # Total objective evals = 2 golden-section seeds + 1 new probe per step
    # (each step reuses one prior point); the first eval is the cold fit, the
    # rest are warm reevaluates.
    total_evals = 2 + n_steps
    # (i) the closure took the cold branch once, then always warm.
    assert calls["cold"] == 1
    assert calls["warm"] == total_evals - 1
    assert len(eval_iters) == total_evals
    # (ii) one held master: the reported cut count is constant across the
    # whole search — a rebuild-per-probe (or dropped cuts) would vary it.
    assert eval_cuts == [eval_cuts[0]] * total_evals, (
        f"RHS search (e) {backend}: n_active_cuts varied across the search"
        f" ({eval_cuts}); a single held master must carry the same cut set"
    )
    # (iii) the cold/warm decomposition in the driver's own diagnostics: the
    # first eval priced its cuts from scratch, every later one reused them.
    assert eval_iters[0] > 1, (
        f"RHS search (e) {backend}: first eval reported {eval_iters[0]}"
        " iterations — a cold fit must price its cuts over several row-gen"
        " passes"
    )
    # On this ψ-class the rewritten superset already bounds every probe's
    # optimum, so each warm reevaluate is exactly one solve+price+converge
    # pass.
    warm_evals = eval_iters[1:]
    assert warm_evals == [1] * (total_evals - 1), (
        f"RHS search (e) {backend}: warm eval iterations {warm_evals} !="
        f" {[1] * (total_evals - 1)} — every warm reevaluate on the"
        " shocks-only ψ-class is a single solve+price+converge pass"
    )
    # A sensible landing: the objective rises with ψ, so the minimiser sits at
    # the bracket floor (within the golden-section resolution after n_steps
    # contractions).
    assert abs(psi_star - 0.8) < 0.05, (
        f"RHS search (e) {backend}: psi_star {psi_star!r} did not land near the"
        " expected ψ-search minimum (the bracket floor)"
    )
