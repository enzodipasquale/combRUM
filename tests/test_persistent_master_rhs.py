"""The RHS-rewrite persistent-master speed layer, end-to-end.

``set_rhs``, ``build_fit_context(master=)`` + observed-objective construction,
and :class:`~combrum.engine.persistent.PersistentMasterFit` are integrated.
This module is tests-only and imports no helper from another test module. It
builds a self-contained toy ψ-problem and validates, over a real solve, the
speed layer's promise: one master held across an outer ψ search tracks a cold
rebuild-per-eval to within ``PARITY_BAND`` while doing strictly less work.

The valid ψ-class (used exactly here) is the shocks-only outer search:
``shocks_ψ = ψ·shocks_ψ0`` so each cut's ``ε@ψ = ψ·ε@ψ0``, while φ /
``c_theta`` / the θ-box / ``agent_weights`` are ψ-invariant (only ε scales).
So:

* ``rhs_transform = lambda row, ψ: ψ · ε@ψ0(row)`` — the bundle-free
  closed-form ε rule. The transform must multiply the ψ0 baseline, not a
  live cut's ``CutRow.epsilon`` (which already reflects the last ``set_rhs``
  rewrite and would compound over the sequence). ``_Psi0BaselineRhs`` tracks
  ε@ψ0 per cut; see its docstring.
* ``geometry_signature = lambda ψ: <the fixed observables as bytes>`` — an
  optional G2 check. The toy geometry is the observables r (φ = b·r), which
  the shocks-only ψ never touches, so the fingerprint is byte-identical across
  ψ and the G2 guard passes. Omitting the callback makes that invariance a
  caller precondition instead of a checked fingerprint.

The observed bundles here are deliberately perturbed off the oracle's argmax
(two flipped rows), so the objective varies across the panel (the coverage
the parity gate needs) while staying squarely in the valid shocks-only ψ-class.

This file focuses on:

* (a) **objective parity, ψ-panel** — persistent objective (one cold ``fit`` then
  a ``reevaluate`` sequence) vs rebuild objective (a cold rebuild-per-eval)
  banded to ``PARITY_BAND`` at every ψ, and exercised (the objective varies
  well above the band).
* (c) **speed + cut-count carry** — ``n_active_cuts`` monotone
  non-decreasing across the reevaluate sequence and, on the shocks-only ψ-class,
  held constant at the ψ0 cold-fit count (the carried superset is reused whole;
  this ψ-class prices no new cut, so it neither grows nor drops). The ψ0 count
  is pinned against an independent cold rebuild at ψ0. Plus a deterministic
  speed proxy: warm ``reevaluate`` iterations
  ≤ the cold rebuild's per ψ (warm-start does no-more row-gen). Wall-time is
  recorded for the record, never gated (it would be flaky).
* (d) **NSlack-only** — a OneSlack into the driver is rejected (TypeError).
* (e) **end-to-end example** — a numpy-only golden-section outer ψ-search that
  drives one cold ``fit`` then ``reevaluate`` across the search, lands a
  sensible ψ, and uses exactly one cold fit + N warm reevaluates (the
  speed-layer demonstration).
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
from _support.families import load_family
from combrum.masters import gurobi as gurobi_backend
from combrum.masters import highs as highs_backend
from combrum.parameters import Parameters
from combrum.transport import SerialTransport
from combrum.transport.base import CutRow

from pathlib import Path

FAMILY_DIR = Path(__file__).resolve().parent / "fixtures" / "families"

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

# Both real backends where the markers allow (highs at minimum) — the speed
# layer is backend-agnostic, so the gates ride both LP hosts.
REAL_BACKENDS = (
    pytest.param("gurobi", marks=needs_gurobi),
    pytest.param("highs", marks=needs_highs),
)

# The ψ-panel: psi0=1.0 first (the cold fit) plus several ψ≠ψ0 on both sides
# of 1, so the reevaluate sequence exercises tighten (ψ>1) and loosen (ψ<1) of
# the carried cuts' RHS. The shocks-only ψ-class prices no new cut, so the
# carried superset stays the whole ψ0 set across the run (constant count).
PSI_PANEL = (1.0, 1.1, 0.9, 1.25, 0.8)


# --- self-contained toy ψ-fixtures (mirrors the persistent-master patterns) ---


def _config() -> LoopConfig:
    return LoopConfig(max_iterations=MAX_ITERATIONS)


def _toy_arrays() -> dict:
    return dict(load_family("toy", FAMILY_DIR))


def _perturbed_observed(arrays: dict) -> np.ndarray:
    """The data: the toy observed bundles with two rows flipped off-argmax.

    Flipping two observed rows off the oracle's argmax makes the objective
    genuinely ψ-varying without leaving the valid shocks-only ψ-class:
    observed bundles are data, not required to be the oracle's argmax, and
    only the shocks scale with ψ.
    """
    observed = np.asarray(arrays["observed"]).copy()
    observed[0] = ~observed[0]
    observed[5] = ~observed[5]
    return observed


def _psi_problem(psi: float) -> FamilyProblem:
    """A ψ-coherent toy problem over ``shocks_ψ = ψ·shocks_ψ0``.

    The toy features are φ = b·r (ψ-invariant — r is the geometry) and
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

    The driver rewrites every carried cut's RHS via ``set_rhs``, which updates
    the backend's ``_installed`` mirror in place, so a live cut's
    ``CutRow.epsilon`` reflects the last ψ's rewrite, not ε@ψ0. A naive
    ``ψ·row.epsilon`` would therefore compound over the reevaluate sequence
    (``ψ_k·ψ_{k-1}·…·ε@ψ0``) and the carried cuts would no longer describe ψ —
    the master would diverge (we observed it hit ``max_iterations``).

    The ψ-class contract is the caller's to honor, so this tracks ε@ψ0 per cut
    key and multiplies ψ by that stable baseline. ``snapshot(ψ)`` is called
    after each fit/reevaluate: a cut newly priced by row-gen at ψ carries
    ``ε@ψ``, so its ε@ψ0 = ``ε@ψ / ψ`` (the toy's exact ``ε@ψ = ψ·ε@ψ0``);
    existing keys keep their stashed ε@ψ0. The transform then reads the
    baseline — the correct closed form over the accumulating cut superset.
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


# --- (a) criterion parity over the ψ-panel (exercised) ----------------------


@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_a_criterion_parity_psi_panel_is_banded_and_nonvacuous(
    backend: str,
) -> None:
    """Q_persistent tracks Q_rebuild to ≤ PARITY_BAND at every ψ, exercisedly.

    One cold ``fit(ψ0)`` then a ``reevaluate`` sequence over the panel (the cut
    superset accumulates) vs a cold rebuild-per-eval at each ψ. The
    RHS-rewritten ψ0 cuts stay valid lower bounds at ψ, so warm row-gen
    converges to the same master objective as the rebuild (the vertex / active
    set may differ, only the objective is gated). Non-vacuity: the objective
    must genuinely vary across the panel, so the banded match is a real
    result, not ψ leaving the objective flat.
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
    # Non-vacuity: the objective genuinely varies (the ≤1e-9 match is a real
    # result, not a constant-objective artifact). The band is 1e-9; a spread
    # orders of magnitude above it proves ψ moves the criterion.
    assert spread > 1e3 * PARITY_BAND, (
        f"RHS parity {backend}: objective spread {spread!r} is not above"
        " the band — the panel is unexercised (ψ does not move the objective)"
    )
    # The convergence flag, exercised in BOTH states so a hard-wired
    # converged=True cannot pass unnoticed.
    #   True state: every panel ψ genuinely converged (the warm reuse solved,
    #     did not bail at the cap).
    assert all(r.converged for r in persistent)
    #   False state: a cold fit at ψ0 under max_iterations=1 CANNOT converge —
    #     this toy needs several row-gen iterations (test_c shows 7-8), so a
    #     single-iteration walk must report converged is False. The floor is
    #     hand-derived (one iteration is below the true iteration count), not
    #     read off the driver; a driver that reports True after hitting the cap
    #     is broken.
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
        " converged=True, but the toy needs several row-gen iterations to"
        " certify — the convergence flag is not the loop's real verdict"
    )
    # Sanity that the cap actually bit (guards against the fit silently
    # converging in one iteration and making the False check unexercised).
    assert capped_result.iterations == 1


# --- (c) speed proxy + cut-count monotonicity ---------------------------------


@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_c_warm_iterations_le_cold_and_cut_count_monotonic(backend: str) -> None:
    """Speed proxy: warm iters ≤ cold iters per ψ; cut count monotone non-dec.

    The deterministic, non-flaky speed claim: the warm ``reevaluate`` warm-
    starts from the RHS-rewritten ψ0 cut superset, so it does no-more row-gen
    than a cold rebuild — its iteration count is ≤ the cold rebuild's at every
    ψ (here the warm reuse converges in a single iteration after the cold fit).
    And the carried cut set is reused whole: ``n_active_cuts`` is monotone
    non-decreasing across the reevaluate sequence and, on this shocks-only
    ψ-class (which prices no new cut), stays constant at the ψ0 cold-fit count
    — the warm reuse neither drops a carried cut nor rebuilds from scratch. The
    ψ0 count is pinned against an independent cold rebuild at ψ0, so the pin is
    not derived from the driver under test. Wall-time is recorded for the record
    but not gated — a timing assert would be flaky under host load.
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
    # The independent ψ0 floor: a separate cold rebuild at ψ0=1.0 (PSI_PANEL[0]),
    # not read off the persistent driver — this is what the carried superset must
    # equal, derived from a distinct build.
    cold_psi0_cuts = cold_rebuilds[0][2]

    print(
        f"\nRHS speed (c) {backend}: warm_iters={warm_iters}"
        f" cold_iters={cold_iters} cut_seq={cut_seq}"
        f" | wall warm={warm_wall * 1e3:.1f}ms cold={cold_wall * 1e3:.1f}ms"
        " (recorded, not gated)"
    )

    # The deterministic speed proxy, bounded on BOTH sides so an under-reported
    # iteration count (the persistent driver publishing fewer iters than it ran)
    # cannot masquerade as a speedup.
    #   lower: every reported iteration count is a real solve (>= 1). A row-gen
    #     master always runs at least one iteration; 0 or a negative published
    #     value means the count was fabricated, not measured.
    #   upper: warm does no-more row-gen than cold (the speed claim).
    for psi, warm_i, cold_i in zip(PSI_PANEL, warm_iters, cold_iters):
        assert warm_i >= 1, (
            f"RHS speed {backend} ψ={psi}: warm iters {warm_i} < 1 — a real"
            " solve reports at least one iteration; this count was not measured"
        )
        assert cold_i >= 1, (
            f"RHS speed {backend} ψ={psi}: cold iters {cold_i} < 1"
        )
        assert warm_i <= cold_i, (
            f"RHS speed {backend} ψ={psi}: warm iters {warm_i} >"
            f" cold iters {cold_i} (warm reuse should do no-more row-gen)"
        )
    # The ψ0 iteration anchor: the persistent cold fit(ψ0) and the independent
    # cold rebuild at ψ0 solve the same problem from scratch with identical
    # row-gen, so they MUST report the same iteration count. cold_iters[0] is
    # read straight off run_fit's diagnostics in _q_cold_rebuild — it never
    # passes through PersistentMasterFit._publish, so it is an oracle the driver
    # cannot influence. This equality (not merely `<=`) is what forces the
    # published warm count to equal the true count at ψ0.
    assert warm_iters[0] == cold_iters[0], (
        f"RHS speed {backend}: persistent cold fit(ψ0) reported"
        f" {warm_iters[0]} iters but the independent cold rebuild at ψ0 reported"
        f" {cold_iters[0]} — the driver's published iteration count is not the"
        " count it actually ran"
    )
    # The full warm-iteration vector, pinned against a hand-derived oracle so an
    # entire class of iteration-count corruptions dies (not just the one-sided
    # >=1 / <=cold pair, which a fabricated constant 1 slips through — in this
    # toy 1 is simultaneously the >=1 floor AND the truth).
    #
    # Every warm reevaluate on the shocks-only ψ-class does ZERO row-gen: the
    # RHS-rewritten ψ0 cut superset already bounds the new ψ-optimum, so the
    # driver loop (driver.py: one `iterations += 1` per pass, convergence when a
    # full sweep prices no violated cut) solves once, prices, admits nothing, and
    # stops — exactly one pass. So the true published warm count is 1 by the loop
    # contract, not read off the driver. The ψ0 slot is the cold fit, anchored to
    # the independent cold rebuild above. The result is a wholesale pin of the
    # published iteration vector against a fully external expectation.
    expected_iters = [cold_iters[0]] + [1] * (len(PSI_PANEL) - 1)
    assert warm_iters == expected_iters, (
        f"RHS speed {backend}: published warm iterations {warm_iters} !="
        f" hand-derived expected {expected_iters} — a warm reevaluate on the"
        " shocks-only ψ-class does zero row-gen (one solve+price+converge pass ="
        " 1 iteration); any other value means the count was fabricated or the"
        " warm-start payoff was lost"
    )
    # Cut-count carry. Three checks, strongest last:
    # (1) monotone non-decreasing — no carried cut is ever dropped.
    for prev, nxt in zip(cut_seq, cut_seq[1:]):
        assert nxt >= prev, (
            f"RHS cut-count {backend}: not monotone non-decreasing:"
            f" {cut_seq}"
        )
    # (2) the cold fit(ψ0) reproduces the independent cold rebuild at ψ0 (the
    # superset floor is a real, externally-derived count, not a self-report).
    assert cut_seq[0] == cold_psi0_cuts, (
        f"RHS cut-count {backend}: cold fit(ψ0) n_active_cuts {cut_seq[0]} !="
        f" independent cold rebuild at ψ0 {cold_psi0_cuts}"
    )
    # (3) warm reuse carries the ψ0 superset whole: on this shocks-only ψ-class
    # (which prices no new cut) the count stays exactly the ψ0 floor at every ψ.
    # This fails if reevaluate rebuilds from scratch (a backend whose cold count
    # varies with ψ would break the constant), drops carried cuts, or lets the
    # superset drift — the accumulation claim is now pinned, not merely tolerated
    # by a `>=` that equality satisfies vacuously.
    assert cut_seq == [cold_psi0_cuts] * len(PSI_PANEL), (
        f"RHS cut-count {backend}: warm reuse did not carry the ψ0 superset"
        f" whole: cut_seq={cut_seq}, expected constant at ψ0 floor"
        f" {cold_psi0_cuts}"
    )
    # Non-vacuity of the proxy: a later warm reevaluate is strictly cheaper than
    # its cold rebuild (the warm-start payoff is real, not a tie everywhere).
    assert any(
        warm_i < cold_i for warm_i, cold_i in zip(warm_iters[1:], cold_iters[1:])
    )


# A ψ far enough from ψ0 that the RHS-rewritten ψ0 cut superset no longer bounds
# the new optimum, so the warm reevaluate must price new cuts (verified in-class:
# ε scales by ψ, geometry/weights/box fixed, so G1/G2/G3 still pass). Chosen so
# BOTH real backends grow their cut set and take >1 warm iteration here.
PSI_ROWGEN = 10.0


@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_c_warm_iterations_tied_to_cut_growth(backend: str) -> None:
    """A warm reevaluate that prices new cuts must publish >1 iteration.

    test_c's shocks-only panel converges every warm reevaluate in exactly one
    pass, so ``iterations == 1`` is simultaneously the truth and the floor — a
    driver that fabricated a constant ``1`` for warm reevaluates would pass it.
    This case forces a warm reevaluate to do real row-gen (a large ψ step whose
    rewritten ψ0 cuts under-bound the new optimum) and ties the published
    iteration count to an INDEPENDENT field the fabrication cannot fake: the
    carried-cut count.

    The oracle is a loop invariant, not a self-report. A pass installs a cut
    only when its reduced cost exceeds ``ctx.tolerance`` (nslack.py), so that
    pass's worst violation exceeds tolerance and the driver's stop rule
    (driver.py) cannot certify convergence on it — a converged warm reevaluate
    whose active-cut count GREW therefore ran at least two passes. ``iterations``
    and ``n_active_cuts`` are separate published fields; cross-checking one
    against the other kills a driver that fabricates the iteration count while
    leaving the (correct) cut count alone.
    """
    arrays = _toy_arrays()
    observed = _perturbed_observed(arrays)
    shocks0 = np.asarray(arrays["shocks"], dtype=np.float64)

    persistent = _run_persistent_panel(
        observed, backend, shocks0, (1.0, PSI_ROWGEN)
    )
    cold = persistent[0]
    warm = persistent[1]

    # The warm reevaluate genuinely converged and stayed in-band against a cold
    # rebuild at ψ_rowgen — so its extra passes were real row-gen, not padding.
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

    # Independent cut-growth signal (coverage of the tie): the ψ-step really
    # did force new cuts, so the >1-iteration claim below is exercised.
    cut_growth = warm.n_active_cuts - cold.n_active_cuts
    assert cut_growth > 0, (
        f"RHS row-gen {backend}: ψ={PSI_ROWGEN} did not grow the carried cut"
        f" set (cold {cold.n_active_cuts} -> warm {warm.n_active_cuts}); the"
        " case no longer exercises warm row-gen — retune PSI_ROWGEN"
    )

    # The tie: a converged warm reevaluate that grew its cut set ran >= 2 passes.
    # This bound is derived from the loop/install invariant above and read
    # against n_active_cuts, NOT against the driver's own iteration count — so a
    # fabricated constant `iterations=1` (which leaves cut growth intact) fails
    # here even though it survives the shocks-only panel.
    assert warm.iterations >= 2, (
        f"RHS row-gen {backend}: warm reevaluate grew the cut set by"
        f" {cut_growth} yet published only {warm.iterations} iteration(s) — a"
        " pass that installs a cut has violation > tolerance, so a converged"
        " warm solve that admitted cuts cannot have run a single pass; the"
        " published iteration count is fabricated, not measured"
    )

    # And the warm-start still beats a cold rebuild at ψ_rowgen: even with the
    # extra row-gen the warm reuse runs strictly fewer passes than building from
    # scratch, so the >= 2 above is a real speed win, not a regression to cold.
    assert warm.iterations < q_cold[1], (
        f"RHS row-gen {backend}: warm reevaluate took {warm.iterations}"
        f" iterations, not fewer than the cold rebuild's {q_cold[1]} at"
        f" ψ={PSI_ROWGEN} — the warm-start advantage is gone"
    )


# --- (d) NSlack-only -----------------------------------------------------------


@needs_highs
def test_d_oneslack_rejected_at_cold_fit() -> None:
    """A OneSlack into the persistent driver is rejected (TypeError, fail-closed).

    Validation-level pin of the driver's ``_require_nslack``: OneSlack installs one
    aggregate cut whose RHS depends on the priced joint selection, so a per-cut
    ``set_rhs(ε@ψ)`` rewrite is undefined for it — the driver rejects any
    non-NSlack formulation where it first arrives (the cold fit), before any
    master is built.
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
    """The speed-layer demonstration: a numpy-only golden-section ψ-search.

    Minimise the objective over a ψ-interval with a pure-numpy golden-section
    search (no scipy — scipy is not a dependency). The search
    drives one cold ``fit(ψ0)`` then a ``reevaluate`` per probe on a single
    held master (the speed layer: one cold build + N warm reuses), lands a
    sensible ψ, and the driver's own iteration/cut diagnostics prove exactly
    that decomposition.

    The objective increases monotonically in ψ over the toy panel, so the
    minimiser pins the lower end of the bracket — a deterministic, checkable
    landing.
    """
    arrays = _toy_arrays()
    observed = _perturbed_observed(arrays)
    shocks0 = np.asarray(arrays["shocks"], dtype=np.float64)

    rhs = _Psi0BaselineRhs()
    driver = _make_driver(observed, backend, rhs, arrays)
    calls = {"cold": 0, "warm": 0}
    started = {"flag": False}
    # combrum-observable per-eval diagnostics (not the test's own counters):
    # the iteration count and carried-cut count the driver reports for each
    # probe. These, not `calls`, carry the cold/warm split signal.
    eval_iters: list[int] = []
    eval_cuts: list[int] = []

    def objective(psi: float) -> float:
        # one cold fit at the first probe, a warm reevaluate at every later
        # one — the held-master speed path. The driver guards reuse validity
        # (G1/G2/G3) on every reevaluate; a valid shocks-only ψ passes.
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
    # (i) test-side control-flow self-check: the closure took the cold branch
    # once and the warm branch for every later probe. These counters belong to
    # the test fixture; the package behavior is checked in (ii)/(iii) below.
    assert calls["cold"] == 1
    assert calls["warm"] == total_evals - 1
    assert len(eval_iters) == total_evals
    # (ii) one held master, no rebuild-per-probe: the carried-cut count the
    # driver reports is identical across the whole search. A driver that
    # secretly rebuilt (or dropped cuts) would show a varying count.
    assert eval_cuts == [eval_cuts[0]] * total_evals, (
        f"RHS search (e) {backend}: n_active_cuts varied across the search"
        f" ({eval_cuts}); a single held master must carry the same cut set"
    )
    # (iii) the cold/warm decomposition, read off the driver's own iteration
    # diagnostics rather than the test's counters: exactly one probe did cold
    # row-gen and it was the first. A cold build prices its cuts from scratch
    # (many iterations); each warm reevaluate reuses the converged superset and
    # does ZERO row-gen — so the first eval's iteration count exceeds 1 and every
    # later eval reports exactly 1.
    assert eval_iters[0] > 1, (
        f"RHS search (e) {backend}: first eval reported {eval_iters[0]}"
        " iterations — a cold fit must price its cuts over several row-gen"
        " passes"
    )
    # The warm evals are pinned to the hand-derived single-pass count (== 1), a
    # tight oracle anchored to the shocks-only warm-reuse semantics rather than
    # to the cold count. A warm-start that degraded to eval_iters[0]-1 passes the
    # old `it < eval_iters[0]` bound while losing most of the payoff; the exact
    # `== 1` refuses it. On this ψ-class the RHS-rewritten superset already bounds
    # the ψ-optimum, so each warm reevaluate solves once, prices, admits nothing,
    # and stops (driver.py: one pass, converge on a full sweep with no violation).
    warm_evals = eval_iters[1:]
    assert warm_evals == [1] * (total_evals - 1), (
        f"RHS search (e) {backend}: warm eval iterations {warm_evals} != the"
        f" hand-derived single-pass count {[1] * (total_evals - 1)} — every warm"
        " reevaluate on the shocks-only ψ-class does zero row-gen (one"
        " solve+price+converge pass); a value >1 means the warm-start payoff was"
        " lost, a value <1 means the count was fabricated"
    )
    # A sensible landing: the objective rises with ψ, so the minimiser sits at
    # the bracket floor (within the golden-section resolution after n_steps
    # contractions).
    assert abs(psi_star - 0.8) < 0.05, (
        f"RHS search (e) {backend}: psi_star {psi_star!r} did not land near the"
        " expected ψ-search minimum (the bracket floor)"
    )
