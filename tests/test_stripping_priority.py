"""Stripping priority leg: bounded cut-count at equal estimate.

The slack-stripping correctness gate already ships in ``test_cut_policies.py``.
This file pins the priority leg: SlackStrip holds the peak installed
cut-count over the full run below the unstripped accumulation, while the
estimate is preserved (the published objective is unchanged within band,
the fit converges, and theta does not collapse onto the box bounds).

The claim is empirically delicate. NSlack accumulates one binding cut per
agent per iteration and never retires without a policy, so on a small
degenerate fit almost every cut is still load-bearing at the optimum: strip
it and theta collapses to the box bounds and the fit never converges. A
bounded peak at equal estimate needs a fixture where cuts churn — cuts added
early (theta far off) go deeply slack as theta converges, so a loose
SlackStrip (a high percentile, with the hard cap inactive at this tiny scale)
sheds them without touching the binding set.

The fixtures here are sized into that churning regime (larger toy and QKP
fits, where over half the installed cuts are loose at convergence). The gate
is the strict form — ``peak(strip) < peak(off)`` at equal estimate, at two
sizes per family — with the measured peak-count numbers recorded.
"""

from __future__ import annotations

import numpy as np
import pytest

from _family_oracles import qkp_problem, toy_problem
from _walk import run_walk
from combrum.cut_policies import SlackStrip
from combrum.formulations import NSlack
from _support.families import DEFAULT_SEED, qkp_family, toy_family
from _support.probes import measure
from combrum.masters import gurobi as gurobi_backend
from combrum.transport import SerialTransport

GUROBI_AVAILABLE = gurobi_backend.available()

# Stripping changes the solve path (constraints come and go), so equal
# claims are banded, never bitwise: theta is one set-identified vertex of
# the same optimal face and the objective is the pinned quantity.
needs_gurobi = pytest.mark.skipif(
    not GUROBI_AVAILABLE, reason="gurobipy missing or no environment starts"
)
pytestmark = pytest.mark.slow

# Objective bands. The strip-vs-off reproducibility floor measures ~5e-14
# across all four fixtures (two off-runs are byte-identical), so a 1e-11
# band sits ~200x above genuine solver noise yet catches a sub-1e-9
# retirement-path objective corruption. The same 1e-11 gates the independent
# objective oracle below (its recompute floor is ~1e-13). A loose 1e-9 would
# let a ~2e-10 published-objective drift slip through — the objective is the
# pinned "estimate" quantity, so it carries the tight band.
STRIP_PARITY_BAND = 1e-11
OBJECTIVE_ORACLE_BAND = 1e-11

# The local priority fixtures exercise the percentile leg. The
# hard_threshold is a max-live row count, not a slack magnitude; production's
# 60000-row cap is far above this fixture scale, so the count cap is kept
# inactive here. A tighter percentile strips binding cuts and breaks the fit —
# the whole delicacy of the claim.
STRIP_PERCENTILE = 95.0
STRIP_HARD_THRESHOLD = float("inf")

# Fixtures in the churning regime, two sizes per family so the bounded-count
# win is not a single-size artifact. The small parity-default fixtures are
# deliberately not used: there the cut set barely churns and stripping
# breaks the estimate (the documented failure mode).
TOY_SIZES = ((40, 6), (60, 10))
QKP_SIZES = ((20, 6), (30, 6))

# The full churning fixture set (both families, two sizes each) the priority
# legs gate. Shared by the cut-count, row-freeing, and retirement-path gates so
# all three measure the same runs.
FIXTURES = [("toy", n, m) for n, m in TOY_SIZES] + [
    ("qkp", n, m) for n, m in QKP_SIZES
]

# Peak-RSS is a process-lifetime high-water mark, so measuring OFF-then-STRIP
# in one process makes a strict "<" unsound (STRIP_peak >= OFF_peak always).
# It is kept only as a printed diagnostic; the memory claim is gated on the
# master's own row/object accounting. Margin shared with test_penalty (its O(K)
# quadratic term, our bounded extra cuts — both well under it).
RSS_MARGIN_BYTES = 32 * 1024 * 1024

def _arrays(kind: str, n_obs: int, n_items: int) -> dict[str, np.ndarray]:
    builder = qkp_family if kind == "qkp" else toy_family
    return builder(n_obs, n_items, DEFAULT_SEED)


def _problem(kind: str, arrays: dict[str, np.ndarray]) -> object:
    return (qkp_problem if kind == "qkp" else toy_problem)(arrays)


def _walk(arrays: dict[str, np.ndarray], problem: object, **kw) -> object:
    return run_walk(
        arrays, problem, NSlack, SerialTransport(), backend="gurobi", **kw
    )


def _constr_objects_on_instance(master: object) -> int:
    """Count distinct gurobi ``Constr`` objects reachable from the master.

    Walks the instance's own attributes and recurses through plain
    dict/list/tuple/set/frozenset containers to ANY depth (deduping by object
    id, guarding against container cycles). It never descends into the
    ``gurobipy.Model`` itself — that is not a plain container, so the live
    solver rows are not double-counted — which keeps the honest count equal to
    the installed rows.

    A retirement that removes a row but parks the removed ``Constr`` in a
    persistent per-instance graveyard shows up here as a count above the live
    installed rows even though the gurobi model itself was mutated correctly.
    Because the walk is recursive rather than one level deep, a graveyard
    nested inside a list-of-lists, a dict-of-lists, or any deeper plain
    container is caught just like a flat one.
    """
    import gurobipy

    constr_cls = gurobipy.Constr
    container_types = (list, tuple, set, frozenset)
    seen: set[int] = set()
    visited_containers: set[int] = set()
    stack: list[object] = list(master.__dict__.values())
    while stack:
        value = stack.pop()
        if isinstance(value, constr_cls):
            seen.add(id(value))
        elif isinstance(value, dict):
            if id(value) in visited_containers:
                continue
            visited_containers.add(id(value))
            stack.extend(value.values())
            stack.extend(value.keys())
        elif isinstance(value, container_types):
            if id(value) in visited_containers:
                continue
            visited_containers.add(id(value))
            stack.extend(value)
    return len(seen)


def _assert_retirement_frees_rows(master: object) -> None:
    """Every retired cut must free its gurobi row and drop its python object.

    Called right after a ``remove_cuts`` on the live master. Two leak forms
    the process-wide RSS margin cannot see:

    * skipping ``self._model.remove(constr)`` keeps the retired row in the
      solver — caught by ``NumConstrs`` running above ``n_active_cuts``;
    * graveyarding the removed ``Constr`` objects on the instance keeps them
      alive — caught by the reachable-object count running above
      ``n_active_cuts``.

    On an honest in-place retirement both equal the installed-row count, so
    the master carries exactly the rows its own bookkeeping claims.
    """
    installed = master.n_active_cuts
    master._model.update()
    live_rows = int(master._model.NumConstrs)
    assert live_rows == installed, (
        "retired gurobi rows leaked: model holds"
        f" {live_rows} constraints but only {installed} cuts are installed"
    )
    reachable = _constr_objects_on_instance(master)
    assert reachable == installed, (
        "retired constraint objects leaked: master references"
        f" {reachable} Constr objects but only {installed} cuts are installed"
    )


def _strip_vs_off(kind: str, n_obs: int, n_items: int):
    """Run the fixture with stripping OFF and with the loose SlackStrip.

    Returns ``(off, strip)`` outcomes; both share the fixture, differing
    only in whether the retirement policy is installed.
    """
    arrays = _arrays(kind, n_obs, n_items)
    problem = _problem(kind, arrays)
    off = _walk(arrays, problem)
    strip = _walk(
        arrays,
        problem,
        cut_policy=SlackStrip(
            percentile=STRIP_PERCENTILE, hard_threshold=STRIP_HARD_THRESHOLD
        ),
    )
    return off, strip


def _interior(theta: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> bool:
    # The estimate is preserved only if theta did not collapse onto the box:
    # the stripping failure mode (cuts the fit needs are gone) pins one or more
    # coordinates against a box face, so an interior theta is the signal that
    # only redundant cuts were shed.
    #
    # Check against the fixture's actual per-coordinate (lower, upper) box, not
    # a symmetric +/-THETA_BOUND magnitude bound. The QKP box is asymmetric:
    # alpha (index 0) and lambda (last index) are pinned at >= 0, so their true
    # lower face is 0.0, not -THETA_BOUND. A theta pinned on alpha=0 or lambda=0
    # has small magnitude yet has collapsed onto a real box face; the old
    # abs<THETA_BOUND test reported it "interior". A per-coordinate margin off
    # both faces is the collapse oracle the asymmetric family needs.
    lower, upper = bounds
    theta = np.asarray(theta, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    off_lower = np.all(theta - lower > 1e-6)
    off_upper = np.all(upper - theta > 1e-6)
    return bool(off_lower and off_upper)


def _expected_objective(
    kind: str, arrays: dict[str, np.ndarray], problem: object, theta: np.ndarray
) -> float:
    """Recompute the row-generation master objective at ``theta`` independently.

    The master minimises ``c_theta . theta + sum_a u_coef_a * u_a`` subject to
    ``u_a >= phi_b . theta + eps_b`` for every installed cut, with
    ``c_theta = -sum_a phi_a(observed_a)`` and (for NSlack on these fixtures)
    ``theta_coef == u_coef == 1``. At a converged fit each ``u_a`` saturates at
    ``max_b (phi_b . theta + eps_b)`` — exactly the oracle-priced optimal payoff
    at ``theta``. So the whole objective is rebuilt here from the observed
    features and the pricing oracle, touching no master accessor: an
    independent reference for the published objective, not a second combrum run.
    """
    observed = np.asarray(arrays["observed"])
    n_agents = observed.shape[0]
    c_theta = np.zeros(problem.K, dtype=np.float64)
    for a in range(n_agents):
        c_theta -= np.asarray(
            problem.observed_features(a, observed[a]), dtype=np.float64
        )
    linear = float(c_theta @ np.asarray(theta, dtype=np.float64))
    u_sum = sum(
        float(problem.oracle.price(theta, a).payoff) for a in range(n_agents)
    )
    return linear + u_sum


def _assert_estimate_preserved(
    kind: str,
    n_obs: int,
    n_items: int,
    arrays: dict[str, np.ndarray],
    problem: object,
    off,
    strip,
) -> None:
    """The stripped fit publishes the same estimate as the unstripped run.

    Three independent legs, so an over-strip that quietly changes the answer
    cannot pass by satisfying only one:

    * both runs converge;
    * each run's published objective matches the objective recomputed from
      theta + the pricing oracle (``_expected_objective``) — the substantive
      "estimate" quantity pinned against a reference that never reads a master
      accessor, so a retirement-path ObjVal corruption is caught even when the
      strip-vs-off band would absorb it;
    * strip-vs-off objective agreement inside the tight ``STRIP_PARITY_BAND``;
    * theta stays off the box, measured per-coordinate against the fixture's
      own asymmetric ``(lower, upper)`` box (so the QKP alpha>=0 / lambda>=0
      lower faces are covered, not just the symmetric +/-THETA_BOUND ones).
      This last leg is a genuine collapse oracle only where the *unstripped*
      fit is itself interior (asserted here): on a fixture whose set-identified
      optimum already sits on a box face, interior-strip-theta is not a
      collapse signature, so we do not lean on it.
    """
    bounds = problem.theta_bounds
    assert off.converged and strip.converged
    for tag, outcome in (("off", off), ("strip", strip)):
        expected = _expected_objective(
            kind, arrays, problem, outcome.result.theta_hat
        )
        assert abs(outcome.objective - expected) <= OBJECTIVE_ORACLE_BAND, (
            f"{kind} {n_obs}x{n_items}: {tag} objective {outcome.objective!r}"
            f" disagrees with the price-oracle recompute {expected!r} — the"
            " published estimate is not the row-generation optimum"
        )
    assert abs(strip.objective - off.objective) <= STRIP_PARITY_BAND, (
        f"{kind} {n_obs}x{n_items}: stripping moved the objective"
        f" ({strip.objective!r} vs {off.objective!r}) — the estimate must"
        " be preserved"
    )
    off_interior = _interior(off.result.theta_hat, bounds)
    assert _interior(strip.result.theta_hat, bounds), (
        f"{kind} {n_obs}x{n_items}: stripped theta collapsed onto the box"
        f" bound ({strip.result.theta_hat.tolist()}) — cuts the fit needed"
        " were retired"
    )
    return off_interior


# --------------------------------------------------------------------------
# SlackStrip strictly bounds the peak cut-count at an equal estimate
# --------------------------------------------------------------------------


@needs_gurobi
@pytest.mark.parametrize("kind,n_obs,n_items", FIXTURES)
def test_slack_strip_bounds_peak_cut_count_at_equal_estimate(
    kind: str, n_obs: int, n_items: int
) -> None:
    arrays = _arrays(kind, n_obs, n_items)
    problem = _problem(kind, arrays)
    off, strip = _strip_vs_off(kind, n_obs, n_items)
    delta_theta = float(
        np.max(np.abs(strip.result.theta_hat - off.result.theta_hat))
    )
    print(
        f"\n{kind} {n_obs}x{n_items}: peak_off={off.peak_installed_cuts}"
        f" peak_strip={strip.peak_installed_cuts}"
        f" (d={strip.peak_installed_cuts - off.peak_installed_cuts:+d})"
        f" | iters_off={off.iterations} iters_strip={strip.iterations}"
        f" | dobj={strip.objective - off.objective:+.2e}"
        f" dtheta={delta_theta:.2e}"
        f" max|theta|_strip={np.max(np.abs(strip.result.theta_hat)):.2f}"
        f" max|theta|_off={np.max(np.abs(off.result.theta_hat)):.2f}"
    )
    # The estimate is preserved: the stripped fit converges, lands on the
    # independently recomputed objective, agrees with the off run, and keeps
    # theta off the box.
    off_interior = _assert_estimate_preserved(
        kind, n_obs, n_items, arrays, problem, off, strip
    )
    # The priority win: the peak installed cut-count over the run is
    # strictly below the unstripped accumulation. The loose rule shed only
    # the deeply-slack cuts, so the master never grew as wide.
    assert strip.peak_installed_cuts < off.peak_installed_cuts, (
        f"{kind} {n_obs}x{n_items}: stripping did not bound the peak"
        f" cut-count (off={off.peak_installed_cuts},"
        f" strip={strip.peak_installed_cuts})"
    )
    # The loose p95 rule must actually be loose: retiring only near the top of
    # the looseness distribution, it keeps strictly more of the accumulation
    # than an aggressive low-percentile strip. A misplumbed percentile that
    # ignored the configured value (always stripping deep into the near-binding
    # set) would drive both runs to the same peak and fail this — the interior
    # co-gate above cannot see that over-strip on fixtures whose fit stays
    # robustly interior, so the contrast is the guard. The two policies here
    # are set by this test, not read from combrum, so the inequality is an
    # independent oracle for "the configured percentile governs how much sheds".
    aggressive = _walk(
        arrays,
        problem,
        cut_policy=SlackStrip(percentile=5.0, hard_threshold=STRIP_HARD_THRESHOLD),
    )
    assert strip.peak_installed_cuts > aggressive.peak_installed_cuts, (
        f"{kind} {n_obs}x{n_items}: the loose p{STRIP_PERCENTILE:g} strip held"
        f" peak {strip.peak_installed_cuts}, no higher than an aggressive p5"
        f" strip ({aggressive.peak_installed_cuts}) — the configured percentile"
        " did not govern how deep the retirement cut"
    )
    # The interior co-gate is a real collapse oracle only where the unstripped
    # fit is itself interior; record which fixtures those are so the box-baseline
    # case (off theta already on a set-identified vertex) is never silently the
    # one relied upon.
    if kind == "toy" and n_obs == 40:
        # This fixture's unstripped optimum already sits on a box face
        # (set identification, not collapse), so interior-strip-theta is not a
        # collapse signature here — pinned so a change is noticed.
        assert not off_interior, (
            "toy 40x6 off theta was expected on the box (set-identified"
            f" vertex); got interior {off.result.theta_hat.tolist()}"
        )
    else:
        assert off_interior, (
            f"{kind} {n_obs}x{n_items}: unstripped theta is on the box, so the"
            " interior-strip-theta collapse oracle has no live baseline here"
        )


@needs_gurobi
def test_stripping_peak_reduction_grows_with_fixture_scale() -> None:
    # Distinct cross-scale claim (the per-size ``peak_strip < peak_off`` gate is
    # owned by the parametrized test above): the loose rule sheds a strictly
    # larger SHARE of the installed cuts at the larger size. The churning regime
    # the module docstring describes says a larger fixture carries a larger
    # fraction of deeply-slack cuts at convergence, so the percentile rule
    # retires a strictly larger fraction of the accumulation — not merely a
    # larger absolute count, which a bigger problem yields even under a policy
    # that strips a scale-independent handful (the size-accumulation confound).
    # The invariant is the shed FRACTION reduction/off.peak; sizes per family
    # are listed small-then-large, so fraction[1] must strictly exceed
    # fraction[0]. Derived from the churning-scales property, not from any
    # combrum-emitted count.
    readings: list[str] = []
    for kind, sizes in (("toy", TOY_SIZES), ("qkp", QKP_SIZES)):
        fractions: list[float] = []
        for n_obs, n_items in sizes:
            arrays = _arrays(kind, n_obs, n_items)
            problem = _problem(kind, arrays)
            off, strip = _strip_vs_off(kind, n_obs, n_items)
            # Guard the premise (same estimate) so a monotone pass cannot come
            # from a broken/collapsed fit at either size — objective pinned to
            # the independent price-oracle recompute, not just strip-vs-off.
            _assert_estimate_preserved(
                kind, n_obs, n_items, arrays, problem, off, strip
            )
            reduction = off.peak_installed_cuts - strip.peak_installed_cuts
            assert reduction > 0, (
                f"{kind} {n_obs}x{n_items}: no peak reduction"
                f" (off={off.peak_installed_cuts},"
                f" strip={strip.peak_installed_cuts})"
            )
            # Share of the unstripped accumulation the loose rule shed. Dividing
            # out off.peak removes the confound that a larger problem installs
            # more cuts, so a policy stripping a size-independent amount shows a
            # falling (not rising) fraction across scales.
            fraction = reduction / off.peak_installed_cuts
            fractions.append(fraction)
            readings.append(
                f"{kind} {n_obs}x{n_items}:"
                f" peak_off={off.peak_installed_cuts}"
                f" peak_strip={strip.peak_installed_cuts}"
                f" reduction={reduction} fraction={fraction:.4f}"
            )
        small, large = fractions
        assert large > small, (
            f"{kind}: shed fraction did not grow with fixture scale"
            f" (small={small:.4f}, large={large:.4f}) — the churning regime"
            " should retire a strictly larger share of the accumulation at the"
            " larger size"
        )
    print("\n--- stripping priority (shed fraction grows with scale) ---")
    print("\n".join(readings))


@needs_gurobi
def test_loose_strip_is_deterministic() -> None:
    # The stripped fit is reproducible: the policy is pure and reads only the
    # master's own slacks, so two runs publish byte-identical theta and the same
    # peak count — the priority measurement is deterministic, not flaky.
    arrays = _arrays("toy", *TOY_SIZES[1])
    problem = _problem("toy", arrays)
    policy = SlackStrip(
        percentile=STRIP_PERCENTILE, hard_threshold=STRIP_HARD_THRESHOLD
    )
    first = _walk(arrays, problem, cut_policy=policy)
    second = _walk(arrays, problem, cut_policy=policy)
    # Determinism only means anything about the retirement path if that path
    # actually ran: a no-op purge is trivially reproducible too. Anchor the
    # claim to a genuine strip by running the same fixture unstripped and
    # requiring the policy held a strictly lower peak.
    off = _walk(arrays, problem)
    assert first.peak_installed_cuts < off.peak_installed_cuts, (
        "loose strip retired nothing (peak"
        f" {first.peak_installed_cuts} vs unstripped"
        f" {off.peak_installed_cuts}) — the determinism claim would be about a"
        " no-op purge, not the retirement path it names"
    )
    assert (
        first.result.theta_hat.tobytes() == second.result.theta_hat.tobytes()
    )
    assert first.peak_installed_cuts == second.peak_installed_cuts
    assert first.objective == second.objective


# --------------------------------------------------------------------------
# (b): retirement frees the rows it retires
# --------------------------------------------------------------------------


@needs_gurobi
@pytest.mark.parametrize("kind,n_obs,n_items", FIXTURES)
def test_slack_strip_peak_rss_within_bounded_margin(
    kind: str, n_obs: int, n_items: int, monkeypatch
) -> None:
    # The cut-count win only lowers memory if a retired cut actually leaves the
    # solver. Process-wide ru_maxrss cannot show that: it is a lifetime
    # high-water mark read at call end, and OFF always runs before STRIP in this
    # one process, so strip_peak >= off_peak by construction and any margin
    # passes even if every retired row leaked. So the memory claim is gated on
    # the master's own accounting: after each in-place retirement the live
    # gurobi row count and the reachable Constr-object count must both equal the
    # installed-cut count. A leak (retained solver row, or a graveyard of
    # removed objects) breaks that even while the fit still converges to the same
    # bounded peak. RSS is kept only as a printed diagnostic below.
    arrays = _arrays(kind, n_obs, n_items)
    problem = _problem(kind, arrays)

    orig_remove = gurobi_backend.GurobiMaster.remove_cuts

    def _remove_then_check(self, keys):
        removed = orig_remove(self, keys)
        # Only inspect the model after a retirement that actually shed rows;
        # a no-op call leaves nothing to verify.
        if removed:
            _assert_retirement_frees_rows(self)
        return removed

    monkeypatch.setattr(
        gurobi_backend.GurobiMaster, "remove_cuts", _remove_then_check
    )

    off, off_probe = measure(lambda: _walk(arrays, problem))
    strip, strip_probe = measure(
        lambda: _walk(
            arrays,
            problem,
            cut_policy=SlackStrip(
                percentile=STRIP_PERCENTILE,
                hard_threshold=STRIP_HARD_THRESHOLD,
            ),
        )
    )
    print(
        f"\n{kind} {n_obs}x{n_items} rss:"
        f" off={off_probe.peak_rss_bytes / 1e6:.1f}MB"
        f" strip={strip_probe.peak_rss_bytes / 1e6:.1f}MB"
        f" (d={(strip_probe.peak_rss_bytes - off_probe.peak_rss_bytes) / 1e6:+.1f}MB"
        f" margin={RSS_MARGIN_BYTES / 1e6:.0f}MB)"
        f" | peak_off={off.peak_installed_cuts}"
        f" peak_strip={strip.peak_installed_cuts}"
    )
    # The estimate is preserved on these fixtures; carry the same
    # estimate-preserved oracle the primary gate uses (independent objective
    # recompute + strip-vs-off band + interior theta) so a cut-count "win" here
    # cannot come from a collapsed fit: a policy that strips binding rows still
    # converges to a lower peak, but its theta lands on the box and the
    # objective drifts off the price-oracle value.
    _assert_estimate_preserved(
        kind, n_obs, n_items, arrays, problem, off, strip
    )
    assert strip.peak_installed_cuts < off.peak_installed_cuts
    # A loose sanity bound only: not the memory oracle (see above). ru_maxrss is
    # a lifetime peak, so this cannot fall below off, only be bounded above it.
    assert (
        strip_probe.peak_rss_bytes
        <= off_probe.peak_rss_bytes + RSS_MARGIN_BYTES
    )


def _probe_graveyard_depths(
    master: object, removed_constrs: list
) -> dict[str, int]:
    """Snapshot the reachable-Constr count on the live master under each planted
    graveyard, restoring the master to honest between plants.

    ``removed_constrs`` are the actual ``Constr`` objects the just-completed
    retirement dropped from ``_constrs`` — genuinely-retired rows, so parking
    them back on the instance is a true graveyard leak (their ids are distinct
    from every live row's).

    Returns the count for: no graveyard (``honest``), a flat direct-attribute
    graveyard (``flat``), a one-level dict graveyard (``one_level``), and a
    two-level dict->list->list graveyard (``nested``).
    """
    installed = master.n_active_cuts
    out = {
        "installed": installed,
        "n_removed": len(removed_constrs),
        "honest": _constr_objects_on_instance(master),
    }

    master._gy_flat = list(removed_constrs)
    try:
        out["flat"] = _constr_objects_on_instance(master)
    finally:
        del master._gy_flat

    master._gy_one = {"removed": list(removed_constrs)}
    try:
        out["one_level"] = _constr_objects_on_instance(master)
    finally:
        del master._gy_one

    master._gy_nested = {"removed": [list(removed_constrs)]}
    try:
        out["nested"] = _constr_objects_on_instance(master)
        with pytest.raises(AssertionError, match="constraint objects leaked"):
            _assert_retirement_frees_rows(master)
    finally:
        del master._gy_nested

    out["restored"] = _constr_objects_on_instance(master)
    return out


@needs_gurobi
def test_reachable_constr_count_catches_graveyards_at_any_depth() -> None:
    # Pins the row-freeing memory oracle itself: _constr_objects_on_instance
    # must equal the installed rows on an honest master, and must exceed them
    # the instant genuinely retired Constr objects are parked anywhere on the
    # instance. Probed on the live master during a real retirement, so this
    # exercises the same instance state the row-freeing gate sees mid-run.
    import gurobipy

    result: dict[str, dict[str, int]] = {}
    orig_remove = gurobi_backend.GurobiMaster.remove_cuts

    def _probe_remove(self, keys):
        keys = list(keys)
        doomed = [
            self._constrs[k]
            for k in set(keys)
            if isinstance(self._constrs.get(k), gurobipy.Constr)
        ]
        removed = orig_remove(self, keys)
        if removed >= 3 and "probe" not in result:
            self._model.update()
            result["probe"] = _probe_graveyard_depths(self, doomed[:3])
        return removed

    original = gurobi_backend.GurobiMaster.remove_cuts
    gurobi_backend.GurobiMaster.remove_cuts = _probe_remove
    try:
        arrays = _arrays("toy", *TOY_SIZES[1])
        problem = _problem("toy", arrays)
        _walk(
            arrays,
            problem,
            cut_policy=SlackStrip(
                percentile=STRIP_PERCENTILE, hard_threshold=STRIP_HARD_THRESHOLD
            ),
        )
    finally:
        gurobi_backend.GurobiMaster.remove_cuts = original

    assert "probe" in result, "no retirement shed >=3 rows, so the oracle never ran"
    p = result["probe"]
    installed = p["installed"]
    assert installed > 0
    assert p["n_removed"] == 3
    # Honest master: the recursive count is exactly the installed rows.
    assert p["honest"] == installed
    # Flat and one-level graveyards were already caught by the old scan; the
    # recursive scan must not regress on them.
    assert p["flat"] == installed + 3
    assert p["one_level"] == installed + 3
    # A two-level nested graveyard is caught only because the walk recurses to
    # any depth; a one-level scan would report exactly ``installed`` here.
    assert p["nested"] == installed + 3, (
        "nested Constr graveyard evaded the reachable-object count:"
        f" got {p['nested']}, installed {installed} — a depth-limited scan lets"
        " a list-of-lists graveyard leak through"
    )
    # Pulling the graveyard restores the honest equality: the count tracks the
    # real leak, it is not stuck high.
    assert p["restored"] == installed


# --------------------------------------------------------------------------
# (c): wall-clock within a soft sanity ceiling
# --------------------------------------------------------------------------


@needs_gurobi
@pytest.mark.parametrize("kind,n_obs,n_items", FIXTURES)
def test_slack_strip_retirement_stays_on_in_place_path(
    kind: str, n_obs: int, n_items: int, monkeypatch
) -> None:
    # The wall claim, as a deterministic structural gate across the whole
    # fixture set (an 8x-of-OFF wall ceiling passes even at many-x regressions,
    # so it bounds nothing). What actually governs per-iteration wall is the
    # retirement PATH: in-place ``remove_cuts`` (O(retired), warm basis kept) vs
    # a cold ``reinstall`` (dispose + full rebuild). An 8x-of-OFF wall ceiling
    # passes even at many-x regressions, so it bounds nothing; a tight wall
    # bound would only flake. Spy on both paths instead — a stripping walk must
    # route every retirement through remove_cuts and never cold-rebuild, and
    # must actually retire rows.
    calls = {"remove_cuts": 0, "rows_removed": 0, "reinstall": 0}
    orig_remove = gurobi_backend.GurobiMaster.remove_cuts
    orig_reinstall = gurobi_backend.GurobiMaster.reinstall

    def _counting_remove(self, keys):
        keys = list(keys)
        calls["remove_cuts"] += 1
        removed = orig_remove(self, keys)
        calls["rows_removed"] += removed
        return removed

    def _counting_reinstall(self, rows):
        calls["reinstall"] += 1
        return orig_reinstall(self, rows)

    monkeypatch.setattr(
        gurobi_backend.GurobiMaster, "remove_cuts", _counting_remove
    )
    monkeypatch.setattr(
        gurobi_backend.GurobiMaster, "reinstall", _counting_reinstall
    )
    off, strip = _strip_vs_off(kind, n_obs, n_items)
    print(
        f"\n{kind} {n_obs}x{n_items} retire-path:"
        f" remove_cuts_calls={calls['remove_cuts']}"
        f" rows_removed={calls['rows_removed']}"
        f" reinstall_calls={calls['reinstall']}"
    )
    assert off.converged and strip.converged
    # Retirement actually happened in place: rows were shed via remove_cuts.
    assert calls["rows_removed"] > 0, (
        f"{kind} {n_obs}x{n_items}: SlackStrip retired nothing via"
        " remove_cuts — the in-place path was never exercised"
    )
    # And never through the cold-rebuild path (warm-start is the only
    # legitimate reinstall, and no walk here warm-starts).
    assert calls["reinstall"] == 0, (
        f"{kind} {n_obs}x{n_items}: SlackStrip cold-rebuilt the master"
        f" (reinstall x{calls['reinstall']}) instead of in-place remove_cuts"
    )
    # The rows shed cannot exceed the reduction in peak width the run realized;
    # ties the structural path back to the measured cut-count win.
    assert strip.peak_installed_cuts < off.peak_installed_cuts
