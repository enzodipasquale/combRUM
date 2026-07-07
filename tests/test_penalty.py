"""Proximal penalty: decay-to-pure-LP correctness and the
fewer-iterations priority win it exists for.

Both legs drive the penalty through the test-local walk's decay schedule
(``qp_weight``/``decay``/``penalty_ref``): the weight decays linearly to
exactly zero over ``decay`` iterations, so the terminating solve is always
a pure LP whose duals are true LP duals. The schedule lives in the walk —
not in a shipped driver — because the production penalty driver does not
exist yet; ``MasterBackend.set_penalty`` is the only contract surface it
touches.

The correctness leg uses a deliberately degenerate fixture (one theta
coordinate is unpinned at the optimum, so the optimal face is a flat
continuum): the penalty selects a determinate point on that face at no
change to the unpenalised objective, the terminating solve is a verified
pure LP, and HiGHS — which has no quadratic support — hard-errors rather
than approximate. The priority leg pins why the penalty exists: on a
slow-converging fixture it reaches the same objective in strictly fewer
row-generation iterations. That iteration count is the deterministic primary
signal; wall-clock and RSS are soft, generously banded guards.
"""

from __future__ import annotations

import numpy as np
import pytest

from _family_oracles import toy_problem
from _walk import run_walk
from combrum.formulations import NSlack
from _support.families import DEFAULT_SEED, toy_family
from _support.probes import measure
from combrum.masters import gurobi as gurobi_backend
from combrum.masters import highs as highs_backend
from combrum.masters import make_master
from combrum.transport import LocalCluster, SerialTransport

GUROBI_AVAILABLE = gurobi_backend.available()
HIGHS_AVAILABLE = highs_backend.available()

# The penalty is a quadratic objective term, so every penalty gate needs a
# backend with native quadratic support: gurobi runs them, highs is only
# ever exercised for its by-design hard error.
needs_gurobi = pytest.mark.skipif(
    not GUROBI_AVAILABLE, reason="gurobipy missing or no environment starts"
)
needs_highs = pytest.mark.skipif(
    not HIGHS_AVAILABLE, reason="highspy missing or broken"
)
pytestmark = pytest.mark.slow

# Parity band shared with the rest of the suite: equal-objective claims
# are banded at 1e-9, never asserted bitwise (the penalty changes the
# solve path, so float paths differ by ~1e-13 even at an identical optimum).
PARITY_BAND = 1e-9
FLAT_FACE_RAW_OBJECTIVE = -1.0

# Decay schedule for the flat-face fixture: weight 10 at the start, decaying
# linearly to 0 over `decay` iterations, so the terminating solve is a pure LP.
FLAT_QP_WEIGHT = 10.0
FLAT_DECAY = 3
FLAT_REF = "static"


# --------------------------------------------------------------------------
# Correctness fixture: a flat (degenerate) optimal face
# --------------------------------------------------------------------------


def flat_face_arrays() -> dict[str, np.ndarray]:
    """A 2-item toy whose optimum leaves item 1's theta coordinate free.

    Construction (all ``r = +1``, so the toy demand rule is "item k chosen
    iff ``theta_k + nu[a, k] > 0``"):

    - Item 0 is tightly identified. Agents 0 and 2 select it
      (``nu = -0.5`` -> needs ``theta_0 > 0.5``); agent 1 does not
      (``nu = -3.0`` -> needs ``theta_0 <= 3.0``). The optimal interval
      for ``theta_0`` is the single edge point the master lands on,
      ``0.5``.
    - Item 1 is selected by no agent (``nu = -8.0`` everywhere ->
      ``observed = False`` is rationalised by any ``theta_1 <= 8``). With
      the box lower bound at ``-10`` the optimal interval for ``theta_1``
      is the whole ``[-10, 8]`` — a continuum. And because no
      agent selects item 1, ``c_theta[1] = 0`` and every installed cut has
      ``phi[1] = 0``, so nothing in the master pins ``theta_1``.

    The master objective is therefore exactly constant across ``theta_1``
    in that interval: a genuinely flat optimal face. A pure-LP solve lands
    on one vertex of it; a proximal penalty selects a determinate point of
    it at no objective cost. ``theta_true = [1.0, 0.0]`` sits strictly
    inside, so the data is rationalisable (the walk converges).
    """
    r = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    nu = np.array([[-0.5, -8.0], [-3.0, -8.0], [-0.5, -8.0]])
    observed = np.array(
        [[True, False], [False, False], [True, False]]
    )
    arrays = {
        "observables": r,
        "shocks": nu.reshape(3, 1, 2),
        "observed": observed,
        "theta_true": np.array([1.0, 0.0]),
    }
    for arr in arrays.values():
        arr.setflags(write=False)
    return arrays


def _flat_walk(transport: object, arrays: dict[str, np.ndarray], **penalty):
    return run_walk(
        arrays,
        toy_problem(arrays),
        NSlack,
        transport,
        backend="gurobi",
        **penalty,
    )


def _fixture_c_theta(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """The master's linear theta objective, rebuilt off the fixture arrays.

    The observed-bundle phi total enters as the negated linear theta cost;
    counting it here (independent of the master builder) gives tests an
    oracle for c_theta.
    """
    problem = toy_problem(arrays)
    observed = arrays["observed"]
    c_theta = np.zeros(problem.K, dtype=np.float64)
    for a in range(observed.shape[0]):
        phi_obs = problem.observed_features(a, observed[a])
        c_theta -= np.asarray(phi_obs, dtype=np.float64)
    return c_theta


def _fresh_master(
    arrays: dict[str, np.ndarray],
    box: tuple[np.ndarray, np.ndarray] | None = None,
):
    """A pure-LP master carrying the fixture's linear objective only.

    The walk builds c_theta the same way (the observed-bundle phi total
    enters as the linear theta objective); rebuilding it here lets a test
    install a known cut set and solve the same relaxation two ways. Pass
    ``box`` to override the fixture's default theta bounds — a tightened box
    lets a test drive a priced coordinate onto a bound with a nonzero
    reduced cost.
    """
    problem = toy_problem(arrays)
    c_theta = _fixture_c_theta(arrays)
    return make_master(
        problem.K,
        problem.theta_bounds if box is None else box,
        c_theta,
        lambda agent_id: 1.0,
        backend="gurobi",
        # Mirror the walk's warm-started primal simplex, so the pure-LP
        # vertex this test observes is the one the walk would publish.
        params={"Method": 0, "LPWarmStart": 2},
    )


# --------------------------------------------------------------------------
# (a): the optimal face is flat, and a proximal ref selects a point
# --------------------------------------------------------------------------


@needs_gurobi
def test_flat_face_is_degenerate_and_penalty_selects_a_determinate_point() -> (
    None
):
    # Solve the same converged relaxation two ways — pure LP vs a small
    # proximal penalty aimed elsewhere on the face — and show they return
    # distinct theta at an identical linear objective: a genuinely flat face.
    arrays = flat_face_arrays()
    converged = _flat_walk(SerialTransport(), arrays)
    assert converged.converged
    active = converged.result.active_set
    assert active is not None and len(active) > 0
    # Every installed cut touches only theta_0 — item 1 is never priced —
    # which is exactly why theta_1 is left free at the optimum.
    assert all(float(row.phi[1]) == 0.0 for row in active)

    with _fresh_master(arrays) as lp_master:
        lp_master.add_cuts(active)
        lp_master.solve()
        theta_lp = lp_master.theta()
        lp_value = lp_master.objective()

    # Aim the proximal reference at a point of the free interval far from the
    # LP vertex; a small weight only breaks the tie within the flat face, so
    # the linear objective is unchanged while theta moves.
    free_target = np.array([float(theta_lp[0]), 5.0], dtype=np.float64)
    weight = 0.01
    with _fresh_master(arrays) as prox_master:
        prox_master.add_cuts(active)
        prox_master.set_penalty(free_target, weight)
        prox_master.solve()
        theta_prox = prox_master.theta()
        # Strip the penalty term back off to recover the linear objective
        # the LP would report at this point.
        penalty_term = weight * float(((theta_prox - free_target) ** 2).sum())
        prox_lp_value = prox_master.objective() - penalty_term

    # The vertex moved (the face has positive width)...
    assert not np.array_equal(theta_lp, theta_prox)
    assert abs(float(theta_prox[1]) - float(theta_lp[1])) > 1.0
    # ...the proximal reference pulled the free coordinate to its determinate
    # target (it lies inside the free interval, so the penalty reaches it)...
    assert abs(float(theta_prox[1]) - 5.0) <= 1e-6
    # ...and the linear objective is identical: moving along the face costs
    # nothing, so the penalty changes the selected point, not the value.
    assert abs(prox_lp_value - lp_value) <= PARITY_BAND


# --------------------------------------------------------------------------
# (b): the decay walk — unique answer, pure-LP terminating solve, equal obj
# --------------------------------------------------------------------------


@needs_gurobi
def test_penalty_walk_is_unique_pure_lp_and_objective_matches_no_penalty() -> (
    None
):
    arrays = flat_face_arrays()
    no_penalty = _flat_walk(SerialTransport(), arrays)
    assert no_penalty.converged

    def penalised() -> object:
        return _flat_walk(
            SerialTransport(),
            arrays,
            qp_weight=FLAT_QP_WEIGHT,
            decay=FLAT_DECAY,
            penalty_ref=FLAT_REF,
        )

    first = penalised()
    second = penalised()
    assert first.converged

    # Determinism: the penalised fit is bitwise reproducible — the penalty
    # adds no nondeterminism, so two runs agree to the byte.
    assert first.result.theta_hat.tobytes() == second.result.theta_hat.tobytes()
    assert first.objective == second.objective

    # The terminating solve is a pure LP: the weight finished decaying to 0
    # before convergence was accepted, so the published theta carries no
    # residual quadratic term (and the duals below are therefore valid).
    assert first.final_penalty_weight == 0.0

    # The penalty changes the path, not the optimal value: the penalised
    # and unpenalised fits reach the same unpenalised optimum.
    assert abs(first.objective - no_penalty.objective) <= PARITY_BAND
    # The published value is the raw row-generation master objective; this
    # fixture's unpenalised LP optimum is -1.0.
    assert first.objective == pytest.approx(
        FLAT_FACE_RAW_OBJECTIVE,
        abs=PARITY_BAND,
    )


# --------------------------------------------------------------------------
# (c): the terminating LP duals are valid (nonneg + complementary slack)
# --------------------------------------------------------------------------


@needs_gurobi
def test_penalty_walk_terminating_duals_are_valid_lp_duals() -> None:
    # The point of reverting to a pure LP for the final solve: its duals
    # are true LP duals. Validity is checked structurally — nonnegativity
    # and complementary slackness against the installed rows — not by
    # bytewise equality with the no-penalty fit, because the degenerate
    # face lets the two solves sit on different vertices with different
    # bases (that divergence is exactly what the fixture is built to show).
    arrays = flat_face_arrays()
    outcome = _flat_walk(
        SerialTransport(),
        arrays,
        qp_weight=FLAT_QP_WEIGHT,
        decay=FLAT_DECAY,
        penalty_ref=FLAT_REF,
    )
    assert outcome.converged
    assert outcome.final_penalty_weight == 0.0

    result = outcome.result
    theta = result.theta_hat
    dual = result.dual
    pi_by_key = {
        (row.agent_id, row.bundle_key): float(pi)
        for row, pi in zip(result.active_set, dual.pis)
    }
    for row in result.active_set:
        pi = pi_by_key[(row.agent_id, row.bundle_key)]
        # Inequality-constraint multipliers of a minimisation are >= 0.
        assert pi >= -PARITY_BAND
        # Row slack at the published theta: u_a - (phi.theta + eps) >= 0.
        u_a = float(result.slack[row.agent_id])
        slack = u_a - float(row.phi @ theta + row.epsilon)
        assert slack >= -PARITY_BAND
        # Complementary slackness: a positive dual forces a binding row.
        assert abs(pi * slack) <= PARITY_BAND
    # Box reduced costs are finite wherever theta sits on a bound (the free
    # coordinate sits at a bound with a degenerate zero reduced cost).
    for coordinate, value in dual.bound_duals.items():
        assert 0 <= coordinate < theta.shape[0]
        assert np.isfinite(value)

    # --- non-degeneracy + KKT stationarity (the signal) --------------------
    # The checks above are all vacuously satisfied by an all-zero dual
    # vector (0 >= -band; pi*slack == 0 for any slack; the slack loop is a
    # pure primal recomputation), so an accessor that silently returns zeros
    # would pass. Pin the dual to the fixture's actual LP optimum instead.
    #
    # c_theta is the observed-bundle feature total: with r == 1 everywhere,
    # c_theta[k] == -(number of agents whose observed bundle selects item k).
    # Counted directly off the fixture arrays here, independent of the master
    # builder that combrum runs.
    observed = arrays["observed"]
    n_agents_selecting = observed.astype(np.int64).sum(axis=0)
    c_theta = -n_agents_selecting.astype(np.float64)
    assert c_theta[0] == -2.0  # items 0/2 select item 0
    assert c_theta[1] == 0.0  # no agent selects item 1

    # theta_0 is the identified coordinate; it lands strictly inside the box,
    # so its bound reduced cost is zero and stationarity reads off the cut
    # duals alone. theta_1 is the free coordinate that sits on a bound.
    lower, upper = -10.0, 10.0
    assert lower + 1e-6 < float(theta[0]) < upper - 1e-6

    # The optimum is genuinely dual-active: at least one row carries strictly
    # positive mass. The all-zero dual vector is exactly the invalid case
    # this test exists to reject.
    assert any(pi > PARITY_BAND for pi in dual.pis)

    # KKT stationarity on the interior priced coordinate: with the box
    # reduced cost zero there, the cut duals must offset the linear cost,
    # sum_r pi_r * phi_r[0] == -c_theta[0] == 2.0. The all-zero vector gives
    # 0 != 2 and fails. Derived from the fixture's c_theta, not the solver.
    stationarity_theta0 = sum(
        pi_by_key[(row.agent_id, row.bundle_key)] * float(row.phi[0])
        for row in result.active_set
    )
    assert abs(stationarity_theta0 - (-c_theta[0])) <= PARITY_BAND
    # The free coordinate is unpriced (every phi[1] == 0), so its stationarity
    # is trivially 0 == -c_theta[1] == 0 regardless of the duals.
    stationarity_theta1 = sum(
        pi_by_key[(row.agent_id, row.bundle_key)] * float(row.phi[1])
        for row in result.active_set
    )
    assert abs(stationarity_theta1 - (-c_theta[1])) <= PARITY_BAND

    # Bound-dual value correctness (not just finiteness). A box reduced cost is
    # the KKT stationarity residual the active bound must carry:
    #   RC[k] == c_theta[k] - sum_r pi_r * phi_r[k].
    # For the free coordinate (unpriced: c_theta[1] == 0 and every phi[1] == 0)
    # that residual is exactly 0 — a degenerate zero reduced cost. Both the
    # target (0.0) and the residual formula come off the fixture's own c_theta
    # and cut duals, never from the master's reduced-cost accessor, so a
    # _bound_duals_now that publishes a wrong-but-finite RC is rejected here.
    for coordinate, value in dual.bound_duals.items():
        residual = c_theta[coordinate] - sum(
            pi_by_key[(row.agent_id, row.bundle_key)] * float(row.phi[coordinate])
            for row in result.active_set
        )
        assert value == pytest.approx(residual, abs=PARITY_BAND)
    # The one active bound here is the free coordinate at its lower bound, whose
    # reduced cost is degenerately zero; pin that concrete value directly.
    assert dual.bound_duals[1] == pytest.approx(0.0, abs=PARITY_BAND)

    # --- nonzero bound reduced cost: value-correctness with signal ---------
    # The walk's own optimum only ever puts a coordinate on a bound with a
    # DEGENERATE zero reduced cost (theta_1, unpriced), so every check above
    # is satisfied by a _bound_duals_now that returns 0.0 for every bound
    # coordinate. Drive the priced coordinate theta_0 onto a bound with a
    # genuinely nonzero reduced cost and pin the whole bound_duals dict off
    # the same fixture residual, so a constant-0 (or otherwise wrong) box
    # reduced cost is caught here.
    #
    # Re-solving the converged relaxation with theta_0's box upper bound
    # tightened to 0.2 (below the cut floor 0.5) pins theta_0 to its upper
    # bound: the cuts all go slack (every cut dual 0), so stationarity puts
    # the full linear cost onto the bound, RC[0] == c_theta[0] == -2.
    tight_box = (
        np.array([-10.0, -10.0]),
        np.array([0.2, 10.0]),
    )
    with _fresh_master(arrays, box=tight_box) as bound_probe:
        bound_probe.add_cuts(result.active_set)
        bound_probe.solve()
        probe_theta = bound_probe.theta()
        probe_bound_duals = bound_probe.bound_duals()
        probe_cut_duals = bound_probe.dual_values()

    # theta_0 is pinned to the tightened upper bound; theta_1 still sits on
    # its lower bound. Both are box-active, so bound_duals carries exactly
    # {0, 1} — pin the key set, not just membership.
    assert abs(float(probe_theta[0]) - 0.2) <= 1e-6
    assert abs(float(probe_theta[1]) - (-10.0)) <= 1e-6
    assert set(probe_bound_duals) == {0, 1}
    # Every cut is slack at this pinned point, so all cut duals vanish and
    # the residual oracle reduces to c_theta alone — an independent hand
    # value of (-2.0, 0.0), never read back off the master's accessor.
    for pi in probe_cut_duals.values():
        assert abs(pi) <= PARITY_BAND
    expected_bound_duals = {
        k: c_theta[k]
        - sum(
            probe_cut_duals[(row.agent_id, row.bundle_key)] * float(row.phi[k])
            for row in result.active_set
        )
        for k in probe_bound_duals
    }
    assert expected_bound_duals[0] == pytest.approx(-2.0, abs=PARITY_BAND)
    assert expected_bound_duals[1] == pytest.approx(0.0, abs=PARITY_BAND)
    # Pin the FULL dict wholesale: exact keys and every value against the
    # independent residual. A constant-0 accessor (RC[0] -> 0), a scaled one
    # (RC[0] -> -4), or a dropped/spurious key all fail here — not just the
    # one named failure.
    assert probe_bound_duals == pytest.approx(expected_bound_duals, abs=PARITY_BAND)
    # And the load-bearing single value: theta_0's reduced cost is genuinely
    # nonzero, so a _bound_duals_now that silently publishes 0.0 dies here.
    assert probe_bound_duals[0] == pytest.approx(-2.0, abs=PARITY_BAND)
    assert abs(probe_bound_duals[0]) > 1.0


# --------------------------------------------------------------------------
# (d): HiGHS has no quadratic support — set_penalty(weight>0) hard-errors
# --------------------------------------------------------------------------


@needs_highs
def test_highs_penalty_hard_errors_no_silent_approximation() -> None:
    # HiGHS cannot host the quadratic term, so the walk's first
    # set_penalty(weight>0) must raise NotImplementedError rather than
    # silently approximate — an approximated objective would report duals of
    # a different problem than the caller thinks was solved.
    arrays = flat_face_arrays()
    with pytest.raises(NotImplementedError):
        run_walk(
            arrays,
            toy_problem(arrays),
            NSlack,
            SerialTransport(),
            backend="highs",
            qp_weight=FLAT_QP_WEIGHT,
            decay=FLAT_DECAY,
            penalty_ref=FLAT_REF,
        )


# --------------------------------------------------------------------------
# (e): the penalised fit is bitwise rank-invariant (serial vs cluster)
# --------------------------------------------------------------------------


@needs_gurobi
@pytest.mark.parametrize("size", [2])
def test_penalty_walk_rank_invariant_bitwise(size: int) -> None:
    # The penalty is a root-only objective edit (the master lives on rank 0
    # alone), so it adds no cross-rank reduction: the published theta and
    # objective must come back bitwise identical whether the agents run
    # serially or sharded across a fake cluster.
    arrays = flat_face_arrays()
    penalty = dict(
        qp_weight=FLAT_QP_WEIGHT, decay=FLAT_DECAY, penalty_ref=FLAT_REF
    )
    serial = _flat_walk(SerialTransport(), arrays, **penalty)
    results = LocalCluster(size).run(
        lambda transport: _flat_walk(transport, arrays, **penalty)
    )
    assert len(results) == size
    for outcome in results:
        assert (
            outcome.result.theta_hat.tobytes()
            == serial.result.theta_hat.tobytes()
        )
        assert outcome.objective == serial.objective
        assert outcome.iterations == serial.iterations
        assert outcome.final_penalty_weight == serial.final_penalty_weight


# --------------------------------------------------------------------------
# Priority fixture + config: a slow-converging fit where the penalty wins
# --------------------------------------------------------------------------

# Slow-convergence fixture sizes. The small toy needs visibly many
# row-generation iterations (the modular family pins theta agent-by-agent),
# and the penalty win holds at >= 2 sizes — it must not vanish at scale.
G5_SIZES = (12, 24)
G5_N_ITEMS = 8
# The penalty schedule that wins across the swept sizes (selected from the
# dynamic-vs-static measurement below; static is the stronger performer on
# this family). qp_weight=1 is a light proximal regulariser; decay=3 hands
# the tail back to a pure LP early.
G5_QP_WEIGHT = 1.0
G5_DECAY = 3
G5_REF = "static"

# Wall-clock is a loose sanity ceiling, not a regression band — at these
# millisecond fixture sizes the suite already flakes on timing under load,
# and the iteration win is the real claim. The LP baseline is floored before
# scaling so a sub-millisecond LP fit can't make the ratio explode on a
# single QP-side scheduler hiccup.
WALL_SANITY_FACTOR = 8.0
WALL_FLOOR_SECONDS = 0.05
# RSS is monotone (a process-lifetime high-water mark) and dominated by the
# Python+gurobi baseline; the penalty adds only the O(K) quadratic term, so
# a few-MB ceiling over the LP run's peak is generous and structural.
RSS_MARGIN_BYTES = 32 * 1024 * 1024


def _slow_arrays(n_obs: int) -> dict[str, np.ndarray]:
    return toy_family(n_obs, G5_N_ITEMS, DEFAULT_SEED)


def _slow_walk(arrays: dict[str, np.ndarray], **penalty):
    return run_walk(
        arrays,
        toy_problem(arrays),
        NSlack,
        SerialTransport(),
        backend="gurobi",
        **penalty,
    )


def _lp_and_qp(arrays: dict[str, np.ndarray], ref: str = G5_REF):
    lp = _slow_walk(arrays)
    qp = _slow_walk(
        arrays, qp_weight=G5_QP_WEIGHT, decay=G5_DECAY, penalty_ref=ref
    )
    return lp, qp


# --------------------------------------------------------------------------
# (a): strictly fewer iterations at equal objective — the priority win
# --------------------------------------------------------------------------


@needs_gurobi
@pytest.mark.parametrize("n_obs", G5_SIZES)
def test_penalty_converges_in_fewer_iterations_at_equal_objective(
    n_obs: int,
) -> None:
    # The primary, deterministic signal: the penalised fit reaches the same
    # objective in strictly fewer row-generation iterations. This is an
    # iteration count (not wall-clock), so it cannot flake under load.
    arrays = _slow_arrays(n_obs)
    lp, qp = _lp_and_qp(arrays)
    assert lp.converged and qp.converged
    assert qp.final_penalty_weight == 0.0
    assert qp.iterations < lp.iterations, (
        f"n_obs={n_obs}: penalty did not save iterations"
        f" (lp={lp.iterations}, qp={qp.iterations})"
    )
    assert abs(qp.objective - lp.objective) <= PARITY_BAND


@needs_gurobi
def test_penalty_iteration_win_holds_across_scales() -> None:
    # Cross-size *aggregate* invariant — deliberately distinct from the
    # per-size win asserted by test_penalty_converges_in_fewer_iterations_...
    # (which already runs at every swept size via @parametrize). What that
    # parametrization cannot express is that the sweep, taken as a whole, is
    # a real multi-scale probe with no counterexample size: this test pins
    # those aggregate facts.
    savings: list[int] = []
    readings: list[str] = []
    for n_obs in G5_SIZES:
        arrays = _slow_arrays(n_obs)
        lp, qp = _lp_and_qp(arrays)
        assert lp.converged and qp.converged
        # The terminating solve is a pure LP at every scale (weight decayed
        # to zero before convergence was accepted) — the parametrized win
        # test checks this too, but the "across scales" claim is empty
        # unless it holds for the whole sweep, so it is re-pinned per size.
        assert qp.final_penalty_weight == 0.0
        assert abs(qp.objective - lp.objective) <= PARITY_BAND
        savings.append(lp.iterations - qp.iterations)
        readings.append(
            f"n_obs={n_obs}: lp_iters={lp.iterations}"
            f" qp_iters={qp.iterations} dobj={qp.objective - lp.objective:+.2e}"
        )
    print("\n".join(readings))

    # The "across scales" premise itself: the sweep must genuinely span more
    # than one fixture size, or the claim is unexercised. Guards a silent
    # collapse of G5_SIZES to a single element.
    assert len(savings) == len(G5_SIZES) >= 2
    # The aggregate win: *every* swept size is a strict win, stated once over
    # the whole sweep. min(savings) >= 1 is the definition of "no counter-
    # example size", derived from the invariant this test's name claims — not
    # from any recorded iteration count.
    assert min(savings) >= 1, f"a size lost the iteration win: {readings}"


# --------------------------------------------------------------------------
# (b): wall-clock (soft) and RSS (generous) guards
# --------------------------------------------------------------------------


@needs_gurobi
@pytest.mark.parametrize("n_obs", G5_SIZES)
def test_penalty_wall_clock_within_soft_sanity_ceiling(n_obs: int) -> None:
    # soft guard: the penalised fit must not run away — it should stay
    # within a generous multiple of the LP fit's wall clock. Deliberately
    # loose (the iteration win is the real claim); a tight timing band at
    # millisecond scale is the suite's known flake, so it is not used here.
    arrays = _slow_arrays(n_obs)
    lp, lp_probe = measure(lambda: _slow_walk(arrays))
    qp, qp_probe = measure(
        lambda: _slow_walk(
            arrays,
            qp_weight=G5_QP_WEIGHT,
            decay=G5_DECAY,
            penalty_ref=G5_REF,
        )
    )
    ceiling = max(lp_probe.wall_seconds, WALL_FLOOR_SECONDS) * WALL_SANITY_FACTOR
    assert qp_probe.wall_seconds <= ceiling
    # The ceiling alone can't tell a working penalty from a dead one: a
    # no-op set_penalty leaves the walk on the pure-LP path, which still
    # converges within this band. Pin the effect the penalty exists for —
    # a strict iteration win at a matched objective — so a disabled penalty
    # (qp.iterations == lp.iterations) fails here, not just silently drifts.
    assert lp.converged and qp.converged
    assert qp.final_penalty_weight == 0.0
    assert abs(qp.objective - lp.objective) <= PARITY_BAND
    assert qp.iterations < lp.iterations


@needs_gurobi
@pytest.mark.parametrize("n_obs", G5_SIZES)
def test_penalty_peak_rss_within_bounded_margin(n_obs: int) -> None:
    # The penalty adds only the O(K) quadratic term — no O(n_agents) blow-up
    # — so the penalised fit's peak RSS stays within a bounded margin of the
    # LP fit's. RSS is a lifetime high-water mark, so the LP run is measured
    # first and the margin is generous; this gates structure, not a tight
    # number.
    arrays = _slow_arrays(n_obs)
    lp, lp_probe = measure(lambda: _slow_walk(arrays))
    qp, qp_probe = measure(
        lambda: _slow_walk(
            arrays,
            qp_weight=G5_QP_WEIGHT,
            decay=G5_DECAY,
            penalty_ref=G5_REF,
        )
    )
    assert (
        qp_probe.peak_rss_bytes
        <= lp_probe.peak_rss_bytes + RSS_MARGIN_BYTES
    )
    # RSS is a monotone lifetime high-water mark and the LP run is measured
    # first, so qp_rss >= lp_rss is structural: this ceiling passes even when
    # set_penalty is a dead no-op that saves nothing. Pin the penalty's actual
    # effect — a strict iteration win at a matched objective — so a disabled
    # penalty (qp.iterations == lp.iterations) fails here rather than sliding
    # under a margin no fixture-scale penalty bug can breach.
    assert lp.converged and qp.converged
    assert qp.final_penalty_weight == 0.0
    assert abs(qp.objective - lp.objective) <= PARITY_BAND
    assert qp.iterations < lp.iterations


# --------------------------------------------------------------------------
# Dynamic-vs-static comparison: the data that decides the default ref
# --------------------------------------------------------------------------


@needs_gurobi
def test_dynamic_vs_static_reference_comparison_data() -> None:
    # Not a winner-picking gate: it runs the same slow-convergence fixtures
    # under both proximal references and records iteration counts and final
    # objectives, so the project owner can ratify the default ref from
    # measured evidence. Both must hit the same objective (within band) and
    # terminate on a pure LP; which one saves more iterations is reported,
    # never hard-coded.
    readings: list[str] = []
    # Per-size flags backing the three ratification invariants below. The
    # comparison is only meaningful evidence if the penalty actually did
    # something and the two references actually differ; a no-op or
    # ref-ignoring set_penalty makes the recorded counts fiction.
    static_wins = 0
    refs_differ = 0
    for n_obs in G5_SIZES:
        arrays = _slow_arrays(n_obs)
        lp = _slow_walk(arrays)
        static = _slow_walk(
            arrays,
            qp_weight=G5_QP_WEIGHT,
            decay=G5_DECAY,
            penalty_ref="static",
        )
        dynamic = _slow_walk(
            arrays,
            qp_weight=G5_QP_WEIGHT,
            decay=G5_DECAY,
            penalty_ref="dynamic",
        )
        for label, run in (("static", static), ("dynamic", dynamic)):
            assert run.converged
            assert run.final_penalty_weight == 0.0
            assert abs(run.objective - lp.objective) <= PARITY_BAND
        # The static ref — the one this file ratifies as the default — must
        # save iterations at every swept size, or the "penalty helps" premise
        # the ratification data rests on is false. A dead-no-op set_penalty
        # leaves the walk on the pure-LP path (static == lp), which fails here.
        assert static.iterations < lp.iterations, (
            f"n_obs={n_obs}: static ref saved no iterations"
            f" (lp={lp.iterations}, static={static.iterations})"
        )
        # The dynamic ref must never do worse than the plain LP: a proximal
        # anchor cannot legitimately inflate the row-generation count past the
        # unpenalised path.
        assert dynamic.iterations <= lp.iterations, (
            f"n_obs={n_obs}: dynamic ref lost to lp"
            f" (lp={lp.iterations}, dynamic={dynamic.iterations})"
        )
        # The comparison the test exists to make is only real if the two
        # references produce measurably different behaviour. A set_penalty that
        # ignores its ref argument (e.g. always anchors at the origin) collapses
        # static and dynamic to the same count — reject that here.
        assert static.iterations != dynamic.iterations, (
            f"n_obs={n_obs}: static and dynamic refs behaved identically"
            f" (both {static.iterations}) — the ref argument had no effect"
        )
        static_wins += 1
        refs_differ += 1
        readings.append(
            f"n_obs={n_obs}: lp_iters={lp.iterations}"
            f" static_iters={static.iterations}"
            f" dynamic_iters={dynamic.iterations}"
            f" | static_obj={static.objective:+.3e}"
            f" dynamic_obj={dynamic.objective:+.3e}"
            f" lp_obj={lp.objective:+.3e}"
        )
    # The "across sizes" claim is unexercised unless the sweep genuinely spans
    # more than one size and the invariants held at every one of them.
    assert len(readings) == len(G5_SIZES) >= 2
    assert static_wins == len(G5_SIZES)
    assert refs_differ == len(G5_SIZES)
    print("\n--- dynamic vs static (ratification data) ---")
    print("\n".join(readings))
