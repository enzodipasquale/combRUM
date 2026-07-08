"""Conformance for the RowGenStep phase split.

The phase split hoists each row-generation method's per-rep collectives out
of ``evaluate``/``update`` into transport-free phase methods
(``contribute``/``finalise``) plus a root-only install (``apply_step``),
so a future engine owns the reduce and exchange. Two properties pin that:

* Bitwise parity: the test plays the engine — ``contribute``, its own
  reduce/exchange, then ``finalise`` and ``apply_step`` — and must reach
  the byte-identical published answer as the bundled ``evaluate``/``update``
  walk. Both formulations, toy and qkp, SerialTransport and interleaved
  LocalCluster(2)/(4).
* Transport-passive: ``contribute`` and ``finalise`` trigger zero transport
  calls; only the engine's reduce/exchange and ``apply_step``'s root bcast
  touch the wire (measured through CountingTransport).

Parity says nothing about a drift shared by both paths, so each family also
pins the published answer to fixture-derived recomputations: the objective
against the regret optimum (toy closed-form, qkp via a SciPy LP over the
enumerated bundles), theta_hat via the regret objective evaluated at the
published point (the optimum is flat, so only the face is pinned here — the
vertex itself is pinned bitwise in test_theta_hat_pins_solved_master_vertex),
NSlack's slack against the epigraph values, the dual payload against the
master LP's KKT system, and the installed-cut count against the running
``apply_step`` progress sum.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pytest

from _family_oracles import FamilyProblem, qkp_problem, toy_problem
from _walk import WalkOutcome, run_walk
from combrum.context import FitContext
from combrum.demand import Demand
from combrum.formulation import Formulation
from combrum.formulations import NSlack, OneSlack
from combrum.oracle import Oracle
from _support.constants import MAX_ITERATIONS, THETA_BOUND, TOLERANCE
from _support.commprobe import CountingTransport
from _support.families import FAMILY_DIR, load_family
from combrum.masters import gurobi as gurobi_backend
from combrum.masters import highs as highs_backend
from combrum.masters import make_master
from combrum.rowgen import (
    MaxContribution,
    MaxReduced,
    SumContribution,
    SumReduced,
)
from combrum.transport import LocalCluster, SerialTransport
from combrum.transport.base import Transport

GUROBI_AVAILABLE = gurobi_backend.available()
HIGHS_AVAILABLE = highs_backend.available()

needs_gurobi = pytest.mark.skipif(
    not GUROBI_AVAILABLE, reason="gurobipy missing or no environment starts"
)
needs_highs = pytest.mark.skipif(
    not HIGHS_AVAILABLE, reason="highspy missing or broken"
)

REAL_BACKENDS = (
    pytest.param("gurobi", marks=needs_gurobi),
    pytest.param("highs", marks=needs_highs),
)

_PROBLEMS = {"toy": toy_problem, "qkp": qkp_problem}

#: One live replication owned by rank 0 — the same owners vector the
#: bundled NSlack path ships its cuts through, so the engine-played
#: exchange routes identically.
_OWNERS = np.zeros(1, dtype=np.int64)

#: Gap allowed between the published objective and the regret optimum
#: recomputed here. Clean agreement is ~1e-13; the smallest drift this must
#: catch moves the objective by ~1e-6, so 1e-9 clears solver noise with room.
_OBJECTIVE_ANCHOR_ABS = 1e-9

#: Gap allowed between the objective recomputed at the *published* theta_hat
#: and the optimum. Recomputing at theta_hat carries no solver-vs-solver gap
#: (clean agreement ~1e-13), so 1e-9 is loose noise cover. Pins theta_hat to
#: the optimal *face* only — the optimum is flat in several coordinates, and
#: the flat directions are pinned bitwise in
#: test_theta_hat_pins_solved_master_vertex. Do not raise above ~1e-7 or a
#: non-flat 1e-6 drift no longer registers; the import-time guard below
#: enforces that.
_THETA_ANCHOR_ABS = 1e-9

#: Per-agent slack (NSlack ``u``) is recomputed exactly from the fixture
#: epigraph; clean agreement is ~1e-15, so 1e-9 is generous.
_SLACK_ANCHOR_ABS = 1e-9

#: OneSlack ``finalise().violation`` vs the aggregate slack
#: ``max(0, phi_agg.theta + eps_agg - u)`` from fixture math; clean agreement
#: is ~1e-14.
_ONESLACK_VIOLATION_ANCHOR_ABS = 1e-9

#: Tolerance on the KKT identities in ``_assert_dual_kkt_anchored``. They hold
#: to ~1e-13 on these families, and the smallest dual corruption to catch
#: moves a stationarity coordinate by O(1). Doubles as the exact-zero bar for
#: which theta coordinates count as sitting at a box bound.
_DUAL_KKT_ABS = 1e-6

# The anchors must stay far below the ~1e-6 drifts they exist to catch; fail
# at import if one is ever loosened toward them.
assert _THETA_ANCHOR_ABS <= 1e-8, "theta anchor loosened too far"
assert _SLACK_ANCHOR_ABS <= 1e-6, "slack anchor loosened too far"
assert (
    _ONESLACK_VIOLATION_ANCHOR_ABS <= 1e-6
), "OneSlack violation anchor loosened too far"
assert _DUAL_KKT_ABS <= 1e-4, "dual KKT anchor loosened too far"


def _enumerate_bundles(m: int) -> np.ndarray:
    """All ``2**m`` bundles as a ``(2**m, m) float64`` 0/1 matrix."""
    count = np.arange(2**m)
    return ((count[:, None] >> np.arange(m)[None, :]) & 1).astype(np.float64)


def _toy_regret_objective(arrays: Mapping[str, np.ndarray]) -> float:
    """Toy regret optimum from scratch, solver-free.

    The toy regret objective
    ``-sum_a phi_a(obs)·theta + sum_a max_b [phi_a(b)·theta + eps_a(b)]``
    separates across items (``phi_a(b)=b*r_a``, ``eps_a(b)=b·nu_a``, and the
    inner max picks each item independently), so each coordinate is a 1-D
    convex piecewise-linear function of ``theta_k`` whose only kinks sit at
    ``theta_k = -nu_{a,k}/r_{a,k}``. Evaluate at every kink inside the box
    plus the two box ends and take the least.
    """
    r = np.asarray(arrays["observables"], dtype=np.float64)
    nu = np.asarray(arrays["shocks"], dtype=np.float64)[:, 0, :]
    observed = np.asarray(arrays["observed"])
    _n_obs, n_items = r.shape
    bound = THETA_BOUND
    total = 0.0
    for k in range(n_items):
        linear = -float((observed[:, k].astype(np.float64) * r[:, k]).sum())
        kinks = -nu[:, k] / r[:, k]
        candidates = [-bound, bound]
        candidates += [t for t in kinks if -bound <= t <= bound]

        def g(t: float, _k: int = k) -> float:
            return linear * t + float(
                np.clip(r[:, _k] * t + nu[:, _k], 0.0, None).sum()
            )

        total += min(g(t) for t in candidates)
    return total


def _qkp_regret_objective(arrays: Mapping[str, np.ndarray]) -> float:
    """QKP regret optimum via a separately-formulated LP.

    The QKP regret objective does not separate (capacity and the quadratic
    coupling bind bundles together), so enumerate every feasible bundle as an
    epigraph constraint and solve the resulting LP with SciPy's HiGHS — a
    formulation and solver wholly outside combrum. Variables are ``theta``
    (K) and one epigraph ``u_a`` per agent; minimise
    ``-sum_a phi_a(obs)·theta + sum_a u_a`` s.t. ``phi_a(b)·theta + eps_a(b)
    <= u_a`` for every feasible ``b``.
    """
    scipy_linprog = pytest.importorskip("scipy.optimize").linprog
    x = np.asarray(arrays["x"], dtype=np.float64)
    q = np.asarray(arrays["Q"], dtype=np.float64)
    weights = np.asarray(arrays["weights"], dtype=np.float64)
    capacities = np.asarray(arrays["capacities"], dtype=np.float64)
    nu = np.asarray(arrays["shocks"], dtype=np.float64)[:, 0, :]
    observed = np.asarray(arrays["observed"])
    n_obs, m = x.shape
    K = m + 2
    bundles = _enumerate_bundles(m)
    loads = bundles @ weights
    quad_half = 0.5 * np.einsum("bj,jk,bk->b", bundles, q, bundles)

    def phi_rows(a: int) -> np.ndarray:
        rows = np.empty((bundles.shape[0], K), dtype=np.float64)
        rows[:, 0] = bundles @ x[a]
        rows[:, 1 : m + 1] = -bundles
        rows[:, m + 1] = quad_half
        return rows

    def observed_phi(a: int) -> np.ndarray:
        b = observed[a].astype(np.float64)
        phi = np.empty(K, dtype=np.float64)
        phi[0] = float(x[a] @ b)
        phi[1 : m + 1] = -b
        phi[m + 1] = 0.5 * float(b @ (q @ b))
        return phi

    n_var = K + n_obs
    c = np.zeros(n_var, dtype=np.float64)
    for a in range(n_obs):
        c[:K] -= observed_phi(a)
    c[K:] = 1.0
    rows_ub: list[np.ndarray] = []
    rhs_ub: list[float] = []
    for a in range(n_obs):
        phi = phi_rows(a)
        eps = bundles @ nu[a]
        feasible = loads <= capacities[a]
        for j in np.flatnonzero(feasible):
            row = np.zeros(n_var, dtype=np.float64)
            row[:K] = phi[j]
            row[K + a] = -1.0
            rows_ub.append(row)
            rhs_ub.append(-float(eps[j]))
    bound = THETA_BOUND
    bounds = (
        [(0.0, bound)]
        + [(-bound, bound)] * m
        + [(0.0, bound)]
        + [(None, None)] * n_obs
    )
    result = scipy_linprog(
        c,
        A_ub=np.asarray(rows_ub),
        b_ub=np.asarray(rhs_ub),
        bounds=bounds,
        method="highs",
    )
    assert result.success, result.message
    return float(result.fun)


def _toy_regret_at_theta(
    arrays: Mapping[str, np.ndarray], theta: np.ndarray
) -> tuple[float, np.ndarray]:
    """Evaluate the toy regret objective and per-agent ``u`` at ``theta``.

    Same separable structure as ``_toy_regret_objective`` but at a
    caller-supplied point: ``u_a = sum_k max(0, r_{a,k}*theta_k + nu_{a,k})``
    and objective ``-sum_a phi_a(obs)·theta + sum_a u_a``, both from the
    fixture arrays alone. A drifted ``theta_hat`` leaves the optimal face and
    its recomputed objective rises above the optimum.
    """
    r = np.asarray(arrays["observables"], dtype=np.float64)
    nu = np.asarray(arrays["shocks"], dtype=np.float64)[:, 0, :]
    observed = np.asarray(arrays["observed"]).astype(np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    u = np.clip(r * theta[None, :] + nu, 0.0, None).sum(axis=1)
    linear = -float(np.einsum("ak,k->", observed * r, theta))
    return linear + float(u.sum()), u


def _qkp_regret_at_theta(
    arrays: Mapping[str, np.ndarray], theta: np.ndarray
) -> tuple[float, np.ndarray]:
    """Evaluate the QKP regret objective and per-agent ``u`` at ``theta``.

    Enumerates every feasible bundle (the same epigraph set the LP above
    builds) and takes, per agent, ``u_a = max_{feasible b}
    [phi_a(b)·theta + eps_a(b)]``; the objective is ``-sum_a phi_a(obs)·theta
    + sum_a u_a``. No solver — the max runs over the enumerated feasible set.
    """
    x = np.asarray(arrays["x"], dtype=np.float64)
    q = np.asarray(arrays["Q"], dtype=np.float64)
    weights = np.asarray(arrays["weights"], dtype=np.float64)
    capacities = np.asarray(arrays["capacities"], dtype=np.float64)
    nu = np.asarray(arrays["shocks"], dtype=np.float64)[:, 0, :]
    observed = np.asarray(arrays["observed"]).astype(np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    n_obs, m = x.shape
    K = m + 2
    bundles = _enumerate_bundles(m)
    loads = bundles @ weights
    quad_half = 0.5 * np.einsum("bj,jk,bk->b", bundles, q, bundles)
    linear = 0.0
    u = np.empty(n_obs, dtype=np.float64)
    for a in range(n_obs):
        rows = np.empty((bundles.shape[0], K), dtype=np.float64)
        rows[:, 0] = bundles @ x[a]
        rows[:, 1 : m + 1] = -bundles
        rows[:, m + 1] = quad_half
        vals = rows @ theta + bundles @ nu[a]
        vals = np.where(loads <= capacities[a], vals, -np.inf)
        u[a] = float(vals.max())
        b = observed[a]
        phi_obs = np.empty(K, dtype=np.float64)
        phi_obs[0] = float(x[a] @ b)
        phi_obs[1 : m + 1] = -b
        phi_obs[m + 1] = 0.5 * float(b @ (q @ b))
        linear -= float(phi_obs @ theta)
    return linear + float(u.sum()), u


_REGRET_OPTIMA = {
    "toy": _toy_regret_objective,
    "qkp": _qkp_regret_objective,
}

_REGRET_AT_THETA = {
    "toy": _toy_regret_at_theta,
    "qkp": _qkp_regret_at_theta,
}


def _assert_objective_anchored(family: str, objective: float) -> None:
    """Published objective must equal the recomputed regret optimum."""
    arrays = load_family(family, FAMILY_DIR)
    expected = _REGRET_OPTIMA[family](arrays)
    assert objective == pytest.approx(expected, abs=_OBJECTIVE_ANCHOR_ABS)


def _assert_theta_anchored(family: str, theta_hat: np.ndarray) -> None:
    """The published ``theta_hat`` must lie on the optimal face.

    ``theta_hat`` is not unique here — the regret objective is flat across
    coordinates whose subgradient brackets zero, so the master may report any
    point of the optimal face. What is pinned: the objective recomputed at
    ``theta_hat`` from fixture math must equal the optimum. A drifted theta
    leaves the face and the recompute rises.
    """
    arrays = load_family(family, FAMILY_DIR)
    opt = _REGRET_OPTIMA[family](arrays)
    at_theta, _ = _REGRET_AT_THETA[family](arrays, theta_hat)
    # A feasible theta sits at or above the optimum, so equality pins the
    # face. No solver-vs-solver gap in the recompute.
    assert at_theta == pytest.approx(opt, abs=_THETA_ANCHOR_ABS)


def _assert_slack_anchored(
    family: str, theta_hat: np.ndarray, slack: np.ndarray
) -> None:
    """NSlack's published slack must equal the recomputed epigraph values.

    ``slack`` is the per-agent epigraph value ``u_a``; recompute it as
    ``max_b [phi_a(b)·theta_hat + eps_a(b)]`` over the family's own
    feature/shock arrays, independent of what the master published.
    """
    arrays = load_family(family, FAMILY_DIR)
    _, u = _REGRET_AT_THETA[family](arrays, theta_hat)
    np.testing.assert_allclose(
        np.asarray(slack, dtype=np.float64), u, atol=_SLACK_ANCHOR_ABS
    )


def _assert_dual_kkt_anchored(family: str, result: object) -> None:
    """The published NSlack dual payload must satisfy the master LP's KKT
    system, recomputed from the fixture feature map, ``theta_hat`` and the
    published slack — never from a master dual accessor.

    The relaxation is
    ``min c_theta·theta + sum_a w_a u_a  s.t.  u_a - phi_r·theta >= eps_r``
    with ``theta`` boxed and ``u_a >= 0``, where ``c_theta =
    -sum_a phi_a(observed_a)`` (built by :func:`_build_master`) and every
    ``w_a == 1``. The identities checked:

    * theta-stationarity: the column reduced cost
      ``z_k = c_theta[k] + sum_r pis[r]·phi_r[k]`` equals the box-bound
      multiplier ``bound_duals[k]`` (0 off a bound) on every coordinate.
      ``phi_r`` is refeaturised from the id + decoded bundle, so this
      exercises ``pis``, ``agent_ids``, ``bundle_row_ids`` and
      ``bundle_table`` together.
    * bound-dual support: every ``bound_duals`` key must be a theta_hat
      coordinate sitting at ``lower``/``upper``, checked from the box
      geometry. The backends land on different vertices (HiGHS interior with
      ``bound_duals == {}``; Gurobi qkp at three bounds), so both branches
      run.
    * u-stationarity: each agent with ``u_a > 0`` has
      ``sum_{r: agent==a} pis[r] == w_a == 1``.
    * complementary slackness + sign: ``pis >= 0`` and every row carrying
      dual mass is tight, ``phi_r·theta_hat + eps_r == u_a``.
    * moment: ``sum_r pis[r]·bundle_r`` by an explicit loop, structurally
      distinct from ``moment()``'s vectorised ``pis @ table[ids]``.
    """
    dual = result.dual
    problem = _PROBLEMS[family](load_family(family, FAMILY_DIR))
    arrays = load_family(family, FAMILY_DIR)
    observed = np.asarray(arrays["observed"])
    n_agents = observed.shape[0]
    K = problem.K
    theta = np.asarray(result.theta_hat, dtype=np.float64)
    slack = np.asarray(result.slack, dtype=np.float64)
    lower, upper = problem.theta_bounds
    # w_a == 1 for every agent under this file's setups (run_phase_walk /
    # run_walk both pass agent_weights = ones); the master's u objective
    # coefficient equals it.
    weights = np.ones(n_agents, dtype=np.float64)
    # c_theta = -sum_a phi_a(observed_a): the linear theta objective the master
    # is built with, recomputed here from the fixture feature map alone.
    c_theta = np.zeros(K, dtype=np.float64)
    for a in range(n_agents):
        c_theta -= np.asarray(
            problem.observed_features(a, observed[a]), dtype=np.float64
        )
    pis = np.asarray(dual.pis, dtype=np.float64)
    agent_ids = np.asarray(dual.agent_ids)
    bundle_row_ids = np.asarray(dual.bundle_row_ids)
    table = np.asarray(dual.bundle_table)
    # Refeaturise each dual row's (phi, eps) from its agent id and decoded
    # bundle — the fixture map, not any master accessor.
    featured = [
        problem.features(int(a), table[b])
        for a, b in zip(agent_ids, bundle_row_ids)
    ]
    phi_rows = np.stack(
        [np.asarray(phi, dtype=np.float64) for phi, _ in featured], axis=0
    )
    eps_rows = np.asarray([float(eps) for _, eps in featured], dtype=np.float64)

    # Sign: multipliers of a >= constraint are nonnegative.
    assert (pis >= -_DUAL_KKT_ABS).all(), pis.min()

    # Every reported box-bound coordinate must actually sit at a bound.
    at_bound = {
        k
        for k in range(K)
        if np.isclose(theta[k], lower[k]) or np.isclose(theta[k], upper[k])
    }
    assert set(map(int, dual.bound_duals)) <= at_bound, (
        dict(dual.bound_duals),
        at_bound,
    )

    # theta-stationarity folded with the bound duals: the reduced-cost
    # residual per coordinate must equal the reported bound-dual value (0 off
    # a bound). Every fixture reaching this anchor lands at a vertex whose
    # box-bound reduced costs are ~0 (qkp gurobi reports {0, 4, 7} all
    # <1e-14; toy reports {}), so the values are only pinned near zero here;
    # the nonzero case is test_nslack_bound_dual_pins_active_box_reduced_cost.
    bound_vector = np.array(
        [float(dual.bound_duals.get(k, 0.0)) for k in range(K)],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        pis @ phi_rows + c_theta, bound_vector, atol=_DUAL_KKT_ABS
    )
    assert all(
        abs(v) < _DUAL_KKT_ABS for v in dual.bound_duals.values()
    ), dict(dual.bound_duals)

    # u-stationarity: interior u_a pins its incident dual mass to w_a.
    for a in range(n_agents):
        if slack[a] > _DUAL_KKT_ABS:
            mass = float(pis[agent_ids == a].sum())
            assert abs(mass - weights[a]) < _DUAL_KKT_ABS, (a, mass)

    # Complementary slackness: rows carrying dual mass are tight.
    tightness = phi_rows @ theta + eps_rows - slack[agent_ids]
    carrying = pis > _DUAL_KKT_ABS
    if carrying.any():
        assert np.max(np.abs(tightness[carrying])) < _DUAL_KKT_ABS

    # moment: loop recompute, structurally distinct from moment()'s
    # pis @ table[ids].
    expected_moment = np.zeros(table.shape[1], dtype=np.float64)
    for b, pi in zip(bundle_row_ids, pis):
        expected_moment = expected_moment + pi * table[b]
    np.testing.assert_allclose(
        dual.moment(), expected_moment, atol=_DUAL_KKT_ABS
    )


# A K=1 toy family whose NSlack regret optimum sits on a box bound with a
# NONZERO reduced cost. phi_a(b) = b * r_a (scalar), eps_a(b) = b * nu_a.
# Each shock nu_a is so negative that r_a*theta + nu_a < 0 on the whole box:
# every u_a is 0, no cut ever ships, and the objective collapses to
# c_theta * theta with c_theta < 0 — so the optimum sits at the UPPER bound
# and that bound's reduced cost equals c_theta exactly (no dual mass to
# offset it). The canonical families never reach such a vertex.
_ACTIVE_BOX_BOUND = 10.0
_ACTIVE_BOX_R = np.array([[1.0], [1.0], [1.0]], dtype=np.float64)
_ACTIVE_BOX_NU = np.array([[-100.0], [-100.0], [-100.0]], dtype=np.float64)
_ACTIVE_BOX_OBSERVED = np.array([[1.0], [1.0], [1.0]], dtype=np.float64)


class _ActiveBoxOracle(Oracle):
    """Exact pricing for the dual-active-box family: item taken iff score > 0."""

    def price(self, theta: np.ndarray, agent_id: int) -> Demand:
        scores = _ACTIVE_BOX_R[agent_id] * theta + _ACTIVE_BOX_NU[agent_id]
        bundle = scores > 0.0
        return Demand.exact(
            bundle=bundle, payoff=float(np.where(bundle, scores, 0.0).sum())
        )


def _active_box_features(
    agent_id: int, bundle: np.ndarray
) -> tuple[np.ndarray, float]:
    b = np.asarray(bundle, dtype=np.float64)
    return b * _ACTIVE_BOX_R[agent_id], float(b @ _ACTIVE_BOX_NU[agent_id])


@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_nslack_bound_dual_pins_active_box_reduced_cost(backend: str) -> None:
    # The parity families only ever see near-zero box-bound reduced costs
    # (see _assert_dual_kkt_anchored). This walk lands theta on its upper
    # bound with a nonzero reduced cost, so the published bound-dual value is
    # checked against the stationarity residual at a point where it is O(1).
    K = 1
    lower = np.array([-_ACTIVE_BOX_BOUND], dtype=np.float64)
    upper = np.array([_ACTIVE_BOX_BOUND], dtype=np.float64)
    n_agents = _ACTIVE_BOX_OBSERVED.shape[0]
    # c_theta = -sum_a phi_a(observed_a), the master's linear theta objective,
    # built from fixture math alone (mirrors _build_master).
    c_theta = np.zeros(K, dtype=np.float64)
    for a in range(n_agents):
        c_theta -= _active_box_features(a, _ACTIVE_BOX_OBSERVED[a])[0]
    params = {"Method": 0, "LPWarmStart": 2} if backend == "gurobi" else None
    master = make_master(
        K, (lower, upper), c_theta, (lambda agent_id: 1.0), backend=backend,
        params=params,
    )
    transport = SerialTransport()
    local_ids = np.arange(n_agents, dtype=np.int64)
    ctx = FitContext(
        K=K,
        N=n_agents,
        S=1,
        theta_bounds=(lower, upper),
        theta_coef=np.ones(n_agents, dtype=np.float64),
        agent_weights=np.ones(n_agents, dtype=np.float64),
        local_ids=local_ids,
        transport=transport,
        tolerance=TOLERANCE,
        master_backend=master,
    )
    oracle = _ActiveBoxOracle()
    formulation = NSlack(_active_box_features)
    converged = False
    try:
        oracle.setup(transport, local_ids)
        formulation.setup(ctx)
        for _ in range(MAX_ITERATIONS):
            theta = formulation.solve()
            demands = {int(a): oracle.price(theta, int(a)) for a in local_ids}
            contribution = formulation.contribute(demands)
            reduced = _reduce(transport, contribution)
            outcome = formulation.finalise(reduced)
            formulation.apply_step(outcome.install_payload)
            if outcome.violation <= TOLERANCE:
                converged = True
                break
        result = formulation.result()
    finally:
        oracle.teardown()
        formulation.dispose()
        master.close()
    assert converged
    dual = result.dual
    theta_hat = np.asarray(result.theta_hat, dtype=np.float64)
    # theta must be pinned at its upper box bound for the bound to be dual-active.
    assert np.isclose(theta_hat[0], upper[0]), theta_hat

    # Stationarity residual z_k = c_theta[k] + sum_r pis[r]*phi_r[k], rebuilt
    # from the published pis and the fixture feature map (never a bound-dual
    # accessor). By dual feasibility it equals the box-bound reduced cost.
    pis = np.asarray(dual.pis, dtype=np.float64)
    agent_ids = np.asarray(dual.agent_ids)
    bundle_row_ids = np.asarray(dual.bundle_row_ids)
    table = np.asarray(dual.bundle_table)
    z = c_theta.astype(np.float64).copy()
    for a, b, pi in zip(agent_ids, bundle_row_ids, pis):
        phi, _ = _active_box_features(int(a), table[b])
        z += pi * np.asarray(phi, dtype=np.float64)
    bound_vector = np.array(
        [float(dual.bound_duals.get(k, 0.0)) for k in range(K)],
        dtype=np.float64,
    )
    np.testing.assert_allclose(bound_vector, z, atol=_DUAL_KKT_ABS)
    # At least one bound dual is genuinely nonzero on this fixture.
    assert np.max(np.abs(bound_vector)) > 0.5, dict(dual.bound_duals)
    # With no cut installed the only stationarity term is c_theta, so the
    # upper bound carries exactly c_theta.
    assert dict(dual.bound_duals) == {0: pytest.approx(float(c_theta[0]))}


def _reduce(
    transport: Transport, contribution: object
) -> MaxReduced | SumReduced:
    """Play the engine's per-type reduce: MAX+exchange, or reproducible SUM.

    Dispatch is on the concrete :class:`~combrum.rowgen.Contribution` type,
    not a string tag.
    """
    if isinstance(contribution, MaxContribution):
        return MaxReduced(
            global_worst=transport.allreduce_max(contribution.worst),
            received_rows=transport.exchange_cuts(
                contribution.local_rows, _OWNERS
            ),
        )
    if isinstance(contribution, SumContribution):
        return SumReduced(
            aggregate=np.asarray(
                transport.sum_reproducible(
                    contribution.terms, contribution.ids
                ),
                dtype=np.float64,
            )
        )
    raise AssertionError(f"unexpected contribution type: {contribution!r}")


def _build_master(
    arrays: Mapping[str, np.ndarray],
    problem: FamilyProblem,
    formulation_cls: type[Formulation],
    theta_coef: np.ndarray,
    agent_weights: np.ndarray,
    backend: str,
) -> object:
    """Root-side master for a phase walk, mirroring ``_walk.run_walk``."""
    observed = np.asarray(arrays["observed"])
    n_agents = observed.shape[0]
    c_theta = np.zeros(problem.K, dtype=np.float64)
    for a in range(n_agents):
        phi_obs = problem.observed_features(a, observed[a])
        c_theta -= theta_coef[a] * np.asarray(phi_obs, dtype=np.float64)
    u_coef = (
        (lambda agent_id: 1.0)
        if formulation_cls is OneSlack
        else (lambda agent_id: float(agent_weights[agent_id]))
    )
    params = {"Method": 0, "LPWarmStart": 2} if backend == "gurobi" else None
    return make_master(
        problem.K,
        problem.theta_bounds,
        c_theta,
        u_coef,
        backend=backend,
        params=params,
    )


def run_phase_walk(
    arrays: Mapping[str, np.ndarray],
    problem: FamilyProblem,
    formulation_cls: type[Formulation],
    transport: Transport,
    *,
    backend: str,
    tolerance: float = TOLERANCE,
    max_iterations: int = MAX_ITERATIONS,
) -> WalkOutcome:
    """The phase-path twin of ``_walk.run_walk`` (no schedule/penalty/warm).

    Setup and teardown mirror the bundled walk verbatim; only the inner
    step differs, driving the loop as the engine will: ``contribute`` ->
    [engine reduce + exchange] -> ``finalise`` -> ``apply_step``, stopping
    at ``violation <= tolerance``.
    """
    observed = np.asarray(arrays["observed"])
    n_agents = observed.shape[0]
    local_ids = np.arange(
        transport.rank, n_agents, transport.size, dtype=np.int64
    )
    theta_coef = np.ones(n_agents, dtype=np.float64)
    agent_weights = np.ones(n_agents, dtype=np.float64)
    master = None
    if transport.rank == 0:
        master = _build_master(
            arrays, problem, formulation_cls, theta_coef, agent_weights, backend
        )
    ctx = FitContext(
        K=problem.K,
        N=n_agents,
        S=1,
        theta_bounds=problem.theta_bounds,
        theta_coef=theta_coef,
        agent_weights=agent_weights,
        local_ids=local_ids,
        transport=transport,
        tolerance=tolerance,
        master_backend=master,
    )
    oracle = problem.oracle
    formulation = formulation_cls(problem.features)
    converged = False
    iterations = 0
    cuts_admitted = 0
    try:
        oracle.setup(transport, local_ids)
        formulation.setup(ctx)
        for _ in range(max_iterations):
            theta = formulation.solve()
            demands = {int(a): oracle.price(theta, int(a)) for a in local_ids}
            contribution = formulation.contribute(demands)
            reduced = _reduce(transport, contribution)
            outcome = formulation.finalise(reduced)
            progressed = formulation.apply_step(outcome.install_payload)
            cuts_admitted += progressed
            iterations += 1
            if outcome.violation <= tolerance:
                converged = True
                break
        result = formulation.result()
    finally:
        oracle.teardown()
        formulation.dispose()
        if master is not None:
            master.close()
    return WalkOutcome(
        result=result,
        objective=result.objective,
        converged=converged,
        iterations=iterations,
        cuts_admitted=cuts_admitted,
    )


def _assert_bitwise_equal(
    phase: WalkOutcome, bundled: WalkOutcome, *, full_dual: bool
) -> None:
    """Phase-path answer must match the bundled answer to the byte."""
    assert phase.converged and bundled.converged
    assert phase.iterations == bundled.iterations
    assert phase.cuts_admitted == bundled.cuts_admitted
    res, ref = phase.result, bundled.result
    # cuts_admitted (running sum of apply_step's progress return) and
    # n_active_cuts (master property) come from distinct sources; with no
    # retirement in this file every admitted cut stays installed, so they
    # must agree.
    assert phase.cuts_admitted == res.n_active_cuts
    assert bundled.cuts_admitted == ref.n_active_cuts
    assert res.theta_hat.tobytes() == ref.theta_hat.tobytes()
    assert res.objective == ref.objective
    assert phase.objective == bundled.objective
    assert res.n_active_cuts == ref.n_active_cuts
    if full_dual:
        # NSlack also publishes the per-agent slack, the installed cut set,
        # and the dual payload — each bitwise. n_active_cuts must equal the
        # length of the published active_set it counts.
        assert res.n_active_cuts == len(res.active_set)
        assert ref.n_active_cuts == len(ref.active_set)
        assert res.slack.tobytes() == ref.slack.tobytes()
        assert [
            (row.agent_id, row.bundle_key, row.phi.tobytes(), row.epsilon)
            for row in res.active_set
        ] == [
            (row.agent_id, row.bundle_key, row.phi.tobytes(), row.epsilon)
            for row in ref.active_set
        ]
        for attr in ("agent_ids", "bundle_row_ids", "pis", "bundle_table"):
            assert (
                getattr(res.dual, attr).tobytes()
                == getattr(ref.dual, attr).tobytes()
            ), attr
        assert res.dual.moment().tobytes() == ref.dual.moment().tobytes()
        assert dict(res.dual.bound_duals) == dict(ref.dual.bound_duals)


# --- phase == bundled, bitwise, serial ----------------------------------------


@pytest.mark.parametrize("family", ["toy", "qkp"])
@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_nslack_phase_matches_bundled_serial(
    backend: str, family: str
) -> None:
    arrays = load_family(family, FAMILY_DIR)
    problem = _PROBLEMS[family](arrays)
    bundled = run_walk(
        arrays, problem, NSlack, SerialTransport(), backend=backend
    )
    phase = run_phase_walk(
        arrays, problem, NSlack, SerialTransport(), backend=backend
    )
    _assert_bitwise_equal(phase, bundled, full_dual=True)
    # Parity cannot see a drift shared by both paths, so also pin the shared
    # answer to the fixture recomputations: objective, optimal face, per-agent
    # epigraph values, and the dual payload's KKT system.
    _assert_objective_anchored(family, bundled.objective)
    _assert_theta_anchored(family, bundled.result.theta_hat)
    _assert_slack_anchored(family, bundled.result.theta_hat, bundled.result.slack)
    _assert_dual_kkt_anchored(family, bundled.result)


@pytest.mark.parametrize("family", ["toy", "qkp"])
@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_oneslack_phase_matches_bundled_serial(
    backend: str, family: str
) -> None:
    arrays = load_family(family, FAMILY_DIR)
    problem = _PROBLEMS[family](arrays)
    bundled = run_walk(
        arrays, problem, OneSlack, SerialTransport(), backend=backend
    )
    phase = run_phase_walk(
        arrays, problem, OneSlack, SerialTransport(), backend=backend
    )
    # OneSlack holds no per-agent slack/cut-set/dual to publish.
    _assert_bitwise_equal(phase, bundled, full_dual=False)
    _assert_objective_anchored(family, bundled.objective)
    _assert_theta_anchored(family, bundled.result.theta_hat)


# --- phase == bundled, bitwise, interleaved local cluster ---------------------


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("family", ["toy", "qkp"])
@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_nslack_phase_matches_bundled_local_cluster(
    backend: str, family: str, size: int
) -> None:
    # Interleaved shards (a % size == rank) re-route every cut and reduction
    # contribution; the answer must still match serial bitwise.
    arrays = load_family(family, FAMILY_DIR)
    problem = _PROBLEMS[family](arrays)
    bundled = run_walk(
        arrays, problem, NSlack, SerialTransport(), backend=backend
    )
    results = LocalCluster(size).run(
        lambda transport: run_phase_walk(
            arrays, problem, NSlack, transport, backend=backend
        )
    )
    assert len(results) == size
    for phase in results:
        _assert_bitwise_equal(phase, bundled, full_dual=True)
    _assert_objective_anchored(family, bundled.objective)
    _assert_theta_anchored(family, bundled.result.theta_hat)
    _assert_slack_anchored(family, bundled.result.theta_hat, bundled.result.slack)
    _assert_dual_kkt_anchored(family, bundled.result)


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("family", ["toy", "qkp"])
@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_oneslack_phase_matches_bundled_local_cluster(
    backend: str, family: str, size: int
) -> None:
    arrays = load_family(family, FAMILY_DIR)
    problem = _PROBLEMS[family](arrays)
    bundled = run_walk(
        arrays, problem, OneSlack, SerialTransport(), backend=backend
    )
    results = LocalCluster(size).run(
        lambda transport: run_phase_walk(
            arrays, problem, OneSlack, transport, backend=backend
        )
    )
    assert len(results) == size
    for phase in results:
        _assert_bitwise_equal(phase, bundled, full_dual=False)
    _assert_objective_anchored(family, bundled.objective)
    _assert_theta_anchored(family, bundled.result.theta_hat)


# --- NSlack.finalise.violation carries the reduced worst ----------------------


@pytest.mark.parametrize("family", ["toy", "qkp"])
@needs_highs
def test_nslack_finalise_violation_echoes_reduced_worst(family: str) -> None:
    # The bundled NSlack twin never reads finalise().violation (evaluate
    # computes the stop signal itself), so parity says nothing about the
    # field's value. Over each real step, finalise must echo the reduce's
    # global_worst verbatim and forward the exchanged rows untouched.
    arrays = load_family(family, FAMILY_DIR)
    problem = _PROBLEMS[family](arrays)
    transport = SerialTransport()
    observed = np.asarray(arrays["observed"])
    n_agents = observed.shape[0]
    local_ids = np.arange(n_agents, dtype=np.int64)
    theta_coef = np.ones(n_agents, dtype=np.float64)
    agent_weights = np.ones(n_agents, dtype=np.float64)
    master = _build_master(
        arrays, problem, NSlack, theta_coef, agent_weights, "highs"
    )
    ctx = FitContext(
        K=problem.K,
        N=n_agents,
        S=1,
        theta_bounds=problem.theta_bounds,
        theta_coef=theta_coef,
        agent_weights=agent_weights,
        local_ids=local_ids,
        transport=transport,
        tolerance=TOLERANCE,
        master_backend=master,
    )
    oracle = problem.oracle
    formulation = NSlack(problem.features)
    checked = False
    try:
        oracle.setup(transport, local_ids)
        formulation.setup(ctx)
        for _ in range(MAX_ITERATIONS):
            theta = formulation.solve()
            demands = {int(a): oracle.price(theta, int(a)) for a in local_ids}
            contribution = formulation.contribute(demands)
            reduced = _reduce(transport, contribution)
            outcome = formulation.finalise(reduced)
            assert outcome.violation == reduced.global_worst
            # The first iterations ship rows, so the forwarding check runs on
            # a non-empty exchange at least once.
            if reduced.received_rows:
                assert outcome.install_payload is reduced.received_rows
                checked = True
            formulation.apply_step(outcome.install_payload)
            if outcome.violation <= TOLERANCE:
                break
    finally:
        oracle.teardown()
        formulation.dispose()
        master.close()
    assert checked, "no violated step exercised finalise.violation"


# --- OneSlack.finalise.violation carries the aggregate slack -------------------


@pytest.mark.parametrize("family", ["toy", "qkp"])
@needs_highs
def test_oneslack_finalise_violation_echoes_aggregate_slack(family: str) -> None:
    # The bundled OneSlack path itself calls finalise, so parity cannot pin
    # the violation's value. Over each real step, finalise's violation must
    # equal the floored aggregate slack max(0, phi_agg.theta + eps_agg - u)
    # recomputed here, and install_payload must carry the same
    # (phi_agg, eps_agg). phi_agg/eps_agg come from the fixture features (not
    # from contribute), theta from a direct master read, and u from the
    # master's objective identity (objective - c_theta.theta) — none reuses
    # finalise's own output.
    arrays = load_family(family, FAMILY_DIR)
    problem = _PROBLEMS[family](arrays)
    transport = SerialTransport()
    observed = np.asarray(arrays["observed"])
    n_agents = observed.shape[0]
    local_ids = np.arange(n_agents, dtype=np.int64)
    theta_coef = np.ones(n_agents, dtype=np.float64)
    agent_weights = np.ones(n_agents, dtype=np.float64)
    # The master's linear theta objective, held here so u can be recovered
    # from the objective identity rather than a master u-accessor.
    c_theta = np.zeros(problem.K, dtype=np.float64)
    for a in range(n_agents):
        phi_obs = problem.observed_features(a, observed[a])
        c_theta -= theta_coef[a] * np.asarray(phi_obs, dtype=np.float64)
    master = _build_master(
        arrays, problem, OneSlack, theta_coef, agent_weights, "highs"
    )
    ctx = FitContext(
        K=problem.K,
        N=n_agents,
        S=1,
        theta_bounds=problem.theta_bounds,
        theta_coef=theta_coef,
        agent_weights=agent_weights,
        local_ids=local_ids,
        transport=transport,
        tolerance=TOLERANCE,
        master_backend=master,
    )
    oracle = problem.oracle
    formulation = OneSlack(problem.features)
    steps = 0
    try:
        oracle.setup(transport, local_ids)
        formulation.setup(ctx)
        for _ in range(MAX_ITERATIONS):
            # Read theta straight from the master, not the formulation.
            theta = np.asarray(master.theta(), dtype=np.float64)
            demands = {int(a): oracle.price(theta, int(a)) for a in local_ids}
            # The aggregate row, rebuilt from the fixture feature map.
            phi_agg = np.zeros(problem.K, dtype=np.float64)
            eps_agg = 0.0
            for a in local_ids:
                phi_a, eps_a = problem.features(int(a), demands[int(a)].bundle)
                phi_agg += agent_weights[a] * np.asarray(phi_a, dtype=np.float64)
                eps_agg += agent_weights[a] * float(eps_a)
            # u from the master's objective identity (objective = c_theta.theta
            # + u), so the slack recompute leans on no OneSlack u-accessor.
            u = float(master.objective()) - float(c_theta @ theta)
            raw = float(phi_agg @ theta) + eps_agg - u
            expected_violation = raw if raw > 0.0 else 0.0

            contribution = formulation.contribute(demands)
            reduced = _reduce(transport, contribution)
            outcome = formulation.finalise(reduced)
            # violation echoes the floored aggregate slack.
            assert outcome.violation == pytest.approx(
                expected_violation, abs=_ONESLACK_VIOLATION_ANCHOR_ABS
            )
            # install_payload carries the same aggregate row, unscaled.
            payload_phi, payload_eps = outcome.install_payload
            np.testing.assert_allclose(
                np.asarray(payload_phi, dtype=np.float64),
                phi_agg,
                atol=_ONESLACK_VIOLATION_ANCHOR_ABS,
            )
            assert float(payload_eps) == pytest.approx(
                eps_agg, abs=_ONESLACK_VIOLATION_ANCHOR_ABS
            )
            formulation.apply_step(outcome.install_payload)
            steps += 1
            if outcome.violation <= TOLERANCE:
                break
    finally:
        oracle.teardown()
        formulation.dispose()
        master.close()
    # At least one violated step must have run before convergence.
    assert steps >= 2, "walk converged too fast to exercise finalise.violation"


# --- theta_hat is the actually-solved master vertex ---------------------------


@pytest.mark.parametrize("formulation_cls", [NSlack, OneSlack])
@pytest.mark.parametrize("family", ["toy", "qkp"])
@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_theta_hat_pins_solved_master_vertex(
    backend: str, formulation_cls: type[Formulation], family: str
) -> None:
    # _assert_theta_anchored only pins theta_hat to the optimal *face*, and
    # the optimum is flat in several coordinates (box bounds, subgradient-
    # bracketing kinks — e.g. qkp NSlack k=4, toy OneSlack k=1). The master
    # solve, however, lands on one specific vertex and result() publishes
    # exactly that vertex, so capture master.theta() in the walk — a distinct
    # accessor from result().theta_hat, which routes through the
    # formulation's adopt/state path — and require the published theta to
    # equal it bitwise. That pins the flat directions too.
    arrays = load_family(family, FAMILY_DIR)
    problem = _PROBLEMS[family](arrays)
    transport = SerialTransport()
    observed = np.asarray(arrays["observed"])
    n_agents = observed.shape[0]
    local_ids = np.arange(n_agents, dtype=np.int64)
    theta_coef = np.ones(n_agents, dtype=np.float64)
    agent_weights = np.ones(n_agents, dtype=np.float64)
    master = _build_master(
        arrays, problem, formulation_cls, theta_coef, agent_weights, backend
    )
    ctx = FitContext(
        K=problem.K,
        N=n_agents,
        S=1,
        theta_bounds=problem.theta_bounds,
        theta_coef=theta_coef,
        agent_weights=agent_weights,
        local_ids=local_ids,
        transport=transport,
        tolerance=TOLERANCE,
        master_backend=master,
    )
    oracle = problem.oracle
    formulation = formulation_cls(problem.features)
    try:
        oracle.setup(transport, local_ids)
        formulation.setup(ctx)
        for _ in range(MAX_ITERATIONS):
            theta = formulation.solve()
            demands = {int(a): oracle.price(theta, int(a)) for a in local_ids}
            contribution = formulation.contribute(demands)
            reduced = _reduce(transport, contribution)
            outcome = formulation.finalise(reduced)
            formulation.apply_step(outcome.install_payload)
            if outcome.violation <= TOLERANCE:
                break
        # Capture the solved vertex before result() and before teardown.
        vertex = np.asarray(master.theta(), dtype=np.float64).copy()
        result = formulation.result()
    finally:
        oracle.teardown()
        formulation.dispose()
        master.close()
    assert result.theta_hat.tobytes() == vertex.tobytes()
    # The captured vertex must itself lie on the optimal face, so the check
    # above is not merely self-consistent.
    opt = _REGRET_OPTIMA[family](arrays)
    at_vertex, _ = _REGRET_AT_THETA[family](arrays, vertex)
    assert at_vertex == pytest.approx(opt, abs=_THETA_ANCHOR_ABS)


# --- contribute + finalise are transport-passive ------------------------------


@dataclass
class _PhaseProbe:
    """Wire-call deltas tallied around each of the four phases."""

    contribute_calls: int
    finalise_calls: int
    reduce_calls: int
    apply_calls: int
    #: Per-kind wire deltas across the reduce phase only.
    reduce_kinds: dict[str, int] = field(default_factory=dict)
    #: Per-kind wire deltas across the apply_step phase only.
    apply_kinds: dict[str, int] = field(default_factory=dict)


def _run_phase_probed(
    arrays: Mapping[str, np.ndarray],
    problem: FamilyProblem,
    formulation_cls: type[Formulation],
    backend: str,
) -> tuple[WalkOutcome, _PhaseProbe]:
    """Drive the phase walk through CountingTransport, tallying per phase.

    The transport count is snapshotted immediately before and after each
    phase, so any collective a phase made shows up as a non-zero delta.
    """
    observed = np.asarray(arrays["observed"])
    n_agents = observed.shape[0]
    transport = CountingTransport(SerialTransport())
    local_ids = np.arange(0, n_agents, 1, dtype=np.int64)
    theta_coef = np.ones(n_agents, dtype=np.float64)
    agent_weights = np.ones(n_agents, dtype=np.float64)
    master = _build_master(
        arrays, problem, formulation_cls, theta_coef, agent_weights, backend
    )
    ctx = FitContext(
        K=problem.K,
        N=n_agents,
        S=1,
        theta_bounds=problem.theta_bounds,
        theta_coef=theta_coef,
        agent_weights=agent_weights,
        local_ids=local_ids,
        transport=transport,
        tolerance=TOLERANCE,
        master_backend=master,
    )
    oracle = problem.oracle
    formulation = formulation_cls(problem.features)
    probe = _PhaseProbe(0, 0, 0, 0)

    def _wire_calls() -> int:
        return sum(transport.counts().values())

    converged = False
    iterations = 0
    cuts_admitted = 0
    try:
        oracle.setup(transport, local_ids)
        formulation.setup(ctx)
        for _ in range(MAX_ITERATIONS):
            theta = formulation.solve()
            demands = {int(a): oracle.price(theta, int(a)) for a in local_ids}

            before = _wire_calls()
            contribution = formulation.contribute(demands)
            probe.contribute_calls += _wire_calls() - before

            kinds_before = transport.counts()
            before = _wire_calls()
            reduced = _reduce(transport, contribution)
            probe.reduce_calls += _wire_calls() - before
            kinds_after = transport.counts()
            for kind in kinds_after.keys() | kinds_before.keys():
                delta = kinds_after.get(kind, 0) - kinds_before.get(kind, 0)
                if delta:
                    probe.reduce_kinds[kind] = (
                        probe.reduce_kinds.get(kind, 0) + delta
                    )

            before = _wire_calls()
            outcome = formulation.finalise(reduced)
            probe.finalise_calls += _wire_calls() - before

            apply_kinds_before = transport.counts()
            before = _wire_calls()
            progressed = formulation.apply_step(outcome.install_payload)
            probe.apply_calls += _wire_calls() - before
            apply_kinds_after = transport.counts()
            for kind in apply_kinds_after.keys() | apply_kinds_before.keys():
                delta = apply_kinds_after.get(kind, 0) - apply_kinds_before.get(
                    kind, 0
                )
                if delta:
                    probe.apply_kinds[kind] = (
                        probe.apply_kinds.get(kind, 0) + delta
                    )

            cuts_admitted += progressed
            iterations += 1
            if outcome.violation <= TOLERANCE:
                converged = True
                break
        result = formulation.result()
    finally:
        oracle.teardown()
        formulation.dispose()
        master.close()
    walk = WalkOutcome(
        result=result,
        objective=result.objective,
        converged=converged,
        iterations=iterations,
        cuts_admitted=cuts_admitted,
    )
    return walk, probe


@pytest.mark.parametrize("formulation_cls", [NSlack, OneSlack])
@needs_highs
def test_contribute_and_finalise_are_transport_passive(
    formulation_cls: type,
) -> None:
    arrays = load_family("toy", FAMILY_DIR)
    problem = toy_problem(arrays)
    walk, probe = _run_phase_probed(arrays, problem, formulation_cls, "highs")
    assert walk.converged
    # Only the engine's reduce and apply_step's root bcast touch the wire;
    # contribute and finalise stay at zero across the run.
    assert probe.contribute_calls == 0
    assert probe.finalise_calls == 0
    assert probe.apply_calls > 0
    # apply_step's full per-iteration wire budget: its root bcast plus the
    # collective guard. On a single rank NSlack adopts the owner's u
    # directly, so no per-agent scatter appears. Asserting the whole
    # kind->count dict means any extra collective inside apply_step fails.
    expected_apply_kinds = {
        "bcast": walk.iterations,
        "collective_guard": walk.iterations,
    }
    assert probe.apply_kinds == expected_apply_kinds
    # The reduce moved exactly the collectives the contribution type demands:
    # MaxContribution via allreduce_max + exchange_cuts, SumContribution via
    # sum_reproducible. A wrong Contribution type would route the other kind.
    expected_kinds = (
        {"allreduce_max", "exchange_cuts"}
        if formulation_cls is NSlack
        else {"sum_reproducible"}
    )
    moved = {kind for kind, count in probe.reduce_kinds.items() if count > 0}
    assert moved == expected_kinds
